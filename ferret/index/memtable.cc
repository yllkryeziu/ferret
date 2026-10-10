// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The memtable publishes each complete document with a release store. Its
// fixed directories and immutable published records permit concurrent reads
// without a record lock; only posting-list snapshots acquire a reader lock.

#include "ferret/index/memtable.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

#include "ferret/base/env.h"
#include "ferret/git/oid.h"
#include "ferret/index/blob_registry.h"
#include "ferret/index/segment.h"
#include "ferret/index/segment_writer.h"
#include "ferret/postings/ids.h"
#include "ferret/query/evaluator.h"
#include "ferret/query/trigram_query.h"
#include "ferret/symbols/symbol_extractor.h"
#include "ferret/text/chunker.h"
#include "ferret/text/fold.h"
#include "ferret/text/language.h"
#include "ferret/text/lexer.h"
#include "ferret/text/line_counter.h"
#include "ferret/text/trigram.h"

namespace ferret {
namespace {
constexpr size_t kArenaBlockBytes = 1 << 20;

void CollectTrigrams(const TrigramQuery& query,
                     absl::flat_hash_set<Trigram>& trigrams) {
  trigrams.insert(query.trigrams().begin(), query.trigrams().end());
  for (const TrigramQuery& sub : query.subs()) CollectTrigrams(sub, trigrams);
}
}  // namespace

MemtableQuery::MemtableQuery(
    const TrigramQuery& query,
    absl::flat_hash_map<Trigram, std::vector<ChunkId>> lists)
    : lists_(std::move(lists)) {
  evaluator_ =
      std::make_unique<TrigramQueryEvaluator<absl::Span<const ChunkId>>>(
          query, [this](Trigram trigram) -> absl::Span<const ChunkId> {
            auto it = lists_.find(trigram);
            return it == lists_.end() ? absl::Span<const ChunkId>()
                                      : absl::MakeConstSpan(it->second);
          });
}

Memtable::Memtable(const MemtableOptions& options,
                   std::shared_ptr<BlobRegistry> blobs)
    : options_(options),
      docs_(options.max_docs),
      blob_of_doc_(options.max_docs),
      blobs_(blobs ? std::move(blobs) : std::make_shared<BlobRegistry>()),
      chunks_(options.max_chunks),
      arena_(std::make_unique<std::unique_ptr<char[]>[]>(
          static_cast<size_t>(options.max_docs) +
          (options.max_content_bytes + kArenaBlockBytes - 1) /
              kArenaBlockBytes)),
      symbols_(options.max_docs),
      symbols_state_(options.max_docs) {
  CHECK(options.max_docs > 0);
  CHECK(options.max_chunks > 0);
  CHECK(options.max_content_bytes > 0);
  CHECK_LT(options.flush_docs, options.max_docs);
  CHECK_LT(options.flush_bytes, options.max_content_bytes);
  CHECK(options.segment.chunk_bytes > 0);
  CHECK(options.segment.chunk_bytes <= std::numeric_limits<uint32_t>::max());
}

absl::StatusOr<DocId> Memtable::Add(const SegmentDocument& document,
                                    bool defer_symbols) {
  if (document.hint_path.empty() ||
      document.content.size() >= std::numeric_limits<uint32_t>::max()) {
    return absl::InvalidArgumentError("invalid memtable document");
  }
  const std::vector<ChunkSpan> spans =
      SplitIntoChunks(document.content, options_.segment.chunk_bytes);
  if (next_doc_ == options_.max_docs ||
      spans.size() > options_.max_chunks - next_chunk_ ||
      document.content.size() > options_.max_content_bytes - content_bytes_) {
    return absl::ResourceExhaustedError("memtable is full");
  }
  // Do not reuse a block for an oversized blob. Regular blobs never cross
  // block boundaries; every returned view points into one allocation.
  char* content = nullptr;
  if (document.content.size() > kArenaBlockBytes) {
    arena_[next_block_] = std::make_unique<char[]>(document.content.size());
    content = arena_[next_block_++].get();
    used_in_block_ = kArenaBlockBytes;
    allocated_content_bytes_ += document.content.size();
  } else {
    if (next_block_ == 0 ||
        document.content.size() > kArenaBlockBytes - used_in_block_) {
      arena_[next_block_] = std::make_unique<char[]>(kArenaBlockBytes);
      ++next_block_;
      used_in_block_ = 0;
      allocated_content_bytes_ += kArenaBlockBytes;
    }
    content = arena_[next_block_ - 1].get() + used_in_block_;
    used_in_block_ += document.content.size();
  }
  if (!document.content.empty()) {
    std::memcpy(content, document.content.data(), document.content.size());
  }
  const absl::string_view bytes(content, document.content.size());
  const DocId id = next_doc_;
  SegmentDoc record = {
      .hint_path = {},
      .content = bytes,
      .oid = document.rank_key.oid,
      .tier = document.rank_key.tier,
      .prior = document.rank_key.prior,
      .language = document.language,
      .long_lines = document.long_lines,
      .first_chunk = next_chunk_,
      .num_chunks = static_cast<uint32_t>(spans.size()),
      .num_lines = static_cast<uint32_t>(CountNewlines(bytes)) +
                   (!bytes.empty() && bytes.back() != '\n')};
  // Paths are allocated independently and never changed after publication.
  // Keep the allocation in the doc record rather than a movable string.
  auto path = std::make_unique<char[]>(document.hint_path.size());
  std::memcpy(path.get(), document.hint_path.data(), document.hint_path.size());
  record.hint_path = {path.get(), document.hint_path.size()};
  LexState state = LexState::kCode;
  posting_scratch_.clear();
  for (const ChunkSpan& span : spans) {
    const absl::string_view text = bytes.substr(
        static_cast<size_t>(span.offset), static_cast<size_t>(span.size));
    chunks_.Set(next_chunk_,
                {.chunk = {.doc = id,
                           .content = text,
                           .offset_in_doc = static_cast<uint64_t>(span.offset),
                           .first_line = static_cast<uint32_t>(span.first_line),
                           .entry_state = state}});
    chunk_trigrams_.clear();
    extractor_.Extract(text, chunk_trigrams_);
    extractor_.ExtractShortLines(text, chunk_trigrams_);
    for (Trigram trigram : chunk_trigrams_) {
      posting_scratch_.emplace_back(trigram, next_chunk_);
    }
    state = AdvanceLexState(document.language, text, state);
    ++next_chunk_;
  }
  size_t dictionary_bytes;
  {
    absl::MutexLock lock(postings_mu_);
    for (const auto& [trigram, chunk_id] : posting_scratch_) {
      postings_[trigram].push_back(chunk_id);
    }
    first_docs_.try_emplace(record.oid, id);
    dictionary_bytes = postings_.size() * 32;
  }
  posting_bytes_ += posting_scratch_.size() * sizeof(ChunkId);
  // Store path ownership alongside the record's view, before publication.
  docs_.Set(id, {.doc = record,
                 .rank_key = document.rank_key,
                 .arrival = absl::Now(),
                 .path = std::move(path)});
  blob_of_doc_.Set(id, blobs_->Intern(record.oid));
  ++next_doc_;
  content_bytes_ += bytes.size();
  path_bytes_ += document.hint_path.size();
  published_memory_bytes_.store(
      allocated_content_bytes_ +
          static_cast<size_t>(next_doc_) * sizeof(DocRecord) +
          static_cast<size_t>(next_chunk_) * sizeof(ChunkRecord) +
          posting_bytes_ + dictionary_bytes + path_bytes_,
      std::memory_order_relaxed);
  if (options_.segment.write_symbols && HasSymbolGrammar(document.language)) {
    absl::MutexLock lock(symbols_mu_);
    symbols_state_[id] = 1;
    ++pending_symbols_;
  }
  published_docs_.store(next_doc_, std::memory_order_release);
  if (options_.segment.write_symbols &&
      (!options_.async_symbols || !defer_symbols) &&
      HasSymbolGrammar(document.language)) {
    SymbolExtractor symbol_extractor;
    absl::StatusOr<std::vector<ExtractedSymbol>> symbols =
        symbol_extractor.Extract(document.language, bytes);
    AttachSymbols(
        id, symbols.ok() ? std::move(*symbols) : std::vector<ExtractedSymbol>{},
        !symbols.ok());
  }
  return id;
}

void Memtable::AttachSymbols(DocId id, std::vector<ExtractedSymbol> symbols,
                             bool failed) {
  absl::MutexLock lock(symbols_mu_);
  DCHECK_EQ(symbols_state_[id], 1);
  DCHECK_LT(id, published_docs());
  symbols_[id] = std::move(symbols);
  for (uint32_t i = 0; i < symbols_[id].size(); ++i) {
    names_.emplace(Fold(symbols_[id][i].name), std::pair(id, i));
  }
  symbols_state_[id] = failed ? 3 : 2;
  --pending_symbols_;
  if (failed) ++failed_symbols_;
}

bool Memtable::SymbolsPending(DocId id) const {
  absl::MutexLock lock(symbols_mu_);
  return symbols_state_[id] == 1;
}

uint32_t Memtable::pending_symbols() const {
  absl::MutexLock lock(symbols_mu_);
  return pending_symbols_;
}

uint32_t Memtable::failed_symbols() const {
  absl::MutexLock lock(symbols_mu_);
  return failed_symbols_;
}

void Memtable::FindSymbols(
    absl::string_view name, bool case_insensitive, uint32_t watermark,
    absl::FunctionRef<void(DocId, const ExtractedSymbol&)> fn) const {
  std::vector<std::pair<DocId, ExtractedSymbol>> found;
  {
    absl::MutexLock lock(symbols_mu_);
    const auto [begin, end] = names_.equal_range(Fold(name));
    for (auto it = begin; it != end; ++it) {
      const DocId id = it->second.first;
      const ExtractedSymbol& symbol = symbols_[id][it->second.second];
      if (id < watermark && (case_insensitive || symbol.name == name)) {
        found.emplace_back(id, symbol);
      }
    }
  }
  for (const auto& [id, symbol] : found) fn(id, symbol);
}

BlobId Memtable::blob_of_doc(DocId id) const {
  DCHECK_LT(id, published_docs());
  return blob_of_doc_.Get(id);
}

uint32_t Memtable::num_chunks(uint32_t watermark) const {
  DCHECK(watermark <= published_docs());
  if (watermark == 0) return 0;
  const SegmentDoc& last = docs_.Get(watermark - 1).doc;
  return last.first_chunk + last.num_chunks;
}

SegmentDoc Memtable::doc(DocId id) const {
  DCHECK(id < published_docs());
  SegmentDoc result = docs_.Get(id).doc;
  result.symbols_pending = SymbolsPending(id);
  return result;
}

SegmentChunk Memtable::chunk(ChunkId id) const {
  DCHECK(id < num_chunks(published_docs()));
  return chunks_.Get(id).chunk;
}

std::unique_ptr<MemtableQuery> Memtable::NewEvaluator(
    const TrigramQuery& query, uint32_t watermark) const {
  const uint32_t limit = num_chunks(watermark);
  absl::flat_hash_set<Trigram> trigrams;
  CollectTrigrams(query, trigrams);
  absl::flat_hash_map<Trigram, std::vector<ChunkId>> copies;
  {
    absl::ReaderMutexLock lock(postings_mu_);
    for (Trigram trigram : trigrams) {
      auto it = postings_.find(trigram);
      if (it == postings_.end()) continue;
      const auto end =
          std::lower_bound(it->second.begin(), it->second.end(), limit);
      copies.emplace(trigram, std::vector<ChunkId>(it->second.begin(), end));
    }
  }
  return std::unique_ptr<MemtableQuery>(
      new MemtableQuery(query, std::move(copies)));
}

std::optional<DocId> Memtable::FindDoc(const Oid& oid,
                                       uint32_t watermark) const {
  DCHECK(watermark <= published_docs());
  absl::ReaderMutexLock lock(postings_mu_);
  auto it = first_docs_.find(oid);
  if (it == first_docs_.end() || it->second >= watermark) return std::nullopt;
  return it->second;
}

bool Memtable::ShouldFlush() const {
  return published_docs() >= options_.flush_docs ||
         memory_bytes() >= options_.flush_bytes;
}

std::optional<absl::Time> Memtable::oldest_document_time() const {
  if (published_docs() == 0) return std::nullopt;
  return docs_.Get(0).arrival;
}

size_t Memtable::memory_bytes() const {
  return published_memory_bytes_.load(std::memory_order_relaxed);
}

absl::StatusOr<SegmentStats> Memtable::WriteSegment(
    Env& env, absl::string_view path, uint32_t watermark,
    absl::FunctionRef<bool(DocId)> keep) const {
  if (watermark > published_docs()) {
    return absl::InvalidArgumentError("watermark exceeds published documents");
  }
  {
    absl::MutexLock lock(symbols_mu_);
    symbols_mu_.Await(absl::Condition(this, &Memtable::SymbolsReady));
  }
  std::vector<DocId> order;
  order.reserve(watermark);
  for (DocId id = 0; id < watermark; ++id) {
    if (keep(id)) order.push_back(id);
  }
  std::sort(order.begin(), order.end(), [this](DocId a, DocId b) {
    const RankKey& left = docs_.Get(a).rank_key;
    const RankKey& right = docs_.Get(b).rank_key;
    return left == right ? a < b : left < right;
  });
  SegmentWriterOptions segment_options = options_.segment;
  segment_options.extract_symbols = false;
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<SegmentWriter> writer,
                        SegmentWriter::Create(env, path, segment_options));
  absl::flat_hash_set<Oid> seen;
  for (DocId id : order) {
    const DocRecord& entry = docs_.Get(id);
    if (!seen.insert(entry.doc.oid).second) continue;
    ABSL_RETURN_IF_ERROR(writer->Add({.hint_path = entry.doc.hint_path,
                                      .content = entry.doc.content,
                                      .rank_key = entry.rank_key,
                                      .language = entry.doc.language,
                                      .long_lines = entry.doc.long_lines}));
    if (segment_options.write_symbols) {
      absl::MutexLock lock(symbols_mu_);
      writer->AddSymbols(symbols_[id]);
      if (symbols_state_[id] == 3) writer->MarkSymbolsFailed();
    }
  }
  return writer->Finish();
}

}  // namespace ferret

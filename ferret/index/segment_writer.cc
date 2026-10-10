// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/segment_writer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/crc/crc32c.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/types/span.h"

#include "ferret/base/env.h"
#include "ferret/index/internal/build_util.h"
#include "ferret/index/internal/code_tokens.h"
#include "ferret/index/internal/run_merge.h"
#include "ferret/index/internal/segment_format.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/segment.h"
#include "ferret/postings/ids.h"
#include "ferret/postings/posting_list.h"
#include "ferret/postings/posting_list_encoder.h"
#include "ferret/symbols/symbol_extractor.h"
#include "ferret/text/chunker.h"
#include "ferret/text/fold.h"
#include "ferret/text/language.h"
#include "ferret/text/lexer.h"
#include "ferret/text/trigram.h"

namespace ferret {
namespace {

namespace sf = segment_format;

using segment_internal::NumLines;
using segment_internal::ScopedTimer;

constexpr uint64_t kMaxId = std::numeric_limits<uint32_t>::max();
// Appends are written to the file in pieces of about this size.
constexpr size_t kWriteBufferBytes = size_t{1} << 20;

// Returns min(count, 4096) zero bytes.
absl::string_view ZeroBytes(uint64_t count) {
  static constexpr char kZeros[4096] = {};
  return absl::string_view(
      kZeros, static_cast<size_t>(std::min<uint64_t>(count, sizeof(kZeros))));
}

}  // namespace

absl::StatusOr<std::unique_ptr<SegmentWriter>> SegmentWriter::Create(
    Env& env, absl::string_view path, const SegmentWriterOptions& options) {
  if (options.chunk_bytes < 1 ||
      static_cast<uint64_t>(options.chunk_bytes) > kMaxId) {
    return absl::InvalidArgumentError(
        absl::StrCat("chunk_bytes out of range: ", options.chunk_bytes));
  }
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<WritableFile> file,
                        env.NewWritableFile(path));
  std::unique_ptr<SegmentWriter> writer(
      new SegmentWriter(std::move(file), options));
  // The header and section table are written over the first page at the
  // end, when they are known; content starts on the next section boundary.
  ABSL_RETURN_IF_ERROR(writer->AppendZeros(sf::kSectionAlignment));
  ABSL_RETURN_IF_ERROR(writer->BeginSection(sf::SectionKind::kContent));
  return writer;
}

SegmentWriter::SegmentWriter(std::unique_ptr<WritableFile> file,
                             const SegmentWriterOptions& options)
    : file_(std::move(file)),
      options_(options),
      postings_(options.max_partition_postings) {
  buffer_.reserve(kWriteBufferBytes);
}

SegmentWriter::~SegmentWriter() {
  // An unfinished segment is abandoned: close the descriptor and leave the
  // partial file to the caller.
  if (!finished_) file_->Close().IgnoreError();
}

absl::Status SegmentWriter::Add(const SegmentDocument& doc,
                                bool extract_symbols) {
  if (document_mode_ == DocumentMode::kChunked ||
      document_mode_ == DocumentMode::kPrecomputed) {
    return absl::FailedPreconditionError(
        "cannot mix content and copied documents");
  }
  if (finished_) return absl::FailedPreconditionError("Finish() was called");
  if (!status_.ok()) return status_;
  if (doc.hint_path.empty()) {
    return absl::InvalidArgumentError("empty hint path");
  }
  if (last_rank_key_.has_value() && doc.rank_key < *last_rank_key_) {
    return absl::InvalidArgumentError(
        absl::StrCat("document out of rank order: ", doc.hint_path));
  }
  // Content below 2^32 - 1 bytes keeps every chunk size and line number in
  // its 32-bit field.
  if (doc.content.size() >= kMaxId) {
    return absl::InvalidArgumentError(
        absl::StrCat("content too large: ", doc.hint_path));
  }
  std::vector<ChunkSpan> chunks;
  {
    ScopedTimer timer(&timings_.chunking_ns);
    chunks = SplitIntoChunks(doc.content, options_.chunk_bytes);
  }
  if (num_docs_ == kMaxId || num_chunks_ + uint64_t{chunks.size()} > kMaxId ||
      paths_.size() + doc.hint_path.size() > kMaxId) {
    return absl::ResourceExhaustedError("segment is full");
  }

  uint32_t num_lines = 0;
  {
    ScopedTimer timer(&timings_.chunking_ns);
    num_lines = NumLines(doc.content);
  }
  const sf::DocRecord doc_record = segment_internal::MakeDocRecord(
      num_chunks_, static_cast<uint32_t>(chunks.size()), num_lines,
      static_cast<uint32_t>(paths_.size()), doc.rank_key, doc.language,
      doc.long_lines);
  sf::AppendRecord(doc_record, doc_records_);
  LexState state = LexState::kCode;
  for (const ChunkSpan& chunk : chunks) {
    const absl::string_view text = doc.content.substr(
        static_cast<size_t>(chunk.offset), static_cast<size_t>(chunk.size));
    const sf::ChunkRecord chunk_record = segment_internal::MakeChunkRecord(
        content_bytes_, chunk, num_docs_, state);
    sf::AppendRecord(chunk_record, chunk_records_);
    chunk_trigrams_.clear();
    {
      ScopedTimer timer(&timings_.trigrams_ns);
      extractor_.Extract(text, chunk_trigrams_);
      if (options_.write_short_line_trigrams) {
        extractor_.ExtractShortLines(text, chunk_trigrams_);
      }
    }
    {
      ScopedTimer timer(&timings_.postings_add_ns);
      postings_.AddChunk(num_chunks_, chunk_trigrams_);
    }
    ++num_chunks_;
    ScopedTimer timer(&timings_.code_tokens_ns);
    state = options_.write_code_tokens
                ? AddCodeTokens(doc.language, text, state)
                : AdvanceLexState(doc.language, text, state);
  }
  oids_.append(doc.rank_key.oid.raw());
  paths_.append(doc.hint_path);
  content_bytes_ += doc.content.size();
  ++num_docs_;
  last_rank_key_ = doc.rank_key;
  document_mode_ = DocumentMode::kContent;
  {
    ScopedTimer timer(&timings_.content_ns);
    ABSL_RETURN_IF_ERROR(Append(doc.content));
  }
  if (options_.write_symbols && options_.extract_symbols && extract_symbols &&
      HasSymbolGrammar(doc.language)) {
    ScopedTimer timer(&timings_.symbols_ns);
    absl::StatusOr<std::vector<ExtractedSymbol>> extracted =
        symbol_extractor_.Extract(doc.language, doc.content);
    if (extracted.ok()) {
      AddSymbols(*extracted);
    } else {
      MarkSymbolsFailed();
    }
  }
  return absl::OkStatus();
}

absl::Status SegmentWriter::AddPrecomputed(
    const SegmentDocument& doc, uint32_t num_lines,
    absl::Span<const PrecomputedChunk> chunks) {
  if (document_mode_ == DocumentMode::kContent ||
      document_mode_ == DocumentMode::kChunked) {
    return absl::FailedPreconditionError(
        "cannot mix precomputed documents with other documents");
  }
  if (finished_) return absl::FailedPreconditionError("Finish() was called");
  if (!status_.ok()) return status_;
  if (doc.hint_path.empty()) {
    return absl::InvalidArgumentError("empty hint path");
  }
  if (last_rank_key_.has_value() && doc.rank_key < *last_rank_key_) {
    return absl::InvalidArgumentError(
        absl::StrCat("document out of rank order: ", doc.hint_path));
  }
  if (doc.content.size() >= kMaxId) {
    return absl::InvalidArgumentError(
        absl::StrCat("content too large: ", doc.hint_path));
  }
  if (num_docs_ == kMaxId || num_chunks_ + uint64_t{chunks.size()} > kMaxId ||
      paths_.size() + doc.hint_path.size() > kMaxId) {
    return absl::ResourceExhaustedError("segment is full");
  }
  sf::AppendRecord(segment_internal::MakeDocRecord(
                       num_chunks_, static_cast<uint32_t>(chunks.size()),
                       num_lines, static_cast<uint32_t>(paths_.size()),
                       doc.rank_key, doc.language, doc.long_lines),
                   doc_records_);
  for (const PrecomputedChunk& chunk : chunks) {
    sf::AppendRecord(segment_internal::MakeChunkRecord(
                         content_bytes_,
                         ChunkSpan{.offset = chunk.offset_in_doc,
                                   .size = chunk.size,
                                   .first_line = chunk.first_line},
                         num_docs_, chunk.entry_state),
                     chunk_records_);
    ++num_chunks_;
  }
  oids_.append(doc.rank_key.oid.raw());
  paths_.append(doc.hint_path);
  content_bytes_ += doc.content.size();
  ++num_docs_;
  last_rank_key_ = doc.rank_key;
  document_mode_ = DocumentMode::kPrecomputed;
  ScopedTimer timer(&timings_.content_ns);
  return Append(doc.content);
}

void SegmentWriter::SetPostings(
    std::vector<segment_internal::PostingAccumulator::Run> runs,
    std::vector<ChunkId> chunk_of_provisional) {
  precomputed_runs_ = std::move(runs);
  chunk_of_provisional_ = std::move(chunk_of_provisional);
}

void SegmentWriter::SetCodeTokenMaps(
    std::vector<segment_internal::CodeTokenMap> maps,
    std::vector<DocId> doc_of_provisional) {
  precomputed_tokens_ = std::move(maps);
  doc_of_provisional_ = std::move(doc_of_provisional);
}

absl::Status SegmentWriter::AddChunked(const SegmentDoc& doc,
                                       absl::Span<const SegmentChunk> chunks) {
  if (document_mode_ == DocumentMode::kContent ||
      document_mode_ == DocumentMode::kPrecomputed) {
    return absl::FailedPreconditionError(
        "cannot mix content and copied documents");
  }
  if (finished_) return absl::FailedPreconditionError("Finish() was called");
  if (!status_.ok()) return status_;
  const RankKey key{.tier = doc.tier, .prior = doc.prior, .oid = doc.oid};
  if (doc.hint_path.empty() ||
      (last_rank_key_.has_value() && key < *last_rank_key_)) {
    return absl::InvalidArgumentError(
        "document out of rank order or empty path");
  }
  if (num_docs_ == kMaxId || num_chunks_ + uint64_t{chunks.size()} > kMaxId ||
      paths_.size() + doc.hint_path.size() > kMaxId ||
      doc.content.size() >= kMaxId) {
    return absl::ResourceExhaustedError("segment is full");
  }
  if (chunks.size() != doc.num_chunks) {
    return absl::InvalidArgumentError("copied chunk count mismatch");
  }
  const sf::DocRecord record = {
      .first_chunk = num_chunks_,
      .num_chunks = static_cast<uint32_t>(chunks.size()),
      .num_lines = doc.num_lines,
      .path_offset = static_cast<uint32_t>(paths_.size()),
      .prior = doc.prior,
      .tier = static_cast<uint8_t>(doc.tier),
      .language = static_cast<uint8_t>(doc.language),
      .flags =
          static_cast<uint8_t>((doc.long_lines ? sf::kDocLongLines : 0) |
                               (doc.symbols_failed ? sf::kSymbolsFailed : 0)),
      .pad = {},
  };
  sf::AppendRecord(record, doc_records_);
  for (const SegmentChunk& chunk : chunks) {
    const sf::ChunkRecord chunk_record = {
        .content_offset = content_bytes_ + chunk.offset_in_doc,
        .content_size = static_cast<uint32_t>(chunk.content.size()),
        .first_line = chunk.first_line,
        .doc = num_docs_,
        .flags = static_cast<uint8_t>(chunk.entry_state),
        .pad = {},
    };
    sf::AppendRecord(chunk_record, chunk_records_);
    ++num_chunks_;
    if (options_.write_code_tokens) {
      AddCodeTokens(doc.language, chunk.content, chunk.entry_state);
    }
  }
  oids_.append(doc.oid.raw());
  paths_.append(doc.hint_path);
  content_bytes_ += doc.content.size();
  ++num_docs_;
  last_rank_key_ = key;
  document_mode_ = DocumentMode::kChunked;
  return Append(doc.content);
}

absl::StatusOr<SegmentStats> SegmentWriter::Finish() {
  if (document_mode_ == DocumentMode::kChunked) {
    return absl::FailedPreconditionError(
        "copied documents need FinishMerged()");
  }
  if (document_mode_ == DocumentMode::kPrecomputed) {
    return FinishWithLists([&](SegmentStats& stats) {
      const int64_t start = absl::GetCurrentTimeNanos();
      const absl::Status status = segment_internal::MergeRuns(
          precomputed_runs_, chunk_of_provisional_, options_.finish_pool,
          options_.finish_threads,
          [&](Trigram trigram, size_t count, uint64_t inline_postings,
              absl::string_view encoded) {
            ScopedTimer timer(&timings_.lists_encode_ns);
            return WriteEncodedList(trigram, count, inline_postings, encoded,
                                    stats);
          });
      timings_.lists_merge_ns += absl::GetCurrentTimeNanos() - start;
      precomputed_runs_.clear();
      return status;
    });
  }
  return FinishWithLists([&](SegmentStats& stats) {
    const int64_t start = absl::GetCurrentTimeNanos();
    const int64_t encode_before = timings_.lists_encode_ns;
    const absl::Status status = postings_.ForEachList(
        [&](Trigram trigram, absl::Span<const ChunkId> ids) {
          ScopedTimer timer(&timings_.lists_encode_ns);
          return WritePostingList(trigram, ids, stats);
        });
    timings_.lists_merge_ns += absl::GetCurrentTimeNanos() - start -
                               (timings_.lists_encode_ns - encode_before);
    return status;
  });
}

absl::StatusOr<SegmentStats> SegmentWriter::FinishMerged(
    absl::FunctionRef<absl::StatusOr<bool>(Trigram&, std::vector<ChunkId>&)>
        next_list,
    bool short_line_trigrams) {
  if (document_mode_ == DocumentMode::kContent ||
      document_mode_ == DocumentMode::kPrecomputed) {
    return absl::FailedPreconditionError("content documents need Finish()");
  }
  short_line_trigrams_ =
      short_line_trigrams && options_.write_short_line_trigrams;
  return FinishWithLists([&](SegmentStats& stats) -> absl::Status {
    Trigram trigram = 0;
    std::vector<ChunkId> ids;
    while (true) {
      ids.clear();
      ABSL_ASSIGN_OR_RETURN(bool has_list, next_list(trigram, ids));
      if (!has_list) return absl::OkStatus();
      ABSL_RETURN_IF_ERROR(WritePostingList(trigram, ids, stats));
    }
  });
}

absl::StatusOr<SegmentStats> SegmentWriter::FinishWithLists(
    absl::FunctionRef<absl::Status(SegmentStats&)> write_lists) {
  if (finished_) return absl::FailedPreconditionError("Finish() was called");
  finished_ = true;
  if (!status_.ok()) {
    file_->Close().IgnoreError();
    return status_;
  }
  SegmentStats stats;
  stats.num_docs = num_docs_;
  stats.num_chunks = num_chunks_;
  stats.content_bytes = content_bytes_;
  const absl::Status status = WriteTail(stats, write_lists);
  const absl::Status close_status = file_->Close();
  ABSL_RETURN_IF_ERROR(status);
  ABSL_RETURN_IF_ERROR(close_status);
  stats.file_bytes = file_size_;
  return stats;
}

absl::Status SegmentWriter::WriteTail(
    SegmentStats& stats,
    absl::FunctionRef<absl::Status(SegmentStats&)> write_lists) {
  {
    ScopedTimer timer(&timings_.tail_sections_ns);
    ABSL_RETURN_IF_ERROR(AppendZeros(sf::kContentPadding));
    EndSection();
    ABSL_RETURN_IF_ERROR(
        WriteSection(sf::SectionKind::kChunks, chunk_records_));
    ABSL_RETURN_IF_ERROR(WriteSection(sf::SectionKind::kDocs, doc_records_));
    ABSL_RETURN_IF_ERROR(WriteSection(sf::SectionKind::kOids, oids_));
    ABSL_RETURN_IF_ERROR(WriteSection(sf::SectionKind::kPaths, paths_));
  }
  if (options_.write_symbols) {
    ScopedTimer timer(&timings_.symbols_write_ns);
    ABSL_RETURN_IF_ERROR(WriteSymbols());
  }
  if (options_.write_code_tokens) {
    ScopedTimer timer(&timings_.code_tokens_write_ns);
    ABSL_RETURN_IF_ERROR(WriteCodeTokens());
  }
  ABSL_RETURN_IF_ERROR(BeginSection(sf::SectionKind::kPostings));
  ABSL_RETURN_IF_ERROR(write_lists(stats));
  {
    ScopedTimer timer(&timings_.dictionary_ns);
    ABSL_RETURN_IF_ERROR(WriteDictionary(stats));
    ABSL_RETURN_IF_ERROR(WriteHeaderAndFooter(stats));
  }
  ScopedTimer timer(&timings_.sync_ns);
  return file_->Sync();
}

void SegmentWriter::MarkSymbolsFailed() { MarkSymbolsFailedFor(num_docs_ - 1); }

void SegmentWriter::MarkSymbolsFailedFor(DocId doc) {
  ++failed_symbols_;
  const size_t offset = size_t{doc} * sizeof(sf::DocRecord);
  sf::DocRecord record = sf::ReadRecord<sf::DocRecord>(
      absl::string_view(doc_records_).substr(offset), 0);
  record.flags |= sf::kSymbolsFailed;
  std::memcpy(doc_records_.data() + offset, &record, sizeof(record));
}

void SegmentWriter::AddSymbols(absl::Span<const ExtractedSymbol> symbols) {
  AddSymbolsFor(num_docs_ - 1, symbols);
}

void SegmentWriter::AddSymbolsFor(DocId doc,
                                  absl::Span<const ExtractedSymbol> symbols) {
  for (const ExtractedSymbol& symbol : symbols) {
    symbols_.push_back({Fold(symbol.name), symbol, doc});
  }
}

absl::Status SegmentWriter::WriteSymbols() {
  std::sort(symbols_.begin(), symbols_.end(),
            [](const PendingSymbol& a, const PendingSymbol& b) {
              if (a.folded != b.folded) return a.folded < b.folded;
              if (a.symbol.name != b.symbol.name) {
                return a.symbol.name < b.symbol.name;
              }
              if (a.doc != b.doc) return a.doc < b.doc;
              return a.symbol.line < b.symbol.line;
            });
  std::string records;
  std::string names;
  for (const PendingSymbol& pending : symbols_) {
    if (names.size() + pending.folded.size() + pending.symbol.name.size() >
        std::numeric_limits<uint32_t>::max()) {
      return absl::ResourceExhaustedError("symbol name pool is full");
    }
    const sf::SymbolRecord record = {
        .folded_offset = static_cast<uint32_t>(names.size()),
        .name_offset =
            static_cast<uint32_t>(names.size() + pending.folded.size()),
        .doc = pending.doc,
        .line = pending.symbol.line,
        .folded_size = static_cast<uint32_t>(pending.folded.size()),
        .name_size = static_cast<uint32_t>(pending.symbol.name.size()),
        .kind = static_cast<uint8_t>(pending.symbol.kind),
        .flags =
            pending.symbol.is_definition ? sf::kSymbolDefinition : uint8_t{0},
        .pad = {},
    };
    sf::AppendRecord(record, records);
    names.append(pending.folded);
    names.append(pending.symbol.name);
  }
  if (symbols_.size() > std::numeric_limits<uint32_t>::max()) {
    return absl::ResourceExhaustedError("too many symbols");
  }
  std::string section;
  sf::AppendRecord(static_cast<uint32_t>(symbols_.size()), section);
  sf::AppendRecord(static_cast<uint32_t>(names.size()), section);
  section.append(records);
  section.append(names);
  return WriteSection(sf::SectionKind::kSymbols, section);
}

LexState SegmentWriter::AddCodeTokens(Language language, absl::string_view text,
                                      LexState entry) {
  // AddChunked() calls this after the document's records but before the
  // document counter advances, and Add() likewise: the document is
  // num_docs_ in both.
  return segment_internal::AddCodeTokens(language, text, entry, num_docs_,
                                         token_scratch_, code_tokens_);
}

absl::Status SegmentWriter::WriteCodeTokens() {
  absl::StatusOr<std::string> section;
  if (document_mode_ == DocumentMode::kPrecomputed) {
    section = segment_internal::BuildCodeTokensSection(
        precomputed_tokens_, doc_of_provisional_, options_.finish_pool,
        options_.finish_threads);
    precomputed_tokens_.clear();
  } else {
    section = segment_internal::BuildCodeTokensSection(
        absl::MakeConstSpan(&code_tokens_, 1), {}, nullptr, 1);
    code_tokens_.clear();
  }
  if (!section.ok()) return section.status();
  return WriteSection(sf::SectionKind::kCodeTokens, *section);
}

absl::Status SegmentWriter::Append(absl::string_view data) {
  if (!status_.ok()) return status_;
  if (open_section_.has_value()) {
    open_section_crc_ = static_cast<uint32_t>(
        absl::ExtendCrc32c(absl::crc32c_t{open_section_crc_}, data));
  }
  file_size_ += data.size();
  if (buffer_.size() + data.size() <= kWriteBufferBytes) {
    buffer_.append(data);
    return absl::OkStatus();
  }
  ABSL_RETURN_IF_ERROR(FlushBuffer());
  if (data.size() < kWriteBufferBytes) {
    buffer_.append(data);
    return absl::OkStatus();
  }
  status_.Update(file_->Append(data));
  return status_;
}

absl::Status SegmentWriter::AppendZeros(uint64_t count) {
  while (count > 0) {
    const absl::string_view zeros = ZeroBytes(count);
    ABSL_RETURN_IF_ERROR(Append(zeros));
    count -= zeros.size();
  }
  return absl::OkStatus();
}

absl::Status SegmentWriter::FlushBuffer() {
  if (!status_.ok()) return status_;
  if (buffer_.empty()) return absl::OkStatus();
  status_.Update(file_->Append(buffer_));
  buffer_.clear();
  return status_;
}

absl::Status SegmentWriter::BeginSection(sf::SectionKind kind) {
  DCHECK(!open_section_.has_value());
  const uint64_t padding =
      (sf::kSectionAlignment - file_size_ % sf::kSectionAlignment) %
      sf::kSectionAlignment;
  ABSL_RETURN_IF_ERROR(AppendZeros(padding));
  open_section_ = sf::SectionEntry{.kind = static_cast<uint32_t>(kind),
                                   .flags = 0,
                                   .offset = file_size_,
                                   .length = 0,
                                   .crc32c = 0,
                                   .reserved = 0};
  open_section_crc_ = 0;
  return absl::OkStatus();
}

void SegmentWriter::EndSection() {
  DCHECK(open_section_.has_value());
  open_section_->length = section_size();
  open_section_->crc32c = open_section_crc_;
  sections_.push_back(*open_section_);
  open_section_.reset();
}

uint64_t SegmentWriter::section_size() const {
  DCHECK(open_section_.has_value());
  return file_size_ - open_section_->offset;
}

absl::Status SegmentWriter::WriteSection(sf::SectionKind kind,
                                         absl::string_view bytes) {
  ABSL_RETURN_IF_ERROR(BeginSection(kind));
  ABSL_RETURN_IF_ERROR(Append(bytes));
  EndSection();
  return absl::OkStatus();
}

absl::Status SegmentWriter::WritePostingList(Trigram trigram,
                                             absl::Span<const ChunkId> ids,
                                             SegmentStats& stats) {
  if (ids.empty() || ids.size() > kMaxPostingListSize ||
      trigram >= (1u << 24) || ids.back() >= num_chunks_ ||
      !std::is_sorted(ids.begin(), ids.end()) ||
      std::adjacent_find(ids.begin(), ids.end()) != ids.end() ||
      (!dictionary_.empty() &&
       trigram <=
           sf::ReadRecord<sf::TrigramEntry>(
               dictionary_, dictionary_.size() / sizeof(sf::TrigramEntry) - 1)
               .trigram_and_flags)) {
    return absl::InvalidArgumentError("invalid posting list");
  }
  if (IsInlinedPostingList(ids.size())) {
    uint64_t inline_postings = ids[0];
    if (ids.size() == 2) inline_postings |= uint64_t{ids[1]} << 32;
    return WriteEncodedList(trigram, ids.size(), inline_postings, {}, stats);
  }
  std::string encoded;
  AppendEncodedPostingList(ids, encoded);
  return WriteEncodedList(trigram, ids.size(), 0, encoded, stats);
}

absl::Status SegmentWriter::WriteEncodedList(Trigram trigram, size_t count,
                                             uint64_t inline_postings,
                                             absl::string_view encoded,
                                             SegmentStats& stats) {
  sf::TrigramEntry entry = {.trigram_and_flags = trigram,
                            .count = static_cast<uint32_t>(count),
                            .postings = 0};
  if (IsInlinedPostingList(count)) {
    entry.postings = inline_postings;
  } else {
    if (count >= kPostingBlockSize) {
      const uint64_t misalignment = section_size() % kPostingPayloadAlignment;
      if (misalignment != 0) {
        ABSL_RETURN_IF_ERROR(
            AppendZeros(kPostingPayloadAlignment - misalignment));
      }
    }
    entry.postings = section_size();
    ABSL_RETURN_IF_ERROR(Append(encoded));
  }
  sf::AppendRecord(entry, dictionary_);
  stats.num_postings += count;
  return absl::OkStatus();
}

absl::Status SegmentWriter::WriteDictionary(SegmentStats& stats) {
  stats.postings_bytes = section_size();
  EndSection();
  stats.num_trigrams =
      static_cast<uint32_t>(dictionary_.size() / sizeof(sf::TrigramEntry));
  ABSL_RETURN_IF_ERROR(WriteSection(sf::SectionKind::kTrigrams, dictionary_));
  std::vector<uint32_t> index(sf::kTrigramIndexSize, 0);
  for (size_t i = 0; i < stats.num_trigrams; ++i) {
    const auto entry = sf::ReadRecord<sf::TrigramEntry>(dictionary_, i);
    ++index[(entry.trigram_and_flags >> 8) + 1];
  }
  for (size_t b = 1; b < index.size(); ++b) index[b] += index[b - 1];
  return WriteSection(
      sf::SectionKind::kTrigramIndex,
      absl::string_view(reinterpret_cast<const char*>(index.data()),
                        index.size() * sizeof(uint32_t)));
}

absl::Status SegmentWriter::WriteHeaderAndFooter(const SegmentStats& stats) {
  std::string table;
  for (const sf::SectionEntry& section : sections_) {
    sf::AppendRecord(section, table);
  }
  sf::SegmentFooter footer = {};
  std::memcpy(footer.magic, sf::kFooterMagic, sizeof(footer.magic));
  footer.file_length = file_size_ + sizeof(footer);
  footer.section_table_crc32c =
      static_cast<uint32_t>(absl::ComputeCrc32c(table));
  std::string footer_bytes;
  sf::AppendRecord(footer, footer_bytes);
  ABSL_RETURN_IF_ERROR(Append(footer_bytes));
  ABSL_RETURN_IF_ERROR(FlushBuffer());

  sf::SegmentHeader header = {};
  std::memcpy(header.magic, sf::kHeaderMagic, sizeof(header.magic));
  header.format_version = sf::kFormatVersion;
  header.flags = sf::kHeaderNoNewlineTrigrams |
                 (short_line_trigrams_ ? sf::kHeaderShortLineTrigrams : 0u);
  header.chunk_bytes = static_cast<uint32_t>(options_.chunk_bytes);
  header.num_docs = stats.num_docs;
  header.num_chunks = stats.num_chunks;
  header.num_trigrams = stats.num_trigrams;
  header.num_sections = static_cast<uint32_t>(sections_.size());
  std::string header_bytes;
  sf::AppendRecord(header, header_bytes);
  header.crc32c = static_cast<uint32_t>(
      absl::ComputeCrc32c(absl::string_view(header_bytes)
                              .substr(0, offsetof(sf::SegmentHeader, crc32c))));
  header_bytes.clear();
  sf::AppendRecord(header, header_bytes);
  header_bytes.append(table);
  DCHECK_LE(header_bytes.size(), sf::kSectionAlignment);
  status_.Update(file_->WriteAt(0, header_bytes));
  return status_;
}

}  // namespace ferret

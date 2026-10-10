// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/segment.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/crc/crc32c.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "ferret/base/env.h"
#include "ferret/base/mapped_file.h"
#include "ferret/git/oid.h"
#include "ferret/index/internal/segment_format.h"
#include "ferret/index/rank_key.h"
#include "ferret/postings/ids.h"
#include "ferret/postings/posting_list.h"
#include "ferret/query/evaluator.h"
#include "ferret/query/trigram_query.h"
#include "ferret/symbols/symbol_kind.h"
#include "ferret/text/fold.h"
#include "ferret/text/language.h"
#include "ferret/text/lexer.h"
#include "ferret/text/literal_finder.h"
#include "ferret/text/trigram.h"

namespace ferret {
namespace {

namespace sf = segment_format;

absl::Status Corrupt(absl::string_view what) {
  return absl::DataLossError(absl::StrCat("corrupt segment: ", what));
}

uint32_t ReadU32(absl::string_view bytes, size_t index) {
  return sf::ReadRecord<uint32_t>(bytes, index);
}

// Calls `fn` for every trigram in `query`, possibly more than once.
void ForEachTrigram(const TrigramQuery& query,
                    absl::FunctionRef<void(Trigram)> fn) {
  for (const Trigram trigram : query.trigrams()) fn(trigram);
  for (const TrigramQuery& sub : query.subs()) ForEachTrigram(sub, fn);
}

}  // namespace

uint32_t Segment::FormatVersion() { return sf::kFormatVersion; }

absl::StatusOr<std::unique_ptr<Segment>> Segment::Open(Env& env,
                                                       absl::string_view path) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<MappedFile> file,
                        env.NewMappedFile(path));
  std::unique_ptr<Segment> segment(new Segment(std::move(file)));
  ABSL_RETURN_IF_ERROR(segment->ParseLayout()).SetPrepend() << path << ": ";
  ABSL_RETURN_IF_ERROR(segment->ValidateRecords()).SetPrepend() << path << ": ";
  segment->PackBasenames();
  return segment;
}

absl::Status Segment::ParseLayout() {
  const absl::string_view data = file_->data();
  const uint64_t size = data.size();
  if (size < sf::kSectionAlignment + sizeof(sf::SegmentFooter)) {
    return Corrupt(absl::StrCat("file too small: ", size, " bytes"));
  }

  const auto header = sf::ReadRecord<sf::SegmentHeader>(data, 0);
  if (std::memcmp(header.magic, sf::kHeaderMagic, sizeof(header.magic)) != 0) {
    return Corrupt("bad header magic");
  }
  if (static_cast<uint32_t>(absl::ComputeCrc32c(data.substr(
          0, offsetof(sf::SegmentHeader, crc32c)))) != header.crc32c) {
    return Corrupt("header checksum mismatch");
  }
  if (header.format_version != sf::kFormatVersion) {
    return absl::FailedPreconditionError(absl::StrCat(
        "unsupported segment format version ", header.format_version,
        "; this build reads version ", sf::kFormatVersion));
  }
  if (header.num_sections > sf::kMaxSections) {
    return Corrupt(absl::StrCat("too many sections: ", header.num_sections));
  }
  num_docs_ = header.num_docs;
  num_chunks_ = header.num_chunks;
  num_trigrams_ = header.num_trigrams;
  chunk_bytes_ = header.chunk_bytes;
  short_line_trigrams_ = (header.flags & sf::kHeaderShortLineTrigrams) != 0;

  const size_t footer_offset = size - sizeof(sf::SegmentFooter);
  const auto footer =
      sf::ReadRecord<sf::SegmentFooter>(data.substr(footer_offset), 0);
  if (std::memcmp(footer.magic, sf::kFooterMagic, sizeof(footer.magic)) != 0) {
    return Corrupt("bad footer magic");
  }
  if (footer.file_length != size) {
    return Corrupt(absl::StrCat("footer says ", footer.file_length,
                                " bytes, file has ", size));
  }
  const absl::string_view table =
      data.substr(sizeof(sf::SegmentHeader),
                  header.num_sections * sizeof(sf::SectionEntry));
  if (static_cast<uint32_t>(absl::ComputeCrc32c(table)) !=
      footer.section_table_crc32c) {
    return Corrupt("section table checksum mismatch");
  }

  // The sections this reader needs, each exactly once; unknown kinds are
  // skipped.
  struct Wanted {
    sf::SectionKind kind;
    absl::string_view* bytes;
    bool found;
  };
  std::array<Wanted, 8> wanted = {{
      {sf::SectionKind::kContent, &content_, false},
      {sf::SectionKind::kChunks, &chunks_, false},
      {sf::SectionKind::kDocs, &docs_, false},
      {sf::SectionKind::kOids, &oids_, false},
      {sf::SectionKind::kPaths, &paths_, false},
      {sf::SectionKind::kPostings, &postings_, false},
      {sf::SectionKind::kTrigrams, &trigrams_, false},
      {sf::SectionKind::kTrigramIndex, &trigram_index_, false},
  }};
  for (uint32_t i = 0; i < header.num_sections; ++i) {
    const auto entry = sf::ReadRecord<sf::SectionEntry>(table, i);
    if (entry.offset % sf::kSectionAlignment != 0 ||
        entry.offset < sf::kSectionAlignment || entry.offset > footer_offset ||
        entry.length > footer_offset - entry.offset) {
      return Corrupt(absl::StrCat("section ", i, " out of bounds"));
    }
    if (entry.kind == static_cast<uint32_t>(sf::SectionKind::kSymbols)) {
      if (has_symbols_) return Corrupt("duplicate symbols section");
      has_symbols_ = true;
      symbols_ = data.substr(entry.offset, entry.length);
    }
    if (entry.kind == static_cast<uint32_t>(sf::SectionKind::kCodeTokens)) {
      if (has_code_tokens_) return Corrupt("duplicate code tokens section");
      ABSL_RETURN_IF_ERROR(
          ParseCodeTokens(data.substr(entry.offset, entry.length)));
    }
    for (Wanted& w : wanted) {
      if (static_cast<uint32_t>(w.kind) != entry.kind) continue;
      if (w.found) {
        return Corrupt(absl::StrCat("duplicate section kind ", entry.kind));
      }
      w.found = true;
      *w.bytes = data.substr(entry.offset, entry.length);
    }
  }
  for (const Wanted& w : wanted) {
    if (!w.found) {
      return Corrupt(
          absl::StrCat("missing section kind ", static_cast<uint32_t>(w.kind)));
    }
  }

  const auto check_size = [](absl::string_view name, absl::string_view bytes,
                             uint64_t count, size_t record_size) {
    return bytes.size() == count * record_size
               ? absl::OkStatus()
               : Corrupt(absl::StrCat(name, " section has ", bytes.size(),
                                      " bytes, expected ", count, " * ",
                                      record_size));
  };
  ABSL_RETURN_IF_ERROR(
      check_size("chunks", chunks_, num_chunks_, sizeof(sf::ChunkRecord)));
  ABSL_RETURN_IF_ERROR(
      check_size("docs", docs_, num_docs_, sizeof(sf::DocRecord)));
  ABSL_RETURN_IF_ERROR(check_size("oids", oids_, num_docs_, Oid::kSize));
  ABSL_RETURN_IF_ERROR(check_size("trigrams", trigrams_, num_trigrams_,
                                  sizeof(sf::TrigramEntry)));
  ABSL_RETURN_IF_ERROR(check_size("trigram index", trigram_index_,
                                  sf::kTrigramIndexSize, sizeof(uint32_t)));
  if (content_.size() < sf::kContentPadding) {
    return Corrupt("content section lacks its padding");
  }
  content_.remove_suffix(sf::kContentPadding);

  // The bucket table must be a nondecreasing map onto [0, num_trigrams].
  if (ReadU32(trigram_index_, 0) != 0 ||
      ReadU32(trigram_index_, sf::kTrigramIndexSize - 1) != num_trigrams_) {
    return Corrupt("trigram index does not span the dictionary");
  }
  for (size_t b = 1; b < sf::kTrigramIndexSize; ++b) {
    if (ReadU32(trigram_index_, b) < ReadU32(trigram_index_, b - 1)) {
      return Corrupt("trigram index is not sorted");
    }
  }
  if (has_symbols_) {
    if (symbols_.size() < 8) return Corrupt("truncated symbols header");
    num_symbols_ = ReadU32(symbols_, 0);
    const uint32_t pool_bytes = ReadU32(symbols_, 1);
    const uint64_t record_bytes =
        uint64_t{num_symbols_} * sizeof(sf::SymbolRecord);
    if (record_bytes + pool_bytes + 8 != symbols_.size()) {
      return Corrupt("symbols section size mismatch");
    }
    symbol_pool_ = symbols_.substr(8 + record_bytes);
    symbols_ = symbols_.substr(8, record_bytes);
    absl::string_view previous_folded;
    absl::string_view previous_name;
    uint32_t previous_doc = 0;
    uint32_t previous_line = 0;
    for (uint32_t i = 0; i < num_symbols_; ++i) {
      const sf::SymbolRecord record =
          sf::ReadRecord<sf::SymbolRecord>(symbols_, i);
      if (record.folded_offset > symbol_pool_.size() ||
          record.folded_size > symbol_pool_.size() - record.folded_offset ||
          record.name_offset > symbol_pool_.size() ||
          record.name_size > symbol_pool_.size() - record.name_offset ||
          record.doc >= num_docs_ || record.line == 0 ||
          record.name_size == 0 ||
          (record.doc < num_docs_ &&
           record.line >
               sf::ReadRecord<sf::DocRecord>(docs_, record.doc).num_lines) ||
          record.kind > static_cast<uint8_t>(SymbolKind::kVariable) ||
          (record.flags & ~sf::kSymbolDefinition) != 0) {
        return Corrupt("symbols record out of bounds");
      }
      const absl::string_view folded =
          symbol_pool_.substr(record.folded_offset, record.folded_size);
      const absl::string_view name =
          symbol_pool_.substr(record.name_offset, record.name_size);
      if (Fold(name) != folded ||
          (i != 0 &&
           (folded < previous_folded ||
            (folded == previous_folded &&
             (name < previous_name ||
              (name == previous_name && (record.doc < previous_doc ||
                                         (record.doc == previous_doc &&
                                          record.line < previous_line)))))))) {
        return Corrupt("symbols dictionary is not sorted");
      }
      previous_folded = folded;
      previous_name = name;
      previous_doc = record.doc;
      previous_line = record.line;
    }
  }
  return absl::OkStatus();
}

absl::Status Segment::ParseCodeTokens(absl::string_view section) {
  if (section.size() < sizeof(sf::CodeTokensHeader)) {
    return Corrupt("truncated code tokens header");
  }
  code_tokens_bytes_ = static_cast<int64_t>(section.size());
  const auto header = sf::ReadRecord<sf::CodeTokensHeader>(section, 0);
  const uint64_t entry_bytes =
      uint64_t{header.num_tokens} * sizeof(sf::CodeTokenEntry);
  if (sizeof(header) + entry_bytes + header.pool_bytes +
          uint64_t{header.postings_bytes} !=
      section.size()) {
    return Corrupt("code tokens section size mismatch");
  }
  section.remove_prefix(sizeof(header));
  const absl::string_view entries = section.substr(0, entry_bytes);
  const absl::string_view pool = section.substr(entry_bytes, header.pool_bytes);
  const absl::string_view postings =
      section.substr(entry_bytes + header.pool_bytes);
  // Keys tile the pool and lists tile the posting bytes, each entry giving
  // where its own ends, with keys strictly increasing. Every list takes at
  // least one byte and ends the posting bytes if it is last. Lists are
  // decoded, and their ids checked, when read.
  uint32_t key_begin = 0;
  uint32_t list_begin = 0;
  absl::string_view previous;
  for (uint32_t i = 0; i < header.num_tokens; ++i) {
    const auto entry = sf::ReadRecord<sf::CodeTokenEntry>(entries, i);
    if (entry.key_end <= key_begin || entry.key_end > pool.size() ||
        entry.key_end - key_begin > header.max_token_bytes ||
        entry.postings_end <= list_begin ||
        entry.postings_end > postings.size()) {
      return Corrupt("code tokens entry out of place");
    }
    const absl::string_view key =
        pool.substr(key_begin, entry.key_end - key_begin);
    if (i != 0 && key <= previous) {
      return Corrupt("code tokens dictionary is not sorted");
    }
    previous = key;
    key_begin = entry.key_end;
    list_begin = entry.postings_end;
  }
  if (key_begin != pool.size() || list_begin != postings.size()) {
    return Corrupt("code tokens pool or lists have a tail");
  }
  code_token_entries_ = entries;
  code_token_pool_ = pool;
  code_token_postings_ = postings;
  num_code_tokens_ = header.num_tokens;
  max_code_token_bytes_ = header.max_token_bytes;
  has_code_tokens_ = true;
  return absl::OkStatus();
}

absl::string_view Segment::CodeTokenKey(uint32_t index) const {
  const uint32_t begin = index == 0 ? 0
                                    : sf::ReadRecord<sf::CodeTokenEntry>(
                                          code_token_entries_, index - 1)
                                          .key_end;
  const uint32_t end =
      sf::ReadRecord<sf::CodeTokenEntry>(code_token_entries_, index).key_end;
  return code_token_pool_.substr(begin, end - begin);
}

bool Segment::DecodeCodeTokenList(uint32_t index,
                                  std::vector<DocId>& docs) const {
  docs.clear();
  size_t pos = index == 0 ? 0
                          : sf::ReadRecord<sf::CodeTokenEntry>(
                                code_token_entries_, index - 1)
                                .postings_end;
  const size_t end =
      sf::ReadRecord<sf::CodeTokenEntry>(code_token_entries_, index)
          .postings_end;
  uint64_t doc = 0;
  while (pos < end) {
    uint64_t delta = 0;
    int shift = 0;
    while (true) {
      if (pos >= end || shift > 28) return false;
      const auto byte = static_cast<uint8_t>(code_token_postings_[pos++]);
      delta |= uint64_t{byte & 0x7Fu} << shift;
      shift += 7;
      if ((byte & 0x80) == 0) break;
    }
    doc += delta;
    if ((!docs.empty() && delta == 0) || doc >= num_docs_) return false;
    docs.push_back(static_cast<DocId>(doc));
  }
  return true;
}

bool Segment::FindCodeTokenDocs(absl::string_view folded_token,
                                std::vector<DocId>& docs) const {
  docs.clear();
  if (!has_code_tokens_ || folded_token.empty() ||
      folded_token.size() > max_code_token_bytes_) {
    return false;
  }
  uint32_t low = 0;
  uint32_t high = num_code_tokens_;
  while (low < high) {
    const uint32_t mid = low + (high - low) / 2;
    const absl::string_view key = CodeTokenKey(mid);
    if (key < folded_token) {
      low = mid + 1;
    } else if (key > folded_token) {
      high = mid;
    } else {
      return DecodeCodeTokenList(mid, docs);
    }
  }
  return true;
}

bool Segment::ForEachCodeToken(
    absl::FunctionRef<void(absl::string_view, absl::Span<const DocId>)> fn)
    const {
  if (!has_code_tokens_) return false;
  std::vector<DocId> docs;
  for (uint32_t i = 0; i < num_code_tokens_; ++i) {
    if (!DecodeCodeTokenList(i, docs)) return false;
    fn(CodeTokenKey(i), docs);
  }
  return true;
}

absl::Status Segment::ValidateRecords() const {
  // Documents tile the chunks in order, each document's chunks tile its
  // content in order, and hint paths tile the path pool.
  ChunkId next_chunk = 0;
  uint64_t next_path = 0;
  for (DocId d = 0; d < num_docs_; ++d) {
    const auto doc = sf::ReadRecord<sf::DocRecord>(docs_, d);
    if (doc.first_chunk != next_chunk ||
        doc.num_chunks > num_chunks_ - doc.first_chunk) {
      return Corrupt(absl::StrCat("doc ", d, " has chunks out of place"));
    }
    if (doc.path_offset != next_path) {
      return Corrupt(absl::StrCat("doc ", d, " has its path out of place"));
    }
    const uint64_t path_end =
        d + 1 < num_docs_
            ? sf::ReadRecord<sf::DocRecord>(docs_, d + 1).path_offset
            : paths_.size();
    if (path_end <= doc.path_offset || path_end > paths_.size()) {
      return Corrupt(absl::StrCat("doc ", d, " has a bad path"));
    }
    next_path = path_end;
    if (doc.tier >= kNumRankTiers) {
      return Corrupt(absl::StrCat("doc ", d, " has tier ", doc.tier));
    }
    uint64_t expected_offset = 0;
    for (uint32_t i = 0; i < doc.num_chunks; ++i) {
      const ChunkId c = doc.first_chunk + i;
      const auto chunk = sf::ReadRecord<sf::ChunkRecord>(chunks_, c);
      if (chunk.doc != d) {
        return Corrupt(absl::StrCat("chunk ", c, " names doc ", chunk.doc,
                                    ", expected ", d));
      }
      if (i == 0) {
        expected_offset = chunk.content_offset;
      } else if (chunk.content_offset != expected_offset) {
        return Corrupt(absl::StrCat("chunk ", c, " is not contiguous"));
      }
      if (chunk.content_size == 0 || chunk.content_offset > content_.size() ||
          chunk.content_size > content_.size() - chunk.content_offset) {
        return Corrupt(absl::StrCat("chunk ", c, " is out of bounds"));
      }
      if (chunk.first_line == 0) {
        return Corrupt(absl::StrCat("chunk ", c, " has line 0"));
      }
      expected_offset += chunk.content_size;
    }
    next_chunk += doc.num_chunks;
  }
  if (next_chunk != num_chunks_) {
    return Corrupt("chunks outside every document");
  }
  if (num_docs_ == 0 && !paths_.empty()) {
    return Corrupt("paths without documents");
  }
  // The writer never exceeds this, and PackBasenames() relies on it.
  if (paths_.size() > std::numeric_limits<uint32_t>::max()) {
    return Corrupt(absl::StrCat("paths section has ", paths_.size(),
                                " bytes, more than 2^32 - 1"));
  }
  return absl::OkStatus();
}

void Segment::ForEachSymbol(absl::FunctionRef<void(SegmentSymbol)> fn) const {
  for (uint32_t i = 0; i < num_symbols_; ++i) {
    const sf::SymbolRecord record =
        sf::ReadRecord<sf::SymbolRecord>(symbols_, i);
    fn({.name = symbol_pool_.substr(record.name_offset, record.name_size),
        .doc = record.doc,
        .line = record.line,
        .kind = static_cast<SymbolKind>(record.kind),
        .is_definition = (record.flags & sf::kSymbolDefinition) != 0});
  }
}

void Segment::FindSymbols(absl::string_view name, bool case_insensitive,
                          absl::FunctionRef<void(SegmentSymbol)> fn) const {
  const std::string folded = Fold(name);
  uint32_t left = 0;
  uint32_t right = num_symbols_;
  while (left < right) {
    const uint32_t middle = left + (right - left) / 2;
    const sf::SymbolRecord record =
        sf::ReadRecord<sf::SymbolRecord>(symbols_, middle);
    if (symbol_pool_.substr(record.folded_offset, record.folded_size) <
        folded) {
      left = middle + 1;
    } else {
      right = middle;
    }
  }
  for (uint32_t i = left; i < num_symbols_; ++i) {
    const sf::SymbolRecord record =
        sf::ReadRecord<sf::SymbolRecord>(symbols_, i);
    if (symbol_pool_.substr(record.folded_offset, record.folded_size) !=
        folded) {
      break;
    }
    const absl::string_view stored =
        symbol_pool_.substr(record.name_offset, record.name_size);
    if (case_insensitive || stored == name) {
      fn({.name = stored,
          .doc = record.doc,
          .line = record.line,
          .kind = static_cast<SymbolKind>(record.kind),
          .is_definition = (record.flags & sf::kSymbolDefinition) != 0});
    }
  }
}

void Segment::PackBasenames() {
  // No basename is longer than its path; shrink after an exact fill.
  basenames_.reserve(paths_.size());
  basename_ends_.reserve(num_docs_);
  for (DocId d = 0; d < num_docs_; ++d) {
    const absl::string_view path = doc(d).hint_path;
    const size_t slash = path.rfind('/');
    basenames_.append(
        slash == absl::string_view::npos ? path : path.substr(slash + 1));
    basename_ends_.push_back(static_cast<uint32_t>(basenames_.size()));
  }
  basenames_.shrink_to_fit();
}

void Segment::ForEachBasenameContaining(
    absl::string_view needle, LiteralFinder::Mode mode,
    absl::FunctionRef<void(DocId)> fn) const {
  if (needle.empty()) {
    for (DocId d = 0; d < num_docs_; ++d) fn(d);
    return;
  }
  const LiteralFinder finder(needle, mode);
  size_t from = 0;
  while (true) {
    const size_t hit = finder.Find(basenames_, from);
    if (hit == absl::string_view::npos) return;
    // The first basename ending after the hit holds its first byte. If the hit
    // crosses into the next basename, so does every later hit starting in
    // this one: either way, the scan resumes at the next basename.
    const auto end =
        std::upper_bound(basename_ends_.begin(), basename_ends_.end(), hit);
    DCHECK(end != basename_ends_.end());
    if (hit + needle.size() <= *end) {
      fn(static_cast<DocId>(end - basename_ends_.begin()));
    }
    from = *end;
  }
}

absl::string_view Segment::basename(DocId doc) const {
  DCHECK_LT(doc, num_docs_);
  const size_t begin = doc == 0 ? 0 : basename_ends_[doc - 1];
  return absl::string_view(basenames_)
      .substr(begin, basename_ends_[doc] - begin);
}

RankKey Segment::rank_key(DocId doc) const {
  DCHECK_LT(doc, num_docs_);
  const auto record = sf::ReadRecord<sf::DocRecord>(docs_, doc);
  return RankKey{
      .tier = static_cast<RankTier>(record.tier),
      .prior = record.prior,
      .oid = Oid::FromRaw(oids_.substr(size_t{doc} * Oid::kSize, Oid::kSize))};
}

void Segment::ForEachBasenameContainingAny(
    absl::Span<const absl::string_view> needles, LiteralFinder::Mode mode,
    DocId begin, DocId end, absl::FunctionRef<void(DocId)> fn) const {
  end = std::min(end, num_docs_);
  if (begin >= end) return;
  if (needles.empty()) {
    for (DocId d = begin; d < end; ++d) fn(d);
    return;
  }
  // The range's names are the bytes between the end of document begin - 1 and
  // the end of document end - 1. A hit cannot leave them, and one that
  // crosses into the next name is discarded as in ForEachBasenameContaining.
  const size_t first_byte = begin == 0 ? 0 : basename_ends_[begin - 1];
  const absl::string_view names =
      absl::string_view(basenames_).substr(0, basename_ends_[end - 1]);
  const auto ends_begin = basename_ends_.begin() + begin;
  const auto ends_end = basename_ends_.begin() + end;
  std::vector<DocId> hits;
  for (const absl::string_view needle : needles) {
    if (needle.empty()) {
      for (DocId d = begin; d < end; ++d) hits.push_back(d);
      continue;
    }
    const LiteralFinder finder(needle, mode);
    size_t from = first_byte;
    // Hits come in increasing order, so the name holding each is found by
    // galloping forward from the last, which costs O(1) when hits are dense
    // and O(log gap) when they are not, where a search from the start would
    // miss the cache on every hit.
    auto name_end = ends_begin;
    while (true) {
      const size_t hit = finder.Find(names, from);
      if (hit == absl::string_view::npos) break;
      size_t step = 1;
      auto low = name_end;
      auto high = name_end;
      while (high != ends_end && *high <= hit) {
        low = high;
        high = static_cast<size_t>(ends_end - high) > step
                   ? high + static_cast<std::ptrdiff_t>(step)
                   : ends_end;
        step *= 2;
      }
      name_end = std::upper_bound(low, high, hit);
      DCHECK(name_end != ends_end);
      if (hit + needle.size() <= *name_end) {
        hits.push_back(static_cast<DocId>(name_end - basename_ends_.begin()));
      }
      from = *name_end;
    }
  }
  if (needles.size() > 1) {
    std::sort(hits.begin(), hits.end());
    hits.erase(std::unique(hits.begin(), hits.end()), hits.end());
  }
  for (const DocId d : hits) fn(d);
}

int64_t Segment::basename_bytes() const {
  return static_cast<int64_t>(basenames_.capacity()) +
         static_cast<int64_t>(basename_ends_.capacity() * sizeof(uint32_t));
}

SegmentDoc Segment::doc(DocId doc) const {
  DCHECK_LT(doc, num_docs_);
  const auto record = sf::ReadRecord<sf::DocRecord>(docs_, doc);
  const uint64_t path_end =
      doc + 1 < num_docs_
          ? sf::ReadRecord<sf::DocRecord>(docs_, doc + 1).path_offset
          : paths_.size();
  SegmentDoc out;
  out.hint_path =
      paths_.substr(record.path_offset, path_end - record.path_offset);
  out.oid = Oid::FromRaw(oids_.substr(size_t{doc} * Oid::kSize, Oid::kSize));
  out.tier = static_cast<RankTier>(record.tier);
  out.prior = record.prior;
  out.language = static_cast<Language>(record.language);
  out.long_lines = (record.flags & sf::kDocLongLines) != 0;
  out.symbols_failed = (record.flags & sf::kSymbolsFailed) != 0;
  out.symbols_pending =
      !has_symbols_ || (record.flags & sf::kSymbolsPending) != 0;
  out.first_chunk = record.first_chunk;
  out.num_chunks = record.num_chunks;
  out.num_lines = record.num_lines;
  if (record.num_chunks > 0) {
    const auto first =
        sf::ReadRecord<sf::ChunkRecord>(chunks_, record.first_chunk);
    const auto last = sf::ReadRecord<sf::ChunkRecord>(
        chunks_, record.first_chunk + record.num_chunks - 1);
    out.content = content_.substr(
        first.content_offset,
        last.content_offset + last.content_size - first.content_offset);
  }
  return out;
}

SegmentChunk Segment::chunk(ChunkId chunk) const {
  DCHECK_LT(chunk, num_chunks_);
  const auto record = sf::ReadRecord<sf::ChunkRecord>(chunks_, chunk);
  const auto doc = sf::ReadRecord<sf::DocRecord>(docs_, record.doc);
  const auto first = sf::ReadRecord<sf::ChunkRecord>(chunks_, doc.first_chunk);
  return SegmentChunk{
      .doc = record.doc,
      .content = content_.substr(record.content_offset, record.content_size),
      .offset_in_doc = record.content_offset - first.content_offset,
      .first_line = record.first_line,
      .entry_state =
          static_cast<LexState>(record.flags & sf::kChunkLexStateMask),
  };
}

DocId Segment::doc_of_chunk(ChunkId chunk) const {
  DCHECK_LT(chunk, num_chunks_);
  return sf::ReadRecord<sf::ChunkRecord>(chunks_, chunk).doc;
}

Oid Segment::doc_oid(DocId doc) const {
  DCHECK_LT(doc, num_docs_);
  return Oid::FromRaw(oids_.substr(size_t{doc} * Oid::kSize, Oid::kSize));
}

std::pair<ChunkId, uint32_t> Segment::doc_chunks(DocId doc) const {
  DCHECK_LT(doc, num_docs_);
  const auto record = sf::ReadRecord<sf::DocRecord>(docs_, doc);
  return {record.first_chunk, record.num_chunks};
}

void Segment::FindRepeatedOids() const {
  absl::flat_hash_set<absl::string_view> seen;
  seen.reserve(num_docs_);
  std::vector<uint64_t> repeated((size_t{num_docs_} + 63) / 64);
  bool any = false;
  for (uint32_t doc = 0; doc < num_docs_; ++doc) {
    if (seen.insert(oids_.substr(size_t{doc} * Oid::kSize, Oid::kSize))
            .second) {
      continue;
    }
    repeated[doc / 64] |= uint64_t{1} << (doc % 64);
    any = true;
  }
  if (any) repeated_ = std::move(repeated);
}

bool Segment::has_duplicate_oids() const {
  absl::call_once(repeated_once_, [&] { FindRepeatedOids(); });
  return !repeated_.empty();
}

bool Segment::is_repeated_oid(DocId doc) const {
  if (!has_duplicate_oids()) return false;
  DCHECK_LT(doc, num_docs_);
  return (repeated_[doc / 64] >> (doc % 64) & 1) != 0;
}

Trigram Segment::trigram_at(uint32_t index) const {
  DCHECK_LT(index, num_trigrams_);
  return sf::ReadRecord<sf::TrigramEntry>(trigrams_, index).trigram_and_flags &
         kTrigramMask;
}

absl::StatusOr<PostingListView> Segment::FindPostingList(
    Trigram trigram) const {
  if ((trigram & ~kTrigramMask) != 0) return PostingListView();
  const size_t bucket = trigram >> 8;
  size_t low = ReadU32(trigram_index_, bucket);
  size_t high = ReadU32(trigram_index_, bucket + 1);
  while (low < high) {
    const size_t mid = low + (high - low) / 2;
    const auto entry = sf::ReadRecord<sf::TrigramEntry>(trigrams_, mid);
    const Trigram found = entry.trigram_and_flags & kTrigramMask;
    if (found < trigram) {
      low = mid + 1;
    } else if (found > trigram) {
      high = mid;
    } else if (IsInlinedPostingList(entry.count)) {
      const ChunkId ids[2] = {static_cast<ChunkId>(entry.postings),
                              static_cast<ChunkId>(entry.postings >> 32)};
      const auto bad_inline = [trigram] {
        return Corrupt(absl::StrCat("bad inline postings for trigram ",
                                    TrigramToString(trigram)));
      };
      if (entry.count == 0 || (entry.count == 1 && ids[1] != 0)) {
        return bad_inline();
      }
      absl::StatusOr<PostingListView> view =
          PostingListView::FromInlined(absl::MakeConstSpan(ids, entry.count));
      if (!view.ok() || ids[entry.count - 1] >= num_chunks_) {
        return bad_inline();
      }
      return view;
    } else {
      if (entry.postings >= postings_.size()) {
        return Corrupt(absl::StrCat("postings offset out of bounds for ",
                                    TrigramToString(trigram)));
      }
      absl::StatusOr<PostingListView> view =
          PostingListView::FromEncoded(postings_.substr(entry.postings));
      if (!view.ok()) {
        return Corrupt(absl::StrCat("posting list of ",
                                    TrigramToString(trigram), ": ",
                                    view.status().message()));
      }
      if (view->size() != entry.count ||
          view->block_last(view->num_blocks() - 1) >= num_chunks_) {
        return Corrupt(absl::StrCat("posting list of ",
                                    TrigramToString(trigram),
                                    " disagrees with its dictionary entry"));
      }
      return view;
    }
  }
  return PostingListView();
}

absl::StatusOr<TrigramQueryEvaluator<PostingListView>> Segment::NewEvaluator(
    const TrigramQuery& query) const {
  std::vector<Trigram> trigrams;
  ForEachTrigram(query, [&](Trigram trigram) { trigrams.push_back(trigram); });
  absl::flat_hash_map<Trigram, PostingListView> lists;
  for (const Trigram trigram : trigrams) {
    if (lists.contains(trigram)) continue;
    ABSL_ASSIGN_OR_RETURN(const PostingListView list, FindPostingList(trigram));
    lists.emplace(trigram, list);
  }
  return TrigramQueryEvaluator<PostingListView>(
      query, [&](Trigram trigram) { return lists.at(trigram); });
}

absl::Status Segment::VerifyChecksums() const {
  const absl::string_view data = file_->data();
  const auto header = sf::ReadRecord<sf::SegmentHeader>(data, 0);
  const absl::string_view table =
      data.substr(sizeof(sf::SegmentHeader),
                  header.num_sections * sizeof(sf::SectionEntry));
  for (uint32_t i = 0; i < header.num_sections; ++i) {
    const auto entry = sf::ReadRecord<sf::SectionEntry>(table, i);
    const absl::string_view bytes = data.substr(entry.offset, entry.length);
    if (static_cast<uint32_t>(absl::ComputeCrc32c(bytes)) != entry.crc32c) {
      return Corrupt(absl::StrCat("checksum mismatch in section ", i, " (kind ",
                                  entry.kind, ")"));
    }
  }
  return absl::OkStatus();
}

}  // namespace ferret

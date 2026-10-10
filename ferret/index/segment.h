// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Read access to an immutable segment file (.fseg), mapped read-only.
//
// Validation is split by cost. Open() checks the header, the section table
// and the footer, and every chunk and document record and hint path, in
// time linear in the number of chunks and documents. After that, doc() and
// chunk() cannot fail and never return bytes outside the file. Dictionary
// entries and posting lists are checked when FindPostingList() reads them,
// so opening does not touch the dictionary or the postings. Corrupt content
// or postings bytes can yield wrong results but never out-of-bounds reads;
// VerifyChecksums() detects them.
//
// Open() also packs the basename of every hint path into memory, so that a
// query can find the documents whose file name contains a literal without
// touching the mapping.

#ifndef FERRET_INDEX_SEGMENT_H_
#define FERRET_INDEX_SEGMENT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "ferret/base/env.h"
#include "ferret/base/mapped_file.h"
#include "ferret/git/oid.h"
#include "ferret/index/rank_key.h"
#include "ferret/postings/ids.h"
#include "ferret/postings/posting_list.h"
#include "ferret/query/evaluator.h"
#include "ferret/query/trigram_query.h"
#include "ferret/symbols/symbol_kind.h"
#include "ferret/text/language.h"
#include "ferret/text/lexer.h"
#include "ferret/text/literal_finder.h"
#include "ferret/text/trigram.h"

namespace ferret {

// A document (blob) of a segment. Views point into the mapping.
struct SegmentDoc {
  absl::string_view hint_path;
  // All of the document's bytes.
  absl::string_view content;
  Oid oid;
  RankTier tier = RankTier::kSource;
  uint16_t prior = 0;
  Language language = Language::kUnknown;
  bool long_lines = false;
  bool symbols_failed = false;
  bool symbols_pending = false;
  // The document's chunks are [first_chunk, first_chunk + num_chunks).
  ChunkId first_chunk = 0;
  uint32_t num_chunks = 0;
  uint32_t num_lines = 0;
};

// A chunk of a segment. Views point into the mapping.
struct SegmentChunk {
  DocId doc = 0;
  absl::string_view content;
  // The chunk's offset in its document's content.
  uint64_t offset_in_doc = 0;
  // The 1-based line number of the chunk's first byte.
  uint32_t first_line = 1;
  // The lexer state at the chunk's first byte.
  LexState entry_state = LexState::kCode;
};

struct SegmentSymbol {
  absl::string_view name;
  DocId doc;
  uint32_t line;
  SymbolKind kind;
  bool is_definition;
};

// Immutable after Open(), so thread-safe.
class Segment final {
 public:
  // Maps and validates the segment at `path`. DataLoss if the file is not a
  // well-formed segment, with the reason; FailedPrecondition for a format
  // version this build does not read.
  static absl::StatusOr<std::unique_ptr<Segment>> Open(Env& env,
                                                       absl::string_view path);

  Segment(const Segment&) = delete;
  Segment& operator=(const Segment&) = delete;

  // Returns the .fseg format version this build writes and reads.
  static uint32_t FormatVersion();

  uint32_t num_docs() const { return num_docs_; }
  uint32_t num_chunks() const { return num_chunks_; }
  uint32_t num_trigrams() const { return num_trigrams_; }
  // Whether every line of two folded bytes "ab" is indexed as the pseudo-
  // trigram "ab\n" of its chunk (TrigramExtractor::ExtractShortLines). If so,
  // every chunk holding a line that contains a two-byte folded literal is in
  // the union of the posting lists of the trigrams "abX" and "Xab", with X
  // any byte other than '\n' that folding does not rewrite, and of "ab\n".
  bool has_short_line_trigrams() const { return short_line_trigrams_; }
  // The chunk size target the segment was built with.
  uint32_t chunk_bytes() const { return chunk_bytes_; }

  // Requires doc < num_docs() (DCHECKed). O(1).
  SegmentDoc doc(DocId doc) const;

  // Requires chunk < num_chunks() (DCHECKed). O(1).
  SegmentChunk chunk(ChunkId chunk) const;

  // Single fields of the records above, for scans that visit many documents
  // and need one or two of them: one record read each instead of several.
  // Each requires its argument in range (DCHECKed). O(1).
  DocId doc_of_chunk(ChunkId chunk) const;
  Oid doc_oid(DocId doc) const;
  // The chunks of `doc`: [first, first + count).
  std::pair<ChunkId, uint32_t> doc_chunks(DocId doc) const;
  // Whether two documents share an Oid, and whether `doc` shares the Oid of a
  // document with a lower DocId. Copies of an Oid have identical content, and
  // so identical candidate chunks. Ingestion adds a blob once, so no segment it
  // writes has copies, but nothing in the format forbids them. Computed on the
  // first call, O(num_docs) once; thread-safe.
  bool has_duplicate_oids() const;
  bool is_repeated_oid(DocId doc) const;

  // The last component of document `doc`'s hint path, from the packed
  // basenames: the same bytes as doc(doc).hint_path after its last '/', with
  // no record reads. Requires doc < num_docs() (DCHECKed). O(1).
  absl::string_view basename(DocId doc) const;

  // The RankKey of document `doc`: doc()'s tier, prior and oid, reading only
  // those fields. Requires doc < num_docs() (DCHECKed). O(1).
  RankKey rank_key(DocId doc) const;

  bool has_symbols() const { return has_symbols_; }
  // Calls fn for each occurrence of name, using binary search on the folded
  // name. A legacy segment without the section has no stored occurrences.
  void FindSymbols(absl::string_view name, bool case_insensitive,
                   absl::FunctionRef<void(SegmentSymbol)> fn) const;
  void ForEachSymbol(absl::FunctionRef<void(SegmentSymbol)> fn) const;

  bool has_code_tokens() const { return has_code_tokens_; }
  // Sets `docs` to the increasing DocIds of every document where a code
  // identifier folds to `folded_token` (the Fold() of the identifier): a
  // superset of the documents where some case variant of the token is labeled
  // kCodeIdentifier by MatchClassifier. Returns false, leaving `docs`
  // unspecified, when the segment cannot say: it has no kCodeTokens section (a
  // legacy segment), the token is empty or longer than the section's limit, or
  // the list is corrupt. Returns true with no ids for a token that never
  // occurs. O(log(tokens) + ids).
  bool FindCodeTokenDocs(absl::string_view folded_token,
                         std::vector<DocId>& docs) const;

  // Calls fn with each stored folded token in increasing order and its
  // DocIds. Returns false, possibly after some calls, if the segment has no
  // section or a list is corrupt. For tests and statistics.
  bool ForEachCodeToken(
      absl::FunctionRef<void(absl::string_view, absl::Span<const DocId>)> fn)
      const;
  // The size of the kCodeTokens section, 0 if absent.
  int64_t code_tokens_bytes() const { return code_tokens_bytes_; }

  // Returns the trigram at `index` in dictionary order. Requires index <
  // num_trigrams() (DCHECKed). O(1).
  Trigram trigram_at(uint32_t index) const;

  // Returns the posting list of `trigram`, empty if no chunk contains it.
  // DataLoss if its dictionary entry or encoding is corrupt.
  // O(log(trigrams in the bucket) + blocks in the list).
  absl::StatusOr<PostingListView> FindPostingList(Trigram trigram) const;

  // Returns an evaluator of `query` over this segment's posting lists.
  // DataLoss if any of the lists is corrupt. The evaluator must not outlive
  // the segment.
  absl::StatusOr<TrigramQueryEvaluator<PostingListView>> NewEvaluator(
      const TrigramQuery& query) const;

  // Calls `fn` once for each document whose basename (the last component of
  // its hint path) contains `needle`, compared as LiteralFinder does in
  // `mode`, in increasing DocId order. Scans the packed basenames, not the
  // mapping: O(basename bytes + log(num_docs()) per document reported), or
  // O(num_docs()) for an empty needle, which every basename contains.
  void ForEachBasenameContaining(absl::string_view needle,
                                 LiteralFinder::Mode mode,
                                 absl::FunctionRef<void(DocId)> fn) const;

  // Calls `fn` once for each document in [begin, end) whose basename contains
  // at least one of `needles` (compared in `mode`), in increasing DocId order.
  // An empty `needles` is every document. Like ForEachBasenameContaining, over
  // a range of documents and several needles: the ranges of a partition
  // report exactly the documents that one call over the whole segment does.
  // O(needles * range basename bytes + log(num_docs()) per hit).
  void ForEachBasenameContainingAny(absl::Span<const absl::string_view> needles,
                                    LiteralFinder::Mode mode, DocId begin,
                                    DocId end,
                                    absl::FunctionRef<void(DocId)> fn) const;

  // Returns the heap bytes held by the packed basenames: about their total
  // length plus 4 bytes per document.
  int64_t basename_bytes() const;

  // Checks the CRC32C of every section: DataLoss naming the first section
  // that does not match. Reads the whole file.
  absl::Status VerifyChecksums() const;

  // The mapping, for madvise.
  const MappedFile& file() const { return *file_; }

 private:
  explicit Segment(std::unique_ptr<MappedFile> file) : file_(std::move(file)) {}

  // Validates the header, footer and section table and sets the fields
  // below; then the records.
  absl::Status ParseLayout();
  absl::Status ValidateRecords() const;
  absl::Status ParseCodeTokens(absl::string_view section);
  // Decodes the list of the `index`-th code token; false if it is corrupt.
  bool DecodeCodeTokenList(uint32_t index, std::vector<DocId>& docs) const;
  absl::string_view CodeTokenKey(uint32_t index) const;
  // Requires validated records.
  void PackBasenames();

  const std::unique_ptr<MappedFile> file_;
  // Every document's basename in DocId order, without separators; document d
  // owns [basename_ends_[d - 1], basename_ends_[d]), starting from 0. The
  // total fits in 32 bits because it is at most the paths section's size.
  std::string basenames_;
  std::vector<uint32_t> basename_ends_;
  uint32_t num_docs_ = 0;
  uint32_t num_chunks_ = 0;
  uint32_t num_trigrams_ = 0;
  uint32_t chunk_bytes_ = 0;
  bool short_line_trigrams_ = false;
  // Section bytes. content_ excludes the trailing padding.
  absl::string_view content_;
  absl::string_view chunks_;
  absl::string_view docs_;
  absl::string_view oids_;
  absl::string_view paths_;
  absl::string_view postings_;
  absl::string_view trigrams_;
  absl::string_view trigram_index_;
  absl::string_view symbols_;
  absl::string_view symbol_pool_;
  uint32_t num_symbols_ = 0;
  bool has_symbols_ = false;
  // The kCodeTokens entries, key pool and posting bytes.
  absl::string_view code_token_entries_;
  absl::string_view code_token_pool_;
  absl::string_view code_token_postings_;
  uint32_t num_code_tokens_ = 0;
  uint32_t max_code_token_bytes_ = 0;
  int64_t code_tokens_bytes_ = 0;
  bool has_code_tokens_ = false;
  void FindRepeatedOids() const;
  // The documents with the Oid of an earlier one, as a bitmap over DocIds;
  // empty if there are none. Written once under repeated_once_.
  mutable absl::once_flag repeated_once_;
  mutable std::vector<uint64_t> repeated_;
};

}  // namespace ferret

#endif  // FERRET_INDEX_SEGMENT_H_

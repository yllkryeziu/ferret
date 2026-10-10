// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Writes a segment file (.fseg) from documents that arrive in rank order.
//
// Content streams to the file as documents are added; chunk, document and
// path records are kept until Finish() writes the remaining sections. The
// normal path accumulates postings; the already-chunked path supplies sorted
// posting lists at FinishMerged(). Both share the section and footer writer.
// Memory on the normal path is about 4 bytes per posting plus 48 bytes per
// chunk.

#ifndef FERRET_INDEX_SEGMENT_WRITER_H_
#define FERRET_INDEX_SEGMENT_WRITER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "ferret/base/env.h"
#include "ferret/base/thread_pool.h"
#include "ferret/index/internal/code_tokens.h"
#include "ferret/index/internal/posting_accumulator.h"
#include "ferret/index/internal/segment_format.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/segment.h"
#include "ferret/symbols/symbol_extractor.h"
#include "ferret/text/language.h"
#include "ferret/text/lexer.h"
#include "ferret/text/trigram.h"
#include "ferret/text/trigram_extractor.h"

namespace ferret {

struct SegmentWriterOptions {
  // The chunk size target for SplitIntoChunks. In [1, UINT32_MAX].
  int64_t chunk_bytes = int64_t{8} * 1024;
  // Postings buffered before a partition is sorted (PostingAccumulator).
  size_t max_partition_postings = size_t{1} << 24;
  // Disabling this writes a legacy segment, useful for compatibility tests.
  bool write_symbols = true;
  // A memtable supplies symbols extracted before its flush.
  bool extract_symbols = true;
  // Disabling this omits the kCodeTokens section, as an older build wrote
  // segments; useful for compatibility tests.
  bool write_code_tokens = true;
  // Disabling this omits the pseudo-trigrams of lines of two folded bytes and
  // the header flag, as an older build wrote segments; useful for
  // compatibility tests.
  bool write_short_line_trigrams = true;
  // Workers for the parts of Finish() that can run in parallel on a segment
  // built from partitions (AddPartition). Not owned; may be null, which
  // finishes on the calling thread. `finish_threads` includes the caller.
  ThreadPool* finish_pool = nullptr;
  int finish_threads = 1;
};

// One blob to index.
struct SegmentDocument {
  // The repo-relative path the blob is indexed under. Not empty.
  absl::string_view hint_path;
  // Shorter than UINT32_MAX bytes.
  absl::string_view content;
  // The caller computes it, including the Oid of `content`.
  RankKey rank_key;
  Language language = Language::kUnknown;
  bool long_lines = false;
};

// Thread time spent in the phases of a build, for profiling. Nanoseconds.
struct SegmentWriterTimings {
  // Add(): SplitIntoChunks and NumLines.
  int64_t chunking_ns = 0;
  // Add(): TrigramExtractor::Extract (folding and dedup).
  int64_t trigrams_ns = 0;
  // Add(): PostingAccumulator::AddChunk, including partition sorts.
  int64_t postings_add_ns = 0;
  // Add(): the code-token lexer and token map.
  int64_t code_tokens_ns = 0;
  // Add(): Append(content), including its CRC and the file write.
  int64_t content_ns = 0;
  // Add(): in-writer symbol extraction (zero for a bulk build).
  int64_t symbols_ns = 0;
  // Finish(): the chunk, doc, oid and path sections.
  int64_t tail_sections_ns = 0;
  // Finish(): sorting and writing the symbol section.
  int64_t symbols_write_ns = 0;
  // Finish(): sorting and writing the code-token section.
  int64_t code_tokens_write_ns = 0;
  // Finish(): merging runs into lists.
  int64_t lists_merge_ns = 0;
  // Finish(): encoding and appending lists.
  int64_t lists_encode_ns = 0;
  // Finish(): dictionary, trigram index, header and footer.
  int64_t dictionary_ns = 0;
  // Finish(): fsync.
  int64_t sync_ns = 0;
};

// One chunk of a document whose chunks were computed elsewhere.
struct PrecomputedChunk {
  // Byte offset in the document's content, and length.
  uint32_t offset_in_doc = 0;
  uint32_t size = 0;
  // 1-based number of the chunk's first line.
  uint32_t first_line = 1;
  // The lex state at the chunk's first byte.
  LexState entry_state = LexState::kCode;
};

struct SegmentStats {
  uint32_t num_docs = 0;
  uint32_t num_chunks = 0;
  uint32_t num_trigrams = 0;
  uint64_t num_postings = 0;
  uint64_t content_bytes = 0;
  // The size of the kPostings section.
  uint64_t postings_bytes = 0;
  uint64_t file_bytes = 0;
};

// Thread-compatible.
class SegmentWriter final {
 public:
  // Creates the file at `path`, replacing any file there. InvalidArgument if
  // options.chunk_bytes is out of range. `env` must outlive the writer.
  static absl::StatusOr<std::unique_ptr<SegmentWriter>> Create(
      Env& env, absl::string_view path,
      const SegmentWriterOptions& options = {});

  SegmentWriter(const SegmentWriter&) = delete;
  SegmentWriter& operator=(const SegmentWriter&) = delete;
  ~SegmentWriter();

  // Appends `doc`, which becomes the next DocId. Documents must arrive in
  // nondecreasing rank_key order. Returns InvalidArgument, and adds nothing,
  // for a document out of order, with an empty hint path, or with content
  // too large, and ResourceExhausted when the segment would exceed 2^32 - 1
  // documents or chunks or 4 GiB of paths. An I/O error is returned by this
  // and every later call.
  // When occurrences were extracted by a bulk reader, set extract_symbols to
  // false and follow with AddSymbols() for this document.
  absl::Status Add(const SegmentDocument& doc, bool extract_symbols = true);

  // Writes the rest of the file, syncs it and closes it. The new directory
  // entry is durable only after the caller syncs the directory. The writer
  // accepts no calls afterwards.
  absl::StatusOr<SegmentStats> Finish();

  // Copies an already-chunked document without extracting trigrams or
  // recomputing metadata. Chunks must cover content in order; documents must
  // arrive in rank order. Views need only remain alive for this call.
  absl::Status AddChunked(const SegmentDoc& doc,
                          absl::Span<const SegmentChunk> chunks);

  // Attaches occurrences to the last appended document. Used by compaction
  // to copy stored symbols without parsing the document again.
  void AddSymbols(absl::Span<const ExtractedSymbol> symbols);
  // Number of documents whose extraction failed in this writer.
  uint32_t failed_symbols() const { return failed_symbols_; }
  void MarkSymbolsFailed();

  const SegmentWriterTimings& timings() const { return timings_; }

  // Appends `doc` like Add(), but from work done elsewhere: its chunks are
  // `chunks` (what SplitIntoChunks would give), `num_lines` its line count,
  // and its postings and code tokens come later from SetPostings() and
  // SetCodeTokenMaps(), under the provisional ids the caller chose. Documents
  // must arrive in rank order. Precomputed documents and Add() cannot be
  // mixed. The errors of Add().
  absl::Status AddPrecomputed(const SegmentDocument& doc, uint32_t num_lines,
                              absl::Span<const PrecomputedChunk> chunks);

  // Supplies the postings of the precomputed documents: runs of lists over
  // provisional chunk ids, and for each provisional id the chunk id it has in
  // this segment. Call once, before Finish(), after the last
  // AddPrecomputed().
  void SetPostings(std::vector<segment_internal::PostingAccumulator::Run> runs,
                   std::vector<ChunkId> chunk_of_provisional);

  // Supplies the code tokens of the precomputed documents, from maps keyed by
  // provisional document ids, and the DocId of each. Same calling rules as
  // SetPostings().
  void SetCodeTokenMaps(std::vector<segment_internal::CodeTokenMap> maps,
                        std::vector<DocId> doc_of_provisional);

  // Like AddSymbols() and MarkSymbolsFailed() for document `doc`.
  void AddSymbolsFor(DocId doc, absl::Span<const ExtractedSymbol> symbols);
  void MarkSymbolsFailedFor(DocId doc);

  // Number of documents appended so far.
  uint32_t num_docs() const { return num_docs_; }

  // Finishes a copied segment. Calls `next_list` until it returns false;
  // each true result supplies a nonempty list with strictly increasing ids
  // below the output chunk count, in strictly increasing trigram order. The
  // callback runs synchronously. Returns InvalidArgument for invalid lists.
  // `short_line_trigrams` says whether the lists include the pseudo-trigrams
  // of every chunk's short lines, as for Finish(); it is true only if every
  // copied segment had them.
  absl::StatusOr<SegmentStats> FinishMerged(
      absl::FunctionRef<absl::StatusOr<bool>(Trigram&, std::vector<ChunkId>&)>
          next_list,
      bool short_line_trigrams = false);

 private:
  SegmentWriter(std::unique_ptr<WritableFile> file,
                const SegmentWriterOptions& options);

  // File output. Appends are buffered, and the bytes appended while a
  // section is open extend its CRC.
  absl::Status Append(absl::string_view data);
  absl::Status AppendZeros(uint64_t count);
  absl::Status FlushBuffer();
  // Pads the file to the next multiple of kSectionAlignment and opens a
  // section of `kind` there.
  absl::Status BeginSection(segment_format::SectionKind kind);
  void EndSection();
  // The number of bytes appended to the open section so far.
  uint64_t section_size() const;
  absl::Status WriteSection(segment_format::SectionKind kind,
                            absl::string_view bytes);
  // Writes everything after the content, up to and including the Sync().
  absl::StatusOr<SegmentStats> FinishWithLists(
      absl::FunctionRef<absl::Status(SegmentStats&)> write_lists);
  absl::Status WriteTail(
      SegmentStats& stats,
      absl::FunctionRef<absl::Status(SegmentStats&)> write_lists);
  absl::Status WritePostingList(Trigram trigram, absl::Span<const ChunkId> ids,
                                SegmentStats& stats);
  // Appends the dictionary entry and payload of an encoded list.
  absl::Status WriteEncodedList(Trigram trigram, size_t count,
                                uint64_t inline_postings,
                                absl::string_view encoded, SegmentStats& stats);
  absl::Status WriteDictionary(SegmentStats& stats);
  absl::Status WriteSymbols();
  absl::Status WriteCodeTokens();
  // Records the code tokens of the last appended document's chunk, lexing it
  // from `entry` as the verifier does, and returns the state after it.
  LexState AddCodeTokens(Language language, absl::string_view text,
                         LexState entry);
  absl::Status WriteHeaderAndFooter(const SegmentStats& stats);

  const std::unique_ptr<WritableFile> file_;
  const SegmentWriterOptions options_;
  // The first error; once set, every call returns it.
  absl::Status status_;
  bool finished_ = false;
  // Whether the segment will carry kHeaderShortLineTrigrams.
  bool short_line_trigrams_ = options_.write_short_line_trigrams;

  std::string buffer_;
  // The file size, including buffer_.
  uint64_t file_size_ = 0;
  std::optional<segment_format::SectionEntry> open_section_;
  uint32_t open_section_crc_ = 0;
  std::vector<segment_format::SectionEntry> sections_;

  SegmentWriterTimings timings_;
  TrigramExtractor extractor_;
  SymbolExtractor symbol_extractor_;
  struct PendingSymbol {
    std::string folded;
    ExtractedSymbol symbol;
    DocId doc;
  };
  std::vector<PendingSymbol> symbols_;
  uint32_t failed_symbols_ = 0;
  // Folded code-identifier tokens to the ascending DocIds that contain one.
  segment_internal::CodeTokenMap code_tokens_;
  // Precomputed documents: their code tokens and postings, over provisional
  // ids, and the maps to this segment's ids.
  std::vector<segment_internal::CodeTokenMap> precomputed_tokens_;
  std::vector<DocId> doc_of_provisional_;
  std::vector<segment_internal::PostingAccumulator::Run> precomputed_runs_;
  std::vector<ChunkId> chunk_of_provisional_;
  std::string token_scratch_;
  std::vector<Trigram> chunk_trigrams_;
  segment_internal::PostingAccumulator postings_;
  std::optional<RankKey> last_rank_key_;
  uint32_t num_docs_ = 0;
  uint32_t num_chunks_ = 0;
  uint64_t content_bytes_ = 0;
  std::string chunk_records_;
  std::string doc_records_;
  std::string oids_;
  std::string paths_;
  std::string dictionary_;
  enum class DocumentMode { kUnset, kContent, kChunked, kPrecomputed };
  DocumentMode document_mode_ = DocumentMode::kUnset;
};

}  // namespace ferret

#endif  // FERRET_INDEX_SEGMENT_WRITER_H_

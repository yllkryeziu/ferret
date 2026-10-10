// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Does the per-document work of SegmentWriter::Add() before the document's
// place in the segment is known.
//
// The bulk build reads each file once, on many threads, and cannot rank the
// files until all have been read. A DocumentPrecomputer chunks a file,
// extracts its trigrams and code tokens, and records them under provisional
// ids: chunks numbered from a counter shared by the threads, documents named
// by the caller. SegmentWriter::AddPrecomputed(), SetPostings() and
// SetCodeTokenMaps() later place the documents in rank order and map the
// provisional ids, which gives the bytes Add() would have written.

#ifndef FERRET_INDEX_DOC_PRECOMPUTER_H_
#define FERRET_INDEX_DOC_PRECOMPUTER_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "ferret/index/internal/code_tokens.h"
#include "ferret/index/internal/posting_accumulator.h"
#include "ferret/index/segment_writer.h"
#include "ferret/text/language.h"
#include "ferret/text/trigram_extractor.h"

namespace ferret {

// What a DocumentPrecomputer learns about one document.
struct PrecomputedDocument {
  uint32_t num_lines = 0;
  // The provisional id of the first chunk; the others follow it.
  uint32_t first_chunk = 0;
  std::vector<PrecomputedChunk> chunks;
};

// Thread-compatible: one per thread.
class DocumentPrecomputer final {
 public:
  // `options` supplies chunk_bytes, write_code_tokens and
  // write_short_line_trigrams. `extractor` and `next_chunk` must outlive this
  // object; the counter is shared by every precomputer of a build.
  DocumentPrecomputer(const SegmentWriterOptions& options,
                      TrigramExtractor& extractor,
                      std::atomic<uint32_t>& next_chunk);

  DocumentPrecomputer(const DocumentPrecomputer&) = delete;
  DocumentPrecomputer& operator=(const DocumentPrecomputer&) = delete;

  // Chunks `content` and records its trigrams and code tokens. Its tokens
  // are recorded under `provisional_doc`, which must be unique across
  // calls. InvalidArgument for
  // content that is too large, ResourceExhausted when the chunk counter would
  // pass 2^32 - 1.
  absl::StatusOr<PrecomputedDocument> Precompute(absl::string_view content,
                                                 Language language,
                                                 uint32_t provisional_doc);

  // The runs of postings so far, in order, leaving none behind.
  std::vector<segment_internal::PostingAccumulator::Run> TakeRuns();

  // The code tokens of every document so far.
  segment_internal::CodeTokenMap TakeCodeTokens();

  // Thread time per phase, in the fields Add() fills.
  const SegmentWriterTimings& timings() const { return timings_; }

 private:
  const SegmentWriterOptions options_;
  SegmentWriterTimings timings_;
  TrigramExtractor& extractor_;
  std::atomic<uint32_t>& next_chunk_;
  std::vector<Trigram> chunk_trigrams_;
  std::string token_scratch_;
  segment_internal::CodeTokenMap tokens_;
  segment_internal::PostingAccumulator postings_;
};

}  // namespace ferret

#endif  // FERRET_INDEX_DOC_PRECOMPUTER_H_

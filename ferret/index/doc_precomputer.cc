// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/doc_precomputer.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "ferret/index/internal/build_util.h"
#include "ferret/index/internal/code_tokens.h"
#include "ferret/index/internal/posting_accumulator.h"
#include "ferret/index/segment_writer.h"
#include "ferret/text/chunker.h"
#include "ferret/text/language.h"
#include "ferret/text/lexer.h"
#include "ferret/text/trigram_extractor.h"

namespace ferret {
namespace {

using segment_internal::ScopedTimer;

constexpr uint64_t kMaxId = std::numeric_limits<uint32_t>::max();

// Pairs buffered per worker before they are sorted into a run: 16 bytes each
// while sorting.
constexpr size_t kMaxRunPostings = size_t{1} << 21;

}  // namespace

DocumentPrecomputer::DocumentPrecomputer(const SegmentWriterOptions& options,
                                         TrigramExtractor& extractor,
                                         std::atomic<uint32_t>& next_chunk)
    : options_(options),
      extractor_(extractor),
      next_chunk_(next_chunk),
      postings_(std::min(options.max_partition_postings, kMaxRunPostings)) {}

absl::StatusOr<PrecomputedDocument> DocumentPrecomputer::Precompute(
    absl::string_view content, Language language, uint32_t provisional_doc) {
  if (content.size() >= kMaxId) {
    return absl::InvalidArgumentError("content too large");
  }
  std::vector<ChunkSpan> spans;
  PrecomputedDocument result;
  {
    ScopedTimer timer(&timings_.chunking_ns);
    spans = SplitIntoChunks(content, options_.chunk_bytes);
    result.num_lines = segment_internal::NumLines(content);
  }
  const uint32_t first =
      next_chunk_.fetch_add(static_cast<uint32_t>(spans.size()));
  if (uint64_t{first} + spans.size() > kMaxId) {
    return absl::ResourceExhaustedError("segment is full");
  }
  result.first_chunk = first;
  result.chunks.reserve(spans.size());
  LexState state = LexState::kCode;
  uint32_t chunk_id = first;
  for (const ChunkSpan& span : spans) {
    const absl::string_view text = content.substr(
        static_cast<size_t>(span.offset), static_cast<size_t>(span.size));
    result.chunks.push_back(
        {.offset_in_doc = static_cast<uint32_t>(span.offset),
         .size = static_cast<uint32_t>(span.size),
         .first_line = static_cast<uint32_t>(span.first_line),
         .entry_state = state});
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
      postings_.AddChunk(chunk_id, chunk_trigrams_);
    }
    ++chunk_id;
    ScopedTimer timer(&timings_.code_tokens_ns);
    state = options_.write_code_tokens
                ? segment_internal::AddCodeTokens(language, text, state,
                                                  provisional_doc,
                                                  token_scratch_, tokens_)
                : AdvanceLexState(language, text, state);
  }
  return result;
}

std::vector<segment_internal::PostingAccumulator::Run>
DocumentPrecomputer::TakeRuns() {
  ScopedTimer timer(&timings_.postings_add_ns);
  return postings_.TakeRuns();
}

segment_internal::CodeTokenMap DocumentPrecomputer::TakeCodeTokens() {
  return std::move(tokens_);
}

}  // namespace ferret

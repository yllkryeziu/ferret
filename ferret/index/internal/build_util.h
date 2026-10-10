// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Small pieces shared by the segment writer and the bulk build's partition
// builder, so that both write the same records.

#ifndef FERRET_INDEX_INTERNAL_BUILD_UTIL_H_
#define FERRET_INDEX_INTERNAL_BUILD_UTIL_H_

#include <cstddef>
#include <cstdint>

#include "absl/strings/string_view.h"
#include "absl/time/clock.h"

#include "ferret/index/internal/segment_format.h"
#include "ferret/index/rank_key.h"
#include "ferret/text/chunker.h"
#include "ferret/text/language.h"
#include "ferret/text/lexer.h"
#include "ferret/text/line_counter.h"

namespace ferret::segment_internal {

// Adds the nanoseconds between construction and destruction to `*sink`.
class ScopedTimer {
 public:
  explicit ScopedTimer(int64_t* sink)
      : sink_(sink), start_(absl::GetCurrentTimeNanos()) {}
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;
  ~ScopedTimer() { *sink_ += absl::GetCurrentTimeNanos() - start_; }

 private:
  int64_t* const sink_;
  const int64_t start_;
};

inline uint32_t NumLines(absl::string_view content) {
  const auto newlines = static_cast<uint32_t>(CountNewlines(content));
  return !content.empty() && content.back() != '\n' ? newlines + 1 : newlines;
}

inline segment_format::DocRecord MakeDocRecord(
    uint32_t first_chunk, uint32_t num_chunks, uint32_t num_lines,
    uint32_t path_offset, const RankKey& rank_key, Language language,
    bool long_lines) {
  return {
      .first_chunk = first_chunk,
      .num_chunks = num_chunks,
      .num_lines = num_lines,
      .path_offset = path_offset,
      .prior = rank_key.prior,
      .tier = static_cast<uint8_t>(rank_key.tier),
      .language = static_cast<uint8_t>(language),
      .flags = long_lines ? segment_format::kDocLongLines : uint8_t{0},
      .pad = {},
  };
}

inline segment_format::ChunkRecord MakeChunkRecord(uint64_t content_offset,
                                                   const ChunkSpan& chunk,
                                                   uint32_t doc,
                                                   LexState state) {
  return {
      .content_offset = content_offset + static_cast<uint64_t>(chunk.offset),
      .content_size = static_cast<uint32_t>(chunk.size),
      .first_line = static_cast<uint32_t>(chunk.first_line),
      .doc = doc,
      .flags = static_cast<uint8_t>(static_cast<uint8_t>(state) &
                                    segment_format::kChunkLexStateMask),
      .pad = {},
  };
}

}  // namespace ferret::segment_internal

#endif  // FERRET_INDEX_INTERNAL_BUILD_UTIL_H_

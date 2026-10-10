// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/chunker.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"

#include "ferret/text/line_counter.h"

namespace ferret {

std::vector<ChunkSpan> SplitIntoChunks(absl::string_view content,
                                       int64_t chunk_bytes) {
  CHECK_GT(chunk_bytes, 0);
  const size_t min_bytes = static_cast<size_t>(chunk_bytes);
  std::vector<ChunkSpan> chunks;
  int64_t line = 1;
  size_t begin = 0;
  while (begin < content.size()) {
    size_t end = content.size();
    if (content.size() - begin > min_bytes) {
      const size_t newline = content.find('\n', begin + min_bytes);
      if (newline != absl::string_view::npos) end = newline + 1;
    }
    const absl::string_view chunk = content.substr(begin, end - begin);
    chunks.push_back({.offset = static_cast<int64_t>(begin),
                      .size = static_cast<int64_t>(chunk.size()),
                      .first_line = line});
    line += CountNewlines(chunk);
    begin = end;
  }
  return chunks;
}

}  // namespace ferret

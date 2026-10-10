// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Splits a file's content into chunks, the posting unit of the index.
//
// Verification bytes dominate query latency, and the 2% of files larger than
// 64 KiB hold half of Chromium's bytes and nearly every common trigram, so
// postings name chunks rather than whole files. A chunk that starts at offset
// s ends just after the first '\n' at or after offset s + chunk_bytes, or at
// the end of the content if there is no such '\n' or s + chunk_bytes is not
// inside the content. Consequences:
//  * Every chunk but the last is the shortest run of whole lines longer than
//    chunk_bytes, and content of at most chunk_bytes bytes is one chunk.
//  * A line longer than chunk_bytes yields one oversize chunk.
//  * A line never spans chunks. Since no indexed trigram contains '\n', every
//    indexed trigram lies inside one chunk, which makes chunk candidates
//    exactly as sound as file candidates.
//
// A chunk-size sweep measured this rule (8 KiB chunks: +51% postings,
// 1.5-9.7x fewer candidate bytes on Chromium).

#ifndef FERRET_TEXT_CHUNKER_H_
#define FERRET_TEXT_CHUNKER_H_

#include <cstdint>
#include <vector>

#include "absl/strings/string_view.h"

namespace ferret {

// One chunk of a file.
struct ChunkSpan {
  // Byte offset of the chunk in the content.
  int64_t offset = 0;
  // Byte length, at least 1.
  int64_t size = 0;
  // The 1-based number of the chunk's first line, as editors and grep show
  // it. A match's line is first_line plus the number of '\n' bytes between
  // the chunk start and the match.
  int64_t first_line = 1;

  friend bool operator==(const ChunkSpan&, const ChunkSpan&) = default;
};

// Returns the chunks of `content` in order. They tile it exactly, so empty
// content has no chunks. Requires chunk_bytes > 0. O(|content|). Thread-safe.
std::vector<ChunkSpan> SplitIntoChunks(absl::string_view content,
                                       int64_t chunk_bytes);

}  // namespace ferret

#endif  // FERRET_TEXT_CHUNKER_H_

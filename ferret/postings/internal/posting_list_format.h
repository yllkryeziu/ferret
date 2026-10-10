// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Layout details of an encoded posting list shared by the encoder and
// PostingListView. The format itself is documented in posting_list.h.

#ifndef FERRET_POSTINGS_INTERNAL_POSTING_LIST_FORMAT_H_
#define FERRET_POSTINGS_INTERNAL_POSTING_LIST_FORMAT_H_

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "ferret/postings/bitpack.h"
#include "ferret/postings/ids.h"
#include "ferret/postings/posting_list.h"

namespace ferret::postings_internal {

static_assert(std::endian::native == std::endian::little,
              "posting lists are stored little-endian");
static_assert(kPostingBlockSize == kBitPackBlockSize);

// The id "before" a list's first id. Gaps are taken from it modulo 2^32, so
// the first stored gap - 1 is the first id itself.
inline constexpr ChunkId kBeforeFirstId = std::numeric_limits<ChunkId>::max();

// One skip table entry, stored as two little-endian uint32s.
struct SkipEntry {
  uint32_t last_chunk;
  uint32_t offset16_and_width;  // offset16 << 8 | bit width.
};
static_assert(sizeof(SkipEntry) == 8);

inline constexpr int kSkipWidthBits = 8;
inline constexpr uint32_t kMaxOffset16 = (uint32_t{1} << 24) - 1;
// Block b starts at most 32 * b units into the payload region, so every list
// of up to kMaxPostingListSize ids has representable offsets.
static_assert(kMaxBitPackWidth *
                  (kMaxPostingListSize / kPostingBlockSize - 1) <=
              kMaxOffset16);

constexpr uint32_t PackOffset16AndWidth(uint32_t offset16, int bit_width) {
  return offset16 << kSkipWidthBits | static_cast<uint32_t>(bit_width);
}
constexpr uint32_t Offset16(const SkipEntry& entry) {
  return entry.offset16_and_width >> kSkipWidthBits;
}
constexpr int BitWidth(const SkipEntry& entry) {
  return static_cast<int>(entry.offset16_and_width &
                          ((uint32_t{1} << kSkipWidthBits) - 1));
}

// Returns the byte offset of the payload region of a blocked list whose count
// varint is `count_bytes` long and which has `num_full_blocks` skip entries.
// The skip table ends there; the zero padding before it is what aligns the
// payload region (and, as entries are 8 bytes, the skip table).
constexpr int64_t PayloadRegionOffset(int64_t count_bytes,
                                      int64_t num_full_blocks) {
  const int64_t skip_end =
      count_bytes + num_full_blocks * static_cast<int64_t>(sizeof(SkipEntry));
  return (skip_end + kPostingPayloadAlignment - 1) / kPostingPayloadAlignment *
         kPostingPayloadAlignment;
}

inline SkipEntry LoadSkipEntry(const uint8_t* skip_table, int block) {
  SkipEntry entry;
  std::memcpy(&entry,
              skip_table + sizeof(SkipEntry) * static_cast<size_t>(block),
              sizeof(entry));
  return entry;
}

}  // namespace ferret::postings_internal

#endif  // FERRET_POSTINGS_INTERNAL_POSTING_LIST_FORMAT_H_

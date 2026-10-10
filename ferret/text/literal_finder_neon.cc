// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// For the 16 candidate offsets o, o+1, ..., o+15, one load at o + rare1_offset
// and one at o + rare2_offset line up each candidate's two filter bytes in the
// same lane, so two compares and an AND test all 16 at once. A block reads up
// to o + 15 + (needle length - 1), so it stays inside the haystack as long as
// o + 15 is a valid candidate. The last partial block is handled by re-running
// the filter on the final full block with the already rejected lanes masked
// off, so nothing is read past the end.

#include <arm_neon.h>

#include <bit>
#include <cstddef>
#include <cstdint>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"

#include "ferret/text/internal/literal_finder_kernels.h"

namespace ferret::text_internal {
namespace {

// Narrows a 16-lane comparison result to a 64-bit mask with bit 4*i + 3 set
// for each matching lane i. NEON has no movemask; shifting each 16-bit pair
// right by 4 and narrowing keeps one nibble per lane.
uint64_t LaneMask(uint8x16_t matches) {
  const uint8x8_t nibbles = vshrn_n_u16(vreinterpretq_u16_u8(matches), 4);
  return vget_lane_u64(vreinterpret_u64_u8(nibbles), 0) & 0x8888888888888888u;
}

template <bool kFoldAscii>
size_t Find(const LiteralNeedle& needle, absl::string_view haystack,
            size_t from) {
  const size_t length = needle.bytes.size();
  if (from > haystack.size() || haystack.size() - from < length) {
    return absl::string_view::npos;
  }
  // The last offset at which the needle fits.
  const size_t last = haystack.size() - length;
  const char* const hay = haystack.data();
  const uint8_t* const first =
      reinterpret_cast<const uint8_t*>(hay) + needle.rare1_offset;
  const uint8_t* const second =
      reinterpret_cast<const uint8_t*>(hay) + needle.rare2_offset;
  const uint8_t byte1 = static_cast<uint8_t>(needle.bytes[needle.rare1_offset]);
  const uint8_t byte2 = static_cast<uint8_t>(needle.bytes[needle.rare2_offset]);
  const uint8x16_t want1 = vdupq_n_u8(byte1);
  const uint8x16_t want2 = vdupq_n_u8(byte2);
  // The needle is folded, so a letter in it is lowercase, and setting bit 0x20
  // maps exactly A-Z onto a-z. Non-letters are compared unchanged.
  auto case_bit = [](uint8_t b) -> uint8_t {
    return kFoldAscii && b >= 'a' && b <= 'z' ? 0x20 : 0;
  };
  const uint8x16_t fold1 = vdupq_n_u8(case_bit(byte1));
  const uint8x16_t fold2 = vdupq_n_u8(case_bit(byte2));

  // Lanes are 0xFF where the candidate at offset + lane passes the filter.
  auto filter = [&](size_t offset) {
    uint8x16_t a = vld1q_u8(first + offset);
    uint8x16_t b = vld1q_u8(second + offset);
    if constexpr (kFoldAscii) {
      a = vorrq_u8(a, fold1);
      b = vorrq_u8(b, fold2);
    }
    return vandq_u8(vceqq_u8(a, want1), vceqq_u8(b, want2));
  };
  // Verifies the candidates of `mask`, a LaneMask for the block at `offset`,
  // in increasing order and returns the first match.
  auto verify = [&](uint64_t mask, size_t offset) -> size_t {
    for (; mask != 0; mask &= mask - 1) {
      const size_t candidate =
          offset + static_cast<size_t>(std::countr_zero(mask)) / 4;
      if (LiteralMatchesAt<kFoldAscii>(needle.bytes, hay + candidate)) {
        return candidate;
      }
    }
    return absl::string_view::npos;
  };

  // Every candidate before `offset` has been rejected.
  size_t offset = from;
  for (; offset + 63 <= last; offset += 64) {
    const uint8x16_t pass0 = filter(offset);
    const uint8x16_t pass1 = filter(offset + 16);
    const uint8x16_t pass2 = filter(offset + 32);
    const uint8x16_t pass3 = filter(offset + 48);
    if (vmaxvq_u8(vorrq_u8(vorrq_u8(pass0, pass1), vorrq_u8(pass2, pass3))) ==
        0) {
      continue;
    }
    size_t match = verify(LaneMask(pass0), offset);
    if (match == absl::string_view::npos) {
      match = verify(LaneMask(pass1), offset + 16);
    }
    if (match == absl::string_view::npos) {
      match = verify(LaneMask(pass2), offset + 32);
    }
    if (match == absl::string_view::npos) {
      match = verify(LaneMask(pass3), offset + 48);
    }
    if (match != absl::string_view::npos) return match;
  }
  for (; offset + 15 <= last; offset += 16) {
    const size_t match = verify(LaneMask(filter(offset)), offset);
    if (match != absl::string_view::npos) return match;
  }
  if (offset > last) return absl::string_view::npos;
  if (last >= 15) {
    // Fewer than 16 candidates remain. The final full block overlaps the
    // previous one; drop its lanes below `offset`, which were rejected.
    const size_t block = last - 15;
    const size_t rejected = offset - block;
    DCHECK(rejected >= 1 && rejected <= 15);
    return verify(LaneMask(filter(block)) & (~uint64_t{0} << (4 * rejected)),
                  block);
  }
  // The haystack is too short for a single block.
  for (; offset <= last; ++offset) {
    if (LiteralMatchesAt<kFoldAscii>(needle.bytes, hay + offset)) {
      return offset;
    }
  }
  return absl::string_view::npos;
}

}  // namespace

size_t FindLiteralNeon(const LiteralNeedle& needle, absl::string_view haystack,
                       size_t from) {
  DCHECK(!needle.bytes.empty());
  DCHECK_LT(needle.rare1_offset, needle.bytes.size());
  DCHECK_LT(needle.rare2_offset, needle.bytes.size());
  return needle.fold_ascii ? Find<true>(needle, haystack, from)
                           : Find<false>(needle, haystack, from);
}

}  // namespace ferret::text_internal

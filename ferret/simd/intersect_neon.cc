// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/simd/intersect.h"

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

#include "absl/base/nullability.h"
#include "absl/types/span.h"

#include "ferret/simd/internal/compress_store.h"

namespace ferret {

size_t IntersectNeon(absl::Span<const uint32_t> a, absl::Span<const uint32_t> b,
                     uint32_t* absl_nonnull out) {
  // The block loop covers whole 4-element blocks of both inputs.
  const size_t a_blocks_end = a.size() & ~size_t{3};
  const size_t b_blocks_end = b.size() & ~size_t{3};
  size_t i = 0;
  size_t j = 0;
  size_t count = 0;
  while (i < a_blocks_end && j < b_blocks_end) {
    const uint32x4_t block_a = vld1q_u32(a.data() + i);
    const uint32x4_t block_b = vld1q_u32(b.data() + j);
    // Every lane of block_a meets every lane of block_b in one of the four
    // rotations of block_b.
    uint32x4_t match = vceqq_u32(block_a, block_b);
    match =
        vorrq_u32(match, vceqq_u32(block_a, vextq_u32(block_b, block_b, 1)));
    match =
        vorrq_u32(match, vceqq_u32(block_a, vextq_u32(block_b, block_b, 2)));
    match =
        vorrq_u32(match, vceqq_u32(block_a, vextq_u32(block_b, block_b, 3)));
    // Each element of `a` matches at most one element of `b`, and matches
    // against later blocks of `b` are larger, so the output stays strictly
    // increasing.
    count += simd_internal::CompressStore(block_a, match, out + count);
    // The block with the smaller maximum cannot match anything past the
    // other input's current block, so it is finished; on a tie both are. The
    // advance has no branches because which block finishes first is
    // unpredictable when the inputs interleave.
    const uint32_t a_max = a[i + 3];
    const uint32_t b_max = b[j + 3];
    i += a_max <= b_max ? 4 : 0;
    j += b_max <= a_max ? 4 : 0;
  }
  // Fewer than 4 elements remain in one input, and the scalar merge finishes
  // both remainders. It cannot repeat an element already written, since those
  // are smaller than everything left in the input that advanced last. It
  // stays within the caller's buffer: it stores only below
  // min(|a| - i, |b| - j) (see IntersectScalar), and count <= min(i, j) + 3,
  // because every match so far lies in a[0, i) and b[0, j) except for at most
  // 3 in the current block of the input that did not advance, whose maximum
  // is unmatched.
  return count + IntersectScalar(a.subspan(i), b.subspan(j), out + count);
}

}  // namespace ferret

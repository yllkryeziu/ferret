// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/simd/union.h"

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/types/span.h"

#include "ferret/simd/internal/compress_store.h"

namespace ferret {
namespace {

// Returns the 8 elements of `a` and `b` in order: val[0] holds the 4 smallest
// and val[1] the 4 largest. Requires `a` and `b` each sorted. `a` followed by
// `b` reversed is a bitonic sequence, which Batcher's bitonic merge (1968)
// sorts in three compare-exchange stages.
uint32x4x2_t MergeBlocks(uint32x4_t a, uint32x4_t b) {
  // Stage 1: lane k of `a` against lane 3 - k of `b`. Every element of `lo`
  // is <= every element of `hi`, and each of them is bitonic.
  const uint32x4_t b_reversed = vrev64q_u32(vextq_u32(b, b, 2));
  const uint32x4_t lo = vminq_u32(a, b_reversed);
  const uint32x4_t hi = vmaxq_u32(a, b_reversed);
  // Stage 2: lanes 2 apart, within `lo` and within `hi`.
  const uint32x4_t lo01_hi01 = vreinterpretq_u32_u64(
      vzip1q_u64(vreinterpretq_u64_u32(lo), vreinterpretq_u64_u32(hi)));
  const uint32x4_t lo23_hi23 = vreinterpretq_u32_u64(
      vzip2q_u64(vreinterpretq_u64_u32(lo), vreinterpretq_u64_u32(hi)));
  const uint32x4_t min2 = vminq_u32(lo01_hi01, lo23_hi23);
  const uint32x4_t max2 = vmaxq_u32(lo01_hi01, lo23_hi23);
  // Stage 3: adjacent lanes, within `lo` (now min2[0], min2[1], max2[0],
  // max2[1]) and within `hi` (the other four).
  const uint32x4_t even = vtrn1q_u32(min2, max2);
  const uint32x4_t odd = vtrn2q_u32(min2, max2);
  const uint32x4_t min3 = vminq_u32(even, odd);
  const uint32x4_t max3 = vmaxq_u32(even, odd);
  return {{vzip1q_u32(min3, max3), vzip2q_u32(min3, max3)}};
}

// Stores the lanes of the sorted `block` that differ from their predecessor,
// where lane 3 of `previous` precedes lane 0, and returns how many. Applied
// to consecutive blocks of a sorted stream, this drops exactly the repeats.
// Always writes out[0, 4).
size_t StoreDistinct(uint32x4_t previous, uint32x4_t block,
                     uint32_t* absl_nonnull out) {
  const uint32x4_t predecessors = vextq_u32(previous, block, 3);
  const uint32x4_t distinct = vmvnq_u32(vceqq_u32(block, predecessors));
  return simd_internal::CompressStore(block, distinct, out);
}

}  // namespace

size_t UnionNeon(absl::Span<const uint32_t> a, absl::Span<const uint32_t> b,
                 uint32_t* absl_nonnull out) {
  if (a.size() < 4 || b.size() < 4) return UnionScalar(a, b, out);
  // The merge loop consumes whole 4-element blocks of both inputs.
  const size_t a_blocks_end = a.size() & ~size_t{3};
  const size_t b_blocks_end = b.size() & ~size_t{3};
  uint32x4x2_t merged = MergeBlocks(vld1q_u32(a.data()), vld1q_u32(b.data()));
  // Lane 0 of the first block has no predecessor; x - 1 never equals x.
  size_t count = StoreDistinct(
      vdupq_n_u32(vgetq_lane_u32(merged.val[0], 0) - 1), merged.val[0], out);
  size_t i = 4;
  size_t j = 4;
  // Invariant: everything stored is smaller than every element not yet
  // loaded, and merged.val[1] holds the loaded elements not yet stored.
  // Loading the next block from the input whose next element is smaller
  // keeps it: the new block is smaller than the rest of its own input, and
  // the pending block is smaller than the rest of the other input, so the 4
  // smallest of the 8 are smaller than both.
  while (i < a_blocks_end && j < b_blocks_end) {
    const uint32x4_t stored = merged.val[0];
    const bool take_a = a[i] <= b[j];
    const uint32_t* next = take_a ? a.data() + i : b.data() + j;
    i += take_a ? 4 : 0;
    j += take_a ? 0 : 4;
    merged = MergeBlocks(vld1q_u32(next), merged.val[1]);
    count += StoreDistinct(stored, merged.val[0], out + count);
  }
  // What is left: the pending block, which may repeat the last stored element
  // or itself, and two remainders, one of them shorter than 4. Merge the
  // deduplicated pending block with the short remainder, then that with the
  // long one.
  uint32_t pending[4];
  const size_t num_pending =
      StoreDistinct(merged.val[0], merged.val[1], pending);
  absl::Span<const uint32_t> short_rest = a.subspan(i);
  absl::Span<const uint32_t> long_rest = b.subspan(j);
  if (short_rest.size() > long_rest.size()) std::swap(short_rest, long_rest);
  uint32_t tail[UnionCapacity(4, 3)];
  const size_t tail_size =
      UnionScalar(absl::MakeConstSpan(pending, num_pending), short_rest, tail);
  // Every stored element consumed an input element that is neither pending
  // nor remaining, so count + tail_size + long_rest.size() <= |a| + |b| and
  // the final merge has the capacity its contract requires.
  return count + UnionScalar(absl::MakeConstSpan(tail, tail_size), long_rest,
                             out + count);
}

}  // namespace ferret

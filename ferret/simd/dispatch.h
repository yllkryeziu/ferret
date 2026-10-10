// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The constants that tie the id-list kernels (intersect.h, union.h) to their
// callers: the output slack every kernel may write into, and the crossover
// at which Intersect() stops merging and starts galloping.
//
// The crossover comes from the ratio sweep in intersect_benchmark.cc. Re-run
// the sweep with --config=opt on an idle machine before changing it.

#ifndef FERRET_SIMD_DISPATCH_H_
#define FERRET_SIMD_DISPATCH_H_

#include <algorithm>
#include <cstddef>

namespace ferret {

// Every kernel's output buffer holds this many elements beyond the largest
// possible result. The NEON kernels store whole 4-lane vectors, so a kernel
// that returns n may also write out[n, n + kSimdSlack).
inline constexpr size_t kSimdSlack = 4;

// The length ratio from which Intersect() gallops instead of merging.
inline constexpr size_t kIntersectGallopRatio = 16;

// Returns true if Intersect() gallops for inputs of these sizes: when the
// longer input is at least kIntersectGallopRatio times the length of the
// shorter, or the shorter is empty. A merge costs O(|a| + |b|) and galloping
// O(|shorter| * log(ratio)), so galloping wins once the ratio is large
// enough.
constexpr bool IntersectShouldGallop(size_t size_a, size_t size_b) {
  // Equivalent to max >= ratio * min for integers, without the overflow.
  return std::max(size_a, size_b) / kIntersectGallopRatio >=
         std::min(size_a, size_b);
}

}  // namespace ferret

#endif  // FERRET_SIMD_DISPATCH_H_

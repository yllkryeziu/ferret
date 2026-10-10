// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Intersection kernels over sorted id lists.
//
// Every kernel has the same contract. `a` and `b` are strictly increasing.
// `out` points to at least IntersectCapacity(a.size(), b.size()) writable
// elements and overlaps neither input. The kernel writes the intersection of
// `a` and `b`, strictly increasing, to out[0, n) and returns n. It may
// overwrite the rest of the buffer. Kernels never allocate. They touch only
// their arguments, so concurrent calls on distinct output buffers are safe.
//
// IntersectScalar is the reference implementation: every other kernel returns
// exactly what it returns. Callers use Intersect(); the individual kernels are
// public so that the tests can check each one and the benchmark can measure
// the crossover that Intersect() dispatches on.

#ifndef FERRET_SIMD_INTERSECT_H_
#define FERRET_SIMD_INTERSECT_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "absl/base/nullability.h"
#include "absl/types/span.h"

#include "ferret/simd/dispatch.h"

namespace ferret {

// Returns the output capacity, in elements, that the kernels below require
// for inputs of sizes `size_a` and `size_b`.
constexpr size_t IntersectCapacity(size_t size_a, size_t size_b) {
  return std::min(size_a, size_b) + kSimdSlack;
}

// Returns the intersection using the kernel that suits the input sizes:
// galloping when IntersectShouldGallop(), and otherwise a merge (NEON where
// available).
[[nodiscard]] size_t Intersect(absl::Span<const uint32_t> a,
                               absl::Span<const uint32_t> b,
                               uint32_t* absl_nonnull out);

// A merge whose loop has no data-dependent branches: every step stores the
// current element of `a` unconditionally and advances the output only on a
// match. O(|a| + |b|).
[[nodiscard]] size_t IntersectScalar(absl::Span<const uint32_t> a,
                                     absl::Span<const uint32_t> b,
                                     uint32_t* absl_nonnull out);

// For each element of the shorter input, an exponential search of the longer
// input forward from where the previous search ended, followed by a binary
// search (Bentley and Yao 1976). O(m log(n / m)) for m = min(|a|, |b|) and
// n = max(|a|, |b|).
[[nodiscard]] size_t IntersectGalloping(absl::Span<const uint32_t> a,
                                        absl::Span<const uint32_t> b,
                                        uint32_t* absl_nonnull out);

#if defined(__aarch64__)
// The 4x4 block intersection of Katsov (2012), as described by Lemire,
// Boytsov and Kurz (SPE 2016): each block of 4 elements of `a` is compared
// against the 4 rotations of a block of `b`, and the matching lanes are
// compacted into `out` with a table-driven byte shuffle. The block with the
// smaller maximum advances. O(|a| + |b|). AArch64 only, where Advanced SIMD
// is always present.
[[nodiscard]] size_t IntersectNeon(absl::Span<const uint32_t> a,
                                   absl::Span<const uint32_t> b,
                                   uint32_t* absl_nonnull out);
#endif  // defined(__aarch64__)

}  // namespace ferret

#endif  // FERRET_SIMD_INTERSECT_H_

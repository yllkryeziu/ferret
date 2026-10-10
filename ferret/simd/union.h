// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Union kernels over sorted id lists, used for two-way ORs in query evaluation.
//
// Every kernel has the same contract. `a` and `b` are strictly increasing.
// `out` points to at least UnionCapacity(a.size(), b.size()) writable
// elements and overlaps neither input. The kernel writes the union of `a` and
// `b`, strictly increasing and without duplicates, to out[0, n) and returns
// n. It may overwrite the rest of the buffer. Kernels never allocate. They
// touch only their arguments, so concurrent calls on distinct output buffers
// are safe.
//
// UnionScalar is the reference implementation: every other kernel returns
// exactly what it returns. Callers use Union().

#ifndef FERRET_SIMD_UNION_H_
#define FERRET_SIMD_UNION_H_

#include <cstddef>
#include <cstdint>

#include "absl/base/nullability.h"
#include "absl/types/span.h"

#include "ferret/simd/dispatch.h"

namespace ferret {

// Returns the output capacity, in elements, that the kernels below require
// for inputs of sizes `size_a` and `size_b`.
constexpr size_t UnionCapacity(size_t size_a, size_t size_b) {
  return size_a + size_b + kSimdSlack;
}

// Returns the union using UnionNeon on AArch64 and UnionScalar elsewhere.
[[nodiscard]] size_t Union(absl::Span<const uint32_t> a,
                           absl::Span<const uint32_t> b,
                           uint32_t* absl_nonnull out);

// A merge whose loop has no data-dependent branches: every step stores the
// smaller current element and advances past it in each input that holds it.
// O(|a| + |b|).
[[nodiscard]] size_t UnionScalar(absl::Span<const uint32_t> a,
                                 absl::Span<const uint32_t> b,
                                 uint32_t* absl_nonnull out);

#if defined(__aarch64__)
// A vectorized merge (Inoue and Taura, VLDB 2015): a bitonic network merges
// the next block of 4 elements with the 4 largest pending ones, the lower
// half of the result is stored with duplicates compacted away, and the input
// whose next element is smaller supplies the next block. O(|a| + |b|).
// AArch64 only, where Advanced SIMD is always present.
[[nodiscard]] size_t UnionNeon(absl::Span<const uint32_t> a,
                               absl::Span<const uint32_t> b,
                               uint32_t* absl_nonnull out);
#endif  // defined(__aarch64__)

}  // namespace ferret

#endif  // FERRET_SIMD_UNION_H_

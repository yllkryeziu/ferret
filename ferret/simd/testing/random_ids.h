// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Deterministic random id lists for differential kernel tests and
// benchmarks.
//
// Everything here draws from std::mt19937_64, whose output sequence the C++
// standard fixes, through hand-written bounded draws, because the standard
// distributions differ across library implementations. A seed therefore
// reproduces the same lists on every platform.

#ifndef FERRET_SIMD_TESTING_RANDOM_IDS_H_
#define FERRET_SIMD_TESTING_RANDOM_IDS_H_

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "absl/types/span.h"

namespace ferret_testing {

// Returns an integer drawn uniformly from [lo, hi]. Requires lo <= hi.
uint64_t UniformInt(std::mt19937_64& rng, uint64_t lo, uint64_t hi);

// Returns `size` strictly increasing ids, the first equal to `first` and each
// gap drawn uniformly from [1, max_gap]; max_gap = 1 yields consecutive ids.
// Requires max_gap >= 1 and first + (size - 1) * max_gap <= UINT32_MAX.
std::vector<uint32_t> RandomIncreasingIds(std::mt19937_64& rng, size_t size,
                                          uint32_t first, uint32_t max_gap);

// Returns a uniformly random `size`-element subset of `ids`, in the order of
// `ids`, using Knuth's selection sampling (TAOCP vol. 2, Algorithm 3.4.2S).
// Requires size <= ids.size().
std::vector<uint32_t> RandomSubset(std::mt19937_64& rng,
                                   absl::Span<const uint32_t> ids, size_t size);

// Two strictly increasing id lists.
struct IdListPair {
  std::vector<uint32_t> a;
  std::vector<uint32_t> b;
};

// Returns a pair of lists that exercises what the kernels branch on: lengths
// from 0 to 10,000, length ratios from 1 to 1024 with either list the longer,
// overlap ranging from none to one list containing the other, runs of
// consecutive ids, and ids at both ends of the uint32_t range.
IdListPair RandomIdListPair(std::mt19937_64& rng);

// Returns pairs of independent lists over the same id range, for benchmarks.
// In each pair, `a` has shorter_size * ratio ids with gaps of mean
// `mean_gap`, and `b` has shorter_size ids with gaps of mean
// mean_gap * ratio, so each id of `b` is in `a` with probability about
// 1 / mean_gap. There is at least one pair, and together the pairs hold at
// least `min_total_ids` ids. A benchmark cycles through them so that the
// branch predictor cannot learn its input (see intersect_benchmark.cc).
// Requires shorter_size >= 1, ratio >= 1, mean_gap >= 1, and every possible
// id to fit in uint32_t.
std::vector<IdListPair> RandomListPairsOverSameRange(std::mt19937_64& rng,
                                                     size_t shorter_size,
                                                     size_t ratio,
                                                     size_t mean_gap,
                                                     size_t min_total_ids);

}  // namespace ferret_testing

#endif  // FERRET_SIMD_TESTING_RANDOM_IDS_H_

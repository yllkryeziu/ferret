// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/simd/testing/random_ids.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"

namespace ferret_testing {
namespace {

constexpr uint64_t kMaxUint64 = std::numeric_limits<uint64_t>::max();
constexpr uint32_t kMaxId = std::numeric_limits<uint32_t>::max();

// Gaps drawn uniformly from [1, 2m - 1] have mean m.
uint32_t MaxGapForMean(size_t mean_gap) {
  CHECK_GE(mean_gap, 1u);
  CHECK_LE(2 * mean_gap - 1, kMaxId);
  return static_cast<uint32_t>(2 * mean_gap - 1);
}

}  // namespace

uint64_t UniformInt(std::mt19937_64& rng, uint64_t lo, uint64_t hi) {
  CHECK_LE(lo, hi);
  const uint64_t span = hi - lo;
  if (span == kMaxUint64) return rng();
  const uint64_t range = span + 1;
  // Rejecting draws at or above the largest multiple of `range` makes every
  // residue equally likely.
  const uint64_t limit = kMaxUint64 - kMaxUint64 % range;
  uint64_t draw = 0;
  do {
    draw = rng();
  } while (draw >= limit);
  return lo + draw % range;
}

std::vector<uint32_t> RandomIncreasingIds(std::mt19937_64& rng, size_t size,
                                          uint32_t first, uint32_t max_gap) {
  CHECK_GE(max_gap, 1u);
  if (size > 0) CHECK_LE(first + uint64_t{size - 1} * max_gap, kMaxId);
  std::vector<uint32_t> ids;
  ids.reserve(size);
  uint32_t id = first;
  for (size_t k = 0; k < size; ++k) {
    if (k > 0) id += static_cast<uint32_t>(UniformInt(rng, 1, max_gap));
    ids.push_back(id);
  }
  return ids;
}

std::vector<uint32_t> RandomSubset(std::mt19937_64& rng,
                                   absl::Span<const uint32_t> ids,
                                   size_t size) {
  CHECK_LE(size, ids.size());
  std::vector<uint32_t> subset;
  subset.reserve(size);
  for (size_t k = 0; subset.size() < size; ++k) {
    // Select ids[k] with probability (ids still needed) / (ids left).
    if (UniformInt(rng, 1, ids.size() - k) <= size - subset.size()) {
      subset.push_back(ids[k]);
    }
  }
  return subset;
}

IdListPair RandomIdListPair(std::mt19937_64& rng) {
  // Drawing the order of magnitude first makes short lists, where the NEON
  // kernels run mostly their scalar tails, as common as long ones.
  const size_t longer_size = UniformInt(
      rng, 0,
      std::min<uint64_t>(10'000, uint64_t{1} << UniformInt(rng, 0, 14)));
  const size_t shorter_size =
      longer_size / UniformInt(rng, 1, uint64_t{1} << UniformInt(rng, 0, 10));

  // Both lists are subsets of one pool. A pool the size of the longer list
  // makes it contain the shorter one; a pool 16 times larger leaves few ids in
  // common.
  constexpr uint64_t kPoolFactors[] = {1, 2, 16};
  const size_t pool_size = UniformInt(
      rng, longer_size, longer_size * kPoolFactors[UniformInt(rng, 0, 2)]);
  // A maximum gap of 1 makes the pool one run of consecutive ids.
  constexpr uint64_t kMaxGaps[] = {1, 2, 3, 1000, 65536};
  const uint32_t max_gap = static_cast<uint32_t>(
      std::min<uint64_t>(kMaxGaps[UniformInt(rng, 0, 4)],
                         kMaxId / std::max<size_t>(pool_size, 1)));
  std::vector<uint32_t> pool = RandomIncreasingIds(rng, pool_size, 0, max_gap);
  if (!pool.empty()) {
    const uint32_t max_offset = kMaxId - pool.back();
    uint32_t offset = 0;  // The pool starts at id 0.
    switch (UniformInt(rng, 0, 2)) {
      case 0:
        break;
      case 1:
        offset = max_offset;  // The pool ends at UINT32_MAX.
        break;
      default:
        offset = static_cast<uint32_t>(UniformInt(rng, 0, max_offset));
        break;
    }
    for (uint32_t& id : pool) id += offset;
  }

  IdListPair pair = {.a = RandomSubset(rng, pool, longer_size),
                     .b = RandomSubset(rng, pool, shorter_size)};
  if (UniformInt(rng, 0, 1) == 1) std::swap(pair.a, pair.b);
  return pair;
}

std::vector<IdListPair> RandomListPairsOverSameRange(std::mt19937_64& rng,
                                                     size_t shorter_size,
                                                     size_t ratio,
                                                     size_t mean_gap,
                                                     size_t min_total_ids) {
  CHECK_GE(shorter_size, 1u);
  const uint32_t longer_max_gap = MaxGapForMean(mean_gap);
  const uint32_t shorter_max_gap = MaxGapForMean(mean_gap * ratio);
  std::vector<IdListPair> pairs;
  size_t total_ids = 0;
  do {
    pairs.push_back(
        {.a = RandomIncreasingIds(rng, shorter_size * ratio, 0, longer_max_gap),
         .b = RandomIncreasingIds(rng, shorter_size, 0, shorter_max_gap)});
    total_ids += shorter_size * ratio + shorter_size;
  } while (total_ids < min_total_ids);
  return pairs;
}

}  // namespace ferret_testing

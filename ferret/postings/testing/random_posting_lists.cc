// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/postings/testing/random_posting_lists.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "absl/log/check.h"

#include "ferret/postings/ids.h"

namespace ferret_testing {

uint64_t UniformBelow(std::mt19937_64& rng, uint64_t bound) {
  DCHECK_GT(bound, 0u);
  return rng() % bound;
}

std::vector<ferret::ChunkId> RandomPostingList(std::mt19937_64& rng,
                                               size_t size, int gap_bits,
                                               ferret::ChunkId first) {
  CHECK(0 <= gap_bits && gap_bits <= 32) << gap_bits;
  std::vector<ferret::ChunkId> ids;
  if (size == 0) return ids;
  constexpr uint64_t kMaxId = std::numeric_limits<ferret::ChunkId>::max();
  CHECK_LE(first + (size - 1), kMaxId);
  // The id space left over once every id takes the smallest step.
  uint64_t slack = kMaxId - first - (size - 1);
  ids.reserve(size);
  ids.push_back(first);
  for (size_t i = 1; i < size; ++i) {
    const uint64_t extra =
        std::min(UniformBelow(rng, uint64_t{1} << gap_bits), slack);
    slack -= extra;
    ids.push_back(static_cast<ferret::ChunkId>(ids.back() + 1 + extra));
  }
  return ids;
}

}  // namespace ferret_testing

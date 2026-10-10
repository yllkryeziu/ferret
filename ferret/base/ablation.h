// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Process-wide switches that turn off one kernel-level lever of the search
// ablation: SIMD decode, NEON intersect, the skip-table probe. Only the
// ablation benchmark (`ferret_bench search --scalar_decode` and friends) and
// the tests that check the switches change them. Every switch leaves results
// identical; each only chooses a slower implementation of the same function.
// They are read with relaxed loads on the query path, so flip them only while
// no search runs.

#ifndef FERRET_BASE_ABLATION_H_
#define FERRET_BASE_ABLATION_H_

#include <atomic>

namespace ferret {

struct Ablation {
  // Unpack posting-list blocks with the scalar decoder instead of NEON.
  static inline std::atomic<bool> scalar_decode{false};
  // Merge sorted id lists with the scalar kernel instead of NEON.
  static inline std::atomic<bool> scalar_intersect{false};
  // Never probe a posting list's skip table candidate by candidate: decode
  // the part of the list between the first and last candidate and intersect.
  static inline std::atomic<bool> no_skip_probe{false};
};

// How often this thread took the slower path of a switch, so a test or a
// benchmark can tell that a switch took effect. Thread-local, so counting
// costs no cache-line traffic between search threads.
struct AblationCounts {
  // Blocks unpacked by the scalar decoder because of scalar_decode.
  int64_t scalar_blocks = 0;
  // Merges done by the scalar kernel because of scalar_intersect.
  int64_t scalar_merges = 0;
  // Intersections that would have probed the skip table.
  int64_t unprobed_intersections = 0;
};

inline AblationCounts& ThisThreadAblationCounts() {
  static thread_local AblationCounts counts;
  return counts;
}

inline bool AblationOn(const std::atomic<bool>& lever) {
  return lever.load(std::memory_order_relaxed);
}

}  // namespace ferret

#endif  // FERRET_BASE_ABLATION_H_

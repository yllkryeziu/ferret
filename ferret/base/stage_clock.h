// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Per-thread nanosecond accumulators for the posting-list stages of a query:
// decoding a list, and intersecting or merging id lists (the benchmark's
// stacked latency). A search that wants the breakdown enables the clock of
// its calling thread (SearcherOptions::stage_timing); otherwise each hook
// costs one thread-local load and a branch. The accumulators count the
// thread's own wall time in the stage; nested stages are not double counted
// because decode and intersect timers never nest.

#ifndef FERRET_BASE_STAGE_CLOCK_H_
#define FERRET_BASE_STAGE_CLOCK_H_

#include <chrono>
#include <cstdint>

namespace ferret {

struct StageClock {
  bool enabled = false;
  int64_t decode_ns = 0;
  int64_t intersect_ns = 0;
};

inline StageClock& ThisThreadStageClock() {
  static thread_local StageClock clock;
  return clock;
}

// Adds the scope's wall time to `field` of this thread's clock, if enabled.
class StageTimer final {
 public:
  explicit StageTimer(int64_t StageClock::* field)
      : clock_(ThisThreadStageClock()), field_(field) {
    if (clock_.enabled) start_ = std::chrono::steady_clock::now();
  }
  StageTimer(const StageTimer&) = delete;
  StageTimer& operator=(const StageTimer&) = delete;
  ~StageTimer() {
    if (!clock_.enabled) return;
    clock_.*field_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - start_)
                          .count();
  }

 private:
  StageClock& clock_;
  int64_t StageClock::* field_;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace ferret

#endif  // FERRET_BASE_STAGE_CLOCK_H_

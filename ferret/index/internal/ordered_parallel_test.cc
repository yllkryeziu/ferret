// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/internal/ordered_parallel.h"

#include <atomic>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "ferret/base/thread_pool.h"

namespace ferret::segment_internal {
namespace {

TEST(ParallelOrderedTest, CommitsInOrderWithinTheWindow) {
  for (const int threads : {1, 2, 4, 8}) {
    for (const int64_t window : {1, 2, 5, 100}) {
      ThreadPool pool(threads, ThreadQos::kUtility);
      std::atomic<int64_t> committed{0};
      std::atomic<int64_t> max_ahead{0};
      std::vector<int64_t> seen;
      const absl::Status status = ParallelOrdered<int64_t>(
          &pool, threads, 200, window,
          [&](int worker, int64_t i) -> absl::StatusOr<int64_t> {
            EXPECT_GE(worker, 0);
            EXPECT_LT(worker, threads);
            // A step starts only when fewer than `window` steps are
            // uncommitted before it.
            const int64_t ahead = i - committed.load();
            int64_t seen_max = max_ahead.load();
            while (ahead > seen_max &&
                   !max_ahead.compare_exchange_weak(seen_max, ahead)) {
            }
            return i * 2;
          },
          [&](int64_t i, int64_t&& value) {
            EXPECT_EQ(value, i * 2);
            seen.push_back(i);
            committed = i + 1;
            return absl::OkStatus();
          });
      EXPECT_TRUE(status.ok());
      ASSERT_EQ(seen.size(), 200u);
      for (int64_t i = 0; i < 200; ++i) EXPECT_EQ(seen[i], i);
      EXPECT_LT(max_ahead.load(), window);
    }
  }
}

TEST(ParallelOrderedTest, RunsInlineWithoutAPool) {
  std::vector<int64_t> seen;
  const absl::Status status = ParallelOrdered<int64_t>(
      nullptr, 8, 5, 3,
      [](int worker, int64_t i) -> absl::StatusOr<int64_t> {
        EXPECT_EQ(worker, 0);
        return i;
      },
      [&](int64_t i, int64_t&& value) {
        EXPECT_EQ(i, value);
        seen.push_back(i);
        return absl::OkStatus();
      });
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(seen, (std::vector<int64_t>{0, 1, 2, 3, 4}));
}

TEST(ParallelOrderedTest, StopsAtTheFirstError) {
  ThreadPool pool(3, ThreadQos::kUtility);
  std::atomic<int64_t> commits{0};
  const absl::Status produce_error = ParallelOrdered<int64_t>(
      &pool, 4, 100, 3,
      [](int, int64_t i) -> absl::StatusOr<int64_t> {
        if (i == 10) return absl::DataLossError("produce");
        return i;
      },
      [&](int64_t, int64_t&&) {
        ++commits;
        return absl::OkStatus();
      });
  EXPECT_EQ(produce_error.code(), absl::StatusCode::kDataLoss);
  EXPECT_LE(commits.load(), 10);

  const absl::Status commit_error = ParallelOrdered<int64_t>(
      &pool, 4, 100, 3,
      [](int, int64_t i) -> absl::StatusOr<int64_t> { return i; },
      [](int64_t i, int64_t&&) {
        return i == 7 ? absl::AbortedError("commit") : absl::OkStatus();
      });
  EXPECT_EQ(commit_error.code(), absl::StatusCode::kAborted);
}

}  // namespace
}  // namespace ferret::segment_internal

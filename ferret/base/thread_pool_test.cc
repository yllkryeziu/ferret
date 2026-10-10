// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/base/thread_pool.h"

#if defined(__APPLE__)
#include <sys/qos.h>
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/notification.h"
#include "gtest/gtest.h"

namespace ferret {
namespace {

TEST(ThreadPoolTest, RunsEveryScheduledTask) {
  std::atomic<int> runs = 0;
  {
    ThreadPool pool(4, ThreadQos::kUtility);
    EXPECT_EQ(pool.num_threads(), 4);
    for (int i = 0; i < 1000; ++i) {
      pool.Schedule([&runs] { runs.fetch_add(1, std::memory_order_relaxed); });
    }
  }
  EXPECT_EQ(runs.load(), 1000);
}

TEST(ThreadPoolTest, DestructorDrainsTheQueue) {
  std::atomic<int> runs = 0;
  absl::Notification release;
  {
    ThreadPool pool(1, ThreadQos::kUtility);
    pool.Schedule([&release] { release.WaitForNotification(); });
    for (int i = 0; i < 100; ++i) {
      pool.Schedule([&runs] { runs.fetch_add(1, std::memory_order_relaxed); });
    }
    // The destructor starts while the queue is (almost certainly) full.
    release.Notify();
  }
  EXPECT_EQ(runs.load(), 100);
}

// Counts one run and schedules the next of `remaining` links.
void RunChainLink(ThreadPool& pool, std::atomic<int>& runs, int remaining) {
  runs.fetch_add(1, std::memory_order_relaxed);
  if (remaining > 1) {
    pool.Schedule(
        [&pool, &runs, remaining] { RunChainLink(pool, runs, remaining - 1); });
  }
}

TEST(ThreadPoolTest, TasksScheduledDuringShutdownStillRun) {
  constexpr int kChainLength = 50;
  std::atomic<int> runs = 0;
  {
    ThreadPool pool(2, ThreadQos::kUtility);
    // Each link schedules the next, racing with the destructor.
    pool.Schedule([&pool, &runs] { RunChainLink(pool, runs, kChainLength); });
  }
  EXPECT_EQ(runs.load(), kChainLength);
}

#if defined(__APPLE__)
class ThreadPoolQosTest
    : public ::testing::TestWithParam<std::pair<ThreadQos, qos_class_t>> {};

TEST_P(ThreadPoolQosTest, TasksRunAtThePoolQos) {
  const auto [qos, expected] = GetParam();
  const qos_class_t caller_qos = qos_class_self();
  qos_class_t observed = QOS_CLASS_UNSPECIFIED;
  {
    ThreadPool pool(1, qos);
    pool.Schedule([&observed] { observed = qos_class_self(); });
  }
  EXPECT_EQ(observed, expected);
  EXPECT_EQ(qos_class_self(), caller_qos);
}

INSTANTIATE_TEST_SUITE_P(
    AllClasses, ThreadPoolQosTest,
    ::testing::Values(std::pair(ThreadQos::kUserInteractive,
                                QOS_CLASS_USER_INTERACTIVE),
                      std::pair(ThreadQos::kUtility, QOS_CLASS_UTILITY),
                      std::pair(ThreadQos::kBackground, QOS_CLASS_BACKGROUND)));
#endif

TEST(ThreadPoolTest, ParallelForCoversEachIndexExactlyOnce) {
  ThreadPool pool(4, ThreadQos::kUserInteractive);
  for (const int64_t n : {0, 1, 2, 7, 64, 1000}) {
    for (const int64_t grain : {1, 3, 64, 5000}) {
      for (const int max_parallelism : {1, 2, 5, 16}) {
        SCOPED_TRACE(::testing::Message()
                     << "n " << n << " grain " << grain << " parallelism "
                     << max_parallelism);
        std::vector<std::atomic<int>> hits(static_cast<size_t>(n));
        std::atomic<bool> bad_call = false;
        pool.ParallelFor(n, grain, max_parallelism,
                         [&](int worker, int64_t begin, int64_t end) {
                           if (worker < 0 || worker >= max_parallelism ||
                               begin < 0 || begin >= end || end > n ||
                               end - begin > grain) {
                             bad_call.store(true);
                             return;
                           }
                           for (int64_t i = begin; i < end; ++i) {
                             hits[static_cast<size_t>(i)].fetch_add(1);
                           }
                         });
        EXPECT_FALSE(bad_call.load());
        for (int64_t i = 0; i < n; ++i) {
          ASSERT_EQ(hits[static_cast<size_t>(i)].load(), 1) << "index " << i;
        }
      }
    }
  }
}

TEST(ThreadPoolTest, ParallelForWorkerStateNeedsNoLocking) {
  ThreadPool pool(4, ThreadQos::kUserInteractive);
  constexpr int kParallelism = 3;
  // Plain integers: if two calls ever shared a worker index concurrently,
  // TSan would report the race.
  std::vector<int64_t> covered(kParallelism, 0);
  pool.ParallelFor(100000, 7, kParallelism,
                   [&covered](int worker, int64_t begin, int64_t end) {
                     covered[static_cast<size_t>(worker)] += end - begin;
                   });
  int64_t total = 0;
  for (const int64_t count : covered) total += count;
  EXPECT_EQ(total, 100000);
}

TEST(ThreadPoolTest, ParallelForRunsRangesConcurrently) {
  absl::Notification second_range_started;
  ThreadPool pool(1, ThreadQos::kUserInteractive);
  // Whoever takes range 0 waits for range 1, which only a second worker can
  // start, so this returns only if the caller and a pool thread both work.
  pool.ParallelFor(2, 1, 2,
                   [&second_range_started](int, int64_t begin, int64_t) {
                     if (begin == 0) {
                       second_range_started.WaitForNotification();
                     } else {
                       second_range_started.Notify();
                     }
                   });
  EXPECT_TRUE(second_range_started.HasBeenNotified());
}

TEST(ThreadPoolTest, ParallelForFinishesAloneWhenThePoolIsBusy) {
  // Declared before the pool, whose destructor joins the thread that waits.
  absl::Notification release;
  std::atomic<int64_t> covered = 0;
  std::atomic<bool> helped = false;
  ThreadPool pool(1, ThreadQos::kUserInteractive);
  pool.Schedule([&release] { release.WaitForNotification(); });
  pool.ParallelFor(100, 1, 4, [&](int worker, int64_t begin, int64_t end) {
    if (worker != 0) helped.store(true);
    covered.fetch_add(end - begin);
  });
  EXPECT_EQ(covered.load(), 100);
  EXPECT_FALSE(helped.load());
  release.Notify();
}

TEST(ThreadPoolTest, ParallelForFromPoolTasksDoesNotDeadlock) {
  constexpr int kThreads = 2;
  std::atomic<int64_t> covered = 0;
  absl::BlockingCounter done(kThreads);
  ThreadPool pool(kThreads, ThreadQos::kUserInteractive);
  // Every pool thread becomes a ParallelFor caller whose helpers queue
  // behind the other caller.
  for (int i = 0; i < kThreads; ++i) {
    pool.Schedule([&] {
      pool.ParallelFor(1000, 10, kThreads + 1,
                       [&covered](int, int64_t begin, int64_t end) {
                         covered.fetch_add(end - begin);
                       });
      done.DecrementCount();
    });
  }
  done.Wait();
  EXPECT_EQ(covered.load(), kThreads * 1000);
}

TEST(ThreadPoolTest, ParallelForWithoutParallelismRunsOnTheCaller) {
  ThreadPool pool(4, ThreadQos::kUserInteractive);
  const std::thread::id caller = std::this_thread::get_id();
  bool all_on_caller = true;
  int calls = 0;
  pool.ParallelFor(10, 3, 1, [&](int worker, int64_t, int64_t) {
    all_on_caller =
        all_on_caller && worker == 0 && std::this_thread::get_id() == caller;
    ++calls;
  });
  EXPECT_TRUE(all_on_caller);
  EXPECT_EQ(calls, 4);
}

}  // namespace
}  // namespace ferret

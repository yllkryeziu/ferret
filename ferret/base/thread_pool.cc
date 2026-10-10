// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/base/thread_pool.h"

#if defined(__APPLE__)
#include <pthread/qos.h>
#include <sys/qos.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"

namespace ferret {

void ApplyThreadQos(ThreadQos qos) {
#if defined(__APPLE__)
  qos_class_t qos_class = QOS_CLASS_UNSPECIFIED;
  switch (qos) {
    case ThreadQos::kUserInteractive:
      qos_class = QOS_CLASS_USER_INTERACTIVE;
      break;
    case ThreadQos::kUserInitiated:
      qos_class = QOS_CLASS_USER_INITIATED;
      break;
    case ThreadQos::kUtility:
      qos_class = QOS_CLASS_UTILITY;
      break;
    case ThreadQos::kBackground:
      qos_class = QOS_CLASS_BACKGROUND;
      break;
  }
  const int result = pthread_set_qos_class_self_np(qos_class, 0);
  DCHECK_EQ(result, 0) << "pthread_set_qos_class_self_np";
#else
  static_cast<void>(qos);
#endif
}

namespace {

// What one ParallelFor call shares with the pool tasks it schedules. A task
// can start after the call has returned, and then only checks `finished_`,
// so the job is reference-counted by the call and its tasks instead of
// living in the caller's frame.
class ParallelForJob {
 public:
  ParallelForJob(int64_t n, int64_t grain,
                 absl::FunctionRef<void(int, int64_t, int64_t)> fn)
      : n_(n), grain_(grain), fn_(fn) {}
  ParallelForJob(const ParallelForJob&) = delete;
  ParallelForJob& operator=(const ParallelForJob&) = delete;

  // Runs ranges on a pool thread, unless the caller has already finished.
  void Help() ABSL_LOCKS_EXCLUDED(mu_) {
    int worker = 0;
    {
      absl::MutexLock lock(mu_);
      if (finished_) return;
      worker = ++helpers_joined_;
      ++helpers_running_;
    }
    RunRanges(worker);
    absl::MutexLock lock(mu_);
    --helpers_running_;
  }

  // Runs ranges on the calling thread as worker 0, then waits for the
  // helpers that joined.
  void RunAndWait() ABSL_LOCKS_EXCLUDED(mu_) {
    RunRanges(0);
    absl::MutexLock lock(mu_);
    finished_ = true;
    mu_.Await(absl::Condition(this, &ParallelForJob::HelpersDoneLocked));
  }

 private:
  void RunRanges(int worker) {
    // Relaxed ordering suffices: the cursor only has to hand out each range
    // once. What `fn_` writes reaches the caller through mu_, which each
    // helper takes after its last range.
    for (int64_t begin = next_.fetch_add(grain_, std::memory_order_relaxed);
         begin < n_;
         begin = next_.fetch_add(grain_, std::memory_order_relaxed)) {
      fn_(worker, begin, begin + std::min(grain_, n_ - begin));
    }
  }

  bool HelpersDoneLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return helpers_running_ == 0;
  }

  const int64_t n_;
  const int64_t grain_;
  const absl::FunctionRef<void(int, int64_t, int64_t)> fn_;
  std::atomic<int64_t> next_{0};
  absl::Mutex mu_;
  int helpers_joined_ ABSL_GUARDED_BY(mu_) = 0;
  int helpers_running_ ABSL_GUARDED_BY(mu_) = 0;
  bool finished_ ABSL_GUARDED_BY(mu_) = false;
};

}  // namespace

ThreadPool::ThreadPool(int num_threads, ThreadQos qos) : qos_(qos) {
  CHECK_GE(num_threads, 1);
  threads_.reserve(static_cast<size_t>(num_threads));
  for (int i = 0; i < num_threads; ++i) {
    threads_.emplace_back([this] { WorkLoop(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    absl::MutexLock lock(mu_);
    stopping_ = true;
  }
  for (std::thread& thread : threads_) thread.join();
}

void ThreadPool::Schedule(absl::AnyInvocable<void()> task) {
  DCHECK(task != nullptr);
  absl::MutexLock lock(mu_);
  queue_.push_back(std::move(task));
}

void ThreadPool::ParallelFor(
    int64_t n, int64_t grain, int max_parallelism,
    absl::FunctionRef<void(int, int64_t, int64_t)> fn) {
  // CHECK, not DCHECK: this runs once per call, and a zero grain would spin
  // forever handing out empty ranges.
  CHECK_GE(grain, 1);
  CHECK_GE(max_parallelism, 1);
  if (n <= 0) return;
  const int64_t num_ranges = (n - 1) / grain + 1;
  const int64_t num_helpers =
      std::min<int64_t>({max_parallelism - 1, num_threads(), num_ranges - 1});
  auto job = std::make_shared<ParallelForJob>(n, grain, fn);
  for (int64_t i = 0; i < num_helpers; ++i) {
    Schedule([job] { job->Help(); });
  }
  job->RunAndWait();
}

void ThreadPool::WorkLoop() {
  ApplyThreadQos(qos_);
  while (true) {
    absl::AnyInvocable<void()> task;
    {
      absl::MutexLock lock(mu_);
      mu_.Await(absl::Condition(this, &ThreadPool::WorkAvailableLocked));
      // Stopping with nothing left. A task still running elsewhere may
      // schedule more, but its own thread comes back for it.
      if (queue_.empty()) return;
      task = std::move(queue_.front());
      queue_.pop_front();
    }
    task();
  }
}

}  // namespace ferret

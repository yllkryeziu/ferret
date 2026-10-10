// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// ThreadPool: a fixed set of threads that share one scheduling class.
//
// Ferret runs two kinds of pools: query verification at user-interactive QoS
// on as many threads as there are performance cores, and flushes and
// compactions at utility QoS, so that background merges yield to queries on
// the performance cores. ParallelFor is the fan-out that query verification
// needs: workers pull fixed-size ranges from a shared cursor, and the caller
// works too, so a query never waits for a pool thread to become free before
// making progress.

#ifndef FERRET_BASE_THREAD_POOL_H_
#define FERRET_BASE_THREAD_POOL_H_

#include <cstdint>
#include <deque>
#include <thread>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/synchronization/mutex.h"

namespace ferret {

// The scheduling class of a pool's threads. On macOS it sets the thread's QoS
// class, which decides between performance and efficiency cores and how the
// thread competes under load; elsewhere it has no effect.
enum class ThreadQos : uint8_t {
  // Work a user is waiting on right now: query verification.
  kUserInteractive,
  // Work a user started and will soon wait for: the readers of a first
  // registration.
  kUserInitiated,
  // Long-running work whose progress matters: flushes, compactions, and
  // madvise prefetch of new segments.
  kUtility,
  // Work nobody waits for.
  kBackground,
};

// Sets the calling thread's scheduling class. Elsewhere this is a no-op.
void ApplyThreadQos(ThreadQos qos);

// Thread-safe.
class ThreadPool final {
 public:
  // Starts `num_threads` (>= 1) threads that run at `qos`.
  ThreadPool(int num_threads, ThreadQos qos);
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  // Runs every task scheduled so far, including tasks that tasks schedule
  // meanwhile, then joins the threads.
  ~ThreadPool();

  int num_threads() const { return static_cast<int>(threads_.size()); }

  // Runs `task` on a pool thread, in FIFO order with other tasks.
  void Schedule(absl::AnyInvocable<void()> task) ABSL_LOCKS_EXCLUDED(mu_);

  // Calls fn(worker, begin, end) on disjoint ranges that cover [0, n)
  // exactly, each at most `grain` (>= 1) long, and returns when all calls
  // have returned. Up to `max_parallelism` (>= 1) workers run the calls: the
  // calling thread, which is worker 0 and runs at its own QoS, and up to
  // max_parallelism - 1 pool threads, which get workers 1, 2, ... in the
  // order they join. A worker makes its calls one at a time, so state indexed
  // by `worker` (such as a per-worker RE2 copy) needs no locking.
  //
  // The caller waits only for pool threads that have already joined, never
  // for queued ones (which find nothing left to do when they finally run), so
  // a busy pool slows ParallelFor down but cannot deadlock it, even when it
  // is called from a pool task.
  void ParallelFor(int64_t n, int64_t grain, int max_parallelism,
                   absl::FunctionRef<void(int, int64_t, int64_t)> fn)
      ABSL_LOCKS_EXCLUDED(mu_);

 private:
  void WorkLoop() ABSL_LOCKS_EXCLUDED(mu_);
  bool WorkAvailableLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return !queue_.empty() || stopping_;
  }

  const ThreadQos qos_;
  absl::Mutex mu_;
  std::deque<absl::AnyInvocable<void()>> queue_ ABSL_GUARDED_BY(mu_);
  bool stopping_ ABSL_GUARDED_BY(mu_) = false;
  // Written only by the constructor, before any other access.
  std::vector<std::thread> threads_;
};

}  // namespace ferret

#endif  // FERRET_BASE_THREAD_POOL_H_

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Runs independent steps in parallel and applies their results in order.
//
// The bulk build and the parallel finish use it: steps 0..n-1 are claimed in
// increasing order by up to `threads` workers, each step's value is built
// concurrently, and `commit` receives the values in index order, one call at
// a time, on whichever worker finished the step that unblocked it. At most
// `window` steps are in flight or waiting to be committed, which bounds the
// memory of the values.

#ifndef FERRET_INDEX_INTERNAL_ORDERED_PARALLEL_H_
#define FERRET_INDEX_INTERNAL_ORDERED_PARALLEL_H_

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"

#include "ferret/base/thread_pool.h"

namespace ferret::segment_internal {

// Calls `produce(worker, i)` for i in [0, n) on up to `threads` workers
// (worker is in [0, threads), and a worker makes its calls one at a time),
// then `commit(i, value)` in increasing i. Runs inline when `pool` is null or
// `threads` is 1. Returns the first error from either function, after which
// no more steps start; steps already running finish. Requires window >= 1.
template <typename T>
absl::Status ParallelOrdered(
    ThreadPool* pool, int threads, int64_t n, int64_t window,
    absl::FunctionRef<absl::StatusOr<T>(int worker, int64_t i)> produce,
    absl::FunctionRef<absl::Status(int64_t i, T&& value)> commit) {
  if (pool == nullptr || threads <= 1 || n <= 1) {
    for (int64_t i = 0; i < n; ++i) {
      absl::StatusOr<T> value = produce(0, i);
      if (!value.ok()) return value.status();
      absl::Status status = commit(i, std::move(*value));
      if (!status.ok()) return status;
    }
    return absl::OkStatus();
  }
  struct State {
    absl::Mutex mu;
    std::vector<std::optional<T>> slots;
    int64_t next_commit = 0;
    int64_t window = 1;
    bool committing = false;
    absl::Status error;
  } state;
  state.window = std::max<int64_t>(window, 1);
  state.slots.resize(static_cast<size_t>(n));
  pool->ParallelFor(
      n, /*grain=*/1, threads, [&](int worker, int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          const std::pair<State*, int64_t> wait = {&state, i};
          state.mu.lock();
          state.mu.Await(absl::Condition(
              +[](const std::pair<State*, int64_t>* w) {
                const State& s = *w->first;
                return w->second < s.next_commit + s.window || !s.error.ok();
              },
              &wait));
          const bool failed = !state.error.ok();
          state.mu.unlock();
          if (failed) return;
          absl::StatusOr<T> value = produce(worker, i);
          absl::MutexLock lock(state.mu);
          if (!value.ok()) {
            state.error.Update(value.status());
            continue;
          }
          state.slots[static_cast<size_t>(i)] = std::move(*value);
          if (state.committing) continue;
          state.committing = true;
          while (
              state.next_commit < n && state.error.ok() &&
              state.slots[static_cast<size_t>(state.next_commit)].has_value()) {
            const int64_t k = state.next_commit;
            T item = std::move(*state.slots[static_cast<size_t>(k)]);
            state.slots[static_cast<size_t>(k)].reset();
            state.mu.unlock();
            const absl::Status status = commit(k, std::move(item));
            state.mu.lock();
            state.error.Update(status);
            ++state.next_commit;
          }
          state.committing = false;
        }
      });
  return state.error;
}

}  // namespace ferret::segment_internal

#endif  // FERRET_INDEX_INTERNAL_ORDERED_PARALLEL_H_

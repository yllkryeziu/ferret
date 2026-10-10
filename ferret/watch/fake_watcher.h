// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Test watcher with explicitly queued hints and a worker thread that delivers
// them on request. It uses the same batching rules as the live watchers.

#ifndef FERRET_WATCH_FAKE_WATCHER_H_
#define FERRET_WATCH_FAKE_WATCHER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <thread>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

#include "ferret/watch/internal/event_batcher.h"
#include "ferret/watch/watcher.h"

namespace ferret {

// Test double: Push queues hints; Deliver or Sync flushes on its own thread.
// Implements the Watcher lifecycle; the destructor calls Stop.
class FakeWatcher final : public Watcher {
 public:
  // Opens a watcher for a canonical directory, without starting capture.
  // Returns an error if the root or options are invalid.
  static absl::StatusOr<std::unique_ptr<FakeWatcher>> Open(
      const std::string& root, const WatcherOptions& options = {});
  ~FakeWatcher() override;
  // Starts capture; see Watcher::Start for lifecycle and error semantics.
  absl::Status Start(WatchCallback callback) override;
  // Stops capture and joins all callback work; see Watcher::Stop.
  void Stop() override;
  // Delivers changes through a barrier; see Watcher::Sync.
  absl::Status Sync(absl::Duration timeout) override;
  // Returns the canonical root for the lifetime of this watcher.
  const std::string& root() const override { return root_; }

  // Queues a hint for the next explicit delivery while not stopped.
  void Push(WatchEvent event);
  // Requests delivery on the worker thread, without waiting for it.
  void Deliver();

 private:
  FakeWatcher(std::string root, const WatcherOptions& options);
  void Run();
  bool WakeLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  std::string root_;
  absl::Mutex mu_;
  watch_internal::EventBatcher batcher_ ABSL_GUARDED_BY(mu_);
  WatchCallback callback_;
  std::thread thread_;
  bool started_ ABSL_GUARDED_BY(mu_) = false;
  bool stopping_ ABSL_GUARDED_BY(mu_) = false;
  uint64_t requested_ ABSL_GUARDED_BY(mu_) = 0;
  uint64_t completed_ ABSL_GUARDED_BY(mu_) = 0;
};

}  // namespace ferret

#endif  // FERRET_WATCH_FAKE_WATCHER_H_

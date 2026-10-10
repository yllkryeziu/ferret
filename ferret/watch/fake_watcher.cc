// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/watch/fake_watcher.h"

#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"

#include "ferret/watch/internal/canonical_root.h"
#include "ferret/watch/internal/validate_options.h"
#include "ferret/watch/watcher.h"

namespace ferret {
FakeWatcher::FakeWatcher(std::string root, const WatcherOptions& options)
    : root_(std::move(root)),
      batcher_(options.quiet_window, options.max_delay,
               [] { return absl::Now(); }) {}

absl::StatusOr<std::unique_ptr<FakeWatcher>> FakeWatcher::Open(
    const std::string& root, const WatcherOptions& options) {
  absl::Status valid = watch_internal::ValidateOptions(options);
  if (!valid.ok()) return valid;
  auto canonical = watch_internal::CanonicalRoot(root);
  if (!canonical.ok()) return canonical.status();
  return std::unique_ptr<FakeWatcher>(new FakeWatcher(*canonical, options));
}

FakeWatcher::~FakeWatcher() { Stop(); }

absl::Status FakeWatcher::Start(WatchCallback callback) {
  absl::MutexLock lock(mu_);
  if (started_ || stopping_) {
    return absl::FailedPreconditionError("watcher already started or stopped");
  }
  if (callback == nullptr) {
    return absl::InvalidArgumentError("watch callback is null");
  }
  callback_ = std::move(callback);
  started_ = true;
  thread_ = std::thread([this] { Run(); });
  return absl::OkStatus();
}

void FakeWatcher::Stop() {
  {
    absl::MutexLock lock(mu_);
    stopping_ = true;
  }
  if (thread_.joinable()) {
    thread_.join();
  }
}

void FakeWatcher::Push(WatchEvent event) {
  absl::MutexLock lock(mu_);
  if (!stopping_) {
    batcher_.Add(std::move(event));
  }
}

void FakeWatcher::Deliver() {
  absl::MutexLock lock(mu_);
  if (started_ && !stopping_) {
    ++requested_;
  }
}

bool FakeWatcher::WakeLocked() const {
  return stopping_ || requested_ != completed_;
}

absl::Status FakeWatcher::Sync(absl::Duration timeout) {
  absl::MutexLock lock(mu_);
  if (!started_ || stopping_) {
    return absl::FailedPreconditionError("watcher is not running");
  }
  const uint64_t ticket = ++requested_;
  const auto done = [this, ticket]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return stopping_ || completed_ >= ticket;
  };
  if (!mu_.AwaitWithDeadline(absl::Condition(&done), absl::Now() + timeout)) {
    return absl::DeadlineExceededError("watcher sync timed out");
  }
  if (stopping_) {
    return absl::FailedPreconditionError("watcher stopped during sync");
  }
  return absl::OkStatus();
}

void FakeWatcher::Run() {
  while (true) {
    std::vector<WatchEvent> events;
    uint64_t ticket;
    {
      absl::MutexLock lock(mu_);
      mu_.Await(absl::Condition(this, &FakeWatcher::WakeLocked));
      if (stopping_) {
        return;
      }
      ticket = requested_;
      events = batcher_.Flush();
    }
    if (!events.empty()) {
      callback_(std::move(events));
    }
    {
      absl::MutexLock lock(mu_);
      completed_ = ticket;
    }
  }
}

}  // namespace ferret

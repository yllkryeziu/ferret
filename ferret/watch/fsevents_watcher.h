// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// FSEvents watcher for macOS. File-level events are batched as reconciliation
// hints; a cookie file provides the Sync barrier. Stream callbacks run on a
// serial dispatch queue, while client callbacks run on a separate thread.

#ifndef FERRET_WATCH_FSEVENTS_WATCHER_H_
#define FERRET_WATCH_FSEVENTS_WATCHER_H_

#include <CoreServices/CoreServices.h>
#include <dispatch/dispatch.h>

#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

#include "ferret/watch/internal/event_batcher.h"
#include "ferret/watch/watcher.h"

namespace ferret {

// Implements the Watcher lifecycle; the destructor calls Stop.
class FsEventsWatcher final : public Watcher {
 public:
  // Opens a watcher for a canonical directory, without starting capture.
  // Returns an error if the root, options, or resolved cookie directory are
  // invalid; the cookie directory must already exist inside the root.
  static absl::StatusOr<std::unique_ptr<FsEventsWatcher>> Open(
      const std::string& root, const WatcherOptions& options = {});
  ~FsEventsWatcher() override;
  // Starts capture; see Watcher::Start for lifecycle and error semantics.
  absl::Status Start(WatchCallback callback) override;
  // Stops capture and joins all callback work; see Watcher::Stop.
  void Stop() override;
  // Delivers changes through a barrier; see Watcher::Sync.
  absl::Status Sync(absl::Duration timeout) override;
  // Returns the canonical root for the lifetime of this watcher.
  const std::string& root() const override { return root_; }

 private:
  FsEventsWatcher(std::string root, const WatcherOptions& options,
                  std::string cookie_directory);
  static void OnStream(ConstFSEventStreamRef stream, void* context,
                       size_t count, void* paths,
                       const FSEventStreamEventFlags flags[],
                       const FSEventStreamEventId ids[]);
  void OnEvents(size_t count, char** paths,
                const FSEventStreamEventFlags flags[]);
  void Run();
  bool WakeLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Removes the cookie files that finished Syncs left behind.
  static void RemoveCookies(const std::vector<std::string>& cookies);

  const std::string root_;
  const std::string cookie_directory_;
  absl::Mutex sync_mu_;
  absl::Mutex mu_;
  watch_internal::EventBatcher batcher_ ABSL_GUARDED_BY(mu_);
  WatchCallback callback_;
  std::thread thread_;
  dispatch_queue_t queue_ = nullptr;
  FSEventStreamRef stream_ = nullptr;
  bool started_ ABSL_GUARDED_BY(mu_) = false;
  bool stopping_ ABSL_GUARDED_BY(mu_) = false;
  bool deadline_changed_ ABSL_GUARDED_BY(mu_) = false;
  uint64_t cookie_counter_ ABSL_GUARDED_BY(mu_) = 0;
  uint64_t seen_ ABSL_GUARDED_BY(mu_) = 0;
  uint64_t completed_ ABSL_GUARDED_BY(mu_) = 0;
  std::string cookie_ ABSL_GUARDED_BY(mu_);
  // Cookies whose Sync returned. The delivery thread unlinks them, so the
  // unlink is not on the caller's path; Stop unlinks any that are left.
  std::vector<std::string> finished_cookies_ ABSL_GUARDED_BY(mu_);
};

}  // namespace ferret

#endif  // FERRET_WATCH_FSEVENTS_WATCHER_H_

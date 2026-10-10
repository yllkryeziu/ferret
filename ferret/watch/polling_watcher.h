// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Portable watcher that compares filesystem snapshots without following
// symlinked directories. A background thread scans periodically or on Sync;
// the scan and client callbacks are serialized on that thread.

#ifndef FERRET_WATCH_POLLING_WATCHER_H_
#define FERRET_WATCH_POLLING_WATCHER_H_

#include <sys/stat.h>

#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

#include "ferret/watch/internal/event_batcher.h"
#include "ferret/watch/watcher.h"

namespace ferret {

// Implements the Watcher lifecycle; the destructor calls Stop.
class PollingWatcher final : public Watcher {
 public:
  // Opens a watcher for a canonical directory, without starting capture.
  // Returns an error if the root or options are invalid.
  static absl::StatusOr<std::unique_ptr<PollingWatcher>> Open(
      const std::string& root, WatcherOptions options = {});
  ~PollingWatcher() override;

  // Starts capture; see Watcher::Start for lifecycle and error semantics.
  absl::Status Start(WatchCallback callback) override;
  // Stops capture and joins all callback work; see Watcher::Stop.
  void Stop() override;
  // Delivers changes through a barrier; see Watcher::Sync.
  absl::Status Sync(absl::Duration timeout) override;
  // Returns the canonical root for the lifetime of this watcher.
  const std::string& root() const override { return root_; }

 private:
  struct Entry {
    mode_t mode;
    off_t size;
    timespec mtime;
    timespec ctime;
    ino_t inode;
    bool operator==(const Entry& other) const {
      return mode == other.mode && size == other.size && inode == other.inode &&
             mtime.tv_sec == other.mtime.tv_sec &&
             mtime.tv_nsec == other.mtime.tv_nsec &&
             ctime.tv_sec == other.ctime.tv_sec &&
             ctime.tv_nsec == other.ctime.tv_nsec;
    }
  };
  using Snapshot = absl::flat_hash_map<std::string, Entry>;

  PollingWatcher(std::string root, WatcherOptions options);
  absl::Status Scan(Snapshot* snapshot) const;
  absl::Status ScanDir(const std::string& relative, Snapshot* snapshot) const;
  absl::Status ScanAndReport();
  void Run();
  bool WakeLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const std::string root_;
  const WatcherOptions options_;
  watch_internal::EventBatcher batcher_;
  Snapshot previous_;
  WatchCallback callback_;
  absl::Mutex mu_;
  std::thread thread_;
  bool started_ ABSL_GUARDED_BY(mu_) = false;
  bool stopping_ ABSL_GUARDED_BY(mu_) = false;
  uint64_t requested_ ABSL_GUARDED_BY(mu_) = 0;
  uint64_t completed_ ABSL_GUARDED_BY(mu_) = 0;
  absl::Status scan_status_ ABSL_GUARDED_BY(mu_);
};

}  // namespace ferret

#endif  // FERRET_WATCH_POLLING_WATCHER_H_

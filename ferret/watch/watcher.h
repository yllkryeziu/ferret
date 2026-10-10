// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Watchers capture filesystem changes as hints to re-examine paths, never as
// descriptions of operations. Callbacks receive serialized batches: duplicate
// paths are coalesced, and a subtree hint absorbs hints for its descendants.
// Changes may be coalesced or reported more than once across batches. With
// FSEvents, start capture before the initial filesystem snapshot (subscribe,
// then snapshot) so changes during the scan can be reconciled afterward.
// Polling establishes its baseline before Start returns. Callbacks run on a
// watcher-owned thread; do not call Stop or Sync from a callback.

#ifndef FERRET_WATCH_WATCHER_H_
#define FERRET_WATCH_WATCHER_H_

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"

namespace ferret {

// Paths are relative to the canonical root, slash-separated and without ./.
// An event requests re-examination; it does not describe the operation.
struct PathChanged {
  std::string rel_path;
};
struct SubtreeChanged {
  // The empty prefix denotes the entire root.
  std::string rel_prefix;
};
using WatchEvent = std::variant<PathChanged, SubtreeChanged>;
using WatchCallback = absl::AnyInvocable<void(std::vector<WatchEvent>)>;

struct WatcherOptions {
  // The quiet period before FSEvents or polling delivers a batch. The fake
  // watcher accepts this setting but flushes only on Deliver or Sync.
  absl::Duration quiet_window = absl::Milliseconds(20);
  // The maximum wait for an FSEvents or polling batch, even under activity.
  // The fake watcher flushes explicitly instead of waiting for this delay.
  absl::Duration max_delay = absl::Milliseconds(50);
  // The interval between background scans in the polling watcher only.
  absl::Duration interval = absl::Seconds(1);
  // Directory names the polling watcher does not descend into.
  std::vector<std::string> pruned_dirs = {".git"};
  // An existing relative directory inside the root where FSEvents creates
  // sync cookies. Polling and fake watchers do not create cookies.
  std::string cookie_dir = ".git/ferret";
};

// A watcher is noncopyable. Start and Sync may be called from client threads;
// callbacks are serialized on a watcher-owned thread. Do not call Stop or
// Sync from a callback. Do not destroy the watcher while another client call
// is still in progress.
class Watcher {
 public:
  Watcher() = default;
  Watcher(const Watcher&) = delete;
  Watcher& operator=(const Watcher&) = delete;
  virtual ~Watcher() = default;

  // Starts capturing before returning. A caller can snapshot the filesystem
  // after this call and reconcile events captured during that snapshot.
  // Returns InvalidArgument for a null callback, or FailedPrecondition if
  // called twice or after Stop. The callback must outlive Stop.
  virtual absl::Status Start(WatchCallback callback) = 0;
  // Stops capturing and joins all watcher work. Idempotent; no callback runs
  // after this returns. Must not be called from the callback thread.
  virtual void Stop() = 0;
  // Waits until changes made before this call have reached the callback, or
  // until timeout. Returns FailedPrecondition when not running and
  // DeadlineExceeded on timeout. Polling forces a scan and returns its scan
  // error after delivering a whole-root subtree hint if the scan fails;
  // unlike FSEvents it cannot see changes entirely between scans.
  virtual absl::Status Sync(absl::Duration timeout) = 0;
  // Returns the canonical absolute root, valid for the watcher's lifetime.
  virtual const std::string& root() const = 0;
};

// Opens the native FSEvents watcher on macOS, or the polling watcher elsewhere.
// Returns a watcher that has not started, with a canonical root, or an error
// for invalid options or a root that cannot be resolved. On macOS cookie_dir
// must exist and resolve inside the root. The caller owns the returned watcher.
absl::StatusOr<std::unique_ptr<Watcher>> NewWatcher(
    const std::string& root, const WatcherOptions& options = {});

}  // namespace ferret

#endif  // FERRET_WATCH_WATCHER_H_

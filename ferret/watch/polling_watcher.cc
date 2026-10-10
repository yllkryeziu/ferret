// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/watch/polling_watcher.h"

#include <dirent.h>
#include <errno.h>
#include <sys/dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "ferret/watch/internal/canonical_root.h"
#include "ferret/watch/internal/validate_options.h"
#include "ferret/watch/watcher.h"

namespace ferret {
PollingWatcher::PollingWatcher(std::string root, WatcherOptions options)
    : root_(std::move(root)),
      options_(std::move(options)),
      batcher_(options_.quiet_window, options_.max_delay,
               [] { return absl::Now(); }) {}

absl::StatusOr<std::unique_ptr<PollingWatcher>> PollingWatcher::Open(
    const std::string& root, WatcherOptions options) {
  absl::Status valid = watch_internal::ValidateOptions(options);
  if (!valid.ok()) return valid;
  auto canonical = watch_internal::CanonicalRoot(root);
  if (!canonical.ok()) return canonical.status();
  if (options.interval <= absl::ZeroDuration()) {
    return absl::InvalidArgumentError("polling interval must be positive");
  }
  return std::unique_ptr<PollingWatcher>(
      new PollingWatcher(*canonical, std::move(options)));
}

PollingWatcher::~PollingWatcher() { Stop(); }

absl::Status PollingWatcher::ScanDir(const std::string& relative,
                                     Snapshot* snapshot) const {
  const std::string directory =
      relative.empty() ? root_ : absl::StrCat(root_, "/", relative);
  DIR* dir = opendir(directory.c_str());
  if (dir == nullptr) {
    return absl::ErrnoToStatus(errno, absl::StrCat("opendir(", directory, ")"));
  }
  absl::Status status;
  while (true) {
    errno = 0;
    dirent* child = readdir(dir);
    if (child == nullptr) {
      if (errno != 0) {
        status = absl::ErrnoToStatus(errno,
                                     absl::StrCat("readdir(", directory, ")"));
      }
      break;
    }
    const std::string name = child->d_name;
    if (name == "." || name == "..") {
      continue;
    }
    const std::string path =
        relative.empty() ? name : absl::StrCat(relative, "/", name);
    const std::string absolute = absl::StrCat(root_, "/", path);
    struct stat st;
    if (lstat(absolute.c_str(), &st) != 0) {
      if (errno == ENOENT) {
        continue;  // The entry vanished during the scan.
      }
      status =
          absl::ErrnoToStatus(errno, absl::StrCat("lstat(", absolute, ")"));
      break;
    }
#ifdef __APPLE__
    const timespec mtime = st.st_mtimespec;
    const timespec ctime = st.st_ctimespec;
#else
    const timespec mtime = st.st_mtim;
    const timespec ctime = st.st_ctim;
#endif
    (*snapshot)[path] = Entry{st.st_mode, st.st_size, mtime, ctime, st.st_ino};
    if (S_ISDIR(st.st_mode) &&
        std::find(options_.pruned_dirs.begin(), options_.pruned_dirs.end(),
                  name) == options_.pruned_dirs.end()) {
      status = ScanDir(path, snapshot);
      if (!status.ok()) {
        break;
      }
    }
  }
  closedir(dir);
  return status;
}

absl::Status PollingWatcher::Scan(Snapshot* snapshot) const {
  absl::Status status = ScanDir("", snapshot);
  if (!status.ok()) return status;
  // The rest of .git is pruned, but HEAD and index change during checkout,
  // commit, and reset. Observe their signatures without scanning objects.
  const auto dir = snapshot->find(".git");
  if (dir == snapshot->end() || !S_ISDIR(dir->second.mode)) {
    return absl::OkStatus();
  }
  for (const char* path : {".git/HEAD", ".git/index"}) {
    struct stat st;
    if (lstat(absl::StrCat(root_, "/", path).c_str(), &st) != 0) {
      if (errno == ENOENT) continue;
      return absl::ErrnoToStatus(errno, absl::StrCat("lstat(", path, ")"));
    }
#ifdef __APPLE__
    const timespec mtime = st.st_mtimespec;
    const timespec ctime = st.st_ctimespec;
#else
    const timespec mtime = st.st_mtim;
    const timespec ctime = st.st_ctim;
#endif
    (*snapshot)[path] = Entry{st.st_mode, st.st_size, mtime, ctime, st.st_ino};
  }
  return absl::OkStatus();
}

absl::Status PollingWatcher::ScanAndReport() {
  Snapshot current;
  absl::Status status = Scan(&current);
  if (!status.ok()) {
    batcher_.Add(SubtreeChanged{""});
    return status;
  }
  for (const auto& [path, entry] : current) {
    const auto it = previous_.find(path);
    if (it == previous_.end() && S_ISDIR(entry.mode)) {
      batcher_.Add(SubtreeChanged{path});
    } else if (it == previous_.end() || !(it->second == entry)) {
      batcher_.Add(PathChanged{path});
    }
  }
  for (const auto& [path, entry] : previous_) {
    if (current.contains(path)) {
      continue;
    }
    // A missing directory absorbs the changes to all its former children.
    if (S_ISDIR(entry.mode)) {
      batcher_.Add(SubtreeChanged{path});
    } else {
      batcher_.Add(PathChanged{path});
    }
  }
  previous_ = std::move(current);
  return absl::OkStatus();
}

absl::Status PollingWatcher::Start(WatchCallback callback) {
  absl::MutexLock lock(mu_);
  if (started_ || stopping_) {
    return absl::FailedPreconditionError("watcher already started or stopped");
  }
  if (callback == nullptr) {
    return absl::InvalidArgumentError("watch callback is null");
  }
  absl::Status status = Scan(&previous_);
  if (!status.ok()) return status;
  callback_ = std::move(callback);
  started_ = true;
  thread_ = std::thread([this] { Run(); });
  return absl::OkStatus();
}

void PollingWatcher::Stop() {
  {
    absl::MutexLock lock(mu_);
    stopping_ = true;
  }
  if (thread_.joinable()) {
    thread_.join();
  }
}

bool PollingWatcher::WakeLocked() const {
  return stopping_ || requested_ != completed_;
}
absl::Status PollingWatcher::Sync(absl::Duration timeout) {
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
  return scan_status_;
}

void PollingWatcher::Run() {
  absl::Time next_scan = absl::Now() + options_.interval;
  while (true) {
    bool scan = false;
    bool sync_scan = false;
    uint64_t ticket = 0;
    {
      absl::MutexLock lock(mu_);
      mu_.AwaitWithDeadline(absl::Condition(this, &PollingWatcher::WakeLocked),
                            std::min(next_scan, batcher_.deadline()));
      if (stopping_) {
        return;
      }
      ticket = requested_;
      sync_scan = ticket != completed_;
      scan = sync_scan || absl::Now() >= next_scan;
    }
    absl::Status status;
    if (scan) {
      status = ScanAndReport();
      next_scan = absl::Now() + options_.interval;
    }
    if (sync_scan || batcher_.Due()) {
      std::vector<WatchEvent> events = batcher_.Flush();
      if (!events.empty()) {
        callback_(std::move(events));
      }
    }
    if (sync_scan) {
      absl::MutexLock lock(mu_);
      scan_status_ = status;
      completed_ = ticket;
    }
  }
}

}  // namespace ferret

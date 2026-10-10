// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/watch/fsevents_watcher.h"

#include <CoreFoundation/CFArray.h>
#include <CoreFoundation/CFBase.h>
#include <CoreFoundation/CFString.h>
#include <dispatch/dispatch.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"

#include "ferret/watch/internal/canonical_root.h"
#include "ferret/watch/internal/cookie_directory.h"
#include "ferret/watch/internal/validate_options.h"
#include "ferret/watch/watch_trace.h"
#include "ferret/watch/watcher.h"

namespace ferret {
FsEventsWatcher::FsEventsWatcher(std::string root,
                                 const WatcherOptions& options,
                                 std::string cookie_directory)
    : root_(std::move(root)),
      cookie_directory_(std::move(cookie_directory)),
      batcher_(options.quiet_window, options.max_delay,
               [] { return absl::Now(); }) {}

absl::StatusOr<std::unique_ptr<FsEventsWatcher>> FsEventsWatcher::Open(
    const std::string& root, const WatcherOptions& options) {
  absl::Status valid = watch_internal::ValidateOptions(options);
  if (!valid.ok()) return valid;
  auto canonical = watch_internal::CanonicalRoot(root);
  if (!canonical.ok()) return canonical.status();
  if (options.cookie_dir.empty() || options.cookie_dir.front() == '/' ||
      absl::StrContains(options.cookie_dir, "..")) {
    return absl::InvalidArgumentError(
        "cookie_dir must be a relative directory inside the root");
  }
  auto directory = watch_internal::CanonicalRoot(
      absl::StrCat(*canonical, "/", options.cookie_dir));
  if (!directory.ok()) {
    return directory.status();
  }
  const std::string root_prefix =
      *canonical == "/" ? *canonical : absl::StrCat(*canonical, "/");
  if (directory->compare(0, root_prefix.size(), root_prefix) != 0) {
    return absl::InvalidArgumentError(
        "cookie_dir resolves outside the watch root");
  }
  return std::unique_ptr<FsEventsWatcher>(
      new FsEventsWatcher(*canonical, options, std::move(*directory)));
}

FsEventsWatcher::~FsEventsWatcher() { Stop(); }

void FsEventsWatcher::OnStream(ConstFSEventStreamRef, void* context,
                               size_t count, void* paths,
                               const FSEventStreamEventFlags flags[],
                               const FSEventStreamEventId[]) {
  static_cast<FsEventsWatcher*>(context)->OnEvents(
      count, static_cast<char**>(paths), flags);
}

void FsEventsWatcher::OnEvents(size_t count, char** paths,
                               const FSEventStreamEventFlags flags[]) {
  absl::MutexLock lock(mu_);
  if (stopping_) {
    return;
  }
  constexpr FSEventStreamEventFlags kRescan =
      kFSEventStreamEventFlagMustScanSubDirs |
      kFSEventStreamEventFlagUserDropped |
      kFSEventStreamEventFlagKernelDropped | kFSEventStreamEventFlagRootChanged;
  const std::string cookie_prefix = absl::StrCat(cookie_directory_, "/cookie-");
  const std::string root_prefix =
      root_ == "/" ? root_ : absl::StrCat(root_, "/");
  for (size_t i = 0; i < count; ++i) {
    const std::string absolute = paths[i];
    if (absolute == cookie_) {
      watch_trace::Mark(watch_trace::kCookieSeen);
      ++seen_;
      cookie_.clear();
      continue;
    }
    if (absolute.compare(0, cookie_prefix.size(), cookie_prefix) == 0) {
      continue;
    }
    if (absolute != root_ &&
        absolute.compare(0, root_prefix.size(), root_prefix) != 0) {
      continue;
    }
    watch_trace::Mark(watch_trace::kEventSeen);
    const std::string relative =
        absolute == root_ ? "" : absolute.substr(root_prefix.size());
    if ((flags[i] & kRescan) != 0 ||
        (flags[i] & kFSEventStreamEventFlagItemIsDir) != 0) {
      batcher_.Add(SubtreeChanged{relative});
    } else {
      batcher_.Add(PathChanged{relative});
    }
    deadline_changed_ = true;
  }
}

absl::Status FsEventsWatcher::Start(WatchCallback callback) {
  absl::MutexLock lock(mu_);
  if (started_ || stopping_) {
    return absl::FailedPreconditionError("watcher already started or stopped");
  }
  if (callback == nullptr) {
    return absl::InvalidArgumentError("watch callback is null");
  }
  CFStringRef path = CFStringCreateWithCString(
      kCFAllocatorDefault, root_.c_str(), kCFStringEncodingUTF8);
  if (path == nullptr) {
    return absl::InternalError("cannot encode FSEvents root");
  }
  const void* values[] = {path};
  CFArrayRef paths =
      CFArrayCreate(kCFAllocatorDefault, values, 1, &kCFTypeArrayCallBacks);
  CFRelease(path);
  if (paths == nullptr) {
    return absl::InternalError("cannot create FSEvents path list");
  }
  FSEventStreamContext context = {0, this, nullptr, nullptr, nullptr};
  stream_ = FSEventStreamCreate(
      kCFAllocatorDefault, &FsEventsWatcher::OnStream, &context, paths,
      kFSEventStreamEventIdSinceNow, 0.01,
      kFSEventStreamCreateFlagFileEvents | kFSEventStreamCreateFlagNoDefer |
          kFSEventStreamCreateFlagWatchRoot);
  CFRelease(paths);
  if (stream_ == nullptr) {
    return absl::InternalError("cannot create FSEvents stream");
  }
  queue_ =
      dispatch_queue_create("ferret.watch.fsevents", DISPATCH_QUEUE_SERIAL);
  FSEventStreamSetDispatchQueue(stream_, queue_);
  if (!FSEventStreamStart(stream_)) {
    FSEventStreamInvalidate(stream_);
    FSEventStreamRelease(stream_);
    stream_ = nullptr;
    dispatch_release(queue_);
    queue_ = nullptr;
    return absl::UnavailableError("cannot start FSEvents stream");
  }
  callback_ = std::move(callback);
  started_ = true;
  thread_ = std::thread([this] { Run(); });
  return absl::OkStatus();
}

void FsEventsWatcher::Stop() {
  {
    absl::MutexLock lock(mu_);
    if (stopping_) {
      return;
    }
    stopping_ = true;
  }
  if (stream_ != nullptr) {
    FSEventStreamStop(stream_);
    FSEventStreamInvalidate(stream_);
    FSEventStreamRelease(stream_);
    stream_ = nullptr;
  }
  if (queue_ != nullptr) {
    dispatch_sync_f(queue_, nullptr, [](void*) {});
    dispatch_release(queue_);
    queue_ = nullptr;
  }
  if (thread_.joinable()) {
    thread_.join();
  }
  // No Sync can run now: it needs `started_ && !stopping_`, and a caller must
  // not destroy or stop the watcher during a Sync.
  std::vector<std::string> leftover;
  {
    absl::MutexLock lock(mu_);
    leftover.swap(finished_cookies_);
  }
  RemoveCookies(leftover);
}

void FsEventsWatcher::RemoveCookies(const std::vector<std::string>& cookies) {
  for (const std::string& cookie : cookies) unlink(cookie.c_str());
}

bool FsEventsWatcher::WakeLocked() const {
  return stopping_ || seen_ != completed_ || deadline_changed_ ||
         !finished_cookies_.empty();
}

absl::Status FsEventsWatcher::Sync(absl::Duration timeout) {
  // Serialize cookie creation with other Sync calls, including their cleanup.
  absl::MutexLock sync_lock(sync_mu_);
  std::string cookie;
  uint64_t ticket;
  {
    absl::MutexLock lock(mu_);
    if (!started_ || stopping_) {
      return absl::FailedPreconditionError("watcher is not running");
    }
    ticket = seen_ + 1;
    cookie = absl::StrCat(cookie_directory_, "/cookie-", getpid(), "-",
                          ++cookie_counter_);
    cookie_ = cookie;
  }
  const auto open_cookie = [&cookie] {
    return open(cookie.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  };
  int fd = open_cookie();
  if (fd < 0 && errno == ENOENT) {
    // `git clean -dx` or `rm -rf` removed the directory; put it back.
    if (watch_internal::MakeCookieDirectory(cookie_directory_).ok()) {
      fd = open_cookie();
    } else {
      errno = ENOENT;
    }
  }
  if (fd < 0) {
    absl::MutexLock lock(mu_);
    cookie_.clear();
    return absl::ErrnoToStatus(errno, absl::StrCat("open(", cookie, ")"));
  }
  close(fd);
  watch_trace::Mark(watch_trace::kCookieCreated);
  absl::Status result;
  {
    absl::MutexLock lock(mu_);
    const auto done = [this, ticket]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return stopping_ || completed_ >= ticket;
    };
    if (!mu_.AwaitWithDeadline(absl::Condition(&done), absl::Now() + timeout)) {
      result = absl::DeadlineExceededError("FSEvents cookie sync timed out");
    } else if (stopping_) {
      result = absl::FailedPreconditionError("watcher stopped during sync");
    }
    cookie_.clear();
    finished_cookies_.push_back(std::move(cookie));
  }
  watch_trace::Mark(watch_trace::kSyncWoke);
  return result;
}

void FsEventsWatcher::Run() {
  while (true) {
    std::vector<WatchEvent> events;
    std::vector<std::string> cookies;
    uint64_t ticket;
    bool deliver = true;
    {
      absl::MutexLock lock(mu_);
      mu_.AwaitWithDeadline(absl::Condition(this, &FsEventsWatcher::WakeLocked),
                            batcher_.deadline());
      if (stopping_) {
        return;
      }
      cookies.swap(finished_cookies_);
      deadline_changed_ = false;
      ticket = seen_;
      if (ticket == completed_ && !batcher_.Due()) {
        deliver = false;
      } else {
        events = batcher_.Flush();
      }
    }
    // Unlinking is off the Sync caller's path and outside the lock. After a
    // delivery it also follows the ticket's completion.
    if (!deliver) {
      RemoveCookies(cookies);
      continue;
    }
    if (!events.empty()) {
      watch_trace::Mark(watch_trace::kDelivering);
      watch_trace::Get().callbacks.fetch_add(1, std::memory_order_relaxed);
      callback_(std::move(events));
    }
    {
      absl::MutexLock lock(mu_);
      if (ticket != completed_) batcher_.EndBurst();
      completed_ = ticket;
      watch_trace::Mark(watch_trace::kCookieDone);
    }
    RemoveCookies(cookies);
  }
}

}  // namespace ferret

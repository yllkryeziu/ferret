// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/watch/fsevents_watcher.h"

#include <unistd.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <thread>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/watch/watcher.h"

namespace ferret {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

WatcherOptions CookieOptions() {
  WatcherOptions options;
  options.cookie_dir = "cookies";
  return options;
}

// Holds the watcher's callback thread in its first batch until Release, so
// the watcher cannot acknowledge a Sync cookie. Release before the watcher
// stops, and destroy after it. Thread-safe.
class BlockingCallback {
 public:
  BlockingCallback() = default;
  BlockingCallback(const BlockingCallback&) = delete;
  BlockingCallback& operator=(const BlockingCallback&) = delete;

  WatchCallback Get() {
    return [this](const std::vector<WatchEvent>&) {
      absl::MutexLock lock(mu_);
      entered_ = true;
      mu_.Await(absl::Condition(&released_));
    };
  }
  bool WaitUntilEntered() {
    absl::MutexLock lock(mu_);
    return mu_.AwaitWithTimeout(absl::Condition(&entered_), absl::Seconds(15));
  }
  void Release() {
    absl::MutexLock lock(mu_);
    released_ = true;
  }

 private:
  absl::Mutex mu_;
  bool entered_ ABSL_GUARDED_BY(mu_) = false;
  bool released_ ABSL_GUARDED_BY(mu_) = false;
};

TEST(FsEventsWatcherTest, OpenRejectsInvalidRootAndDebounceOptions) {
  ferret_testing::TempDir dir;
  ASSERT_THAT(dir.MakeDirs("cookies"), IsOk());
  ASSERT_THAT(dir.WriteFile("file", "x"), IsOk());
  EXPECT_THAT(FsEventsWatcher::Open(dir.Join("missing"), CookieOptions()),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("missing")));
  EXPECT_THAT(FsEventsWatcher::Open(dir.Join("file"), CookieOptions()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("not a directory")));

  WatcherOptions options = CookieOptions();
  options.quiet_window = absl::ZeroDuration();
  EXPECT_THAT(
      FsEventsWatcher::Open(dir.path(), options),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("debounce")));
  options.quiet_window = absl::Milliseconds(20);
  options.max_delay = -absl::Milliseconds(1);
  EXPECT_THAT(
      FsEventsWatcher::Open(dir.path(), options),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("debounce")));
}

TEST(FsEventsWatcherTest, OpenRequiresExistingCookieDirectoryInsideRoot) {
  ferret_testing::TempDir dir;
  ASSERT_THAT(dir.MakeDirs("root/cookies"), IsOk());
  ASSERT_THAT(dir.MakeDirs("outside"), IsOk());
  WatcherOptions options;
  // The root itself is not inside the root: cookies there would share a
  // prefix with top-level files named cookie-*.
  for (const char* cookie_dir : {"", "/tmp", "../outside", "."}) {
    SCOPED_TRACE(cookie_dir);
    options.cookie_dir = cookie_dir;
    EXPECT_THAT(FsEventsWatcher::Open(dir.Join("root"), options),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  options.cookie_dir = "missing";
  EXPECT_THAT(FsEventsWatcher::Open(dir.Join("root"), options),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("missing")));

  options.cookie_dir = "cookies";
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<FsEventsWatcher> watcher,
                              FsEventsWatcher::Open(dir.Join("root"), options));
  EXPECT_EQ(watcher->root(), dir.Join("root"));
}

TEST(FsEventsWatcherTest, StartAndSyncFollowTheLifecycle) {
  ferret_testing::TempDir dir;
  ASSERT_THAT(dir.MakeDirs("cookies"), IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<FsEventsWatcher> watcher,
      FsEventsWatcher::Open(dir.path(), CookieOptions()));
  EXPECT_THAT(watcher->Sync(absl::Seconds(1)),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(watcher->Start(nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));

  ASSERT_THAT(watcher->Start([](const std::vector<WatchEvent>&) {}), IsOk());
  EXPECT_THAT(watcher->Start([](const std::vector<WatchEvent>&) {}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(watcher->Sync(absl::Seconds(15)), IsOk());

  watcher->Stop();
  watcher->Stop();
  EXPECT_THAT(watcher->Sync(absl::Seconds(1)),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(watcher->Start([](const std::vector<WatchEvent>&) {}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(FsEventsWatcherTest, SyncTimesOutWhileCallbackIsBusyThenRecovers) {
  ferret_testing::TempDir dir;
  ASSERT_THAT(dir.MakeDirs("cookies"), IsOk());
  BlockingCallback callback;
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<FsEventsWatcher> watcher,
      FsEventsWatcher::Open(dir.path(), CookieOptions()));
  // Stop joins the callback thread, so release it first on every exit path.
  absl::Cleanup release = [&callback] { callback.Release(); };
  ASSERT_THAT(watcher->Start(callback.Get()), IsOk());
  ASSERT_THAT(dir.WriteFile("file", "x"), IsOk());
  ASSERT_TRUE(callback.WaitUntilEntered());

  EXPECT_THAT(watcher->Sync(absl::Milliseconds(100)),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
  callback.Release();
  EXPECT_THAT(watcher->Sync(absl::Seconds(15)), IsOk());
  // Stop removes the cookies of both Syncs, so the directory is empty.
  watcher->Stop();
  EXPECT_EQ(rmdir(dir.Join("cookies").c_str()), 0);
}

// The delivery thread unlinks a finished Sync's cookie after Sync returned.
TEST(FsEventsWatcherTest, FinishedCookiesAreRemovedAfterSyncReturns) {
  ferret_testing::TempDir dir;
  ASSERT_THAT(dir.MakeDirs("cookies"), IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<FsEventsWatcher> watcher,
      FsEventsWatcher::Open(dir.path(), CookieOptions()));
  ASSERT_THAT(watcher->Start([](const std::vector<WatchEvent>&) {}), IsOk());
  for (int i = 0; i < 20; ++i) {
    ASSERT_THAT(watcher->Sync(absl::Seconds(15)), IsOk());
  }
  const auto cookies = [&dir] {
    size_t count = 0;
    for (const auto& entry :
         std::filesystem::directory_iterator(dir.Join("cookies"))) {
      (void)entry;
      ++count;
    }
    return count;
  };
  // Without a new Sync or delivery, only the unlink of the last cookie can
  // still be pending; it happens on the delivery thread.
  const absl::Time deadline = absl::Now() + absl::Seconds(15);
  while (cookies() != 0 && absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(5));
  }
  EXPECT_EQ(cookies(), size_t{0});
  watcher->Stop();
  EXPECT_EQ(rmdir(dir.Join("cookies").c_str()), 0);
}

// With the delivery thread held in a callback, nothing unlinks the cookies of
// returned Syncs. Stop must not leave them behind.
TEST(FsEventsWatcherTest, StopRemovesCookiesThatWereNeverUnlinked) {
  ferret_testing::TempDir dir;
  ASSERT_THAT(dir.MakeDirs("cookies"), IsOk());
  BlockingCallback callback;
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<FsEventsWatcher> watcher,
      FsEventsWatcher::Open(dir.path(), CookieOptions()));
  absl::Cleanup release = [&callback] { callback.Release(); };
  ASSERT_THAT(watcher->Start(callback.Get()), IsOk());
  ASSERT_THAT(dir.WriteFile("file", "x"), IsOk());
  ASSERT_TRUE(callback.WaitUntilEntered());
  EXPECT_THAT(watcher->Sync(absl::Milliseconds(50)),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_THAT(watcher->Sync(absl::Milliseconds(50)),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
  callback.Release();
  watcher->Stop();
  EXPECT_EQ(rmdir(dir.Join("cookies").c_str()), 0);
}

// `git clean -dx` or `rm -rf` in a worktree removes the cookie directory.
TEST(FsEventsWatcherTest, SyncRecreatesAMissingCookieDirectory) {
  ferret_testing::TempDir dir;
  ASSERT_THAT(dir.MakeDirs("cookies"), IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<FsEventsWatcher> watcher,
      FsEventsWatcher::Open(dir.path(), CookieOptions()));
  ASSERT_THAT(watcher->Start([](const std::vector<WatchEvent>&) {}), IsOk());
  ASSERT_EQ(rmdir(dir.Join("cookies").c_str()), 0);
  EXPECT_THAT(watcher->Sync(absl::Seconds(15)), IsOk());
  EXPECT_EQ(access(dir.Join("cookies/.gitignore").c_str(), F_OK), 0);
  watcher->Stop();
}

TEST(FsEventsWatcherTest, SyncFailsWhileCookieParentIsMissing) {
  ferret_testing::TempDir dir;
  ASSERT_THAT(dir.MakeDirs("parent/cookies"), IsOk());
  WatcherOptions options = CookieOptions();
  options.cookie_dir = "parent/cookies";
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<FsEventsWatcher> watcher,
                              FsEventsWatcher::Open(dir.path(), options));
  ASSERT_THAT(watcher->Start([](const std::vector<WatchEvent>&) {}), IsOk());
  ASSERT_EQ(rmdir(dir.Join("parent/cookies").c_str()), 0);
  ASSERT_EQ(rmdir(dir.Join("parent").c_str()), 0);
  EXPECT_THAT(watcher->Sync(absl::Seconds(15)),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("cookie-")));
  ASSERT_THAT(dir.MakeDirs("parent/cookies"), IsOk());
  EXPECT_THAT(watcher->Sync(absl::Seconds(15)), IsOk());
}

TEST(FsEventsWatcherTest, StopEndsPendingSync) {
  ferret_testing::TempDir dir;
  ASSERT_THAT(dir.MakeDirs("cookies"), IsOk());
  BlockingCallback callback;
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<FsEventsWatcher> watcher,
      FsEventsWatcher::Open(dir.path(), CookieOptions()));
  absl::Cleanup release = [&callback] { callback.Release(); };
  ASSERT_THAT(watcher->Start(callback.Get()), IsOk());
  ASSERT_THAT(dir.WriteFile("file", "x"), IsOk());
  ASSERT_TRUE(callback.WaitUntilEntered());

  absl::Status synced;
  std::thread syncer([&] { synced = watcher->Sync(absl::Seconds(60)); });
  // Let Sync start waiting for its cookie. If Stop runs first instead, Sync
  // still fails with FailedPrecondition.
  std::this_thread::sleep_for(
      absl::ToChronoMilliseconds(absl::Milliseconds(100)));
  // Stop blocks joining the callback thread until the callback is released.
  std::thread stopper([&] { watcher->Stop(); });
  syncer.join();
  EXPECT_THAT(synced, StatusIs(absl::StatusCode::kFailedPrecondition));
  callback.Release();
  stopper.join();
}

}  // namespace
}  // namespace ferret

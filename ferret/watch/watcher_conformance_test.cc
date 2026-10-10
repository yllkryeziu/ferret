// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "ferret/testing/temp_dir.h"
#include "ferret/watch/polling_watcher.h"
#include "ferret/watch/watcher.h"
#ifdef __APPLE__
#include "ferret/watch/fsevents_watcher.h"
#endif

namespace ferret {
namespace {

class WatcherConformanceTest : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    ASSERT_TRUE(dir_.MakeDirs("cookies").ok());
    Open(dir_.path());
  }

  void Open(const std::string& root) {
    WatcherOptions options;
    options.cookie_dir = "cookies";
    options.interval = absl::Hours(1);  // Sync forces polling scans.
    if (GetParam() == 0) {
      auto opened = PollingWatcher::Open(root, options);
      ASSERT_TRUE(opened.ok()) << opened.status();
      watcher_ = std::move(*opened);
    } else {
#ifdef __APPLE__
      auto opened = FsEventsWatcher::Open(root, options);
      ASSERT_TRUE(opened.ok()) << opened.status();
      watcher_ = std::move(*opened);
#endif
    }
    ASSERT_NE(watcher_, nullptr);
    ASSERT_TRUE(watcher_
                    ->Start([this](std::vector<WatchEvent> events) {
                      absl::MutexLock lock(mu_);
                      delivered_.insert(delivered_.end(),
                                        std::make_move_iterator(events.begin()),
                                        std::make_move_iterator(events.end()));
                    })
                    .ok());
  }

  void Sync() { ASSERT_TRUE(watcher_->Sync(absl::Seconds(15)).ok()); }

  void Covers(const std::string& path) {
    absl::MutexLock lock(mu_);
    for (const WatchEvent& event : delivered_) {
      if (const auto* changed = std::get_if<PathChanged>(&event)) {
        if (changed->rel_path == path) return;
      } else {
        const std::string& prefix = std::get<SubtreeChanged>(event).rel_prefix;
        if (prefix.empty() || path == prefix ||
            (path.size() > prefix.size() &&
             path.compare(0, prefix.size(), prefix) == 0 &&
             path[prefix.size()] == '/')) {
          return;
        }
      }
    }
    ADD_FAILURE() << "not covered: " << path << " (" << delivered_.size()
                  << " events)";
  }

  ferret_testing::TempDir dir_;
  absl::Mutex mu_;
  std::vector<WatchEvent> delivered_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<Watcher> watcher_;
};

TEST_P(WatcherConformanceTest, AtomicSave) {
  ASSERT_TRUE(dir_.WriteFile("file.tmp", "new").ok());
  ASSERT_EQ(rename(dir_.Join("file.tmp").c_str(), dir_.Join("file").c_str()),
            0);
  Sync();
  Covers("file");
}

TEST_P(WatcherConformanceTest, VimSave) {
  ASSERT_TRUE(dir_.WriteFile("file", "old").ok());
  Sync();
  ASSERT_EQ(rename(dir_.Join("file").c_str(), dir_.Join("file~").c_str()), 0);
  ASSERT_TRUE(dir_.WriteFile("file", "new").ok());
  ASSERT_EQ(unlink(dir_.Join("file~").c_str()), 0);
  Sync();
  Covers("file");
  // A polling scan cannot observe a backup created and removed between scans.
  if (GetParam() != 0) Covers("file~");
}

TEST_P(WatcherConformanceTest, TruncateThenAppend) {
  ASSERT_TRUE(dir_.WriteFile("file", "old").ok());
  Sync();
  ASSERT_TRUE(dir_.WriteFile("file", "longer content").ok());
  Sync();
  Covers("file");
}

TEST_P(WatcherConformanceTest, CaseOnlyRename) {
  ASSERT_TRUE(dir_.WriteFile("Lower", "content").ok());
  Sync();
  ASSERT_EQ(rename(dir_.Join("Lower").c_str(), dir_.Join("lower").c_str()), 0);
  Sync();
  Covers("lower");
}

TEST_P(WatcherConformanceTest, DirectoryRename) {
  ASSERT_TRUE(dir_.WriteFile("before/nested/file", "x").ok());
  Sync();
  ASSERT_EQ(rename(dir_.Join("before").c_str(), dir_.Join("after").c_str()), 0);
  Sync();
  Covers("before/nested/file");
  Covers("after/nested/file");
}

TEST_P(WatcherConformanceTest, RemoveSubtree) {
  ASSERT_TRUE(dir_.WriteFile("tree/nested/file", "x").ok());
  Sync();
  ASSERT_EQ(unlink(dir_.Join("tree/nested/file").c_str()), 0);
  ASSERT_EQ(rmdir(dir_.Join("tree/nested").c_str()), 0);
  ASSERT_EQ(rmdir(dir_.Join("tree").c_str()), 0);
  Sync();
  Covers("tree/nested/file");
}

TEST_P(WatcherConformanceTest, SymlinkedRoot) {
  watcher_->Stop();
  ASSERT_TRUE(dir_.MakeDirs("actual/cookies").ok());
  ASSERT_EQ(symlink(dir_.Join("actual").c_str(), dir_.Join("alias").c_str()),
            0);
  Open(dir_.Join("alias"));
  EXPECT_EQ(watcher_->root(), dir_.Join("actual"));
  ASSERT_TRUE(dir_.WriteFile("actual/file", "x").ok());
  Sync();
  Covers("file");
}

TEST_P(WatcherConformanceTest, BurstOfTenThousand) {
  ASSERT_TRUE(dir_.MakeDirs("burst").ok());
  for (int i = 0; i < 10000; ++i) {
    ASSERT_TRUE(dir_.WriteFile(absl::StrCat("burst/", i), "x").ok());
  }
  Sync();
  for (int i = 0; i < 10000; ++i) Covers(absl::StrCat("burst/", i));
}

TEST_P(WatcherConformanceTest, SyncNeverDeliversCookiePaths) {
  for (int i = 0; i < 5; ++i) {
    ASSERT_TRUE(dir_.WriteFile(absl::StrCat("file-", i), "x").ok());
    Sync();
  }
  // The removal of the last cookie can arrive after its Sync has returned.
  std::this_thread::sleep_for(
      absl::ToChronoMilliseconds(absl::Milliseconds(100)));
  watcher_->Stop();
  absl::MutexLock lock(mu_);
  ASSERT_FALSE(delivered_.empty());
  for (const WatchEvent& event : delivered_) {
    const std::string& path = std::holds_alternative<PathChanged>(event)
                                  ? std::get<PathChanged>(event).rel_path
                                  : std::get<SubtreeChanged>(event).rel_prefix;
    EXPECT_EQ(path.find("cookies/cookie-"), std::string::npos) << path;
  }
}

#ifdef __APPLE__
TEST(FsEventsWatcherTest, ResolvesCookieDirectoryInsideRoot) {
  ferret_testing::TempDir dir;
  ferret_testing::TempDir outside;
  ASSERT_TRUE(dir.MakeDirs("real/cookies").ok());
  ASSERT_EQ(
      symlink(dir.Join("real/cookies").c_str(), dir.Join("alias").c_str()), 0);
  ASSERT_EQ(symlink(outside.path().c_str(), dir.Join("escape").c_str()), 0);
  WatcherOptions options;
  options.cookie_dir = "escape";
  auto escaped = FsEventsWatcher::Open(dir.path(), options);
  ASSERT_FALSE(escaped.ok());
  EXPECT_EQ(escaped.status().code(), absl::StatusCode::kInvalidArgument);

  options.cookie_dir = "alias";
  // Both outlive the watcher, whose destructor joins the callback thread.
  absl::Mutex mu;
  std::vector<WatchEvent> delivered;
  auto opened = FsEventsWatcher::Open(dir.path(), options);
  ASSERT_TRUE(opened.ok()) << opened.status();
  std::unique_ptr<FsEventsWatcher> watcher = std::move(*opened);
  ASSERT_TRUE(watcher
                  ->Start([&](std::vector<WatchEvent> events) {
                    absl::MutexLock lock(mu);
                    delivered.insert(delivered.end(),
                                     std::make_move_iterator(events.begin()),
                                     std::make_move_iterator(events.end()));
                  })
                  .ok());
  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(dir.WriteFile(absl::StrCat("file-", i), "x").ok());
    ASSERT_TRUE(watcher->Sync(absl::Seconds(15)).ok());
  }
  std::this_thread::sleep_for(
      absl::ToChronoMilliseconds(absl::Milliseconds(100)));
  watcher->Stop();
  absl::MutexLock lock(mu);
  ASSERT_FALSE(delivered.empty());
  for (const WatchEvent& event : delivered) {
    const std::string& path = std::holds_alternative<PathChanged>(event)
                                  ? std::get<PathChanged>(event).rel_path
                                  : std::get<SubtreeChanged>(event).rel_prefix;
    EXPECT_EQ(path.find("real/cookies/cookie-"), std::string::npos) << path;
  }
}

TEST(FsEventsWatcherTest, DestroyWhileStreamIsActive) {
  ferret_testing::TempDir dir;
  ASSERT_TRUE(dir.MakeDirs("cookies").ok());
  WatcherOptions options;
  options.cookie_dir = "cookies";
  std::atomic<int> callbacks = 0;
  for (int iteration = 0; iteration < 50; ++iteration) {
    auto opened = FsEventsWatcher::Open(dir.path(), options);
    ASSERT_TRUE(opened.ok()) << opened.status();
    std::unique_ptr<FsEventsWatcher> watcher = std::move(*opened);
    ASSERT_TRUE(
        watcher->Start([&](const std::vector<WatchEvent>&) { ++callbacks; })
            .ok());
    std::atomic<bool> writing = true;
    std::atomic<bool> write_failed = false;
    std::thread writer([&] {
      for (int i = 0; writing.load(); ++i) {
        if (!dir.WriteFile(absl::StrCat("stream-", iteration, "-", i), "x")
                 .ok()) {
          write_failed = true;
          return;
        }
        std::this_thread::sleep_for(
            absl::ToChronoMilliseconds(absl::Milliseconds(1)));
      }
    });
    // Let the stream enter its dispatch callback before tearing it down.
    std::this_thread::sleep_for(
        absl::ToChronoMilliseconds(absl::Milliseconds(50)));
    watcher.reset();
    writing = false;
    writer.join();
    EXPECT_FALSE(write_failed);
  }
  EXPECT_GT(callbacks.load(), 0);
}
#endif

TEST_P(WatcherConformanceTest, IdleSyncAndStop) {
  Sync();
  {
    absl::MutexLock lock(mu_);
    for (const WatchEvent& event : delivered_) {
      if (const auto* path = std::get_if<PathChanged>(&event)) {
        EXPECT_EQ(path->rel_path.find("cookie-"), std::string::npos);
      } else {
        EXPECT_EQ(std::get<SubtreeChanged>(event).rel_prefix.find("cookie-"),
                  std::string::npos);
      }
    }
  }
  watcher_->Stop();
  {
    absl::MutexLock lock(mu_);
    delivered_.clear();
  }
  ASSERT_TRUE(dir_.WriteFile("after-stop", "x").ok());
  absl::MutexLock lock(mu_);
  EXPECT_TRUE(delivered_.empty());
  EXPECT_FALSE(watcher_->Sync(absl::Seconds(1)).ok());
}

TEST(WatcherFactoryTest, StartsAndSynchronizes) {
  ferret_testing::TempDir dir;
  ASSERT_TRUE(dir.MakeDirs("cookies").ok());
  WatcherOptions options;
  options.cookie_dir = "cookies";
  // Outlives the watcher, whose destructor joins the callback thread.
  std::vector<WatchEvent> delivered;
  auto opened = NewWatcher(dir.path(), options);
  ASSERT_TRUE(opened.ok()) << opened.status();
  std::unique_ptr<Watcher> watcher = std::move(*opened);
  ASSERT_TRUE(watcher
                  ->Start([&](std::vector<WatchEvent> events) {
                    delivered.insert(delivered.end(),
                                     std::make_move_iterator(events.begin()),
                                     std::make_move_iterator(events.end()));
                  })
                  .ok());
  ASSERT_TRUE(dir.WriteFile("file", "x").ok());
  ASSERT_TRUE(watcher->Sync(absl::Seconds(5)).ok());
  EXPECT_FALSE(delivered.empty());
}

#ifdef __APPLE__
INSTANTIATE_TEST_SUITE_P(AllWatchers, WatcherConformanceTest,
                         ::testing::Values(0, 1));
#else
INSTANTIATE_TEST_SUITE_P(AllWatchers, WatcherConformanceTest,
                         ::testing::Values(0));
#endif

}  // namespace
}  // namespace ferret

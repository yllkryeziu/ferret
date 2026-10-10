// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/watch/polling_watcher.h"

#include <sys/stat.h>
#include <unistd.h>

#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
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
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::HasSubstr;
using ::testing::VariantWith;

TEST(PollingWatcherTest, IgnoresPrunedDirectoriesAndDoesNotFollowSymlinks) {
  ferret_testing::TempDir dir;
  ASSERT_TRUE(dir.WriteFile(".git/old", "x").ok());
  ASSERT_TRUE(dir.WriteFile("outside/file", "x").ok());
  ASSERT_EQ(symlink(dir.Join("outside").c_str(), dir.Join("link").c_str()), 0);
  // Outlives the watcher, whose destructor joins the callback thread.
  std::vector<WatchEvent> delivered;
  auto opened = PollingWatcher::Open(dir.path());
  ASSERT_TRUE(opened.ok()) << opened.status();
  std::unique_ptr<PollingWatcher> watcher = std::move(*opened);
  ASSERT_TRUE(watcher
                  ->Start([&](std::vector<WatchEvent> events) {
                    delivered.insert(delivered.end(),
                                     std::make_move_iterator(events.begin()),
                                     std::make_move_iterator(events.end()));
                  })
                  .ok());
  ASSERT_TRUE(watcher->Sync(absl::Seconds(2)).ok());
  EXPECT_TRUE(delivered.empty());  // The first scan was a baseline.
  ASSERT_TRUE(dir.WriteFile(".git/new", "x").ok());
  ASSERT_TRUE(dir.WriteFile("outside/file", "changed").ok());
  ASSERT_TRUE(watcher->Sync(absl::Seconds(2)).ok());
  bool found = false;
  for (const WatchEvent& event : delivered) {
    if (const auto* path = std::get_if<PathChanged>(&event)) {
      EXPECT_NE(path->rel_path, ".git/new");
      EXPECT_NE(path->rel_path, "link/file");
      if (path->rel_path == "outside/file") found = true;
    }
  }
  EXPECT_TRUE(found);
}

TEST(PollingWatcherTest, PrunesGitObjectsButReportsHeadAndIndex) {
  ferret_testing::TempDir dir;
  ASSERT_TRUE(dir.WriteFile(".git/HEAD", "first").ok());
  ASSERT_TRUE(dir.WriteFile(".git/index", "first").ok());
  ASSERT_TRUE(dir.WriteFile(".git/objects/object", "first").ok());
  std::vector<WatchEvent> delivered;
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PollingWatcher> watcher,
                              PollingWatcher::Open(dir.path()));
  ASSERT_TRUE(watcher
                  ->Start([&](std::vector<WatchEvent> events) {
                    delivered.insert(delivered.end(),
                                     std::make_move_iterator(events.begin()),
                                     std::make_move_iterator(events.end()));
                  })
                  .ok());
  ASSERT_TRUE(dir.WriteFile(".git/HEAD", "second").ok());
  ASSERT_TRUE(dir.WriteFile(".git/index", "second").ok());
  ASSERT_TRUE(dir.WriteFile(".git/objects/object", "second").ok());
  ASSERT_TRUE(watcher->Sync(absl::Seconds(2)).ok());
  bool head = false;
  bool index = false;
  for (const WatchEvent& event : delivered) {
    if (const auto* file = std::get_if<PathChanged>(&event)) {
      head |= file->rel_path == ".git/HEAD";
      index |= file->rel_path == ".git/index";
      EXPECT_NE(file->rel_path, ".git/objects/object");
    }
  }
  EXPECT_TRUE(head);
  EXPECT_TRUE(index);
}

TEST(PollingWatcherTest, RejectsNonDirectoryRootAndInvalidInterval) {
  ferret_testing::TempDir dir;
  ASSERT_TRUE(dir.WriteFile("file", "x").ok());
  auto invalid = PollingWatcher::Open(dir.Join("file"));
  ASSERT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.status().code(), absl::StatusCode::kInvalidArgument);
  WatcherOptions options;
  options.interval = absl::ZeroDuration();
  invalid = PollingWatcher::Open(dir.path(), options);
  ASSERT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.status().code(), absl::StatusCode::kInvalidArgument);
  options.interval = absl::Seconds(1);
  options.quiet_window = absl::ZeroDuration();
  invalid = PollingWatcher::Open(dir.path(), options);
  ASSERT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(PollingWatcherTest, StartAndSyncFollowTheLifecycle) {
  ferret_testing::TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PollingWatcher> watcher,
                              PollingWatcher::Open(dir.path()));
  EXPECT_THAT(watcher->Sync(absl::Seconds(1)),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(watcher->Start(nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));

  ASSERT_THAT(watcher->Start([](const std::vector<WatchEvent>&) {}), IsOk());
  EXPECT_THAT(watcher->Start([](const std::vector<WatchEvent>&) {}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(watcher->Sync(absl::Seconds(2)), IsOk());

  watcher->Stop();
  watcher->Stop();
  EXPECT_THAT(watcher->Sync(absl::Seconds(1)),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(watcher->Start([](const std::vector<WatchEvent>&) {}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(PollingWatcherTest, SyncReportsScanErrorAfterWholeRootHint) {
  if (geteuid() == 0) GTEST_SKIP() << "root ignores directory permissions";
  ferret_testing::TempDir dir;
  ASSERT_THAT(dir.MakeDirs("locked"), IsOk());
  // Outlives the watcher, whose destructor joins the callback thread.
  std::vector<WatchEvent> delivered;
  auto collect = [&delivered](std::vector<WatchEvent> events) {
    delivered.insert(delivered.end(), std::make_move_iterator(events.begin()),
                     std::make_move_iterator(events.end()));
  };
  WatcherOptions options;
  options.interval = absl::Hours(1);  // Only Sync scans.
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PollingWatcher> watcher,
                              PollingWatcher::Open(dir.path(), options));
  ASSERT_THAT(watcher->Start(collect), IsOk());

  const std::string locked = dir.Join("locked");
  ASSERT_EQ(chmod(locked.c_str(), 0), 0);
  // TempDir cannot delete a directory it cannot read.
  absl::Cleanup unlock = [&locked] { chmod(locked.c_str(), 0755); };
  EXPECT_THAT(watcher->Sync(absl::Seconds(2)),
              StatusIs(absl::StatusCode::kPermissionDenied,
                       HasSubstr(absl::StrCat("opendir(", locked, ")"))));
  EXPECT_THAT(delivered, ElementsAre(VariantWith<SubtreeChanged>(
                             Field(&SubtreeChanged::rel_prefix, ""))));

  ASSERT_EQ(chmod(locked.c_str(), 0755), 0);
  EXPECT_THAT(watcher->Sync(absl::Seconds(2)), IsOk());
}

}  // namespace
}  // namespace ferret

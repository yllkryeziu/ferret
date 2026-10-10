// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/watch/fake_watcher.h"

#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "ferret/testing/temp_dir.h"
#include "ferret/watch/watcher.h"

namespace ferret {
namespace {

TEST(FakeWatcherTest, SyncFlushesOnceAndStopPreventsDelivery) {
  ferret_testing::TempDir dir;
  // Outlives the watcher, whose destructor joins the callback thread.
  std::vector<WatchEvent> delivered;
  auto opened = FakeWatcher::Open(dir.path());
  ASSERT_TRUE(opened.ok()) << opened.status();
  std::unique_ptr<FakeWatcher> watcher = std::move(*opened);
  ASSERT_TRUE(watcher
                  ->Start([&](std::vector<WatchEvent> events) {
                    delivered.insert(delivered.end(),
                                     std::make_move_iterator(events.begin()),
                                     std::make_move_iterator(events.end()));
                  })
                  .ok());
  watcher->Push(PathChanged{"a"});
  watcher->Push(PathChanged{"a"});
  ASSERT_TRUE(watcher->Sync(absl::Seconds(2)).ok());
  ASSERT_EQ(delivered.size(), 1u);
  EXPECT_EQ(std::get<PathChanged>(delivered[0]).rel_path, "a");
  watcher->Push(PathChanged{"b"});
  watcher->Stop();
  watcher->Push(PathChanged{"c"});
  EXPECT_EQ(delivered.size(), 1u);
  EXPECT_FALSE(watcher->Sync(absl::Seconds(1)).ok());
}

TEST(FakeWatcherTest, DeliverFlushesWithoutSync) {
  ferret_testing::TempDir dir;
  // Outlives the watcher, whose destructor joins the callback thread.
  int batches = 0;
  auto opened = FakeWatcher::Open(dir.path());
  ASSERT_TRUE(opened.ok());
  std::unique_ptr<FakeWatcher> watcher = std::move(*opened);
  ASSERT_TRUE(
      watcher->Start([&](const std::vector<WatchEvent>&) { ++batches; }).ok());
  watcher->Push(SubtreeChanged{""});
  watcher->Deliver();
  ASSERT_TRUE(watcher->Sync(absl::Seconds(2)).ok());
  EXPECT_EQ(batches, 1);
}

}  // namespace
}  // namespace ferret

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/watch/internal/event_batcher.h"

#include <string>
#include <vector>

#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "ferret/watch/watcher.h"

namespace ferret {
namespace {

TEST(EventBatcherTest, QuietWindowAndMaximumDelay) {
  absl::Time now = absl::UnixEpoch();
  watch_internal::EventBatcher batcher(
      absl::Milliseconds(20), absl::Milliseconds(50), [&now] { return now; });
  batcher.Add(PathChanged{"a"});
  EXPECT_TRUE(batcher.Due());
  for (int i = 0; i < 7; ++i) {
    batcher.Add(PathChanged{"initial" + std::to_string(i)});
  }
  ASSERT_EQ(batcher.Flush().size(), 8u);
  batcher.Add(PathChanged{"a"});
  now += absl::Milliseconds(19);
  EXPECT_FALSE(batcher.Due());
  batcher.Add(PathChanged{"b"});
  now += absl::Milliseconds(19);
  EXPECT_FALSE(batcher.Due());
  batcher.Add(PathChanged{"b"});
  for (int i = 0; i < 6; ++i) {
    batcher.Add(PathChanged{"more" + std::to_string(i)});
  }
  now += absl::Milliseconds(12);
  EXPECT_TRUE(batcher.Due());
  EXPECT_EQ(batcher.Flush().size(), 8u);
  EXPECT_FALSE(batcher.Due());
  batcher.Add(PathChanged{"c"});
  for (int i = 0; i < 4; ++i) {
    now += absl::Milliseconds(12);
    batcher.Add(PathChanged{"c"});
    EXPECT_FALSE(batcher.Due());
  }
  now += absl::Milliseconds(2);
  EXPECT_TRUE(batcher.Due());
}

TEST(EventBatcherTest, FirstEventAfterQuietIsImmediate) {
  absl::Time now = absl::UnixEpoch();
  watch_internal::EventBatcher batcher(
      absl::Milliseconds(20), absl::Milliseconds(50), [&now] { return now; });
  batcher.Add(PathChanged{"first"});
  EXPECT_TRUE(batcher.Due());
  EXPECT_EQ(batcher.Flush().size(), 1u);
  now += absl::Milliseconds(1);
  batcher.Add(PathChanged{"burst"});
  EXPECT_TRUE(batcher.Due());
  now += absl::Milliseconds(20);
  EXPECT_TRUE(batcher.Due());
  EXPECT_EQ(batcher.Flush().size(), 1u);
  now += absl::Milliseconds(21);
  batcher.Add(PathChanged{"next"});
  EXPECT_TRUE(batcher.Due());
}

TEST(EventBatcherTest, SyncBarrierEndsBurst) {
  absl::Time now = absl::UnixEpoch();
  watch_internal::EventBatcher batcher(
      absl::Milliseconds(20), absl::Milliseconds(50), [&now] { return now; });
  batcher.Add(PathChanged{"first"});
  for (int i = 0; i < 7; ++i) {
    batcher.Add(PathChanged{"initial" + std::to_string(i)});
  }
  batcher.Flush();
  now += absl::Milliseconds(1);
  batcher.Add(PathChanged{"second"});
  EXPECT_FALSE(batcher.Due());
  batcher.Flush();
  batcher.EndBurst();
  batcher.Add(PathChanged{"next"});
  EXPECT_TRUE(batcher.Due());
}

TEST(EventBatcherTest, ThousandWritesCoalesceWhileBusy) {
  absl::Time now = absl::UnixEpoch();
  watch_internal::EventBatcher batcher(
      absl::Milliseconds(20), absl::Milliseconds(50), [&now] { return now; });
  batcher.Add(PathChanged{"first"});
  for (int i = 0; i < 7; ++i) {
    batcher.Add(PathChanged{"initial" + std::to_string(i)});
  }
  EXPECT_TRUE(batcher.Due());
  ASSERT_EQ(batcher.Flush().size(), 8u);
  for (int i = 0; i < 1000; ++i) {
    batcher.Add(PathChanged{"file" + std::to_string(i)});
    now += absl::Microseconds(10);
    EXPECT_FALSE(batcher.Due());
  }
  EXPECT_EQ(batcher.Flush().size(), 1000u);
}

TEST(EventBatcherTest, SubtreeAbsorbsDescendantsOnComponentBoundaries) {
  absl::Time now = absl::UnixEpoch();
  watch_internal::EventBatcher batcher(
      absl::Milliseconds(20), absl::Milliseconds(50), [&now] { return now; });
  batcher.Add(PathChanged{"dir/file"});
  batcher.Add(PathChanged{"dir/file"});
  batcher.Add(PathChanged{"directory/file"});
  batcher.Add(SubtreeChanged{"dir"});
  batcher.Add(PathChanged{"dir/other"});
  std::vector<WatchEvent> events = batcher.Flush();
  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(std::get<SubtreeChanged>(events[0]).rel_prefix, "dir");
  EXPECT_EQ(std::get<PathChanged>(events[1]).rel_path, "directory/file");
  batcher.Add(SubtreeChanged{"dir/nested"});
  batcher.Add(SubtreeChanged{"dir"});
  batcher.Add(SubtreeChanged{""});
  events = batcher.Flush();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(std::get<SubtreeChanged>(events[0]).rel_prefix, "");
}

}  // namespace
}  // namespace ferret

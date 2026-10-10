// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/reaper.h"

#include <atomic>
#include <thread>

#include "absl/synchronization/notification.h"
#include "gtest/gtest.h"

namespace ferret {
namespace {

TEST(ReaperTest, RunsOffSubmittingThreadAndDrainsOnDestruction) {
  const std::thread::id caller = std::this_thread::get_id();
  std::atomic<int> count{0};
  std::thread::id executor;
  absl::Notification first_done;
  {
    Reaper reaper;
    reaper.Submit([&] {
      executor = std::this_thread::get_id();
      first_done.Notify();
    });
    first_done.WaitForNotification();
    for (int i = 0; i < 100; ++i) {
      reaper.Submit([&] { count.fetch_add(1, std::memory_order_relaxed); });
    }
  }
  EXPECT_NE(executor, caller);
  EXPECT_EQ(count.load(std::memory_order_relaxed), 100);
}

}  // namespace
}  // namespace ferret

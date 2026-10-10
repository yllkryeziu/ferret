// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/base/system_info.h"

#include <unistd.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include <bit>
#include <cstddef>
#include <cstdint>
#include <thread>

#include "gtest/gtest.h"

namespace ferret {
namespace {

TEST(SystemInfoTest, PerformanceCoreCountIsBetweenOneAndAllCores) {
  const int count = PerformanceCoreCount();
  EXPECT_GE(count, 1);
  if (const unsigned int all = std::thread::hardware_concurrency(); all > 0) {
    EXPECT_LE(count, static_cast<int>(all));
  }
}

#if defined(__APPLE__)
TEST(SystemInfoTest, PerformanceCoreCountPrefersPerflevel0) {
  int perflevel0 = 0;
  size_t size = sizeof(perflevel0);
  if (sysctlbyname("hw.perflevel0.physicalcpu", &perflevel0, &size, nullptr,
                   0) != 0) {
    GTEST_SKIP() << "no performance levels on this machine";
  }
  EXPECT_EQ(PerformanceCoreCount(), perflevel0);
}
#endif

TEST(SystemInfoTest, PageSizeIsThePowerOfTwoTheKernelReports) {
  const int64_t page_size = PageSize();
  EXPECT_TRUE(std::has_single_bit(static_cast<uint64_t>(page_size)));
  EXPECT_EQ(page_size, getpagesize());
#if defined(__APPLE__) && defined(__aarch64__)
  EXPECT_EQ(page_size, 16384);
#endif
}

}  // namespace
}  // namespace ferret

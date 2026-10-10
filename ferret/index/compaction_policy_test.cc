// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/compaction_policy.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

namespace ferret {
namespace {

TEST(CompactionPolicyTest, SelectsAtMostOneJobPerLane) {
  constexpr uint64_t kMib = uint64_t{1024} * 1024;
  struct Case {
    const char* name;
    std::vector<CompactionSegment> segments;
    std::vector<std::vector<uint64_t>> expected;
    // NOLINTNEXTLINE(readability-redundant-member-init): Required by build.
    std::vector<CompactionSelection::Lane> lanes = {};
  };
  const std::vector<Case> cases = {
      {"empty", {}, {}},
      {"tier",
       {{4, kMib, 10, 0}, {3, kMib, 10, 0}, {2, kMib, 10, 0}, {1, kMib, 10, 0}},
       {{1, 2, 3, 4}}},
      {"boundary", {{1, 16 * kMib, 4, 1}}, {{1}}},
      {"belowThreshold", {{1, kMib, 5, 1}}, {}},
      {"zeroDocs", {{1, kMib, 0, 0}}, {}},
      {"twoLanes",
       {{1, kMib, 4, 1}, {2, 300 * kMib, 4, 1}},
       {{1}, {2}},
       {CompactionSelection::Lane::kSmall, CompactionSelection::Lane::kLarge}},
      {"laneBoundary",
       {{1, 256 * kMib, 4, 1}},
       {{1}},
       {CompactionSelection::Lane::kLarge}},
      {"oldestRewrite", {{2, kMib, 4, 1}, {1, kMib, 4, 1}}, {{1}}},
      {"forcedBeatsRewrites",
       {{1, kMib, 4, 1},
        {2, 18 * kMib, 4, 1},
        {3, 80 * kMib, 4, 1},
        {4, 300 * kMib, 4, 1},
        {5, 2 * kMib, 4, 1},
        {6, 22 * kMib, 4, 1},
        {7, 90 * kMib, 4, 1},
        {8, 350 * kMib, 4, 1},
        {9, 1024 * kMib, 4, 1}},
       {{1, 5, 2, 6}, {4}}},
      {"forcedSmallest",
       {{9, kMib, 10, 0},
        {8, 18 * kMib, 10, 0},
        {7, 18 * kMib, 10, 0},
        {6, 18 * kMib, 10, 0},
        {5, 80 * kMib, 10, 0},
        {4, 80 * kMib, 10, 0},
        {3, 300 * kMib, 10, 0},
        {2, 300 * kMib, 10, 0},
        {1, 300 * kMib, 10, 0}},
       {{9, 6, 7, 8}}},
  };
  for (const Case& test : cases) {
    SCOPED_TRACE(test.name);
    auto jobs = PickCompactions(test.segments);
    std::vector<std::vector<uint64_t>> numbers;
    numbers.reserve(jobs.size());
    for (const auto& job : jobs) numbers.push_back(job.file_numbers);
    EXPECT_EQ(numbers, test.expected);
    if (!test.lanes.empty()) {
      std::vector<CompactionSelection::Lane> lanes;
      lanes.reserve(jobs.size());
      for (const auto& job : jobs) lanes.push_back(job.lane);
      EXPECT_EQ(lanes, test.lanes);
    }
  }
}

TEST(CompactionPolicyTest, UsesConfiguredThresholds) {
  constexpr uint64_t kMib = uint64_t{1024} * 1024;
  CompactionOptions options;
  options.tier_base_bytes = 8 * kMib;
  options.tier_growth_factor = 2;
  options.tier_min_segments = 2;
  options.rewrite_dead_percent = 50;
  options.max_live_segments = 3;
  options.small_lane_bytes = 6 * kMib;
  options.forced_merge_segments = 2;
  const std::vector<CompactionSegment> segments = {
      {2, 2 * kMib, 10, 0}, {1, kMib, 10, 0}, {3, 16 * kMib, 10, 5}};
  const auto jobs = PickCompactions(segments, options);
  ASSERT_EQ(jobs.size(), size_t{2});
  EXPECT_EQ(jobs[0].file_numbers, (std::vector<uint64_t>{1, 2}));
  EXPECT_EQ(jobs[0].lane, CompactionSelection::Lane::kSmall);
  EXPECT_EQ(jobs[1].file_numbers, (std::vector<uint64_t>{3}));
  EXPECT_EQ(jobs[1].lane, CompactionSelection::Lane::kLarge);
}

}  // namespace
}  // namespace ferret

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Selects tier merges, forced merges and rewrites in priority order without
// mutating the live segment set.

#include "ferret/index/compaction_policy.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "absl/types/span.h"

namespace ferret {
namespace {

uint32_t Tier(uint64_t bytes, const CompactionOptions& options) {
  uint32_t tier = 0;
  uint64_t boundary = options.tier_base_bytes;
  while (options.tier_growth_factor > 1 && boundary != 0 &&
         boundary <= std::numeric_limits<uint64_t>::max() /
                         options.tier_growth_factor &&
         bytes >= boundary * options.tier_growth_factor) {
    boundary *= options.tier_growth_factor;
    ++tier;
  }
  return tier;
}

}  // namespace

std::vector<CompactionSelection> PickCompactions(
    absl::Span<const CompactionSegment> segments,
    const CompactionOptions& options) {
  std::vector<CompactionSegment> sorted(segments.begin(), segments.end());
  std::sort(sorted.begin(), sorted.end(), [&](const auto& a, const auto& b) {
    if (Tier(a.bytes, options) != Tier(b.bytes, options)) {
      return Tier(a.bytes, options) < Tier(b.bytes, options);
    }
    return a.file_number < b.file_number;
  });
  std::vector<std::vector<CompactionSegment>> jobs;
  for (size_t first = 0; first < sorted.size();) {
    size_t end = first + 1;
    while (end < sorted.size() && Tier(sorted[end].bytes, options) ==
                                      Tier(sorted[first].bytes, options)) {
      ++end;
    }
    if (end - first >= options.tier_min_segments) {
      jobs.emplace_back(sorted.begin() + static_cast<std::ptrdiff_t>(first),
                        sorted.begin() + static_cast<std::ptrdiff_t>(end));
      sorted.erase(sorted.begin() + static_cast<std::ptrdiff_t>(first),
                   sorted.begin() + static_cast<std::ptrdiff_t>(end));
    } else {
      first = end;
    }
  }
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    return a.file_number < b.file_number;
  });
  // When tiers cannot reduce an overfull set, a merge takes precedence over
  // single-segment rewrites, which cannot reduce the live segment count.
  if (jobs.empty() && segments.size() > options.max_live_segments &&
      sorted.size() >= options.forced_merge_segments &&
      options.forced_merge_segments > 1) {
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
      if (a.bytes != b.bytes) return a.bytes < b.bytes;
      return a.file_number < b.file_number;
    });
    jobs.emplace_back(sorted.begin(),
                      sorted.begin() + static_cast<std::ptrdiff_t>(
                                           options.forced_merge_segments));
    sorted.erase(sorted.begin(),
                 sorted.begin() + static_cast<std::ptrdiff_t>(
                                      options.forced_merge_segments));
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
      return a.file_number < b.file_number;
    });
  }
  for (size_t i = 0; i < sorted.size();) {
    const auto& s = sorted[i];
    if (s.num_docs > 0 &&
        uint64_t{s.dead_docs} * 100 >=
            uint64_t{s.num_docs} * options.rewrite_dead_percent) {
      jobs.push_back({s});
      sorted.erase(sorted.begin() + static_cast<std::ptrdiff_t>(i));
    } else {
      ++i;
    }
  }
  std::vector<CompactionSelection> result;
  for (const auto& job : jobs) {
    uint64_t total = 0;
    for (const auto& s : job) total += s.bytes;
    const auto lane = total < options.small_lane_bytes
                          ? CompactionSelection::Lane::kSmall
                          : CompactionSelection::Lane::kLarge;
    if (std::any_of(result.begin(), result.end(), [&](const auto& selection) {
          return selection.lane == lane;
        })) {
      continue;
    }
    CompactionSelection selection{.lane = lane};
    for (const auto& s : job) selection.file_numbers.push_back(s.file_number);
    result.push_back(std::move(selection));
  }
  return result;
}

}  // namespace ferret

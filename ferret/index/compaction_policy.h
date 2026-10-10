// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Pure tiered compaction selection. Each call proposes at most one job per
// lane; the caller decides when to schedule them.

#ifndef FERRET_INDEX_COMPACTION_POLICY_H_
#define FERRET_INDEX_COMPACTION_POLICY_H_

#include <cstdint>
#include <vector>

#include "absl/time/time.h"
#include "absl/types/span.h"

namespace ferret {

struct CompactionOptions {
  // Disables background compaction when false.
  bool enabled = true;
  // Delay before the first retry after a failed compaction.
  absl::Duration retry_initial_delay = absl::Milliseconds(200);
  // Maximum delay between failed compaction retries.
  absl::Duration retry_max_delay = absl::Minutes(1);
  // The base size of tier zero.
  uint64_t tier_base_bytes = uint64_t{4} * 1024 * 1024;
  // The multiplicative size step between tiers.
  uint64_t tier_growth_factor = 4;
  // The minimum segment count that triggers a tier merge.
  uint32_t tier_min_segments = 4;
  // The minimum dead-document percentage that triggers a rewrite.
  uint32_t rewrite_dead_percent = 25;
  // The maximum live segment count before a forced merge.
  uint32_t max_live_segments = 8;
  // Input total below this size belongs to the small lane.
  uint64_t small_lane_bytes = uint64_t{256} * 1024 * 1024;
  // The number of smallest segments in a forced merge.
  uint32_t forced_merge_segments = 4;
};

struct CompactionSegment {
  // The unique file number, used to break ties and order rewrites.
  uint64_t file_number;
  // The segment file size used for tier and lane selection.
  uint64_t bytes;
  // The total number of documents, including dead documents.
  uint32_t num_docs;
  // The number of documents eligible to be dropped.
  uint32_t dead_docs;
};

struct CompactionSelection {
  enum class Lane {
    // The input total is strictly below small_lane_bytes.
    kSmall,
    // The input total is at least small_lane_bytes.
    kLarge,
  };
  // The lane on which this job may run.
  Lane lane;
  // The file numbers of the disjoint inputs, in selection order.
  std::vector<uint64_t> file_numbers;
};

// Selects up to one job per lane, independent of input order. Tier merges
// precede forced merges and rewrites; the smallest eligible tier wins, with
// ties broken by file number. Jobs do not share inputs. A forced merge takes
// the smallest `forced_merge_segments` remaining files, breaking size ties
// by file number, if no tier merge exists. Rewrites run oldest first.
// O(n²) time and O(n) memory. Thread-safe.
std::vector<CompactionSelection> PickCompactions(
    absl::Span<const CompactionSegment> segments,
    const CompactionOptions& options = {});

}  // namespace ferret

#endif  // FERRET_INDEX_COMPACTION_POLICY_H_

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Merges the sorted runs of a partitioned build into encoded posting lists,
// in parallel.
//
// The runs come from workers that numbered chunks as they went (provisional
// ids); `remap` gives each provisional id its place in the segment's rank
// order. The trigram space is cut into ranges. For each range a task gathers,
// for every trigram, its pieces from all runs, maps their ids, sorts the list
// and encodes it. Ranges are committed in order, so the sink sees the lists in
// trigram order, as PostingAccumulator::ForEachList() delivers them.

#ifndef FERRET_INDEX_INTERNAL_RUN_MERGE_H_
#define FERRET_INDEX_INTERNAL_RUN_MERGE_H_

#include <cstddef>
#include <cstdint>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "ferret/base/thread_pool.h"
#include "ferret/index/internal/posting_accumulator.h"
#include "ferret/postings/ids.h"
#include "ferret/text/trigram.h"

namespace ferret::segment_internal {

// Receives one list. `count` ids. If the list is inlined
// (IsInlinedPostingList(count)), `inline_postings` holds them as the
// dictionary stores them and `encoded` is empty; otherwise `encoded` is the
// list's encoding.
using ListSink = absl::FunctionRef<absl::Status(Trigram trigram, size_t count,
                                                uint64_t inline_postings,
                                                absl::string_view encoded)>;

// Calls `sink` for each trigram that has postings, in increasing order. The
// ids in `runs` must be below remap.size(), and mapped ids distinct within a
// trigram. The runs must stay alive and unchanged. `pool` may be null.
// Returns the first error from `sink`, or InvalidArgument for a list longer
// than the format allows.
absl::Status MergeRuns(absl::Span<const PostingAccumulator::Run> runs,
                       absl::Span<const ChunkId> remap, ThreadPool* pool,
                       int threads, ListSink sink);

}  // namespace ferret::segment_internal

#endif  // FERRET_INDEX_INTERNAL_RUN_MERGE_H_

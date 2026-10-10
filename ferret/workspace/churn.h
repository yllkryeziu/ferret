// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The churn-and-merge benchmark: simulated agents query one registered worktree
// at a fixed rate while a large `git checkout` changes it, and the latencies
// are split into four windows around the checkout.
//
// The client schedule and the window accounting are plain functions so that
// they can be tested exactly; RunChurn drives a real Workspace.

#ifndef FERRET_WORKSPACE_CHURN_H_
#define FERRET_WORKSPACE_CHURN_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

#include "ferret/search/search_types.h"
#include "ferret/workspace/workspace.h"

namespace ferret {

struct ClientScheduleOptions {
  uint64_t seed = 1;
  int client = 0;
  // Mean queries per second. Positive.
  double rate_hz = 2.0;
  // 0 sends at exact multiples of the period; 1 moves each send by up to half
  // a period either way. In [0, 1].
  double jitter = 0.5;
  // Intended send times at or after this are not generated.
  absl::Duration length = absl::Seconds(10);
};

struct ScheduledQuery {
  // Intended send time, from the start of the run.
  absl::Duration at;
  // Index into the query list.
  size_t query = 0;
};

// The send times and queries of one client: open loop, so they depend on the
// seed and the options only, never on how long a response takes. Send i is at
// phase + i / rate_hz, moved by jitter, where the phase is drawn in
// [0, 1 / rate_hz). Times ascend. A longer `length` extends the same schedule:
// the shorter one is a prefix. `num_queries` must be positive.
std::vector<ScheduledQuery> ClientSchedule(const ClientScheduleOptions& options,
                                           size_t num_queries);

enum class QueryOutcome { kOk, kStale, kFailed };

struct QuerySample {
  int client = 0;
  // Intended send time, actual start, and the time the Search call took, all
  // from the start of the run.
  absl::Duration intended;
  absl::Duration started;
  absl::Duration latency;
  QueryOutcome outcome = QueryOutcome::kOk;
};

// Offsets from the start of the run.
struct ChurnEvents {
  // Just before the checkout starts.
  absl::Duration checkout_start;
  // The view has absorbed the checkout (see RunChurn).
  absl::Duration ingest_done;
  // No compaction is queued or running any more, if a compaction ran after
  // the ingest; otherwise the same as ingest_done (an empty merge window).
  absl::Duration merge_done;
  // The clients stop issuing queries.
  absl::Duration end;
};

enum ChurnWindow : size_t { kBefore, kIngest, kMerge, kAfter, kNumWindows };

const char* ChurnWindowName(ChurnWindow window);

struct WindowStats {
  absl::Duration begin;
  absl::Duration end;
  int64_t count = 0;
  int64_t stale = 0;
  int64_t failed = 0;
  // Percentiles of QuerySample::latency, by nearest rank over the window's
  // queries; zero when it has none.
  absl::Duration p50;
  absl::Duration p99;
  absl::Duration max;
  // The p99 of started + latency - intended: the response time a client with
  // a fixed send schedule sees, including the wait behind its own slow
  // queries.
  absl::Duration p99_response;
};

// Assigns each sample to the window of its intended send time:
// [0, checkout_start), [checkout_start, ingest_done), [ingest_done,
// merge_done), [merge_done, end]. A query that starts in one window and ends
// in the next belongs to the first. Requires the events to be ordered.
std::array<WindowStats, kNumWindows> SplitWindows(
    absl::Span<const QuerySample> samples, const ChurnEvents& events);

struct ChurnOptions {
  int clients = 10;
  double rate_hz = 2.0;
  double jitter = 0.5;
  uint64_t seed = 1;
  // Queries before the checkout and after the merge.
  absl::Duration before = absl::Seconds(10);
  absl::Duration after = absl::Seconds(10);
  // The requests the clients cycle through, each client in its own seeded
  // order.
  std::vector<SearchRequest> requests;
  // Performs the checkout. Called once, on the calling thread, with no
  // client paused.
  absl::AnyInvocable<absl::Status() const> checkout;
  // A literal that exists only after the checkout, in a file the checkout
  // adds. The ingest is not done until a search finds it.
  std::string marker;
  // Ingest is done when the marker is visible and no generation has been
  // assigned or applied for this long.
  absl::Duration quiet = absl::Milliseconds(500);
  absl::Duration poll = absl::Milliseconds(20);
  absl::Duration ingest_timeout = absl::Seconds(300);
};

struct ChurnResult {
  std::vector<QuerySample> samples;
  ChurnEvents events;
  std::array<WindowStats, kNumWindows> windows;
  // The checkout call's duration, and when the marker was first visible.
  absl::Duration checkout_took;
  absl::Duration marker_visible;
  bool ingest_timed_out = false;
  // Compactions completed and their bytes, from the checkout's start to
  // ingest_done, and from ingest_done to merge_done.
  uint64_t compactions_in_ingest = 0;
  uint64_t compactions_in_merge = 0;
  uint64_t compaction_bytes_in_ingest = 0;
  uint64_t compaction_bytes_in_merge = 0;
  // The worktree's view size before and after.
  int64_t view_paths_before = 0;
  int64_t view_paths_after = 0;
};

// Runs the clients, waits `before`, calls options.checkout, waits for the
// ingest and then for the compactions, runs the clients for `after`, and
// splits the samples. The worktree must be registered, and watched if the
// checkout is to be picked up without a Notify. Returns the checkout's error
// if it fails (the clients are stopped first).
absl::StatusOr<ChurnResult> RunChurn(Workspace& workspace, WorktreeId tree,
                                     const ChurnOptions& options);

}  // namespace ferret

#endif  // FERRET_WORKSPACE_CHURN_H_

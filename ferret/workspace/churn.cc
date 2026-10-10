// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/workspace/churn.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

#include "ferret/workspace/workspace.h"

namespace ferret {

std::vector<ScheduledQuery> ClientSchedule(const ClientScheduleOptions& options,
                                           size_t num_queries) {
  CHECK_GT(options.rate_hz, 0.0);
  CHECK_GE(options.jitter, 0.0);
  CHECK_LE(options.jitter, 1.0);
  CHECK_GT(num_queries, size_t{0});
  std::seed_seq seeds{static_cast<uint32_t>(options.seed),
                      static_cast<uint32_t>(options.seed >> 32),
                      static_cast<uint32_t>(options.client)};
  std::mt19937_64 rng(seeds);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const double period = 1.0 / options.rate_hz;
  const double phase = unit(rng) * period;
  std::vector<ScheduledQuery> schedule;
  for (int64_t i = 0;; ++i) {
    // Both draws happen for every send, so a longer schedule extends a
    // shorter one.
    const double moved = (unit(rng) - 0.5) * options.jitter * period;
    const size_t query = static_cast<size_t>(rng() % num_queries);
    const double at =
        std::max(0.0, phase + static_cast<double>(i) * period + moved);
    if (absl::Seconds(at) >= options.length) break;
    schedule.push_back({absl::Seconds(at), query});
  }
  return schedule;
}

const char* ChurnWindowName(ChurnWindow window) {
  static constexpr const char* kNames[] = {"before", "ingest", "merge",
                                           "after"};
  return kNames[window];
}

namespace {

absl::Duration Rank(std::vector<absl::Duration>& sorted, int percent) {
  if (sorted.empty()) return absl::ZeroDuration();
  return sorted[std::min(sorted.size() - 1,
                         sorted.size() * static_cast<size_t>(percent) / 100)];
}

}  // namespace

std::array<WindowStats, kNumWindows> SplitWindows(
    absl::Span<const QuerySample> samples, const ChurnEvents& events) {
  const std::array<absl::Duration, kNumWindows + 1> bounds = {
      absl::ZeroDuration(), events.checkout_start, events.ingest_done,
      events.merge_done, events.end};
  std::array<WindowStats, kNumWindows> out;
  std::array<std::vector<absl::Duration>, kNumWindows> latencies;
  std::array<std::vector<absl::Duration>, kNumWindows> responses;
  for (size_t w = 0; w < kNumWindows; ++w) {
    out[w].begin = bounds[w];
    out[w].end = bounds[w + 1];
  }
  for (const QuerySample& sample : samples) {
    size_t w = kNumWindows - 1;
    for (size_t i = 0; i + 1 < kNumWindows; ++i) {
      if (sample.intended < bounds[i + 1]) {
        w = i;
        break;
      }
    }
    ++out[w].count;
    if (sample.outcome == QueryOutcome::kStale) ++out[w].stale;
    if (sample.outcome == QueryOutcome::kFailed) ++out[w].failed;
    latencies[w].push_back(sample.latency);
    responses[w].push_back(sample.started + sample.latency - sample.intended);
  }
  for (size_t w = 0; w < kNumWindows; ++w) {
    std::sort(latencies[w].begin(), latencies[w].end());
    std::sort(responses[w].begin(), responses[w].end());
    out[w].p50 = Rank(latencies[w], 50);
    out[w].p99 = Rank(latencies[w], 99);
    out[w].max =
        latencies[w].empty() ? absl::ZeroDuration() : latencies[w].back();
    out[w].p99_response = Rank(responses[w], 99);
  }
  return out;
}

absl::StatusOr<ChurnResult> RunChurn(Workspace& workspace, WorktreeId tree,
                                     const ChurnOptions& options) {
  CHECK(!options.requests.empty());
  CHECK(options.checkout);
  CHECK_GE(options.clients, 1);
  ChurnResult result;
  {
    auto status = workspace.WorktreeStatus(tree);
    if (!status.ok()) return status.status();
    result.view_paths_before = status->view_paths;
  }
  const absl::Time t0 = absl::Now();
  const auto since = [&] { return absl::Now() - t0; };
  std::atomic<bool> stop{false};
  std::vector<std::vector<QuerySample>> per_client(
      static_cast<size_t>(options.clients));
  std::vector<std::thread> threads;
  // Long enough for any run: the ingest and merge waits are bounded.
  const absl::Duration horizon =
      options.before + options.ingest_timeout + absl::Hours(2) + options.after;
  for (int client = 0; client < options.clients; ++client) {
    threads.emplace_back([&, client] {
      const std::vector<ScheduledQuery> schedule =
          ClientSchedule({.seed = options.seed,
                          .client = client,
                          .rate_hz = options.rate_hz,
                          .jitter = options.jitter,
                          .length = horizon},
                         options.requests.size());
      std::vector<QuerySample>& samples =
          per_client[static_cast<size_t>(client)];
      for (const ScheduledQuery& query : schedule) {
        while (!stop.load(std::memory_order_acquire)) {
          const absl::Duration wait = query.at - since();
          if (wait <= absl::ZeroDuration()) break;
          absl::SleepFor(std::min(wait, absl::Milliseconds(20)));
        }
        if (stop.load(std::memory_order_acquire)) return;
        QuerySample sample;
        sample.client = client;
        sample.intended = query.at;
        sample.started = since();
        auto found = workspace.Search(tree, options.requests[query.query]);
        sample.latency = since() - sample.started;
        if (!found.ok()) {
          sample.outcome = QueryOutcome::kFailed;
        } else if (found->stale) {
          sample.outcome = QueryOutcome::kStale;
        }
        samples.push_back(sample);
      }
    });
  }
  const auto finish = [&] {
    stop.store(true, std::memory_order_release);
    for (std::thread& thread : threads) thread.join();
    threads.clear();
  };

  absl::SleepFor(options.before - since());
  const WorkspaceStats at_checkout = workspace.Stats();
  result.events.checkout_start = since();
  const absl::Status checked_out = options.checkout();
  result.checkout_took = since() - result.events.checkout_start;
  if (!checked_out.ok()) {
    finish();
    return checked_out;
  }

  // Ingest: wait for the marker, then for the generations to stop moving.
  SearchRequest probe;
  probe.pattern = options.marker;
  probe.context_lines = 0;
  probe.max_files = 1;
  uint64_t last_applied = 0;
  uint64_t last_assigned = 0;
  absl::Duration last_change = since();
  // The counters as they were when the generations last moved: a compaction
  // that completes after that belongs to the merge, not the ingest.
  WorkspaceStats at_ingest = workspace.Stats();
  bool marker_seen = false;
  for (;;) {
    absl::SleepFor(options.poll);
    const absl::Duration now = since();
    if (now - result.events.checkout_start > options.ingest_timeout) {
      result.ingest_timed_out = true;
      result.events.ingest_done = now;
      break;
    }
    auto status = workspace.WorktreeStatus(tree);
    if (!status.ok()) continue;
    if (status->applied_generation != last_applied ||
        status->assigned_generation != last_assigned) {
      last_applied = status->applied_generation;
      last_assigned = status->assigned_generation;
      last_change = now;
      at_ingest = workspace.Stats();
    }
    if (!marker_seen && !options.marker.empty()) {
      auto found = workspace.Search(tree, probe);
      if (found.ok() && !found->response.files.empty()) {
        marker_seen = true;
        result.marker_visible = now - result.events.checkout_start;
      }
    }
    const bool settled =
        status->applied_generation == status->assigned_generation &&
        status->queued_paths == 0;
    if ((marker_seen || options.marker.empty()) && settled &&
        now - last_change >= options.quiet) {
      result.events.ingest_done = last_change;
      break;
    }
  }
  const absl::Duration waiting_from = since();
  workspace.WaitForCompactions();
  const absl::Duration waited = since() - waiting_from;
  const WorkspaceStats at_merge = workspace.Stats();
  // The merge window is empty unless a compaction ran after the ingest: the
  // wait for the settle time above is not merge time.
  const bool merged = at_merge.index.completed_compactions >
                          at_ingest.index.completed_compactions ||
                      waited > options.poll;
  result.events.merge_done = merged
                                 ? std::max(since(), result.events.ingest_done)
                                 : result.events.ingest_done;
  result.compactions_in_ingest = at_ingest.index.completed_compactions -
                                 at_checkout.index.completed_compactions;
  result.compaction_bytes_in_ingest =
      at_ingest.index.compaction_bytes - at_checkout.index.compaction_bytes;
  result.compactions_in_merge = at_merge.index.completed_compactions -
                                at_ingest.index.completed_compactions;
  result.compaction_bytes_in_merge =
      at_merge.index.compaction_bytes - at_ingest.index.compaction_bytes;
  {
    auto status = workspace.WorktreeStatus(tree);
    if (status.ok()) result.view_paths_after = status->view_paths;
  }

  absl::SleepFor(options.after);
  result.events.end = since();
  finish();
  for (auto& samples : per_client) {
    result.samples.insert(result.samples.end(), samples.begin(), samples.end());
  }
  std::sort(result.samples.begin(), result.samples.end(),
            [](const QuerySample& a, const QuerySample& b) {
              return a.intended != b.intended ? a.intended < b.intended
                                              : a.client < b.client;
            });
  result.windows = SplitWindows(result.samples, result.events);
  return result;
}

}  // namespace ferret

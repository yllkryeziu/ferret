// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Usage: workspace_benchmark INDEX_DIR WT1 WT2 WT3 [DIFF_WT]
//        workspace_benchmark --regq INDEX_DIR WARM_WT NEW_WT --queries FILE
//        [--qos utility|initiated|interactive] [--clients N] [--rate HZ]
//        [--before_s S] [--after_s S] [--seed N] [--csv FILE]
//        workspace_benchmark --g3 INDEX_DIR LABEL=WT ...
//        workspace_benchmark --freshness INDEX_DIR WT [TRIALS]
//        workspace_benchmark --churn INDEX_DIR WT --queries FILE
//        --checkout REV --marker TOKEN [--clients N] [--rate HZ]
//        [--before_s S] [--after_s S] [--seed N] [--csv FILE] [--quiet_ms N]
//        workspace_benchmark --g2g3 COUNT INDEX_DIR WT1 ... WTN CHURN_WT
//        workspace_benchmark --watch INDEX_DIR SCRATCH_WT
//        workspace_benchmark --profile INDEX_DIR SCRATCH_WT [CALLS
//        [ROW_FILTER]] workspace_benchmark --gc INDEX_DIR OLD_WT REMAINING_WT
//        workspace_benchmark --restore INDEX_DIR WT1 [WT2]
//
// G3: registers WT1 into an empty index (k=1) and then waits for the
// compactions its segments trigger, reported separately; then WT2 and WT3
// (k=2, k=3, the same commit), each followed by a bare ScanWorktree of the
// same tree; then DIFF_WT, a clean worktree at another commit whose changed
// blobs are hinted but not indexed (O(diff)); then WT1 again after reopening.
// G2: 1,000 single-file edits and 200 batches of 16 to WT1, timed from Notify
// until the generation is applied and checked with a kNone search: idle,
// while WT3 is registered and unregistered in a loop, and while a
// registration of WT3 is held inside the bulk lane (its line reports the
// hold separately and only bulk-lane counters). Every edited file is
// restored. The --g2g3 mode registers COUNT (1..10) worktrees and uses an
// additional unregistered scratch worktree for G2 churn. Watch mode also
// times write-to-kNone-searchable and
// write-to-kSync-returning over 1,000 edits. The caller creates and removes
// the scratch worktrees; never run
// this in a corpus checkout.
//
// --g3 registers the labelled worktrees in order into one index and prints,
// per registration, the wall time split into scan, read and ingest, the CPU
// time of the process, the bytes read, the bytes the index directory grew by,
// and the segments, before and after the compactions it triggers.
// --freshness times 1 edit at a time from the edit to a kNone search that
// sees it, through Notify and through the watcher, and prints one FRESH line
// per trial with its stages. --churn is described in ferret/workspace/churn.h;
// it prints CHURN lines, and with --csv writes every query.

#include "ferret/workspace/workspace.h"

#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "bench/ferret/query_set.h"

#include "ferret/base/env.h"
#include "ferret/base/subprocess.h"
#include "ferret/git/worktree_files.h"
#include "ferret/git/worktree_scanner.h"
#include "ferret/index/index.h"
#include "ferret/search/search_types.h"
#include "ferret/text/eligibility.h"
#include "ferret/watch/watch_trace.h"
#include "ferret/watch/watcher.h"
#include "ferret/workspace/churn.h"

namespace ferret {
namespace {

constexpr absl::string_view kGateFile = "__workspace_benchmark_gate.cc";

struct Sample {
  std::string path;
  std::string original;
};

int64_t Millis(absl::Duration duration) {
  return absl::ToInt64Milliseconds(duration);
}

uint64_t Delta(uint64_t after, uint64_t before) { return after - before; }

// Prints the bulk-lane counters of one registration. `held` is time a test
// hook paused the registration; it is excluded from `wall` but included in
// read_ms. Memtable flushes and compactions serve both lanes, so flush_bytes
// and the compaction counters also count work done meanwhile for the other.
void PrintRegistration(absl::string_view label, absl::Duration wall,
                       absl::Duration held, const WorkspaceStats& before,
                       const WorkspaceStats& after) {
  std::string skipped;
  for (int reason = 1; reason < kNumSkipReasons; ++reason) {
    const size_t i = static_cast<size_t>(reason);
    if (after.skipped[i] == before.skipped[i]) continue;
    absl::StrAppend(&skipped, skipped.empty() ? "" : ",",
                    SkipReasonName(static_cast<SkipReason>(reason)), ":",
                    after.skipped[i] - before.skipped[i]);
  }
  absl::PrintF(
      "%s: wall_ms=%d held_ms=%d scan_ms=%d classify_ms=%d read_ms=%d "
      "segment_ms=%d memtable_ms=%d publish_ms=%d bulk_reads=%d rereads=%d "
      "hint_reads=%d hints_reused=%d signatures_reused=%d changed=%d "
      "bulk_bytes=%d segment_bytes=%d flush_bytes=%d segments=%d "
      "compactions=%d compaction_bytes=%d skipped=%s "
      "wb_wait_ms=%d wb_add_ms=%d wb_finish_ms=%d wb_reread_thread_ms=%d "
      "wb_symwait_thread_ms=%d wb_symparse_thread_ms=%d\n",
      label, Millis(wall - held), Millis(held),
      Millis(after.scan_time - before.scan_time),
      Millis(after.classify_time - before.classify_time),
      Millis(after.read_time - before.read_time),
      Millis(after.segment_write_time - before.segment_write_time),
      Millis(after.memtable_add_time - before.memtable_add_time),
      Millis(after.publish_time - before.publish_time),
      Delta(after.bulk_files_read, before.bulk_files_read),
      Delta(after.bulk_files_reread, before.bulk_files_reread),
      Delta(after.hint_reads, before.hint_reads),
      Delta(after.hints_reused, before.hints_reused),
      Delta(after.signatures_reused, before.signatures_reused),
      Delta(after.changed_before_write, before.changed_before_write),
      Delta(after.bulk_bytes_ingested, before.bulk_bytes_ingested),
      Delta(after.index.bulk_segment_bytes, before.index.bulk_segment_bytes),
      Delta(after.index.flush_bytes, before.index.flush_bytes),
      Delta(after.index.bulk_segments, before.index.bulk_segments),
      Delta(after.index.completed_compactions,
            before.index.completed_compactions),
      Delta(after.index.compaction_bytes, before.index.compaction_bytes),
      skipped.empty() ? "none" : skipped,
      Millis(after.bulk_wait_time - before.bulk_wait_time),
      Millis(after.bulk_add_time - before.bulk_add_time),
      Millis(after.bulk_finish_time - before.bulk_finish_time),
      Millis(after.bulk_reread_thread_time - before.bulk_reread_thread_time),
      Millis(after.bulk_symbol_wait_thread_time -
             before.bulk_symbol_wait_thread_time),
      Millis(after.bulk_symbol_parse_thread_time -
             before.bulk_symbol_parse_thread_time));
  std::fflush(stdout);
}

// Registers `root`, prints its counters, then waits for the compactions it
// left running and prints how long that took and what they wrote.
WorktreeId RegisterTimed(Workspace& workspace, absl::string_view label,
                         absl::string_view root) {
  const WorkspaceStats before = workspace.Stats();
  const absl::Time start = absl::Now();
  absl::StatusOr<WorktreeId> tree = workspace.Register(root);
  const absl::Duration wall = absl::Now() - start;
  CHECK_OK(tree.status());
  const WorkspaceStats registered = workspace.Stats();
  PrintRegistration(label, wall, absl::ZeroDuration(), before, registered);
  workspace.WaitForCompactions();
  const absl::Duration total = absl::Now() - start;
  const WorkspaceStats quiet = workspace.Stats();
  absl::PrintF(
      "%s then compaction: wait_ms=%d total_ms=%d compactions=%d "
      "compaction_bytes=%d segment_count=%d\n",
      label, Millis(total - wall), Millis(total),
      Delta(quiet.index.completed_compactions,
            registered.index.completed_compactions),
      Delta(quiet.index.compaction_bytes, registered.index.compaction_bytes),
      quiet.index.segment_count);
  std::fflush(stdout);
  return *tree;
}

void ScanTimed(absl::string_view label, absl::string_view root) {
  const absl::Time start = absl::Now();
  absl::StatusOr<WorktreeScan> scan = ScanWorktree(root);
  const absl::Duration wall = absl::Now() - start;
  CHECK_OK(scan.status());
  const ScanReport& report = scan->report;
  absl::PrintF(
      "%s ScanWorktree: wall_ms=%d files=%d hinted=%d dirty=%d untracked=%d "
      "unmerged=%d assume_unchanged=%d conversion=%d case_collisions=%d "
      "git_ms=%d attributes_ms=%d lstat_ms=%d\n",
      label, Millis(wall), report.files, report.hinted, report.dirty,
      report.untracked, report.unmerged, report.assume_unchanged,
      report.conversion, report.case_collisions, Millis(report.git_commands),
      Millis(report.attributes), Millis(report.lstat));
  std::fflush(stdout);
}

void WriteFileOrDie(const std::string& path, absl::string_view bytes) {
  auto file = Env::Default()->NewWritableFile(path);
  CHECK_OK(file.status());
  CHECK_OK((*file)->Append(bytes));
  CHECK_OK((*file)->Close());
}

// Up to `count` small eligible tracked files, in a seeded random order.
std::vector<Sample> Samples(absl::string_view root, size_t count) {
  absl::StatusOr<std::vector<std::string>> paths = ListWorktreeFiles(root);
  CHECK_OK(paths.status());
  std::shuffle(paths->begin(), paths->end(), std::mt19937_64(4173));
  std::vector<Sample> samples;
  for (const std::string& path : *paths) {
    if (samples.size() == count) break;
    auto bytes = Env::Default()->ReadFile(absl::StrCat(root, "/", path));
    if (bytes.ok() && !bytes->empty() && bytes->size() < size_t{16} * 1024 &&
        bytes->find('\0') == std::string::npos) {
      samples.push_back({path, std::move(*bytes)});
    }
  }
  CHECK(!samples.empty());
  return samples;
}

void AwaitOrDie(Workspace& workspace, WorktreeId tree, uint64_t generation) {
  absl::StatusOr<uint64_t> applied =
      workspace.AwaitGeneration(tree, generation, absl::Seconds(60));
  CHECK_OK(applied.status());
  CHECK_GE(*applied, generation);
}

void PrintLatency(absl::string_view label, std::vector<int64_t> micros) {
  CHECK(!micros.empty());
  std::sort(micros.begin(), micros.end());
  absl::PrintF("%s: n=%d p50_us=%d p90_us=%d p99_us=%d max_us=%d\n", label,
               micros.size(), micros[micros.size() / 2],
               micros[micros.size() * 9 / 10], micros[micros.size() * 99 / 100],
               micros.back());
  std::fflush(stdout);
}

// Appends a unique token line to random samples, notifies, and times until
// the generation is applied; then checks with a kNone search and restores.
void RunEdits(Workspace& workspace, WorktreeId tree, absl::string_view root,
              const std::vector<Sample>& samples, int batch_size, int batches,
              absl::string_view label) {
  std::mt19937_64 rng(4173 + static_cast<uint64_t>(batch_size) + label.size());
  std::vector<int64_t> times;
  times.reserve(static_cast<size_t>(batches));
  const std::string prefix = absl::StrCat("ferretbenchtoken", rng() % 100000);
  for (int batch = 0; batch < batches; ++batch) {
    std::vector<size_t> touched;
    while (touched.size() < static_cast<size_t>(batch_size) &&
           touched.size() < samples.size()) {
      const size_t which = static_cast<size_t>(rng() % samples.size());
      if (std::find(touched.begin(), touched.end(), which) == touched.end()) {
        touched.push_back(which);
      }
    }
    const std::string token = absl::StrCat(prefix, "b", batch, "x");
    std::vector<std::string> paths;
    for (const size_t which : touched) {
      const Sample& sample = samples[which];
      WriteFileOrDie(absl::StrCat(root, "/", sample.path),
                     absl::StrCat(sample.original, "\n", token, "\n"));
      paths.push_back(sample.path);
    }
    const absl::Time start = absl::Now();
    absl::StatusOr<uint64_t> generation = workspace.Notify(tree, paths);
    CHECK_OK(generation.status());
    AwaitOrDie(workspace, tree, *generation);
    times.push_back(absl::ToInt64Microseconds(absl::Now() - start));
    SearchRequest request;
    request.pattern = token;
    request.max_files = batch_size + 1;
    absl::StatusOr<WorkspaceSearchResult> found =
        workspace.Search(tree, request);
    CHECK_OK(found.status());
    CHECK_EQ(found->response.files.size(), paths.size()) << token;
    for (const size_t which : touched) {
      WriteFileOrDie(absl::StrCat(root, "/", samples[which].path),
                     samples[which].original);
    }
    absl::StatusOr<uint64_t> restored = workspace.Notify(tree, paths);
    CHECK_OK(restored.status());
    AwaitOrDie(workspace, tree, *restored);
  }
  PrintLatency(label, std::move(times));
}

void WatchEdits(Workspace& workspace, WorktreeId tree, absl::string_view root,
                const std::vector<Sample>& samples) {
  std::vector<int64_t> none;
  std::vector<int64_t> sync;
  none.reserve(1000);
  sync.reserve(1000);
  for (int i = 0; i < 1000; ++i) {
    const Sample& sample = samples[static_cast<size_t>(i) % samples.size()];
    const std::string path = absl::StrCat(root, "/", sample.path);
    for (int mode = 0; mode < 2; ++mode) {
      const std::string token = absl::StrCat("ferretwatchbench", i, "m", mode);
      WriteFileOrDie(path, absl::StrCat(sample.original, "\n", token, "\n"));
      const absl::Time written = absl::Now();
      SearchRequest request;
      request.pattern = token;
      request.context_lines = 0;
      if (mode == 0) {
        for (;;) {
          auto result = workspace.Search(tree, request);
          CHECK_OK(result.status());
          if (!result->response.files.empty()) break;
          CHECK_LT(absl::Now() - written, absl::Seconds(10));
          absl::SleepFor(absl::Microseconds(100));
        }
        none.push_back(absl::ToInt64Microseconds(absl::Now() - written));
      } else {
        auto result = workspace.Search(tree, request,
                                       {.mode = SearchConsistencyMode::kSync,
                                        .timeout = absl::Seconds(10)});
        CHECK_OK(result.status());
        CHECK(!result->stale);
        CHECK_EQ(result->response.files.size(), size_t{1});
        sync.push_back(absl::ToInt64Microseconds(absl::Now() - written));
      }
      WriteFileOrDie(path, sample.original);
      auto restored = workspace.Search(
          tree, request,
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(10)});
      CHECK_OK(restored.status());
      CHECK(!restored->stale);
      CHECK(restored->response.files.empty());
    }
  }
  PrintLatency("G2 watcher write to kNone", std::move(none));
  PrintLatency("G2 watcher write to kSync", std::move(sync));
}

// Stage profile of the kSync barrier and of write -> searchable, from the
// marks in watch_trace.h. Each row makes `calls` measurements; `gap` is the
// quiet time before a measurement (0: straight after the previous Sync).
struct ProfileRow {
  std::vector<int64_t> total;
  std::vector<std::array<int64_t, watch_trace::kStageCount + 1>> marks;
  std::vector<int64_t> callbacks;
};

constexpr int kEndStage = watch_trace::kStageCount;

const char* StageName(int stage) {
  static const char* const names[] = {"sync_start",
                                      "cookie_created",
                                      "event_seen",
                                      "cookie_seen",
                                      "delivering",
                                      "cookie_done",
                                      "sync_woke",
                                      "callback",
                                      "queued",
                                      "dequeued",
                                      "pinned",
                                      "index_refreshed",
                                      "ignore_ready",
                                      "batch_setup",
                                      "file_read",
                                      "read",
                                      "added",
                                      "published",
                                      "completed",
                                      "unpinned",
                                      "sync_returned",
                                      "excludes_done",
                                      "generation_done",
                                      "snapshot_done",
                                      "end"};
  static_assert(std::size(names) == watch_trace::kStageCount + 1);
  return names[stage];
}

int64_t Percentile(std::vector<int64_t> v, int pct) {
  if (v.empty()) return -1;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, v.size() * static_cast<size_t>(pct) / 100)];
}

void PrintProfile(absl::string_view label, const ProfileRow& row) {
  absl::PrintF(
      "PROFILE %s: n=%d total_us p50=%d p90=%d p99=%d max=%d "
      "callbacks p50=%d max=%d\n",
      label, row.total.size(), Percentile(row.total, 50) / 1000,
      Percentile(row.total, 90) / 1000, Percentile(row.total, 99) / 1000,
      Percentile(row.total, 100) / 1000, Percentile(row.callbacks, 50),
      Percentile(row.callbacks, 100));
  // FSEvents delivery is not unimodal: histogram its two stages in 1 ms bins.
  for (const int stage : {static_cast<int>(watch_trace::kEventSeen),
                          static_cast<int>(watch_trace::kCookieSeen)}) {
    std::array<int, 21> bins = {};
    int seen = 0;
    for (const auto& m : row.marks) {
      const int64_t at = m[static_cast<size_t>(stage)];
      if (at <= 0) continue;
      ++seen;
      ++bins[static_cast<size_t>(std::min<int64_t>(20, at / 1'000'000))];
    }
    if (seen == 0) continue;
    std::string line;
    for (int bin = 0; bin <= 20; ++bin) {
      absl::StrAppend(&line, " ", bins[static_cast<size_t>(bin)]);
    }
    absl::PrintF("  hist %-12s (1 ms bins, last = 20+):%s\n", StageName(stage),
                 line);
  }
  // Cumulative offsets, ordered by median; a stage seen in under half of the
  // measurements is left out.
  std::vector<std::pair<int64_t, int>> order;
  for (int stage = 0; stage <= kEndStage; ++stage) {
    std::vector<int64_t> v;
    for (const auto& m : row.marks) {
      if (m[static_cast<size_t>(stage)] > 0) {
        v.push_back(m[static_cast<size_t>(stage)]);
      }
    }
    if (v.size() * 2 < row.marks.size()) continue;
    order.emplace_back(Percentile(v, 50), stage);
  }
  std::sort(order.begin(), order.end());
  int previous = -1;
  for (const auto& [median, stage] : order) {
    std::vector<int64_t> at;
    std::vector<int64_t> step;
    for (const auto& m : row.marks) {
      const int64_t now = m[static_cast<size_t>(stage)];
      if (now <= 0) continue;
      at.push_back(now);
      if (previous >= 0 && m[static_cast<size_t>(previous)] > 0) {
        step.push_back(
            std::max<int64_t>(0, now - m[static_cast<size_t>(previous)]));
      }
    }
    absl::PrintF(
        "  %-16s present=%4d at_us p50=%6d p90=%6d p99=%6d | "
        "step_us p50=%6d p90=%6d p99=%6d\n",
        StageName(stage), at.size(), Percentile(at, 50) / 1000,
        Percentile(at, 90) / 1000, Percentile(at, 99) / 1000,
        Percentile(step, 50) / 1000, Percentile(step, 90) / 1000,
        Percentile(step, 99) / 1000);
    previous = stage;
  }
  std::fflush(stdout);
}

// Starts a measurement; offsets are taken from the t0 the caller supplies.
void TraceBegin() {
  watch_trace::Clear();
  watch_trace::Get().enabled.store(true);
}

void TraceEnd(ProfileRow& row, int64_t t0, int64_t end) {
  watch_trace::Get().enabled.store(false);
  std::array<int64_t, watch_trace::kStageCount + 1> marks = {};
  for (int stage = 0; stage < watch_trace::kStageCount; ++stage) {
    const int64_t at = watch_trace::Get().nanos[stage].load();
    marks[static_cast<size_t>(stage)] =
        at == 0 ? 0 : std::max<int64_t>(1, at - t0);
  }
  marks[static_cast<size_t>(kEndStage)] = end - t0;
  row.total.push_back(end - t0);
  row.marks.push_back(marks);
  row.callbacks.push_back(watch_trace::Get().callbacks.load());
}

// A stale kSync result is reported with the state that explains it, and the
// run goes on: stale means the cookie timed out, the watcher failed, or a
// generation stayed unapplied until the deadline.
int g_stale_results = 0;

void NoteStale(Workspace& workspace, WorktreeId tree, const char* where,
               const WorkspaceSearchResult& result, int64_t started) {
  if (!result.stale) return;
  ++g_stale_results;
  const WorkspaceStats stats = workspace.Stats();
  auto status = workspace.WorktreeStatus(tree);
  absl::PrintF(
      "STALE %s: after_ms=%d result_generation=%d applied=%d assigned=%d "
      "queued_paths=%d watcher_running=%d watcher_failed=%d sync_timeouts=%d "
      "watcher_failures=%d last_watcher_error=%s\n",
      where, (absl::GetCurrentTimeNanos() - started) / 1'000'000,
      result.generation, status.ok() ? status->applied_generation : 0,
      status.ok() ? status->assigned_generation : 0,
      status.ok() ? status->queued_paths : 0,
      status.ok() && status->watcher_running,
      status.ok() && status->watcher_failed, stats.sync_timeouts,
      stats.watcher_failures, stats.last_watcher_error.ToString());
  std::fflush(stdout);
}

enum class ProfileKind {
  kNoneNoMatch,
  kSyncNoMatch,
  kWriteNone,
  kWriteSync,
  kWriteSyncNoMatch
};

ProfileRow RunProfile(Workspace& workspace, WorktreeId tree,
                      absl::string_view root,
                      const std::vector<Sample>& samples, ProfileKind kind,
                      int calls, absl::Duration gap, bool preceded_by_sync) {
  ProfileRow row;
  SearchRequest none_match;
  none_match.pattern = "ferretprofilenomatchzq";
  none_match.context_lines = 0;
  const SearchConsistency sync{.mode = SearchConsistencyMode::kSync,
                               .timeout = absl::Seconds(10)};
  for (int i = 0; i < calls; ++i) {
    if (gap > absl::ZeroDuration()) absl::SleepFor(gap);
    if (kind == ProfileKind::kNoneNoMatch) {
      TraceBegin();
      const int64_t t0 = absl::GetCurrentTimeNanos();
      CHECK_OK(workspace.Search(tree, none_match).status());
      TraceEnd(row, t0, absl::GetCurrentTimeNanos());
    } else if (kind == ProfileKind::kSyncNoMatch) {
      if (preceded_by_sync) {
        CHECK_OK(workspace.Search(tree, none_match, sync).status());
      }
      TraceBegin();
      const int64_t t0 = absl::GetCurrentTimeNanos();
      auto result = workspace.Search(tree, none_match, sync);
      const int64_t end = absl::GetCurrentTimeNanos();
      CHECK_OK(result.status());
      NoteStale(workspace, tree, "row", *result, t0);
      TraceEnd(row, t0, end);
    } else {
      const Sample& sample = samples[static_cast<size_t>(i) % samples.size()];
      const std::string path = absl::StrCat(root, "/", sample.path);
      const std::string token =
          absl::StrCat("ferretprofiletoken", i, "k", static_cast<int>(kind));
      SearchRequest request;
      request.pattern = token;
      request.context_lines = 0;
      TraceBegin();
      WriteFileOrDie(path, absl::StrCat(sample.original, "\n", token, "\n"));
      const int64_t t0 = absl::GetCurrentTimeNanos();
      int64_t end;
      if (kind == ProfileKind::kWriteNone) {
        for (;;) {
          auto result = workspace.Search(tree, request);
          CHECK_OK(result.status());
          end = absl::GetCurrentTimeNanos();
          if (!result->response.files.empty()) break;
          CHECK_LT(end - t0, int64_t{10'000'000'000});
          absl::SleepFor(absl::Microseconds(100));
        }
      } else {
        auto result = workspace.Search(
            tree, kind == ProfileKind::kWriteSync ? request : none_match, sync);
        end = absl::GetCurrentTimeNanos();
        CHECK_OK(result.status());
        NoteStale(workspace, tree, "row", *result, t0);
        if (kind == ProfileKind::kWriteSync) {
          CHECK_EQ(result->response.files.size(), size_t{1}) << token;
        }
      }
      TraceEnd(row, t0, end);
      WriteFileOrDie(path, sample.original);
      const int64_t restore_started = absl::GetCurrentTimeNanos();
      auto restored = workspace.Search(tree, request, sync);
      CHECK_OK(restored.status());
      NoteStale(workspace, tree, "restore", *restored, restore_started);
      CHECK(restored->response.files.empty()) << token << " " << path;
    }
  }
  return row;
}

void ProfileAll(Workspace& workspace, WorktreeId tree, absl::string_view root,
                const std::vector<Sample>& samples, int calls,
                absl::string_view filter) {
  using K = ProfileKind;
  const auto run = [&](absl::string_view label, K kind, absl::Duration gap,
                       bool preceded) {
    if (!filter.empty() && !absl::StrContains(label, filter)) return;
    PrintProfile(label, RunProfile(workspace, tree, root, samples, kind, calls,
                                   gap, preceded));
  };
  run("kNone no-match", K::kNoneNoMatch, absl::ZeroDuration(), false);
  run("kSync no-match, idle (100ms gap)", K::kSyncNoMatch,
      absl::Milliseconds(100), false);
  run("kSync no-match, right after another kSync", K::kSyncNoMatch,
      absl::ZeroDuration(), true);
  run("write -> kSync no-match, idle", K::kWriteSyncNoMatch,
      absl::Milliseconds(100), false);
  run("write -> kSync searchable, idle", K::kWriteSync, absl::Milliseconds(100),
      false);
  run("write -> kSync searchable, right after a kSync", K::kWriteSync,
      absl::ZeroDuration(), false);
  run("write -> kNone searchable, idle", K::kWriteNone, absl::Milliseconds(100),
      false);
  run("write -> kNone searchable, right after a kSync", K::kWriteNone,
      absl::ZeroDuration(), false);
  absl::PrintF("stale kSync results: %d\n", g_stale_results);
}

// Counts native callback batches during 1,000 distinct writes to a scratch
// directory. A sync cookie drains the stream before the count is printed.
void WatchBurst(absl::string_view root) {
  const std::string cookies = absl::StrCat(root, "/.watch-burst-cookies");
  const std::string dir = absl::StrCat(root, "/.watch-burst-files");
  CHECK_EQ(mkdir(cookies.c_str(), 0700), 0);
  CHECK_EQ(mkdir(dir.c_str(), 0700), 0);
  WatcherOptions options;
  options.cookie_dir = ".watch-burst-cookies";
  absl::StatusOr<std::unique_ptr<Watcher>> watcher =
      NewWatcher(std::string(root), options);
  CHECK_OK(watcher.status());
  std::atomic<int> batches{0};
  CHECK_OK((*watcher)->Start([&batches](const std::vector<WatchEvent>& events) {
    if (!events.empty()) ++batches;
  }));
  for (int i = 0; i < 1000; ++i) {
    WriteFileOrDie(absl::StrCat(dir, "/", i, ".cc"), "bursttoken\n");
  }
  CHECK_OK((*watcher)->Sync(absl::Seconds(10)));
  absl::PrintF("G2 watcher burst: writes=1000 batches=%d\n", batches.load());
  (*watcher)->Stop();
  for (int i = 0; i < 1000; ++i) {
    CHECK_EQ(unlink(absl::StrCat(dir, "/", i, ".cc").c_str()), 0);
  }
  CHECK_EQ(rmdir(dir.c_str()), 0);
  CHECK_EQ(rmdir(cookies.c_str()), 0);
}

// Compares a two-view index after collecting the removed view with a fresh
// index built from the surviving worktree alone.
void RunGcComparison(absl::string_view index_dir, absl::string_view old_root,
                     absl::string_view remaining_root) {
  WorkspaceOptions options;
  options.index.gc_grace_period = absl::Milliseconds(50);
  options.index.compaction.rewrite_dead_percent = 1;
  auto opened = Workspace::Open(*Env::Default(), index_dir, options);
  CHECK_OK(opened.status());
  std::unique_ptr<Workspace> workspace = std::move(*opened);
  const WorktreeId old = RegisterTimed(*workspace, "gc old", old_root);
  const WorktreeId remaining =
      RegisterTimed(*workspace, "gc remaining", remaining_root);
  const auto bytes = [&](WorktreeId id) {
    auto status = workspace->WorktreeStatus(id);
    CHECK_OK(status.status());
    return status->index_bytes;
  };
  const int64_t combined = bytes(remaining);
  CHECK_OK(workspace->Unregister(old));
  absl::SleepFor(absl::Milliseconds(150));
  workspace->WaitForCompactions();
  const int64_t collected = bytes(remaining);
  const IndexStats stats = workspace->Stats().index;
  absl::PrintF(
      "gc bytes: combined=%d collected=%d reclaimed=%d "
      "live_docs=%d dead_docs=%d\n",
      combined, collected, stats.reclaimed_bytes, stats.live_docs,
      stats.dead_docs);
  workspace.reset();
  opened = Workspace::Open(*Env::Default(), absl::StrCat(index_dir, "-fresh"),
                           options);
  CHECK_OK(opened.status());
  workspace = std::move(*opened);
  const WorktreeId fresh =
      RegisterTimed(*workspace, "gc fresh", remaining_root);
  absl::PrintF("gc fresh bytes: %d\n", bytes(fresh));
}

void RunAllEdits(Workspace& workspace, WorktreeId tree, absl::string_view root,
                 const std::vector<Sample>& samples, absl::string_view when) {
  RunEdits(workspace, tree, root, samples, 1, 1000,
           absl::StrCat("G2 single ", when));
  RunEdits(workspace, tree, root, samples, 16, 200,
           absl::StrCat("G2 batch16 ", when));
}

// CPU time (user + system) of this process, in milliseconds.
double CpuMillis() {
  rusage usage;
  CHECK_EQ(getrusage(RUSAGE_SELF, &usage), 0);
  const auto ms = [](const timeval& t) {
    return static_cast<double>(t.tv_sec) * 1e3 +
           static_cast<double>(t.tv_usec) / 1e3;
  };
  return ms(usage.ru_utime) + ms(usage.ru_stime);
}

// Bytes of every regular file below `dir`.
int64_t DirectoryBytes(const std::string& dir) {
  int64_t total = 0;
  std::error_code error;
  for (std::filesystem::recursive_directory_iterator it(dir, error), end;
       !error && it != end; it.increment(error)) {
    std::error_code file_error;
    if (it->is_regular_file(file_error)) {
      const auto size = it->file_size(file_error);
      if (!file_error) total += static_cast<int64_t>(size);
    }
  }
  return total;
}

// One G3 row: registers `root` into the workspace and prints what it cost.
// scan is the git or directory scan; read is the parallel stat, read and
// hash; ingest is everything after that (classification, segment and
// memtable writes, view publish). cpu_ms is the process's CPU time during the
// registration; the flush of its memtable and the compactions it triggers are
// counted separately (compaction_cpu_ms covers both). segment_bytes_written is
// what the registration wrote (direct segments and flushes) before any
// compaction; index_bytes_added is the net change of the published index
// after the compactions, which can be negative.
// published_bytes is the segment bytes of the published index, which
// compactions shrink as well as grow; unlike the directory it does not count
// segments awaiting deletion. `*published` carries the previous row's value
// in and this row's out.
WorktreeId RegisterG3(Workspace& workspace, const std::string& index_dir,
                      absl::string_view label, absl::string_view root,
                      int64_t* published) {
  const int64_t bytes_before = DirectoryBytes(index_dir);
  const WorkspaceStats before = workspace.Stats();
  const double cpu_before = CpuMillis();
  const absl::Time start = absl::Now();
  absl::StatusOr<WorktreeId> tree = workspace.Register(root);
  const absl::Duration wall = absl::Now() - start;
  const double cpu_registered = CpuMillis();
  CHECK_OK(tree.status());
  const WorkspaceStats registered = workspace.Stats();
  // Everything the registration ingested must be on disk to be counted.
  const absl::Time flush_start = absl::Now();
  CHECK_OK(workspace.FlushIndex());
  const absl::Duration flush = absl::Now() - flush_start;
  const int64_t bytes_flushed = DirectoryBytes(index_dir);
  const IndexStats flushed = workspace.Stats().index;
  // Segment bytes this registration wrote, before any compaction rewrote
  // them: direct bulk segments plus memtable flushes.
  const int64_t written =
      Delta(flushed.bulk_segment_bytes, before.index.bulk_segment_bytes) +
      Delta(flushed.flush_bytes, before.index.flush_bytes);
  const absl::Time wait_start = absl::Now();
  workspace.WaitForCompactions();
  const absl::Duration wait = absl::Now() - wait_start;
  const double cpu_compacted = CpuMillis();
  const WorkspaceStats quiet = workspace.Stats();
  const int64_t bytes_after = DirectoryBytes(index_dir);
  const auto d = [&](auto field) {
    return Delta(field(registered), field(before));
  };
  const auto d2 = [&](absl::Duration WorkspaceStats::* field) {
    return registered.*field - before.*field;
  };
  const int64_t ingest_ms =
      Millis(registered.classify_time - before.classify_time) +
      Millis(registered.segment_write_time - before.segment_write_time) +
      Millis(registered.memtable_add_time - before.memtable_add_time) +
      Millis(registered.publish_time - before.publish_time);
  auto status = workspace.WorktreeStatus(*tree);
  CHECK_OK(status.status());
  const int64_t published_added = status->index_bytes - *published;
  *published = status->index_bytes;
  absl::PrintF(
      "G3 %s: wall_ms=%d scan_ms=%d read_ms=%d ingest_ms=%d cpu_ms=%d "
      "files_read=%d bytes_read=%d bytes_ingested=%d bytes_reread=%d "
      "hints_reused=%d signatures_reused=%d published_bytes=%d "
      "segment_bytes_written=%d index_bytes_added=%d "
      "index_dir_bytes_added=%d "
      "index_dir_bytes_added_before_compaction=%d flush_ms=%d "
      "compaction_wait_ms=%d compaction_cpu_ms=%d compaction_bytes=%d "
      "segments=%d segments_before_flush=%d view_paths=%d "
      "wb_wait_ms=%d wb_add_ms=%d wb_finish_ms=%d wb_reread_thread_ms=%d "
      "wb_symwait_thread_ms=%d wb_symparse_thread_ms=%d "
      "classify_ms=%d segment_ms=%d memtable_ms=%d publish_ms=%d\n",
      label, Millis(wall), Millis(registered.scan_time - before.scan_time),
      Millis(registered.read_time - before.read_time), ingest_ms,
      static_cast<int64_t>(cpu_registered - cpu_before),
      d([](const WorkspaceStats& s) { return s.bulk_files_read; }),
      d([](const WorkspaceStats& s) {
        return s.bulk_bytes_read + s.bulk_bytes_reread;
      }),
      d([](const WorkspaceStats& s) { return s.bulk_bytes_ingested; }),
      d([](const WorkspaceStats& s) { return s.bulk_bytes_reread; }),
      d([](const WorkspaceStats& s) { return s.hints_reused; }),
      d([](const WorkspaceStats& s) { return s.signatures_reused; }),
      status->index_bytes, written, published_added, bytes_after - bytes_before,
      bytes_flushed - bytes_before, Millis(flush), Millis(wait),
      static_cast<int64_t>(cpu_compacted - cpu_registered),
      Delta(quiet.index.compaction_bytes, registered.index.compaction_bytes),
      quiet.index.segment_count, registered.index.segment_count,
      status->view_paths, Millis(d2(&WorkspaceStats::bulk_wait_time)),
      Millis(d2(&WorkspaceStats::bulk_add_time)),
      Millis(d2(&WorkspaceStats::bulk_finish_time)),
      Millis(d2(&WorkspaceStats::bulk_reread_thread_time)),
      Millis(d2(&WorkspaceStats::bulk_symbol_wait_thread_time)),
      Millis(d2(&WorkspaceStats::bulk_symbol_parse_thread_time)),
      Millis(d2(&WorkspaceStats::classify_time)),
      Millis(d2(&WorkspaceStats::segment_write_time)),
      Millis(d2(&WorkspaceStats::memtable_add_time)),
      Millis(d2(&WorkspaceStats::publish_time)));
  std::fflush(stdout);
  return *tree;
}

// Freshness trials, one edit at a time: the edit is in the worktree, the
// clock starts, and it stops when a kNone search sees the new token. Notify
// trials call Notify and wait for the generation; watcher trials wait for
// the watcher to deliver the event. The edit is restored, untimed, after each
// trial.
void RunFreshness(Workspace& workspace, WorktreeId tree, absl::string_view root,
                  const std::vector<Sample>& samples, int trials) {
  const auto restore = [&](const Sample& sample, const std::string& path,
                           bool watched) {
    WriteFileOrDie(path, sample.original);
    if (!watched) {
      auto generation = workspace.Notify(tree, {sample.path});
      CHECK_OK(generation.status());
      AwaitOrDie(workspace, tree, *generation);
    } else {
      SearchRequest request;
      request.pattern = "ferretfreshnessrestore";
      auto restored = workspace.Search(
          tree, request,
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(10)});
      CHECK_OK(restored.status());
    }
  };
  for (int trial = 0; trial < trials; ++trial) {
    const Sample& sample = samples[static_cast<size_t>(trial) % samples.size()];
    const std::string path = absl::StrCat(root, "/", sample.path);
    const std::string token = absl::StrCat("ferretfreshnessn", trial, "x");
    SearchRequest request;
    request.pattern = token;
    request.context_lines = 0;
    WriteFileOrDie(path, absl::StrCat(sample.original, "\n", token, "\n"));
    const int64_t t0 = absl::GetCurrentTimeNanos();
    absl::StatusOr<uint64_t> generation = workspace.Notify(tree, {sample.path});
    const int64_t notified = absl::GetCurrentTimeNanos();
    CHECK_OK(generation.status());
    AwaitOrDie(workspace, tree, *generation);
    const int64_t applied = absl::GetCurrentTimeNanos();
    auto found = workspace.Search(tree, request);
    const int64_t searched = absl::GetCurrentTimeNanos();
    CHECK_OK(found.status());
    CHECK_EQ(found->response.files.size(), size_t{1}) << token;
    absl::PrintF(
        "FRESH ferret_notify trial=%d total_us=%d notify_us=%d apply_us=%d "
        "search_us=%d\n",
        trial, (searched - t0) / 1000, (notified - t0) / 1000,
        (applied - notified) / 1000, (searched - applied) / 1000);
    restore(sample, path, /*watched=*/false);
  }
  std::fflush(stdout);
}

void RunWatcherFreshness(Workspace& workspace, WorktreeId tree,
                         absl::string_view root,
                         const std::vector<Sample>& samples, int trials) {
  for (int trial = 0; trial < trials; ++trial) {
    const Sample& sample = samples[static_cast<size_t>(trial) % samples.size()];
    const std::string path = absl::StrCat(root, "/", sample.path);
    const std::string token = absl::StrCat("ferretfreshnessw", trial, "x");
    SearchRequest request;
    request.pattern = token;
    request.context_lines = 0;
    TraceBegin();
    WriteFileOrDie(path, absl::StrCat(sample.original, "\n", token, "\n"));
    const int64_t t0 = absl::GetCurrentTimeNanos();
    int64_t end;
    for (;;) {
      auto result = workspace.Search(tree, request);
      CHECK_OK(result.status());
      end = absl::GetCurrentTimeNanos();
      if (!result->response.files.empty()) break;
      CHECK_LT(end - t0, int64_t{10'000'000'000});
      absl::SleepFor(absl::Microseconds(100));
    }
    watch_trace::Get().enabled.store(false);
    const int64_t callback =
        watch_trace::Get().nanos[watch_trace::kCallbackStart].load();
    const int64_t completed =
        watch_trace::Get().nanos[watch_trace::kCompleted].load();
    // event: the write until Ferret's callback; apply: the callback until the
    // generation is applied; visible: until the polling search sees it.
    const int64_t event = callback > 0 ? callback - t0 : 0;
    const int64_t apply =
        completed > 0 && callback > 0 ? completed - callback : 0;
    const int64_t visible = completed > 0 ? end - completed : 0;
    absl::PrintF(
        "FRESH ferret_watcher trial=%d total_us=%d event_us=%d apply_us=%d "
        "search_us=%d\n",
        trial, (end - t0) / 1000, event / 1000, apply / 1000, visible / 1000);
    WriteFileOrDie(path, sample.original);
    SearchRequest gone;
    gone.pattern = token;
    auto restored = workspace.Search(
        tree, gone,
        {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(10)});
    CHECK_OK(restored.status());
  }
  std::fflush(stdout);
}

// G1's request shape: top-20 files, 2 lines of context, estimated totals.
SearchRequest G1Request(const ferret_bench::QuerySpec& query) {
  SearchRequest request;
  request.pattern = query.pattern.pattern;
  request.regex = !query.pattern.literal;
  request.case_sensitive = query.pattern.case_sensitive;
  request.context_lines = 2;
  request.max_files = 20;
  request.max_lines_per_file = std::numeric_limits<int>::max();
  request.count_sample = true;
  return request;
}

struct ChurnFlags {
  std::string queries;
  std::string checkout;
  std::string marker;
  std::string csv;
  int clients = 10;
  double rate = 2.0;
  int before_s = 10;
  int after_s = 10;
  int quiet_ms = 500;
  uint64_t seed = 1;
};

int64_t Us(absl::Duration d) { return absl::ToInt64Microseconds(d); }

// Query latency on an already registered worktree while another one is
// registered cold: the churn clients run for `before_s`, the registration of
// `new_root` is the "checkout", and the windows are before, during the
// registration, during the compactions it triggers, and after.
int RunRegisterWhileQuerying(const std::string& index_dir,
                             const std::string& warm_root,
                             const std::string& new_root,
                             const ChurnFlags& flags, const std::string& qos) {
  auto text = Env::Default()->ReadFile(flags.queries);
  CHECK_OK(text.status());
  auto specs = ferret_bench::ParseQuerySet(*text);
  CHECK_OK(specs.status());
  ChurnOptions options;
  options.clients = flags.clients;
  options.rate_hz = flags.rate;
  options.seed = flags.seed;
  options.before = absl::Seconds(flags.before_s);
  options.after = absl::Seconds(flags.after_s);
  options.quiet = absl::Milliseconds(flags.quiet_ms);
  for (const auto& spec : *specs) options.requests.push_back(G1Request(spec));
  WorkspaceOptions workspace_options;
  workspace_options.serial_bulk_write = qos == "serial";
  workspace_options.reader_qos =
      qos == "interactive" ? ThreadQos::kUserInteractive
      : qos == "initiated" ? ThreadQos::kUserInitiated
                           : ThreadQos::kUtility;
  auto opened =
      Workspace::Open(*Env::Default(), index_dir, std::move(workspace_options));
  CHECK_OK(opened.status());
  std::unique_ptr<Workspace> workspace = std::move(*opened);
  auto tree = workspace->Register(warm_root);
  CHECK_OK(tree.status());
  workspace->WaitForCompactions();
  for (int pass = 0; pass < 2; ++pass) {
    for (const SearchRequest& request : options.requests) {
      CHECK_OK(workspace->Search(*tree, request).status());
    }
  }
  WorkspaceStats before_registration;
  absl::Time registration_start;
  absl::Duration registration_wall;
  WorkspaceStats after_registration;
  options.checkout = [&]() -> absl::Status {
    before_registration = workspace->Stats();
    registration_start = absl::Now();
    auto id = workspace->Register(new_root);
    registration_wall = absl::Now() - registration_start;
    after_registration = workspace->Stats();
    return id.status();
  };
  absl::PrintF(
      "REGQ config: qos=%s clients=%d rate_hz=%.2f before_s=%d "
      "after_s=%d queries=%d\n",
      qos, flags.clients, flags.rate, flags.before_s, flags.after_s,
      options.requests.size());
  std::fflush(stdout);
  auto result = RunChurn(*workspace, *tree, options);
  CHECK_OK(result.status());
  PrintRegistration("REGQ registration", registration_wall,
                    absl::ZeroDuration(), before_registration,
                    after_registration);
  absl::PrintF(
      "REGQ event: registration_start_ms=%d registration_ms=%d "
      "merge_done_ms=%d end_ms=%d\n",
      Millis(result->events.checkout_start), Millis(result->checkout_took),
      Millis(result->events.merge_done), Millis(result->events.end));
  for (size_t w = 0; w < kNumWindows; ++w) {
    const WindowStats& window = result->windows[w];
    absl::PrintF(
        "REGQ window %s: begin_ms=%d end_ms=%d n=%d p50_us=%d p99_us=%d "
        "max_us=%d p99_response_us=%d stale=%d failed=%d\n",
        ChurnWindowName(static_cast<ChurnWindow>(w)), Millis(window.begin),
        Millis(window.end), window.count, Us(window.p50), Us(window.p99),
        Us(window.max), Us(window.p99_response), window.stale, window.failed);
  }
  if (!flags.csv.empty()) {
    std::ofstream csv(flags.csv);
    csv << "client,intended_us,started_us,latency_us,outcome\n";
    for (const QuerySample& sample : result->samples) {
      csv << sample.client << ',' << Us(sample.intended) << ','
          << Us(sample.started) << ',' << Us(sample.latency) << ','
          << static_cast<int>(sample.outcome) << '\n';
    }
  }
  return 0;
}

int RunChurnMode(const std::string& index_dir, const std::string& root,
                 const ChurnFlags& flags) {
  auto text = Env::Default()->ReadFile(flags.queries);
  CHECK_OK(text.status());
  auto specs = ferret_bench::ParseQuerySet(*text);
  CHECK_OK(specs.status());
  ChurnOptions options;
  options.clients = flags.clients;
  options.rate_hz = flags.rate;
  options.seed = flags.seed;
  options.before = absl::Seconds(flags.before_s);
  options.after = absl::Seconds(flags.after_s);
  options.quiet = absl::Milliseconds(flags.quiet_ms);
  options.marker = flags.marker;
  for (const auto& spec : *specs) options.requests.push_back(G1Request(spec));
  options.checkout = [&]() -> absl::Status {
    auto result = RunSubprocess(
        {"git", "-C", root, "checkout", "--quiet", "--detach", flags.checkout});
    if (!result.ok()) return result.status();
    if (result->exit_code != 0) {
      return absl::InternalError(
          absl::StrCat("git checkout: ", result->stderr_data));
    }
    return absl::OkStatus();
  };
  auto opened = Workspace::Open(*Env::Default(), index_dir);
  CHECK_OK(opened.status());
  std::unique_ptr<Workspace> workspace = std::move(*opened);
  const absl::Time registered_at = absl::Now();
  auto tree = workspace->Register(root, {.watch = true});
  CHECK_OK(tree.status());
  workspace->WaitForCompactions();
  absl::PrintF("CHURN register: wall_ms=%d view_paths=%d segments=%d\n",
               Millis(absl::Now() - registered_at),
               workspace->WorktreeStatus(*tree)->view_paths,
               workspace->Stats().index.segment_count);
  // Warm the clients' queries once so the first window is not a cold start.
  for (const SearchRequest& request : options.requests) {
    CHECK_OK(workspace->Search(*tree, request).status());
  }
  absl::PrintF(
      "CHURN config: clients=%d rate_hz=%.2f seed=%d before_s=%d after_s=%d "
      "queries=%d checkout=%s marker=%s quiet_ms=%d\n",
      flags.clients, flags.rate, flags.seed, flags.before_s, flags.after_s,
      options.requests.size(), flags.checkout, flags.marker, flags.quiet_ms);
  std::fflush(stdout);
  auto result = RunChurn(*workspace, *tree, options);
  CHECK_OK(result.status());
  const WorkspaceStats stats = workspace->Stats();
  absl::PrintF(
      "CHURN event: checkout_start_ms=%d checkout_took_ms=%d "
      "marker_visible_ms=%d ingest_done_ms=%d merge_done_ms=%d end_ms=%d "
      "ingest_timed_out=%d compactions_in_ingest=%d compactions_in_merge=%d "
      "compaction_bytes_in_ingest=%d compaction_bytes_in_merge=%d "
      "view_paths_before=%d view_paths_after=%d rescans_bulk_files_read=%d "
      "interactive_files_read=%d sync_timeouts=%d watcher_failures=%d "
      "ingest_failures=%d\n",
      Millis(result->events.checkout_start), Millis(result->checkout_took),
      Millis(result->marker_visible), Millis(result->events.ingest_done),
      Millis(result->events.merge_done), Millis(result->events.end),
      result->ingest_timed_out, result->compactions_in_ingest,
      result->compactions_in_merge, result->compaction_bytes_in_ingest,
      result->compaction_bytes_in_merge, result->view_paths_before,
      result->view_paths_after, stats.bulk_files_read,
      stats.interactive_files_read, stats.sync_timeouts, stats.watcher_failures,
      stats.ingest_failures);
  for (size_t w = 0; w < kNumWindows; ++w) {
    const WindowStats& window = result->windows[w];
    absl::PrintF(
        "CHURN window %s: begin_ms=%d end_ms=%d n=%d p50_us=%d p99_us=%d "
        "max_us=%d p99_response_us=%d stale=%d failed=%d\n",
        ChurnWindowName(static_cast<ChurnWindow>(w)), Millis(window.begin),
        Millis(window.end), window.count, Us(window.p50), Us(window.p99),
        Us(window.max), Us(window.p99_response), window.stale, window.failed);
  }
  if (!flags.csv.empty()) {
    std::ofstream csv(flags.csv);
    csv << "client,intended_us,started_us,latency_us,outcome\n";
    for (const QuerySample& sample : result->samples) {
      csv << sample.client << ',' << Us(sample.intended) << ','
          << Us(sample.started) << ',' << Us(sample.latency) << ','
          << static_cast<int>(sample.outcome) << '\n';
    }
  }
  std::fflush(stdout);
  return 0;
}
}  // namespace
}  // namespace ferret

// Measures a full cold registration, reopening and resyncing the same views,
// and a fresh scan-based registration against the populated index. All views
// and the index live below the caller's scratch directory.
int RunRestoreBenchmark(int argc, char** argv) {
  CHECK(argc == 4 || argc == 5);
  const std::string index_dir = argv[2];
  const size_t count = static_cast<size_t>(argc - 3);
  absl::InitializeLog();
  const absl::Time first = absl::Now();
  auto opened = ferret::Workspace::Open(*ferret::Env::Default(), index_dir);
  CHECK_OK(opened.status());
  for (int i = 3; i < argc; ++i) {
    auto id = (*opened)->Register(argv[i]);
    CHECK_OK(id.status());
  }
  absl::PrintF("cold_registration_ms=%d\n",
               absl::ToInt64Milliseconds(absl::Now() - first));
  opened->reset();

  const absl::Time restore_start = absl::Now();
  opened = ferret::Workspace::Open(*ferret::Env::Default(), index_dir);
  CHECK_OK(opened.status());
  const absl::Time deadline = absl::Now() + absl::Minutes(10);
  while ((*opened)->Roots().size() != count && absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  CHECK_EQ((*opened)->Roots().size(), count);
  absl::PrintF("restore_open_to_published_ms=%d bulk_bytes=%d\n",
               absl::ToInt64Milliseconds(absl::Now() - restore_start),
               (*opened)->Stats().bulk_bytes_ingested);
  for (const auto& [root, id] : (*opened)->Roots()) {
    CHECK_OK((*opened)->Unregister(id));
  }
  const absl::Time fresh_start = absl::Now();
  for (int i = 3; i < argc; ++i) {
    auto id = (*opened)->Register(argv[i]);
    CHECK_OK(id.status());
  }
  absl::PrintF("fresh_same_index_ms=%d\n",
               absl::ToInt64Milliseconds(absl::Now() - fresh_start));
  opened->reset();

  auto index = ferret::Index::Open(*ferret::Env::Default(), index_dir);
  CHECK_OK(index.status());
  const absl::Time install_start = absl::Now();
  CHECK_OK(index->first->RecordView({"/view-install-timing", ""}));
  absl::PrintF("view_manifest_install_us=%d\n",
               absl::ToInt64Microseconds(absl::Now() - install_start));
  CHECK_OK(index->first->ForgetView("/view-install-timing"));
  return 0;
}

int main(int argc, char** argv) {
  using ferret::Workspace;
  if ((argc == 4 || argc == 5) && std::string(argv[1]) == "--restore") {
    return RunRestoreBenchmark(argc, argv);
  }
  if (argc == 5 && std::string(argv[1]) == "--gc") {
    absl::InitializeLog();
    ferret::RunGcComparison(argv[2], argv[3], argv[4]);
    return 0;
  }
  if (argc >= 4 && std::string(argv[1]) == "--g3") {
    absl::InitializeLog();
    const std::string index_dir = argv[2];
    ferret::WorkspaceOptions options;
    int first = 3;
    // Optional: --qos=utility|initiated|interactive, for the reader threads.
    if (std::string(argv[3]).rfind("--qos=", 0) == 0) {
      const std::string qos = std::string(argv[3]).substr(6);
      options.serial_bulk_write = qos == "serial";
      options.reader_qos =
          qos == "interactive" ? ferret::ThreadQos::kUserInteractive
          : qos == "initiated" ? ferret::ThreadQos::kUserInitiated
                               : ferret::ThreadQos::kUtility;
      ++first;
    }
    auto opened =
        Workspace::Open(*ferret::Env::Default(), index_dir, std::move(options));
    CHECK_OK(opened.status());
    int64_t published = 0;
    for (int i = first; i < argc; ++i) {
      const std::string arg = argv[i];
      const size_t eq = arg.find('=');
      CHECK(eq != std::string::npos) << "LABEL=WT expected: " << arg;
      ferret::RegisterG3(**opened, index_dir, arg.substr(0, eq),
                         arg.substr(eq + 1), &published);
    }
    return 0;
  }
  if (argc >= 4 && argc <= 5 && std::string(argv[1]) == "--freshness") {
    absl::InitializeLog();
    const int trials = argc == 5 ? std::stoi(argv[4]) : 200;
    auto opened = Workspace::Open(*ferret::Env::Default(), argv[2]);
    CHECK_OK(opened.status());
    auto tree = (*opened)->Register(argv[3], {.watch = true});
    CHECK_OK(tree.status());
    const std::vector<ferret::Sample> samples =
        ferret::Samples(argv[3], static_cast<size_t>(trials));
    ferret::RunFreshness(**opened, *tree, argv[3], samples, trials);
    ferret::RunWatcherFreshness(**opened, *tree, argv[3], samples, trials);
    return 0;
  }
  if (argc >= 5 && std::string(argv[1]) == "--regq") {
    ferret::ChurnFlags flags;
    std::string qos = "utility";
    for (int i = 5; i < argc; i += 2) {
      CHECK_LT(i + 1, argc) << "flag without a value: " << argv[i];
      const std::string flag = argv[i];
      const std::string value = argv[i + 1];
      if (flag == "--queries")
        flags.queries = value;
      else if (flag == "--csv")
        flags.csv = value;
      else if (flag == "--qos")
        qos = value;
      else if (flag == "--clients")
        flags.clients = std::stoi(value);
      else if (flag == "--rate")
        flags.rate = std::stod(value);
      else if (flag == "--before_s")
        flags.before_s = std::stoi(value);
      else if (flag == "--after_s")
        flags.after_s = std::stoi(value);
      else if (flag == "--seed")
        flags.seed = std::stoull(value);
      else
        CHECK(false) << "unknown flag " << flag;
    }
    CHECK(!flags.queries.empty());
    absl::InitializeLog();
    return ferret::RunRegisterWhileQuerying(argv[2], argv[3], argv[4], flags,
                                            qos);
  }
  if (argc >= 4 && std::string(argv[1]) == "--churn") {
    ferret::ChurnFlags flags;
    for (int i = 4; i < argc; i += 2) {
      CHECK_LT(i + 1, argc) << "flag without a value: " << argv[i];
      const std::string flag = argv[i];
      const std::string value = argv[i + 1];
      if (flag == "--queries")
        flags.queries = value;
      else if (flag == "--checkout")
        flags.checkout = value;
      else if (flag == "--marker")
        flags.marker = value;
      else if (flag == "--csv")
        flags.csv = value;
      else if (flag == "--clients")
        flags.clients = std::stoi(value);
      else if (flag == "--rate")
        flags.rate = std::stod(value);
      else if (flag == "--before_s")
        flags.before_s = std::stoi(value);
      else if (flag == "--after_s")
        flags.after_s = std::stoi(value);
      else if (flag == "--quiet_ms")
        flags.quiet_ms = std::stoi(value);
      else if (flag == "--seed")
        flags.seed = std::stoull(value);
      else
        CHECK(false) << "unknown flag " << flag;
    }
    CHECK(!flags.queries.empty() && !flags.checkout.empty());
    absl::InitializeLog();
    return ferret::RunChurnMode(argv[2], argv[3], flags);
  }
  if (argc >= 6 && std::string(argv[1]) == "--g2g3") {
    const int count = std::stoi(argv[2]);
    CHECK_GE(count, 1);
    CHECK_LE(count, 10);
    CHECK_EQ(argc, count + 5);
    absl::InitializeLog();
    const std::string index_dir = argv[3];
    const std::string first_path = argv[4];
    const std::string churn_path = argv[count + 4];
    absl::Notification bulk_paused;
    absl::Notification bulk_release;
    std::atomic<bool> gate{false};
    ferret::WorkspaceOptions options;
    options.read_hook = std::make_shared<ferret::WorkspaceReadHook>(
        [&](absl::string_view path, bool bulk) {
          if (bulk && path == ferret::kGateFile && gate.exchange(false)) {
            bulk_paused.Notify();
            bulk_release.WaitForNotification();
          }
        });
    auto opened = Workspace::Open(*ferret::Env::Default(), index_dir, options);
    CHECK_OK(opened.status());
    std::unique_ptr<Workspace> workspace = std::move(*opened);
    const ferret::WorktreeId first =
        ferret::RegisterTimed(*workspace, "k=1", first_path);
    for (int i = 2; i <= count; ++i) {
      const std::string label = absl::StrCat("k=", i);
      ferret::RegisterTimed(*workspace, label, argv[3 + i]);
      ferret::ScanTimed(label, argv[3 + i]);
    }
    const std::vector<ferret::Sample> samples =
        ferret::Samples(first_path, 1000);
    ferret::RunAllEdits(*workspace, first, first_path, samples, "idle");
    std::atomic<bool> stop{false};
    std::thread churn([&] {
      while (!stop.load()) {
        auto id = workspace->Register(churn_path);
        CHECK_OK(id.status());
        CHECK_OK(workspace->Unregister(*id));
      }
    });
    ferret::RunAllEdits(*workspace, first, first_path, samples,
                        "while registering");
    stop.store(true);
    churn.join();
    const std::string gate_path =
        absl::StrCat(churn_path, "/", ferret::kGateFile);
    ferret::WriteFileOrDie(gate_path, "bulk gate content\n");
    absl::Cleanup remove_gate = [&] { unlink(gate_path.c_str()); };
    gate.store(true);
    std::thread paused([&] {
      auto id = workspace->Register(churn_path);
      CHECK_OK(id.status());
    });
    CHECK(bulk_paused.WaitForNotificationWithTimeout(absl::Seconds(60)));
    ferret::RunAllEdits(*workspace, first, first_path, samples,
                        "bulk lane held");
    bulk_release.Notify();
    paused.join();
    std::move(remove_gate).Invoke();
    return 0;
  }
  using ferret::WorktreeId;
  if (argc >= 4 && argc <= 6 && std::string(argv[1]) == "--profile") {
    absl::InitializeLog();
    auto opened = Workspace::Open(*ferret::Env::Default(), argv[2]);
    CHECK_OK(opened.status());
    auto tree = (*opened)->Register(argv[3], {.watch = true});
    CHECK_OK(tree.status());
    ferret::ProfileAll(**opened, *tree, argv[3], ferret::Samples(argv[3], 1000),
                       argc >= 5 ? std::stoi(argv[4]) : 1000,
                       argc == 6 ? argv[5] : "");
    return 0;
  }
  CHECK((argc == 4 && std::string(argv[1]) == "--watch") || argc == 5 ||
        argc == 6)
      << "[--watch] INDEX_DIR WT1 [WT2 WT3 [DIFF_WT]]";
  if (argc == 4) {
    absl::InitializeLog();
    auto opened = Workspace::Open(*ferret::Env::Default(), argv[2]);
    CHECK_OK(opened.status());
    auto tree = (*opened)->Register(argv[3], {.watch = true});
    CHECK_OK(tree.status());
    ferret::WatchEdits(**opened, *tree, argv[3],
                       ferret::Samples(argv[3], 1000));
    opened->reset();
    ferret::WatchBurst(argv[3]);
    return 0;
  }
  const std::string wt1 = argv[2];
  const std::string wt3 = argv[4];
  absl::InitializeLog();
  absl::Notification bulk_paused;
  absl::Notification bulk_release;
  std::atomic<bool> gate{false};
  // Written by the hook before bulk_paused, and by main before bulk_release.
  absl::Time paused_at;
  absl::Time released_at;
  ferret::WorkspaceOptions options;
  options.read_hook = std::make_shared<ferret::WorkspaceReadHook>(
      [&](absl::string_view path, bool bulk) {
        if (bulk && path == ferret::kGateFile && gate.exchange(false)) {
          paused_at = absl::Now();
          bulk_paused.Notify();
          bulk_release.WaitForNotification();
        }
      });
  auto opened = Workspace::Open(*ferret::Env::Default(), argv[1], options);
  CHECK_OK(opened.status());
  std::unique_ptr<Workspace> workspace = std::move(*opened);

  // G3. Each registration starts once the previous one's compactions are
  // done, so no scan overlaps a merge.
  const WorktreeId first = ferret::RegisterTimed(*workspace, "k=1", wt1);
  ferret::RegisterTimed(*workspace, "k=2", argv[3]);
  ferret::ScanTimed("k=2", argv[3]);
  WorktreeId third = ferret::RegisterTimed(*workspace, "k=3", wt3);
  ferret::ScanTimed("k=3", wt3);
  if (argc == 6) {
    const WorktreeId diff = ferret::RegisterTimed(*workspace, "diff", argv[5]);
    ferret::ScanTimed("diff", argv[5]);
    CHECK_OK(workspace->Unregister(diff));
  }

  // G2.
  const std::vector<ferret::Sample> samples = ferret::Samples(wt1, 1000);
  ferret::RunAllEdits(*workspace, first, wt1, samples, "idle");

  CHECK_OK(workspace->Unregister(third));
  std::atomic<bool> stop{false};
  std::atomic<int> registrations{0};
  std::thread churn([&] {
    while (!stop.load()) {
      absl::StatusOr<WorktreeId> id = workspace->Register(wt3);
      CHECK_OK(id.status());
      CHECK_OK(workspace->Unregister(*id));
      registrations.fetch_add(1);
    }
  });
  ferret::RunAllEdits(*workspace, first, wt1, samples, "while registering");
  stop.store(true);
  churn.join();
  absl::PrintF("concurrent registrations of k=3: %d\n", registrations.load());

  // An untracked file makes the registration read in the bulk lane, where
  // the hook holds it while the edits run. The registration's own wall time
  // excludes the hold, and its counters are the bulk lane's only.
  const std::string gate_path = absl::StrCat(wt3, "/", ferret::kGateFile);
  ferret::WriteFileOrDie(gate_path, "bulk gate content\n");
  absl::Cleanup remove_gate = [&] { unlink(gate_path.c_str()); };
  gate.store(true);
  const ferret::WorkspaceStats before_paused = workspace->Stats();
  std::thread paused([&] {
    const absl::Time start = absl::Now();
    absl::StatusOr<WorktreeId> id = workspace->Register(wt3);
    const absl::Duration wall = absl::Now() - start;
    CHECK_OK(id.status());
    // Register returns after the hook's wait, which follows released_at.
    ferret::PrintRegistration("k=3 held during G2", wall,
                              released_at - paused_at, before_paused,
                              workspace->Stats());
  });
  CHECK(bulk_paused.WaitForNotificationWithTimeout(absl::Seconds(60)));
  ferret::RunAllEdits(*workspace, first, wt1, samples, "bulk lane held");
  released_at = absl::Now();
  bulk_release.Notify();
  paused.join();
  std::move(remove_gate).Invoke();

  workspace.reset();
  auto reopened = Workspace::Open(*ferret::Env::Default(), argv[1]);
  CHECK_OK(reopened.status());
  ferret::RegisterTimed(**reopened, "reopened k=1", wt1);
  return 0;
}

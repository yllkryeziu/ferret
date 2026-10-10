// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Stage timestamps for profiling the write -> searchable path and the
// `kSync` barrier. Disabled by default: a mark is then one relaxed load.
// A benchmark enables it, clears it before an operation, and reads the marks
// afterwards. A later mark of a stage overwrites an earlier one, and
// `callbacks` counts deliveries, so a write that arrives in two FSEvents
// deliveries is visible. The marks are for one operation at a time.

#ifndef FERRET_WATCH_WATCH_TRACE_H_
#define FERRET_WATCH_WATCH_TRACE_H_

#include <atomic>
#include <cstdint>

#include "absl/time/clock.h"

namespace ferret::watch_trace {

enum Stage : int {
  kSyncStart,       // Workspace::SyncWatcher entered.
  kCookieCreated,   // Cookie file created and closed.
  kEventSeen,       // The FSEvents queue saw a file event (not a cookie).
  kCookieSeen,      // The FSEvents queue saw this Sync's cookie.
  kDelivering,      // The delivery thread took the batch to deliver.
  kCookieDone,      // The delivery thread finished the cookie's ticket.
  kSyncWoke,        // The Sync caller woke up.
  kCallbackStart,   // Workspace::OnWatchEvents entered.
  kQueued,          // The Notify job is queued for the interactive lane.
  kDequeued,        // The interactive lane took the job.
  kPinned,          // Interactive batch: GC pinned.
  kIndexRefreshed,  // Interactive batch: indexed-blob set refreshed.
  kIgnoreReady,     // Interactive batch: ignore matcher ready.
  kBatchSetup,      // Interactive batch: index refresh, ignore rules, tracked.
  kFileRead,        // The last file's bytes are read (before PrepareDocument).
  kRead,            // Files read and prepared (trigrams, tokens, symbols).
  kAdded,           // New content is in the memtable.
  kPublished,       // The view batch is published.
  kCompleted,       // The generation is applied.
  kUnpinned,        // Interactive batch: GC resumed, lane free again.
  kSyncReturned,    // Watcher::Sync returned inside SyncWatcher.
  kExcludesDone,    // External exclude files compared.
  kGenerationDone,  // The generation wait in Search returned.
  kSnapshotDone,    // View and ReadState captured.
  kStageCount
};

struct Trace {
  std::atomic<bool> enabled{false};
  std::atomic<int64_t> nanos[kStageCount] = {};
  std::atomic<int64_t> callbacks{0};
};

inline Trace& Get() {
  static Trace trace;
  return trace;
}

inline void Mark(Stage stage) {
  Trace& trace = Get();
  if (trace.enabled.load(std::memory_order_relaxed)) {
    trace.nanos[stage].store(absl::GetCurrentTimeNanos(),
                             std::memory_order_relaxed);
  }
}

inline void Clear() {
  Trace& trace = Get();
  for (auto& nanos : trace.nanos) nanos.store(0, std::memory_order_relaxed);
  trace.callbacks.store(0, std::memory_order_relaxed);
}

}  // namespace ferret::watch_trace

#endif  // FERRET_WATCH_WATCH_TRACE_H_

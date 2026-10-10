// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Debounces and coalesces filesystem hints. Subtrees absorb their descendants;
// balanced sets maintain deterministic output order at logarithmic insertion
// cost plus the cost of checking overlapping prefixes.

#ifndef FERRET_WATCH_INTERNAL_EVENT_BATCHER_H_
#define FERRET_WATCH_INTERNAL_EVENT_BATCHER_H_

#include <vector>

#include "absl/container/btree_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/time/time.h"

#include "ferret/watch/watcher.h"

namespace ferret::watch_internal {

// Not thread-safe: the owner serializes Add, Due, and Flush. The clock is
// injected so that deadline behavior can be tested without sleeping.
class EventBatcher {
 public:
  // Requires positive delays and a clock callable that outlives this batcher.
  EventBatcher(absl::Duration quiet, absl::Duration maximum,
               absl::AnyInvocable<absl::Time() const> now);

  // Adds a hint, dropping duplicates and hints covered by a subtree. Each
  // addition restarts the quiet window but not the maximum delay. The first
  // event after an idle period is due immediately.
  void Add(WatchEvent event);
  // Returns when the batch is due: at its first event outside a burst, else
  // the earlier of the quiet and maximum deadlines. InfiniteFuture when the
  // batch is empty.
  absl::Time deadline() const;
  // Reports whether the current clock has reached the deadline.
  bool Due() const;
  // Returns all hints and resets the deadlines. Subtree hints come first in
  // lexicographic order, then path hints in lexicographic order.
  std::vector<WatchEvent> Flush();
  // A completed sync barrier ends a burst even if the next edit is soon.
  void EndBurst();

 private:
  absl::Duration quiet_;
  absl::Duration maximum_;
  absl::AnyInvocable<absl::Time() const> now_;
  absl::Time first_ = absl::InfiniteFuture();
  absl::Time last_ = absl::InfiniteFuture();
  // After a multi-file delivery, debounce until activity goes quiet.
  bool burst_started_ = false;
  absl::btree_set<std::string> paths_;
  absl::btree_set<std::string> subtrees_;
};

}  // namespace ferret::watch_internal

#endif  // FERRET_WATCH_INTERNAL_EVENT_BATCHER_H_

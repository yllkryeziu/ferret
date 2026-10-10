// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/watch/internal/event_batcher.h"

#include <algorithm>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/time/time.h"

#include "ferret/watch/watcher.h"

namespace ferret::watch_internal {
namespace {

bool Covers(const std::string& prefix, const std::string& path) {
  return prefix.empty() || path == prefix ||
         (path.size() > prefix.size() &&
          path.compare(0, prefix.size(), prefix) == 0 &&
          path[prefix.size()] == '/');
}

}  // namespace

EventBatcher::EventBatcher(absl::Duration quiet, absl::Duration maximum,
                           absl::AnyInvocable<absl::Time() const> now)
    : quiet_(quiet), maximum_(maximum), now_(std::move(now)) {
  CHECK_GT(quiet, absl::ZeroDuration());
  CHECK_GT(maximum, absl::ZeroDuration());
}

void EventBatcher::Add(WatchEvent event) {
  const absl::Time now = now_();
  if (first_ == absl::InfiniteFuture()) {
    if (now - last_ >= quiet_) burst_started_ = false;
    first_ = now;
  }
  last_ = now;
  if (auto* path = std::get_if<PathChanged>(&event)) {
    for (const std::string& subtree : subtrees_) {
      if (Covers(subtree, path->rel_path)) {
        return;
      }
    }
    paths_.insert(std::move(path->rel_path));
    return;
  }
  std::string prefix = std::move(std::get<SubtreeChanged>(event).rel_prefix);
  for (const std::string& subtree : subtrees_) {
    if (Covers(subtree, prefix)) {
      return;
    }
  }
  for (auto it = subtrees_.begin(); it != subtrees_.end();) {
    if (Covers(prefix, *it)) {
      it = subtrees_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = paths_.begin(); it != paths_.end();) {
    if (Covers(prefix, *it)) {
      it = paths_.erase(it);
    } else {
      ++it;
    }
  }
  subtrees_.insert(std::move(prefix));
}

absl::Time EventBatcher::deadline() const {
  if (first_ == absl::InfiniteFuture()) {
    return first_;
  }
  if (!burst_started_) return first_;
  return std::min(first_ + maximum_, last_ + quiet_);
}

bool EventBatcher::Due() const { return now_() >= deadline(); }

void EventBatcher::EndBurst() { burst_started_ = false; }

std::vector<WatchEvent> EventBatcher::Flush() {
  std::vector<WatchEvent> events;
  events.reserve(subtrees_.size() + paths_.size());
  for (const std::string& subtree : subtrees_) {
    events.emplace_back(SubtreeChanged{subtree});
  }
  for (const std::string& path : paths_) {
    events.emplace_back(PathChanged{path});
  }
  subtrees_.clear();
  paths_.clear();
  first_ = absl::InfiniteFuture();
  // A single-file edit can produce a follow-up hint. Do not hold that hint
  // behind a quiet window; only a multi-file delivery enters burst mode.
  burst_started_ = events.size() >= 8;
  return events;
}

}  // namespace ferret::watch_internal

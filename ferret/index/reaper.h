// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Background segment destruction and obsolete-file cleanup for immutable
// versions. Last releases never unmap on query threads.

#ifndef FERRET_INDEX_REAPER_H_
#define FERRET_INDEX_REAPER_H_

#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <thread>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/synchronization/mutex.h"

#include "ferret/base/env.h"
#include "ferret/index/segment.h"

namespace ferret {

// Runs cleanup off the thread that released the last reference. All segments
// and the Env must be destroyed before the Reaper. Thread-safe.
class Reaper final {
 public:
  Reaper();
  Reaper(const Reaper&) = delete;
  Reaper& operator=(const Reaper&) = delete;
  ~Reaper();

  // Queues `task` for the worker, in submission order. Returns without
  // waiting for it; destruction drains all queued tasks. Thread-safe.
  void Submit(absl::AnyInvocable<void() &&> task) ABSL_LOCKS_EXCLUDED(mu_);

 private:
  bool ReadyLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void Run();

  absl::Mutex mu_;
  std::deque<absl::AnyInvocable<void() &&>> queue_ ABSL_GUARDED_BY(mu_);
  bool stopping_ ABSL_GUARDED_BY(mu_) = false;
  std::thread worker_;
};

// The installer holds an old Version until after MarkObsolete(), so the last
// release cannot race ahead of the mark. The reaper reads this flag with
// acquire ordering after destroying the segment; only a committed removal
// sets it. The flag can also be set after a query releases its reference.
// The Env and Reaper must outlive this handle and all copies of segment().
// Thread-safe.
class ReapedSegment final {
 public:
  // Takes ownership of `segment` at `path`; the reaper destroys its mapping
  // after the last reference, and unlinks the path only if marked obsolete.
  ReapedSegment(Env& env, Reaper& reaper, std::string path,
                std::unique_ptr<const Segment> segment);
  ReapedSegment(const ReapedSegment&) = delete;
  ReapedSegment& operator=(const ReapedSegment&) = delete;

  // Returns the read-only segment; its last release queues destruction.
  const std::shared_ptr<const Segment>& segment() const { return segment_; }
  // Marks a committed removal. After the last segment reference is released,
  // the reaper unmaps it and best-effort unlinks its file. Thread-safe.
  void MarkObsolete();

 private:
  struct State {
    std::atomic<bool> obsolete{false};
  };
  std::shared_ptr<State> state_;
  std::shared_ptr<const Segment> segment_;
};

}  // namespace ferret

#endif  // FERRET_INDEX_REAPER_H_

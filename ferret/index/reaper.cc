// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/reaper.h"

#include <atomic>
#include <memory>
#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/synchronization/mutex.h"

#include "ferret/base/env.h"
#include "ferret/index/segment.h"

namespace ferret {

Reaper::Reaper() : worker_([this] { Run(); }) {}

Reaper::~Reaper() {
  {
    absl::MutexLock lock(mu_);
    stopping_ = true;
  }
  worker_.join();
}

void Reaper::Submit(absl::AnyInvocable<void() &&> task) {
  absl::MutexLock lock(mu_);
  queue_.push_back(std::move(task));
}

bool Reaper::ReadyLocked() const { return stopping_ || !queue_.empty(); }

void Reaper::Run() {
  for (;;) {
    absl::AnyInvocable<void() &&> task;
    {
      absl::MutexLock lock(mu_);
      mu_.Await(absl::Condition(this, &Reaper::ReadyLocked));
      if (queue_.empty()) return;
      task = std::move(queue_.front());
      queue_.pop_front();
    }
    std::move(task)();
  }
}

ReapedSegment::ReapedSegment(Env& env, Reaper& reaper, std::string path,
                             std::unique_ptr<const Segment> segment)
    : state_(std::make_shared<State>()),
      segment_(segment.release(),
               [state = state_, &env, &reaper,
                path = std::move(path)](const Segment* ptr) mutable {
                 reaper.Submit([state = std::move(state), &env,
                                path = std::move(path), ptr]() mutable {
                   delete ptr;
                   if (state->obsolete.load(std::memory_order_acquire)) {
                     static_cast<void>(env.DeleteFile(path));
                   }
                 });
               }) {}

void ReapedSegment::MarkObsolete() {
  state_->obsolete.store(true, std::memory_order_release);
}

}  // namespace ferret

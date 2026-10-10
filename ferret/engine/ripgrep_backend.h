// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Live, unindexed worktree searches using ripgrep's plain match stream.
#ifndef FERRET_ENGINE_RIPGREP_BACKEND_H_
#define FERRET_ENGINE_RIPGREP_BACKEND_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

#include "ferret/engine/engine.h"

namespace ferret {

class RipgrepCache;

class RipgrepBackend final : public SearchBackend {
 public:
  // An empty path resolves FERRET_RG, then searches PATH. No index is built.
  explicit RipgrepBackend(std::string rg_path = "")
      : rg_path_(std::move(rg_path)) {}

  absl::Status Register(absl::string_view root);
  std::vector<std::string> Roots() const override;
  absl::StatusOr<WorkspaceSearchResult> Search(
      absl::string_view root, const SearchRequest& request,
      const SearchConsistency& consistency) const override;
  // One entry per root. All index counters are zero, except skipped_paths:
  // the number of files the root's latest search left out because rg's bytes
  // disagreed with a later read. This backend has no index, generation,
  // queue, or watcher.
  absl::StatusOr<std::vector<WorktreeIndexStatus>> IndexStatus() const override;

 private:
  std::string rg_path_;
  mutable absl::Mutex mu_;
  std::vector<std::string> roots_ ABSL_GUARDED_BY(mu_);
  std::map<std::string, std::shared_ptr<RipgrepCache>> caches_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace ferret

#endif  // FERRET_ENGINE_RIPGREP_BACKEND_H_

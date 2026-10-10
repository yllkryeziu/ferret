// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// In-process search across registered worktrees. A repository's worktrees
// share one Workspace and one content-addressed index.
#ifndef FERRET_ENGINE_ENGINE_H_
#define FERRET_ENGINE_ENGINE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

#include "ferret/base/env.h"
#include "ferret/search/search_types.h"
#include "ferret/workspace/workspace.h"

namespace ferret {

// A snapshot of a registered worktree's published view and ingest progress.
struct WorktreeIndexStatus {
  // Canonical registered root.
  std::string root;
  // Paths and distinct blobs visible in the published view.
  int64_t file_count = 0;
  int64_t blob_count = 0;
  // Bytes and segments in the index's published version.
  int64_t index_bytes = 0;
  int64_t segment_count = 0;
  uint64_t pending_symbols = 0;
  uint64_t failed_symbols = 0;
  // Last applied and assigned worktree generations.
  uint64_t applied_generation = 0;
  uint64_t assigned_generation = 0;
  // Paths waiting in the interactive lane, unstable paths, and skipped paths.
  int64_t queued_paths = 0;
  int64_t unstable_paths = 0;
  int64_t skipped_paths = 0;
  // True while the watcher runs; false after failover or when unwatched.
  bool watcher_running = false;
  bool watcher_failed = false;
};

// Configures index placement and Workspace hooks. Move into the engine once.
struct EngineOptions {
  // The directory that holds one index per repository, in a subdirectory
  // named by RepositoryCacheDir's hash. Empty selects <cache>/ferret.
  std::string index_dir;
  // Options shared by all owned workspaces, including test hooks.
  WorkspaceOptions workspace;
};

// Selects a cache directory from explicit environment values on either OS.
std::string CacheDirectory(absl::string_view home,
                           absl::string_view xdg_cache_home, bool macos);

// All methods are thread-safe; the implementation must outlive its callers.
class SearchBackend {
 public:
  // Destruction requires concurrent calls to have completed.
  virtual ~SearchBackend() = default;
  // Lists published roots without blocking on registration scans.
  virtual std::vector<std::string> Roots() const = 0;
  // Searches a root; errors for unknown roots and invalid requests.
  virtual absl::StatusOr<WorkspaceSearchResult> Search(
      absl::string_view root, const SearchRequest& request,
      const SearchConsistency& consistency) const = 0;
  virtual bool SupportsFindSymbol() const { return false; }
  virtual absl::StatusOr<FindSymbolResponse> FindSymbol(
      absl::string_view, const FindSymbolRequest&,
      const SearchConsistency&) const {
    return absl::UnimplementedError("symbol lookup is unavailable");
  }
  // Returns snapshot counters without filesystem scanning.
  virtual absl::StatusOr<std::vector<WorktreeIndexStatus>> IndexStatus()
      const = 0;
};

// Returns the canonical git common directory. A non-repository root
// is rejected. Thread-safe; git receives an argv, never a shell command.
absl::StatusOr<std::string> GitCommonDir(absl::string_view root);
// The cache key of the common git directory, or of a plain directory.
absl::StatusOr<std::string> RepositoryKeyForRoot(absl::string_view root);

// The first 16 hex digits of SHA-256(realpath(git common dir)).
absl::StatusOr<std::string> RepositoryCacheDir(absl::string_view root);

// All operations are thread-safe and may overlap registration. Destruction
// requires all calls to have returned; the borrowed Env must outlive the
// engine.
class Engine final {
 public:
  // Borrows env; the options and workspaces belong to the engine.
  explicit Engine(Env& env, EngineOptions options = {})
      : env_(env), options_(std::move(options)) {}
  // Convenience overload for callers overriding only the index directory.
  Engine(Env& env, std::string index_dir)
      : Engine(env, EngineOptions{.index_dir = std::move(index_dir)}) {}
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Opens the repository index and begins restoring its persisted views.
  absl::Status OpenRepository(absl::string_view root);

  // Registers a canonical root with watching. An already published root returns
  // OK without scanning; AlreadyExists means registration is still in progress.
  // Failures release the reservation.
  absl::Status Register(absl::string_view root);
  // Returns published roots, excluding registrations in progress.
  std::vector<std::string> Roots() const;
  // Searches a published root; NotFound for an unknown root.
  absl::StatusOr<WorkspaceSearchResult> Search(
      absl::string_view root, const SearchRequest& request,
      const SearchConsistency& consistency) const;
  absl::StatusOr<FindSymbolResponse> FindSymbol(
      absl::string_view root, const FindSymbolRequest& request,
      const SearchConsistency& consistency) const;
  // Snapshots view and index counters without scanning the filesystem.
  absl::StatusOr<std::vector<WorktreeIndexStatus>> IndexStatus() const;
  // Explicit notifications for clients without a watcher. The returned
  // generation can be passed to Workspace's kNotify consistency mode.
  absl::StatusOr<uint64_t> Notify(absl::string_view root,
                                  const std::vector<std::string>& paths);

 private:
  struct Registration {
    std::string root;
    WorktreeId id;
    // Shares ownership with the engine map; safe after releasing mu_.
    std::shared_ptr<Workspace> workspace;
  };
  absl::StatusOr<Registration> Find(absl::string_view root) const;

  Env& env_;
  EngineOptions options_;
  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::shared_ptr<Workspace>> workspaces_
      ABSL_GUARDED_BY(mu_);
  // Workspaces in opening order, for deterministic restored-root enumeration.
  std::vector<std::string> workspace_order_ ABSL_GUARDED_BY(mu_);
  std::vector<Registration> registrations_ ABSL_GUARDED_BY(mu_);
  // Includes reservations until registration succeeds or fails.
  absl::flat_hash_set<std::string> roots_ ABSL_GUARDED_BY(mu_);
};

// Borrows an engine that must outlive the backend and its concurrent callers.
class EngineBackend final : public SearchBackend {
 public:
  // Borrows an engine that outlives the backend.
  explicit EngineBackend(Engine& engine) : engine_(engine) {}
  // Forwards thread-safe root enumeration.
  std::vector<std::string> Roots() const override { return engine_.Roots(); }
  // Forwards a search, including its freshness flag.
  absl::StatusOr<WorkspaceSearchResult> Search(
      absl::string_view root, const SearchRequest& request,
      const SearchConsistency& consistency) const override {
    return engine_.Search(root, request, consistency);
  }
  bool SupportsFindSymbol() const override { return true; }
  absl::StatusOr<FindSymbolResponse> FindSymbol(
      absl::string_view root, const FindSymbolRequest& request,
      const SearchConsistency& consistency) const override {
    return engine_.FindSymbol(root, request, consistency);
  }
  // Forwards status snapshots.
  absl::StatusOr<std::vector<WorktreeIndexStatus>> IndexStatus()
      const override {
    return engine_.IndexStatus();
  }

 private:
  Engine& engine_;
};

// The measured edit-to-kSync p99 is 27 ms on the Linux scratch worktree.
// Allow one second for scheduling and interactive ingestion under load
// without imposing an unbounded stall.
inline constexpr absl::Duration kServingSyncTimeout = absl::Seconds(1);
// Production searches wait for the Workspace watcher barrier.
inline constexpr SearchConsistencyMode kServingConsistency =
    SearchConsistencyMode::kSync;

}  // namespace ferret
#endif  // FERRET_ENGINE_ENGINE_H_

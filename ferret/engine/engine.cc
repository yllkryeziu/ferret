// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
#include "ferret/engine/engine.h"

#include <stdlib.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "openssl/sha.h"

#include "ferret/base/subprocess.h"
#include "ferret/search/search_types.h"
#include "ferret/workspace/workspace.h"

namespace ferret {
namespace {

absl::StatusOr<std::string> Canonical(absl::string_view path) {
  const std::string path_string(path);
  // realpath is declared by the platform stdlib.h.
  // NOLINTNEXTLINE(misc-include-cleaner)
  char* resolved = realpath(path_string.c_str(), nullptr);
  if (resolved == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat("cannot resolve ", path));
  }
  std::string canonical(resolved);
  free(resolved);
  return canonical;
}

// The first 16 hex digits of SHA-256(common).
std::string RepositoryKey(absl::string_view common) {
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest;
  SHA256(reinterpret_cast<const unsigned char*>(common.data()), common.size(),
         digest.data());
  constexpr char kHex[] = "0123456789abcdef";
  std::string hash;
  for (int i = 0; i < 8; ++i) {
    hash.push_back(kHex[digest[i] >> 4]);
    hash.push_back(kHex[digest[i] & 15]);
  }
  return hash;
}

std::string CachePath(absl::string_view common) {
  const char* home = getenv("HOME");
  const char* xdg = getenv("XDG_CACHE_HOME");
  return absl::StrCat(
      CacheDirectory(home == nullptr ? "" : home, xdg == nullptr ? "" : xdg,
#if defined(__APPLE__)
                     true
#else
                     false
#endif
                     ),
      "/ferret/", RepositoryKey(common));
}

}  // namespace

std::string CacheDirectory(absl::string_view home,
                           absl::string_view xdg_cache_home, bool macos) {
  return macos ? absl::StrCat(home, "/Library/Caches")
               : (xdg_cache_home.empty() ? absl::StrCat(home, "/.cache")
                                         : std::string(xdg_cache_home));
}

absl::StatusOr<std::string> GitCommonDir(absl::string_view root) {
  ABSL_ASSIGN_OR_RETURN(std::string canonical, Canonical(root));
  const std::vector<std::string> argv = {"git",
                                         "-C",
                                         canonical,
                                         "rev-parse",
                                         "--path-format=absolute",
                                         "--git-common-dir"};
  ABSL_ASSIGN_OR_RETURN(SubprocessResult result, RunSubprocess(argv));
  if (result.exit_code != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("not a git repository: ", root));
  }
  return Canonical(absl::StripAsciiWhitespace(result.stdout_data));
}

absl::StatusOr<std::string> RepositoryCacheDir(absl::string_view root) {
  ABSL_ASSIGN_OR_RETURN(std::string common, GitCommonDir(root));
  return CachePath(common);
}

absl::StatusOr<std::string> RepositoryKeyForRoot(absl::string_view root) {
  ABSL_ASSIGN_OR_RETURN(std::string canonical, Canonical(root));
  absl::StatusOr<std::string> git_dir = GitCommonDir(canonical);
  return RepositoryKey(git_dir.ok() ? *git_dir : canonical);
}

absl::Status Engine::OpenRepository(absl::string_view root) {
  ABSL_ASSIGN_OR_RETURN(std::string canonical, Canonical(root));
  absl::StatusOr<std::string> git_dir = GitCommonDir(canonical);
  const std::string common = git_dir.ok() ? *git_dir : canonical;
  absl::MutexLock lock(mu_);
  if (workspaces_.contains(common)) return absl::OkStatus();
  const std::string dir =
      options_.index_dir.empty()
          ? CachePath(common)
          : absl::StrCat(options_.index_dir, "/", RepositoryKey(common));
  ABSL_ASSIGN_OR_RETURN(auto opened,
                        Workspace::Open(env_, dir, options_.workspace));
  workspaces_.emplace(common, std::move(opened));
  workspace_order_.push_back(common);
  return absl::OkStatus();
}

absl::Status Engine::Register(absl::string_view root) {
  ABSL_ASSIGN_OR_RETURN(std::string canonical, Canonical(root));
  // A plain directory has no git common dir; key its index by its root.
  absl::StatusOr<std::string> git_dir = GitCommonDir(canonical);
  const std::string common = git_dir.ok() ? *git_dir : canonical;
  std::string dir;
  std::shared_ptr<Workspace> workspace;
  {
    absl::MutexLock lock(mu_);
    if (!roots_.insert(canonical).second) {
      for (const Registration& registration : registrations_) {
        if (registration.root == canonical) return absl::OkStatus();
      }
      return absl::AlreadyExistsError(absl::StrCat("registered: ", canonical));
    }
    dir = options_.index_dir.empty()
              ? CachePath(common)
              : absl::StrCat(options_.index_dir, "/", RepositoryKey(common));
    auto it = workspaces_.find(common);
    if (it == workspaces_.end()) {
      absl::StatusOr<std::unique_ptr<Workspace>> opened =
          Workspace::Open(env_, dir, options_.workspace);
      if (!opened.ok()) {
        roots_.erase(canonical);
        return opened.status();
      }
      it = workspaces_.emplace(common, std::move(*opened)).first;
      workspace_order_.push_back(common);
    }
    workspace = it->second;
  }
  absl::StatusOr<WorktreeId> id =
      workspace->Register(canonical, {.watch = true});
  if (absl::IsAlreadyExists(id.status())) {
    for (const auto& [published_root, published] : workspace->Roots()) {
      if (published_root == canonical) {
        id = published;
        break;
      }
    }
  }
  {
    absl::MutexLock lock(mu_);
    if (!id.ok()) {
      roots_.erase(canonical);
      // Keep the workspace open: restored views may still live in its index.
      return id.status();
    }
    registrations_.push_back({canonical, *id, workspace});
  }
  return absl::OkStatus();
}

std::vector<std::string> Engine::Roots() const {
  absl::MutexLock lock(mu_);
  std::vector<std::string> roots;
  absl::flat_hash_set<std::string> seen;
  for (const Registration& registration : registrations_) {
    if (seen.insert(registration.root).second) {
      roots.push_back(registration.root);
    }
  }
  for (const std::string& common : workspace_order_) {
    for (const auto& [root, id] : workspaces_.find(common)->second->Roots()) {
      if (seen.insert(root).second) roots.push_back(root);
    }
  }
  return roots;
}

absl::StatusOr<Engine::Registration> Engine::Find(
    absl::string_view root) const {
  absl::MutexLock lock(mu_);
  for (const Registration& registration : registrations_) {
    if (registration.root == root) return registration;
  }
  for (const auto& [common, workspace] : workspaces_) {
    for (const auto& [published, id] : workspace->Roots()) {
      if (published == root) return Registration{published, id, workspace};
    }
  }
  return absl::NotFoundError(absl::StrCat("unknown worktree: ", root));
}

absl::StatusOr<WorkspaceSearchResult> Engine::Search(
    absl::string_view root, const SearchRequest& request,
    const SearchConsistency& consistency) const {
  ABSL_ASSIGN_OR_RETURN(Registration registration, Find(root));
  return registration.workspace->Search(registration.id, request, consistency);
}

absl::StatusOr<FindSymbolResponse> Engine::FindSymbol(
    absl::string_view root, const FindSymbolRequest& request,
    const SearchConsistency& consistency) const {
  ABSL_ASSIGN_OR_RETURN(Registration registration, Find(root));
  return registration.workspace->FindSymbol(registration.id, request,
                                            consistency);
}

absl::StatusOr<uint64_t> Engine::Notify(absl::string_view root,
                                        const std::vector<std::string>& paths) {
  ABSL_ASSIGN_OR_RETURN(Registration registration, Find(root));
  ABSL_ASSIGN_OR_RETURN(uint64_t generation,
                        registration.workspace->Notify(registration.id, paths));
  return generation;
}

absl::StatusOr<std::vector<WorktreeIndexStatus>> Engine::IndexStatus() const {
  std::vector<Registration> registrations;
  {
    absl::MutexLock lock(mu_);
    registrations = registrations_;
    absl::flat_hash_set<std::string> seen;
    for (const Registration& registration : registrations) {
      seen.insert(registration.root);
    }
    for (const std::string& common : workspace_order_) {
      const auto& workspace = workspaces_.find(common)->second;
      for (const auto& [root, id] : workspace->Roots()) {
        if (seen.insert(root).second) {
          registrations.push_back({root, id, workspace});
        }
      }
    }
  }
  std::vector<WorktreeIndexStatus> statuses;
  for (const Registration& registration : registrations) {
    const Workspace* workspace = registration.workspace.get();
    ABSL_ASSIGN_OR_RETURN(WorkspaceWorktreeStatus view,
                          workspace->WorktreeStatus(registration.id));
    WorktreeIndexStatus status;
    status.root = registration.root;
    status.file_count = view.view_paths;
    status.blob_count = view.view_blobs;
    status.applied_generation = view.applied_generation;
    status.assigned_generation = view.assigned_generation;
    status.queued_paths = view.queued_paths;
    status.unstable_paths = view.unstable_paths;
    status.skipped_paths = view.skipped_paths;
    status.watcher_running = view.watcher_running;
    status.watcher_failed = view.watcher_failed;
    status.segment_count = view.segment_count;
    status.pending_symbols = view.pending_symbols;
    status.failed_symbols = view.failed_symbols;
    status.index_bytes = view.index_bytes;
    statuses.push_back(std::move(status));
  }
  return statuses;
}

}  // namespace ferret

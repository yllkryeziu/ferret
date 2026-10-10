// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The set of files in a git worktree that search should see: what git
// tracks there, plus untracked files that are not ignored.
//
// This is the static listing that `ferret index` builds from. The
// incremental scanner (dirty detection, OID skip hints, case-insensitive
// collisions) builds on it.

#ifndef FERRET_GIT_WORKTREE_FILES_H_
#define FERRET_GIT_WORKTREE_FILES_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace ferret {

// Returns the repo-relative paths, sorted bytewise and unique, of the
// worktree at `root`:
//   * index entries that are regular files (mode 100644 or 100755), except
//     skip-worktree entries, which a sparse checkout leaves absent; each
//     path of an unmerged entry appears once;
//   * untracked files that .gitignore, $GIT_DIR/info/exclude and
//     core.excludesFile do not exclude.
// Symlinks and gitlinks are omitted. A tracked file deleted from the
// worktree is still listed; readers must tolerate NotFound.
//
// Runs git with an argv, never a shell. Returns FailedPrecondition if `root`
// is not inside a git worktree, and an error for a SHA-256 repository,
// whose object ids ParseLsFiles() rejects. Thread-safe.
absl::StatusOr<std::vector<std::string>> ListWorktreeFiles(
    absl::string_view root);

struct DirectoryFilesOptions {
  // Repo-relative path of the index directory, if it lies inside `root`.
  std::string index_directory;
  // Fold case when matching ignore rules; read core.ignoreCase in the caller.
  bool ignore_case = false;
  // Optional $GIT_DIR/info/exclude, for callers listing git worktrees.
  std::string info_exclude_path;
  // Extra lowest-priority gitignore rules, added after the built-in default
  // excludes.
  std::vector<std::string> default_excludes;
};

// Lists the regular files below a non-git root, sorted by root-relative path.
// Ignored directories are pruned; symlinks are neither listed nor followed.
// Uses .gitignore, .ferretignore, and the built-in default excludes.
// Thread-safe; each call has its own ignore cache.
absl::StatusOr<std::vector<std::string>> ListDirectoryFiles(
    absl::string_view root, const DirectoryFilesOptions& options = {});

}  // namespace ferret

#endif  // FERRET_GIT_WORKTREE_FILES_H_

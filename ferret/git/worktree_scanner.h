// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Incremental worktree enumeration with index-Oid hints.

#ifndef FERRET_GIT_WORKTREE_SCANNER_H_
#define FERRET_GIT_WORKTREE_SCANNER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"

#include "ferret/git/oid.h"

namespace ferret {

enum class ScanReadReason {
  kDirty,            // Tracked bytes changed, or the tracked path is missing.
  kUntracked,        // The path is not in the index.
  kUnmerged,         // The path has unmerged index stages.
  kAssumeUnchanged,  // The index requests no worktree stat checks.
  kConversion        // Checkout attributes or config may transform the blob.
};

struct ScannedWorktreeFile {
  std::string path;                 // Root-relative path, with index spelling.
  std::optional<int64_t> size;      // Disk size; absent for missing paths.
  std::optional<absl::Time> mtime;  // Disk mtime; absent for missing paths.
  std::optional<Oid> skip_hint;     // Index blob Oid if reading may be skipped.
  std::optional<ScanReadReason> read_reason;  // Why no hint was issued.
};

struct ScanReport {
  int64_t files = 0;             // Returned regular or missing paths.
  int64_t hinted = 0;            // Paths with a skip hint.
  int64_t dirty = 0;             // Paths requiring a read due to disk changes.
  int64_t untracked = 0;         // Paths requiring a read outside the index.
  int64_t unmerged = 0;          // Paths with unmerged index stages.
  int64_t assume_unchanged = 0;  // Paths whose index stat data is not trusted.
  int64_t conversion = 0;        // Paths with possible checkout conversion.
  int64_t attribute_candidates = 0;  // Paths sent to git check-attr.
  int64_t skip_worktree = 0;         // Sparse index entries omitted.
  int64_t symlink = 0;               // Index symlinks omitted.
  int64_t not_regular = 0;           // Disk nonregular paths omitted.
  int64_t gitlink = 0;               // Index gitlinks omitted.
  int64_t case_collisions = 0;  // Index spellings omitted due to collisions.
  absl::Duration git_commands =
      absl::ZeroDuration();  // Format, status, and index time.
  absl::Duration attributes = absl::ZeroDuration();  // Conversion check time.
  absl::Duration lstat = absl::ZeroDuration();  // Parallel stat and case time.
};

struct ScanWorktreeOptions {
  // Fraction of hints verified by reading and hashing, in [0, 1]. Unset
  // selects 1% in debug builds and 0% in optimized builds.
  std::optional<double> verify_hint_fraction;
  int lstat_threads = 8;  // Maximum number of parallel stat workers.
};

struct WorktreeScan {
  std::vector<ScannedWorktreeFile> files;  // Bytewise sorted, unique paths.
  // Root-relative directories that git does not enter, without a trailing
  // '/': index gitlinks and untracked nested repositories. Bytewise sorted
  // and unique. No file below one is listed.
  std::vector<std::string> nested_repositories;
  ScanReport report;  // Counts and phase timings.
};

// Enumerates the same files as ListWorktreeFiles, including missing tracked
// paths. A hint is issued only for a clean, present, correctly spelled,
// stage-0 index regular file without assume-unchanged or skip-worktree. Index
// symlinks and gitlinks and disk nonregular files are omitted. Dirty, deleted,
// unmerged, untracked, or case-mismatched paths cannot receive hints. Nor can
// paths affected by core.autocrlf other than unset/false/input, core.eol=crlf,
// or effective eol=crlf, filter, ident, or working-tree-encoding attributes.
// Tracked and untracked .gitattributes, info/attributes, global attributes,
// and system attributes are consulted; unknown source paths select every hint
// candidate for git check-attr. Hints are sampled deterministically by Oid.
// Requires 0 <= verify_hint_fraction <= 1 and lstat_threads >= 1. Thread-safe.
absl::StatusOr<WorktreeScan> ScanWorktree(
    absl::string_view root, const ScanWorktreeOptions& options = {});

}  // namespace ferret

#endif  // FERRET_GIT_WORKTREE_SCANNER_H_

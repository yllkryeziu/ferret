// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Parser for the output of `git ls-files -s -v -z`: the index entries of a
// worktree with their modes, object ids, stages, and tags.
//
// Each record is "<tag> <mode> <oid> <stage>\t<path>\0". With -v the tag is
// 'H' (cached), 'S' (skip-worktree), or 'M' (unmerged, stage 1-3), and is
// lowercase when the entry is marked assume-unchanged. With -z paths are
// printed verbatim: they may contain spaces, newlines, or any byte but NUL.
//
// The input is untrusted (a repository can carry a crafted index), so the
// parser validates every field and never trusts a length it has not checked.

#ifndef FERRET_GIT_LS_FILES_H_
#define FERRET_GIT_LS_FILES_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "ferret/git/oid.h"

namespace ferret {

struct LsFilesEntry {
  // The mode as git prints it, e.g. 0100644, 0100755, 0120000 (symlink), or
  // 0160000 (gitlink).
  uint32_t mode = 0;
  Oid oid;
  // 0 for a merged entry; 1-3 for the base, ours, and theirs of a conflict.
  int stage = 0;
  // The entry is outside a sparse checkout and absent from the worktree.
  bool skip_worktree = false;
  // `git update-index --assume-unchanged`: git no longer checks the file for
  // changes, so the index says nothing about its content.
  bool assume_unchanged = false;
  // Normalized repo-relative path.
  std::string path;
};

// Parses the complete output of `git ls-files -s -v -z`, preserving order.
// Returns InvalidArgument, naming the offending record, if any record is
// malformed: an unknown tag, a tag that disagrees with the stage, a mode
// that is not six octal digits, an object id that is not 40 hex digits
// (SHA-256 repositories are rejected earlier, by the scanner), or a path
// that is not a normalized repo-relative path. O(output.size()).
// Thread-safe.
absl::StatusOr<std::vector<LsFilesEntry>> ParseLsFiles(
    absl::string_view output);

}  // namespace ferret

#endif  // FERRET_GIT_LS_FILES_H_

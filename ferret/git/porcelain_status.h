// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Parser for the output of
// `git --no-optional-locks status --porcelain=v2 -z -uall --no-renames`:
// which paths differ between the index and the worktree, which are
// unmerged, and which are untracked or ignored.
//
// Record formats (git-status(1), "Porcelain Format Version 2"), each ending
// in NUL:
//   1 <XY> <sub> <mH> <mI> <mW> <hH> <hI> <path>       changed
//   u <XY> <sub> <m1> <m2> <m3> <mW> <h1> <h2> <h3> <path>   unmerged
//   ? <path>                                            untracked
//   ! <path>                                            ignored (--ignored)
//   # <header>                                          skipped
// Rename records ("2 ...") cannot occur with --no-renames and are rejected.
// Paths are verbatim with -z and may contain spaces and newlines.
//
// The input is untrusted, so every field this parser keeps is validated and
// the fields it drops are checked for shape.

#ifndef FERRET_GIT_PORCELAIN_STATUS_H_
#define FERRET_GIT_PORCELAIN_STATUS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace ferret {

enum class PorcelainStatusKind {
  kChanged,
  kUnmerged,
  kUntracked,
  kIgnored,
};

struct PorcelainStatusEntry {
  PorcelainStatusKind kind = PorcelainStatusKind::kChanged;
  // X and Y of the record: the index against HEAD, and the worktree against
  // the index. One of ".MTADRCU", where '.' means unchanged. Always '.' for
  // untracked and ignored entries.
  char index_status = '.';
  char worktree_status = '.';
  // The worktree mode git observed (mW), e.g. 0100644; 0 if the worktree has
  // no file there. Always 0 for untracked and ignored entries.
  uint32_t worktree_mode = 0;
  // Normalized repo-relative path. Untracked and ignored directories that
  // git did not descend into (nested repositories, or ignored directories)
  // keep their trailing '/'.
  std::string path;
};

// Parses the complete status output, preserving order and skipping header
// records. Returns InvalidArgument, naming the offending record, if any
// record is malformed or is a rename record. O(output.size()). Thread-safe.
absl::StatusOr<std::vector<PorcelainStatusEntry>> ParsePorcelainStatus(
    absl::string_view output);

}  // namespace ferret

#endif  // FERRET_GIT_PORCELAIN_STATUS_H_

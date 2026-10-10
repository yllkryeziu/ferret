// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark file lists: the exact files both engines index, one repo-relative
// '/'-separated path per line. A list is generated from a git worktree by
// applying one eligibility policy (ferret/text/eligibility.h), normally the
// `zoekt` policy, under which Zoekt skips none of the listed files. The
// validation matches bench/zoekt/filelist.go.

#ifndef BENCH_FERRET_FILE_LIST_H_
#define BENCH_FERRET_FILE_LIST_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "ferret/base/env.h"
#include "ferret/text/eligibility.h"

namespace ferret_bench {

// Parses a file list. InvalidArgument, naming the line, for an empty list,
// an empty path, a path with CR or NUL, a path that is absolute or not
// clean (an empty, ".", or ".." component, or a trailing '/'), or a
// duplicate.
absl::StatusOr<std::vector<std::string>> ParseFileList(absl::string_view text);

// Returns `paths` as a file list: one path per line, each ending in '\n'.
std::string FormatFileList(const std::vector<std::string>& paths);

// Returns the paths of `paths` that the search tool's `path_glob` argument
// `glob` admits (ferret/git/path_glob.h), in their order. The replay
// (eval/harness/replay.py) uses this to give Zoekt the exact file set of a
// scoped search without porting the glob semantics. InvalidArgument for an
// invalid glob.
absl::StatusOr<std::vector<std::string>> FilterByPathGlob(
    const std::vector<std::string>& paths, absl::string_view glob);

struct EligibleFiles {
  // Sorted by path bytes.
  std::vector<std::string> paths;
  // Files of the worktree listing.
  int64_t files_listed = 0;
  // Files the policy skips, indexed by SkipReason (entry kNone stays 0).
  std::array<int64_t, ferret::kNumSkipReasons> files_skipped = {};
};

// Returns the files of the git worktree at `root` (ListWorktreeFiles) that
// `options` makes eligible. A listed file that cannot be read, such as a
// tracked file deleted from the worktree, is an error: a benchmark corpus
// must be a clean checkout. O(total bytes).
absl::StatusOr<EligibleFiles> ListEligibleFiles(
    ferret::Env& env, absl::string_view root,
    const ferret::EligibilityOptions& options);

}  // namespace ferret_bench

#endif  // BENCH_FERRET_FILE_LIST_H_

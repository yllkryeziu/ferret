// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Filters repo-relative file paths by a glob in .gitignore syntax: the
// meaning of the search tool's `path_glob` argument.
//
//   * A glob without a '/' matches a file name or a directory name at any
//     depth: "*.h", "BUILD", "testdata".
//   * Any other glob matches from the repository root: '*' and '?' do not
//     cross '/', and "**" matches any number of directories: "src/**/*.cc",
//     "/docs/*.md".
//   * A trailing '/' matches only directories: "third_party/".
//   * A file matches when the glob matches its path or any directory above
//     it, so "src" and "src/" select everything under src/.
//   * A leading '!' inverts the filter: "!*_test.cc".

#ifndef FERRET_GIT_PATH_GLOB_H_
#define FERRET_GIT_PATH_GLOB_H_

#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "ferret/git/internal/path_pattern.h"

namespace ferret {

// A regular value type. Thread-compatible.
class PathGlob final {
 public:
  // InvalidArgument if `glob` is empty, "!" alone, or contains a NUL byte.
  static absl::StatusOr<PathGlob> Parse(absl::string_view glob);

  PathGlob(const PathGlob&) = default;
  PathGlob& operator=(const PathGlob&) = default;

  // Returns whether the file at repo-relative `path` passes the filter.
  // O(depth of path) glob matches.
  bool Matches(absl::string_view path) const;

 private:
  explicit PathGlob(git_internal::PathPattern pattern)
      : pattern_(std::move(pattern)) {}

  git_internal::PathPattern pattern_;
};

}  // namespace ferret

#endif  // FERRET_GIT_PATH_GLOB_H_

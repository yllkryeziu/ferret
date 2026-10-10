// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Lazily loaded gitignore rules for arbitrary paths within a directory tree.
// Rules share git's path-pattern and wildmatch implementations with PathGlob.
// A lookup reads at most the ignore files along the path's ancestry; later
// lookups reuse them until their directory is invalidated.

#ifndef FERRET_GIT_IGNORE_H_
#define FERRET_GIT_IGNORE_H_

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "ferret/git/internal/path_pattern.h"

namespace ferret {

struct IgnoreOptions {
  // The root of the tree; paths passed to IsIgnored are relative to this root.
  std::string root;
  // Optional path to $GIT_DIR/info/exclude. No git configuration or
  // environment variables are consulted by the matcher.
  std::string info_exclude_path;
  // Optional path to the lower-priority core.excludesFile.
  std::string global_excludes_path;
  // Additional lowest-priority rules, useful for non-git directories.
  std::vector<std::string> default_excludes;
  // Also load .ferretignore in each directory, after its .gitignore.
  bool ferretignore = false;
  // Fold case when matching; the caller reads this from core.ignoreCase.
  bool ignore_case = false;
};

// A missing file returns nullopt; all other read failures return a Status.
// The callback must remain usable throughout the matcher's lifetime.
using IgnoreFileReader =
    absl::AnyInvocable<absl::StatusOr<std::optional<std::string>>(
        absl::string_view path)>;

// Reads regular files from disk. Missing files return nullopt. Symlinked
// .gitignore and .ferretignore files are not followed, as with git.
// Thread-safe.
absl::StatusOr<std::optional<std::string>> ReadIgnoreFile(
    absl::string_view path);

// Thread-compatible: concurrent callers must synchronize IsIgnored and
// Invalidate externally. The file reader is invoked only on cache misses.
class IgnoreMatcher final {
 public:
  explicit IgnoreMatcher(IgnoreOptions options,
                         IgnoreFileReader reader = ReadIgnoreFile);

  IgnoreMatcher(const IgnoreMatcher&) = delete;
  IgnoreMatcher& operator=(const IgnoreMatcher&) = delete;

  // Returns whether a root-relative path is excluded. The caller supplies
  // the path's actual type; an excluded ancestor cannot be re-included.
  // Paths must be nonempty, normalized, and must not end with '/'. The
  // lookup examines the matching rules at each path component.
  absl::StatusOr<bool> IsIgnored(absl::string_view rel_path, bool is_dir);

  // Drops the cached .gitignore and .ferretignore in `rel_dir` ("" for root).
  // The next lookup through that directory reloads both files. Invalidating
  // the root also reloads info/exclude and the global excludes file.
  void Invalidate(absl::string_view rel_dir);

 private:
  using Rules = std::vector<git_internal::PathPattern>;
  struct DirectoryRules {
    Rules gitignore;
    Rules ferretignore;
  };

  absl::StatusOr<const DirectoryRules*> RulesFor(absl::string_view dir);
  absl::StatusOr<bool> IsExcludedHere(absl::string_view path, bool is_dir,
                                      absl::string_view parent);

  IgnoreOptions options_;
  IgnoreFileReader reader_;
  Rules defaults_;
  std::optional<Rules> global_;
  std::optional<Rules> info_;
  std::map<std::string, DirectoryRules> directories_;
};

}  // namespace ferret

#endif  // FERRET_GIT_IGNORE_H_

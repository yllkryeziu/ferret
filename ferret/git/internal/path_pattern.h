// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// One .gitignore or .gitattributes pattern, matched the way git matches it.
//
// Git parses both file kinds' patterns with parse_path_pattern() and matches
// them with match_basename() / match_pathname() in dir.c:
//   * A leading '!' negates the pattern; a trailing '/' restricts it to
//     directories and is not part of the glob.
//   * A pattern without any other '/' matches the last path component at any
//     depth below the directory that holds the pattern file.
//   * Any other pattern is anchored to that directory; a leading '/' only
//     marks the anchoring. The glob is matched with WM_PATHNAME.
// Git first compares the literal prefix (up to the first glob character) and
// then runs wildmatch on the remainder as a new pattern, which changes the
// meaning of a '**' right after the prefix ("a/b**/c" behaves like
// "a/b" + "**/c"). This class reproduces that split.

#ifndef FERRET_GIT_INTERNAL_PATH_PATTERN_H_
#define FERRET_GIT_INTERNAL_PATH_PATTERN_H_

#include <cstddef>
#include <string>

#include "absl/strings/string_view.h"

namespace ferret::git_internal {

// A regular value type. Thread-compatible.
class PathPattern final {
 public:
  // Parses `text`, one pattern whose comment and trailing-space handling the
  // caller has already done, defined in a file in directory `base` (repo
  // relative, without a trailing '/'; empty for the root and for files that
  // apply to the whole tree).
  PathPattern(absl::string_view text, absl::string_view base);

  PathPattern(const PathPattern&) = default;
  PathPattern& operator=(const PathPattern&) = default;
  PathPattern(PathPattern&&) = default;
  PathPattern& operator=(PathPattern&&) = default;

  // True if the pattern started with '!'. Only .gitignore gives this a
  // meaning; git drops negative .gitattributes patterns.
  bool negative() const { return negative_; }

  // Returns whether the pattern matches the repo-relative `path`, which names
  // a directory iff `is_dir`. `casefold` is git's core.ignorecase.
  bool Matches(absl::string_view path, bool is_dir, bool casefold) const;

 private:
  bool MatchesBasename(absl::string_view basename, bool casefold) const;
  bool MatchesPathname(absl::string_view path, bool casefold) const;

  std::string glob_;  // Without the leading '!' and the trailing '/'.
  std::string base_;
  size_t literal_prefix_ = 0;  // Bytes of glob_ before the first glob char.
  bool negative_ = false;
  bool directories_only_ = false;
  bool basename_only_ = false;      // No '/' in glob_.
  bool star_then_literal_ = false;  // "*" followed by glob-free text.
};

}  // namespace ferret::git_internal

#endif  // FERRET_GIT_INTERNAL_PATH_PATTERN_H_

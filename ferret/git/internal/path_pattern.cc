// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/internal/path_pattern.h"

#include <cstddef>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"

#include "ferret/git/internal/wildmatch.h"

namespace ferret::git_internal {
namespace {

// git's simple_length(): the length of the prefix free of glob characters.
size_t LiteralPrefixLength(absl::string_view glob) {
  const size_t special = glob.find_first_of("*?[\\");
  return special == absl::string_view::npos ? glob.size() : special;
}

// git's fspathncmp() over equal-length ranges.
bool PathBytesEqual(absl::string_view a, absl::string_view b, bool casefold) {
  return casefold ? absl::EqualsIgnoreCase(a, b) : a == b;
}

}  // namespace

PathPattern::PathPattern(absl::string_view text, absl::string_view base)
    : base_(base) {
  negative_ = absl::ConsumePrefix(&text, "!");
  // Git measures the literal prefix before dropping the trailing '/' and then
  // clamps it; '/' is not a glob character, so measuring afterwards agrees.
  directories_only_ = absl::ConsumeSuffix(&text, "/");
  basename_only_ = !absl::StrContains(text, '/');
  literal_prefix_ = LiteralPrefixLength(text);
  star_then_literal_ = absl::StartsWith(text, "*") &&
                       LiteralPrefixLength(text.substr(1)) == text.size() - 1;
  glob_ = std::string(text);
}

bool PathPattern::Matches(absl::string_view path, bool is_dir,
                          bool casefold) const {
  if (directories_only_ && !is_dir) return false;
  if (basename_only_) {
    const size_t slash = path.rfind('/');
    return MatchesBasename(
        slash == absl::string_view::npos ? path : path.substr(slash + 1),
        casefold);
  }
  return MatchesPathname(path, casefold);
}

// git's match_basename().
bool PathPattern::MatchesBasename(absl::string_view basename,
                                  bool casefold) const {
  if (literal_prefix_ == glob_.size()) {
    return basename.size() == glob_.size() &&
           PathBytesEqual(glob_, basename, casefold);
  }
  if (star_then_literal_) {
    const absl::string_view suffix = absl::string_view(glob_).substr(1);
    return suffix.size() <= basename.size() &&
           PathBytesEqual(suffix,
                          basename.substr(basename.size() - suffix.size()),
                          casefold);
  }
  return Wildmatch(glob_, basename, {.pathname = false, .casefold = casefold});
}

// git's match_pathname().
bool PathPattern::MatchesPathname(absl::string_view path, bool casefold) const {
  absl::string_view glob = glob_;
  size_t prefix = literal_prefix_;
  if (absl::ConsumePrefix(&glob, "/")) {
    DCHECK_GT(prefix, 0u);
    --prefix;
  }
  if (path.size() < base_.size() + 1 ||
      (!base_.empty() && path[base_.size()] != '/') ||
      !PathBytesEqual(path.substr(0, base_.size()), base_, casefold)) {
    return false;
  }
  absl::string_view name = base_.empty() ? path : path.substr(base_.size() + 1);
  if (prefix > 0) {
    if (prefix > name.size() ||
        !PathBytesEqual(glob.substr(0, prefix), name.substr(0, prefix),
                        casefold)) {
      return false;
    }
    glob.remove_prefix(prefix);
    name.remove_prefix(prefix);
    if (glob.empty() && name.empty()) return true;
  }
  return Wildmatch(glob, name, {.pathname = true, .casefold = casefold});
}

}  // namespace ferret::git_internal

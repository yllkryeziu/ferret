// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Git's glob matcher: wildmatch(), as used by .gitignore and .gitattributes.
//
// Deciding "ignored" or "converted" differently from git would silently index
// build output, drop real source, or trust a wrong skip hint, so this is a
// statement-by-statement reimplementation of git's wildmatch.c (Rich Salz's
// matcher as extended by Wayne Davison for rsync), quirks included:
//   * '**' crosses directories only when it forms a whole path component
//     ("a/**/b", "**/b", "a/**"); elsewhere it behaves like '*'.
//   * An escaped character and the characters inside a bracket expression are
//     compared without case folding, even under casefold (so "[A]" does not
//     match "A" case-insensitively), except for ranges and [:upper:].
//   * Bracket expressions accept '!' or '^' for negation, a leading ']' as a
//     literal, and git's [:class:] names over ASCII only, independent of the
//     locale. An unterminated expression or an unknown class matches nothing.
// Behavior is pinned by git's t3070 vectors (wildmatch_test.cc) and by the
// differential test against `git check-ignore` in ignore_test.cc.
//
// Complexity: the matcher backtracks over '*' positions, but git's two abort
// codes stop a star from retrying once no later start can succeed, which
// keeps adversarial patterns such as "*a*a*a*b" from going exponential.
// Recursion depth is at most the number of runs of '*' in the pattern;
// callers bound the pattern length.

#ifndef FERRET_GIT_INTERNAL_WILDMATCH_H_
#define FERRET_GIT_INTERNAL_WILDMATCH_H_

#include "absl/strings/string_view.h"

namespace ferret::git_internal {

struct WildmatchOptions {
  // '*', '?' and bracket expressions never match '/', and '**' matches across
  // directories (git's WM_PATHNAME).
  bool pathname = false;
  // ASCII letters match regardless of case (git's WM_CASEFOLD).
  bool casefold = false;
};

// Returns true if all of `text` matches `pattern`. Neither may contain a NUL
// byte (DCHECKed): git compares C strings, so NUL would end both early.
bool Wildmatch(absl::string_view pattern, absl::string_view text,
               WildmatchOptions options);

}  // namespace ferret::git_internal

#endif  // FERRET_GIT_INTERNAL_WILDMATCH_H_

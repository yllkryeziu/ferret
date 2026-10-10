// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/internal/path_pattern.h"

#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

namespace ferret {
namespace {

using ::ferret::git_internal::PathPattern;

bool FileMatches(const PathPattern& pattern, absl::string_view path) {
  return pattern.Matches(path, /*is_dir=*/false, /*casefold=*/false);
}

TEST(PathPatternTest, PatternWithoutSlashMatchesBasenameAtAnyDepth) {
  const PathPattern pattern("*.o", "");
  EXPECT_TRUE(FileMatches(pattern, "a.o"));
  EXPECT_TRUE(FileMatches(pattern, "x/y/a.o"));
  EXPECT_FALSE(FileMatches(pattern, "a.oo"));
  EXPECT_FALSE(FileMatches(pattern, "a.o/b"));
}

TEST(PathPatternTest, PatternWithSlashIsAnchoredToItsDirectory) {
  const PathPattern pattern("doc/*.txt", "");
  EXPECT_TRUE(FileMatches(pattern, "doc/a.txt"));
  EXPECT_FALSE(FileMatches(pattern, "doc/sub/a.txt"));
  EXPECT_FALSE(FileMatches(pattern, "x/doc/a.txt"));

  const PathPattern leading_slash("/a", "");
  EXPECT_TRUE(FileMatches(leading_slash, "a"));
  EXPECT_FALSE(FileMatches(leading_slash, "b/a"));
}

TEST(PathPatternTest, BaseDirectoryPrefixesAnchoredPatterns) {
  const PathPattern anchored("/b", "a");
  EXPECT_TRUE(FileMatches(anchored, "a/b"));
  EXPECT_FALSE(FileMatches(anchored, "b"));
  EXPECT_FALSE(FileMatches(anchored, "ab/b"));
  EXPECT_FALSE(FileMatches(anchored, "a/c/b"));

  // Basename patterns ignore the base: the caller consults a directory's
  // patterns only for paths below it.
  EXPECT_TRUE(FileMatches(PathPattern("b", "a"), "a/c/b"));
}

TEST(PathPatternTest, TrailingSlashRestrictsToDirectories) {
  const PathPattern pattern("build/", "");
  EXPECT_TRUE(pattern.Matches("build", /*is_dir=*/true, false));
  EXPECT_TRUE(pattern.Matches("x/build", /*is_dir=*/true, false));
  EXPECT_FALSE(pattern.Matches("build", /*is_dir=*/false, false));
}

TEST(PathPatternTest, LeadingBangNegates) {
  const PathPattern pattern("!keep.o", "");
  EXPECT_TRUE(pattern.negative());
  EXPECT_TRUE(FileMatches(pattern, "keep.o"));
  EXPECT_FALSE(PathPattern(R"(\!keep.o)", "").negative());
  EXPECT_TRUE(FileMatches(PathPattern(R"(\!keep.o)", ""), "!keep.o"));
}

TEST(PathPatternTest, CasefoldAppliesToLiteralsAndBase) {
  const PathPattern pattern("/Src/*.C", "Lib");
  EXPECT_FALSE(pattern.Matches("lib/src/a.c", false, /*casefold=*/false));
  EXPECT_TRUE(pattern.Matches("lib/src/a.c", false, /*casefold=*/true));
  EXPECT_TRUE(PathPattern("README", "").Matches("readme", false, true));
}

TEST(PathPatternTest, DoubleStarAfterLiteralPrefixStartsANewPattern) {
  // Git compares "a/b" literally and wildmatches "**/c" on its own, so the
  // "**/" counts as leading; verified against git 2.49 check-ignore.
  const PathPattern pattern("a/b**/c", "");
  EXPECT_TRUE(FileMatches(pattern, "a/bx/y/c"));
  EXPECT_TRUE(FileMatches(pattern, "a/bx/c"));
  EXPECT_FALSE(FileMatches(pattern, "a/x/c"));
}

TEST(PathPatternTest, EmptyPatternMatchesNothing) {
  EXPECT_FALSE(FileMatches(PathPattern("", ""), "a"));
  EXPECT_FALSE(FileMatches(PathPattern("!", ""), "a"));
  EXPECT_FALSE(PathPattern("/", "").Matches("a", /*is_dir=*/true, false));
}

}  // namespace
}  // namespace ferret

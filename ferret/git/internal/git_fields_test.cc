// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/internal/git_fields.h"

#include <optional>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ferret {
namespace {

using ::ferret::git_internal::IsNormalizedRepoPath;
using ::ferret::git_internal::ParseGitMode;
using ::testing::Optional;

TEST(GitFieldsTest, AcceptsNormalizedPaths) {
  EXPECT_TRUE(IsNormalizedRepoPath("a"));
  EXPECT_TRUE(IsNormalizedRepoPath("src/main.cc"));
  EXPECT_TRUE(IsNormalizedRepoPath("with space/new\nline/\xC3\xA9.txt"));
  EXPECT_TRUE(IsNormalizedRepoPath(".gitignore"));
  EXPECT_TRUE(IsNormalizedRepoPath("a/.github/x"));
  EXPECT_TRUE(IsNormalizedRepoPath("..."));
}

TEST(GitFieldsTest, RejectsEmptyAndSlashes) {
  EXPECT_FALSE(IsNormalizedRepoPath(""));
  EXPECT_FALSE(IsNormalizedRepoPath("/etc/passwd"));
  EXPECT_FALSE(IsNormalizedRepoPath("dir/"));
  EXPECT_FALSE(IsNormalizedRepoPath("a//b"));
}

TEST(GitFieldsTest, RejectsDotComponents) {
  EXPECT_FALSE(IsNormalizedRepoPath("."));
  EXPECT_FALSE(IsNormalizedRepoPath("../x"));
  EXPECT_FALSE(IsNormalizedRepoPath("a/./b"));
  EXPECT_FALSE(IsNormalizedRepoPath("a/.."));
}

TEST(GitFieldsTest, RejectsGitDirectoryInAnyCase) {
  EXPECT_FALSE(IsNormalizedRepoPath(".git"));
  EXPECT_FALSE(IsNormalizedRepoPath(".git/config"));
  EXPECT_FALSE(IsNormalizedRepoPath("sub/.GIT/hooks/pre-commit"));
  EXPECT_FALSE(IsNormalizedRepoPath("sub/.Git"));
}

TEST(GitFieldsTest, RejectsNul) {
  EXPECT_FALSE(IsNormalizedRepoPath(std::string("a\0b", 3)));
}

TEST(GitFieldsTest, ParsesSixOctalDigitModes) {
  EXPECT_THAT(ParseGitMode("100644"), Optional(0100644u));
  EXPECT_THAT(ParseGitMode("160000"), Optional(0160000u));
  EXPECT_THAT(ParseGitMode("000000"), Optional(0u));
}

TEST(GitFieldsTest, RejectsMalformedModes) {
  EXPECT_EQ(ParseGitMode(""), std::nullopt);
  EXPECT_EQ(ParseGitMode("10644"), std::nullopt);
  EXPECT_EQ(ParseGitMode("1006440"), std::nullopt);
  EXPECT_EQ(ParseGitMode("100648"), std::nullopt);
  EXPECT_EQ(ParseGitMode("-10644"), std::nullopt);
}

}  // namespace
}  // namespace ferret

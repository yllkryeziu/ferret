// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/worktree_files.h"

#include <unistd.h>

#include <optional>
#include <string>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/git/testing/git_repo_builder.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"

namespace ferret {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::ferret_testing::GitRepoBuilder;
using ::ferret_testing::TempDir;
using ::testing::ElementsAre;

class ListWorktreeFilesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    FERRET_ASSERT_OK_AND_ASSIGN(repo_, GitRepoBuilder::Init(temp_dir_, "repo"));
  }

  GitRepoBuilder& repo() {
    CHECK(repo_.has_value());
    return *repo_;
  }

  TempDir temp_dir_;
  std::optional<GitRepoBuilder> repo_;
};

TEST_F(ListWorktreeFilesTest, ListsTrackedAndUntrackedButNotIgnored) {
  ABSL_ASSERT_OK(repo().WriteFile(".gitignore", "*.log\nbuild/\n"));
  ABSL_ASSERT_OK(repo().WriteFile("src/main.cc", "int main() {}\n"));
  ABSL_ASSERT_OK(repo().WriteFile("README", "hi\n"));
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().Commit("initial"));
  ABSL_ASSERT_OK(repo().WriteFile("notes.txt", "untracked\n"));
  ABSL_ASSERT_OK(repo().WriteFile("debug.log", "ignored\n"));
  ABSL_ASSERT_OK(repo().WriteFile("build/out.o", "ignored\n"));

  EXPECT_THAT(ListWorktreeFiles(repo().root()),
              IsOkAndHolds(ElementsAre(".gitignore", "README", "notes.txt",
                                       "src/main.cc")));
}

TEST_F(ListWorktreeFilesTest, OmitsSymlinksGitlinksAndSkipWorktree) {
  ABSL_ASSERT_OK(repo().WriteFile("a.txt", "a\n"));
  ABSL_ASSERT_OK(repo().WriteFile("sparse.txt", "s\n"));
  ABSL_ASSERT_OK(repo().Symlink("a.txt", "link"));
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().AddGitlink("vendor/sub",
                                   "0123456789abcdef0123456789abcdef01234567"));
  ABSL_ASSERT_OK(repo().Commit("initial"));
  ABSL_ASSERT_OK(repo().SetSkipWorktree("sparse.txt"));

  EXPECT_THAT(ListWorktreeFiles(repo().root()),
              IsOkAndHolds(ElementsAre("a.txt")));
}

TEST_F(ListWorktreeFilesTest, KeepsUnusualPathsVerbatim) {
  const std::string odd = "dir with space/new\nline-\xC3\xA9.txt";
  ABSL_ASSERT_OK(repo().WriteFile(odd, "x\n"));
  ABSL_ASSERT_OK(repo().Add());
  EXPECT_THAT(ListWorktreeFiles(repo().root()), IsOkAndHolds(ElementsAre(odd)));
}

TEST_F(ListWorktreeFilesTest, EmptyRepositoryHasNoFiles) {
  EXPECT_THAT(ListWorktreeFiles(repo().root()),
              IsOkAndHolds(::testing::IsEmpty()));
}

TEST(ListWorktreeFilesStandaloneTest, NonRepositoryIsFailedPrecondition) {
  TempDir temp_dir;
  EXPECT_THAT(ListWorktreeFiles(temp_dir.path()),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(ListDirectoryFilesTest, PrunesIgnoredDirectoriesAndSkipsSymlinks) {
  TempDir temp;
  ABSL_ASSERT_OK(
      temp.WriteFile(".gitignore", "hidden/\n!hidden/keep\n*.log\n"));
  ABSL_ASSERT_OK(temp.WriteFile(".ferretignore", "!keep.log\n"));
  ABSL_ASSERT_OK(temp.WriteFile("hidden/keep", "x"));
  ABSL_ASSERT_OK(temp.WriteFile("keep.log", "x"));
  ABSL_ASSERT_OK(temp.WriteFile("other.log", "x"));
  ABSL_ASSERT_OK(temp.WriteFile("src/main.cc", "x"));
  ABSL_ASSERT_OK(temp.WriteFile("node_modules/pkg/index.js", "x"));
  ABSL_ASSERT_OK(temp.WriteFile("out/result", "x"));
  ABSL_ASSERT_OK(temp.WriteFile("bazel-bin/result", "x"));
  ABSL_ASSERT_OK(temp.WriteFile(".git/config", "x"));
  ABSL_ASSERT_OK(temp.WriteFile("idx/segment", "x"));
  ABSL_ASSERT_OK(temp.WriteFile("custom/a", "x"));
  ASSERT_EQ(symlink(temp.Join("src").c_str(), temp.Join("alias").c_str()), 0);
  ASSERT_EQ(
      symlink(temp.Join("src/main.cc").c_str(), temp.Join("link.cc").c_str()),
      0);
  EXPECT_THAT(
      ListDirectoryFiles(temp.path(), {.index_directory = "idx",
                                       .default_excludes = {"custom/"}}),
      IsOkAndHolds(ElementsAre(".ferretignore", ".gitignore", "keep.log",
                               "src/main.cc")));
}

}  // namespace
}  // namespace ferret

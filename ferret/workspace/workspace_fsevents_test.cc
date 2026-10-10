// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include <unistd.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status_matchers.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/git/testing/git_repo_builder.h"
#include "ferret/search/search_types.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/workspace/workspace.h"

namespace ferret {
namespace {

using ::absl_testing::IsOkAndHolds;

void ExpectSync(Workspace& workspace, WorktreeId tree, const std::string& token,
                const std::vector<std::string>& paths) {
  SearchRequest request;
  request.pattern = token;
  request.context_lines = 0;
  auto result = workspace.Search(
      tree, request,
      {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(10)});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_FALSE(result->stale);
  std::vector<std::string> found;
  for (const FileResult& file : result->response.files) {
    found.push_back(file.path);
  }
  EXPECT_EQ(found, paths) << token;
}

TEST(WorkspaceFseventsTest, WritesRenamesAndDeletesWithoutNotify) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      ferret_testing::GitRepoBuilder repo,
      ferret_testing::GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("base.cc", "oldtoken\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  FERRET_ASSERT_OK_AND_ASSIGN(ferret_testing::GitRepoBuilder other,
                              repo.AddWorktree("other"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("index")));
  FERRET_ASSERT_OK_AND_ASSIGN(
      WorktreeId tree, workspace->Register(other.root(), {.watch = true}));
  EXPECT_TRUE(workspace->Stats().last_watcher_error.ok())
      << workspace->Stats().last_watcher_error;
  ABSL_ASSERT_OK(other.WriteFile("base.cc", "editedtoken\n"));
  ExpectSync(*workspace, tree, "editedtoken", {"base.cc"});
  ABSL_ASSERT_OK(other.WriteFile("folder/new.cc", "createdtoken\n"));
  ExpectSync(*workspace, tree, "createdtoken", {"folder/new.cc"});
  ABSL_ASSERT_OK(other.Rename("folder/new.cc", "folder/renamed.cc"));
  ExpectSync(*workspace, tree, "createdtoken", {"folder/renamed.cc"});
  ABSL_ASSERT_OK(other.Rename("folder", "moved"));
  ExpectSync(*workspace, tree, "createdtoken", {"moved/renamed.cc"});
  ABSL_ASSERT_OK(other.DeleteFile("moved/renamed.cc"));
  ExpectSync(*workspace, tree, "createdtoken", {});
  ASSERT_EQ(rmdir(other.Path("moved").c_str()), 0);
  ExpectSync(*workspace, tree, "createdtoken", {});
  ABSL_ASSERT_OK(other.DeleteFile("base.cc"));
  ExpectSync(*workspace, tree, "editedtoken", {});
  ABSL_ASSERT_OK(workspace->Unregister(tree));
}

// In a linked worktree `.git` is a file, so the sync cookies live in
// `.ferret-watch/` inside the worktree, where agents run `git status` and
// `git clean`. Once `git clean -d` had deleted it, every serving search in
// an early four-agent run waited out its timeout and came back stale.
TEST(WorkspaceFseventsTest, CookieDirectorySurvivesGitCleanAndDeletion) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      ferret_testing::GitRepoBuilder repo,
      ferret_testing::GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("base.cc", "oldtoken\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  FERRET_ASSERT_OK_AND_ASSIGN(ferret_testing::GitRepoBuilder other,
                              repo.AddWorktree("other"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("index")));
  FERRET_ASSERT_OK_AND_ASSIGN(
      WorktreeId tree, workspace->Register(other.root(), {.watch = true}));
  const std::string cookies = other.Path(".ferret-watch");
  ExpectSync(*workspace, tree, "oldtoken", {"base.cc"});
  EXPECT_THAT(other.Git({"status", "--porcelain", "--untracked-files=all"}),
              IsOkAndHolds(""));
  ABSL_ASSERT_OK(other.Git({"clean", "-f", "-d"}).status());
  EXPECT_EQ(access(cookies.c_str(), F_OK), 0) << "git clean -d removed it";

  // `git clean -dx` or `rm -rf` still deletes it.
  ABSL_ASSERT_OK(other.Git({"clean", "-f", "-d", "-x"}).status());
  ASSERT_NE(access(cookies.c_str(), F_OK), 0);
  ABSL_ASSERT_OK(other.WriteFile("base.cc", "aftertoken\n"));
  ExpectSync(*workspace, tree, "aftertoken", {"base.cc"});
  EXPECT_EQ(workspace->Stats().watcher_failures, 0u)
      << workspace->Stats().last_watcher_error;
  ABSL_ASSERT_OK(workspace->Unregister(tree));
  EXPECT_NE(access(cookies.c_str(), F_OK), 0) << "left behind";
}

}  // namespace
}  // namespace ferret

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/workspace/workspace.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/base/testing/fault_injection_env.h"
#include "ferret/git/blob_hash.h"
#include "ferret/git/oid.h"
#include "ferret/git/testing/git_repo_builder.h"
#include "ferret/git/worktree_files.h"
#include "ferret/index/index.h"
#include "ferret/index/rank_key.h"
#include "ferret/search/search_types.h"
#include "ferret/symbols/symbol_extractor.h"
#include "ferret/symbols/symbol_kind.h"
#include "ferret/testing/brute_force_search.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/testing/test_seed.h"
#include "ferret/text/eligibility.h"
#include "ferret/text/language.h"
#include "ferret/text/trigram_extractor.h"

namespace ferret {
namespace {

using ::absl_testing::StatusIs;
using ::ferret_testing::GitRepoBuilder;
using ::ferret_testing::TempDir;

// Long enough that a kNotify search is stale only if its generation is never
// applied, even under sanitizers.
constexpr absl::Duration kWait = absl::Seconds(30);

SearchRequest Request(absl::string_view pattern) {
  SearchRequest request;
  request.pattern = std::string(pattern);
  request.context_lines = 0;
  request.max_files = 1000;
  return request;
}

std::shared_ptr<WorkspaceReadHook> Hook(WorkspaceReadHook hook) {
  return std::make_shared<WorkspaceReadHook>(std::move(hook));
}

std::vector<std::string> Paths(const SearchResponse& response) {
  std::vector<std::string> paths;
  paths.reserve(response.files.size());
  for (const FileResult& file : response.files) paths.push_back(file.path);
  std::sort(paths.begin(), paths.end());
  return paths;
}

void ExpectPaths(Workspace& workspace, WorktreeId tree,
                 absl::string_view pattern,
                 const std::vector<std::string>& expected,
                 uint64_t generation = 0) {
  auto result = workspace.Search(tree, Request(pattern),
                                 {.mode = SearchConsistencyMode::kNotify,
                                  .min_generation = generation,
                                  .timeout = kWait});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_FALSE(result->stale);
  EXPECT_GE(result->generation, generation);
  EXPECT_EQ(Paths(result->response), expected) << pattern;
}

absl::StatusOr<uint64_t> NotifyPaths(Workspace& workspace, WorktreeId tree,
                                     const std::vector<std::string>& paths) {
  return workspace.Notify(tree, paths);
}

TEST(WorkspaceTest, WorktreeStatusTracksViewAndGenerations) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("a.cc", "sharedneedle\n"));
  ABSL_ASSERT_OK(repo.WriteFile("b.cc", "sharedneedle\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("idx")));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id, workspace->Register(repo.root()));
  FERRET_ASSERT_OK_AND_ASSIGN(WorkspaceWorktreeStatus first,
                              workspace->WorktreeStatus(id));
  EXPECT_EQ(first.view_paths, 2);
  EXPECT_EQ(first.view_blobs, 1);
  EXPECT_EQ(first.assigned_generation, uint64_t{0});
  EXPECT_FALSE(first.watcher_running);
  EXPECT_EQ(first.index_bytes, 0);  // Small content remains in the memtable.
  ABSL_ASSERT_OK(repo.DeleteFile("a.cc"));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t generation,
                              NotifyPaths(*workspace, id, {"a.cc"}));
  ExpectPaths(*workspace, id, "sharedneedle", {"b.cc"}, generation);
  FERRET_ASSERT_OK_AND_ASSIGN(WorkspaceWorktreeStatus second,
                              workspace->WorktreeStatus(id));
  EXPECT_EQ(second.view_paths, 1);
  EXPECT_EQ(second.view_blobs, 1);
  EXPECT_GE(second.applied_generation, generation);
  EXPECT_GE(second.assigned_generation, generation);
  ABSL_ASSERT_OK(workspace->Unregister(id));
  EXPECT_THAT(workspace->WorktreeStatus(id),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(WorkspaceTest, BulkBytesReadCountOnlyWhatARegistrationReads) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  const std::string a(1000, 'a');
  const std::string b(3000, 'b');
  ABSL_ASSERT_OK(repo.WriteFile("a.cc", absl::StrCat(a, "\n")));
  ABSL_ASSERT_OK(repo.WriteFile("b.cc", absl::StrCat(b, "\n")));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder same, repo.AddWorktree("same"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("idx")));
  ASSERT_TRUE(workspace->Register(repo.root()).ok());
  // First registration: both files are read for their identity, then again to
  // be written.
  WorkspaceStats stats = workspace->Stats();
  EXPECT_EQ(stats.bulk_bytes_read, uint64_t{4002});
  EXPECT_EQ(stats.bulk_bytes_reread, uint64_t{4002});
  // The same commit in another worktree reads nothing: every blob is known.
  ASSERT_TRUE(workspace->Register(same.root()).ok());
  stats = workspace->Stats();
  EXPECT_EQ(stats.bulk_bytes_read, uint64_t{4002});
  EXPECT_EQ(stats.bulk_bytes_reread, uint64_t{4002});
  // A commit that edits one file reads that file, and only it.
  const std::string edited = absl::StrCat(std::string(500, 'c'), "\n");
  ABSL_ASSERT_OK(repo.WriteFile("a.cc", edited));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("edit"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder other, repo.AddWorktree("other"));
  ASSERT_TRUE(workspace->Register(other.root()).ok());
  stats = workspace->Stats();
  EXPECT_EQ(stats.bulk_bytes_read, uint64_t{4002 + 501});
  EXPECT_EQ(stats.bulk_bytes_reread, uint64_t{4002 + 501});
}

TEST(WorkspaceTest, FlushIndexWritesTheMemtableToDisk) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("a.cc", "flushword\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("idx")));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id, workspace->Register(repo.root()));
  FERRET_ASSERT_OK_AND_ASSIGN(WorkspaceWorktreeStatus before,
                              workspace->WorktreeStatus(id));
  EXPECT_EQ(before.index_bytes, 0);
  ABSL_ASSERT_OK(workspace->FlushIndex());
  FERRET_ASSERT_OK_AND_ASSIGN(WorkspaceWorktreeStatus after,
                              workspace->WorktreeStatus(id));
  EXPECT_GT(after.index_bytes, 0);
  ExpectPaths(*workspace, id, "flushword", {"a.cc"});
}

TEST(WorkspaceTest, SharedGitViewsAndPlainRoot) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("a.cc", "sharedneedle\n"));
  ABSL_ASSERT_OK(repo.WriteFile("b.cc", "sharedneedle\n"));
  ABSL_ASSERT_OK(repo.WriteFile("dir/inner.cc", "innerneedle\n"));
  ABSL_ASSERT_OK(repo.WriteFile(".gitignore", "*.log\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder other, repo.AddWorktree("other"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("index")));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId a, workspace->Register(repo.root()));
  const WorkspaceStats first = workspace->Stats();
  EXPECT_GE(first.hint_reads, uint64_t{4});
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId b, workspace->Register(other.root()));
  const WorkspaceStats second = workspace->Stats();
  EXPECT_EQ(second.bulk_files_read, first.bulk_files_read);
  EXPECT_EQ(second.hint_reads, first.hint_reads);
  EXPECT_EQ(second.hints_reused - first.hints_reused, uint64_t{4});
  ExpectPaths(*workspace, a, "sharedneedle", {"a.cc", "b.cc"});
  ExpectPaths(*workspace, b, "sharedneedle", {"a.cc", "b.cc"});
  EXPECT_THAT(workspace->Register(repo.root()),
              StatusIs(absl::StatusCode::kAlreadyExists));
  EXPECT_THAT(workspace->Register(repo.Path("dir")),
              StatusIs(absl::StatusCode::kInvalidArgument));

  ABSL_ASSERT_OK(repo.WriteFile("a.cc", "changedneedle\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t edit,
                              NotifyPaths(*workspace, a, {"a.cc"}));
  ExpectPaths(*workspace, a, "changedneedle", {"a.cc"}, edit);
  ExpectPaths(*workspace, b, "changedneedle", {});
  ABSL_ASSERT_OK(repo.Rename("b.cc", "c.cc"));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t rename,
                              NotifyPaths(*workspace, a, {"b.cc", "c.cc"}));
  ExpectPaths(*workspace, a, "sharedneedle", {"c.cc"}, rename);
  ABSL_ASSERT_OK(repo.WriteFile("ignored.log", "ignoredneedle\n"));
  ABSL_ASSERT_OK(repo.WriteFile(".git/info/stray", "gitdirneedle\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      uint64_t ignore,
      NotifyPaths(*workspace, a, {"ignored.log", ".git/info/stray"}));
  ExpectPaths(*workspace, a, "ignoredneedle", {}, ignore);
  ExpectPaths(*workspace, a, "gitdirneedle", {}, ignore);
  ABSL_ASSERT_OK(repo.Symlink("a.cc", "link.cc"));
  ABSL_ASSERT_OK(repo.Symlink("dir", "linkdir"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      uint64_t link,
      NotifyPaths(*workspace, a, {"link.cc", "linkdir/inner.cc"}));
  ExpectPaths(*workspace, a, "changedneedle", {"a.cc"}, link);
  ExpectPaths(*workspace, a, "innerneedle", {"dir/inner.cc"}, link);
  EXPECT_THAT(workspace->Notify(a, std::vector<std::string>{"x.cc", "../up"}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  ABSL_ASSERT_OK(repo.DeleteFile("a.cc"));
  ABSL_ASSERT_OK(temp.MakeDirs("repo/a.cc"));
  ABSL_ASSERT_OK(repo.WriteFile("a.cc/child.cc", "directoryneedle\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t directory,
                              NotifyPaths(*workspace, a, {"a.cc"}));
  ExpectPaths(*workspace, a, "directoryneedle", {"a.cc/child.cc"}, directory);
  ExpectPaths(*workspace, a, "changedneedle", {});
  FERRET_ASSERT_OK_AND_ASSIGN(
      const uint64_t applied,
      workspace->AwaitGeneration(a, directory, absl::ZeroDuration()));
  EXPECT_EQ(applied, directory);
  // The directory becomes a file again, and then dir moves out of the
  // worktree: the rescans these schedule delete the files that were below.
  ABSL_ASSERT_OK(repo.DeleteFile("a.cc/child.cc"));
  ASSERT_EQ(rmdir(repo.Path("a.cc").c_str()), 0);
  ABSL_ASSERT_OK(repo.WriteFile("a.cc", "filebackneedle\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t file_again,
                              NotifyPaths(*workspace, a, {"a.cc"}));
  ExpectPaths(*workspace, a, "needle", {"a.cc", "c.cc", "dir/inner.cc"},
              file_again);
  ExpectPaths(*workspace, a, "filebackneedle", {"a.cc"});
  ABSL_ASSERT_OK(
      Env::Default()->RenameFile(repo.Path("dir"), temp.Join("outside")));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t moved_out,
                              NotifyPaths(*workspace, a, {"dir"}));
  ExpectPaths(*workspace, a, "needle", {"a.cc", "c.cc"}, moved_out);

  ABSL_ASSERT_OK(temp.WriteFile("plain/x.cc", "plainneedle\n"));
  ABSL_ASSERT_OK(temp.WriteFile("plain/node_modules/y.cc", "plainneedle\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId plain,
                              workspace->Register(temp.Join("plain")));
  ExpectPaths(*workspace, plain, "plainneedle", {"x.cc"});
  ABSL_ASSERT_OK(workspace->Unregister(b));
  EXPECT_THAT(workspace->Search(b, Request("sharedneedle")),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(workspace->Unregister(b), StatusIs(absl::StatusCode::kNotFound));
  ExpectPaths(*workspace, a, "sharedneedle", {"c.cc"});
}

// Git lists nothing inside gitlinks and nested repositories; neither do
// registration, Notify, and Rescan.
TEST(WorkspaceTest, NestedRepositoriesStayOutOfViews) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("top.cc", "outerword\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.AddGitlink("vendor/sub",
                                 "0123456789abcdef0123456789abcdef01234567"));
  ABSL_ASSERT_OK(repo.Commit("base"));
  ABSL_ASSERT_OK(repo.WriteFile("vendor/sub/inside.cc", "nestedword\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder nested,
                              GitRepoBuilder::Init(temp, "repo/nested"));
  ABSL_ASSERT_OK(nested.WriteFile("inner.cc", "nestedword\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("index")));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id, workspace->Register(repo.root()));
  ExpectPaths(*workspace, id, "nestedword", {});
  // A repository created after the scan is known by its `.git` entry.
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder late,
                              GitRepoBuilder::Init(temp, "repo/lib/late"));
  ABSL_ASSERT_OK(late.WriteFile("deep/late.cc", "nestedword\n"));
  ABSL_ASSERT_OK(repo.WriteFile("lib/own.cc", "nestedword outer\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      uint64_t generation,
      NotifyPaths(*workspace, id,
                  {"vendor/sub/inside.cc", "nested/inner.cc",
                   "lib/late/deep/late.cc", "lib/own.cc"}));
  ExpectPaths(*workspace, id, "nestedword", {"lib/own.cc"}, generation);
  // A nested repository's own directory schedules no rescan.
  const absl::Duration scanned = workspace->Stats().scan_time;
  FERRET_ASSERT_OK_AND_ASSIGN(generation,
                              NotifyPaths(*workspace, id, {"nested"}));
  ExpectPaths(*workspace, id, "nestedword", {"lib/own.cc"}, generation);
  EXPECT_EQ(workspace->Stats().scan_time, scanned);
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t rescan, workspace->Rescan(id));
  ExpectPaths(*workspace, id, "nestedword", {"lib/own.cc"}, rescan);
  ExpectPaths(*workspace, id, "outerword", {"top.cc"});
}

// On a case-insensitive volume another spelling opens the same file, but it
// is not a path of the worktree. Elsewhere it does not exist.
TEST(WorkspaceTest, NotifyIgnoresOtherSpellings) {
  TempDir temp;
  ABSL_ASSERT_OK(temp.WriteFile("plain/Mixed.cc", "spellingword\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("index")));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id,
                              workspace->Register(temp.Join("plain")));
  ABSL_ASSERT_OK(temp.WriteFile("plain/Mixed.cc", "spellingword changed\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      uint64_t generation,
      NotifyPaths(*workspace, id, {"mixed.cc", "MIXED.CC", "Mixed.cc"}));
  ExpectPaths(*workspace, id, "spellingword", {"Mixed.cc"}, generation);
  ExpectPaths(*workspace, id, "changed", {"Mixed.cc"}, generation);
  EXPECT_EQ(workspace->Stats().interactive_files_read, uint64_t{1});
}

// A failed flush makes every later Index::Add fail until Flush retries it.
// The workspace retries, rereads what it could not add, and holds back the
// generations until then, so kNotify reports stale instead of old content.
TEST(WorkspaceTest, IngestRecoversFromTransientFlushFailure) {
  TempDir temp;
  ABSL_ASSERT_OK(temp.WriteFile("plain/seed.cc", "seedtoken\n"));
  ABSL_ASSERT_OK(temp.WriteFile("later/bulk.cc", "bulktoken\n"));
  ferret_testing::FaultInjectionEnv env(Env::Default());
  WorkspaceOptions options;
  options.index.compaction.enabled = false;
  options.index.memtable.flush_docs = 1;
  options.ingest_retry_initial_delay = absl::Milliseconds(1);
  options.ingest_retry_max_delay = absl::Milliseconds(20);
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(env, temp.Join("index"), std::move(options)));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id,
                              workspace->Register(temp.Join("plain")));
  env.FailAfter(0, absl::ResourceExhaustedError("injected full disk"));
  // Each document fills a memtable whose flush fails; a later Add fails.
  std::string path;
  std::string token;
  uint64_t generation = 0;
  for (int i = 0; workspace->Stats().ingest_failures == 0; ++i) {
    ASSERT_LT(i, 1000);
    path = absl::StrCat("f", i, ".cc");
    token = absl::StrCat("failtoken", i, "x");
    ABSL_ASSERT_OK(temp.WriteFile(absl::StrCat("plain/", path), token));
    FERRET_ASSERT_OK_AND_ASSIGN(generation,
                                NotifyPaths(*workspace, id, {path}));
    const absl::Time deadline = absl::Now() + kWait;
    while (workspace->Stats().ingest_failures == 0 && absl::Now() < deadline) {
      FERRET_ASSERT_OK_AND_ASSIGN(
          const uint64_t applied,
          workspace->AwaitGeneration(id, generation, absl::Milliseconds(1)));
      if (applied >= generation) break;
    }
  }
  FERRET_ASSERT_OK_AND_ASSIGN(
      WorkspaceSearchResult stale,
      workspace->Search(id, Request(token),
                        {.mode = SearchConsistencyMode::kNotify,
                         .min_generation = generation,
                         .timeout = absl::ZeroDuration()}));
  EXPECT_TRUE(stale.stale);
  EXPECT_THAT(Paths(stale.response), ::testing::IsEmpty());
  EXPECT_THAT(workspace->Stats().last_ingest_error,
              StatusIs(absl::StatusCode::kResourceExhausted));
  // The disk frees up: the next retry succeeds and adds the file.
  env.ClearFault();
  ExpectPaths(*workspace, id, token, {path}, generation);
  EXPECT_GT(workspace->Stats().ingest_retries, uint64_t{0});
  ABSL_ASSERT_OK(temp.WriteFile("plain/after.cc", "aftertoken\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t after,
                              NotifyPaths(*workspace, id, {"after.cc"}));
  ExpectPaths(*workspace, id, "aftertoken", {"after.cc"}, after);
  // Small registrations go through Index::Add(kBulk) too.
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId later,
                              workspace->Register(temp.Join("later")));
  ExpectPaths(*workspace, later, "bulktoken", {"bulk.cc"});
}

TEST(WorkspaceTest, ContentMovesBetweenWorktreesAndPaths) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("a.cc", "dupneedle\n"));
  ABSL_ASSERT_OK(repo.WriteFile("b.cc", "dupneedle\n"));
  ABSL_ASSERT_OK(
      repo.WriteFile("bin.dat", std::string("binary\0needle\n", 14)));
  ABSL_ASSERT_OK(repo.WriteFile(".gitignore", "*.log\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder other, repo.AddWorktree("other"));
  ABSL_ASSERT_OK(repo.WriteFile("notes.txt", "untrackedneedle\n"));
  ABSL_ASSERT_OK(repo.WriteFile("debug.log", "untrackedneedle\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("index")));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId a, workspace->Register(repo.root()));
  ExpectPaths(*workspace, a, "dupneedle", {"a.cc", "b.cc"});
  ExpectPaths(*workspace, a, "untrackedneedle", {"notes.txt"});
  ExpectPaths(*workspace, a, "binary", {});
  const WorkspaceStats first = workspace->Stats();
  EXPECT_EQ(first.skipped[static_cast<size_t>(SkipReason::kContainsNul)],
            uint64_t{1});
  // One document per distinct eligible content: a.cc and b.cc share one.
  EXPECT_EQ(first.bulk_bytes_ingested, uint64_t{10 + 16 + 6});

  // The ineligible hint is known by its Oid, so nothing is read.
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId b, workspace->Register(other.root()));
  const WorkspaceStats second = workspace->Stats();
  EXPECT_EQ(second.bulk_files_read, first.bulk_files_read);
  EXPECT_EQ(second.skipped[static_cast<size_t>(SkipReason::kContainsNul)],
            uint64_t{2});
  ExpectPaths(*workspace, b, "untrackedneedle", {});

  ABSL_ASSERT_OK(repo.WriteFile("moved.cc", "movingneedle\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t added,
                              NotifyPaths(*workspace, a, {"moved.cc"}));
  ExpectPaths(*workspace, a, "movingneedle", {"moved.cc"}, added);
  ExpectPaths(*workspace, b, "movingneedle", {});
  const uint64_t ingested = workspace->Stats().interactive_bytes_ingested;
  ABSL_ASSERT_OK(repo.DeleteFile("moved.cc"));
  ABSL_ASSERT_OK(other.WriteFile("sub/elsewhere.cc", "movingneedle\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t removed,
                              NotifyPaths(*workspace, a, {"moved.cc"}));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t moved,
                              NotifyPaths(*workspace, b, {"sub/elsewhere.cc"}));
  ExpectPaths(*workspace, a, "movingneedle", {}, removed);
  ExpectPaths(*workspace, b, "movingneedle", {"sub/elsewhere.cc"}, moved);
  EXPECT_EQ(workspace->Stats().interactive_bytes_ingested, ingested);

  ABSL_ASSERT_OK(other.WriteFile("c1.cc", "twinneedle\n"));
  ABSL_ASSERT_OK(other.WriteFile("c2.cc", "twinneedle\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t twins,
                              NotifyPaths(*workspace, b, {"c2.cc", "c1.cc"}));
  ExpectPaths(*workspace, b, "twinneedle", {"c1.cc", "c2.cc"}, twins);
  EXPECT_EQ(workspace->Stats().interactive_bytes_ingested, ingested + 11);

  ABSL_ASSERT_OK(other.DeleteFile("a.cc"));
  ABSL_ASSERT_OK(other.WriteFile("bin.dat", "nowtextneedle\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t deleted,
                              NotifyPaths(*workspace, b, {"a.cc", "bin.dat"}));
  ExpectPaths(*workspace, b, "dupneedle", {"b.cc"}, deleted);
  ExpectPaths(*workspace, b, "nowtextneedle", {"bin.dat"}, deleted);
  ExpectPaths(*workspace, a, "dupneedle", {"a.cc", "b.cc"});
  ExpectPaths(*workspace, a, "nowtextneedle", {});

  // A rescan finds the same views.
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t rescan_a, workspace->Rescan(a));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t rescan_b, workspace->Rescan(b));
  ExpectPaths(*workspace, a, "needle", {"a.cc", "b.cc", "notes.txt"}, rescan_a);
  ExpectPaths(*workspace, b, "needle",
              {"b.cc", "bin.dat", "c1.cc", "c2.cc", "sub/elsewhere.cc"},
              rescan_b);
}

// Content of one seeded step: small eligible texts, a long line, an empty
// file, and binary content.
std::string SeededContent(std::mt19937_64& rng) {
  switch (rng() % 8) {
    case 0:
      return "beta seedword\nalpha line\n";
    case 1:
      return absl::StrCat("alpha ", std::string(1100, 'x'), " seedword\n");
    case 2:
      return "";
    case 3:
      return std::string("nul seedword\0tail\n", 18);
    default:
      return absl::StrCat("alpha seedword ", rng() % 4, "\n");
  }
}

// The files a view must contain: what git lists on disk, filtered by
// eligibility. A content's rank key comes from the first path that made it
// visible, the bytewise-smallest in its batch, which the caller models by
// listing the stepped worktree first.
std::vector<ferret_testing::ViewCorpusFile> Oracle(
    const GitRepoBuilder& repo, absl::flat_hash_map<Oid, RankKey>& ranks) {
  absl::StatusOr<std::vector<std::string>> listed =
      ListWorktreeFiles(repo.root());
  CHECK_OK(listed.status());
  std::vector<ferret_testing::ViewCorpusFile> corpus;
  TrigramExtractor extractor;
  for (const std::string& name : *listed) {
    absl::StatusOr<std::string> content =
        Env::Default()->ReadFile(repo.Path(name));
    if (!content.ok()) continue;
    const Eligibility eligibility = CheckEligibility(*content, {}, extractor);
    if (eligibility.skip_reason != SkipReason::kNone) continue;
    const Oid oid = HashBlob(*content);
    ranks.try_emplace(oid, ComputeRankKey(name, content->size(),
                                          eligibility.long_lines, oid));
    corpus.push_back({name, *content, ranks.at(oid)});
  }
  return corpus;
}

SearchRequest SeededRequest(std::mt19937_64& rng) {
  SearchRequest request;
  switch (rng() % 4) {
    case 0:
      request.pattern = "seedword";
      break;
    case 1:
      request.pattern = "alpha.*word";
      request.regex = true;
      break;
    case 2:
      request.pattern = "SEEDWORD";
      break;
    default:
      request.pattern = "seedword [0-9]|line";
      request.regex = true;
      break;
  }
  constexpr absl::string_view kGlobs[] = {"", "dir/**", "dir/file?.cc", "*.cc"};
  request.path_glob = std::string(kGlobs[rng() % 4]);
  request.context_lines = static_cast<int>(rng() % 2);
  request.max_files = rng() % 2 == 0 ? 2 : std::numeric_limits<int64_t>::max();
  request.max_lines = rng() % 3 == 0 ? 3 : std::numeric_limits<int64_t>::max();
  request.max_lines_per_file = rng() % 3 == 0 ? 1 : 10;
  request.exact_counts = rng() % 2 == 0;
  return request;
}

void ExpectMatchesOracle(const SearchResponse& found,
                         const SearchResponse& oracle) {
  ASSERT_EQ(found.files.size(), oracle.files.size());
  for (size_t i = 0; i < oracle.files.size(); ++i) {
    SCOPED_TRACE(i);
    const FileResult& actual = found.files[i];
    const FileResult& expected = oracle.files[i];
    EXPECT_EQ(actual.path, expected.path);
    EXPECT_EQ(actual.oid, expected.oid);
    EXPECT_EQ(actual.tier, expected.tier);
    EXPECT_EQ(actual.prior, expected.prior);
    EXPECT_EQ(actual.name_matches, expected.name_matches);
    EXPECT_EQ(actual.score, expected.score);
    EXPECT_EQ(actual.num_matching_lines, expected.num_matching_lines);
    EXPECT_EQ(actual.lines, expected.lines);
  }
  EXPECT_EQ(found.complete, oracle.complete);
  // The oracle does not model early termination, whose counts cover only the
  // verified prefix.
  if (!found.terminated_early) {
    EXPECT_EQ(found.num_matching_files, oracle.num_matching_files);
    EXPECT_EQ(found.num_matching_lines, oracle.num_matching_lines);
  }
}

TEST(WorkspaceTest, SeededEditsMatchExhaustiveViewOracle) {
  const uint64_t seed = ferret_testing::TestSeed(4123);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("dir/base.cc", "alpha seedword\n"));
  ABSL_ASSERT_OK(repo.WriteFile("dir/sub/anchor.cc", "alpha seedword 1\n"));
  ABSL_ASSERT_OK(repo.WriteFile("dir/file0.cc", "alpha seedword 2\n"));
  ABSL_ASSERT_OK(repo.WriteFile(".gitignore", "*.log\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("initial"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder second,
                              repo.AddWorktree("second"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder third, repo.AddWorktree("third"));
  ABSL_ASSERT_OK(
      third.WriteFile("dir/file1.cc", "beta seedword\nalpha line\n"));
  const std::vector<GitRepoBuilder> repos = {repo, second, third};
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("index")));
  absl::flat_hash_map<Oid, RankKey> ranks;
  std::vector<WorktreeId> ids;
  for (const GitRepoBuilder& tree : repos) {
    FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id,
                                workspace->Register(tree.root()));
    ids.push_back(id);
    Oracle(tree, ranks);
  }
  const std::vector<std::string> pool = {
      "dir/file0.cc", "dir/file1.cc",     "dir/file2.cc",     "dir/file3.cc",
      "dir/x.log",    "dir/sub/deep0.cc", "dir/sub/deep1.cc", "dir/sub/y.log",
      "top0.cc",      "top1.cc"};
  const auto random_path = [&] { return pool[rng() % pool.size()]; };
  // Paths that are neither deleted nor renamed, so "dir" stays a directory.
  const auto movable = [&](const GitRepoBuilder& tree) {
    std::vector<std::string> paths;
    for (const std::string& path : pool) {
      if (Env::Default()->FileExists(tree.Path(path)).value_or(false)) {
        paths.push_back(path);
      }
    }
    return paths;
  };
  for (int step = 0; step < 40; ++step) {
    SCOPED_TRACE(step);
    const size_t tree = rng() % repos.size();
    const GitRepoBuilder& edited = repos[tree];
    std::vector<std::string> changed;
    switch (rng() % 7) {
      case 0:
      case 1: {
        changed.push_back(random_path());
        ABSL_ASSERT_OK(edited.WriteFile(changed[0], SeededContent(rng)));
        break;
      }
      case 2: {
        const std::string content = SeededContent(rng);
        for (int i = 0; i < 3; ++i) {
          changed.push_back(random_path());
          ABSL_ASSERT_OK(edited.WriteFile(
              changed.back(), rng() % 2 == 0 ? content : SeededContent(rng)));
        }
        break;
      }
      case 3: {
        const std::vector<std::string> present = movable(edited);
        if (present.empty()) break;
        changed.push_back(present[rng() % present.size()]);
        ABSL_ASSERT_OK(edited.DeleteFile(changed[0]));
        break;
      }
      case 4:
      case 5: {
        const std::vector<std::string> present = movable(edited);
        if (present.empty()) break;
        changed.push_back(present[rng() % present.size()]);
        changed.push_back(random_path());
        if (changed[0] == changed[1]) break;
        ABSL_ASSERT_OK(edited.Rename(changed[0], changed[1]));
        break;
      }
      default:
        // Committed files get skip hints at the next scan.
        ABSL_ASSERT_OK(edited.Add());
        ABSL_ASSERT_OK(edited.Commit(absl::StrCat("step ", step)));
        break;
    }
    const int apply = static_cast<int>(rng() % 4);
    absl::StatusOr<uint64_t> generation =
        apply == 0 || changed.empty() ? workspace->Rescan(ids[tree])
        : apply == 1
            ? workspace->Notify(ids[tree], std::vector<std::string>{"dir"})
            : workspace->Notify(ids[tree], changed);
    ASSERT_TRUE(generation.ok()) << generation.status();
    Oracle(edited, ranks);
    for (size_t view = 0; view < repos.size(); ++view) {
      const std::vector<ferret_testing::ViewCorpusFile> corpus =
          Oracle(repos[view], ranks);
      for (int query = 0; query < 3; ++query) {
        const SearchRequest request = SeededRequest(rng);
        SCOPED_TRACE(absl::StrCat("view ", view, " pattern ", request.pattern,
                                  " glob ", request.path_glob));
        FERRET_ASSERT_OK_AND_ASSIGN(
            SearchResponse oracle,
            ferret_testing::BruteForceViewSearch(corpus, request));
        FERRET_ASSERT_OK_AND_ASSIGN(
            WorkspaceSearchResult found,
            workspace->Search(ids[view], request,
                              {.mode = SearchConsistencyMode::kNotify,
                               .min_generation = view == tree ? *generation : 0,
                               .timeout = kWait}));
        EXPECT_FALSE(found.stale);
        ExpectMatchesOracle(found.response, oracle);
      }
    }
  }
}

TEST(WorkspaceTest, BulkPauseDoesNotDelayNotify) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("first.cc", "initialtoken\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder other, repo.AddWorktree("other"));
  ABSL_ASSERT_OK(other.WriteFile("extra.cc", "bulkcontent\n"));
  absl::Notification inside;
  absl::Notification release;
  WorkspaceOptions options;
  options.read_hook = Hook([&](absl::string_view path, bool bulk) {
    if (bulk && path == "extra.cc" && !inside.HasBeenNotified()) {
      inside.Notify();
      release.WaitForNotification();
    }
  });
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("index"), std::move(options)));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId first,
                              workspace->Register(repo.root()));
  std::thread register_other(
      [&] { EXPECT_TRUE(workspace->Register(other.root()).ok()); });
  ASSERT_TRUE(inside.WaitForNotificationWithTimeout(kWait));
  ABSL_ASSERT_OK(repo.WriteFile("first.cc", "interactivetoken\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t generation,
                              NotifyPaths(*workspace, first, {"first.cc"}));
  ExpectPaths(*workspace, first, "interactivetoken", {"first.cc"}, generation);
  release.Notify();
  register_other.join();
}

TEST(WorkspaceTest, NotifyCompletesWhileBulkWaitsAtFrozenCap) {
  TempDir temp;
  ABSL_ASSERT_OK(temp.WriteFile("a/edit.cc", "beforetoken\n"));
  for (int i = 0; i < 4; ++i) {
    ABSL_ASSERT_OK(
        temp.WriteFile(absl::StrCat("b/f", i, ".cc"), absl::StrCat("bulk", i)));
  }
  auto gate = std::make_shared<absl::Notification>();
  WorkspaceOptions options;
  options.index.compaction.enabled = false;
  options.index.memtable.flush_docs = 1;
  options.index.max_frozen_memtables = 1;
  options.index.flush_install_gate = gate;
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("index"), std::move(options)));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId a,
                              workspace->Register(temp.Join("a")));
  std::thread register_b(
      [&] { EXPECT_TRUE(workspace->Register(temp.Join("b")).ok()); });
  const absl::Time deadline = absl::Now() + kWait;
  while (workspace->Stats().index.bulk_ingest_waits == 0 &&
         absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(1));
  }
  ASSERT_GT(workspace->Stats().index.bulk_ingest_waits, uint64_t{0});
  ABSL_ASSERT_OK(temp.WriteFile("a/edit.cc", "aftertoken\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t generation,
                              NotifyPaths(*workspace, a, {"edit.cc"}));
  ExpectPaths(*workspace, a, "aftertoken", {"edit.cc"}, generation);
  gate->Notify();
  register_b.join();
}

TEST(WorkspaceTest, OlderBulkReadCannotUndoInteractiveEdit) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("race.cc", "beforetoken\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  absl::Notification paused;
  absl::Notification release;
  std::atomic<bool> pause{false};
  WorkspaceOptions options;
  options.read_hook = Hook([&](absl::string_view path, bool bulk) {
    if (bulk && path == "race.cc" && pause.exchange(false)) {
      paused.Notify();
      release.WaitForNotification();
    }
  });
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("index"), std::move(options)));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id, workspace->Register(repo.root()));
  // The old content is already indexed, so the rescan applies its read
  // without rereading the file.
  ABSL_ASSERT_OK(repo.WriteFile("race.cc", "bulkoldtoken\n"));
  ABSL_ASSERT_OK(repo.WriteFile("keep.cc", "bulkoldtoken\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t kept,
                              NotifyPaths(*workspace, id, {"keep.cc"}));
  ExpectPaths(*workspace, id, "bulkoldtoken", {"keep.cc"}, kept);
  pause.store(true);
  std::thread bulk([&] { EXPECT_TRUE(workspace->Rescan(id).ok()); });
  ASSERT_TRUE(paused.WaitForNotificationWithTimeout(kWait));
  // The paused read holds the old inode open; replace the file.
  ABSL_ASSERT_OK(repo.WriteFile("race.tmp", "interactivetoken\n"));
  ABSL_ASSERT_OK(repo.Rename("race.tmp", "race.cc"));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t generation,
                              NotifyPaths(*workspace, id, {"race.cc"}));
  // The interactive lane publishes while the bulk reader remains paused, but
  // the rescan's earlier generation is not applied yet.
  bool visible = false;
  for (int attempt = 0; attempt < 10000 && !visible; ++attempt) {
    auto response = workspace->Search(id, Request("interactivetoken"));
    visible = response.ok() && !response->response.files.empty();
    if (!visible) absl::SleepFor(absl::Milliseconds(1));
  }
  EXPECT_TRUE(visible);
  auto stale = workspace->Search(id, Request("interactivetoken"),
                                 {.mode = SearchConsistencyMode::kNotify,
                                  .min_generation = generation,
                                  .timeout = absl::ZeroDuration()});
  ASSERT_TRUE(stale.ok()) << stale.status();
  EXPECT_TRUE(stale->stale);
  EXPECT_LT(stale->generation, generation);
  ASSERT_EQ(stale->response.files.size(), size_t{1});
  release.Notify();
  bulk.join();
  ExpectPaths(*workspace, id, "interactivetoken", {"race.cc"}, generation);
  ExpectPaths(*workspace, id, "bulkoldtoken", {"keep.cc"});
}

TEST(WorkspaceTest, UnstableReadKeepsPreviousEntryAndRetries) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("a.cc", "stabletoken\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  std::atomic<bool> rewrite{false};
  std::atomic<int> version{0};
  WorkspaceOptions options;
  options.after_read_hook = Hook([&](absl::string_view path, bool bulk) {
    if (!bulk && path == "a.cc" && rewrite.load()) {
      CHECK_OK(repo.WriteFile(
          "a.cc", absl::StrCat("movingtoken", version.fetch_add(1), "\n")));
    }
  });
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("index"), std::move(options)));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id, workspace->Register(repo.root()));
  ABSL_ASSERT_OK(repo.WriteFile("a.cc", "newtoken\n"));
  rewrite.store(true);
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t generation,
                              NotifyPaths(*workspace, id, {"a.cc"}));
  ExpectPaths(*workspace, id, "stabletoken", {"a.cc"}, generation);
  EXPECT_EQ(workspace->Stats().unstable_files, uint64_t{1});
  EXPECT_EQ(version.load(), 3);
  rewrite.store(false);
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t retry,
                              NotifyPaths(*workspace, id, {"a.cc"}));
  ExpectPaths(*workspace, id, "movingtoken", {"a.cc"}, retry);
}

// With a tiny memtable, registration writes direct segments, several of them
// with a tiny segment budget. Files that change between the first read and
// the rank-ordered reread go through the memtable instead.
TEST(WorkspaceTest, SegmentsSplitAndChangedFilesFallBack) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  for (int i = 0; i < 12; ++i) {
    ABSL_ASSERT_OK(repo.WriteFile(absl::StrCat("src/f", i, ".cc"),
                                  absl::StrCat("segword ", i, "\n")));
  }
  ABSL_ASSERT_OK(repo.WriteFile("change.cc", "firstcontent\n"));
  ABSL_ASSERT_OK(repo.WriteFile("dup_a.cc", "duporiginal\n"));
  ABSL_ASSERT_OK(repo.WriteFile("dup_b.cc", "duporiginal\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  absl::Mutex mu;
  absl::flat_hash_map<std::string, int> reads;
  WorkspaceOptions options;
  options.index.memtable.flush_bytes = 1;
  options.bulk_segment_bytes = 40;
  options.bulk_prefetch_bytes = 20;
  options.reader_threads = 3;
  options.read_hook = Hook([&](absl::string_view path, bool bulk) {
    if (!bulk) return;
    int count;
    {
      absl::MutexLock lock(mu);
      count = ++reads[path];
    }
    // The second bulk read is the reread before the write.
    if (count != 2) return;
    if (path == "change.cc") {
      CHECK_OK(repo.WriteFile("change.cc", "changedlater\n"));
    } else if (path == "dup_a.cc") {
      CHECK_OK(repo.WriteFile("dup_a.cc", "dupchanged\n"));
    }
  });
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("index"), std::move(options)));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id, workspace->Register(repo.root()));
  const WorkspaceStats stats = workspace->Stats();
  EXPECT_GE(stats.index.bulk_segments, uint64_t{3});
  EXPECT_EQ(stats.changed_before_write, uint64_t{2});
  EXPECT_EQ(stats.bulk_files_read, uint64_t{15});
  // The 14 distinct contents, and dup_b.cc again for the content dup_a.cc
  // lost.
  EXPECT_EQ(stats.bulk_files_reread, uint64_t{15});
  // The first reads see the original contents (122 bytes of src files, and
  // 13 + 12 + 12); the rereads see what the hook wrote (122 + 13 + 11) and
  // dup_b.cc's content again (12).
  EXPECT_EQ(stats.bulk_bytes_read, uint64_t{159});
  EXPECT_EQ(stats.bulk_bytes_reread, uint64_t{158});
  EXPECT_GT(stats.segment_write_time, absl::ZeroDuration());
  std::vector<std::string> sources;
  sources.reserve(12);
  for (int i = 0; i < 12; ++i) {
    sources.push_back(absl::StrCat("src/f", i, ".cc"));
  }
  std::sort(sources.begin(), sources.end());
  ExpectPaths(*workspace, id, "segword", sources);
  ExpectPaths(*workspace, id, "changedlater", {"change.cc"});
  ExpectPaths(*workspace, id, "firstcontent", {});
  ExpectPaths(*workspace, id, "dupchanged", {"dup_a.cc"});
  ExpectPaths(*workspace, id, "duporiginal", {"dup_b.cc"});
}

// A generated repository with symbols in three languages, several chunks per
// large file, duplicates, and files that are not eligible. Equal calls give
// repositories with equal Oids.
void WriteEquivalenceRepo(GitRepoBuilder& repo) {
  std::mt19937 rng(20261002);
  const auto word = [&rng] {
    static const char* const kWords[] = {"alpha",  "beta",   "gamma", "delta",
                                         "kernel", "buffer", "inode", "sched"};
    return std::string(kWords[rng() % 8]);
  };
  for (int i = 0; i < 36; ++i) {
    std::string text = absl::StrCat("// file ", i, "\n#include <stdio.h>\n");
    absl::StrAppend(&text, "namespace ns", i % 5, " {\nclass Widget", i,
                    " {\n public:\n  int Method", i, "(int x);\n};\n");
    absl::StrAppend(&text, "int Widget", i, "::Method", i, "(int x) {\n");
    for (int line = 0; line < 8 + static_cast<int>(rng() % 40); ++line) {
      absl::StrAppend(&text, "  int ", word(), "_", rng() % 1000, " = x + ",
                      rng() % 100, ";  // ", word(), "\n");
    }
    absl::StrAppend(&text, "  return x;\n}\n}  // namespace\n");
    CHECK_OK(
        repo.WriteFile(absl::StrCat("src/mod", i % 4, "/w", i, ".cc"), text));
  }
  for (int i = 0; i < 8; ++i) {
    std::string text = absl::StrCat("def helper_", i, "(a, b):\n");
    for (int line = 0; line < 20 + i * 5; ++line) {
      absl::StrAppend(&text, "    ", word(), "_", rng() % 50, " = a + b\n");
    }
    absl::StrAppend(&text, "\nclass Runner", i, ":\n    def go(self):\n",
                    "        return ", i, "\n");
    CHECK_OK(repo.WriteFile(absl::StrCat("py/m", i, ".py"), text));
  }
  for (int i = 0; i < 4; ++i) {
    std::string text = absl::StrCat("struct Packet", i, " {\n  int len;\n};\n",
                                    "static int parse_", i, "(void) {\n");
    for (int line = 0; line < 30; ++line) {
      absl::StrAppend(&text, "  ", word(), "(", rng() % 77, ");\n");
    }
    absl::StrAppend(&text, "  return 0;\n}\n");
    CHECK_OK(repo.WriteFile(absl::StrCat("c/p", i, ".c"), text));
  }
  // Several chunks each, so that their documents span chunk boundaries.
  for (int i = 0; i < 3; ++i) {
    std::string text = absl::StrCat("void BigFunction", i, "() {\n");
    while (text.size() < 30000) {
      absl::StrAppend(&text, "  ", word(), "(", rng() % 9999, ");  // ", word(),
                      "\n");
    }
    absl::StrAppend(&text, "}\n");
    CHECK_OK(repo.WriteFile(absl::StrCat("big/b", i, ".cc"), text));
  }
  CHECK_OK(repo.WriteFile("doc/readme.md", "# Notes\nbuffer inode sched\n"));
  CHECK_OK(repo.WriteFile("doc/main.go", "package main\nfunc main() {}\n"));
  CHECK_OK(repo.WriteFile("dup/a.cc", "int SharedDup() { return 1; }\n"));
  CHECK_OK(repo.WriteFile("dup/b.cc", "int SharedDup() { return 1; }\n"));
  CHECK_OK(repo.WriteFile("dup/c.cc", "int SharedDup() { return 1; }\n"));
  CHECK_OK(repo.WriteFile("bin/blob.dat", std::string("a\0b\0c", 5)));
  CHECK_OK(repo.WriteFile("empty.txt", ""));
  CHECK_OK(repo.Add());
  CHECK_OK(repo.Commit("generated"));
}

std::vector<std::string> SegmentFileContents(const std::string& index_dir) {
  std::vector<std::string> names;
  for (const auto& entry : std::filesystem::directory_iterator(index_dir)) {
    if (entry.path().extension() == ".fseg") {
      names.push_back(entry.path().string());
    }
  }
  std::sort(names.begin(), names.end());
  std::vector<std::string> contents;
  for (const std::string& name : names) {
    std::ifstream in(name, std::ios::binary);
    contents.emplace_back(std::istreambuf_iterator<char>(in),
                          std::istreambuf_iterator<char>());
  }
  return contents;
}

struct EquivalenceRun {
  std::vector<std::string> segments;
  WorkspaceStats stats;
  std::vector<std::vector<std::string>> answers;
  std::vector<std::vector<std::string>> symbols;
};

// Registers a generated repository with direct segments of `segment_bytes`,
// the serial reference path or the parallel one, optionally changing the
// files named in `change` between the first read and the reread.
EquivalenceRun RunEquivalence(bool serial, int64_t segment_bytes,
                              const std::vector<std::string>& change) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  absl::StatusOr<GitRepoBuilder> repo = GitRepoBuilder::Init(temp, "repo");
  CHECK_OK(repo.status());
  WriteEquivalenceRepo(*repo);
  absl::Mutex mu;
  absl::flat_hash_map<std::string, int> reads;
  WorkspaceOptions options;
  options.serial_bulk_write = serial;
  options.index.memtable.flush_bytes = 1;
  options.index.compaction.enabled = false;
  options.bulk_segment_bytes = segment_bytes;
  options.bulk_prefetch_bytes = 40000;
  options.reader_threads = 4;
  options.read_hook = Hook([&](absl::string_view path, bool bulk) {
    if (!bulk) return;
    int count;
    {
      absl::MutexLock lock(mu);
      count = ++reads[path];
    }
    if (count != 2) return;
    for (const std::string& name : change) {
      if (path == name) {
        CHECK_OK(repo->WriteFile(name, "int ChangedLater() { return 7; }\n"));
      }
    }
  });
  EquivalenceRun run;
  const std::string index_dir = temp.Join("index");
  {
    absl::StatusOr<std::unique_ptr<Workspace>> workspace =
        Workspace::Open(*Env::Default(), index_dir, std::move(options));
    CHECK_OK(workspace.status());
    absl::StatusOr<WorktreeId> id = (*workspace)->Register(repo->root());
    CHECK_OK(id.status());
    CHECK_OK((*workspace)->FlushIndex());
    run.stats = (*workspace)->Stats();
    for (const char* pattern :
         {"Method", "kernel", "buffer_[0-9]+", "def helper", "BigFunction",
          "SharedDup", "ChangedLater", "package main"}) {
      auto result = (*workspace)->Search(*id, Request(pattern));
      CHECK_OK(result.status());
      run.answers.push_back(Paths(result->response));
    }
    for (const char* name : {"Widget3", "Method7", "Runner2", "parse_1",
                             "BigFunction1", "Packet0", "helper_5"}) {
      auto found = (*workspace)->FindSymbol(*id, {.name = name});
      CHECK_OK(found.status());
      std::vector<std::string>& lines = run.symbols.emplace_back();
      for (const SymbolLocation& where : found->occurrences) {
        lines.push_back(absl::StrCat(where.path, ":", where.line));
      }
      std::sort(lines.begin(), lines.end());
    }
  }
  run.segments = SegmentFileContents(index_dir);
  return run;
}

// The segments a registration writes are the bytes the serial path wrote,
// across several segments, with symbols, code tokens, and several chunks per
// file.
TEST(WorkspaceTest, ParallelDirectSegmentsAreByteIdenticalToTheSerialPath) {
  const EquivalenceRun reference = RunEquivalence(true, 20000, {});
  const EquivalenceRun parallel = RunEquivalence(false, 20000, {});
  ASSERT_GE(reference.stats.index.bulk_segments, uint64_t{4});
  EXPECT_EQ(parallel.stats.index.bulk_segments,
            reference.stats.index.bulk_segments);
  ASSERT_EQ(parallel.segments.size(), reference.segments.size());
  for (size_t i = 0; i < reference.segments.size(); ++i) {
    EXPECT_TRUE(parallel.segments[i] == reference.segments[i])
        << "segment " << i << " differs: " << parallel.segments[i].size()
        << " vs " << reference.segments[i].size() << " bytes";
  }
  EXPECT_EQ(parallel.answers, reference.answers);
  EXPECT_EQ(parallel.symbols, reference.symbols);
  // The generated repository has what the comparison is about.
  EXPECT_FALSE(reference.answers[0].empty());
  EXPECT_EQ(reference.answers[5].size(), size_t{3});
  for (const auto& found : reference.symbols) EXPECT_FALSE(found.empty());
}

// Files that change between the first read and the reread go to the memtable
// and no precomputed table names them. One segment holds the rest, as before.
TEST(WorkspaceTest, ParallelDirectSegmentWithChangedFilesIsByteIdentical) {
  const std::vector<std::string> change = {"src/mod1/w5.cc", "py/m3.py",
                                           "dup/a.cc", "big/b1.cc"};
  const EquivalenceRun reference =
      RunEquivalence(true, int64_t{1} << 30, change);
  const EquivalenceRun parallel =
      RunEquivalence(false, int64_t{1} << 30, change);
  EXPECT_EQ(reference.stats.changed_before_write, uint64_t{4});
  EXPECT_EQ(parallel.stats.changed_before_write, uint64_t{4});
  ASSERT_EQ(parallel.segments.size(), reference.segments.size());
  ASSERT_GE(reference.segments.size(), size_t{2});
  for (size_t i = 0; i < reference.segments.size(); ++i) {
    EXPECT_TRUE(parallel.segments[i] == reference.segments[i])
        << "segment " << i << " differs";
  }
  EXPECT_EQ(parallel.answers, reference.answers);
  EXPECT_EQ(parallel.symbols, reference.symbols);
  EXPECT_EQ(reference.answers[6].size(), size_t{4});
}

// With several segments, a changed file shrinks its planned segment, which
// the serial rule would have filled up; the answers are the same.
TEST(WorkspaceTest,
     ParallelDirectSegmentsAnswerLikeTheSerialPathWhenFilesChange) {
  const std::vector<std::string> change = {"src/mod2/w6.cc", "py/m1.py",
                                           "dup/a.cc", "c/p2.c"};
  const EquivalenceRun reference = RunEquivalence(true, 20000, change);
  const EquivalenceRun parallel = RunEquivalence(false, 20000, change);
  EXPECT_GE(parallel.stats.index.bulk_segments, uint64_t{3});
  EXPECT_EQ(parallel.stats.changed_before_write, uint64_t{4});
  EXPECT_EQ(parallel.answers, reference.answers);
  EXPECT_EQ(parallel.symbols, reference.symbols);
  EXPECT_EQ(parallel.stats.bulk_bytes_ingested,
            reference.stats.bulk_bytes_ingested);
}

// A segment takes files up to exactly its budget, as the serial rule did.
TEST(WorkspaceTest, ParallelDirectSegmentsFillTheirBudgetExactly) {
  for (const bool serial : {true, false}) {
    TempDir temp;
    ferret_testing::IsolateGitEnvironment(temp.Join("home"));
    FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                                GitRepoBuilder::Init(temp, "repo"));
    for (int i = 0; i < 6; ++i) {
      // Twenty bytes each.
      ABSL_ASSERT_OK(
          repo.WriteFile(absl::StrCat("f", i, ".cc"),
                         absl::StrCat("int Exact", i, "_xxxxxxx;\n")));
    }
    ABSL_ASSERT_OK(repo.Add());
    ABSL_ASSERT_OK(repo.Commit("base"));
    WorkspaceOptions options;
    options.serial_bulk_write = serial;
    options.index.memtable.flush_bytes = 1;
    options.index.compaction.enabled = false;
    options.bulk_segment_bytes = 40;
    FERRET_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Workspace> workspace,
        Workspace::Open(*Env::Default(), temp.Join("index"),
                        std::move(options)));
    FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id,
                                workspace->Register(repo.root()));
    EXPECT_EQ(workspace->Stats().index.bulk_segments, uint64_t{3}) << serial;
    EXPECT_EQ(workspace->Stats().bulk_bytes_ingested, uint64_t{120});
    ExpectPaths(*workspace, id, "Exact3", {"f3.cc"});
  }
}

TEST(WorkspaceTest, RescanReusesSignaturesOfUnchangedFiles) {
  TempDir temp;
  for (int i = 0; i < 3; ++i) {
    ABSL_ASSERT_OK(temp.WriteFile(absl::StrCat("plain/f", i, ".cc"),
                                  absl::StrCat("plainword ", i, "\n")));
  }
  ABSL_ASSERT_OK(temp.WriteFile("plain/big.cc", std::string(2000, 'b')));
  ABSL_ASSERT_OK(temp.WriteFile("plain/nul.cc", std::string("a\0b", 3)));
  // Signatures of files changed just before their read are not trusted.
  absl::SleepFor(absl::Milliseconds(300));
  WorkspaceOptions options;
  options.eligibility.max_file_bytes = 1000;
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("index"), std::move(options)));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id,
                              workspace->Register(temp.Join("plain")));
  const WorkspaceStats first = workspace->Stats();
  EXPECT_EQ(first.bulk_files_read, uint64_t{4});
  EXPECT_EQ(first.skipped[static_cast<size_t>(SkipReason::kTooLarge)],
            uint64_t{1});
  ABSL_ASSERT_OK(workspace->Rescan(id).status());
  const WorkspaceStats second = workspace->Stats();
  EXPECT_EQ(second.bulk_files_read, first.bulk_files_read);
  EXPECT_EQ(second.signatures_reused - first.signatures_reused, uint64_t{5});
  ABSL_ASSERT_OK(temp.WriteFile("plain/f1.cc", "plainword changed\n"));
  ABSL_ASSERT_OK(temp.WriteFile("plain/new.cc", "plainword new\n"));
  ABSL_ASSERT_OK(Env::Default()->DeleteFile(temp.Join("plain/f2.cc")));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t generation, workspace->Rescan(id));
  EXPECT_EQ(workspace->Stats().bulk_files_read, first.bulk_files_read + 2);
  ExpectPaths(*workspace, id, "plainword", {"f0.cc", "f1.cc", "new.cc"},
              generation);
  ExpectPaths(*workspace, id, "changed", {"f1.cc"}, generation);
}

TEST(WorkspaceTest, NotifySplitsLargeBatches) {
  TempDir temp;
  ABSL_ASSERT_OK(temp.WriteFile("plain/seed.cc", "seed\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("index")));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id,
                              workspace->Register(temp.Join("plain")));
  std::vector<std::string> paths;
  for (int i = 0; i < 600; ++i) {
    paths.push_back(absl::StrCat("many/f", i, ".cc"));
    ABSL_ASSERT_OK(temp.WriteFile(absl::StrCat("plain/", paths.back()),
                                  absl::StrCat("manyword ", i, "\n")));
  }
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t generation,
                              workspace->Notify(id, paths));
  EXPECT_EQ(generation, uint64_t{3});
  std::sort(paths.begin(), paths.end());
  ExpectPaths(*workspace, id, "manyword", paths, generation);
  EXPECT_EQ(workspace->Stats().interactive_files_read, uint64_t{600});
}

TEST(WorkspaceTest, NotifyOpensOnlyRegularFiles) {
  TempDir temp;
  ABSL_ASSERT_OK(temp.WriteFile("plain/seed.cc", "fifoword\n"));
  absl::Mutex mu;
  std::vector<std::string> opened;  // Guarded by mu.
  WorkspaceOptions options;
  options.read_hook = Hook([&](absl::string_view path, bool bulk) {
    if (bulk) return;
    absl::MutexLock lock(mu);
    opened.emplace_back(path);
  });
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("index"), std::move(options)));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id,
                              workspace->Register(temp.Join("plain")));
  ASSERT_EQ(mkfifo(temp.Join("plain/pipe.cc").c_str(), 0600), 0);
  ABSL_ASSERT_OK(temp.WriteFile("plain/seed.cc", "fifoword changed\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      uint64_t generation,
      NotifyPaths(*workspace, id, {"pipe.cc", "seed.cc", "missing.cc"}));
  ExpectPaths(*workspace, id, "fifoword", {"seed.cc"}, generation);
  EXPECT_EQ(workspace->Stats().interactive_files_read, uint64_t{1});
  absl::MutexLock lock(mu);
  EXPECT_EQ(opened, std::vector<std::string>{"seed.cc"});
}

TEST(WorkspaceTest, NonGitEligibilityAndReopen) {
  TempDir temp;
  ABSL_ASSERT_OK(temp.WriteFile("plain/ok.cc", "lookupword\n"));
  ABSL_ASSERT_OK(
      temp.WriteFile("plain/no.cc", std::string("binary\0word", 11)));
  ABSL_ASSERT_OK(temp.WriteFile("plain/.gitignore", "*.log\n"));
  ABSL_ASSERT_OK(temp.WriteFile("plain/hidden.log", "lookupword\n"));
  {
    FERRET_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Workspace> workspace,
        Workspace::Open(*Env::Default(), temp.Join("index")));
    FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id,
                                workspace->Register(temp.Join("plain")));
    ExpectPaths(*workspace, id, "lookupword", {"ok.cc"});
    EXPECT_EQ(workspace->Stats()
                  .skipped[static_cast<size_t>(SkipReason::kContainsNul)],
              uint64_t{1});
    ABSL_ASSERT_OK(temp.WriteFile("plain/later.log", "lookupword\n"));
    FERRET_ASSERT_OK_AND_ASSIGN(uint64_t ignored,
                                NotifyPaths(*workspace, id, {"later.log"}));
    ExpectPaths(*workspace, id, "lookupword", {"ok.cc"}, ignored);
    // An edited ignore file applies to the paths of the same batch.
    ABSL_ASSERT_OK(temp.WriteFile("plain/.gitignore", "*.tmp\n"));
    FERRET_ASSERT_OK_AND_ASSIGN(
        uint64_t unignored,
        NotifyPaths(*workspace, id, {".gitignore", "later.log"}));
    ExpectPaths(*workspace, id, "lookupword", {"later.log", "ok.cc"},
                unignored);
    ABSL_ASSERT_OK(workspace->Unregister(id));
  }
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> reopened,
      Workspace::Open(*Env::Default(), temp.Join("index")));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id,
                              reopened->Register(temp.Join("plain")));
  ExpectPaths(*reopened, id, "lookupword",
              {"hidden.log", "later.log", "ok.cc"});
  EXPECT_EQ(reopened->Stats().bulk_bytes_ingested, uint64_t{0});
}

TEST(WorkspaceTest, GitReopenIsScanOnly) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("a.cc", "reopenword\n"));
  ABSL_ASSERT_OK(repo.WriteFile("b.bin", std::string("\0\0", 2)));
  ABSL_ASSERT_OK(repo.WriteFile("empty.txt", ""));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  {
    FERRET_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Workspace> workspace,
        Workspace::Open(*Env::Default(), temp.Join("index")));
    FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id,
                                workspace->Register(repo.root()));
    ExpectPaths(*workspace, id, "reopenword", {"a.cc"});
    // The scan's size shows that empty.txt is empty without reading it.
    const WorkspaceStats stats = workspace->Stats();
    EXPECT_EQ(stats.bulk_files_read, uint64_t{2});
    EXPECT_EQ(stats.skipped[static_cast<size_t>(SkipReason::kEmpty)],
              uint64_t{1});
  }
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> reopened,
      Workspace::Open(*Env::Default(), temp.Join("index")));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id, reopened->Register(repo.root()));
  ExpectPaths(*reopened, id, "reopenword", {"a.cc"});
  // Ineligible Oids are not persisted, so only b.bin is read again.
  EXPECT_EQ(reopened->Stats().hint_reads, uint64_t{1});
  EXPECT_EQ(reopened->Stats().bulk_bytes_ingested, uint64_t{0});
}

TEST(WorkspaceTest, ConcurrentSearchNotifyRescanAndRegistration) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  constexpr int kFiles = 4;
  absl::Mutex mu;
  // Every token written to each path, recorded before the write.
  // Guarded by mu.
  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> written;
  const auto write = [&](int file, absl::string_view token) {
    const std::string path = absl::StrCat("live", file, ".cc");
    {
      absl::MutexLock lock(mu);
      written[path].insert(std::string(token));
    }
    CHECK_OK(repo.WriteFile(path, absl::StrCat(token, "\n")));
    return path;
  };
  for (int file = 0; file < kFiles; ++file) {
    write(file, absl::StrCat("stresstoken", file, "x"));
  }
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder other, repo.AddWorktree("other"));
  ABSL_ASSERT_OK(other.WriteFile("extra.cc", "othercontent\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("index")));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id, workspace->Register(repo.root()));
  std::atomic<bool> stopping{false};
  std::thread searches([&] {
    while (!stopping.load()) {
      auto result = workspace->Search(id, Request("stresstoken"));
      ASSERT_TRUE(result.ok()) << result.status();
      absl::MutexLock lock(mu);
      for (const FileResult& file : result->response.files) {
        ASSERT_TRUE(written.contains(file.path)) << file.path;
        for (const ResultLine& line : file.lines) {
          EXPECT_TRUE(written[file.path].contains(line.text)) << line.text;
        }
      }
    }
  });
  std::thread rescans([&] {
    while (!stopping.load()) EXPECT_TRUE(workspace->Rescan(id).ok());
  });
  std::thread register_other([&] {
    for (int i = 0; i < 3; ++i) {
      absl::StatusOr<WorktreeId> other_id = workspace->Register(other.root());
      ASSERT_TRUE(other_id.ok()) << other_id.status();
      EXPECT_TRUE(workspace->Unregister(*other_id).ok());
    }
  });
  std::mt19937_64 rng(ferret_testing::TestSeed(77));
  for (int i = 0; i < 60; ++i) {
    std::vector<std::string> paths;
    std::vector<std::string> tokens;
    for (int k = 0; k < 2; ++k) {
      tokens.push_back(absl::StrCat("stresstoken", i, "y", k));
      paths.push_back(write(static_cast<int>(rng() % kFiles), tokens.back()));
    }
    FERRET_ASSERT_OK_AND_ASSIGN(uint64_t generation,
                                workspace->Notify(id, paths));
    // The second write wins when both went to one path.
    for (int k = paths[0] == paths[1] ? 1 : 0; k < 2; ++k) {
      auto result = workspace->Search(id, Request(tokens[k]),
                                      {.mode = SearchConsistencyMode::kNotify,
                                       .min_generation = generation,
                                       .timeout = kWait});
      ASSERT_TRUE(result.ok()) << result.status();
      EXPECT_FALSE(result->stale);
      EXPECT_EQ(Paths(result->response), std::vector<std::string>{paths[k]});
    }
  }
  stopping.store(true);
  searches.join();
  rescans.join();
  register_other.join();
}

TEST(WorkspaceTest, ShutdownCancelsPendingRegistration) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("pending.cc", "unindexedblob\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  absl::Notification paused;
  absl::Notification release;
  WorkspaceOptions options;
  options.read_hook = Hook([&](absl::string_view path, bool bulk) {
    if (bulk && path == "pending.cc" && !paused.HasBeenNotified()) {
      paused.Notify();
      release.WaitForNotification();
    }
  });
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp.Join("index"), std::move(options)));
  Workspace* const ptr = workspace.get();
  absl::Status outcome;
  std::thread registering(
      [&] { outcome = ptr->Register(repo.root()).status(); });
  ASSERT_TRUE(paused.WaitForNotificationWithTimeout(kWait));
  std::thread unpause([&] {
    absl::SleepFor(absl::Milliseconds(10));
    release.Notify();
  });
  workspace.reset();
  registering.join();
  unpause.join();
  EXPECT_THAT(outcome, StatusIs(absl::StatusCode::kCancelled));
}

}  // namespace
}  // namespace ferret

namespace ferret {
namespace {

// Waits for the GC deadline and any rewrite, not for a fixed scheduler delay.
void ExpectDocumentCount(Workspace& workspace, uint64_t count) {
  const absl::Time deadline = absl::Now() + absl::Seconds(10);
  while (absl::Now() < deadline) {
    workspace.WaitForCompactions();
    const IndexStats index = workspace.Stats().index;
    if (index.live_docs + index.dead_docs == count) return;
    absl::SleepFor(absl::Milliseconds(5));
  }
  const IndexStats index = workspace.Stats().index;
  EXPECT_EQ(index.live_docs + index.dead_docs, count)
      << " gc_scans=" << index.gc_scans << " dead=" << index.dead_docs
      << " segments=" << index.segment_count
      << " completed=" << index.completed_compactions
      << " failures=" << index.compaction_failures
      << " last=" << index.last_compaction_error;
}

TEST(WorkspaceGcTest, MultiplicityAndSharedViews) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("a.cc", "gcsharedneedle\n"));
  ABSL_ASSERT_OK(repo.WriteFile("b.cc", "gcsharedneedle\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder other, repo.AddWorktree("other"));
  WorkspaceOptions options;
  options.index.gc_grace_period = absl::Milliseconds(15);
  options.index.memtable.flush_docs = 1;
  options.index.compaction.rewrite_dead_percent = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto workspace,
      Workspace::Open(*Env::Default(), temp.Join("idx"), std::move(options)));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId first,
                              workspace->Register(repo.root()));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId second,
                              workspace->Register(other.root()));
  workspace->WaitForCompactions();
  ExpectDocumentCount(*workspace, 1);
  ABSL_ASSERT_OK(repo.DeleteFile("a.cc"));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t edit,
                              NotifyPaths(*workspace, first, {"a.cc"}));
  ExpectPaths(*workspace, first, "gcsharedneedle", {"b.cc"}, edit);
  ABSL_ASSERT_OK(workspace->Unregister(first));
  absl::SleepFor(absl::Milliseconds(30));
  workspace->WaitForCompactions();
  ExpectPaths(*workspace, second, "gcsharedneedle", {"a.cc", "b.cc"});
  ExpectDocumentCount(*workspace, 1);
  ABSL_ASSERT_OK(workspace->Unregister(second));
  ExpectDocumentCount(*workspace, 0);
  EXPECT_GT(workspace->Stats().index.reclaimed_bytes, uint64_t{0});
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId reread,
                              workspace->Register(other.root()));
  ExpectPaths(*workspace, reread, "gcsharedneedle", {"a.cc", "b.cc"});
  EXPECT_GT(workspace->Stats().bulk_files_read, uint64_t{0});
  ABSL_ASSERT_OK(workspace->Unregister(reread));
}

TEST(WorkspaceGcTest, ReentryWithinGraceAndRestartReuse) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("a.cc", "gcreentryneedle\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  WorkspaceOptions options;
  options.index.gc_grace_period = absl::Seconds(2);
  options.index.memtable.flush_docs = 1;
  options.index.compaction.rewrite_dead_percent = 1;
  const std::string index_dir = temp.Join("idx");
  {
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto workspace, Workspace::Open(*Env::Default(), index_dir, options));
    FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id,
                                workspace->Register(repo.root()));
    const WorkspaceStats before = workspace->Stats();
    ABSL_ASSERT_OK(repo.DeleteFile("a.cc"));
    FERRET_ASSERT_OK_AND_ASSIGN(uint64_t deleted,
                                NotifyPaths(*workspace, id, {"a.cc"}));
    ExpectPaths(*workspace, id, "gcreentryneedle", {}, deleted);
    ABSL_ASSERT_OK(repo.WriteFile("a.cc", "gcreentryneedle\n"));
    FERRET_ASSERT_OK_AND_ASSIGN(uint64_t restored,
                                NotifyPaths(*workspace, id, {"a.cc"}));
    ExpectPaths(*workspace, id, "gcreentryneedle", {"a.cc"}, restored);
    EXPECT_EQ(workspace->Stats().interactive_bytes_ingested,
              before.interactive_bytes_ingested);
    ABSL_ASSERT_OK(workspace->Unregister(id));
  }
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto reopened, Workspace::Open(*Env::Default(), index_dir, options));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id, reopened->Register(repo.root()));
  ExpectPaths(*reopened, id, "gcreentryneedle", {"a.cc"});
  EXPECT_EQ(reopened->Stats().bulk_files_read, uint64_t{0});
  EXPECT_EQ(reopened->Stats().bulk_bytes_ingested, uint64_t{0});
  ABSL_ASSERT_OK(reopened->Unregister(id));
}

TEST(WorkspaceGcTest, SeededViewsSurviveCollectionAndReRegistration) {
  const uint64_t seed = ferret_testing::TestSeed(517);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  for (int i = 0; i < 5; ++i) {
    ABSL_ASSERT_OK(
        repo.WriteFile(absl::StrCat("file", i, ".cc"), "gcsharedneedle\n"));
  }
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder second,
                              repo.AddWorktree("second"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder third, repo.AddWorktree("third"));
  const std::vector<GitRepoBuilder> repos = {repo, second, third};
  WorkspaceOptions options;
  options.index.gc_grace_period = absl::Milliseconds(8);
  options.index.memtable.flush_docs = 1;
  options.index.compaction.tier_min_segments = 2;
  options.index.compaction.rewrite_dead_percent = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto workspace,
      Workspace::Open(*Env::Default(), temp.Join("idx"), std::move(options)));
  std::vector<WorktreeId> ids;
  for (const GitRepoBuilder& tree : repos) {
    FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id,
                                workspace->Register(tree.root()));
    ids.push_back(id);
  }
  absl::flat_hash_map<Oid, RankKey> ranks;
  for (int step = 0; step < 24; ++step) {
    SCOPED_TRACE(step);
    const size_t which = rng() % repos.size();
    const GitRepoBuilder& tree = repos[which];
    const std::string path = absl::StrCat("file", rng() % 5, ".cc");
    const std::string other = absl::StrCat("file", rng() % 5, ".cc");
    const int action = static_cast<int>(rng() % 6);
    if (action == 0) {
      if (Env::Default()->FileExists(tree.Path(path)).value_or(false)) {
        ABSL_ASSERT_OK(tree.DeleteFile(path));
      }
    } else if (action == 1 && path != other &&
               Env::Default()->FileExists(tree.Path(path)).value_or(false)) {
      ABSL_ASSERT_OK(tree.Rename(path, other));
    } else if (action == 2) {
      FERRET_ASSERT_OK_AND_ASSIGN(std::string output,
                                  tree.Git({"checkout", "-f", "HEAD"}));
      (void)output;
    } else {
      const std::string content =
          rng() % 3 == 0 ? "gcsharedneedle\n"
                         : absl::StrCat("gcunique", step, "needle\n");
      ABSL_ASSERT_OK(tree.WriteFile(path, content));
      if (action == 3) ABSL_ASSERT_OK(tree.WriteFile(other, content));
    }
    if (step % 5 == 0) {
      ABSL_ASSERT_OK(workspace->Unregister(ids[which]));
      FERRET_ASSERT_OK_AND_ASSIGN(ids[which], workspace->Register(tree.root()));
    } else {
      FERRET_ASSERT_OK_AND_ASSIGN(uint64_t generation,
                                  workspace->Rescan(ids[which]));
      FERRET_ASSERT_OK_AND_ASSIGN(
          uint64_t applied,
          workspace->AwaitGeneration(ids[which], generation, kWait));
      EXPECT_GE(applied, generation);
    }
    absl::SleepFor(absl::Milliseconds(10));
    workspace->WaitForCompactions();
    for (size_t view = 0; view < repos.size(); ++view) {
      const auto corpus = Oracle(repos[view], ranks);
      FERRET_ASSERT_OK_AND_ASSIGN(
          SearchResponse expected,
          ferret_testing::BruteForceViewSearch(corpus, Request("needle")));
      FERRET_ASSERT_OK_AND_ASSIGN(
          WorkspaceSearchResult found,
          workspace->Search(ids[view], Request("needle")));
      EXPECT_EQ(Paths(found.response), Paths(expected)) << "view " << view;
    }
  }
  for (WorktreeId id : ids) ABSL_ASSERT_OK(workspace->Unregister(id));
  ExpectDocumentCount(*workspace, 0);
}

TEST(WorkspaceGcTest, ConcurrentEditsAndCompactionsKeepSearchableViews) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("a.cc", "gcthreadneedle\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder other, repo.AddWorktree("other"));
  WorkspaceOptions options;
  options.index.gc_grace_period = absl::Milliseconds(5);
  options.index.memtable.flush_docs = 1;
  options.index.compaction.tier_min_segments = 2;
  options.index.compaction.rewrite_dead_percent = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto workspace,
      Workspace::Open(*Env::Default(), temp.Join("idx"), std::move(options)));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId first,
                              workspace->Register(repo.root()));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId second,
                              workspace->Register(other.root()));
  std::atomic<bool> running{true};
  std::thread compactor([&] {
    while (running.load()) {
      workspace->WaitForCompactions();
      absl::SleepFor(absl::Milliseconds(1));
    }
  });
  const auto edit = [&](const GitRepoBuilder& tree, WorktreeId id) {
    for (int i = 0; i < 16; ++i) {
      ASSERT_TRUE(
          tree.WriteFile("a.cc", absl::StrCat("gcthreadneedle ", i, "\n"))
              .ok());
      auto generation = NotifyPaths(*workspace, id, {"a.cc"});
      ASSERT_TRUE(generation.ok()) << generation.status();
      ExpectPaths(*workspace, id, "gcthreadneedle", {"a.cc"}, *generation);
    }
  };
  std::thread a([&] { edit(repo, first); });
  std::thread b([&] { edit(other, second); });
  a.join();
  b.join();
  running.store(false);
  compactor.join();
  ExpectPaths(*workspace, first, "gcthreadneedle", {"a.cc"});
  ExpectPaths(*workspace, second, "gcthreadneedle", {"a.cc"});
}

TEST(WorkspaceGcTest, RestoresViewsBeforeCollectingAndRemovesMissingRoot) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("tracked.cc", "persistneedle tracked\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder other, repo.AddWorktree("other"));
  ABSL_ASSERT_OK(repo.WriteFile("edit.cc", "persistneedle first edit\n"));
  ABSL_ASSERT_OK(other.WriteFile("edit.cc", "persistneedle second edit\n"));
  const std::string removed = temp.Join("removed");
  ABSL_ASSERT_OK(temp.MakeDirs("removed"));
  WorkspaceOptions options;
  options.index.gc_grace_period = absl::ZeroDuration();
  options.index.memtable.flush_docs = 1;
  options.index.compaction.tier_min_segments = 2;
  {
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto workspace,
        Workspace::Open(*Env::Default(), temp.Join("idx"), options));
    FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId a, workspace->Register(repo.root()));
    FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId b,
                                workspace->Register(other.root()));
    ASSERT_TRUE(workspace->Register(removed).ok());
    ExpectPaths(*workspace, a, "persistneedle", {"edit.cc", "tracked.cc"});
    ExpectPaths(*workspace, b, "persistneedle", {"edit.cc", "tracked.cc"});
    EXPECT_EQ(workspace->Roots().size(), size_t{3});
  }
  ASSERT_EQ(rmdir(removed.c_str()), 0);
  absl::Notification reading;
  absl::Notification continue_reading;
  options.read_hook = Hook([&](absl::string_view path, bool bulk) {
    if (bulk && path == "edit.cc" && !reading.HasBeenNotified()) {
      reading.Notify();
      continue_reading.WaitForNotification();
    }
  });
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto restored,
      Workspace::Open(*Env::Default(), temp.Join("idx"), options));
  ASSERT_TRUE(reading.WaitForNotificationWithTimeout(kWait));
  EXPECT_TRUE(restored->Roots().empty());
  std::atomic<bool> client_done{false};
  absl::StatusOr<WorktreeId> client_id;
  std::thread client([&] {
    client_id = restored->Register(repo.root());
    client_done.store(true);
  });
  EXPECT_FALSE(client_done.load());
  continue_reading.Notify();
  client.join();
  ASSERT_TRUE(client_id.ok()) << client_id.status();
  for (int i = 0; i < 3000 && restored->Roots().size() != 2; ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_EQ(restored->Roots().size(), size_t{2});
  restored->WaitForCompactions();
  absl::flat_hash_map<Oid, RankKey> ranks;
  for (const GitRepoBuilder& tree : {repo, other}) {
    const auto oracle = ferret_testing::BruteForceViewSearch(
        Oracle(tree, ranks), Request("persistneedle"));
    ASSERT_TRUE(oracle.ok()) << oracle.status();
    WorktreeId id = *client_id;
    for (const auto& [root, published] : restored->Roots()) {
      if (root == tree.root()) id = published;
    }
    FERRET_ASSERT_OK_AND_ASSIGN(auto found,
                                restored->Search(id, Request("persistneedle")));
    ExpectMatchesOracle(found.response, *oracle);
  }
  EXPECT_EQ(restored->Stats().bulk_bytes_ingested, uint64_t{0});
  // A direct reopen sees only the two surviving identities in MANIFEST.
  restored.reset();
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto again, Workspace::Open(*Env::Default(), temp.Join("idx"), options));
  for (int i = 0; i < 3000 && again->Roots().size() != 2; ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  EXPECT_EQ(again->Roots().size(), size_t{2});
}

TEST(WorkspaceGcTest, FailedRestoreKeepsViewAndRetriesWithoutCollecting) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("tracked.cc", "retryneedle original\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder other, repo.AddWorktree("other"));
  ABSL_ASSERT_OK(other.WriteFile("tracked.cc", "otherneedle changed\n"));
  ABSL_ASSERT_OK(other.WriteFile("extra.cc", "extra unique\n"));
  const std::string index_dir = temp.Join("idx");
  WorkspaceOptions options;
  options.index.gc_grace_period = absl::ZeroDuration();
  options.index.memtable.flush_docs = 1;
  options.index.compaction.tier_min_segments = 2;
  {
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto initial, Workspace::Open(*Env::Default(), index_dir, options));
    FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id, initial->Register(repo.root()));
    ExpectPaths(*initial, id, "retryneedle", {"tracked.cc"});
  }
  absl::Notification failed;
  absl::Notification failed_again;
  std::atomic<int> failures{0};
  std::atomic<bool> allow{false};
  options.restore_scan_hook =
      std::make_shared<absl::AnyInvocable<absl::Status(absl::string_view)>>(
          [&](absl::string_view) {
            if (!allow.load()) {
              if (++failures == 1) {
                failed.Notify();
              } else {
                failed_again.Notify();
              }
              return absl::UnavailableError("injected scan failure");
            }
            return absl::OkStatus();
          });
  options.ingest_retry_initial_delay = absl::Seconds(10);
  options.ingest_retry_max_delay = absl::Seconds(10);
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto restored, Workspace::Open(*Env::Default(), index_dir, options));
  ASSERT_TRUE(failed.WaitForNotificationWithTimeout(kWait));
  EXPECT_TRUE(restored->Roots().empty());
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId other_id,
                              restored->Register(other.root()));
  restored->WaitForCompactions();
  ExpectPaths(*restored, other_id, "otherneedle", {"tracked.cc"});
  // Shutting down during backoff must leave the failed view in MANIFEST.
  restored.reset();
  {
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto persisted, Index::Open(*Env::Default(), index_dir, options.index));
    ASSERT_EQ(persisted.first->Views().size(), size_t{2});
    EXPECT_EQ(persisted.first->Views()[0].canonical_root, repo.root());
  }
  FERRET_ASSERT_OK_AND_ASSIGN(
      restored, Workspace::Open(*Env::Default(), index_dir, options));
  ASSERT_TRUE(failed_again.WaitForNotificationWithTimeout(kWait));
  allow.store(true);
  // A client request bypasses the ten-second backoff and joins the restore.
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id, restored->Register(repo.root()));
  ExpectPaths(*restored, id, "retryneedle", {"tracked.cc"});
  EXPECT_EQ(restored->Stats().bulk_bytes_ingested, uint64_t{0});
  restored.reset();
  options.restore_scan_hook.reset();
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto index, Index::Open(*Env::Default(), index_dir, options.index));
  ASSERT_EQ(index.first->Views().size(), size_t{2});
  EXPECT_EQ(index.first->Views()[0].canonical_root, repo.root());
}

TEST(WorkspaceGcTest, ClientRegisterDuringFailingRestoreReturnsError) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("tracked.cc", "failingrestoreneedle\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  const std::string index_dir = temp.Join("idx");
  WorkspaceOptions options;
  {
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto initial, Workspace::Open(*Env::Default(), index_dir, options));
    ABSL_ASSERT_OK(initial->Register(repo.root()).status());
  }
  absl::Notification failed;
  std::atomic<bool> allow{false};
  options.restore_scan_hook =
      std::make_shared<absl::AnyInvocable<absl::Status(absl::string_view)>>(
          [&](absl::string_view) {
            if (allow.load()) return absl::OkStatus();
            if (!failed.HasBeenNotified()) failed.Notify();
            return absl::UnavailableError("injected scan failure");
          });
  // Backoff far longer than the wait below, so only the client's request
  // can trigger the attempt that answers it.
  options.ingest_retry_initial_delay = absl::Minutes(10);
  options.ingest_retry_max_delay = absl::Minutes(10);
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto restored, Workspace::Open(*Env::Default(), index_dir, options));
  ASSERT_TRUE(failed.WaitForNotificationWithTimeout(kWait));
  absl::StatusOr<WorktreeId> registered;
  absl::Notification returned;
  std::thread client([&] {
    registered = restored->Register(repo.root());
    returned.Notify();
  });
  const bool answered = returned.WaitForNotificationWithTimeout(kWait);
  if (!answered) allow.store(true);  // Unblock the client before failing.
  client.join();
  ASSERT_TRUE(answered) << "Register waited on a restore that keeps failing";
  EXPECT_TRUE(absl::IsUnavailable(registered.status())) << registered.status();
  // The failure leaves the view persisted and pending; once the scan can
  // succeed, the next client request restores it.
  allow.store(true);
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id, restored->Register(repo.root()));
  ExpectPaths(*restored, id, "failingrestoreneedle", {"tracked.cc"});
}

TEST(WorkspaceGcTest, RetriesFailedScanWithoutClient) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("tracked.cc", "automaticretryneedle\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  const std::string index_dir = temp.Join("idx");
  {
    FERRET_ASSERT_OK_AND_ASSIGN(auto initial,
                                Workspace::Open(*Env::Default(), index_dir));
    ASSERT_TRUE(initial->Register(repo.root()).ok());
  }
  std::atomic<int> attempts{0};
  WorkspaceOptions options;
  options.ingest_retry_initial_delay = absl::Milliseconds(10);
  options.ingest_retry_max_delay = absl::Milliseconds(40);
  options.restore_scan_hook =
      std::make_shared<absl::AnyInvocable<absl::Status(absl::string_view)>>(
          [&](absl::string_view) {
            if (++attempts == 1) {
              return absl::UnavailableError("injected scan failure");
            }
            return absl::OkStatus();
          });
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto restored, Workspace::Open(*Env::Default(), index_dir, options));
  for (int i = 0; i < 1000 && restored->Roots().empty(); ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_EQ(restored->Roots().size(), size_t{1});
  EXPECT_EQ(attempts.load(), 2);
  ExpectPaths(*restored, restored->Roots()[0].second, "automaticretryneedle",
              {"tracked.cc"});
}

TEST(WorkspaceGcTest, UnregisterDuringRestoreForgetsOnlyThatView) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("tracked.cc", "unregisterneedle\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder other, repo.AddWorktree("other"));
  ABSL_ASSERT_OK(other.WriteFile("edit.cc", "anotherneedle\n"));
  const std::string index_dir = temp.Join("idx");
  {
    FERRET_ASSERT_OK_AND_ASSIGN(auto workspace,
                                Workspace::Open(*Env::Default(), index_dir));
    ASSERT_TRUE(workspace->Register(repo.root()).ok());
    ASSERT_TRUE(workspace->Register(other.root()).ok());
  }
  absl::Notification paused;
  absl::Notification release;
  WorkspaceOptions options;
  options.index.gc_grace_period = absl::ZeroDuration();
  options.read_hook = Hook([&](absl::string_view path, bool bulk) {
    if (bulk && path == "edit.cc") {
      paused.Notify();
      release.WaitForNotification();
    }
  });
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto workspace, Workspace::Open(*Env::Default(), index_dir, options));
  ASSERT_TRUE(paused.WaitForNotificationWithTimeout(kWait));
  const auto published = workspace->Roots();
  ASSERT_EQ(published.size(), size_t{1});
  EXPECT_EQ(published[0].first, repo.root());
  ABSL_ASSERT_OK(workspace->Unregister(published[0].second));
  release.Notify();
  for (int i = 0; i < 3000 && workspace->Roots().size() != 1; ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_EQ(workspace->Roots().size(), size_t{1});
  EXPECT_EQ(workspace->Roots()[0].first, other.root());
  workspace.reset();
  options.read_hook.reset();
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto recovered, Workspace::Open(*Env::Default(), index_dir, options));
  for (int i = 0; i < 3000 && recovered->Roots().empty(); ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_EQ(recovered->Roots().size(), size_t{1});
  EXPECT_EQ(recovered->Roots()[0].first, other.root());
}

TEST(WorkspaceTest, PendingSymbolDoesNotBlockTextOrSync) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("initial.txt", "other\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  auto gate = std::make_shared<absl::Notification>();
  auto reached = std::make_shared<absl::Notification>();
  WorkspaceOptions options;
  options.index.symbol_extraction_gate = gate;
  options.index.symbol_extraction_reached = reached;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto workspace,
      Workspace::Open(*Env::Default(), temp.Join("idx"), options));
  absl::Cleanup release([gate] {
    if (!gate->HasBeenNotified()) gate->Notify();
  });
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id, workspace->Register(repo.root()));
  ABSL_ASSERT_OK(repo.WriteFile("pending.cc", "int PendingDefinition() {}\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t generation,
                              workspace->Notify(id, {"pending.cc"}));
  ExpectPaths(*workspace, id, "PendingDefinition", {"pending.cc"}, generation);
  EXPECT_TRUE(reached->WaitForNotificationWithTimeout(kWait));
  auto pending = workspace->FindSymbol(id, {.name = "PendingDefinition"},
                                       {.mode = SearchConsistencyMode::kSync,
                                        .min_generation = generation,
                                        .timeout = kWait});
  ASSERT_TRUE(pending.ok()) << pending.status();
  EXPECT_FALSE(pending->stale);
  EXPECT_EQ(pending->pending_files, 1);
  EXPECT_TRUE(pending->occurrences.empty());
  gate->Notify();
  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  do {
    pending = workspace->FindSymbol(id, {.name = "PendingDefinition"});
    ASSERT_TRUE(pending.ok()) << pending.status();
    if (pending->pending_files != 0) absl::SleepFor(absl::Milliseconds(1));
  } while (pending->pending_files != 0 && absl::Now() < deadline);
  EXPECT_EQ(pending->pending_files, 0);
  ASSERT_EQ(pending->occurrences.size(), 1u);
  EXPECT_EQ(pending->occurrences[0].path, "pending.cc");
}

TEST(WorkspaceTest, FindSymbolFiltersAndOrdersDefinitionsFirst) {
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("a.cc", "int Foo();\n"));
  ABSL_ASSERT_OK(repo.WriteFile(
      "b.cc",
      absl::StrCat("int ", std::string(300, ' '), "Foo() { return 1; }\n")));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("symbols"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto workspace, Workspace::Open(*Env::Default(), temp.Join("index")));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id, workspace->Register(repo.root()));
  FindSymbolRequest request{
      .name = "foo", .case_insensitive = true, .max_results = 1};
  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  absl::StatusOr<FindSymbolResponse> found;
  do {
    found = workspace->FindSymbol(id, request);
    ASSERT_TRUE(found.ok()) << found.status();
    if (found->pending_files != 0) absl::SleepFor(absl::Milliseconds(1));
  } while (found->pending_files != 0 && absl::Now() < deadline);
  ASSERT_EQ(found->pending_files, 0);
  EXPECT_EQ(found->total_count, 2);
  ASSERT_EQ(found->occurrences.size(), 1u);
  EXPECT_EQ(found->occurrences[0].path, "b.cc");
  EXPECT_TRUE(found->occurrences[0].is_definition);
  EXPECT_NE(found->occurrences[0].source_line.find("Foo"), std::string::npos);
  EXPECT_LE(found->occurrences[0].source_line.size(), 256u);
  request.definitions_only = true;
  request.max_results = 50;
  FERRET_ASSERT_OK_AND_ASSIGN(auto definitions,
                              workspace->FindSymbol(id, request));
  EXPECT_EQ(definitions.total_count, 1);
  request.definitions_only = false;
  request.path_glob = "a.cc";
  FERRET_ASSERT_OK_AND_ASSIGN(auto declarations,
                              workspace->FindSymbol(id, request));
  ASSERT_EQ(declarations.occurrences.size(), 1u);
  EXPECT_EQ(declarations.occurrences[0].source_line, "int Foo();");
  EXPECT_FALSE(declarations.occurrences[0].is_definition);
  request.path_glob.clear();
  request.kinds = {SymbolKind::kClass};
  FERRET_ASSERT_OK_AND_ASSIGN(auto classes, workspace->FindSymbol(id, request));
  EXPECT_EQ(classes.total_count, 0);
  request.kinds.clear();
  request.case_insensitive = false;
  FERRET_ASSERT_OK_AND_ASSIGN(auto mismatched,
                              workspace->FindSymbol(id, request));
  EXPECT_EQ(mismatched.total_count, 0);
}

TEST(WorkspaceTest, SeededFindSymbolMatchesExtractorAcrossEditsAndRestart) {
  const uint64_t seed = ferret_testing::TestSeed(0x3a94);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  const std::vector<std::string> paths = {"a.c", "b.cc", "c.py", "d.cc"};
  std::vector<std::string> contents(paths.size());
  const auto edit = [&](size_t i) {
    const std::string name = absl::StrCat("Example", rng() % 5);
    contents[i] = paths[i].ends_with(".py")
                      ? absl::StrCat("def ", name, "():\n    return 2\n")
                      : absl::StrCat("int ", name, "() { return 1; }\nint ",
                                     name, "();\n");
    ABSL_ASSERT_OK(repo.WriteFile(paths[i], contents[i]));
  };
  for (size_t i = 0; i < paths.size(); ++i) edit(i);
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  WorkspaceOptions options;
  options.index.memtable.flush_docs = 3;
  const auto verify = [&](Workspace& workspace, WorktreeId id,
                          const SearchConsistency& consistency) {
    SymbolExtractor extractor;
    std::vector<std::vector<ExtractedSymbol>> expected;
    for (size_t i = 0; i < paths.size(); ++i) {
      auto symbols =
          extractor.Extract(DetectLanguage(paths[i], contents[i]), contents[i]);
      ASSERT_TRUE(symbols.ok()) << symbols.status();
      expected.push_back(*symbols);
    }
    for (int query = 0; query < 12; ++query) {
      const std::string name =
          query % 3 == 0 ? "AbsentSymbol" : absl::StrCat("Example", rng() % 5);
      FindSymbolRequest request{.name = name};
      request.definitions_only = query % 2 == 0;
      request.path_glob = query % 4 == 0 ? "*.cc" : "";
      const absl::Time deadline = absl::Now() + absl::Seconds(10);
      absl::StatusOr<FindSymbolResponse> found;
      do {
        found = workspace.FindSymbol(id, request, consistency);
        ASSERT_TRUE(found.ok()) << found.status();
        if (found->pending_files != 0) absl::SleepFor(absl::Milliseconds(2));
      } while (found->pending_files != 0 && absl::Now() < deadline);
      ASSERT_EQ(found->pending_files, 0);
      std::vector<SymbolLocation> oracle;
      for (size_t i = 0; i < paths.size(); ++i) {
        if (!request.path_glob.empty() && !paths[i].ends_with(".cc")) continue;
        for (const ExtractedSymbol& symbol : expected[i]) {
          if (symbol.name != name ||
              (request.definitions_only && !symbol.is_definition)) {
            continue;
          }
          const size_t start =
              symbol.line == 1 ? 0 : contents[i].find('\n') + 1;
          const size_t end = contents[i].find('\n', start);
          oracle.push_back({paths[i], symbol.line, symbol.kind,
                            symbol.is_definition,
                            contents[i].substr(start, end - start)});
        }
      }
      const auto by_path = [](const SymbolLocation& a,
                              const SymbolLocation& b) {
        return std::tie(a.path, a.line, a.kind, a.is_definition) <
               std::tie(b.path, b.line, b.kind, b.is_definition);
      };
      std::sort(oracle.begin(), oracle.end(), by_path);
      std::vector<SymbolLocation> actual = found->occurrences;
      std::sort(actual.begin(), actual.end(), by_path);
      EXPECT_EQ(actual, oracle) << name;
      EXPECT_EQ(found->total_count, static_cast<int64_t>(oracle.size()));
    }
  };
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto workspace,
      Workspace::Open(*Env::Default(), temp.Join("idx"), options));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId id, workspace->Register(repo.root()));
  verify(*workspace, id, {});
  edit(1);
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t generation,
                              workspace->Notify(id, {paths[1]}));
  verify(*workspace, id,
         {.mode = SearchConsistencyMode::kSync,
          .min_generation = generation,
          .timeout = kWait});
  workspace->WaitForCompactions();
  workspace.reset();
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto reopened,
      Workspace::Open(*Env::Default(), temp.Join("idx"), options));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId restored,
                              reopened->Register(repo.root()));
  verify(*reopened, restored, {});
}

}  // namespace
}  // namespace ferret

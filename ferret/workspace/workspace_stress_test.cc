// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

// Runs concurrent edits, ingestion, searches, view churn, and zero-grace GC.
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/git/testing/git_repo_builder.h"
#include "ferret/search/search_types.h"
#include "ferret/testing/brute_force_search.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/testing/test_seed.h"
#include "ferret/workspace/workspace.h"

namespace ferret {
namespace {

using ::ferret_testing::GitRepoBuilder;
using ::ferret_testing::TempDir;

constexpr std::array<const char*, 3> kFiles{"live.cc", "created.cc",
                                            "transient.cc"};
constexpr int kEdits = 20;

std::string Value(int tree, int file, int revision) {
  return absl::StrCat("stressneedle", tree, "file", file, "rev", revision, "x");
}

SearchRequest Request() {
  return {.pattern = "stressneedle", .max_files = 100, .context_lines = 0};
}

struct StressTree {
  StressTree(const GitRepoBuilder& source, WorktreeId tree_id, int number)
      : repo(source), id(tree_id), number(number) {}
  GitRepoBuilder repo;
  WorktreeId id;
  int number;
  // Only the writer mutates a file. Notify's returned generation publishes
  // the revision floor to concurrent readers without locking out searches.
  std::array<std::atomic<int>, 3> floor{};
  std::atomic<uint64_t> generation{0};
};

void CheckSearch(Workspace& workspace, StressTree& tree,
                 SearchConsistencyMode mode) {
  const uint64_t generation = tree.generation.load();
  std::array<int, 3> floor;
  for (int file = 0; file < 3; ++file) floor[file] = tree.floor[file].load();
  auto result = workspace.Search(tree.id, Request(),
                                 {.mode = mode,
                                  .min_generation = generation,
                                  .timeout = absl::Seconds(30)});
  ASSERT_TRUE(result.ok()) << result.status();
  if (mode == SearchConsistencyMode::kSync) EXPECT_FALSE(result->stale);
  for (const FileResult& found : result->response.files) {
    auto it = std::find(kFiles.begin(), kFiles.end(), found.path);
    ASSERT_NE(it, kFiles.end());
    const int file = static_cast<int>(it - kFiles.begin());
    ASSERT_EQ(found.lines.size(), 1u);
    bool written = false;
    for (int revision = 0; revision <= kEdits; ++revision) {
      if (found.lines[0].text != Value(tree.number, file, revision)) continue;
      written = true;
      // kNone deliberately permits an older published snapshot. kSync must
      // see at least every Notify that finished before this search began.
      if (mode == SearchConsistencyMode::kSync) {
        EXPECT_GE(revision, floor[file]);
      }
    }
    EXPECT_TRUE(written) << found.lines[0].text;
  }
}

TEST(WorkspaceStressTest, LanesViewsGcAndSearches) {
  const uint64_t seed = ferret_testing::TestSeed(493);
  SCOPED_TRACE(seed);
  TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok()) << repo.status();
  ASSERT_TRUE(
      repo->WriteFile(kFiles[0], absl::StrCat(Value(0, 0, 0), "\n")).ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("base").ok());
  auto second = repo->AddWorktree("second");
  auto extra = repo->AddWorktree("extra");
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_TRUE(extra.ok()) << extra.status();
  WorkspaceOptions options;
  options.index.gc_grace_period = absl::ZeroDuration();
  options.index.memtable.flush_docs = 1;
  options.index.compaction.tier_min_segments = 2;
  auto opened =
      Workspace::Open(*Env::Default(), temp.Join("index"), std::move(options));
  ASSERT_TRUE(opened.ok()) << opened.status();
  std::unique_ptr<Workspace> workspace = std::move(*opened);
  auto first_id = workspace->Register(repo->root());
  auto second_id = workspace->Register(second->root());
  ASSERT_TRUE(first_id.ok()) << first_id.status();
  ASSERT_TRUE(second_id.ok()) << second_id.status();
  std::array<StressTree, 2> trees{
      {StressTree(*repo, *first_id, 0), StressTree(*second, *second_id, 1)}};
  // The second worktree initially contains the base revision's bytes.
  ASSERT_TRUE(
      second->WriteFile(kFiles[0], absl::StrCat(Value(1, 0, 0), "\n")).ok());
  auto initial = workspace->Notify(*second_id, {kFiles[0]});
  ASSERT_TRUE(initial.ok()) << initial.status();
  trees[1].generation.store(*initial);
  ASSERT_TRUE(
      workspace->AwaitGeneration(*second_id, *initial, absl::Seconds(30)).ok());
  std::atomic<bool> running{true};
  std::thread reader([&] {
    while (running.load()) {
      for (StressTree& tree : trees) {
        CheckSearch(*workspace, tree, SearchConsistencyMode::kNone);
        CheckSearch(*workspace, tree, SearchConsistencyMode::kSync);
        EXPECT_TRUE(workspace->WorktreeStatus(tree.id).ok());
      }
      const WorkspaceStats stats = workspace->Stats();
      EXPECT_GE(stats.index.live_docs + stats.index.dead_docs, 1u);
      EXPECT_FALSE(stats.index.compaction_disabled);
    }
  });
  std::thread churn([&] {
    for (int i = 0; i < 6; ++i) {
      auto id = workspace->Register(extra->root());
      ASSERT_TRUE(id.ok()) << id.status();
      EXPECT_TRUE(workspace->Unregister(*id).ok());
      workspace->WaitForCompactions();
    }
  });
  const auto write = [&](StressTree& tree, uint64_t local_seed) {
    std::mt19937_64 rng(local_seed);
    for (int revision = 1; revision <= kEdits; ++revision) {
      const int file = static_cast<int>(rng() % kFiles.size());
      // Alternate creates and deletes of untracked files, while the tracked
      // file is always replaced. The one writer per tree owns disk mutations.
      const bool remove = file != 0 && revision % 4 == 0;
      if (remove) {
        auto exists = Env::Default()->FileExists(tree.repo.Path(kFiles[file]));
        ASSERT_TRUE(exists.ok()) << exists.status();
        if (*exists) EXPECT_TRUE(tree.repo.DeleteFile(kFiles[file]).ok());
      } else {
        EXPECT_TRUE(
            tree.repo
                .WriteFile(
                    kFiles[file],
                    absl::StrCat(Value(tree.number, file, revision), "\n"))
                .ok());
      }
      auto generation = workspace->Notify(tree.id, {kFiles[file]});
      ASSERT_TRUE(generation.ok()) << generation.status();
      tree.floor[file].store(revision);
      tree.generation.store(*generation);
      auto result = workspace->Search(tree.id, Request(),
                                      {.mode = SearchConsistencyMode::kSync,
                                       .min_generation = *generation,
                                       .timeout = absl::Seconds(30)});
      ASSERT_TRUE(result.ok()) << result.status();
      EXPECT_FALSE(result->stale);
      const auto found = std::find_if(
          result->response.files.begin(), result->response.files.end(),
          [&](const FileResult& item) { return item.path == kFiles[file]; });
      if (remove) {
        EXPECT_EQ(found, result->response.files.end());
      } else {
        ASSERT_NE(found, result->response.files.end());
        ASSERT_EQ(found->lines.size(), 1u);
        EXPECT_EQ(found->lines[0].text, Value(tree.number, file, revision));
      }
    }
  };
  std::thread first([&] { write(trees[0], seed); });
  std::thread other([&] { write(trees[1], seed + 1); });
  first.join();
  other.join();
  running.store(false);
  reader.join();
  churn.join();
  workspace->WaitForCompactions();
  const IndexStats counters = workspace->Stats().index;
  EXPECT_GT(counters.gc_scans, 0u);
  EXPECT_GT(counters.completed_compactions, 0u);
  for (StressTree& tree : trees) {
    CheckSearch(*workspace, tree, SearchConsistencyMode::kSync);
    std::vector<ferret_testing::CorpusFile> corpus;
    for (const char* name : kFiles) {
      auto content = Env::Default()->ReadFile(tree.repo.Path(name));
      if (content.ok()) corpus.push_back({name, *content});
    }
    auto expected = ferret_testing::BruteForceSearch(corpus, Request());
    auto actual = workspace->Search(
        tree.id, Request(),
        {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(30)});
    ASSERT_TRUE(expected.ok()) << expected.status();
    ASSERT_TRUE(actual.ok()) << actual.status();
    ASSERT_EQ(actual->response.files.size(), expected->size());
    std::vector<std::pair<std::string, int64_t>> got;
    std::vector<std::pair<std::string, int64_t>> want;
    for (const FileResult& file : actual->response.files) {
      got.emplace_back(file.path, file.num_matching_lines);
    }
    for (const FileResult& file : *expected) {
      want.emplace_back(file.path, file.num_matching_lines);
    }
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    EXPECT_EQ(got, want);
  }
}

}  // namespace
}  // namespace ferret

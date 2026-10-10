// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
#include "ferret/engine/engine.h"

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/base/subprocess.h"
#include "ferret/git/testing/git_repo_builder.h"
#include "ferret/search/search_types.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/watch/fake_watcher.h"
#include "ferret/watch/watcher.h"
#include "ferret/workspace/workspace.h"

namespace ferret {
namespace {

TEST(EngineTest, CacheDirectoryFollowsBothPlatformRules) {
  EXPECT_EQ(CacheDirectory("/home/test", "/xdg", true),
            "/home/test/Library/Caches");
  EXPECT_EQ(CacheDirectory("/home/test", "/xdg", false), "/xdg");
  EXPECT_EQ(CacheDirectory("/home/test", "", false), "/home/test/.cache");
}

TEST(EngineTest, RegistersPlainDirectory) {
  ferret_testing::TempDir temp;
  ASSERT_TRUE(temp.MakeDirs("plain").ok());
  ASSERT_TRUE(temp.WriteFile("plain/a.cc", "plainneedle\n").ok());
  Engine engine(*Env::Default(), temp.Join("index"));
  ASSERT_TRUE(engine.Register(temp.Join("plain")).ok());
  SearchRequest request;
  request.pattern = "plainneedle";
  auto result = engine.Search(temp.Join("plain"), request, {});
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->response.files.size(), size_t{1});
  EXPECT_EQ(result->response.files[0].path, "a.cc");
}

TEST(EngineTest, WorkspaceWatcherSynchronizesFirstSearch) {
  ferret_testing::TempDir temp;
  ASSERT_TRUE(temp.MakeDirs("plain").ok());
  ASSERT_TRUE(temp.WriteFile("plain/a.cc", "beforetoken\n").ok());
  FakeWatcher* watcher = nullptr;
  EngineOptions options;
  options.index_dir = temp.Join("index");
  options.workspace.watcher_factory =
      std::make_shared<WorkspaceOptions::WatcherFactory>(
          [&](const std::string& root, const WatcherOptions& watcher_options)
              -> absl::StatusOr<std::unique_ptr<Watcher>> {
            auto opened = FakeWatcher::Open(root, watcher_options);
            if (!opened.ok()) return opened.status();
            watcher = opened->get();
            return std::unique_ptr<Watcher>(std::move(*opened));
          });
  Engine engine(*Env::Default(), std::move(options));
  ASSERT_TRUE(engine.Register(temp.Join("plain")).ok());
  ASSERT_NE(watcher, nullptr);
  ASSERT_TRUE(temp.WriteFile("plain/a.cc", "aftertoken\n").ok());
  watcher->Push(PathChanged{.rel_path = "a.cc"});
  SearchRequest request;
  request.pattern = "aftertoken";
  auto result = engine.Search(
      temp.Join("plain"), request,
      {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_FALSE(result->stale);
  ASSERT_EQ(result->response.files.size(), size_t{1});
  auto status = engine.IndexStatus();
  ASSERT_TRUE(status.ok()) << status.status();
  EXPECT_TRUE((*status)[0].watcher_running);
}

TEST(EngineTest, RegistrationDoesNotBlockOtherRootsOrReserveAfterFailure) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok());
  ASSERT_TRUE(repo->WriteFile("a.cc", "firstneedle\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("base").ok());
  auto sibling = repo->AddWorktree("sibling");
  ASSERT_TRUE(sibling.ok());
  ASSERT_TRUE(sibling->WriteFile("slow.cc", "secondneedle\n").ok());
  auto other = repo->AddWorktree("other");
  ASSERT_TRUE(other.ok());
  absl::Notification paused;
  absl::Notification release;
  std::atomic<bool> notified = false;
  EngineOptions options;
  options.index_dir = temp.Join("index");
  options.workspace.read_hook = std::make_shared<WorkspaceReadHook>(
      [&](absl::string_view path, bool bulk) {
        if (bulk && path == "slow.cc") {
          if (!notified.exchange(true)) paused.Notify();
          release.WaitForNotification();
        }
      });
  Engine engine(*Env::Default(), std::move(options));
  ASSERT_TRUE(engine.Register(repo->root()).ok());
  absl::Status registration;
  std::thread worker([&] { registration = engine.Register(sibling->root()); });
  ASSERT_TRUE(paused.WaitForNotificationWithTimeout(absl::Seconds(10)));
  EXPECT_EQ(engine.Register(sibling->root()).code(),
            absl::StatusCode::kAlreadyExists);
  absl::Status other_status;
  std::thread other_worker(
      [&] { other_status = engine.Register(other->root()); });
  SearchRequest request;
  request.pattern = "firstneedle";
  absl::Notification searched;
  std::thread searcher([&] {
    auto result = engine.Search(repo->root(), request, {});
    EXPECT_TRUE(result.ok()) << result.status();
    if (result.ok()) EXPECT_EQ(result->response.files.size(), size_t{1});
    searched.Notify();
  });
  EXPECT_TRUE(searched.WaitForNotificationWithTimeout(absl::Seconds(5)));
  EXPECT_EQ(engine.Roots().size(), size_t{1});
  release.Notify();
  searcher.join();
  worker.join();
  other_worker.join();
  EXPECT_TRUE(registration.ok()) << registration;
  EXPECT_TRUE(other_status.ok()) << other_status;
  EXPECT_EQ(engine.Roots().size(), size_t{3});
}

TEST(EngineTest, FailedRegistrationReleasesReservationForRetry) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok());
  ASSERT_TRUE(temp.MakeDirs("repo/nested").ok());
  {
    Engine engine(*Env::Default(), temp.Join("index"));
    EXPECT_EQ(engine.Register(temp.Join("repo/nested")).code(),
              absl::StatusCode::kInvalidArgument);
    auto initialized =
        RunSubprocess({"git", "-C", temp.Join("repo/nested"), "init"});
    ASSERT_TRUE(initialized.ok()) << initialized.status();
    ASSERT_EQ(initialized->exit_code, 0);
    const absl::Status retried = engine.Register(temp.Join("repo/nested"));
    EXPECT_TRUE(retried.ok()) << retried;
    EXPECT_EQ(engine.Roots().size(), size_t{1});
    EXPECT_TRUE(engine.Register(temp.Join("repo/nested")).ok());
  }
  // The failed call left no second index owner or leaked LOCK on shutdown.
  Engine reopened(*Env::Default(), temp.Join("index"));
  ASSERT_TRUE(reopened.OpenRepository(repo->root()).ok());
  ASSERT_TRUE(reopened.Register(temp.Join("repo/nested")).ok());
}

TEST(EngineTest, IndexDirectoryOverrideHoldsOneIndexPerRepository) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto first = ferret_testing::GitRepoBuilder::Init(temp, "first");
  auto second = ferret_testing::GitRepoBuilder::Init(temp, "second");
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_TRUE(first->WriteFile("a.cc", "firstrepo\n").ok());
  ASSERT_TRUE(second->WriteFile("b.cc", "secondrepo\n").ok());
  // Each repository's workspace locks its own index directory.
  Engine engine(*Env::Default(), temp.Join("index"));
  const absl::Status registered_first = engine.Register(first->root());
  const absl::Status registered_second = engine.Register(second->root());
  ASSERT_TRUE(registered_first.ok()) << registered_first;
  ASSERT_TRUE(registered_second.ok()) << registered_second;
  SearchRequest request;
  request.pattern = "secondrepo";
  auto found = engine.Search(second->root(), request, {});
  ASSERT_TRUE(found.ok()) << found.status();
  ASSERT_EQ(found->response.files.size(), size_t{1});
  EXPECT_EQ(found->response.files[0].path, "b.cc");
  auto missing = engine.Search(first->root(), request, {});
  ASSERT_TRUE(missing.ok()) << missing.status();
  EXPECT_TRUE(missing->response.files.empty());
}

TEST(EngineTest, SharesIndexAcrossWorktreesAndSearchesViews) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok()) << repo.status();
  ASSERT_TRUE(repo->WriteFile("one.cc", "alpha unique\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("first").ok());
  auto sibling = repo->AddWorktree("sibling");
  ASSERT_TRUE(sibling.ok()) << sibling.status();
  ASSERT_TRUE(sibling->WriteFile("two.cc", "beta unique\n").ok());
  ASSERT_TRUE(Env::Default()->CreateDir(temp.Join("index")).ok());
  Engine engine(*Env::Default(), temp.Join("index"));
  ASSERT_TRUE(engine.Register(repo->root()).ok());
  ASSERT_TRUE(engine.Register(sibling->root()).ok());
  EXPECT_EQ(engine.Roots().size(), size_t{2});
  EXPECT_TRUE(engine.Register(repo->root()).ok());
  SearchRequest request;
  request.pattern = "beta unique";
  SearchConsistency consistency;
  consistency.mode = kServingConsistency;
  auto first = engine.Search(repo->root(), request, consistency);
  auto second = engine.Search(sibling->root(), request, consistency);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_TRUE(first->response.files.empty());
  ASSERT_EQ(second->response.files.size(), size_t{1});
  EXPECT_EQ(second->response.files[0].path, "two.cc");
  auto status = engine.IndexStatus();
  ASSERT_TRUE(status.ok()) << status.status();
  ASSERT_EQ(status->size(), size_t{2});
  EXPECT_EQ((*status)[0].index_bytes, (*status)[1].index_bytes);
  EXPECT_EQ((*status)[0].segment_count, (*status)[1].segment_count);
}

TEST(EngineTest, CacheDirHashesCanonicalCommonDirectory) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok());
  ASSERT_TRUE(repo->Commit("first").ok());
  auto sibling = repo->AddWorktree("sibling");
  ASSERT_TRUE(sibling.ok());
  auto first = RepositoryCacheDir(repo->root());
  auto second = RepositoryCacheDir(sibling->root());
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(*first, *second);
  EXPECT_EQ(first->substr(first->size() - 16).size(), size_t{16});
  EXPECT_EQ(GitCommonDir(repo->root()), GitCommonDir(sibling->root()));
  EXPECT_EQ(GitCommonDir(temp.path()).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(EngineTest, ListsRegistrationsThenRestoredViewsInManifestOrder) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok());
  ASSERT_TRUE(repo->WriteFile("base.cc", "orderneedle\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("base").ok());
  auto b = repo->AddWorktree("b");
  auto c = repo->AddWorktree("c");
  auto d = repo->AddWorktree("d");
  ASSERT_TRUE(b.ok() && c.ok() && d.ok());
  {
    Engine initial(*Env::Default(), temp.Join("index"));
    ASSERT_TRUE(initial.Register(c->root()).ok());
    ASSERT_TRUE(initial.Register(repo->root()).ok());
    ASSERT_TRUE(initial.Register(b->root()).ok());
    EXPECT_EQ(initial.Roots(),
              (std::vector<std::string>{c->root(), repo->root(), b->root()}));
  }
  Engine restored(*Env::Default(), temp.Join("index"));
  ASSERT_TRUE(restored.OpenRepository(repo->root()).ok());
  for (int i = 0; i < 3000 && restored.Roots().size() != 3; ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_EQ(restored.Roots(),
            (std::vector<std::string>{c->root(), repo->root(), b->root()}));
  ASSERT_TRUE(restored.Register(d->root()).ok());
  ASSERT_TRUE(restored.Register(repo->root()).ok());
  ASSERT_TRUE(restored.Register(d->root()).ok());
  const std::vector<std::string> expected = {d->root(), repo->root(), c->root(),
                                             b->root()};
  EXPECT_EQ(restored.Roots(), expected);
  auto statuses = restored.IndexStatus();
  ASSERT_TRUE(statuses.ok()) << statuses.status();
  ASSERT_EQ(statuses->size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ((*statuses)[i].root, expected[i]);
  }
}

TEST(EngineTest, RestoredRootsPublishBeforeClientRegistration) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok()) << repo.status();
  ASSERT_TRUE(repo->WriteFile("base.cc", "enginerestore needle\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("base").ok());
  auto sibling = repo->AddWorktree("sibling");
  ASSERT_TRUE(sibling.ok()) << sibling.status();
  ASSERT_TRUE(sibling->WriteFile("edit.cc", "enginerestore edit\n").ok());
  {
    Engine initial(*Env::Default(), temp.Join("index"));
    ASSERT_TRUE(initial.Register(repo->root()).ok());
    ASSERT_TRUE(initial.Register(sibling->root()).ok());
  }
  std::atomic<int> reads{0};
  EngineOptions options;
  options.index_dir = temp.Join("index");
  options.workspace.index.gc_grace_period = absl::ZeroDuration();
  options.workspace.read_hook =
      std::make_shared<WorkspaceReadHook>([&](absl::string_view, bool bulk) {
        if (bulk) ++reads;
      });
  Engine restored(*Env::Default(), std::move(options));
  ASSERT_TRUE(restored.OpenRepository(repo->root()).ok());
  for (int i = 0; i < 3000 && restored.Roots().size() != 2; ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  EXPECT_EQ(restored.Roots().size(), size_t{2});
  const int after_restore = reads.load();
  ASSERT_TRUE(restored.Register(repo->root()).ok());
  ASSERT_TRUE(restored.Register(sibling->root()).ok());
  EXPECT_EQ(reads.load(), after_restore);
  EXPECT_EQ(restored.Roots().size(), size_t{2});
  SearchRequest request;
  request.pattern = "enginerestore";
  auto found = restored.Search(sibling->root(), request, {});
  ASSERT_TRUE(found.ok()) << found.status();
  EXPECT_EQ(found->response.files.size(), size_t{2});
}

}  // namespace
}  // namespace ferret

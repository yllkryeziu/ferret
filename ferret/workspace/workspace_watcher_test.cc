// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/git/testing/git_repo_builder.h"
#include "ferret/search/search_types.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/watch/fake_watcher.h"
#include "ferret/watch/watcher.h"
#include "ferret/workspace/workspace.h"

namespace ferret {
namespace {

using ::ferret_testing::GitRepoBuilder;
using ::ferret_testing::TempDir;

SearchRequest Request(const std::string& token) {
  SearchRequest request;
  request.pattern = token;
  request.context_lines = 0;
  return request;
}

bool Has(Workspace& workspace, WorktreeId id, const std::string& token,
         SearchConsistency consistency = {}) {
  auto found = workspace.Search(id, Request(token), consistency);
  EXPECT_TRUE(found.ok()) << found.status();
  if (!found.ok()) return false;
  if (consistency.mode == SearchConsistencyMode::kSync) {
    EXPECT_FALSE(found->stale);
  }
  return !found->response.files.empty();
}

uint64_t SyncGeneration(Workspace& workspace, WorktreeId id) {
  auto result = workspace.Search(
      id, Request("generationtoken"),
      {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)});
  EXPECT_TRUE(result.ok()) << result.status();
  if (!result.ok()) return 0;
  EXPECT_FALSE(result->stale);
  return result->generation;
}

// Holds the first sync open so another search can test its own deadline.
// Thread-safe.
class BlockingSyncWatcher final : public Watcher {
 public:
  explicit BlockingSyncWatcher(std::string root) : root_(std::move(root)) {}
  BlockingSyncWatcher(const BlockingSyncWatcher&) = delete;
  BlockingSyncWatcher& operator=(const BlockingSyncWatcher&) = delete;
  BlockingSyncWatcher(BlockingSyncWatcher&&) = delete;
  BlockingSyncWatcher& operator=(BlockingSyncWatcher&&) = delete;
  ~BlockingSyncWatcher() override { Stop(); }

  absl::Status Start(WatchCallback callback) override {
    if (callback == nullptr) return absl::InvalidArgumentError("null callback");
    return absl::OkStatus();
  }
  void Stop() override { Release(); }
  void Release() {
    if (!released_.exchange(true)) release.Notify();
  }
  absl::Status Sync(absl::Duration timeout) override {
    if (!entered_.exchange(true)) entered.Notify();
    if (!release.WaitForNotificationWithTimeout(timeout)) {
      return absl::DeadlineExceededError("blocked sync");
    }
    return absl::OkStatus();
  }
  const std::string& root() const override { return root_; }

  absl::Notification entered;
  absl::Notification release;

 private:
  std::string root_;
  std::atomic<bool> released_{false};
  std::atomic<bool> entered_{false};
};

class WatchFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    ferret_testing::IsolateGitEnvironment(temp_.Join("home"));
    FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                                GitRepoBuilder::Init(temp_, "repo"));
    repo_ = std::make_unique<GitRepoBuilder>(std::move(repo));
    ABSL_ASSERT_OK(repo_->WriteFile("base.cc", "basetoken\n"));
    ABSL_ASSERT_OK(repo_->WriteFile(".gitignore", "*.log\nignored/\n"));
    ABSL_ASSERT_OK(repo_->Add());
    ABSL_ASSERT_OK(repo_->Commit("base"));
  }

  WorkspaceOptions Options() {
    WorkspaceOptions options;
    options.watcher_factory =
        std::make_shared<WorkspaceOptions::WatcherFactory>(
            [this](const std::string& root, const WatcherOptions& settings)
                -> absl::StatusOr<std::unique_ptr<Watcher>> {
              auto fake = FakeWatcher::Open(root, settings);
              if (!fake.ok()) return fake.status();
              fake_ = fake->get();
              return std::unique_ptr<Watcher>(std::move(*fake));
            });
    return options;
  }

  void Open(WorkspaceOptions options) {
    FERRET_ASSERT_OK_AND_ASSIGN(
        workspace_, Workspace::Open(*Env::Default(), temp_.Join("index"),
                                    std::move(options)));
    FERRET_ASSERT_OK_AND_ASSIGN(
        id_, workspace_->Register(repo_->root(), {.watch = true}));
    ASSERT_NE(fake_.load(), nullptr);
  }

  void Event(WatchEvent event) {
    fake_.load()->Push(std::move(event));
    ABSL_ASSERT_OK(fake_.load()->Sync(absl::Seconds(5)));
  }

  TempDir temp_;
  std::unique_ptr<GitRepoBuilder> repo_;
  std::atomic<FakeWatcher*> fake_{nullptr};
  std::unique_ptr<Workspace> workspace_;
  WorktreeId id_{};
};

TEST_F(WatchFixture, PathEventsAndIgnoredAndCookieFiles) {
  Open(Options());
  ABSL_ASSERT_OK(repo_->WriteFile("new.cc", "newtoken\n"));
  EXPECT_FALSE(Has(*workspace_, id_, "newtoken"));
  Event(PathChanged{"new.cc"});
  EXPECT_TRUE(
      Has(*workspace_, id_, "newtoken",
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)}));
  const uint64_t before_ignore = SyncGeneration(*workspace_, id_);
  ABSL_ASSERT_OK(repo_->WriteFile("ignored.log", "ignoredtoken\n"));
  Event(PathChanged{"ignored.log"});
  EXPECT_EQ(SyncGeneration(*workspace_, id_), before_ignore);
  EXPECT_FALSE(Has(*workspace_, id_, "ignoredtoken",
                   {.mode = SearchConsistencyMode::kSync}));
  ABSL_ASSERT_OK(repo_->WriteFile(".git/ferret/cookie-test", "cookietoken\n"));
  Event(PathChanged{".git/ferret/cookie-test"});
  EXPECT_EQ(SyncGeneration(*workspace_, id_), before_ignore);
  EXPECT_FALSE(Has(*workspace_, id_, "cookietoken",
                   {.mode = SearchConsistencyMode::kSync}));
  ABSL_ASSERT_OK(repo_->DeleteFile("new.cc"));
  Event(PathChanged{"new.cc"});
  EXPECT_FALSE(
      Has(*workspace_, id_, "newtoken",
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)}));
}

TEST_F(WatchFixture, SubtreesAndGitMetadata) {
  Open(Options());
  ABSL_ASSERT_OK(repo_->WriteFile("dir/nested.cc", "subtreetoken\n"));
  Event(SubtreeChanged{"dir"});
  EXPECT_TRUE(
      Has(*workspace_, id_, "subtreetoken",
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)}));
  const uint64_t before_metadata = SyncGeneration(*workspace_, id_);
  ABSL_ASSERT_OK(repo_->WriteFile("base.cc", "gitignoredtoken\n"));
  Event(PathChanged{".git/config"});
  EXPECT_EQ(SyncGeneration(*workspace_, id_), before_metadata);
  EXPECT_FALSE(Has(*workspace_, id_, "gitignoredtoken"));
  Event(PathChanged{".git/HEAD"});
  EXPECT_TRUE(
      Has(*workspace_, id_, "gitignoredtoken",
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)}));
  ABSL_ASSERT_OK(repo_->WriteFile("base.cc", "indexeventtoken\n"));
  Event(PathChanged{".git/index"});
  EXPECT_TRUE(
      Has(*workspace_, id_, "indexeventtoken",
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)}));
  ABSL_ASSERT_OK(repo_->DeleteFile("dir/nested.cc"));
  Event(SubtreeChanged{""});
  EXPECT_FALSE(
      Has(*workspace_, id_, "subtreetoken",
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)}));
}

// Keep polling for the background lane rather than imposing a short fixed
// number of attempts. Sanitizer builds can delay the lane substantially.
absl::Duration WatchConvergenceBudget() {
#if defined(__has_feature)
#if __has_feature(thread_sanitizer) || __has_feature(address_sanitizer)
  return absl::Seconds(30);
#endif
#endif
  return absl::Seconds(5);
}

TEST_F(WatchFixture, IndexLockDefersRescanButNotInteractivePaths) {
  Open(Options());
  ABSL_ASSERT_OK(repo_->WriteFile(".git/index.lock", "lock\n"));
  ABSL_ASSERT_OK(repo_->WriteFile("base.cc", "lockedtoken\n"));
  Event(PathChanged{".git/index"});
  auto stale = workspace_->Search(id_, Request("lockedtoken"),
                                  {.mode = SearchConsistencyMode::kSync,
                                   .timeout = absl::Milliseconds(20)});
  ASSERT_TRUE(stale.ok()) << stale.status();
  EXPECT_TRUE(stale->stale);
  EXPECT_TRUE(stale->response.files.empty());
  ABSL_ASSERT_OK(repo_->WriteFile("interactive.cc", "interactivetoken\n"));
  Event(PathChanged{"interactive.cc"});
  bool interactive = false;
  const absl::Time interactive_deadline =
      absl::Now() + WatchConvergenceBudget();
  while (!interactive && absl::Now() < interactive_deadline) {
    interactive = Has(*workspace_, id_, "interactivetoken");
    if (!interactive) absl::SleepFor(absl::Milliseconds(5));
  }
  EXPECT_TRUE(interactive);
  ABSL_ASSERT_OK(repo_->DeleteFile(".git/index.lock"));
  EXPECT_TRUE(Has(*workspace_, id_, "lockedtoken",
                  {.mode = SearchConsistencyMode::kSync,
                   .timeout = WatchConvergenceBudget()}));
}

TEST_F(WatchFixture, StartFailureIsReportedAndFallsBackToRescans) {
  auto options = Options();
  options.watcher_factory = std::make_shared<WorkspaceOptions::WatcherFactory>(
      [](const std::string&,
         const WatcherOptions&) -> absl::StatusOr<std::unique_ptr<Watcher>> {
        return absl::UnavailableError("synthetic watcher failure");
      });
  options.watch_failure_rescan_interval = absl::Milliseconds(20);
  FERRET_ASSERT_OK_AND_ASSIGN(
      workspace_, Workspace::Open(*Env::Default(), temp_.Join("index"),
                                  std::move(options)));
  FERRET_ASSERT_OK_AND_ASSIGN(
      id_, workspace_->Register(repo_->root(), {.watch = true}));
  EXPECT_EQ(workspace_->Stats().watcher_failures, uint64_t{1});
  EXPECT_EQ(workspace_->Stats().last_watcher_error.code(),
            absl::StatusCode::kUnavailable);
  auto stale = workspace_->Search(id_, Request("basetoken"),
                                  {.mode = SearchConsistencyMode::kSync});
  ASSERT_TRUE(stale.ok()) << stale.status();
  EXPECT_TRUE(stale->stale);
  ABSL_ASSERT_OK(repo_->WriteFile("base.cc", "fallbacktoken\n"));
  bool found = false;
  const absl::Time fallback_deadline = absl::Now() + WatchConvergenceBudget();
  while (!found && absl::Now() < fallback_deadline) {
    found = Has(*workspace_, id_, "fallbacktoken");
    if (!found) absl::SleepFor(absl::Milliseconds(20));
  }
  EXPECT_TRUE(found);
}

TEST_F(WatchFixture, IgnoredSubtreeAndIgnoreRuleEdit) {
  Open(Options());
  const uint64_t before_ignore = SyncGeneration(*workspace_, id_);
  ABSL_ASSERT_OK(repo_->WriteFile("ignored/hidden.cc", "hiddentoken\n"));
  Event(SubtreeChanged{"ignored"});
  EXPECT_EQ(SyncGeneration(*workspace_, id_), before_ignore);
  EXPECT_FALSE(
      Has(*workspace_, id_, "hiddentoken",
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)}));
  ABSL_ASSERT_OK(repo_->WriteFile(".gitignore", "*.log\n"));
  Event(PathChanged{".gitignore"});
  Event(SubtreeChanged{"ignored"});
  EXPECT_TRUE(
      Has(*workspace_, id_, "hiddentoken",
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)}));
}

TEST_F(WatchFixture, TrackedFilesMatchingIgnoreRulesStayWatched) {
  ABSL_ASSERT_OK(repo_->WriteFile("tracked.log", "trackedlogold\n"));
  ABSL_ASSERT_OK(repo_->WriteFile("ignored/kept.cc", "trackedkeptold\n"));
  ABSL_ASSERT_OK(
      repo_->Git({"add", "-f", "--", "tracked.log", "ignored/kept.cc"}));
  ABSL_ASSERT_OK(repo_->Commit("tracked despite ignore rules"));
  Open(Options());
  ABSL_ASSERT_OK(repo_->WriteFile("tracked.log", "trackedlognew\n"));
  Event(PathChanged{"tracked.log"});
  EXPECT_TRUE(
      Has(*workspace_, id_, "trackedlognew",
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)}));
  ABSL_ASSERT_OK(repo_->WriteFile("ignored/kept.cc", "trackedkeptnew\n"));
  Event(SubtreeChanged{"ignored"});
  EXPECT_TRUE(
      Has(*workspace_, id_, "trackedkeptnew",
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)}));
}

TEST_F(WatchFixture, IgnoreEditInSameBatchInvalidatesCachedRules) {
  Open(Options());
  ABSL_ASSERT_OK(repo_->WriteFile("ignored.log", "newlyallowedtoken\n"));
  Event(PathChanged{"ignored.log"});
  EXPECT_FALSE(Has(*workspace_, id_, "newlyallowedtoken"));
  ABSL_ASSERT_OK(repo_->WriteFile(".gitignore", "ignored/\n"));
  fake_.load()->Push(PathChanged{"ignored.log"});
  fake_.load()->Push(PathChanged{".gitignore"});
  ABSL_ASSERT_OK(fake_.load()->Sync(absl::Seconds(5)));
  EXPECT_TRUE(
      Has(*workspace_, id_, "newlyallowedtoken",
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)}));
}

TEST_F(WatchFixture, SyncFailureIsStaleAndStartsFallback) {
  Open(Options());
  fake_.load()->Stop();
  auto stale = workspace_->Search(id_, Request("basetoken"),
                                  {.mode = SearchConsistencyMode::kSync,
                                   .timeout = absl::Milliseconds(10)});
  ASSERT_TRUE(stale.ok()) << stale.status();
  EXPECT_TRUE(stale->stale);
  EXPECT_EQ(workspace_->Stats().watcher_failures, uint64_t{1});
  EXPECT_EQ(workspace_->Stats().last_watcher_error.code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST_F(WatchFixture, StoppedWatcherIsReplacedAfterFallbackScan) {
  auto options = Options();
  options.watch_failure_rescan_interval = absl::Milliseconds(20);
  auto factory = options.watcher_factory;
  std::atomic<int> starts{0};
  options.watcher_factory = std::make_shared<WorkspaceOptions::WatcherFactory>(
      [factory, &starts](const std::string& root,
                         const WatcherOptions& settings)
          -> absl::StatusOr<std::unique_ptr<Watcher>> {
        ++starts;
        return (*factory)(root, settings);
      });
  Open(std::move(options));
  fake_.load()->Stop();
  auto stale = workspace_->Search(id_, Request("basetoken"),
                                  {.mode = SearchConsistencyMode::kSync});
  ASSERT_TRUE(stale.ok()) << stale.status();
  EXPECT_TRUE(stale->stale);
  for (int i = 0; i < 200 && starts.load() < 2; ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_GE(starts.load(), 2);
  bool fresh = false;
  for (int i = 0; i < 100 && !fresh; ++i) {
    auto result = workspace_->Search(
        id_, Request("basetoken"),
        {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)});
    ASSERT_TRUE(result.ok()) << result.status();
    fresh = !result->stale;
    if (!fresh) absl::SleepFor(absl::Milliseconds(10));
  }
  EXPECT_TRUE(fresh);
  EXPECT_EQ(workspace_->Stats().watcher_failures, uint64_t{1});
  ABSL_ASSERT_OK(repo_->WriteFile("base.cc", "afterreplacement\n"));
  Event(PathChanged{"base.cc"});
  EXPECT_TRUE(
      Has(*workspace_, id_, "afterreplacement",
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)}));
}

TEST_F(WatchFixture, RepeatedStartFailuresRecoverAfterReplacementScan) {
  std::atomic<int> failures{3};
  auto options = Options();
  options.watch_failure_rescan_interval = absl::Milliseconds(30);
  options.watcher_factory = std::make_shared<WorkspaceOptions::WatcherFactory>(
      [this, &failures](const std::string& root, const WatcherOptions& settings)
          -> absl::StatusOr<std::unique_ptr<Watcher>> {
        if (failures.fetch_sub(1) > 0) {
          return absl::UnavailableError("synthetic start failure");
        }
        auto fake = FakeWatcher::Open(root, settings);
        if (!fake.ok()) return fake.status();
        fake_ = fake->get();
        return std::unique_ptr<Watcher>(std::move(*fake));
      });
  FERRET_ASSERT_OK_AND_ASSIGN(
      workspace_, Workspace::Open(*Env::Default(), temp_.Join("index"),
                                  std::move(options)));
  FERRET_ASSERT_OK_AND_ASSIGN(
      id_, workspace_->Register(repo_->root(), {.watch = true}));
  EXPECT_TRUE(workspace_
                  ->Search(id_, Request("basetoken"),
                           {.mode = SearchConsistencyMode::kSync})
                  ->stale);
  for (int i = 0; i < 200 && fake_.load() == nullptr; ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_NE(fake_.load(), nullptr);
  EXPECT_EQ(workspace_->Stats().watcher_failures, uint64_t{3});
  bool fresh = false;
  for (int i = 0; i < 100 && !fresh; ++i) {
    auto result = workspace_->Search(
        id_, Request("basetoken"),
        {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)});
    ASSERT_TRUE(result.ok()) << result.status();
    fresh = !result->stale;
    if (!fresh) absl::SleepFor(absl::Milliseconds(10));
  }
  EXPECT_TRUE(fresh);
  ABSL_ASSERT_OK(repo_->WriteFile("base.cc", "recoveredtoken\n"));
  Event(PathChanged{"base.cc"});
  EXPECT_TRUE(
      Has(*workspace_, id_, "recoveredtoken",
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)}));
}

TEST_F(WatchFixture, RegistrationDoesNotWaitForeverForStaleIndexLock) {
  auto options = Options();
  options.index_lock_wait = absl::Milliseconds(30);
  ABSL_ASSERT_OK(repo_->WriteFile(".git/index.lock", "stale lock\n"));
  const absl::Time started = absl::Now();
  Open(std::move(options));
  EXPECT_LT(absl::Now() - started, absl::Seconds(5));
  EXPECT_GE(workspace_->Stats().index_lock_timeouts, uint64_t{1});
  EXPECT_TRUE(
      Has(*workspace_, id_, "basetoken",
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)}));
}

TEST_F(WatchFixture, DeferredRescanDoesNotWaitForeverForStaleIndexLock) {
  auto options = Options();
  options.index_lock_wait = absl::Milliseconds(30);
  Open(std::move(options));
  ABSL_ASSERT_OK(repo_->WriteFile(".git/index.lock", "stale lock\n"));
  ABSL_ASSERT_OK(repo_->WriteFile("base.cc", "deferredtoken\n"));
  Event(PathChanged{".git/HEAD"});
  EXPECT_TRUE(
      Has(*workspace_, id_, "deferredtoken",
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)}));
  EXPECT_GE(workspace_->Stats().index_lock_timeouts, uint64_t{1});
}

TEST_F(WatchFixture, IgnoredEventsDoNotInternPaths) {
  Open(Options());
  const size_t before = workspace_->interned_paths();
  for (int i = 0; i < 1000; ++i) {
    fake_.load()->Push(PathChanged{absl::StrCat("ignored/", i, ".cc")});
    fake_.load()->Push(PathChanged{absl::StrCat(".git/objects/", i)});
    fake_.load()->Push(PathChanged{absl::StrCat("ignored/", i, ".log")});
  }
  ABSL_ASSERT_OK(fake_.load()->Sync(absl::Seconds(5)));
  EXPECT_EQ(workspace_->interned_paths(), before);
}

TEST_F(WatchFixture, LockedWorktreeDoesNotBlockOtherDeferredRescan) {
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder second,
                              GitRepoBuilder::Init(temp_, "second"));
  ABSL_ASSERT_OK(second.WriteFile("file.cc", "othertoken\n"));
  ABSL_ASSERT_OK(second.Add());
  ABSL_ASSERT_OK(second.Commit("initial"));
  // The first worktree's stale lock holds its rescan for index_lock_wait.
  // Blocking behind it would take 30 s; the 15 s bound leaves room for slow
  // machines (background QoS stretches the 10 ms lock-poll timers).
  auto options = Options();
  options.index_lock_wait = absl::Seconds(30);
  Open(std::move(options));
  FakeWatcher* first_fake = fake_.load();
  FERRET_ASSERT_OK_AND_ASSIGN(
      WorktreeId second_id,
      workspace_->Register(second.root(), {.watch = true}));
  ABSL_ASSERT_OK(repo_->WriteFile(".git/index.lock", "stale lock\n"));
  ABSL_ASSERT_OK(second.WriteFile(".git/index.lock", "temporary lock\n"));
  ABSL_ASSERT_OK(second.WriteFile("file.cc", "otherchangedtoken\n"));
  // Both worktrees have pending deferrals; only the second lock is removed.
  first_fake->Push(PathChanged{".git/HEAD"});
  ABSL_ASSERT_OK(first_fake->Sync(absl::Seconds(5)));
  Event(PathChanged{".git/index"});
  ABSL_ASSERT_OK(second.DeleteFile(".git/index.lock"));
  const absl::Time start = absl::Now();
  EXPECT_TRUE(Has(
      *workspace_, second_id, "otherchangedtoken",
      {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(10)}));
  EXPECT_LT(absl::Now() - start, absl::Seconds(15));
}

TEST_F(WatchFixture, FailedWorktreeDoesNotDelayAnotherDeferredRescan) {
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder second,
                              GitRepoBuilder::Init(temp_, "second"));
  ABSL_ASSERT_OK(second.WriteFile("file.cc", "othertoken\n"));
  ABSL_ASSERT_OK(second.Add());
  ABSL_ASSERT_OK(second.Commit("initial"));
  // Waiting for the failed root's next rescan would take 30 s; the 15 s
  // bound below leaves room for slow machines.
  auto options = Options();
  auto factory = options.watcher_factory;
  options.watch_failure_rescan_interval = absl::Seconds(30);
  options.index_lock_wait = absl::Seconds(30);
  options.watcher_factory = std::make_shared<WorkspaceOptions::WatcherFactory>(
      [this, factory](const std::string& root, const WatcherOptions& settings)
          -> absl::StatusOr<std::unique_ptr<Watcher>> {
        if (root == repo_->root()) return absl::UnavailableError("failed root");
        return (*factory)(root, settings);
      });
  FERRET_ASSERT_OK_AND_ASSIGN(
      workspace_, Workspace::Open(*Env::Default(), temp_.Join("index"),
                                  std::move(options)));
  FERRET_ASSERT_OK_AND_ASSIGN(
      id_, workspace_->Register(repo_->root(), {.watch = true}));
  FERRET_ASSERT_OK_AND_ASSIGN(
      WorktreeId second_id,
      workspace_->Register(second.root(), {.watch = true}));
  ABSL_ASSERT_OK(second.WriteFile(".git/index.lock", "temporary lock\n"));
  ABSL_ASSERT_OK(second.WriteFile("file.cc", "independenttoken\n"));
  Event(PathChanged{".git/index"});
  ABSL_ASSERT_OK(second.DeleteFile(".git/index.lock"));
  const absl::Time start = absl::Now();
  EXPECT_TRUE(Has(
      *workspace_, second_id, "independenttoken",
      {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(10)}));
  EXPECT_LT(absl::Now() - start, absl::Seconds(15));
}

TEST_F(WatchFixture, UnwatchedSyncWaitsForAssignedGeneration) {
  absl::Notification paused;
  absl::Notification resume;
  std::atomic<bool> gate{true};
  auto options = Options();
  options.read_hook = std::make_shared<WorkspaceReadHook>(
      [&](absl::string_view path, bool bulk) {
        if (!bulk && path == "base.cc" && gate.exchange(false)) {
          paused.Notify();
          resume.WaitForNotification();
        }
      });
  FERRET_ASSERT_OK_AND_ASSIGN(
      workspace_, Workspace::Open(*Env::Default(), temp_.Join("index"),
                                  std::move(options)));
  FERRET_ASSERT_OK_AND_ASSIGN(id_, workspace_->Register(repo_->root()));
  ABSL_ASSERT_OK(repo_->WriteFile("base.cc", "unwatchedtoken\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(uint64_t generation,
                              workspace_->Notify(id_, {"base.cc"}));
  ASSERT_TRUE(paused.WaitForNotificationWithTimeout(absl::Seconds(5)));
  auto stale = workspace_->Search(id_, Request("unwatchedtoken"),
                                  {.mode = SearchConsistencyMode::kSync,
                                   .timeout = absl::Milliseconds(10)});
  ASSERT_TRUE(stale.ok()) << stale.status();
  EXPECT_TRUE(stale->stale);
  EXPECT_LT(stale->generation, generation);
  resume.Notify();
  EXPECT_TRUE(
      Has(*workspace_, id_, "unwatchedtoken",
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)}));
}

TEST_F(WatchFixture, ConcurrentSyncsHaveSeparateTimeouts) {
  BlockingSyncWatcher* blocker = nullptr;
  auto options = Options();
  options.watcher_factory = std::make_shared<WorkspaceOptions::WatcherFactory>(
      [&](const std::string& root,
          const WatcherOptions&) -> absl::StatusOr<std::unique_ptr<Watcher>> {
        auto watcher = std::make_unique<BlockingSyncWatcher>(root);
        blocker = watcher.get();
        return std::unique_ptr<Watcher>(std::move(watcher));
      });
  FERRET_ASSERT_OK_AND_ASSIGN(
      workspace_, Workspace::Open(*Env::Default(), temp_.Join("index"),
                                  std::move(options)));
  FERRET_ASSERT_OK_AND_ASSIGN(
      id_, workspace_->Register(repo_->root(), {.watch = true}));
  std::thread first([&] {
    auto result = workspace_->Search(
        id_, Request("basetoken"),
        {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(1)});
    EXPECT_TRUE(result.ok()) << result.status();
  });
  const bool entered =
      blocker->entered.WaitForNotificationWithTimeout(absl::Seconds(5));
  EXPECT_TRUE(entered);
  if (entered) {
    const absl::Time start = absl::Now();
    auto result = workspace_->Search(id_, Request("basetoken"),
                                     {.mode = SearchConsistencyMode::kSync,
                                      .timeout = absl::Milliseconds(10)});
    EXPECT_TRUE(result.ok()) << result.status();
    if (result.ok()) EXPECT_TRUE(result->stale);
    EXPECT_LT(absl::Now() - start, absl::Milliseconds(250));
  }
  blocker->Release();
  first.join();
}

TEST_F(WatchFixture, CookieTimeoutIsStaleButNextSyncSucceeds) {
  BlockingSyncWatcher* blocker = nullptr;
  auto options = Options();
  options.watcher_factory = std::make_shared<WorkspaceOptions::WatcherFactory>(
      [&](const std::string& root,
          const WatcherOptions&) -> absl::StatusOr<std::unique_ptr<Watcher>> {
        auto watcher = std::make_unique<BlockingSyncWatcher>(root);
        blocker = watcher.get();
        return std::unique_ptr<Watcher>(std::move(watcher));
      });
  FERRET_ASSERT_OK_AND_ASSIGN(
      workspace_, Workspace::Open(*Env::Default(), temp_.Join("index"),
                                  std::move(options)));
  FERRET_ASSERT_OK_AND_ASSIGN(
      id_, workspace_->Register(repo_->root(), {.watch = true}));
  auto result = workspace_->Search(id_, Request("basetoken"),
                                   {.mode = SearchConsistencyMode::kSync,
                                    .timeout = absl::Milliseconds(10)});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->stale);
  EXPECT_EQ(workspace_->Stats().sync_timeouts, uint64_t{1});
  EXPECT_EQ(workspace_->Stats().watcher_failures, uint64_t{0});
  blocker->Release();
  EXPECT_TRUE(
      Has(*workspace_, id_, "basetoken",
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)}));
}

TEST_F(WatchFixture, RegistrationCapturesEventsDuringScan) {
  absl::Notification paused;
  absl::Notification resume;
  std::atomic<bool> gate{true};
  auto options = Options();
  options.read_hook = std::make_shared<WorkspaceReadHook>(
      [&](absl::string_view path, bool bulk) {
        if (bulk && path == "base.cc" && gate.exchange(false)) {
          paused.Notify();
          resume.WaitForNotification();
        }
      });
  FERRET_ASSERT_OK_AND_ASSIGN(
      workspace_, Workspace::Open(*Env::Default(), temp_.Join("index"),
                                  std::move(options)));
  absl::StatusOr<WorktreeId> registered;
  std::thread thread([&] {
    registered = workspace_->Register(repo_->root(), {.watch = true});
  });
  ASSERT_TRUE(paused.WaitForNotificationWithTimeout(absl::Seconds(5)));
  ABSL_ASSERT_OK(repo_->WriteFile("during.cc", "duringtoken\n"));
  Event(PathChanged{"during.cc"});
  resume.Notify();
  thread.join();
  ASSERT_TRUE(registered.ok()) << registered.status();
  EXPECT_TRUE(
      Has(*workspace_, *registered, "duringtoken",
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)}));
}

TEST_F(WatchFixture, ConcurrentWritesAndSyncSearches) {
  Open(Options());
  std::atomic<bool> done{false};
  std::thread writer([&] {
    for (int i = 0; i < 100; ++i) {
      auto file = Env::Default()->NewWritableFile(repo_->Path("stress.cc"));
      if (!file.ok()) break;
      if (!(*file)->Append(absl::StrCat("stresstoken", i, "\n")).ok()) break;
      if (!(*file)->Close().ok()) break;
      fake_.load()->Push(PathChanged{"stress.cc"});
    }
    done.store(true);
  });
  do {
    auto result = workspace_->Search(
        id_, Request("stresstoken"),
        {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)});
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_FALSE(result->stale);
  } while (!done.load());
  writer.join();
  EXPECT_TRUE(
      Has(*workspace_, id_, "stresstoken99",
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(5)}));
}

}  // namespace
}  // namespace ferret

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/base/testing/fault_injection_env.h"
#include "ferret/git/testing/git_repo_builder.h"
#include "ferret/search/search_types.h"
#include "ferret/testing/brute_force_search.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/testing/test_seed.h"
#include "ferret/workspace/workspace.h"

namespace ferret {
namespace {

using ::ferret_testing::FaultInjectionEnv;
using ::ferret_testing::GitRepoBuilder;
using ::ferret_testing::TempDir;

enum class CrashPhase { kRegister, kNotify, kUnregister, kRestore };

TEST(WorkspaceCrashTest, ReRegisterReconcilesDiskAfterEachCrashPhase) {
  const uint64_t seed = ferret_testing::TestSeed(441);
  SCOPED_TRACE(seed);
  const auto exercise = [&](CrashPhase phase, std::optional<int64_t> boundary,
                            int64_t* mutations) {
    SCOPED_TRACE(absl::StrCat("phase ", static_cast<int>(phase),
                              " Env mutation ", boundary.value_or(-1)));
    TempDir temp;
    ferret_testing::IsolateGitEnvironment(temp.Join("home"));
    auto repo = GitRepoBuilder::Init(temp, "repo");
    ASSERT_TRUE(repo.ok()) << repo.status();
    ASSERT_TRUE(repo->WriteFile("live.cc", "needlebefore\n").ok());
    ASSERT_TRUE(repo->Add().ok());
    ASSERT_TRUE(repo->Commit("base").ok());
    FaultInjectionEnv env(Env::Default());
    WorkspaceOptions options;
    absl::Notification notify_entered;
    absl::Notification notify_release;
    std::atomic<bool> notify_armed{false};
    int64_t notify_begin = 0;
    if (phase == CrashPhase::kNotify) {
      options.read_hook = std::make_shared<WorkspaceReadHook>(
          [&](absl::string_view, bool bulk) {
            if (!bulk && notify_armed.exchange(false)) {
              notify_begin = env.mutation_count();
              notify_entered.Notify();
              notify_release.WaitForNotification();
            }
          });
    }
    options.index.compaction.enabled = false;
    options.index.memtable.flush_docs = 1;
    const std::string path = temp.Join("index");
    if (phase == CrashPhase::kRestore) {
      auto initial = Workspace::Open(env, path, options);
      ASSERT_TRUE(initial.ok()) << initial.status();
      ASSERT_TRUE((*initial)->Register(repo->root()).ok());
    }
    const int64_t before = env.mutation_count();
    if (phase == CrashPhase::kRestore) {
      absl::Notification entered;
      absl::Notification release;
      options.restore_scan_hook =
          std::make_shared<absl::AnyInvocable<absl::Status(absl::string_view)>>(
              [&](absl::string_view) {
                entered.Notify();
                release.WaitForNotification();
                return absl::OkStatus();
              });
      auto restoring = Workspace::Open(env, path, options);
      ASSERT_TRUE(restoring.ok()) << restoring.status();
      ASSERT_TRUE(entered.WaitForNotificationWithTimeout(absl::Seconds(30)));
      if (boundary) {
        env.FailAfter(*boundary,
                      absl::UnavailableError("simulated power loss"));
      }
      release.Notify();
    } else {
      auto opened = Workspace::Open(env, path, options);
      ASSERT_TRUE(opened.ok()) << opened.status();
      std::unique_ptr<Workspace> workspace = std::move(*opened);
      if (phase == CrashPhase::kRegister) {
        if (boundary) {
          env.FailAfter(*boundary,
                        absl::UnavailableError("simulated power loss"));
        }
        workspace->Register(repo->root()).IgnoreError();
      } else {
        auto id = workspace->Register(repo->root());
        ASSERT_TRUE(id.ok()) << id.status();
        if (boundary && phase != CrashPhase::kNotify) {
          env.FailAfter(*boundary,
                        absl::UnavailableError("simulated power loss"));
        }
        if (phase == CrashPhase::kNotify) {
          ASSERT_TRUE(repo->WriteFile("live.cc", "needleafter\n").ok());
          for (int i = 0; i < 3; ++i) {
            ASSERT_TRUE(repo->WriteFile(absl::StrCat("edit", i, ".cc"),
                                        absl::StrCat("needleedit", i, "\n"))
                            .ok());
          }
          notify_armed.store(true);
          auto generation = workspace->Notify(
              *id, {"live.cc", "edit0.cc", "edit1.cc", "edit2.cc"});
          const bool entered =
              notify_entered.WaitForNotificationWithTimeout(absl::Seconds(30));
          if (entered && boundary) {
            // The reader is paused inside ingestion. The generation cannot
            // have been applied at the instant this crash fault is armed.
            if (generation.ok()) {
              auto pending = workspace->AwaitGeneration(*id, *generation,
                                                        absl::ZeroDuration());
              if (pending.ok()) EXPECT_LT(*pending, *generation);
            }
            env.FailAfter(*boundary,
                          absl::UnavailableError("simulated power loss"));
          }
          notify_release.Notify();
          ASSERT_TRUE(entered);
          if (generation.ok()) {
            auto applied =
                workspace->AwaitGeneration(*id, *generation, absl::Seconds(30));
            if (!boundary) EXPECT_TRUE(applied.ok()) << applied.status();
          }
        } else {
          workspace->Unregister(*id).IgnoreError();
        }
      }
    }
    if (mutations != nullptr) {
      *mutations = env.mutation_count() -
                   (phase == CrashPhase::kNotify ? notify_begin : before);
    }
    options.restore_scan_hook.reset();
    options.read_hook.reset();
    if (!boundary) return;
    ASSERT_TRUE(env.DropUnsyncedData().ok());
    // Restore may be racing the client registration. Register waits for a
    // fresh attempt, and kSync waits for its published scan.
    auto reopened = Workspace::Open(env, path, options);
    ASSERT_TRUE(reopened.ok()) << reopened.status();
    auto id = (*reopened)->Register(repo->root());
    ASSERT_TRUE(id.ok()) << id.status();
    std::vector<ferret_testing::CorpusFile> corpus;
    for (const std::string& name :
         {"live.cc", "edit0.cc", "edit1.cc", "edit2.cc"}) {
      auto bytes = Env::Default()->ReadFile(repo->Path(name));
      if (bytes.ok()) corpus.push_back({name, *bytes});
    }
    for (const SearchRequest& request :
         {SearchRequest{.pattern = "needle"},
          SearchRequest{.pattern = "needle(before|after|edit[0-9])",
                        .regex = true}}) {
      auto expected = ferret_testing::BruteForceSearch(corpus, request);
      auto actual = (*reopened)->Search(
          *id, request,
          {.mode = SearchConsistencyMode::kSync, .timeout = absl::Seconds(30)});
      ASSERT_TRUE(expected.ok()) << expected.status();
      ASSERT_TRUE(actual.ok()) << actual.status();
      EXPECT_FALSE(actual->stale);
      ASSERT_EQ(actual->response.files.size(), expected->size());
      std::vector<std::string> found_paths;
      std::vector<std::string> expected_paths;
      for (const FileResult& file : actual->response.files) {
        found_paths.push_back(file.path);
      }
      for (const FileResult& file : *expected) {
        expected_paths.push_back(file.path);
      }
      std::sort(found_paths.begin(), found_paths.end());
      std::sort(expected_paths.begin(), expected_paths.end());
      EXPECT_EQ(found_paths, expected_paths);
    }
  };
  for (const CrashPhase phase :
       {CrashPhase::kRegister, CrashPhase::kNotify, CrashPhase::kUnregister,
        CrashPhase::kRestore}) {
    int64_t mutations = 0;
    exercise(phase, std::nullopt, &mutations);
    ASSERT_GT(mutations, 0) << static_cast<int>(phase);
    std::vector<int64_t> boundaries{0, mutations};
    if (mutations <= 40) {
      for (int64_t i = 1; i < mutations; ++i) boundaries.push_back(i);
    } else {
      std::mt19937_64 rng(seed + static_cast<int>(phase));
      std::set<int64_t> chosen{0, mutations};
      while (chosen.size() < 42) {
        chosen.insert(1 + static_cast<int64_t>(rng() % (mutations - 1)));
      }
      boundaries.assign(chosen.begin(), chosen.end());
    }
    // Every Env boundary can split segment writes, syncs, MANIFEST installs,
    // directory syncs, or obsolete-file deletion. The calibration includes
    // the complete ingest generation rather than a fixed 50-ms slice.
    for (int64_t boundary : boundaries) exercise(phase, boundary, nullptr);
  }
}

}  // namespace
}  // namespace ferret

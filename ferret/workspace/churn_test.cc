// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/workspace/churn.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
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

using ::testing::Each;
using ::testing::Ge;
using ::testing::Lt;

TEST(ClientScheduleTest, KeepsTheRateAndStaysInsideTheLength) {
  for (const double rate : {0.5, 2.0, 7.0}) {
    for (int client = 0; client < 10; ++client) {
      const auto schedule = ClientSchedule({.seed = 5,
                                            .client = client,
                                            .rate_hz = rate,
                                            .length = absl::Seconds(100)},
                                           1000);
      EXPECT_NEAR(static_cast<double>(schedule.size()), rate * 100.0, 1.5)
          << rate << " " << client;
      for (size_t i = 0; i < schedule.size(); ++i) {
        EXPECT_GE(schedule[i].at, absl::ZeroDuration());
        EXPECT_LT(schedule[i].at, absl::Seconds(100));
        EXPECT_LT(schedule[i].query, size_t{1000});
        if (i > 0) EXPECT_LE(schedule[i - 1].at, schedule[i].at);
      }
    }
  }
}

TEST(ClientScheduleTest, IsOpenLoopWithBoundedJitter) {
  const double period = 0.5;
  const auto schedule = ClientSchedule({.seed = 9,
                                        .client = 3,
                                        .rate_hz = 2.0,
                                        .jitter = 0.5,
                                        .length = absl::Seconds(200)},
                                       50);
  ASSERT_GT(schedule.size(), size_t{100});
  for (size_t i = 1; i < schedule.size(); ++i) {
    const double gap =
        absl::ToDoubleSeconds(schedule[i].at - schedule[i - 1].at);
    // Neighbouring sends each move by at most a quarter period.
    EXPECT_GE(gap, 0.5 * period - 1e-9);
    EXPECT_LE(gap, 1.5 * period + 1e-9);
  }
  // Some gaps are not the period: the jitter is real.
  int jittered = 0;
  for (size_t i = 1; i < schedule.size(); ++i) {
    const double gap =
        absl::ToDoubleSeconds(schedule[i].at - schedule[i - 1].at);
    if (gap < period * 0.9 || gap > period * 1.1) ++jittered;
  }
  EXPECT_GT(jittered, 20);
}

TEST(ClientScheduleTest, NoJitterSendsAtTheExactPeriod) {
  const auto schedule = ClientSchedule({.seed = 1,
                                        .client = 0,
                                        .rate_hz = 4.0,
                                        .jitter = 0.0,
                                        .length = absl::Seconds(20)},
                                       10);
  ASSERT_EQ(schedule.size(), size_t{80});
  for (size_t i = 1; i < schedule.size(); ++i) {
    EXPECT_NEAR(absl::ToDoubleSeconds(schedule[i].at - schedule[i - 1].at),
                0.25, 1e-6);
  }
}

TEST(ClientScheduleTest, IsSeededPerClientAndExtendsWithLength) {
  const ClientScheduleOptions base{
      .seed = 3, .client = 2, .rate_hz = 2.0, .length = absl::Seconds(30)};
  const auto a = ClientSchedule(base, 100);
  const auto b = ClientSchedule(base, 100);
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].at, b[i].at);
    EXPECT_EQ(a[i].query, b[i].query);
  }
  ClientScheduleOptions longer = base;
  longer.length = absl::Seconds(90);
  const auto prefix = ClientSchedule(longer, 100);
  ASSERT_GT(prefix.size(), a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].at, prefix[i].at);
    EXPECT_EQ(a[i].query, prefix[i].query);
  }
  ClientScheduleOptions other_client = base;
  other_client.client = 3;
  ClientScheduleOptions other_seed = base;
  other_seed.seed = 4;
  EXPECT_NE(ClientSchedule(other_client, 100)[0].at, a[0].at);
  EXPECT_NE(ClientSchedule(other_seed, 100)[0].at, a[0].at);
}

TEST(ClientScheduleTest, ChoosesEveryQuery) {
  const auto schedule = ClientSchedule(
      {.seed = 1, .client = 0, .rate_hz = 2.0, .length = absl::Seconds(2000)},
      8);
  std::vector<int> seen(8, 0);
  for (const ScheduledQuery& query : schedule) ++seen[query.query];
  EXPECT_THAT(seen, Each(Ge(300)));
}

QuerySample Sample(double intended_s, double latency_ms,
                   QueryOutcome outcome = QueryOutcome::kOk,
                   double started_s = -1) {
  QuerySample sample;
  sample.intended = absl::Seconds(intended_s);
  sample.started = absl::Seconds(started_s < 0 ? intended_s : started_s);
  sample.latency = absl::Milliseconds(latency_ms);
  sample.outcome = outcome;
  return sample;
}

TEST(SplitWindowsTest, AssignsByIntendedTimeAndComputesPercentiles) {
  const ChurnEvents events{.checkout_start = absl::Seconds(10),
                           .ingest_done = absl::Seconds(14),
                           .merge_done = absl::Seconds(20),
                           .end = absl::Seconds(30)};
  std::vector<QuerySample> samples;
  // 100 queries before, latencies 1..100 ms.
  for (int i = 1; i <= 100; ++i) samples.push_back(Sample(i * 0.09, i));
  // The boundaries belong to the later window.
  samples.push_back(Sample(10.0, 40));
  samples.push_back(Sample(13.999, 50, QueryOutcome::kStale));
  samples.push_back(Sample(14.0, 60, QueryOutcome::kFailed));
  samples.push_back(Sample(19.5, 70));
  samples.push_back(Sample(20.0, 5));
  samples.push_back(Sample(29.9, 6));
  // A slow query: intended in the merge window, started 300 ms late.
  samples.push_back(Sample(15.0, 80, QueryOutcome::kOk, 15.3));

  const auto windows = SplitWindows(samples, events);
  EXPECT_EQ(windows[kBefore].count, 100);
  EXPECT_EQ(windows[kBefore].p50, absl::Milliseconds(51));
  EXPECT_EQ(windows[kBefore].p99, absl::Milliseconds(100));
  EXPECT_EQ(windows[kBefore].max, absl::Milliseconds(100));
  EXPECT_EQ(windows[kBefore].begin, absl::ZeroDuration());
  EXPECT_EQ(windows[kBefore].end, absl::Seconds(10));
  EXPECT_EQ(windows[kIngest].count, 2);
  EXPECT_EQ(windows[kIngest].stale, 1);
  EXPECT_EQ(windows[kIngest].failed, 0);
  EXPECT_EQ(windows[kIngest].max, absl::Milliseconds(50));
  EXPECT_EQ(windows[kMerge].count, 3);
  EXPECT_EQ(windows[kMerge].failed, 1);
  EXPECT_EQ(windows[kMerge].max, absl::Milliseconds(80));
  EXPECT_EQ(windows[kMerge].p99_response, absl::Milliseconds(380));
  EXPECT_EQ(windows[kAfter].count, 2);
  EXPECT_EQ(windows[kAfter].p99, absl::Milliseconds(6));
  EXPECT_EQ(windows[kAfter].end, absl::Seconds(30));
  int64_t total = 0;
  for (const WindowStats& window : windows) total += window.count;
  EXPECT_EQ(total, static_cast<int64_t>(samples.size()));
}

TEST(SplitWindowsTest, EmptyWindowsAreZero) {
  const ChurnEvents events{.checkout_start = absl::Seconds(1),
                           .ingest_done = absl::Seconds(2),
                           .merge_done = absl::Seconds(2),
                           .end = absl::Seconds(3)};
  const auto windows = SplitWindows({}, events);
  for (const WindowStats& window : windows) {
    EXPECT_EQ(window.count, 0);
    EXPECT_EQ(window.p99, absl::ZeroDuration());
  }
  // A zero-length merge window takes nothing between the two events.
  const auto one =
      SplitWindows(std::vector<QuerySample>{Sample(2.0, 3)}, events);
  EXPECT_EQ(one[kMerge].count, 0);
  EXPECT_EQ(one[kAfter].count, 1);
}

class RunChurnTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ferret_testing::IsolateGitEnvironment(temp_.Join("home"));
  }
  ferret_testing::TempDir temp_;
};

TEST_F(RunChurnTest, MeasuresAWatchedCheckoutInFourWindows) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      ferret_testing::GitRepoBuilder repo,
      ferret_testing::GitRepoBuilder::Init(temp_, "repo"));
  for (int i = 0; i < 30; ++i) {
    ABSL_ASSERT_OK(repo.WriteFile(absl::StrCat("src/f", i, ".cc"),
                                  absl::StrCat("int churn_value_", i, " = ", i,
                                               ";\nconst int shared = 1;\n")));
  }
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  FERRET_ASSERT_OK_AND_ASSIGN(std::string base_out,
                              repo.Git({"rev-parse", "HEAD"}));
  const std::string base(absl::StripAsciiWhitespace(base_out));
  ABSL_ASSERT_OK(repo.Git({"checkout", "-q", "-b", "week"}).status());
  for (int i = 0; i < 10; ++i) {
    ABSL_ASSERT_OK(repo.WriteFile(absl::StrCat("src/f", i, ".cc"),
                                  "int rewritten = 1;\n"));
  }
  ABSL_ASSERT_OK(repo.WriteFile("synthetic/marker.txt", "churnmarkerzq\n"));
  ABSL_ASSERT_OK(repo.DeleteFile("src/f29.cc"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("week"));
  FERRET_ASSERT_OK_AND_ASSIGN(std::string week_out,
                              repo.Git({"rev-parse", "HEAD"}));
  const std::string week(absl::StripAsciiWhitespace(week_out));
  ABSL_ASSERT_OK(repo.Git({"checkout", "-q", "--detach", base}).status());
  FERRET_ASSERT_OK_AND_ASSIGN(ferret_testing::GitRepoBuilder other,
                              repo.AddWorktree("other"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp_.Join("index")));
  FERRET_ASSERT_OK_AND_ASSIGN(
      WorktreeId tree, workspace->Register(other.root(), {.watch = true}));

  ChurnOptions options;
  options.clients = 3;
  options.rate_hz = 20.0;
  options.seed = 11;
  options.before = absl::Milliseconds(400);
  options.after = absl::Milliseconds(400);
  options.quiet = absl::Milliseconds(200);
  options.marker = "churnmarkerzq";
  for (const char* pattern : {"churn_value_3", "shared", "rewritten"}) {
    SearchRequest request;
    request.pattern = pattern;
    options.requests.push_back(request);
  }
  options.checkout = [&] {
    return other.Git({"checkout", "-q", "--detach", week}).status();
  };
  FERRET_ASSERT_OK_AND_ASSIGN(ChurnResult result,
                              RunChurn(*workspace, tree, options));
  EXPECT_FALSE(result.ingest_timed_out);
  EXPECT_GT(result.marker_visible, absl::ZeroDuration());
  EXPECT_LT(result.events.checkout_start, result.events.ingest_done);
  EXPECT_LE(result.events.ingest_done, result.events.merge_done);
  EXPECT_LT(result.events.merge_done, result.events.end);
  EXPECT_GE(result.events.checkout_start, options.before);
  EXPECT_GE(result.events.end - result.events.merge_done, options.after);
  // 3 clients at 20 per second, for the whole run, within one query per
  // client at each end.
  const double seconds = absl::ToDoubleSeconds(result.events.end);
  EXPECT_NEAR(static_cast<double>(result.samples.size()), 3 * 20.0 * seconds,
              3 * 2.0);
  EXPECT_NEAR(static_cast<double>(result.windows[kBefore].count),
              3 * 20.0 * absl::ToDoubleSeconds(result.events.checkout_start),
              3 * 2.0);
  int64_t total = 0;
  for (const WindowStats& window : result.windows) {
    total += window.count;
    EXPECT_EQ(window.failed, 0);
    EXPECT_EQ(window.stale, 0);
  }
  EXPECT_EQ(total, static_cast<int64_t>(result.samples.size()));
  EXPECT_GT(result.windows[kBefore].count, 0);
  EXPECT_GT(result.windows[kAfter].count, 0);
  // The view changed: one file deleted, a marker file added.
  EXPECT_EQ(result.view_paths_after, result.view_paths_before);
  // Every sample's intended time follows the seeded schedule of its client.
  const auto expected = ClientSchedule({.seed = 11,
                                        .client = 0,
                                        .rate_hz = 20.0,
                                        .jitter = 0.5,
                                        .length = absl::Hours(3)},
                                       3);
  std::vector<absl::Duration> client0;
  for (const QuerySample& sample : result.samples) {
    if (sample.client == 0) client0.push_back(sample.intended);
  }
  ASSERT_FALSE(client0.empty());
  ASSERT_LE(client0.size(), expected.size());
  for (size_t i = 0; i < client0.size(); ++i) {
    EXPECT_EQ(client0[i], expected[i].at);
  }
  SearchRequest after;
  after.pattern = "rewritten";
  FERRET_ASSERT_OK_AND_ASSIGN(WorkspaceSearchResult found,
                              workspace->Search(tree, after));
  EXPECT_EQ(found.response.files.size(), size_t{10});
}

TEST_F(RunChurnTest, TheIngestIsNotDoneUntilTheMarkerIsSearchable) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      ferret_testing::GitRepoBuilder repo,
      ferret_testing::GitRepoBuilder::Init(temp_, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("a.cc", "token\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp_.Join("index")));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId tree,
                              workspace->Register(repo.root()));
  ChurnOptions options;
  options.clients = 1;
  options.before = absl::Milliseconds(50);
  options.after = absl::Milliseconds(50);
  options.quiet = absl::Milliseconds(50);
  options.ingest_timeout = absl::Milliseconds(600);
  options.marker = "markerthatneverappearszq";
  SearchRequest request;
  request.pattern = "token";
  options.requests.push_back(request);
  options.checkout = [] { return absl::OkStatus(); };
  FERRET_ASSERT_OK_AND_ASSIGN(ChurnResult result,
                              RunChurn(*workspace, tree, options));
  EXPECT_TRUE(result.ingest_timed_out);
  EXPECT_EQ(result.marker_visible, absl::ZeroDuration());
  EXPECT_GE(result.events.ingest_done - result.events.checkout_start,
            options.ingest_timeout);
  // Without a marker the settle time alone ends the ingest.
  options.marker.clear();
  FERRET_ASSERT_OK_AND_ASSIGN(ChurnResult unmarked,
                              RunChurn(*workspace, tree, options));
  EXPECT_FALSE(unmarked.ingest_timed_out);
  EXPECT_LT(unmarked.events.ingest_done - unmarked.events.checkout_start,
            options.ingest_timeout);
}

TEST_F(RunChurnTest, ACheckoutFailureStopsTheClientsAndIsReturned) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      ferret_testing::GitRepoBuilder repo,
      ferret_testing::GitRepoBuilder::Init(temp_, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("a.cc", "token\n"));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("base"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Workspace> workspace,
      Workspace::Open(*Env::Default(), temp_.Join("index")));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeId tree,
                              workspace->Register(repo.root()));
  ChurnOptions options;
  options.clients = 2;
  options.before = absl::Milliseconds(100);
  SearchRequest request;
  request.pattern = "token";
  options.requests.push_back(request);
  options.checkout = [] { return absl::InternalError("checkout failed"); };
  auto result = RunChurn(*workspace, tree, options);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInternal);
}

}  // namespace
}  // namespace ferret

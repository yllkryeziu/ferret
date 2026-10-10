// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "bench/ferret/search_bench.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "bench/ferret/query_set.h"
#include "bench/ferret/result_set.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/ablation.h"
#include "ferret/base/env.h"
#include "ferret/index/bulk_build.h"
#include "ferret/index/segment.h"
#include "ferret/search/search_types.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"

namespace ferret_bench {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Optional;

const std::vector<QuerySpec> kQueries = {
    {.id = "hit",
     .query_class = "identifier",
     .pattern = {.pattern = "alpha", .literal = true, .case_sensitive = true}},
    {.id = "miss",
     .query_class = "zero",
     .pattern = {
         .pattern = "absent", .literal = true, .case_sensitive = true}}};

TEST(SearchBenchTest, TwoClientsAndRunsHaveStableDigests) {
  ferret_testing::TempDir dir;
  ASSERT_THAT(dir.WriteFile("a.c", "alpha\nbeta alpha\n"), IsOk());
  ASSERT_THAT(dir.WriteFile("b.c", "alpha\n"), IsOk());
  const std::vector<std::string> paths = {"a.c", "b.c"};
  FERRET_ASSERT_OK_AND_ASSIGN(
      const ferret::BulkBuildStats build,
      ferret::BuildSegment(*ferret::Env::Default(), dir.path(), paths,
                           dir.Join("test.fseg")));
  EXPECT_EQ(build.files_indexed, 2);
  FERRET_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<ferret::Segment> segment,
      ferret::Segment::Open(*ferret::Env::Default(), dir.Join("test.fseg")));

  SearchBenchConfig config;
  config.context_lines = 0;
  config.clients = 2;
  config.warmup_passes = 1;
  config.timed_runs = 3;
  config.full_results = true;
  config.k = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(const std::vector<SearchRecord> topk,
                              RunSearchBench(*segment, kQueries, config));
  ASSERT_EQ(topk.size(), size_t{2});
  const SearchRecord& hit = topk[0];
  EXPECT_EQ(hit.id, "hit");
  EXPECT_EQ(hit.query_class, "identifier");
  EXPECT_EQ(hit.engine, "ferret");
  EXPECT_EQ(hit.mode, SearchMode::kTopK);
  EXPECT_EQ(hit.k, 1);
  EXPECT_EQ(hit.context, 0);
  EXPECT_EQ(hit.threads, 1);
  EXPECT_EQ(hit.clients, 2);
  EXPECT_EQ(hit.latencies_ns.size(), size_t{6});
  EXPECT_EQ(hit.file_count, 1);
  EXPECT_EQ(hit.result_count, 2);
  EXPECT_FALSE(hit.truncated);
  const std::vector<LineRef> expected = {LineRef{"a.c", 1}, LineRef{"a.c", 2}};
  EXPECT_THAT(hit.results, Optional(ElementsAreArray(expected)));
  EXPECT_EQ(hit.result_digest, LineSetDigest(expected));
  EXPECT_TRUE(hit.digest_stable);
  EXPECT_EQ(hit.engine_stats.num_matching_lines, 3);
  EXPECT_EQ(topk[1].latencies_ns.size(), size_t{6});
  EXPECT_EQ(topk[1].result_count, 0);
  EXPECT_TRUE(topk[1].digest_stable);
  EXPECT_EQ(topk[1].result_digest, LineSetDigest({}));
  EXPECT_THAT(topk[1].results, Optional(IsEmpty()));

  config.mode = SearchMode::kExhaustive;
  config.max_matches = 1;
  config.full_results = false;
  FERRET_ASSERT_OK_AND_ASSIGN(const std::vector<SearchRecord> exhaustive,
                              RunSearchBench(*segment, kQueries, config));
  const SearchRecord& capped = exhaustive[0];
  EXPECT_FALSE(capped.k.has_value());
  EXPECT_FALSE(capped.results.has_value());
  EXPECT_EQ(capped.result_count, 1);
  EXPECT_EQ(capped.latencies_ns.size(), size_t{6});
  EXPECT_TRUE(capped.truncated);
  EXPECT_TRUE(capped.digest_stable);
  EXPECT_EQ(capped.engine_stats.num_matching_lines, 3);
  EXPECT_EQ(capped.result_digest.size(), size_t{64});
}

TEST(SearchBenchTest, TopKStopsEvaluatingAndExhaustiveDoesNotStop) {
  ferret_testing::TempDir dir;
  ASSERT_THAT(dir.WriteFile("root.cc", "// alpha root\n"), IsOk());
  ASSERT_THAT(dir.WriteFile("third_party/low.cc", "// alpha low\n"), IsOk());
  ASSERT_THAT(dir.WriteFile("third_party/other.cc", "// alpha other\n"),
              IsOk());
  const std::vector<std::string> paths = {"root.cc", "third_party/low.cc",
                                          "third_party/other.cc"};
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto build, ferret::BuildSegment(*ferret::Env::Default(), dir.path(),
                                       paths, dir.Join("rank.fseg")));
  EXPECT_EQ(build.files_indexed, 3);
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto segment,
      ferret::Segment::Open(*ferret::Env::Default(), dir.Join("rank.fseg")));
  SearchBenchConfig config;
  config.k = 1;
  config.warmup_passes = 0;
  config.timed_runs = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(auto topk,
                              RunSearchBench(*segment, {kQueries[0]}, config));
  ASSERT_EQ(topk.size(), 1u);
  EXPECT_TRUE(topk[0].terminated_early);
  EXPECT_EQ(SearchRecordToJson(topk[0])["terminated_early"], true);
  EXPECT_FALSE(
      SearchRecordToJson(topk[0])["engine_stats"].contains("skipped_chunks"));

  config.mode = SearchMode::kExhaustive;
  FERRET_ASSERT_OK_AND_ASSIGN(auto exhaustive,
                              RunSearchBench(*segment, {kQueries[0]}, config));
  ASSERT_EQ(exhaustive.size(), 1u);
  EXPECT_FALSE(exhaustive[0].terminated_early);
  EXPECT_LT(topk[0].engine_stats.candidate_chunks,
            exhaustive[0].engine_stats.candidate_chunks);
  EXPECT_EQ(exhaustive[0].engine_stats.num_matching_files, 3);
}

TEST(SearchBenchTest, JsonUsesRecordSchemaAndOmitsOptionalFields) {
  SearchRecord record;
  record.id = "q";
  record.query_class = "identifier";
  record.mode = SearchMode::kExhaustive;
  record.context = 2;
  record.clients = 2;
  record.latencies_ns = {10, 20};
  record.result_digest = "digest";
  record.cold_view_names_ns = 11;
  record.view_rebuild_ns = 12;
  record.engine_stats = {.candidate_chunks = 1,
                         .verified_chunks = 2,
                         .verified_bytes = 3,
                         .priority_documents = 9,
                         .priority_basenames_tested = 14,
                         .priority_basenames_scanned = 15,
                         .priority_restricted_sources = 16,
                         .priority_scan_ns = 13,
                         .compile_ns = 4,
                         .evaluate_ns = 5,
                         .verify_ns = 6,
                         .num_matching_files = 7,
                         .num_matching_lines = 8};
  const nlohmann::json json = SearchRecordToJson(record);
  EXPECT_EQ(json.size(), size_t{18});
  EXPECT_EQ(json["id"], "q");
  EXPECT_EQ(json["class"], "identifier");
  EXPECT_EQ(json["engine"], "ferret");
  EXPECT_EQ(json["mode"], "exhaustive");
  EXPECT_EQ(json["context"], 2);
  EXPECT_EQ(json["threads"], 1);
  EXPECT_EQ(json["search_threads"], 1);
  EXPECT_EQ(json["clients"], 2);
  EXPECT_EQ(json["latencies_ns"], nlohmann::json::array({10, 20}));
  EXPECT_EQ(json["truncated"], false);
  EXPECT_EQ(json["terminated_early"], false);
  EXPECT_EQ(json["file_count"], 0);
  EXPECT_EQ(json["result_count"], 0);
  EXPECT_EQ(json["result_digest"], "digest");
  EXPECT_EQ(json["digest_stable"], true);
  EXPECT_EQ(json["cold_view_names_ns"], 11);
  EXPECT_EQ(json["view_rebuild_ns"], 12);
  EXPECT_EQ(json["engine_stats"].size(), size_t{13});
  EXPECT_FALSE(json["engine_stats"].contains("skipped_chunks"));
  EXPECT_EQ(json["engine_stats"]["candidate_chunks"], 1);
  EXPECT_EQ(json["engine_stats"]["verified_chunks"], 2);
  EXPECT_EQ(json["engine_stats"]["verified_bytes"], 3);
  EXPECT_EQ(json["engine_stats"]["priority_documents"], 9);
  EXPECT_EQ(json["engine_stats"]["priority_basenames_tested"], 14);
  EXPECT_EQ(json["engine_stats"]["priority_basenames_scanned"], 15);
  EXPECT_EQ(json["engine_stats"]["priority_restricted_sources"], 16);
  EXPECT_EQ(json["engine_stats"]["priority_scan_ns"], 13);
  EXPECT_EQ(json["engine_stats"]["compile_ns"], 4);
  EXPECT_EQ(json["engine_stats"]["evaluate_ns"], 5);
  EXPECT_EQ(json["engine_stats"]["verify_ns"], 6);
  EXPECT_EQ(json["engine_stats"]["num_matching_files"], 7);
  EXPECT_EQ(json["engine_stats"]["num_matching_lines"], 8);
  EXPECT_FALSE(json.contains("k"));
  EXPECT_FALSE(json.contains("results"));
  EXPECT_FALSE(json["engine_stats"].contains("count_only_bytes"));
  EXPECT_FALSE(json["engine_stats"].contains("count_only_documents"));

  record.engine_stats.count_only_bytes = 21;
  record.engine_stats.count_only_documents = 5;
  const nlohmann::json counted = SearchRecordToJson(record);
  EXPECT_EQ(counted["engine_stats"].size(), size_t{15});
  EXPECT_EQ(counted["engine_stats"]["count_only_bytes"], 21);
  EXPECT_EQ(counted["engine_stats"]["count_only_documents"], 5);
  record.engine_stats.count_only_bytes = 0;
  record.engine_stats.count_only_documents = 0;

  record.mode = SearchMode::kTopK;
  record.k = 4;
  record.results = std::vector<LineRef>{{"a.c", 3}};
  const nlohmann::json topk = SearchRecordToJson(record);
  EXPECT_EQ(topk["mode"], "topk");
  EXPECT_EQ(topk["k"], 4);
  EXPECT_EQ(topk["results"], nlohmann::json::array({{"a.c", 3}}));
}

// A segment of `files` files, every one with "alpha" lines and some with
// rarer words, for searches that last long enough to time.
std::unique_ptr<ferret::Segment> BigSegment(ferret_testing::TempDir& dir,
                                            int files) {
  std::vector<std::string> paths;
  for (int i = 0; i < files; ++i) {
    std::string content;
    for (int line = 0; line < 40; ++line) {
      absl::StrAppend(&content, "int alpha_", i, "_", line, " = ", line, ";\n");
    }
    if (i % 7 == 0) absl::StrAppend(&content, "// gamma delta ", i, "\n");
    if (i % 50 == 0) absl::StrAppend(&content, "// zeta alpha ", i, "\n");
    const std::string path = absl::StrCat("d", i % 9, "/f", i, ".cc");
    EXPECT_THAT(dir.WriteFile(path, content), IsOk());
    paths.push_back(path);
  }
  EXPECT_TRUE(ferret::BuildSegment(*ferret::Env::Default(), dir.path(), paths,
                                   dir.Join("big.fseg"))
                  .ok());
  auto segment =
      ferret::Segment::Open(*ferret::Env::Default(), dir.Join("big.fseg"));
  EXPECT_TRUE(segment.ok());
  return std::move(*segment);
}

// The winner's name matches and no later name does, so a top-1 search stops
// after the first file (searcher_test: NameAwareBoundStopsWhereTheFull...).
std::unique_ptr<ferret::Segment> StopSegment(ferret_testing::TempDir& dir) {
  std::vector<std::string> paths = {"needle.cc"};
  EXPECT_THAT(dir.WriteFile("needle.cc", "int needle = 0;\n"), IsOk());
  for (int i = 0; i < 40; ++i) {
    paths.push_back(absl::StrCat("d/f", i, ".cc"));
    EXPECT_THAT(
        dir.WriteFile(paths.back(), absl::StrCat("// needle ", i, "\n")),
        IsOk());
  }
  EXPECT_TRUE(ferret::BuildSegment(*ferret::Env::Default(), dir.path(), paths,
                                   dir.Join("stop.fseg"))
                  .ok());
  auto segment =
      ferret::Segment::Open(*ferret::Env::Default(), dir.Join("stop.fseg"));
  EXPECT_TRUE(segment.ok());
  return std::move(*segment);
}

const std::vector<QuerySpec> kStopQueries = {
    {.id = "stop",
     .query_class = "identifier",
     .pattern = {
         .pattern = "needle", .literal = true, .case_sensitive = true}}};

const std::vector<QuerySpec> kBigQueries = {
    {.id = "ident",
     .query_class = "identifier",
     .pattern = {.pattern = "alpha_3",
                 .literal = true,
                 .case_sensitive = true}},
    {.id = "rare",
     .query_class = "literal",
     .pattern = {.pattern = "zeta alpha",
                 .literal = true,
                 .case_sensitive = false}},
    {.id = "pair",
     .query_class = "literal",
     .pattern = {.pattern = "gamma delta",
                 .literal = true,
                 .case_sensitive = true}},
    {.id = "re",
     .query_class = "regex",
     .pattern = {.pattern = "alpha_[0-9]+_3 =",
                 .literal = false,
                 .case_sensitive = true}}};

TEST(SearchBenchTest, TopKIsG1sConfigurationByDefault) {
  const SearchBenchConfig config;
  EXPECT_EQ(config.k, 20);
  EXPECT_EQ(config.context_lines, 2);
  EXPECT_TRUE(config.count_sample);
  EXPECT_TRUE(config.encode_response);
  ferret_testing::TempDir dir;
  const auto segment = BigSegment(dir, 600);
  SearchBenchConfig run;
  run.warmup_passes = 0;
  run.timed_runs = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(auto estimated,
                              RunSearchBench(*segment, kBigQueries, run));
  // The estimate is on: counting-only verification, which it excludes, is off.
  for (const SearchRecord& record : estimated) {
    EXPECT_EQ(record.engine_stats.count_only_documents, 0) << record.id;
    EXPECT_GT(record.response_bytes, 0) << record.id;
  }
  run.count_sample = false;
  run.encode_response = false;
  FERRET_ASSERT_OK_AND_ASSIGN(auto exact,
                              RunSearchBench(*segment, kBigQueries, run));
  for (size_t i = 0; i < exact.size(); ++i) {
    EXPECT_EQ(exact[i].response_bytes, 0);
    // The estimate never changes which lines a response holds.
    EXPECT_EQ(exact[i].result_digest, estimated[i].result_digest)
        << exact[i].id;
  }
}

TEST(ComputeStagesTest, PartitionsTheTotal) {
  ferret::SearchStats stats;
  stats.compile_time = absl::Microseconds(100);
  stats.dictionary_time = absl::Microseconds(30);
  stats.evaluate_time = absl::Microseconds(500);
  stats.priority_scan_time = absl::Microseconds(120);
  stats.decode_time = absl::Microseconds(50);
  stats.intersect_time = absl::Microseconds(80);
  stats.verify_time = absl::Microseconds(2000);
  stats.count_sample_time = absl::Microseconds(700);
  stats.rank_time = absl::Microseconds(40);
  stats.worker_scan_ns = 3000;
  const StageBreakdown s = ComputeStages(stats, 3600000, 90000);
  EXPECT_EQ(s.plan, 70000);
  EXPECT_EQ(s.dictionary, 30000);
  EXPECT_EQ(s.decode, 50000);
  EXPECT_EQ(s.intersect, 80000);
  // evaluate_time holds the priority scan and the nested decode and intersect.
  EXPECT_EQ(s.window, 250000);
  EXPECT_EQ(s.priority, 120000);
  EXPECT_EQ(s.verify, 2000000);
  EXPECT_EQ(s.sample, 700000);
  EXPECT_EQ(s.rank, 40000);
  EXPECT_EQ(s.encode, 90000);
  EXPECT_EQ(s.other, 170000);
  EXPECT_EQ(s.plan + s.dictionary + s.decode + s.intersect + s.window +
                s.priority + s.verify + s.sample + s.rank + s.encode + s.other,
            s.total);
  EXPECT_EQ(s.worker_scan, 3000);
}

// The stages are measured separately from the clock around the whole search,
// so their sum agreeing with it checks both. The tolerance: whatever no stage
// covers (setup, teardown, clock reads) is at most 10% of a query's wall time
// plus 50 microseconds in at least one of the runs (a preempted run may
// exceed it), and no stage is negative.
TEST(SearchBenchTest, StagesSumToTheMeasuredTotalWithinTolerance) {
  ferret_testing::TempDir dir;
  const auto segment = BigSegment(dir, 1500);
  for (const int threads : {1, 3}) {
    SCOPED_TRACE(threads);
    SearchBenchConfig config;
    config.search_threads = threads;
    config.warmup_passes = 1;
    config.timed_runs = 3;
    config.stage_timing = true;
    FERRET_ASSERT_OK_AND_ASSIGN(auto records,
                                RunSearchBench(*segment, kBigQueries, config));
    for (const SearchRecord& record : records) {
      SCOPED_TRACE(record.id);
      ASSERT_EQ(record.stage_runs.size(), size_t{3});
      for (const StageBreakdown& s : record.stage_runs) {
        for (const int64_t stage :
             {s.plan, s.dictionary, s.decode, s.intersect, s.window, s.priority,
              s.verify, s.sample, s.rank, s.encode}) {
          EXPECT_GE(stage, 0);
        }
        EXPECT_GE(s.other, -s.total / 100);
        EXPECT_GT(s.total, 0);
      }
      // A preemption inside one run lands in `other`, so the upper bound is
      // asserted on the best of the runs.
      bool within = false;
      for (const StageBreakdown& s : record.stage_runs) {
        within = within || s.other <= s.total / 10 + 50'000;
      }
      EXPECT_TRUE(within);
      const nlohmann::json json = SearchRecordToJson(record);
      ASSERT_TRUE(json.contains("stages_ns"));
      // The record carries the run with the median total.
      std::vector<int64_t> totals;
      totals.reserve(record.stage_runs.size());
      for (const StageBreakdown& s : record.stage_runs) {
        totals.push_back(s.total);
      }
      std::sort(totals.begin(), totals.end());
      EXPECT_EQ(json["stages_ns"]["total"], totals[1]);
    }
  }
}

TEST(SearchBenchTest, StageTimingIsOffByDefault) {
  ferret_testing::TempDir dir;
  const auto segment = BigSegment(dir, 100);
  SearchBenchConfig config;
  config.warmup_passes = 0;
  config.timed_runs = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(auto records,
                              RunSearchBench(*segment, kBigQueries, config));
  for (const SearchRecord& record : records) {
    EXPECT_TRUE(record.stage_runs.empty());
    EXPECT_FALSE(SearchRecordToJson(record).contains("stages_ns"));
  }
}

ferret::SearchResponse OneFile(absl::string_view path,
                               const std::vector<int64_t>& lines) {
  ferret::SearchResponse response;
  ferret::FileResult file;
  file.path = std::string(path);
  for (const int64_t number : lines) {
    file.lines.push_back({.number = number, .is_match = true});
  }
  response.files.push_back(std::move(file));
  return response;
}

TEST(CompareWithOracleTest, EqualResponsesAgreeAndPlantedDifferencesAreCaught) {
  const ferret::SearchResponse oracle = OneFile("a.cc", {3, 9});
  EXPECT_TRUE(CompareWithOracle(OneFile("a.cc", {3, 9}), oracle).equal);
  EXPECT_FALSE(CompareWithOracle(OneFile("a.cc", {3, 8}), oracle).equal);
  EXPECT_FALSE(CompareWithOracle(OneFile("a.cc", {3}), oracle).equal);
  EXPECT_FALSE(CompareWithOracle(OneFile("b.cc", {3, 9}), oracle).equal);
  ferret::SearchResponse extra = OneFile("a.cc", {3, 9});
  extra.files.push_back(extra.files[0]);
  extra.files[1].path = "z.cc";
  EXPECT_FALSE(CompareWithOracle(extra, oracle).equal);
  // A context line is not a match line.
  ferret::SearchResponse context = OneFile("a.cc", {3, 9});
  context.files[0].lines[1].is_match = false;
  EXPECT_FALSE(CompareWithOracle(context, oracle).equal);
  // Order of files matters: it is the ranking.
  ferret::SearchResponse two = OneFile("a.cc", {1});
  two.files.push_back(OneFile("b.cc", {1}).files[0]);
  ferret::SearchResponse swapped = two;
  std::swap(swapped.files[0], swapped.files[1]);
  EXPECT_FALSE(CompareWithOracle(swapped, two).equal);

  // Only a complete response that differs is a mismatch.
  ferret::SearchResponse wrong = OneFile("a.cc", {3, 8});
  EXPECT_TRUE(CompareWithOracle(wrong, oracle).mismatch());
  wrong.complete = false;
  const OracleResult cut = CompareWithOracle(wrong, oracle);
  EXPECT_FALSE(cut.mismatch());
  EXPECT_EQ(cut.common_files, 1);
  EXPECT_EQ(cut.oracle_files, 1);
}

TEST(SearchBenchTest, OracleConfirmsCompleteResponsesIncludingEarlyStops) {
  ferret_testing::TempDir dir;
  const auto big = BigSegment(dir, 800);
  const auto stop = StopSegment(dir);
  for (const int threads : {1, 3}) {
    SCOPED_TRACE(threads);
    SearchBenchConfig config;
    config.search_threads = threads;
    config.k = 3;
    config.warmup_passes = 0;
    config.timed_runs = 2;
    config.oracle = true;
    FERRET_ASSERT_OK_AND_ASSIGN(auto records,
                                RunSearchBench(*big, kBigQueries, config));
    config.k = 1;
    FERRET_ASSERT_OK_AND_ASSIGN(auto stops,
                                RunSearchBench(*stop, kStopQueries, config));
    records.insert(records.end(), stops.begin(), stops.end());
    int stopped_early = 0;
    for (const SearchRecord& record : records) {
      SCOPED_TRACE(record.id);
      ASSERT_TRUE(record.oracle.has_value());
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      const OracleResult& oracle = *record.oracle;
      EXPECT_TRUE(oracle.complete);
      EXPECT_TRUE(oracle.equal);
      EXPECT_FALSE(oracle.mismatch());
      EXPECT_GT(oracle.oracle_ns, 0);
      // The oracle sees every matching file; the response may have stopped.
      EXPECT_GE(oracle.oracle_matching_files,
                record.engine_stats.num_matching_files > 0 ? 1 : 0);
      stopped_early += record.terminated_early;
      EXPECT_EQ(SearchRecordToJson(record)["oracle"]["mismatch"], false);
    }
    EXPECT_GT(stopped_early, 0) << "no query exercised an early stop";
  }
}

TEST(SearchBenchTest, OracleIsOffUnlessAskedAndDoesNotChangeResults) {
  ferret_testing::TempDir dir;
  const auto segment = BigSegment(dir, 200);
  SearchBenchConfig config;
  config.warmup_passes = 0;
  config.timed_runs = 2;
  FERRET_ASSERT_OK_AND_ASSIGN(auto plain,
                              RunSearchBench(*segment, kBigQueries, config));
  config.oracle = true;
  FERRET_ASSERT_OK_AND_ASSIGN(auto checked,
                              RunSearchBench(*segment, kBigQueries, config));
  for (size_t i = 0; i < plain.size(); ++i) {
    EXPECT_FALSE(plain[i].oracle.has_value());
    EXPECT_EQ(plain[i].result_digest, checked[i].result_digest);
    EXPECT_EQ(plain[i].latencies_ns.size(), checked[i].latencies_ns.size());
  }
}

// Each ablation switch changes how a search runs, not what it returns: every
// query that both answer completely has the same digest. A switch that does
// nothing would pass this alone, so the kernel switches also show their
// counters, and the others a counter that must move.
TEST(SearchBenchTest, AblationSwitchesKeepDigestsAndTakeEffect) {
  ferret_testing::TempDir dir;
  const auto segment = BigSegment(dir, 1200);
  SearchBenchConfig base;
  base.warmup_passes = 0;
  base.timed_runs = 1;
  base.k = 3;
  FERRET_ASSERT_OK_AND_ASSIGN(auto full,
                              RunSearchBench(*segment, kBigQueries, base));
  struct Switch {
    const char* name;
    void (*apply)(SearchBenchConfig&);
  };
  const Switch switches[] = {
      {"scalar_decode", [](SearchBenchConfig& c) { c.scalar_decode = true; }},
      {"scalar_intersect",
       [](SearchBenchConfig& c) { c.scalar_intersect = true; }},
      {"no_skip_probe", [](SearchBenchConfig& c) { c.skip_probe = false; }},
      {"no_early_termination",
       [](SearchBenchConfig& c) {
         c.early_termination = false;
         c.line_cap_stop = false;
       }},
      {"no_name_aware_bound",
       [](SearchBenchConfig& c) { c.name_aware_bound = false; }},
      {"no_count_only", [](SearchBenchConfig& c) { c.count_only = false; }},
      {"no_parallel_verify",
       [](SearchBenchConfig& c) { c.parallel_verify = false; }},
      {"no_short_query_path",
       [](SearchBenchConfig& c) { c.bigram_posting_share = 0; }},
      {"no_count_sample",
       [](SearchBenchConfig& c) { c.count_sample = false; }}};
  for (const Switch& sw : switches) {
    SCOPED_TRACE(sw.name);
    SearchBenchConfig config = base;
    sw.apply(config);
    ferret::AblationCounts& counts = ferret::ThisThreadAblationCounts();
    const ferret::AblationCounts before = counts;
    FERRET_ASSERT_OK_AND_ASSIGN(auto variant,
                                RunSearchBench(*segment, kBigQueries, config));
    ASSERT_EQ(variant.size(), full.size());
    for (size_t i = 0; i < full.size(); ++i) {
      if (full[i].truncated || variant[i].truncated) continue;
      EXPECT_EQ(variant[i].result_digest, full[i].result_digest) << full[i].id;
    }
    const std::string name = sw.name;
    if (name == "scalar_decode") {
      EXPECT_GT(counts.scalar_blocks, before.scalar_blocks);
    } else if (name == "scalar_intersect") {
      EXPECT_GT(counts.scalar_merges, before.scalar_merges);
    } else if (name == "no_early_termination") {
      for (size_t i = 0; i < full.size(); ++i) {
        EXPECT_FALSE(variant[i].terminated_early);
        EXPECT_GE(variant[i].engine_stats.verified_bytes,
                  full[i].engine_stats.verified_bytes);
      }
    }
  }
  // The kernel switches are put back when a run ends.
  EXPECT_FALSE(ferret::AblationOn(ferret::Ablation::scalar_decode));
  EXPECT_FALSE(ferret::AblationOn(ferret::Ablation::scalar_intersect));
  EXPECT_FALSE(ferret::AblationOn(ferret::Ablation::no_skip_probe));
  const ferret::AblationCounts after = ferret::ThisThreadAblationCounts();
  FERRET_ASSERT_OK_AND_ASSIGN(auto again,
                              RunSearchBench(*segment, kBigQueries, base));
  EXPECT_EQ(ferret::ThisThreadAblationCounts().scalar_blocks,
            after.scalar_blocks);
}

TEST(SearchBenchTest, RejectsInvalidLimits) {
  ferret_testing::TempDir dir;
  ASSERT_THAT(dir.WriteFile("a.c", "abc\n"), IsOk());
  const std::vector<std::string> paths = {"a.c"};
  FERRET_ASSERT_OK_AND_ASSIGN(
      const ferret::BulkBuildStats build,
      ferret::BuildSegment(*ferret::Env::Default(), dir.path(), paths,
                           dir.Join("test.fseg")));
  EXPECT_EQ(build.files_indexed, 1);
  FERRET_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<ferret::Segment> segment,
      ferret::Segment::Open(*ferret::Env::Default(), dir.Join("test.fseg")));
  SearchBenchConfig config;
  config.clients = 0;
  EXPECT_THAT(
      RunSearchBench(*segment, {}, config),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("limits")));
}

}  // namespace
}  // namespace ferret_bench

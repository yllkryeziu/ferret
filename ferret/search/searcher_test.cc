// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/search/searcher.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/base/stage_clock.h"
#include "ferret/git/blob_hash.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/segment.h"
#include "ferret/index/segment_writer.h"
#include "ferret/regex/compile.h"
#include "ferret/regex/testing/regex_gen.h"
#include "ferret/search/ranking.h"
#include "ferret/search/search_types.h"
#include "ferret/testing/brute_force_search.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/testing/test_seed.h"
#include "ferret/text/language.h"
#include "ferret/text/lexer.h"

namespace ferret {
namespace {

using ::absl_testing::StatusIs;
using ::ferret_testing::CorpusFile;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::SizeIs;

SegmentDocument ToSegmentDocument(const CorpusFile& file) {
  return SegmentDocument{
      .hint_path = file.path,
      .content = file.content,
      .rank_key = ComputeRankKey(file.path, file.content.size(),
                                 /*long_lines=*/false, HashBlob(file.content)),
      .language = DetectLanguage(file.path, file.content),
  };
}

// Writes `corpus` to a new segment file in `dir`, in rank order, and opens
// it.
absl::StatusOr<std::unique_ptr<Segment>> BuildSegment(
    const ferret_testing::TempDir& dir, std::vector<CorpusFile> corpus,
    int64_t chunk_bytes = 8192, bool write_code_tokens = true,
    const std::string& name = "test.fseg",
    bool write_short_line_trigrams = true) {
  std::sort(
      corpus.begin(), corpus.end(),
      [](const CorpusFile& a, const CorpusFile& b) {
        return std::forward_as_tuple(ToSegmentDocument(a).rank_key, a.path) <
               std::forward_as_tuple(ToSegmentDocument(b).rank_key, b.path);
      });
  const std::string path = dir.Join(name);
  absl::StatusOr<std::unique_ptr<SegmentWriter>> writer = SegmentWriter::Create(
      *Env::Default(), path,
      {.chunk_bytes = chunk_bytes,
       .write_code_tokens = write_code_tokens,
       .write_short_line_trigrams = write_short_line_trigrams});
  if (!writer.ok()) return writer.status();
  for (const CorpusFile& file : corpus) {
    if (absl::Status s = (*writer)->Add(ToSegmentDocument(file)); !s.ok()) {
      return s;
    }
  }
  if (absl::Status s = (*writer)->Finish().status(); !s.ok()) return s;
  return Segment::Open(*Env::Default(), path);
}

SearchRequest Request(std::string pattern) {
  return SearchRequest{.pattern = std::move(pattern)};
}

std::vector<int64_t> LineNumbers(const FileResult& file) {
  std::vector<int64_t> numbers;
  numbers.reserve(file.lines.size());
  for (const ResultLine& line : file.lines) numbers.push_back(line.number);
  return numbers;
}

std::vector<int64_t> MatchLineNumbers(const FileResult& file) {
  std::vector<int64_t> numbers;
  numbers.reserve(file.lines.size());
  for (const ResultLine& line : file.lines) {
    if (line.is_match) numbers.push_back(line.number);
  }
  return numbers;
}

std::vector<std::string> Paths(const SearchResponse& response) {
  std::vector<std::string> paths;
  paths.reserve(response.files.size());
  for (const FileResult& file : response.files) paths.push_back(file.path);
  return paths;
}

// Expects `got` to have the files of `want`, comparing every field.
void ExpectSameFiles(const SearchResponse& got, const SearchResponse& want) {
  ASSERT_EQ(got.files.size(), want.files.size());
  for (size_t i = 0; i < want.files.size(); ++i) {
    SCOPED_TRACE(want.files[i].path);
    const FileResult& a = got.files[i];
    const FileResult& b = want.files[i];
    EXPECT_EQ(a.path, b.path);
    EXPECT_EQ(a.oid, b.oid);
    EXPECT_EQ(a.tier, b.tier);
    EXPECT_EQ(a.prior, b.prior);
    EXPECT_EQ(a.name_matches, b.name_matches);
    EXPECT_EQ(a.score, b.score);
    EXPECT_EQ(a.num_matching_lines, b.num_matching_lines);
    EXPECT_EQ(a.lines, b.lines);
  }
}

// Lines "line 1\n" through "line <n>\n", with `needle` appended to the given
// lines.
std::string NumberedLines(int n, const std::vector<int>& needle_lines,
                          absl::string_view needle = " needle") {
  std::string content;
  for (int i = 1; i <= n; ++i) {
    absl::StrAppend(&content, "line ", i,
                    std::find(needle_lines.begin(), needle_lines.end(), i) !=
                            needle_lines.end()
                        ? needle
                        : "",
                    "\n");
  }
  return content;
}

class SearcherTest : public ::testing::Test {
 protected:
  ferret_testing::TempDir dir_;
};

TEST_F(SearcherTest, StopsOnlyAfterTheStrictScoreBound) {
  // The equal file earns the name boost under either matcher, even when
  // the regexp has no required literal. Each depth costs a prior of 4,096.
  // The literal query is eligible for the code-token bound, whose ceiling
  // for a priority document is name plus match count, 12,288: the equal file
  // sits three levels below the root. The regexp keeps the full ceiling of
  // 20,480: five levels.
  for (const bool regex : {false, true}) {
    SCOPED_TRACE(regex);
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto segment,
        BuildSegment(
            dir_,
            {{"root.cc", "// needle first\n"},
             {regex ? "a/b/c/d/e/needle_equal.cc" : "a/b/c/needle_equal.cc",
              "// needle equal\n"},
             {regex ? "a/b/c/d/e/f/later.cc" : "a/b/c/d/later.cc",
              "// needle later\n"},
             {"tests/last_test.cc", "// needle test\n"}}));
    SearchRequest request = Request(regex ? "(needle|zzz)" : "needle");
    request.regex = regex;
    request.max_files = 1;
    FERRET_ASSERT_OK_AND_ASSIGN(auto early, SearchSegment(*segment, request));
    request.exact_counts = true;
    FERRET_ASSERT_OK_AND_ASSIGN(auto exact, SearchSegment(*segment, request));
    EXPECT_THAT(Paths(early), ElementsAre("root.cc"));
    EXPECT_TRUE(early.complete);
    EXPECT_TRUE(early.terminated_early);
    EXPECT_EQ(early.stats.verified_chunks, 2);
    EXPECT_EQ(early.stats.priority_documents, 1);
    EXPECT_LT(early.stats.candidate_chunks, exact.stats.candidate_chunks);
    EXPECT_EQ(exact.stats.verified_chunks, 4);
    EXPECT_FALSE(exact.terminated_early);

    request.exact_counts = false;
    request.max_lines = 0;
    FERRET_ASSERT_OK_AND_ASSIGN(auto no_kept, SearchSegment(*segment, request));
    EXPECT_THAT(no_kept.files, IsEmpty());
    EXPECT_FALSE(no_kept.terminated_early);
    EXPECT_EQ(no_kept.stats.verified_chunks, 4);
  }
}

// The winner's name matches and no later name does. Every later file has a
// prior 4,096 below it, so the full ceiling can only tie the winner and must
// verify them all; without the name boost, the first one already falls short.
TEST_F(SearcherTest, NameAwareBoundStopsWhereTheFullCeilingCannot) {
  std::vector<CorpusFile> corpus = {{"needle.cc", "int needle = 0;\n"}};
  for (int i = 0; i < 40; ++i) {
    corpus.push_back(
        {absl::StrCat("d/f", i, ".cc"), absl::StrCat("// needle ", i, "\n")});
  }
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment, BuildSegment(dir_, corpus));
  SearchRequest request = Request("needle");
  request.max_files = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(auto early, SearchSegment(*segment, request));
  request.exact_counts = true;
  FERRET_ASSERT_OK_AND_ASSIGN(auto exact, SearchSegment(*segment, request));
  ASSERT_THAT(Paths(early), ElementsAre("needle.cc"));
  ExpectSameFiles(early, exact);
  const RankKey later = ComputeRankKey(corpus[1].path, corpus[1].content.size(),
                                       false, HashBlob(corpus[1].content));
  EXPECT_FALSE(CannotOutrank(later, kMaxBoost, early.files[0]));
  EXPECT_TRUE(CannotOutrank(later, kMaxBoostWithoutName, early.files[0]));
  EXPECT_TRUE(early.terminated_early);
  EXPECT_EQ(early.stats.priority_documents, 1);
  EXPECT_EQ(early.stats.verified_chunks, 1);
  EXPECT_EQ(exact.stats.verified_chunks, 41);
  EXPECT_EQ(exact.stats.priority_documents, 0);
}

// The residual ceiling is strict as well: a file whose name does not match
// can tie the heap's worst file with the other boosts and win on its path.
TEST_F(SearcherTest, ResidualCeilingIsStrict) {
  std::string tie;
  for (int i = 0; i < 16; ++i) tie += "int needle = 0;\n";
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto segment,
      BuildSegment(dir_, {{"z.cc", "// needle\n"},
                          {"a/b/c/tie.cc", tie},
                          {"a/b/c/d/e/f/late.cc", "// needle late\n"}}));
  SearchRequest request = Request("needle");
  request.max_files = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(auto early, SearchSegment(*segment, request));
  request.exact_counts = true;
  FERRET_ASSERT_OK_AND_ASSIGN(auto exact, SearchSegment(*segment, request));
  // 65535 - 3 * 4096 + kMaxBoostWithoutName equals the root file's prior.
  ASSERT_THAT(Paths(exact), ElementsAre("a/b/c/tie.cc"));
  EXPECT_EQ(exact.files[0].score, 65535);
  ExpectSameFiles(early, exact);
  EXPECT_TRUE(early.terminated_early);
  EXPECT_EQ(early.stats.verified_chunks, 2);
}

// A file deep in rank order whose name matches keeps the full ceiling, and
// here it ties the root file's score and wins on its path. Nonmatching names
// between them are within reach of neither ceiling.
TEST_F(SearcherTest, DeepNameMatchTiesIntoTheTopK) {
  std::string deep;
  for (int i = 0; i < 16; ++i) deep += "int needle = 0;\n";
  std::vector<CorpusFile> corpus = {{"z.cc", "// needle\n"},
                                    {"a/b/c/d/e/needle.cc", deep}};
  for (int i = 0; i < 8; ++i) {
    corpus.push_back(
        {absl::StrCat("a/b/c/mid", i, ".cc"),
         absl::StrCat("// needle ", i, "\n", std::string(4096, 'x'), "\n")});
    corpus.push_back({absl::StrCat("a/b/c/d/e/f/late", i, ".cc"),
                      absl::StrCat("// needle ", i, "\n")});
  }
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment, BuildSegment(dir_, corpus));
  SearchRequest request = Request("needle");
  request.max_files = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(auto early, SearchSegment(*segment, request));
  request.exact_counts = true;
  FERRET_ASSERT_OK_AND_ASSIGN(auto exact, SearchSegment(*segment, request));
  // 65535 - 5 * 4096 plus every boost equals the root file's prior.
  ASSERT_THAT(Paths(exact), ElementsAre("a/b/c/d/e/needle.cc"));
  EXPECT_EQ(exact.files[0].score, 65535);
  ExpectSameFiles(early, exact);
  EXPECT_TRUE(early.terminated_early);
  EXPECT_EQ(early.stats.priority_documents, 1);
  EXPECT_EQ(early.stats.verified_chunks, 10);
  EXPECT_EQ(exact.stats.verified_chunks, 18);
}

// Without a required literal, the compiled matcher still identifies the
// priority name: the residual ceiling alone would miss the winner.
TEST_F(SearcherTest, RegexWithoutRequiredLiteralFindsPriorityName) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto segment,
      BuildSegment(dir_, {{"x.cc", "// needle other\n"},
                          {"a/b/c/other.cc",
                           absl::StrCat("int other = 1;\nint other = 2;\n",
                                        std::string(4096, 'x'), "\n")},
                          {"a/b/c/d/e/late.cc", "// needle other late\n"}}));
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto alternation,
      CompilePattern({.pattern = "(needle|other)", .case_sensitive = false}));
  ASSERT_THAT(alternation->required_literal(), IsEmpty());
  for (const bool regex : {true, false}) {
    SCOPED_TRACE(regex);
    SearchRequest request = Request(regex ? "(needle|other)" : "other");
    request.regex = regex;
    request.max_files = 1;
    FERRET_ASSERT_OK_AND_ASSIGN(auto early, SearchSegment(*segment, request));
    request.exact_counts = true;
    FERRET_ASSERT_OK_AND_ASSIGN(auto exact, SearchSegment(*segment, request));
    // x.cc scores 65535 and other.cc 65535 - 3 * 4096 - 2048 + 17408, but
    // 65535 - 3 * 4096 - 2048 + kMaxBoostWithoutName is below 65535.
    ASSERT_THAT(Paths(exact), ElementsAre("a/b/c/other.cc"));
    ExpectSameFiles(early, exact);
    EXPECT_EQ(early.stats.priority_documents, 1);
  }
}

// An alternation has no required literal. Testing all basenames still makes
// the residual bound useful, both with and without case folding.
TEST_F(SearcherTest, AlternationWithoutLiteralStopsOnResidualBound) {
  std::vector<CorpusFile> corpus = {{"z.cc", "int needle = 0;\n"}};
  for (int i = 0; i < 30; ++i) {
    corpus.push_back({absl::StrCat("a/b/c/mid", i, ".cc"),
                      absl::StrCat("// needle ", i, "\n")});
  }
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment, BuildSegment(dir_, corpus));
  for (const bool case_sensitive : {false, true}) {
    SearchRequest request =
        Request(case_sensitive ? "(needle|other)" : "(NEEDLE|OTHER)");
    request.regex = true;
    request.case_sensitive = case_sensitive;
    request.max_files = 1;
    FERRET_ASSERT_OK_AND_ASSIGN(auto early, SearchSegment(*segment, request));
    request.exact_counts = true;
    FERRET_ASSERT_OK_AND_ASSIGN(auto exact, SearchSegment(*segment, request));
    ExpectSameFiles(early, exact);
    EXPECT_TRUE(early.terminated_early);
    EXPECT_LT(early.stats.verified_bytes * 3, exact.stats.verified_bytes);
    EXPECT_EQ(early.stats.verified_chunks, 1);
  }
}

// A document whose name matches the pattern but which has no candidate chunk
// can never become a result, so it must not hold the rank stop open. The
// unrestricted set keeps it and verifies the files before it; the set of
// candidate documents drops it. Both give the same files.
TEST_F(SearcherTest, PriorityDocumentWithoutCandidatesDoesNotBlockTheStop) {
  std::vector<CorpusFile> corpus = {{"needle.cc", "int needle = 0;\n"},
                                    {"a/needle_deep.cc", "unrelated\n"}};
  for (int i = 0; i < 30; ++i) {
    corpus.push_back(
        {absl::StrCat("d/f", i, ".cc"), absl::StrCat("// needle ", i, "\n")});
  }
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment, BuildSegment(dir_, corpus));
  for (const char* pattern : {"(?:needle|zzz)"}) {
    SCOPED_TRACE(pattern);
    SearchRequest request = Request(pattern);
    request.regex = true;
    request.max_files = 1;
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto full,
        SearchSegment(*segment, request,
                      {.priority_restriction = PriorityRestriction::kNever}));
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto restricted,
        SearchSegment(*segment, request,
                      {.priority_restriction = PriorityRestriction::kAlways}));
    FERRET_ASSERT_OK_AND_ASSIGN(auto automatic,
                                SearchSegment(*segment, request));
    ASSERT_THAT(Paths(full), ElementsAre("needle.cc"));
    ExpectSameFiles(restricted, full);
    ExpectSameFiles(automatic, full);
    EXPECT_EQ(full.stats.priority_documents, 2);
    EXPECT_EQ(restricted.stats.priority_documents, 1);
    EXPECT_EQ(restricted.stats.priority_restricted_sources, 1);
    EXPECT_EQ(full.stats.priority_restricted_sources, 0);
    EXPECT_TRUE(restricted.terminated_early);
    EXPECT_EQ(restricted.stats.verified_chunks, 1);
    EXPECT_GT(full.stats.verified_chunks, restricted.stats.verified_chunks);
    // The counters: the basenames the restricted scan tested are the
    // candidates' (32 at most), not all of them.
    EXPECT_LE(restricted.stats.priority_basenames_tested, 32);
  }
}

// A basename can match while every candidate chunk fails the line matcher.
// It cannot add a file even when its name blocks the bound.
TEST_F(SearcherTest, PriorityNameWithoutMatchingLineDoesNotAddAFile) {
  std::vector<CorpusFile> corpus = {{"z.cc", "needle();\n"},
                                    {"a/b/c/needle.cc", "// needle comment\n"}};
  for (int i = 0; i < 12; ++i) {
    corpus.push_back({absl::StrCat("a/b/c/mid", i, ".cc"),
                      absl::StrCat("needle ", i, "\n")});
  }
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment, BuildSegment(dir_, corpus));
  SearchRequest request = Request("^needle");
  request.regex = true;
  request.max_files = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(auto early, SearchSegment(*segment, request));
  request.exact_counts = true;
  FERRET_ASSERT_OK_AND_ASSIGN(auto exact, SearchSegment(*segment, request));
  ExpectSameFiles(early, exact);
  EXPECT_FALSE(early.terminated_early);
  EXPECT_EQ(early.stats.priority_documents, 1);
  EXPECT_EQ(early.stats.verified_bytes, exact.stats.verified_bytes);
}

// Priority documents are the basenames the response's name test accepts: it
// is case-sensitive or not like the pattern, folds U+212A KELVIN SIGN with
// 'k', and sees only files that pass the glob.
TEST_F(SearcherTest, PriorityDocumentsFollowTheNameMatch) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto segment,
      BuildSegment(dir_, {{"a.cc", "int needle = 0;\nint kernel = 0;\n"},
                          {"src/x/NEEDLE.cc", "// needle kernel 1\n"},
                          {"src/x/needle.cc", "// needle kernel 2\n"},
                          {"lib/x/needle2.cc", "// needle kernel 3\n"},
                          {"src/x/\u212Aernel.cc", "// needle kernel 4\n"},
                          {"src/x/KERNEL.cc", "// needle kernel 5\n"},
                          {"src/x/y/z/w/v/late.cc", "// needle kernel 6\n"}}));
  struct Case {
    const char* pattern;
    bool case_sensitive;
    const char* glob;
    int64_t priority_documents;
  };
  for (const Case& c :
       {Case{"needle", true, "", 2}, Case{"needle", false, "", 3},
        Case{"needle", false, "!lib", 2}, Case{"kernel", false, "", 2},
        Case{"kernel", true, "", 0}}) {
    SCOPED_TRACE(absl::StrCat(c.pattern, " ", c.case_sensitive, " ", c.glob));
    SearchRequest request = Request(c.pattern);
    request.case_sensitive = c.case_sensitive;
    request.path_glob = c.glob;
    request.max_files = 1;
    FERRET_ASSERT_OK_AND_ASSIGN(auto early, SearchSegment(*segment, request));
    request.exact_counts = true;
    FERRET_ASSERT_OK_AND_ASSIGN(auto exact, SearchSegment(*segment, request));
    EXPECT_THAT(Paths(early), ElementsAre("a.cc"));
    ExpectSameFiles(early, exact);
    EXPECT_TRUE(early.terminated_early);
    EXPECT_EQ(early.stats.priority_documents, c.priority_documents);
  }
}

// One segment can hold a blob under two equally ranked paths, as two
// documents with equal RankKeys. Both are priority documents within reach of
// the full ceiling, so both are verified; the stop comes at the next file.
TEST_F(SearcherTest, PriorityDocumentsWithEqualRankKeys) {
  // The priority documents' ceiling under the code-token bound is 12,288,
  // which ties the root file's identifier score one level below it.
  const std::vector<CorpusFile> corpus = {{"w.cc", "int needle;\n"},
                                          {"a/needle.cc", "// needle\n"},
                                          {"c/needle.cc", "// needle\n"},
                                          {"e/f/g/h/late.cc", "// needle\n"}};
  ASSERT_EQ(ToSegmentDocument(corpus[1]).rank_key,
            ToSegmentDocument(corpus[2]).rank_key);
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment, BuildSegment(dir_, corpus));
  SearchRequest request = Request("needle");
  request.max_files = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(auto early, SearchSegment(*segment, request));
  request.exact_counts = true;
  FERRET_ASSERT_OK_AND_ASSIGN(auto exact, SearchSegment(*segment, request));
  ASSERT_THAT(Paths(early), ElementsAre("w.cc"));
  ExpectSameFiles(early, exact);
  EXPECT_FALSE(CannotOutrank(ToSegmentDocument(corpus[1]).rank_key, kMaxBoost,
                             early.files[0]));
  EXPECT_FALSE(CannotOutrank(ToSegmentDocument(corpus[1]).rank_key,
                             kMaxBoostWithoutIdentifier, early.files[0]));
  EXPECT_TRUE(early.terminated_early);
  EXPECT_EQ(early.stats.priority_documents, 2);
  EXPECT_EQ(early.stats.verified_chunks, 3);
  EXPECT_EQ(exact.stats.verified_chunks, 4);
}

TEST_F(SearcherTest, KeepsMatchesWithMergedClippedContext) {
  // Every line is its own chunk with chunk_bytes = 1, so line numbers and
  // context cross chunk boundaries.
  for (const int64_t chunk_bytes : {1, 5, 8192}) {
    SCOPED_TRACE(chunk_bytes);
    FERRET_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Segment> segment,
        BuildSegment(dir_, {{"a.cc", NumberedLines(12, {3, 5, 11})}},
                     chunk_bytes));
    FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse response,
                                SearchSegment(*segment, Request("needle")));
    ASSERT_THAT(response.files, SizeIs(1));
    const FileResult& file = response.files[0];
    EXPECT_EQ(file.path, "a.cc");
    EXPECT_EQ(file.num_matching_lines, 3);
    EXPECT_THAT(LineNumbers(file),
                ElementsAre(1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12));
    EXPECT_THAT(MatchLineNumbers(file), ElementsAre(3, 5, 11));
    EXPECT_EQ(file.lines[2].text, "line 3 needle");
    EXPECT_EQ(file.lines[2].match_begin, size_t{7});
    EXPECT_EQ(file.lines[2].match_end, size_t{13});
    EXPECT_EQ(file.lines[0].text, "line 1");
    EXPECT_EQ(response.num_matching_files, 1);
    EXPECT_EQ(response.num_matching_lines, 3);
    EXPECT_TRUE(response.complete);
    EXPECT_EQ(response.verified_fraction, 1.0);
  }
}

TEST_F(SearcherTest, ContextAtUnterminatedLastLine) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Segment> segment,
      BuildSegment(dir_, {{"a.cc", "one\ntwo\nneedle"}}, /*chunk_bytes=*/1));
  SearchRequest request = Request("needle");
  request.context_lines = 5;
  FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse response,
                              SearchSegment(*segment, request));
  ASSERT_THAT(response.files, SizeIs(1));
  EXPECT_THAT(LineNumbers(response.files[0]), ElementsAre(1, 2, 3));
  EXPECT_EQ(response.files[0].lines[2].text, "needle");
}

TEST_F(SearcherTest, CountSampleEstimatesAndKeepsTopKAcrossThreads) {
  std::vector<CorpusFile> corpus = {{"z.cc", "int needle = 0;\n"}};
  for (int i = 0; i < 128; ++i) {
    const std::string content = i % 4 == 0   ? "int needle = 0;\n"
                                : i % 4 == 1 ? "// needle\n"
                                : i % 4 == 2 ? "auto s = \"needle\";\n"
                                             : "int needles = 0;\n";
    corpus.push_back({absl::StrCat("a/b/c/deep", i, ".cc"),
                      absl::StrCat(content, "// id ", i, "\n")});
  }
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment, BuildSegment(dir_, corpus));
  SearchRequest request = Request("needle");
  request.max_files = 1;
  request.count_sample = true;
  FERRET_ASSERT_OK_AND_ASSIGN(auto inline_result,
                              SearchSegment(*segment, request));
  request.exact_counts = true;
  FERRET_ASSERT_OK_AND_ASSIGN(auto exact, SearchSegment(*segment, request));
  ExpectSameFiles(inline_result, exact);
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto oracle, ferret_testing::BruteForceSearch(corpus, request));
  for (const FileResult& selected : inline_result.files) {
    auto it = std::find_if(oracle.begin(), oracle.end(),
                           [&](const FileResult& candidate) {
                             return candidate.path == selected.path;
                           });
    ASSERT_NE(it, oracle.end());
    EXPECT_EQ(selected.num_matching_lines, it->num_matching_lines);
    ASSERT_EQ(selected.lines.size(), it->lines.size());
    for (size_t i = 0; i < selected.lines.size(); ++i) {
      EXPECT_EQ(selected.lines[i].number, it->lines[i].number);
      EXPECT_EQ(selected.lines[i].text, it->lines[i].text);
    }
  }
  EXPECT_TRUE(inline_result.terminated_early);
  EXPECT_TRUE(inline_result.counts_estimated);
  EXPECT_EQ(inline_result.count_sample_size, 64);
  EXPECT_NEAR(inline_result.num_matching_files, exact.num_matching_files, 15);
  EXPECT_NEAR(inline_result.num_matching_lines, exact.num_matching_lines, 15);
  EXPECT_NEAR(inline_result.token_precision, 0.25, 0.12);
  EXPECT_FALSE(exact.counts_estimated);
  EXPECT_EQ(exact.count_sample_size, 0);
  EXPECT_EQ(exact.num_matching_files, 129);
  EXPECT_DOUBLE_EQ(exact.token_precision, 33.0 / 129.0);
  request.exact_counts = false;
  for (int threads : {1, 4}) {
    SearchExecutor executor(threads);
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto parallel,
        SearchSegment(*segment, request,
                      {.executor = &executor, .parallel_threshold_bytes = 0}));
    ExpectSameFiles(parallel, inline_result);
    EXPECT_EQ(parallel.num_matching_files, inline_result.num_matching_files);
    EXPECT_EQ(parallel.num_matching_lines, inline_result.num_matching_lines);
    EXPECT_EQ(parallel.match_classes, inline_result.match_classes);
  }
  request.max_verify_bytes = 500;
  FERRET_ASSERT_OK_AND_ASSIGN(auto bounded, SearchSegment(*segment, request));
  ExpectSameFiles(bounded, inline_result);
  EXPECT_FALSE(bounded.counts_estimated);
  EXPECT_EQ(bounded.count_sample_size, 0);
  EXPECT_LE(bounded.stats.verified_bytes, request.max_verify_bytes);
  request.max_verify_bytes = SearchRequest().max_verify_bytes;
  request.count_sample = false;
  FERRET_ASSERT_OK_AND_ASSIGN(auto off, SearchSegment(*segment, request));
  ExpectSameFiles(off, inline_result);
  EXPECT_FALSE(off.counts_estimated);
  EXPECT_EQ(off.count_sample_size, 0);
}

TEST_F(SearcherTest, CountSampleEstimatesFalsePositiveCandidates) {
  std::vector<CorpusFile> corpus = {{"z.cc", "needle\n"}};
  for (int i = 0; i < 120; ++i) {
    corpus.push_back(
        {absl::StrCat("a/b/c/deep", i, ".cc"),
         absl::StrCat(i % 2 == 0 ? "// needle\n" : "nee eed edl dle\n",
                      "// id ", i, "\n")});
  }
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment, BuildSegment(dir_, corpus));
  SearchRequest request = Request("needle");
  request.max_files = 1;
  request.count_sample = true;
  FERRET_ASSERT_OK_AND_ASSIGN(auto estimated, SearchSegment(*segment, request));
  EXPECT_TRUE(estimated.terminated_early);
  EXPECT_TRUE(estimated.counts_estimated);
  EXPECT_EQ(estimated.count_sample_size, 64);
  EXPECT_NEAR(estimated.num_matching_files, 61, 20);
  EXPECT_NEAR(estimated.num_matching_lines, 61, 20);
  request.exact_counts = true;
  FERRET_ASSERT_OK_AND_ASSIGN(auto exact, SearchSegment(*segment, request));
  EXPECT_EQ(exact.num_matching_files, 61);
  EXPECT_EQ(exact.num_matching_lines, 61);
  EXPECT_FALSE(exact.counts_estimated);
  ExpectSameFiles(estimated, exact);
}

// A document with several candidate chunks is one document of the sample's
// frame: every document here matches in all its chunks, so the estimated totals
// must equal the exact ones whatever the sample is.
TEST_F(SearcherTest, CountSampleCountsAMultiChunkDocumentOnce) {
  std::vector<CorpusFile> corpus = {{"z.cc", "needle\n"}};
  for (int i = 0; i < 150; ++i) {
    std::string content;
    for (int line = 0; line < 6; ++line) {
      absl::StrAppend(&content, "// needle ", i, " line ", line, "\n");
    }
    corpus.push_back({absl::StrCat("a/b/c/deep", i, ".cc"), content});
  }
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment,
                              BuildSegment(dir_, corpus, /*chunk_bytes=*/1));
  ASSERT_GT(segment->num_chunks(), 3 * segment->num_docs());
  SearchRequest request = Request("needle");
  request.max_files = 1;
  request.count_sample = true;
  request.exact_counts = true;
  FERRET_ASSERT_OK_AND_ASSIGN(auto exact, SearchSegment(*segment, request));
  EXPECT_EQ(exact.num_matching_files, 151);
  EXPECT_EQ(exact.num_matching_lines, 1 + 150 * 6);
  request.exact_counts = false;
  for (int level : {0, 1, 2}) {
    SCOPED_TRACE(level);
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto estimated,
        SearchSegment(*segment, request, {.count_sample_level = level}));
    ASSERT_TRUE(estimated.terminated_early);
    ASSERT_TRUE(estimated.counts_estimated);
    EXPECT_EQ(estimated.count_sample_size, 64);
    EXPECT_EQ(estimated.num_matching_files, exact.num_matching_files);
    EXPECT_EQ(estimated.num_matching_lines, exact.num_matching_lines);
    EXPECT_GT(estimated.stats.count_sample_documents, 64);
    EXPECT_LT(estimated.stats.count_sample_documents, 151);
  }
}

// The count sample's estimator is the expansion estimator over a hash
// sample of 64 documents, so each total is unbiased and its relative standard
// error is about sqrt((1 - n/N) * cv^2 / n) for n = 64 documents of N: at most
// 0.14 for the files, lines and classes of these corpora (cv about 1). The
// mean error over `kSeeds` corpora then has a standard error of 0.14 / sqrt(60)
// = 0.018, so a tolerance of 0.06 on the mean is more than 3 sigma. The per
// corpus error is only bounded loosely (0.6): a heavy document may fall in or
// out of the sample.
TEST_F(SearcherTest, CountSampleEstimatesAreUnbiasedOverManyCorpora) {
  constexpr int kSeeds = 60;
  const uint64_t seed = ferret_testing::TestSeed(0x7a11e5);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  const char* const lines[] = {
      "int needle = 0;",  "// needle here", "auto s = \"a needle\";",
      "int needles = 1;", "other text",     "call(needle, needle);"};
  const char* const roots[] = {"src/", "test/", "third_party/", "docs/"};
  const char* const names[] = {"needle", "Needle", "other", "a_needle_b",
                               "plain"};
  const char* const extensions[] = {".cc", ".h", ".txt"};
  double files = 0, line_total = 0;
  std::array<double, 4> classes = {};
  std::array<double, 4> class_used = {};
  int64_t counted = 0;
  int estimated = 0;
  for (int round = 0; round < kSeeds; ++round) {
    std::vector<CorpusFile> corpus = {{"src/needle.cc", "int needle = 0;\n"}};
    const int documents = 300 + static_cast<int>(rng() % 300);
    for (int i = 0; i < documents; ++i) {
      std::string content;
      const int count = 1 + static_cast<int>(rng() % 8);
      for (int j = 0; j < count; ++j) {
        absl::StrAppend(&content, lines[rng() % 6], " // ", j, "\n");
      }
      // Distinct Oids: documents sharing one are drawn together by the hash.
      absl::StrAppend(&content, "// id ", round, "/", i, "\n");
      corpus.push_back(
          {absl::StrCat(roots[rng() % 4], "d", rng() % 7, "/", names[rng() % 5],
                        i, extensions[rng() % 3]),
           content});
    }
    ferret_testing::TempDir round_dir;
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto segment, BuildSegment(round_dir, corpus, /*chunk_bytes=*/64));
    SearchRequest request = Request("needle");
    request.max_files = 3;
    request.count_sample = true;
    request.exact_counts = true;
    FERRET_ASSERT_OK_AND_ASSIGN(auto exact, SearchSegment(*segment, request));
    request.exact_counts = false;
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto got,
        SearchSegment(*segment, request, {.count_only_min_bytes = 0}));
    ASSERT_TRUE(got.terminated_early);
    estimated += got.counts_estimated;
    counted += got.stats.count_only_documents;
    const auto rel = [](double a, double b) { return a / b - 1; };
    EXPECT_LT(std::abs(rel(got.num_matching_files, exact.num_matching_files)),
              0.6);
    files += rel(got.num_matching_files, exact.num_matching_files);
    line_total += rel(got.num_matching_lines, exact.num_matching_lines);
    for (size_t c = 0; c < 4; ++c) {
      if (exact.match_classes[c] < 20) continue;
      classes[c] += rel(got.match_classes[c], exact.match_classes[c]);
      class_used[c] += 1;
    }
  }
  EXPECT_EQ(estimated, kSeeds);
  EXPECT_NEAR(files / kSeeds, 0, 0.06);
  EXPECT_NEAR(line_total / kSeeds, 0, 0.06);
  for (size_t c = 0; c < 4; ++c) {
    SCOPED_TRACE(c);
    ASSERT_GT(class_used[c], 0);
    EXPECT_NEAR(classes[c] / class_used[c], 0, 0.06);
  }
  EXPECT_GT(counted, 0);
}

TEST_F(SearcherTest, PrecisionCountsEveryOccurrenceOnOneLine) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto segment,
      BuildSegment(dir_, {{"file.cc", "int needle = needle(); // needle\n"}}));
  SearchRequest request = Request("needle");
  request.count_sample = true;
  FERRET_ASSERT_OK_AND_ASSIGN(auto result, SearchSegment(*segment, request));
  EXPECT_FALSE(result.counts_estimated);
  EXPECT_EQ(result.count_sample_size, 0);
  EXPECT_EQ(result.num_matching_lines, 1);
  EXPECT_EQ(result.match_classes[0], 2);
  EXPECT_EQ(result.match_classes[2], 1);
  EXPECT_DOUBLE_EQ(result.token_precision, 2.0 / 3.0);
}

TEST_F(SearcherTest, ClassifiesMatches) {
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Segment> segment,
                              BuildSegment(dir_, {{"a.cc",
                                                   "int needle = 1;\n"
                                                   "needles++;\n"
                                                   "// needle\n"
                                                   "s = \"needle\";\n"}}));
  SearchRequest request = Request("needle");
  request.context_lines = 0;
  FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse response,
                              SearchSegment(*segment, request));
  ASSERT_THAT(response.files, SizeIs(1));
  EXPECT_THAT(
      response.files[0].lines,
      ElementsAre(Field(&ResultLine::match_class, MatchClass::kCodeIdentifier),
                  Field(&ResultLine::match_class, MatchClass::kPartialToken),
                  Field(&ResultLine::match_class, MatchClass::kComment),
                  Field(&ResultLine::match_class, MatchClass::kString)));
}

TEST_F(SearcherTest, LiteralRegexAndCase) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Segment> segment,
      BuildSegment(dir_, {{"a.cc", "Needle\nneedle\nne+dle\nneeedle\n"}}));
  auto match_lines = [&](SearchRequest request) {
    request.context_lines = 0;
    absl::StatusOr<SearchResponse> response = SearchSegment(*segment, request);
    EXPECT_TRUE(response.ok()) << response.status();
    return response.ok() && !response->files.empty()
               ? MatchLineNumbers(response->files[0])
               : std::vector<int64_t>();
  };
  EXPECT_THAT(match_lines(Request("needle")), ElementsAre(1, 2));
  SearchRequest request = Request("needle");
  request.case_sensitive = true;
  EXPECT_THAT(match_lines(request), ElementsAre(2));
  EXPECT_THAT(match_lines(Request("ne+dle")), ElementsAre(3));
  request = Request("ne+dle");
  request.regex = true;
  EXPECT_THAT(match_lines(request), ElementsAre(1, 2, 4));
}

TEST_F(SearcherTest, RanksByTierThenScoreThenPath) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Segment> segment,
      BuildSegment(dir_, {
                             {"third_party/y.cc", "needle();\n"},
                             {"x_test.cc", "needle();\n"},
                             {"a.cc", "needles();\n"},
                             {"b.cc", "needle();\n"},
                             {"c.cc", "needles();\n"},
                         }));
  FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse response,
                              SearchSegment(*segment, Request("needle")));
  EXPECT_THAT(Paths(response), ElementsAre("b.cc", "a.cc", "c.cc", "x_test.cc",
                                           "third_party/y.cc"));
}

TEST_F(SearcherTest, NameMatches) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Segment> segment,
      BuildSegment(dir_, {{"dir/needle.cc", "needles\n"},
                          {"needle/other.cc", "needles\n"}}));
  FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse response,
                              SearchSegment(*segment, Request("needle")));
  ASSERT_THAT(Paths(response), ElementsAre("dir/needle.cc", "needle/other.cc"));
  EXPECT_TRUE(response.files[0].name_matches);
  EXPECT_FALSE(response.files[1].name_matches);
}

TEST_F(SearcherTest, LimitsKeptLinesButCountsAll) {
  // Deeper files have lower priors, so the files are verified in this order.
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Segment> segment,
      BuildSegment(dir_, {{"a.cc", NumberedLines(5, {1, 2, 3, 4, 5})},
                          {"d1/b.cc", NumberedLines(3, {1, 2, 3})},
                          {"d1/d2/c.cc", NumberedLines(2, {1, 2})}}));
  SearchRequest request = Request("needle");
  request.context_lines = 0;
  request.max_lines_per_file = 2;
  request.max_lines = 3;
  request.exact_counts = true;
  FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse response,
                              SearchSegment(*segment, request));
  ASSERT_THAT(Paths(response), ElementsAre("a.cc", "d1/b.cc"));
  EXPECT_THAT(MatchLineNumbers(response.files[0]), ElementsAre(1, 2));
  EXPECT_EQ(response.files[0].num_matching_lines, 5);
  EXPECT_THAT(MatchLineNumbers(response.files[1]), ElementsAre(1));
  EXPECT_EQ(response.files[1].num_matching_lines, 3);
  EXPECT_EQ(response.num_matching_files, 3);
  EXPECT_EQ(response.num_matching_lines, 10);
  EXPECT_TRUE(response.complete);
}

TEST_F(SearcherTest, LineCapStopLeavesTheExhaustiveTopFiles) {
  std::vector<CorpusFile> files;
  // Files differ in match count, so scores differ. A stop before the cap is
  // spent would drop the best file read; later files keep no lines.
  for (int i = 0; i < 12; ++i) {
    std::vector<int> lines;
    for (int l = 1; l <= i + 1; ++l) lines.push_back(l);
    files.push_back({absl::StrCat("d", i, "/f.cc"), NumberedLines(13, lines)});
  }
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Segment> segment,
                              BuildSegment(dir_, files));
  SearchRequest request = Request("needle");
  request.context_lines = 0;
  request.max_files = 1;
  request.max_lines_per_file = 1;
  request.max_lines = 6;
  SearcherOptions off;
  off.stop_at_line_cap = false;
  SearcherOptions on;
  SearchRequest exhaustive = request;
  exhaustive.exact_counts = true;
  FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse full,
                              SearchSegment(*segment, exhaustive, off));
  FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse stopped,
                              SearchSegment(*segment, request, on));
  EXPECT_TRUE(stopped.terminated_early);
  EXPECT_EQ(Paths(stopped), Paths(full));
  ASSERT_EQ(stopped.files.size(), full.files.size());
  for (size_t i = 0; i < full.files.size(); ++i) {
    EXPECT_EQ(MatchLineNumbers(stopped.files[i]),
              MatchLineNumbers(full.files[i]));
  }
}

// The ablation switches change when a search stops, not what it returns. Same
// corpus as above: the winner's name matches and no later name does.
TEST_F(SearcherTest, EarlyTerminationAndNameAwareSwitchesKeepTheFiles) {
  std::vector<CorpusFile> corpus = {{"needle.cc", "int needle = 0;\n"}};
  for (int i = 0; i < 40; ++i) {
    corpus.push_back(
        {absl::StrCat("d/f", i, ".cc"), absl::StrCat("// needle ", i, "\n")});
  }
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment, BuildSegment(dir_, corpus));
  SearchRequest request = Request("needle");
  request.max_files = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(auto early, SearchSegment(*segment, request));
  ASSERT_TRUE(early.terminated_early);
  ASSERT_EQ(early.stats.verified_chunks, 1);

  SearcherOptions no_stop;
  no_stop.early_termination = false;
  FERRET_ASSERT_OK_AND_ASSIGN(auto full,
                              SearchSegment(*segment, request, no_stop));
  EXPECT_FALSE(full.terminated_early);
  EXPECT_EQ(full.stats.verified_chunks, 41);
  EXPECT_EQ(full.stats.priority_documents, 0);
  ExpectSameFiles(full, early);
  EXPECT_EQ(full.num_matching_files, 41);

  SearcherOptions no_names;
  no_names.name_aware_bound = false;
  FERRET_ASSERT_OK_AND_ASSIGN(auto coarse,
                              SearchSegment(*segment, request, no_names));
  EXPECT_FALSE(coarse.terminated_early);
  EXPECT_EQ(coarse.stats.verified_chunks, 41);
  EXPECT_EQ(coarse.stats.priority_documents, 0);
  ExpectSameFiles(coarse, early);

  // Through the parallel pipeline, whose workers apply the same stop test.
  SearchExecutor executor(3);
  for (const SearcherOptions* base : {&no_stop, &no_names}) {
    SearcherOptions options = *base;
    options.executor = &executor;
    options.parallel_threshold_bytes = 0;
    FERRET_ASSERT_OK_AND_ASSIGN(auto parallel,
                                SearchSegment(*segment, request, options));
    EXPECT_FALSE(parallel.terminated_early);
    EXPECT_EQ(parallel.stats.verified_chunks, 41);
    ExpectSameFiles(parallel, early);
  }
}

TEST_F(SearcherTest, StageTimingFillsTheBreakdownOnlyWhenAsked) {
  std::vector<CorpusFile> corpus;
  corpus.reserve(400);
  for (int i = 0; i < 400; ++i) {
    corpus.push_back({absl::StrCat("d", i % 7, "/f", i, ".cc"),
                      absl::StrCat("int needle_haystack = ", i, ";\n")});
  }
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment, BuildSegment(dir_, corpus));
  SearchRequest request = Request("needle_haystack = 1");
  request.max_files = 5;
  FERRET_ASSERT_OK_AND_ASSIGN(auto off, SearchSegment(*segment, request));
  EXPECT_EQ(off.stats.decode_time, absl::ZeroDuration());
  EXPECT_EQ(off.stats.intersect_time, absl::ZeroDuration());
  EXPECT_EQ(off.stats.worker_scan_ns, 0);
  // The always-on timers still run.
  EXPECT_GT(off.stats.dictionary_time, absl::ZeroDuration());
  EXPECT_GT(off.stats.compile_time, off.stats.dictionary_time);

  SearchExecutor executor(3);
  for (SearchExecutor* threads :
       {static_cast<SearchExecutor*>(nullptr), &executor}) {
    SearcherOptions options;
    options.stage_timing = true;
    options.executor = threads;
    options.parallel_threshold_bytes = 0;
    options.grain_chunks = 4;
    options.grain_bytes = 1;
    FERRET_ASSERT_OK_AND_ASSIGN(auto on,
                                SearchSegment(*segment, request, options));
    ExpectSameFiles(on, off);
    EXPECT_GT(on.stats.decode_time + on.stats.intersect_time,
              absl::ZeroDuration());
    EXPECT_LE(on.stats.decode_time + on.stats.intersect_time,
              on.stats.evaluate_time);
    EXPECT_GT(on.stats.verify_time, absl::ZeroDuration());
    EXPECT_GE(on.stats.rank_time, absl::ZeroDuration());
    EXPECT_GE(on.stats.count_sample_time, absl::ZeroDuration());
    if (threads != nullptr) {
      EXPECT_GT(on.stats.parallel_grains, 0);
      EXPECT_GT(on.stats.worker_scan_ns, 0);
    } else {
      EXPECT_EQ(on.stats.worker_scan_ns, 0);
    }
  }
  // A search that did not ask leaves this thread's clock as it found it.
  EXPECT_FALSE(ThisThreadStageClock().enabled);
}

TEST_F(SearcherTest, KeepsTheBestFilesAfterRanking) {
  // b.cc is verified after a.cc but outranks it with an identifier match.
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Segment> segment,
      BuildSegment(dir_, {{"a.cc", "needles();\n"},
                          {"d1/b.cc", "needle();\n"},
                          {"d1/d2/c.cc", "needles();\n"}}));
  SearchRequest request = Request("needle");
  request.max_files = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse response,
                              SearchSegment(*segment, request));
  EXPECT_THAT(Paths(response), ElementsAre("d1/b.cc"));
  // c.cc holds only a partial token: it is in no code-token list, so its
  // ceiling is the match-count boost and the search stops before it.
  EXPECT_TRUE(response.terminated_early);
  EXPECT_EQ(response.num_matching_files, 2);
  request.exact_counts = true;
  FERRET_ASSERT_OK_AND_ASSIGN(response, SearchSegment(*segment, request));
  EXPECT_EQ(response.num_matching_files, 3);
  request.exact_counts = false;
  request.max_files = 0;
  FERRET_ASSERT_OK_AND_ASSIGN(response, SearchSegment(*segment, request));
  EXPECT_THAT(response.files, IsEmpty());
  EXPECT_EQ(response.num_matching_lines, 3);
}

TEST_F(SearcherTest, NonKeptMatchInContextIsContext) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Segment> segment,
      BuildSegment(dir_, {{"a.cc", NumberedLines(3, {1, 2})}}));
  SearchRequest request = Request("needle");
  request.max_lines_per_file = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse response,
                              SearchSegment(*segment, request));
  ASSERT_THAT(response.files, SizeIs(1));
  EXPECT_THAT(LineNumbers(response.files[0]), ElementsAre(1, 2, 3));
  EXPECT_THAT(MatchLineNumbers(response.files[0]), ElementsAre(1));
}

TEST_F(SearcherTest, FiltersByPath) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Segment> segment,
      BuildSegment(dir_, {{"a.cc", "needle\n"},
                          {"b.h", "needle\n"},
                          {"third_party/c.cc", "needle\n"},
                          {"x_test.cc", "needle\n"}}));
  auto paths = [&](absl::string_view glob) {
    SearchRequest request = Request("needle");
    request.path_glob = std::string(glob);
    absl::StatusOr<SearchResponse> response = SearchSegment(*segment, request);
    EXPECT_TRUE(response.ok()) << response.status();
    if (!response.ok()) return std::vector<std::string>();
    EXPECT_EQ(response->stats.candidate_chunks, std::ssize(response->files));
    std::vector<std::string> sorted = Paths(*response);
    std::sort(sorted.begin(), sorted.end());
    return sorted;
  };
  EXPECT_THAT(paths("*.cc"),
              ElementsAre("a.cc", "third_party/c.cc", "x_test.cc"));
  EXPECT_THAT(paths("!third_party"), ElementsAre("a.cc", "b.h", "x_test.cc"));
  EXPECT_THAT(paths("third_party/*"), ElementsAre("third_party/c.cc"));
  EXPECT_THAT(paths("*.py"), IsEmpty());
  SearchRequest request = Request("needle");
  request.path_glob = "!";
  EXPECT_THAT(SearchSegment(*segment, request),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(SearcherTest, BudgetStopsBetweenFiles) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Segment> segment,
      BuildSegment(dir_,
                   {{"a.cc", NumberedLines(4, {1, 2, 3, 4})},
                    {"d1/b.cc", NumberedLines(1, {1})},
                    {"d1/d2/c.cc", NumberedLines(1, {1})}},
                   /*chunk_bytes=*/1));
  SearchRequest request = Request("needle");
  request.max_verify_bytes = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse response,
                              SearchSegment(*segment, request));
  // a.cc is verified completely, across its four chunks, and nothing else.
  EXPECT_FALSE(response.complete);
  ASSERT_THAT(Paths(response), ElementsAre("a.cc"));
  EXPECT_EQ(response.files[0].num_matching_lines, 4);
  EXPECT_EQ(response.num_matching_files, 1);
  EXPECT_EQ(response.stats.candidate_chunks, 6);
  EXPECT_EQ(response.stats.verified_chunks, 4);
  EXPECT_DOUBLE_EQ(response.verified_fraction, 4.0 / 6.0);
}

TEST_F(SearcherTest, NoCandidates) {
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Segment> segment,
                              BuildSegment(dir_, {{"a.cc", "needle\n"}}));
  SearchRequest impossible = Request("[^\\x00-\\x{10FFFF}]");
  impossible.regex = true;
  for (const SearchRequest& request : {Request("haystack"), impossible}) {
    FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse response,
                                SearchSegment(*segment, request));
    EXPECT_THAT(response.files, IsEmpty());
    EXPECT_EQ(response.num_matching_files, 0);
    EXPECT_EQ(response.stats.candidate_chunks, 0);
    EXPECT_TRUE(response.complete);
  }
}

TEST_F(SearcherTest, RejectsInvalidRequests) {
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Segment> segment,
                              BuildSegment(dir_, {{"a.cc", "needle\n"}}));
  std::vector<SearchRequest> requests(9, Request("needle"));
  requests[0].pattern = "";
  requests[1].pattern = "(";
  requests[1].regex = true;
  requests[2].context_lines = -1;
  requests[3].context_lines = kMaxContextLines + 1;
  requests[4].max_lines_per_file = 0;
  requests[5].max_lines = -1;
  requests[6].max_verify_bytes = 0;
  requests[7].pattern = "a\nb";
  requests[8].max_files = -1;
  for (const SearchRequest& request : requests) {
    EXPECT_THAT(SearchSegment(*segment, request),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << absl::CEscape(request.pattern);
  }
}

TEST_F(SearcherTest, EvaluatesEveryWindow) {
  // About 30,000 one-line chunks: more than the first three windows.
  std::vector<CorpusFile> corpus;
  corpus.reserve(2500);
  for (int i = 0; i < 2500; ++i) {
    corpus.push_back(
        {absl::StrCat("f", i, ".cc"),
         NumberedLines(12,
                       i % 1000 == 999 ? std::vector{7} : std::vector<int>{},
                       absl::StrCat(" needle", i))});
  }
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Segment> segment,
                              BuildSegment(dir_, corpus, /*chunk_bytes=*/1));
  ASSERT_GT(segment->num_chunks(), uint32_t{8192 + 16384 + 1});
  SearchRequest request = Request("needle");
  request.context_lines = 0;
  FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse response,
                              SearchSegment(*segment, request));
  std::vector<std::string> paths = Paths(response);
  std::sort(paths.begin(), paths.end());
  EXPECT_THAT(paths, ElementsAre("f1999.cc", "f999.cc"));
  EXPECT_EQ(response.stats.candidate_chunks, 2);
}

// Returns `files` sorted by path, with the fields the oracle does not set
// cleared.
std::vector<FileResult> Normalized(std::vector<FileResult> files) {
  for (FileResult& file : files) {
    file.oid = Oid();
    file.tier = RankTier::kSource;
    file.prior = 0;
    file.name_matches = false;
    file.score = 0;
    for (ResultLine& line : file.lines) line.match_class = {};
  }
  std::sort(
      files.begin(), files.end(),
      [](const FileResult& a, const FileResult& b) { return a.path < b.path; });
  return files;
}

std::string Describe(const std::vector<FileResult>& files) {
  std::string out;
  for (const FileResult& file : files) {
    absl::StrAppend(&out, file.path, " (", file.num_matching_lines, ")\n");
    for (const ResultLine& line : file.lines) {
      absl::StrAppend(&out, "  ", line.number, line.is_match ? ":" : "-",
                      absl::CEscape(line.text), " [", line.match_begin, ",",
                      line.match_end, ")\n");
    }
  }
  return out;
}

// Seeded corpora where most basenames contain a match, at every rank
// position and in every tier: the early-terminated top K equals the
// exhaustive top K, which equals the brute-force definition. Patterns cover
// literals and regexps with and without a required literal, case folding
// (including U+212A KELVIN SIGN), globs and line caps.
TEST_F(SearcherTest, SeededNameHeavyTopKMatchesExhaustive) {
  const uint64_t seed = ferret_testing::TestSeed(771);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  constexpr const char* kDirs[] = {
      "",      "src/",  "src/a/",         "src/a/b/c/", "src/a/b/c/d/e/f/",
      "test/", "docs/", "third_party/x/", "gen/"};
  constexpr const char* kNames[] = {"needle",     "Needle",      "NEEDLE",
                                    "a_needle_b", "other",       "needl",
                                    "kelvin",     "\u212Aelvin", "plain"};
  constexpr const char* kExtensions[] = {".cc", ".h", "_test.cc", ".md",
                                         ".pb.cc"};
  constexpr const char* kLines[] = {"int needle = 1;", "// needle comment",
                                    "needle_value();", "s = \"Needle\";",
                                    "other(NEEDLE);",  "kelvin = 1;",
                                    "x = 1;",          "int other = 2;"};
  const std::vector<SearchRequest> patterns = {
      {.pattern = "needle", .case_sensitive = true},
      {.pattern = "needle"},
      {.pattern = "NEEDLE"},
      {.pattern = "[Nn]eedle", .regex = true, .case_sensitive = true},
      {.pattern = "need(le|LE)", .regex = true},
      {.pattern = "(needle|other)", .regex = true},
      {.pattern = "^needle", .regex = true},
      {.pattern = "kelvin"},
      {.pattern = "^int", .regex = true},
      // Alternations: the priority scan searches each branch's required
      // string, or only the candidate documents' names. U+212A matches the
      // case-insensitive k, so "kelvin" is searched as "elvin".
      {.pattern = "(?:needle|kelvin)", .regex = true},
      {.pattern = "kelvin|plain", .regex = true, .case_sensitive = false},
      {.pattern = "needle_[a-z]|other", .regex = true, .case_sensitive = true},
      {.pattern = "(?i)(NEEDL|Plain)", .regex = true},
      {.pattern = "x[a-z]y|q.*r", .regex = true}};
  constexpr const char* kGlobs[] = {"", "", "src", "!test", "*.cc"};
  constexpr int64_t kUnlimited = int64_t{1} << 40;
  int64_t stops_with_priority = 0;
  int64_t stops_without_priority = 0;
  for (int corpus_index = 0; corpus_index < 4; ++corpus_index) {
    SCOPED_TRACE(corpus_index);
    std::vector<CorpusFile> corpus;
    const int num_files = 60 + static_cast<int>(rng() % 60);
    for (int i = 0; i < num_files; ++i) {
      std::string content;
      for (int line = static_cast<int>(rng() % 20); line >= 0; --line) {
        absl::StrAppend(&content, kLines[rng() % std::size(kLines)], "\n");
      }
      // Sizes span several size classes, so priors interleave by depth.
      absl::StrAppend(&content,
                      std::string(rng() % 3 == 0 ? rng() % 20000 : 0, 'x'),
                      "\nend ", i, "\n");
      corpus.push_back(
          {absl::StrCat(kDirs[rng() % std::size(kDirs)],
                        kNames[rng() % std::size(kNames)], i,
                        kExtensions[rng() % std::size(kExtensions)]),
           std::move(content)});
    }
    // The basename is priority for ^needle, but its candidate chunk has no
    // matching line because the literal is not at the beginning of the line.
    corpus.push_back({"a/b/c/needle_unmatched.cc", "// needle\n"});
    const int64_t chunk_bytes = std::vector<int64_t>{64, 1024, 8192}[rng() % 3];
    // The last corpus is an old segment: no code-token lists and no
    // short-line trigrams.
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto segment,
        BuildSegment(dir_, corpus, chunk_bytes,
                     /*write_code_tokens=*/corpus_index != 3, "test.fseg",
                     /*write_short_line_trigrams=*/corpus_index != 3));
    for (const SearchRequest& pattern : patterns) {
      for (const int64_t k : {int64_t{1}, int64_t{2}, int64_t{3}, int64_t{5},
                              int64_t{20}, kUnlimited}) {
        SearchRequest request = pattern;
        request.max_files = k;
        request.path_glob = kGlobs[rng() % std::size(kGlobs)];
        request.context_lines = static_cast<int>(rng() % 3);
        request.max_lines_per_file = rng() % 2 == 0 ? 1 : 100;
        request.max_lines = rng() % 3 == 0 ? 4 : kUnlimited;
        SCOPED_TRACE(absl::StrCat(
            request.pattern, " regex ", request.regex, " case ",
            request.case_sensitive, " k ", k, " glob ", request.path_glob,
            " lines ", request.max_lines_per_file, "/", request.max_lines));
        FERRET_ASSERT_OK_AND_ASSIGN(auto early,
                                    SearchSegment(*segment, request));
        request.exact_counts = true;
        FERRET_ASSERT_OK_AND_ASSIGN(auto exact,
                                    SearchSegment(*segment, request));
        ASSERT_TRUE(early.complete);
        ExpectSameFiles(early, exact);
        // The priority scan may limit itself to the candidate documents, or
        // never: the files are the same, and the restricted set only stops
        // sooner.
        request.exact_counts = false;
        FERRET_ASSERT_OK_AND_ASSIGN(
            auto unrestricted,
            SearchSegment(
                *segment, request,
                {.priority_restriction = PriorityRestriction::kNever}));
        FERRET_ASSERT_OK_AND_ASSIGN(
            auto restricted, SearchSegment(*segment, request,
                                           {.priority_restriction =
                                                PriorityRestriction::kAlways}));
        ExpectSameFiles(unrestricted, exact);
        ExpectSameFiles(restricted, exact);
        EXPECT_TRUE(restricted.complete);
        EXPECT_LE(restricted.stats.verified_bytes,
                  unrestricted.stats.verified_bytes);
        if (restricted.stats.priority_documents > 0) {
          EXPECT_EQ(restricted.stats.priority_restricted_sources, 1);
        }
        request.exact_counts = true;
        if (!early.terminated_early) {
          EXPECT_EQ(early.num_matching_files, exact.num_matching_files);
          EXPECT_EQ(early.num_matching_lines, exact.num_matching_lines);
        } else if (early.stats.priority_documents > 0) {
          ++stops_with_priority;
        } else {
          ++stops_without_priority;
        }
        // The exhaustive verifier is also the oracle for a byte-limited
        // prefix, where a rank stop must not evade the budget.
        if (corpus_index % 2 == 0 && k != kUnlimited) {
          const int64_t original_budget = request.max_verify_bytes;
          request.max_verify_bytes = 512;
          request.exact_counts = false;
          FERRET_ASSERT_OK_AND_ASSIGN(auto limited,
                                      SearchSegment(*segment, request));
          request.exact_counts = true;
          FERRET_ASSERT_OK_AND_ASSIGN(auto limited_exact,
                                      SearchSegment(*segment, request));
          ExpectSameFiles(limited, limited_exact);
          request.max_verify_bytes = original_budget;
        }
        // The oracle ignores the file and total line limits.
        if (k == kUnlimited && request.max_lines == kUnlimited) {
          FERRET_ASSERT_OK_AND_ASSIGN(
              const std::vector<FileResult> expected,
              ferret_testing::BruteForceSearch(corpus, request));
          EXPECT_EQ(Describe(Normalized(exact.files)),
                    Describe(Normalized(expected)));
        }
      }
    }
  }
  EXPECT_GT(stops_with_priority, 20);
  EXPECT_GT(stops_without_priority, 5);
}

// A root file with an identifier match and files one level down that only
// mention the word in comments, strings and longer identifiers. Each has a
// prior 4,096 below the root, so the previous ceiling (name excluded, 12,288)
// ties the root's score of prior + 8,192 and must verify them all. The
// code-token ceiling is the match-count boost alone, 4,096, and stops at the
// first.
TEST_F(SearcherTest, CodeTokenBoundStopsBeforeCommentsAndPartialTokens) {
  std::vector<CorpusFile> corpus = {{"z.cc", "int needle = 0;\n"}};
  for (int i = 0; i < 20; ++i) {
    const char* line[] = {"// needle ", "const char* s = \"needle ",
                          "int needles", "/* needle "};
    corpus.push_back(
        {absl::StrCat("d/f", i, ".cc"), absl::StrCat(line[i % 4], i, "\n")});
  }
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment, BuildSegment(dir_, corpus));
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto legacy, BuildSegment(dir_, corpus, 8192, /*write_code_tokens=*/false,
                                "legacy.fseg"));
  ASSERT_TRUE(segment->has_code_tokens());
  ASSERT_FALSE(legacy->has_code_tokens());
  SearchRequest request = Request("needle");
  request.max_files = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(auto early, SearchSegment(*segment, request));
  FERRET_ASSERT_OK_AND_ASSIGN(auto old, SearchSegment(*legacy, request));
  request.exact_counts = true;
  FERRET_ASSERT_OK_AND_ASSIGN(auto exact, SearchSegment(*segment, request));
  ASSERT_THAT(Paths(early), ElementsAre("z.cc"));
  ExpectSameFiles(early, exact);
  ExpectSameFiles(old, exact);
  EXPECT_TRUE(early.terminated_early);
  EXPECT_EQ(early.stats.verified_chunks, 1);
  // Only the root's list entry counts.
  EXPECT_EQ(early.stats.code_token_documents, 1);
  // A segment without the section keeps the previous bound.
  EXPECT_FALSE(old.terminated_early);
  EXPECT_EQ(old.stats.verified_chunks, 21);
  EXPECT_EQ(old.stats.code_token_documents, 0);
  // Every other query shape keeps it too, with or without the section: a
  // regexp, a literal that is not one ASCII identifier, and a non-ASCII one.
  for (const SearchRequest& other :
       {SearchRequest{.pattern = "needle", .regex = true},
        SearchRequest{
            .pattern = "needle", .regex = true, .case_sensitive = true},
        SearchRequest{.pattern = "needle "},
        SearchRequest{.pattern = "needle("},
        SearchRequest{.pattern = "1needle"}}) {
    SCOPED_TRACE(other.pattern);
    SearchRequest ineligible = other;
    ineligible.max_files = 1;
    EXPECT_FALSE(CodeTokenBoundKey(ineligible).has_value() && ineligible.regex);
    FERRET_ASSERT_OK_AND_ASSIGN(auto with, SearchSegment(*segment, ineligible));
    FERRET_ASSERT_OK_AND_ASSIGN(auto without,
                                SearchSegment(*legacy, ineligible));
    ExpectSameFiles(with, without);
    EXPECT_EQ(with.stats.verified_chunks, without.stats.verified_chunks);
    EXPECT_EQ(with.stats.code_token_documents, 0);
  }
}

TEST_F(SearcherTest, CodeTokenBoundKeyIsTheFoldedAsciiIdentifier) {
  EXPECT_EQ(CodeTokenBoundKey(Request("FooBar_9")), "foobar_9");
  EXPECT_EQ(CodeTokenBoundKey(Request("_x")), "_x");
  EXPECT_FALSE(CodeTokenBoundKey(Request("")).has_value());
  EXPECT_FALSE(CodeTokenBoundKey(Request("9x")).has_value());
  EXPECT_FALSE(CodeTokenBoundKey(Request("a b")).has_value());
  EXPECT_FALSE(CodeTokenBoundKey(Request("a.b")).has_value());
  // Case-insensitive RE2 matches É for é, which folding does not equate.
  EXPECT_FALSE(CodeTokenBoundKey(Request("caf\xC3\xA9")).has_value());
  SearchRequest regex = Request("foo");
  regex.regex = true;
  EXPECT_FALSE(CodeTokenBoundKey(regex).has_value());
}

// Seeded corpora full of partial-token, comment, string and case-variant
// matches, in several languages and chunk sizes. Every eligible query returns
// the same files with the code-token section, without it, and exhaustively;
// the section never verifies more chunks than the old bound, and often fewer.
TEST_F(SearcherTest, SeededCodeTokenTopKMatchesExhaustive) {
  const uint64_t seed = ferret_testing::TestSeed(20261006);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  constexpr const char* kDirs[] = {"",         "src/",         "src/a/",
                                   "src/a/b/", "src/a/b/c/d/", "test/",
                                   "docs/",    "gen/"};
  constexpr const char* kNames[] = {"flow",  "Flow",   "other",      "a_flow_b",
                                    "plain", "kelvin", "\u212Aelvin"};
  constexpr const char* kExtensions[] = {".cc", ".h",  ".py", ".txt",
                                         ".sh", ".go", ".rs", ".js"};
  constexpr const char* kLines[] = {"int flow = 1;",   "int overflow = 2;",
                                    "// flow control", "s = \"flow\";",
                                    "/* flow",         "still flow */",
                                    "flow_x(y);",      "Flow();",
                                    "FLOW;",           "x = flowing;",
                                    "# flow",          "other();",
                                    "flowchart",       "a.flow();",
                                    "/* start",        "end */",
                                    "kelvin = 1;",     "\u212Aelvin();",
                                    "\"\"\"",          "R\"x(flow)x\";",
                                    "`flow`",          "x = 1;"};
  // Lines that never hold `flow` as a whole identifier in code: comments,
  // strings and longer identifiers. Most files use only these, so most have
  // no identifier match and only the code-token lists can tell.
  constexpr const char* kQuietLines[] = {
      "int overflow = 2;", "// flow control", "s = \"flow\";", "# flow",
      "flow_x(y);",        "x = flowing;",    "flowchart",     "x = 1;",
      "other();"};
  const std::vector<SearchRequest> patterns = {
      {.pattern = "flow"},
      {.pattern = "flow", .case_sensitive = true},
      {.pattern = "Flow", .case_sensitive = true},
      {.pattern = "FLOW"},
      {.pattern = "overflow"},
      {.pattern = "flow_x"},
      {.pattern = "flowing"},
      {.pattern = "other"},
      {.pattern = "kelvin"},
      {.pattern = "kelvin", .case_sensitive = true},
      {.pattern = "o"},
      {.pattern = "x"},
      {.pattern = "start"}};
  constexpr const char* kGlobs[] = {"", "", "src", "!test", "*.cc"};
  constexpr int64_t kUnlimited = int64_t{1} << 40;
  int64_t stops_with_tokens = 0;
  int64_t stops_without_tokens = 0;
  int64_t searches = 0;
  int64_t fewer_chunks = 0;
  for (int corpus_index = 0; corpus_index < 6; ++corpus_index) {
    SCOPED_TRACE(corpus_index);
    std::vector<CorpusFile> corpus;
    const int num_files = 60 + static_cast<int>(rng() % 60);
    for (int i = 0; i < num_files; ++i) {
      std::string content;
      const bool quiet = rng() % 10 != 0;
      for (int line = static_cast<int>(rng() % 20); line >= 0; --line) {
        absl::StrAppend(&content,
                        quiet ? kQuietLines[rng() % std::size(kQuietLines)]
                              : kLines[rng() % std::size(kLines)],
                        "\n");
      }
      absl::StrAppend(&content,
                      std::string(rng() % 3 == 0 ? rng() % 20000 : 0, 'x'),
                      "\nend ", i, "\n");
      corpus.push_back(
          {absl::StrCat(kDirs[rng() % std::size(kDirs)],
                        // Few names match, most are "other" or "plain".
                        kNames[rng() % 4 == 0 ? rng() % std::size(kNames)
                                              : 2 + rng() % 2],
                        i, kExtensions[rng() % std::size(kExtensions)]),
           std::move(content)});
    }
    const int64_t chunk_bytes = std::vector<int64_t>{64, 1024, 8192}[rng() % 3];
    FERRET_ASSERT_OK_AND_ASSIGN(auto segment,
                                BuildSegment(dir_, corpus, chunk_bytes));
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto legacy,
        BuildSegment(dir_, corpus, chunk_bytes, false, "legacy.fseg"));
    for (const SearchRequest& pattern : patterns) {
      for (const int64_t k : {int64_t{1}, int64_t{2}, int64_t{3}, int64_t{5},
                              int64_t{20}, kUnlimited}) {
        SearchRequest request = pattern;
        request.max_files = k;
        request.path_glob = kGlobs[rng() % std::size(kGlobs)];
        request.context_lines = static_cast<int>(rng() % 3);
        request.max_lines_per_file = rng() % 2 == 0 ? 1 : 100;
        request.max_lines = rng() % 3 == 0 ? 4 : kUnlimited;
        SCOPED_TRACE(
            absl::StrCat(request.pattern, " case ", request.case_sensitive,
                         " k ", k, " glob ", request.path_glob, " lines ",
                         request.max_lines_per_file, "/", request.max_lines));
        FERRET_ASSERT_OK_AND_ASSIGN(auto with,
                                    SearchSegment(*segment, request));
        FERRET_ASSERT_OK_AND_ASSIGN(auto without,
                                    SearchSegment(*legacy, request));
        request.exact_counts = true;
        FERRET_ASSERT_OK_AND_ASSIGN(auto exact,
                                    SearchSegment(*segment, request));
        ASSERT_TRUE(with.complete);
        ExpectSameFiles(with, exact);
        ExpectSameFiles(without, exact);
        EXPECT_LE(with.stats.verified_chunks, without.stats.verified_chunks);
        EXPECT_LE(with.stats.verified_bytes, without.stats.verified_bytes);
        EXPECT_TRUE(!without.terminated_early || with.terminated_early);
        if (!with.terminated_early) {
          EXPECT_EQ(with.num_matching_files, exact.num_matching_files);
          EXPECT_EQ(with.num_matching_lines, exact.num_matching_lines);
        }
        fewer_chunks +=
            with.stats.verified_chunks < without.stats.verified_chunks;
        ++searches;
        stops_with_tokens += with.terminated_early;
        stops_without_tokens += without.terminated_early;
      }
    }
  }
  EXPECT_GT(searches, 300);
  EXPECT_GT(stops_without_tokens, 20);
  EXPECT_GE(stops_with_tokens, stops_without_tokens);
  EXPECT_GT(fewer_chunks, 20);
}

// Counting-only verification must change nothing a caller can see: the same
// files and lines, counts, completeness, early stop and verified work as a run
// that classifies and keeps every document, under tight line limits, budgets,
// globs and a segment without code-token lists. Some documents match on many
// lines so that the line allowance binds while counted documents still
// consume it.
TEST_F(SearcherTest, CountOnlyVerificationLeavesTheResponseUnchanged) {
  const uint64_t seed = ferret_testing::TestSeed(20261007);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  constexpr const char* kDirs[] = {"", "src/", "src/a/b/", "test/", "gen/"};
  constexpr const char* kNames[] = {"flow", "Flow", "other", "a_flow_b",
                                    "plain"};
  constexpr const char* kExtensions[] = {".cc", ".h", ".py", ".txt", ".js"};
  constexpr const char* kLines[] = {
      "int flow = 1;", "int overflow = 2;", "// flow control", "s = \"flow\";",
      "/* flow",       "still flow */",     "flow_x(y);",      "Flow();",
      "FLOW;",         "x = flowing;",      "# flow",          "other();",
      "x = 1;",        "a.flow();",         "flow flow flow;"};
  const std::vector<SearchRequest> patterns = {
      {.pattern = "flow"},
      {.pattern = "flow", .case_sensitive = true},
      {.pattern = "Flow", .case_sensitive = true},
      {.pattern = "overflow"},
      {.pattern = "other"},
      {.pattern = "fl[o]w", .regex = true},
      {.pattern = "x"}};
  constexpr const char* kGlobs[] = {"", "", "src", "!test", "*.cc"};
  constexpr int64_t kUnlimited = int64_t{1} << 40;
  const int64_t options_sample_size = SearcherOptions().count_sample_size;
  const SearcherOptions off{.count_only_excluded = false};
  const SearcherOptions on{.count_only_min_bytes = 0};
  int64_t searches = 0;
  int64_t counted_searches = 0;
  int64_t counted_documents = 0;
  int64_t counted_with_binding_cap = 0;
  for (int corpus_index = 0; corpus_index < 4; ++corpus_index) {
    SCOPED_TRACE(corpus_index);
    std::vector<CorpusFile> corpus;
    const int num_files = 80 + static_cast<int>(rng() % 60);
    for (int i = 0; i < num_files; ++i) {
      std::string content;
      const int lines = 1 + static_cast<int>(rng() % (i % 7 == 0 ? 90 : 12));
      for (int line = 0; line < lines; ++line) {
        absl::StrAppend(&content, kLines[rng() % std::size(kLines)], "\n");
      }
      absl::StrAppend(&content, "end ", i, "\n");
      corpus.push_back(
          {absl::StrCat(kDirs[rng() % std::size(kDirs)],
                        kNames[rng() % std::size(kNames)], i,
                        kExtensions[rng() % std::size(kExtensions)]),
           std::move(content)});
    }
    const int64_t chunk_bytes = std::vector<int64_t>{48, 512, 8192}[rng() % 3];
    FERRET_ASSERT_OK_AND_ASSIGN(auto segment,
                                BuildSegment(dir_, corpus, chunk_bytes));
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto legacy,
        BuildSegment(dir_, corpus, chunk_bytes, false, "legacy.fseg"));
    for (const SearchRequest& pattern : patterns) {
      for (const int64_t k :
           {int64_t{1}, int64_t{3}, int64_t{10}, kUnlimited}) {
        SearchRequest request = pattern;
        request.max_files = k;
        request.path_glob = kGlobs[rng() % std::size(kGlobs)];
        request.context_lines = static_cast<int>(rng() % 3);
        request.max_lines_per_file = rng() % 2 == 0 ? 2 : 1000;
        request.max_lines =
            rng() % 3 == 0 ? 7 : (rng() % 2 == 0 ? 60 : kUnlimited);
        request.max_verify_bytes = rng() % 4 == 0 ? 600 : kUnlimited;
        request.count_sample = rng() % 8 == 0;
        SCOPED_TRACE(absl::StrCat(
            request.pattern, " case ", request.case_sensitive, " k ", k,
            " glob ", request.path_glob, " lines ", request.max_lines_per_file,
            "/", request.max_lines, " budget ", request.max_verify_bytes));
        for (const Segment* source : {segment.get(), legacy.get()}) {
          FERRET_ASSERT_OK_AND_ASSIGN(auto want,
                                      SearchSegment(*source, request, off));
          FERRET_ASSERT_OK_AND_ASSIGN(auto got,
                                      SearchSegment(*source, request, on));
          ExpectSameFiles(got, want);
          EXPECT_EQ(got.complete, want.complete);
          EXPECT_EQ(got.terminated_early, want.terminated_early);
          EXPECT_EQ(got.num_matching_files, want.num_matching_files);
          EXPECT_EQ(got.num_matching_lines, want.num_matching_lines);
          // The classes of counted documents are exact unless they are
          // estimated from a sample of them (a count sample), which happens
          // only when more than a sample's worth were counted.
          if (!request.count_sample ||
              got.stats.counted_sample_documents ==
                  got.stats.count_only_documents ||
              got.stats.count_only_documents == 0) {
            EXPECT_EQ(got.match_classes, want.match_classes);
          }
          EXPECT_EQ(got.counts_estimated, want.counts_estimated);
          EXPECT_EQ(got.count_sample_size, want.count_sample_size);
          EXPECT_EQ(got.stats.candidate_chunks, want.stats.candidate_chunks);
          EXPECT_EQ(got.stats.verified_chunks, want.stats.verified_chunks);
          EXPECT_EQ(got.stats.verified_bytes, want.stats.verified_bytes);
          EXPECT_EQ(want.stats.count_only_documents, 0);
          EXPECT_LE(got.stats.count_only_bytes, got.stats.verified_bytes);
          if (request.count_sample && got.stats.count_only_documents > 0) {
            EXPECT_LE(got.stats.counted_sample_documents, options_sample_size);
          }
          ++searches;
          counted_searches += got.stats.count_only_documents > 0;
          counted_documents += got.stats.count_only_documents;
          counted_with_binding_cap += got.stats.count_only_documents > 0 &&
                                      request.max_lines < kUnlimited;
        }
      }
    }
  }
  EXPECT_GT(searches, 200);
  EXPECT_GT(counted_searches, 40);
  EXPECT_GT(counted_documents, 300);
  EXPECT_GT(counted_with_binding_cap, 10);
}

// A line for a corpus searched with `pattern`: mostly strings it matches.
std::string RandomLine(const CompiledPattern& pattern, std::mt19937_64& rng) {
  const uint64_t roll = rng() % 100;
  if (roll < 50) {
    std::optional<std::string> sample =
        ferret_testing::SampleMatch(pattern.matcher().Regexp(), rng);
    if (sample.has_value()) {
      return roll < 35 ? ferret_testing::Embed(*sample, rng)
                       : ferret_testing::Mutate(*std::move(sample), rng);
    }
  }
  return ferret_testing::RandomText(rng);
}

// The whole pipeline, from pattern to ranked files, against the definition
// on random corpora and patterns: any unsound plan, evaluator, chunking,
// verifier, line numbering, or context bug shows up as a difference.
TEST_F(SearcherTest, RandomCorporaMatchBruteForce) {
  std::mt19937_64 rng(ferret_testing::TestSeed(47));
  constexpr const char* kDirs[] = {"", "src/", "src/deep/er/", "third_party/",
                                   "test/"};
  constexpr const char* kExtensions[] = {".cc", ".h", ".py", "_test.cc",
                                         ".txt"};
  constexpr const char* kGlobs[] = {"",      "", "", "*.cc", "!third_party",
                                    "src/**"};
  int64_t corpora = 0;
  int64_t matching_lines = 0;
  int64_t multi_chunk_matches = 0;
  while (corpora < 150) {
    SearchRequest request{.pattern = ferret_testing::RandomPattern(rng),
                          .regex = true,
                          .case_sensitive = rng() % 2 == 0};
    absl::StatusOr<std::unique_ptr<CompiledPattern>> pattern = CompilePattern(
        {.pattern = request.pattern, .case_sensitive = request.case_sensitive});
    if (!pattern.ok()) continue;
    ++corpora;

    std::vector<CorpusFile> corpus;
    const int num_files = 1 + static_cast<int>(rng() % 12);
    for (int f = 0; f < num_files; ++f) {
      std::string content;
      const int num_lines = static_cast<int>(rng() % 30);
      for (int l = 0; l < num_lines; ++l) {
        absl::StrAppend(&content, RandomLine(**pattern, rng),
                        l + 1 < num_lines || rng() % 2 == 0 ? "\n" : "");
      }
      corpus.push_back(
          {absl::StrCat(kDirs[rng() % std::size(kDirs)], "f", f,
                        kExtensions[rng() % std::size(kExtensions)]),
           std::move(content)});
    }
    const int64_t chunk_bytes =
        std::vector<int64_t>{1, 16, 100, 8192}[rng() % 4];
    FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Segment> segment,
                                BuildSegment(dir_, corpus, chunk_bytes));

    // The pattern itself, then a literal taken from the corpus.
    std::vector<SearchRequest> requests = {request};
    const CorpusFile& source = corpus[rng() % corpus.size()];
    if (source.content.size() >= 3) {
      const size_t length = 3 + rng() % 4;
      const size_t begin = rng() % (source.content.size() - 2);
      std::string literal = source.content.substr(begin, length);
      literal = literal.substr(0, literal.find('\n'));
      if (!literal.empty()) {
        requests.push_back(
            {.pattern = literal, .case_sensitive = rng() % 2 == 0});
      }
    }
    for (SearchRequest& r : requests) {
      r.path_glob = kGlobs[rng() % std::size(kGlobs)];
      r.context_lines = static_cast<int>(rng() % 4);
      r.max_lines_per_file =
          rng() % 3 == 0 ? 1 + static_cast<int>(rng() % 3) : 100;
      r.max_lines = int64_t{1} << 40;
      r.max_files = int64_t{1} << 40;
      const absl::StatusOr<std::vector<FileResult>> expected =
          ferret_testing::BruteForceSearch(corpus, r);
      const absl::StatusOr<SearchResponse> actual = SearchSegment(*segment, r);
      ASSERT_EQ(actual.ok(), expected.ok())
          << actual.status() << " vs " << expected.status();
      if (!actual.ok()) continue;
      const std::vector<FileResult> want = Normalized(*expected);
      const std::vector<FileResult> got = Normalized(actual->files);
      ASSERT_EQ(Describe(got), Describe(want))
          << "pattern \"" << absl::CEscape(r.pattern) << "\" regex " << r.regex
          << " case_sensitive " << r.case_sensitive << " glob " << r.path_glob
          << " chunk_bytes " << chunk_bytes;
      int64_t lines = 0;
      for (const FileResult& file : want) lines += file.num_matching_lines;
      EXPECT_EQ(actual->num_matching_files, std::ssize(want));
      EXPECT_EQ(actual->num_matching_lines, lines);
      EXPECT_TRUE(actual->complete);
      EXPECT_TRUE(std::is_sorted(
          actual->files.begin(), actual->files.end(),
          [](const FileResult& a, const FileResult& b) {
            return std::forward_as_tuple(a.tier, -a.score, a.path) <
                   std::forward_as_tuple(b.tier, -b.score, b.path);
          }));
      matching_lines += lines;
      if (chunk_bytes < 100) multi_chunk_matches += lines;
    }
  }
  EXPECT_GT(matching_lines, 1000);
  EXPECT_GT(multi_chunk_matches, 300);
}

// Two folded bytes have no trigram, so such a pattern selects candidates from
// the posting lists of "abX", "Xab" and the pseudo-trigram "ab\n" of lines of
// exactly those two bytes. The pieces make lines that are the pattern alone,
// lines it borders on either side, and the fold orbits of 'k' and 's'; one-byte
// chunks put every line in its own chunk.
TEST_F(SearcherTest, TwoBytePatternsMatchBruteForce) {
  std::mt19937_64 rng(ferret_testing::TestSeed(53));
  const std::vector<std::string> pieces = {
      "ab", "AB", "Ab",       "};",           "})",       "k",
      "K",  "s",  "S",        "\xE2\x84\xAA", "\xC5\xBF", "x",
      " ",  "\t", "\xC3\xA9", "\xC3\x89",     "a",        "b",
      "kk", "sk", "\r",       "ab ab",        "ba",       "aB",
      "BA", "A;", "{X"};
  const std::vector<std::string> patterns = {
      "ab", "AB",  "};",   "})", "kk", "ks", "sk", "SK", "\xC3\xA9", "x ", "  ",
      "ba", "b\n", "a\nb", "\r", "k",  "Ab", "aB", "BA", "A;",       "{X"};
  int64_t narrowed = 0;
  int64_t found = 0;
  for (int round = 0; round < 60; ++round) {
    std::vector<CorpusFile> corpus;
    const int num_files = 1 + static_cast<int>(rng() % 8);
    for (int f = 0; f < num_files; ++f) {
      std::string content;
      const int num_lines = static_cast<int>(rng() % 25);
      for (int l = 0; l < num_lines; ++l) {
        const int num_pieces = static_cast<int>(rng() % 4);
        for (int p = 0; p < num_pieces; ++p) {
          absl::StrAppend(&content, pieces[rng() % pieces.size()]);
        }
        if (l + 1 < num_lines || rng() % 2 == 0) content.push_back('\n');
      }
      corpus.push_back(
          {absl::StrCat("d", rng() % 3, "/f", f, ".cc"), std::move(content)});
    }
    const int64_t chunk_bytes = std::vector<int64_t>{1, 5, 16, 8192}[rng() % 4];
    FERRET_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Segment> segment,
        BuildSegment(dir_, corpus, chunk_bytes, true, "new.fseg"));
    FERRET_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Segment> legacy,
        BuildSegment(dir_, corpus, chunk_bytes, true, "old.fseg",
                     /*write_short_line_trigrams=*/false));
    ASSERT_TRUE(segment->has_short_line_trigrams());
    ASSERT_FALSE(legacy->has_short_line_trigrams());
    for (const std::string& text : patterns) {
      for (const bool regex : {false, true}) {
        for (const bool case_sensitive : {false, true}) {
          SearchRequest request = Request(text);
          request.regex = regex;
          request.case_sensitive = case_sensitive;
          request.max_lines = int64_t{1} << 40;
          request.max_files = int64_t{1} << 40;
          request.max_lines_per_file = 100;
          const absl::StatusOr<std::vector<FileResult>> expected =
              ferret_testing::BruteForceSearch(corpus, request);
          SCOPED_TRACE(absl::StrCat("pattern \"", absl::CEscape(text),
                                    "\" regex ", regex, " case_sensitive ",
                                    case_sensitive, " chunk_bytes ",
                                    chunk_bytes));
          int64_t scan_candidates = -1;
          for (const double share : {0.0, 1e9, 0.25}) {
            for (const Segment* target : {legacy.get(), segment.get()}) {
              SearcherOptions options;
              options.max_bigram_posting_share = share;
              const absl::StatusOr<SearchResponse> actual =
                  SearchSegment(*target, request, options);
              ASSERT_EQ(actual.ok(), expected.ok())
                  << actual.status() << " vs " << expected.status();
              if (!actual.ok()) continue;
              ASSERT_EQ(Describe(Normalized(actual->files)),
                        Describe(Normalized(*expected)))
                  << "share " << share << " legacy "
                  << (target == legacy.get());
              EXPECT_TRUE(actual->complete);
              // A segment without the pseudo-trigrams always scans every chunk.
              if (share == 0.0 && target == legacy.get()) {
                scan_candidates = actual->stats.candidate_chunks;
                found += std::ssize(*expected);
              }
              if (target == legacy.get()) {
                EXPECT_EQ(actual->stats.candidate_chunks, scan_candidates);
              } else if (share == 1e9) {
                EXPECT_LE(actual->stats.candidate_chunks, scan_candidates);
                narrowed += actual->stats.candidate_chunks < scan_candidates;
              }
            }
          }
        }
      }
    }
  }
  EXPECT_GT(found, 500);
  // The union path really ran: it left out chunks the scan visited.
  EXPECT_GT(narrowed, 100);
}

}  // namespace
}  // namespace ferret

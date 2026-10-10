// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/base/thread_pool.h"
#include "ferret/git/blob_hash.h"
#include "ferret/git/oid.h"
#include "ferret/index/blob_registry.h"
#include "ferret/index/index.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/read_state.h"
#include "ferret/index/segment.h"
#include "ferret/index/segment_writer.h"
#include "ferret/index/version_set.h"
#include "ferret/postings/ids.h"
#include "ferret/postings/list_ops.h"
#include "ferret/regex/compile.h"
#include "ferret/search/search_types.h"
#include "ferret/search/searcher.h"
#include "ferret/testing/status_matchers.h"  // IWYU pragma: keep (test assertion macros).
#include "ferret/testing/temp_dir.h"
#include "ferret/testing/test_seed.h"
#include "ferret/text/language.h"
#include "ferret/workspace/path_table.h"
#include "ferret/workspace/view_state.h"

namespace ferret {
namespace {

// The files and completeness: what a response promises whatever the bounds
// let it skip.
void ExpectSameFiles(const SearchResponse& got, const SearchResponse& want) {
  EXPECT_EQ(got.complete, want.complete);
  ASSERT_EQ(got.files.size(), want.files.size());
  for (size_t i = 0; i < got.files.size(); ++i) {
    const FileResult& a = got.files[i];
    const FileResult& b = want.files[i];
    EXPECT_EQ(a.path, b.path);
    EXPECT_EQ(a.oid, b.oid);
    EXPECT_EQ(a.tier, b.tier);
    EXPECT_EQ(a.prior, b.prior);
    EXPECT_EQ(a.name_matches, b.name_matches);
    EXPECT_EQ(a.score, b.score);
    EXPECT_EQ(a.lines, b.lines);
  }
}

void ExpectSameResponse(const SearchResponse& got, const SearchResponse& want) {
  EXPECT_EQ(got.complete, want.complete);
  EXPECT_EQ(got.terminated_early, want.terminated_early);
  EXPECT_EQ(got.verified_fraction, want.verified_fraction);
  EXPECT_EQ(got.num_matching_files, want.num_matching_files);
  EXPECT_EQ(got.num_matching_lines, want.num_matching_lines);
  EXPECT_EQ(got.stats.candidate_chunks, want.stats.candidate_chunks);
  EXPECT_EQ(got.stats.verified_chunks, want.stats.verified_chunks);
  EXPECT_EQ(got.stats.verified_bytes, want.stats.verified_bytes);
  EXPECT_EQ(got.counts_estimated, want.counts_estimated);
  EXPECT_EQ(got.count_sample_size, want.count_sample_size);
  EXPECT_EQ(got.match_classes, want.match_classes);
  EXPECT_EQ(got.stats.priority_documents, want.stats.priority_documents);
  EXPECT_EQ(got.stats.excluded_scans, 0);
  ASSERT_EQ(got.files.size(), want.files.size());
  for (size_t i = 0; i < got.files.size(); ++i) {
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

TEST(SearchParallelTest, SeededSnapshotDifferentialAndConcurrentQueries) {
  const uint64_t seed = ferret_testing::TestSeed(0x99123);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  ferret_testing::TempDir dir;
  const char* const words[] = {"needle", "NEEDLE", "other", "needle_code"};
  for (int shape = 0; shape < 3; ++shape) {
    SCOPED_TRACE(shape);
    IndexOptions index_options;
    index_options.memtable.flush_docs = 1000;
    index_options.memtable.segment.chunk_bytes = 64;
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto opened,
        Index::Open(*Env::Default(), dir.Join(absl::StrCat("parallel", shape)),
                    index_options));
    std::unique_ptr<Index> index = std::move(opened.first);
    const int documents = shape == 1 ? 18 : 96;
    for (int i = 0; i < documents; ++i) {
      std::string content;
      const int lines = shape == 1 ? 1200 : 2 + static_cast<int>(rng() % 18);
      for (int j = 0; j < lines; ++j) {
        absl::StrAppend(&content, words[rng() % 4], " line ", j, "\n");
      }
      // Repeated Oids span both flushed segments and the live memtable.
      if (shape == 2 && i % 3 == 0) content = "needle shared\n";
      const std::string path =
          absl::StrCat(i % 4 == 0 ? "third_party/" : "src/", "file", i % 19,
                       i % 3 == 0 ? ".h" : ".cc");
      SegmentDocument doc{.hint_path = path,
                          .content = content,
                          .rank_key = ComputeRankKey(path, content.size(),
                                                     false, HashBlob(content)),
                          .language = DetectLanguage(path, content)};
      ASSERT_THAT(index->Add(doc), absl_testing::IsOk());
      if (i == documents / 3 || i == 2 * documents / 3) {
        ASSERT_THAT(index->Flush(), absl_testing::IsOk());
      }
    }
    const std::shared_ptr<const ReadState> state = index->GetReadState();
    const char* const patterns[] = {"needle", "NEEDLE", "other", "nee.*le"};
    const char* const globs[] = {"", "*.h", "src/**", "!third_party"};
    std::vector<SearchRequest> requests;
    for (int i = 0; i < 16; ++i) {
      SearchRequest request;
      request.pattern = i <= 3 ? "needle" : patterns[rng() % 4];
      request.regex = i > 3 && i % 4 == 0;
      request.case_sensitive = i > 3 && i % 3 == 0;
      request.path_glob = i <= 3 ? "" : globs[rng() % 4];
      request.context_lines = i % 3;
      request.max_lines = i % 3 == 0 ? 2 : std::numeric_limits<int64_t>::max();
      request.max_lines_per_file = i % 2 == 0 ? 1 : 100;
      request.max_files = i % 3 == 0 ? 1 : 100;
      request.max_verify_bytes = i == 0 ? 1 << 24
                                        : (i % 4 == 0   ? 128
                                           : i % 4 == 1 ? 1024
                                           : i % 4 == 2 ? 16 << 10
                                                        : 1 << 24);
      request.exact_counts = i <= 2 || i % 2 == 0;
      requests.push_back(std::move(request));
    }
    std::vector<SearchResponse> expected;
    // The expected responses classify and keep every verified document; every
    // other run counts the documents that cannot reach the top K.
    const SearcherOptions classify_all{.count_only_excluded = false};
    for (const SearchRequest& request : requests) {
      FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse response,
                                  Search(*state, request, classify_all));
      expected.push_back(std::move(response));
    }
    EXPECT_TRUE(expected[3].terminated_early);
    EXPECT_FALSE(expected[1].complete);
    if (shape == 1) EXPECT_FALSE(expected[2].complete);
    for (int threads : {1, 2, 3, 4, 8}) {
      SCOPED_TRACE(threads);
      SearchExecutor executor(threads);
      for (int mode : {0, 1, 2}) {
        SCOPED_TRACE(mode);
        const SearcherOptions options{
            .count_only_min_bytes = 0,
            .executor = &executor,
            .parallel_threshold_bytes = mode == 0 ? int64_t{2} << 20 : 0,
            .grain_chunks = mode == 2 ? 1 : 64,
            .grain_bytes = mode == 2 ? 1 : 256 << 10,
            .ring_slots_per_worker = mode == 2 ? 2 : 4};
        for (size_t i = 0; i < requests.size(); ++i) {
          SCOPED_TRACE(i);
          FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse response,
                                      Search(*state, requests[i], options));
          ExpectSameResponse(response, expected[i]);
          if (i == 0 && mode == 2 && threads > 1) {
            EXPECT_GT(response.stats.parallel_grains, 2 * threads);
          }
          if (threads == 1 || mode == 0) {
            EXPECT_EQ(response.stats.parallel_grains, 0);
          }
        }
        ThreadPool clients(3, ThreadQos::kUserInteractive);
        std::vector<SearchResponse> concurrent(4);
        clients.ParallelFor(4, 1, 4, [&](int, int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto result = Search(*state, requests[i], options);
            ASSERT_TRUE(result.ok()) << result.status();
            concurrent[i] = std::move(*result);
          }
        });
        for (size_t i = 0; i < concurrent.size(); ++i) {
          ExpectSameResponse(concurrent[i], expected[i]);
        }
      }
    }
    for (int knob = 0; knob < 4; ++knob) {
      SearcherOptions options;
      if (knob == 0) options.parallel_threshold_bytes = -1;
      if (knob == 1) options.grain_chunks = 0;
      if (knob == 2) options.grain_bytes = 0;
      if (knob == 3) options.ring_slots_per_worker = 0;
      EXPECT_EQ(Search(*state, requests[0], options).status().code(),
                absl::StatusCode::kInvalidArgument);
    }
  }
}

// Documents with a path under each tier, some named after the pattern (the
// name-priority documents), spread over segments and the memtable.
struct EarlyStopCorpus {
  std::unique_ptr<Index> index;
  std::vector<Oid> oids;
  std::vector<std::string> paths;
};

EarlyStopCorpus BuildEarlyStopCorpus(const ferret_testing::TempDir& dir,
                                     std::mt19937_64& rng, int documents,
                                     int repeat_stride) {
  IndexOptions index_options;
  index_options.memtable.flush_docs = 1000;
  index_options.memtable.segment.chunk_bytes = 96;
  auto opened = Index::Open(*Env::Default(), dir.Join("early"), index_options);
  EXPECT_TRUE(opened.ok());
  EarlyStopCorpus corpus;
  corpus.index = std::move(opened->first);
  const char* const roots[] = {"src/",  "src/",         "test/",
                               "docs/", "third_party/", "generated/"};
  const char* const names[] = {"needle_core", "util",  "needle",
                               "main",        "other", "my_needle_x"};
  for (int i = 0; i < documents; ++i) {
    std::string content;
    const int lines = 1 + static_cast<int>(rng() % 30);
    for (int j = 0; j < lines; ++j) {
      absl::StrAppend(&content, rng() % 3 == 0 ? "call needle(x);" : "other",
                      " // line ", j, "\n");
    }
    // Repeated content shares an Oid across sources and paths.
    if (i % repeat_stride == 0) content = "needle shared\n";
    const std::string path = absl::StrCat(roots[rng() % 6], names[rng() % 6],
                                          i % 23, i % 2 == 0 ? ".cc" : ".h");
    SegmentDocument doc{.hint_path = path,
                        .content = content,
                        .rank_key = ComputeRankKey(path, content.size(), false,
                                                   HashBlob(content)),
                        .language = DetectLanguage(path, content)};
    EXPECT_THAT(corpus.index->Add(doc), absl_testing::IsOk());
    corpus.oids.push_back(HashBlob(content));
    corpus.paths.push_back(path);
    if (i == documents / 3 || i == 2 * documents / 3) {
      EXPECT_THAT(corpus.index->Flush(), absl_testing::IsOk());
    }
  }
  return corpus;
}

TEST(SearchParallelTest, EarlyStopMatchesOneThreadAtEveryThreadCount) {
  const uint64_t seed = ferret_testing::TestSeed(0x51ab7);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  ferret_testing::TempDir dir;
  const EarlyStopCorpus corpus = BuildEarlyStopCorpus(dir, rng, 240, 7);
  const std::shared_ptr<const ReadState> state = corpus.index->GetReadState();

  // A view with one or two paths per blob; some paths are named after the
  // pattern, and some blobs are invisible.
  PathTable paths;
  ViewWriter writer;
  std::vector<ViewEdit> edits;
  for (size_t i = 0; i < corpus.oids.size(); ++i) {
    if (i % 11 == 0) continue;
    const BlobId blob = corpus.index->blobs().Intern(corpus.oids[i]);
    auto one = paths.Intern(corpus.paths[i]);
    ASSERT_TRUE(one.ok());
    edits.push_back(ViewEdit{*one, blob});
    if (i % 5 == 0) {
      auto two = paths.Intern(absl::StrCat("alias/needle_", i, ".txt"));
      ASSERT_TRUE(two.ok());
      edits.push_back(ViewEdit{*two, blob});
    }
  }
  writer.ApplyBatch(edits);
  const SearchView view{writer.Snapshot(), paths};

  std::vector<SearchRequest> requests;
  for (int i = 0; i < 24; ++i) {
    SearchRequest request;
    request.pattern = i % 4 == 3 ? "nee.le" : "needle";
    request.regex = i % 4 == 3;
    // Alternations have no single required string: the priority scan
    // searches each branch's, or the candidates' names.
    if (i >= 16) {
      constexpr const char* kAlternations[] = {
          "(?:needle_core|my_needle_x)", "(?i)(UTIL|main)", "util|main|other",
          "needle_(core|x)|other"};
      request.pattern = kAlternations[i % 4];
      request.regex = true;
    }
    request.max_files = i % 3 == 0 ? 1 : (i % 3 == 1 ? 5 : 20);
    request.max_lines_per_file = i % 2 == 0 ? 1 : 100;
    request.context_lines = i % 2;
    request.count_sample = i % 2 == 1;
    request.path_glob = i % 5 == 4 ? "src/**" : "";
    // Tight budgets: the second group cuts segment verification.
    request.max_verify_bytes =
        i >= 12 && i < 16 ? 300 + 200 * i : int64_t{1} << 24;
    requests.push_back(std::move(request));
  }
  int early_stops = 0;
  int64_t counted_documents = 0;
  const PriorityRestriction modes[] = {PriorityRestriction::kAuto,
                                       PriorityRestriction::kNever,
                                       PriorityRestriction::kAlways};
  for (const bool scoped : {false, true}) {
    SCOPED_TRACE(scoped);
    // Per restriction, the one-thread responses that classify every document.
    // The restriction never changes the files, only how soon the bound stops.
    std::vector<std::vector<SearchResponse>> expected_by_mode;
    for (const PriorityRestriction mode : modes) {
      const SearcherOptions classify_all{.count_only_excluded = false,
                                         .priority_restriction = mode};
      std::vector<SearchResponse>& expected = expected_by_mode.emplace_back();
      for (const SearchRequest& request : requests) {
        auto response = scoped ? Search(*state, view, request, classify_all)
                               : Search(*state, request, classify_all);
        ASSERT_TRUE(response.ok()) << response.status();
        if (mode == PriorityRestriction::kAuto) {
          early_stops += response->terminated_early;
        }
        expected.push_back(*std::move(response));
      }
    }
    for (size_t m = 1; m < expected_by_mode.size(); ++m) {
      for (size_t i = 0; i < requests.size(); ++i) {
        SCOPED_TRACE(i);
        ExpectSameFiles(expected_by_mode[m][i], expected_by_mode[0][i]);
        // Fewer priority documents can only stop sooner.
        const SearchStats& never = expected_by_mode[1][i].stats;
        const SearchStats& other = expected_by_mode[m][i].stats;
        EXPECT_LE(other.verified_bytes, never.verified_bytes);
        EXPECT_LE(other.priority_documents, never.priority_documents);
      }
    }
    for (size_t m = 0; m < expected_by_mode.size(); ++m) {
      SCOPED_TRACE(m);
      const std::vector<SearchResponse>& expected = expected_by_mode[m];
      const SearcherOptions count_all{.count_only_min_bytes = 0,
                                      .priority_restriction = modes[m]};
      for (size_t i = 0; i < requests.size(); ++i) {
        SCOPED_TRACE(i);
        auto response = scoped ? Search(*state, view, requests[i], count_all)
                               : Search(*state, requests[i], count_all);
        ASSERT_TRUE(response.ok()) << response.status();
        ExpectSameResponse(*response, expected[i]);
        counted_documents += response->stats.count_only_documents;
      }
    }
    for (int threads : {2, 4, 8}) {
      SCOPED_TRACE(threads);
      SearchExecutor executor(threads);
      for (size_t m = 0; m < expected_by_mode.size(); ++m) {
        SCOPED_TRACE(m);
        const std::vector<SearchResponse>& expected = expected_by_mode[m];
        for (int run_ahead : {1, 2, 4}) {
          SCOPED_TRACE(run_ahead);
          for (int grain : {1, 3}) {
            const SearcherOptions options{
                .count_only_min_bytes = 0,
                .priority_restriction = modes[m],
                .priority_range_docs = grain == 1 ? 3 : 32768,
                .executor = &executor,
                .parallel_threshold_bytes = 0,
                .grain_chunks = grain,
                .grain_bytes = 1,
                .ring_slots_per_worker = 4,
                .full_ring_slots_per_worker = run_ahead};
            for (size_t i = 0; i < requests.size(); ++i) {
              SCOPED_TRACE(i);
              auto response = scoped
                                  ? Search(*state, view, requests[i], options)
                                  : Search(*state, requests[i], options);
              ASSERT_TRUE(response.ok()) << response.status();
              ExpectSameResponse(*response, expected[i]);
            }
          }
        }
      }
    }
  }
  EXPECT_GT(early_stops, 4);
  EXPECT_GT(counted_documents, 50);
}

// A view with one or two paths per blob; some paths are named after the
// pattern, and some blobs are invisible.
SearchView MakeView(const EarlyStopCorpus& corpus, PathTable& paths,
                    ViewWriter& writer) {
  std::vector<ViewEdit> edits;
  for (size_t i = 0; i < corpus.oids.size(); ++i) {
    if (i % 11 == 0) continue;
    const BlobId blob = corpus.index->blobs().Intern(corpus.oids[i]);
    auto one = paths.Intern(corpus.paths[i]);
    EXPECT_TRUE(one.ok());
    edits.push_back(ViewEdit{*one, blob});
    if (i % 5 == 0) {
      auto two = paths.Intern(absl::StrCat("alias/needle_", i, ".txt"));
      EXPECT_TRUE(two.ok());
      edits.push_back(ViewEdit{*two, blob});
    }
  }
  writer.ApplyBatch(edits);
  return SearchView{writer.Snapshot(), paths};
}

void ExpectSameEstimate(const SearchResponse& got, const SearchResponse& want) {
  EXPECT_EQ(got.complete, want.complete);
  EXPECT_EQ(got.terminated_early, want.terminated_early);
  EXPECT_EQ(got.counts_estimated, want.counts_estimated);
  EXPECT_EQ(got.count_sample_size, want.count_sample_size);
  EXPECT_EQ(got.num_matching_files, want.num_matching_files);
  EXPECT_EQ(got.num_matching_lines, want.num_matching_lines);
  EXPECT_EQ(got.match_classes, want.match_classes);
  EXPECT_EQ(got.stats.count_sample_documents,
            want.stats.count_sample_documents);
  EXPECT_EQ(got.stats.count_sample_bytes, want.stats.count_sample_bytes);
  ASSERT_EQ(got.files.size(), want.files.size());
  for (size_t i = 0; i < got.files.size(); ++i) {
    EXPECT_EQ(got.files[i].path, want.files[i].path);
    EXPECT_EQ(got.files[i].lines, want.files[i].lines);
  }
}

// Count sample: the bounded heap and the resumed, skipping frame draw the
// reference selection's sample, on segments plus memtable (deduplicating, Oids
// repeated across and within sources), with a glob, a view and a byte budget,
// at every thread count.
TEST(SearchParallelTest, CountSampleLevelsDrawTheSameSample) {
  const uint64_t seed = ferret_testing::TestSeed(0x5a3b1e);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  int estimated = 0;
  int64_t chunks_saved = 0;
  int64_t split_frames = 0;
  int64_t split_view_frames = 0;
  int64_t split_view_repeats = 0;
  for (int round = 0; round < 4; ++round) {
    SCOPED_TRACE(round);
    ferret_testing::TempDir dir;
    const EarlyStopCorpus corpus = BuildEarlyStopCorpus(
        dir, rng, 240 + 60 * round, round == 3 ? 1000000 : 3 + round);
    const std::shared_ptr<const ReadState> state = corpus.index->GetReadState();
    PathTable paths;
    ViewWriter writer;
    const SearchView view = MakeView(corpus, paths, writer);
    std::vector<SearchRequest> requests;
    for (int i = 0; i < 12; ++i) {
      SearchRequest request;
      request.pattern = i % 4 == 3 ? "nee.le" : "needle";
      request.regex = i % 4 == 3;
      request.max_files = i % 3 == 0 ? 1 : (i % 3 == 1 ? 5 : 20);
      request.max_lines_per_file = i % 2 == 0 ? 1 : 100;
      request.count_sample = true;
      request.path_glob = i % 5 == 4 ? "src/**" : "";
      request.max_verify_bytes = i == 7 ? 600 : int64_t{1} << 24;
      requests.push_back(std::move(request));
    }
    SearchExecutor executor(4);
    for (const bool scoped : {false, true}) {
      SCOPED_TRACE(scoped);
      for (size_t i = 0; i < requests.size(); ++i) {
        SCOPED_TRACE(i);
        auto run = [&](const SearcherOptions& options) {
          return scoped ? Search(*state, view, requests[i], options)
                        : Search(*state, requests[i], options);
        };
        FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse reference,
                                    run({.count_sample_level = 0}));
        estimated += reference.counts_estimated;
        for (int level : {1, 2, 3}) {
          SCOPED_TRACE(level);
          for (const SearcherOptions& options :
               {SearcherOptions{.count_sample_level = level},
                SearcherOptions{.count_sample_level = level,
                                .count_sample_parallel_chunks = 1,
                                .executor = &executor,
                                .parallel_threshold_bytes = 0,
                                .grain_chunks = 1,
                                .grain_bytes = 1}}) {
            FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse got, run(options));
            ExpectSameEstimate(got, reference);
            if (level == 3 && got.stats.count_sample_parts > 1) {
              ++split_frames;
              split_view_frames += scoped;
              split_view_repeats += scoped && round < 3;
            }
            if (level == 1) {
              EXPECT_EQ(got.stats.count_sample_chunks,
                        reference.stats.count_sample_chunks);
            } else {
              EXPECT_LE(got.stats.count_sample_chunks,
                        reference.stats.count_sample_chunks);
              chunks_saved += reference.stats.count_sample_chunks -
                              got.stats.count_sample_chunks;
            }
          }
        }
      }
    }
    // The single-segment path, which does not deduplicate.
    ASSERT_FALSE(state->version->segments.empty());
    const Segment& segment = *state->version->segments.front().segment;
    for (size_t i = 0; i < requests.size(); ++i) {
      SCOPED_TRACE(i);
      FERRET_ASSERT_OK_AND_ASSIGN(
          SearchResponse reference,
          SearchSegment(segment, requests[i], {.count_sample_level = 0}));
      estimated += reference.counts_estimated;
      for (int level : {1, 2, 3}) {
        FERRET_ASSERT_OK_AND_ASSIGN(
            SearchResponse got,
            SearchSegment(segment, requests[i], {.count_sample_level = level}));
        ExpectSameEstimate(got, reference);
        FERRET_ASSERT_OK_AND_ASSIGN(
            got, SearchSegment(segment, requests[i],
                               {.count_sample_level = level,
                                .count_sample_parallel_chunks = 1,
                                .executor = &executor,
                                .parallel_threshold_bytes = 0}));
        ExpectSameEstimate(got, reference);
      }
    }
  }
  EXPECT_GT(estimated, 30);
  EXPECT_GT(chunks_saved, 0);
  EXPECT_GT(split_frames, 5);
  EXPECT_GT(split_view_frames, 0);
  // Corpora whose segments repeat Oids split too.
  EXPECT_GT(split_view_repeats, 0);
}

// A view search finds a segment's priority documents through its inverse blob
// table, or by testing every document when the table is absent, and a lone
// segment without copies of an Oid skips deduplication: none of these changes a
// response, with or without a count sample, a glob or exact counts, at one and
// four threads.
TEST(SearchParallelTest,
     ViewSearchesAreTheSameWithoutTheInverseTableOrTheSkip) {
  const uint64_t seed = ferret_testing::TestSeed(0x3d1e7);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  SearchExecutor executor(4);
  int64_t compared = 0;
  int64_t skipped_dedup = 0;
  for (int round = 0; round < 3; ++round) {
    SCOPED_TRACE(round);
    ferret_testing::TempDir dir;
    const EarlyStopCorpus corpus =
        BuildEarlyStopCorpus(dir, rng, 300, round == 2 ? 1000000 : 9);
    const std::shared_ptr<const ReadState> state = corpus.index->GetReadState();
    ASSERT_GE(state->version->segments.size(), 2u);
    PathTable paths;
    ViewWriter writer;
    const SearchView view = MakeView(corpus, paths, writer);
    // The same snapshot without the inverse tables.
    ReadState plain_state = *state;
    auto plain_version = std::make_shared<Version>(*state->version);
    for (VersionSegment& entry : plain_version->segments) {
      entry.docs_of_blob = nullptr;
    }
    plain_state.version = plain_version;
    // The first segment alone: one source, so deduplication can be skipped.
    ReadState lone_state = *state;
    auto lone_version = std::make_shared<Version>(*state->version);
    lone_version->segments.resize(1);
    lone_state.version = lone_version;
    lone_state.frozen.clear();
    lone_state.mem = nullptr;
    lone_state.mem_watermark = 0;
    // The index ingests a blob once per segment, so the segment has no copy.
    EXPECT_FALSE(lone_version->segments[0].segment->has_duplicate_oids());
    for (int i = 0; i < 16; ++i) {
      SCOPED_TRACE(i);
      SearchRequest request;
      request.pattern =
          i % 4 == 3 ? "nee.le" : (i % 4 == 2 ? "util" : "needle");
      request.regex = i % 4 == 3;
      request.case_sensitive = i % 2 == 0;
      request.max_files = i % 3 == 0 ? 1 : (i % 3 == 1 ? 5 : 20);
      request.count_sample = i % 2 == 1;
      request.exact_counts = i % 8 == 6;
      request.path_glob = i % 5 == 4 ? "src/**" : "";
      for (const int threads : {1, 4}) {
        SearcherOptions options{.count_only_min_bytes = 0};
        if (threads > 1) {
          options.executor = &executor;
          options.parallel_threshold_bytes = 0;
          options.grain_chunks = 1;
          options.grain_bytes = 1;
        }
        std::optional<PriorityProbe> with_table;
        std::optional<PriorityProbe> without_table;
        auto keep = [&](std::optional<PriorityProbe>& out) {
          return [&out](const PriorityProbe& built) { out = built; };
        };
        auto first = keep(with_table);
        auto second = keep(without_table);
        const absl::FunctionRef<void(const PriorityProbe&)> first_ref(first);
        const absl::FunctionRef<void(const PriorityProbe&)> second_ref(second);
        SearcherOptions a = options;
        a.priority_probe = &first_ref;
        SearcherOptions b = options;
        b.priority_probe = &second_ref;
        FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse got,
                                    Search(*state, view, request, a));
        FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse want,
                                    Search(plain_state, view, request, b));
        ExpectSameResponse(got, want);
        EXPECT_EQ(with_table.has_value(), without_table.has_value());
        if (with_table.has_value() && without_table.has_value()) {
          EXPECT_EQ(with_table->keys, without_table->keys);
          EXPECT_EQ(with_table->both, without_table->both);
        }
        for (const bool scoped : {false, true}) {
          SearcherOptions off = options;
          off.lone_segment_skips_dedup = false;
          FERRET_ASSERT_OK_AND_ASSIGN(
              SearchResponse skipping,
              scoped ? Search(lone_state, view, request, options)
                     : Search(lone_state, request, options));
          FERRET_ASSERT_OK_AND_ASSIGN(
              SearchResponse deduplicating,
              scoped ? Search(lone_state, view, request, off)
                     : Search(lone_state, request, off));
          ExpectSameResponse(skipping, deduplicating);
          ++skipped_dedup;
        }
        ++compared;
      }
    }
  }
  EXPECT_EQ(compared, 96);
  EXPECT_GT(skipped_dedup, 30);
}

// A segment that does repeat an Oid (the format allows it) keeps its
// deduplication: only the first copy is a result, with or without the skip.
TEST(SearchParallelTest, LoneSegmentWithCopiesStillDeduplicates) {
  ferret_testing::TempDir dir;
  struct Doc {
    std::string path;
    std::string content;
  };
  std::vector<Doc> docs = {{"a/one.cc", "needle shared\n"},
                           {"b/two.cc", "needle shared\n"},
                           {"c/three.cc", "needle shared\n"},
                           {"d/four.cc", "needle other\nneedle more\n"}};
  std::vector<SegmentDocument> to_write;
  for (const Doc& doc : docs) {
    to_write.push_back(
        {.hint_path = doc.path,
         .content = doc.content,
         .rank_key = ComputeRankKey(doc.path, doc.content.size(), false,
                                    HashBlob(doc.content)),
         .language = DetectLanguage(doc.path, doc.content)});
  }
  std::sort(to_write.begin(), to_write.end(),
            [](const SegmentDocument& a, const SegmentDocument& b) {
              return a.rank_key < b.rank_key;
            });
  const std::string file = dir.Join("copies.fseg");
  FERRET_ASSERT_OK_AND_ASSIGN(auto writer,
                              SegmentWriter::Create(*Env::Default(), file));
  for (const SegmentDocument& doc : to_write) {
    ASSERT_THAT(writer->Add(doc), absl_testing::IsOk());
  }
  ASSERT_THAT(writer->Finish().status(), absl_testing::IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Segment::Open(*Env::Default(), file));
  const std::shared_ptr<const Segment> segment = std::move(opened);
  ASSERT_TRUE(segment->has_duplicate_oids());
  ReadState state;
  state.blobs = std::make_shared<BlobRegistry>();
  auto blob_of_doc = std::make_shared<std::vector<BlobId>>();
  for (DocId d = 0; d < segment->num_docs(); ++d) {
    blob_of_doc->push_back(state.blobs->Intern(segment->doc(d).oid));
  }
  auto version = std::make_shared<Version>();
  version->segments.push_back(
      {.segment = segment,
       .blob_of_doc = blob_of_doc,
       .docs_of_blob = std::make_shared<const DocsOfBlob>(*blob_of_doc)});
  state.version = version;
  for (const int64_t max_files : {int64_t{1}, int64_t{10}}) {
    SearchRequest request{.pattern = "needle", .max_files = max_files};
    request.exact_counts = true;
    FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse skipping,
                                Search(state, request));
    FERRET_ASSERT_OK_AND_ASSIGN(
        SearchResponse deduplicating,
        Search(state, request, {.lone_segment_skips_dedup = false}));
    ExpectSameResponse(skipping, deduplicating);
    // Two files: the shared content once, and the other document.
    EXPECT_EQ(skipping.num_matching_files, 2);
    ASSERT_EQ(skipping.files.size(),
              static_cast<size_t>(std::min<int64_t>(max_files, 2)));
  }
}

// Counting-only with estimates: the commit alone decides which documents are
// counted, so the estimated classes of counted documents (a sample of them,
// once more than 64 are counted) are the same at every thread count and
// run-ahead, with and without a view.
TEST(SearchParallelTest, CountedClassEstimatesDoNotDependOnThreads) {
  const uint64_t seed = ferret_testing::TestSeed(0x7c0de5);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  ferret_testing::TempDir dir;
  const EarlyStopCorpus corpus = BuildEarlyStopCorpus(dir, rng, 1500, 13);
  const std::shared_ptr<const ReadState> state = corpus.index->GetReadState();
  PathTable paths;
  ViewWriter writer;
  const SearchView view = MakeView(corpus, paths, writer);
  std::vector<SearchRequest> requests;
  for (int i = 0; i < 6; ++i) {
    SearchRequest request;
    request.pattern = i % 3 == 2 ? "nee.le" : "needle";
    request.regex = i % 3 == 2;
    request.max_files = i % 2 == 0 ? 3 : 10;
    request.max_lines_per_file = 1 + i;
    request.count_sample = true;
    requests.push_back(std::move(request));
  }
  int64_t sampled_counted = 0;
  for (const bool scoped : {false, true}) {
    SCOPED_TRACE(scoped);
    for (size_t i = 0; i < requests.size(); ++i) {
      SCOPED_TRACE(i);
      auto run = [&](const SearcherOptions& options) {
        return scoped ? Search(*state, view, requests[i], options)
                      : Search(*state, requests[i], options);
      };
      FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse want,
                                  run({.count_only_min_bytes = 40000}));
      FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse all,
                                  run({.count_only_excluded = false}));
      // Counting changes only the estimate of the counted documents' classes.
      EXPECT_EQ(want.num_matching_files, all.num_matching_files);
      EXPECT_EQ(want.num_matching_lines, all.num_matching_lines);
      EXPECT_EQ(want.count_sample_size, all.count_sample_size);
      sampled_counted +=
          want.stats.counted_sample_documents < want.stats.count_only_documents;
      for (int threads : {2, 4, 8}) {
        SearchExecutor executor(threads);
        for (int run_ahead : {1, 4}) {
          FERRET_ASSERT_OK_AND_ASSIGN(
              SearchResponse got,
              run({.count_only_min_bytes = 40000,
                   .executor = &executor,
                   .parallel_threshold_bytes = 0,
                   .grain_chunks = 3,
                   .grain_bytes = 1,
                   .full_ring_slots_per_worker = run_ahead}));
          ExpectSameEstimate(got, want);
          EXPECT_EQ(got.stats.count_only_documents,
                    want.stats.count_only_documents);
          EXPECT_EQ(got.stats.counted_sample_documents,
                    want.stats.counted_sample_documents);
        }
      }
    }
  }
  EXPECT_GT(sampled_counted, 4);
}

// The documents of `segment` with a candidate chunk for `request`, as the
// RankKeys the priority set holds.
std::vector<RankKey> CandidateKeys(const Segment& segment,
                                   const SearchRequest& request) {
  auto compiled = CompilePattern({.pattern = request.pattern,
                                  .literal = !request.regex,
                                  .case_sensitive = request.case_sensitive});
  EXPECT_TRUE(compiled.ok());
  auto evaluator = segment.NewEvaluator((*compiled)->query());
  EXPECT_TRUE(evaluator.ok());
  std::vector<RankKey> keys;
  if (segment.num_chunks() == 0) return keys;
  ChunkIdBuffer chunks;
  evaluator->Evaluate(
      {.first = 0, .last = static_cast<ChunkId>(segment.num_chunks() - 1)},
      chunks);
  for (const ChunkId chunk : chunks.ids()) {
    const SegmentDoc doc = segment.doc(segment.chunk(chunk).doc);
    keys.push_back({.tier = doc.tier, .prior = doc.prior, .oid = doc.oid});
  }
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  return keys;
}

// A priority scan that costs what the query costs: the restricted set is the
// full set limited to the candidate documents of each segment, memtables are
// unchanged, and no thread count, task size or restriction changes the set that
// one thread builds.
TEST(SearchParallelTest, PrioritySetIsTheNameMatchesAmongCandidateDocuments) {
  const uint64_t seed = ferret_testing::TestSeed(0x6a11);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  ferret_testing::TempDir dir;
  const EarlyStopCorpus corpus = BuildEarlyStopCorpus(dir, rng, 300, 7);
  const std::shared_ptr<const ReadState> state = corpus.index->GetReadState();
  const size_t segments = state->version->segments.size();
  ASSERT_GE(segments, 2u);

  PathTable paths;
  ViewWriter writer;
  std::vector<ViewEdit> edits;
  for (size_t i = 0; i < corpus.oids.size(); ++i) {
    if (i % 11 == 0) continue;
    const BlobId blob = corpus.index->blobs().Intern(corpus.oids[i]);
    auto one = paths.Intern(corpus.paths[i]);
    ASSERT_TRUE(one.ok());
    edits.push_back(ViewEdit{*one, blob});
    if (i % 5 == 0) {
      auto two = paths.Intern(absl::StrCat("alias/needle_", i, ".txt"));
      ASSERT_TRUE(two.ok());
      edits.push_back(ViewEdit{*two, blob});
    }
  }
  writer.ApplyBatch(edits);
  const SearchView view{writer.Snapshot(), paths};

  struct Pattern {
    const char* text;
    bool regex;
    bool case_sensitive;
  };
  const Pattern patterns[] = {
      {"needle", false, true},
      {"NEEDLE", false, false},
      {"(?:needle_core|my_needle_x)", true, true},
      {"(?i)(UTIL|main)", true, true},
      {"util|main|other", true, true},
      {"needle_(core|x)|other", true, true},
      {"(?:needle|util)\\w+", true, true},
      {"[a-z]+_core", true, true},
      {"x.*y|z.*w", true, true},  // no required string at all
  };
  const char* const globs[] = {"", "src/**", "!test"};
  int probes = 0;
  int shrunk = 0;
  int64_t restricted_sources = 0;
  for (const bool scoped : {false, true}) {
    SCOPED_TRACE(scoped);
    for (const Pattern& pattern : patterns) {
      SCOPED_TRACE(pattern.text);
      for (const int k : {1, 3}) {
        SearchRequest request;
        request.pattern = pattern.text;
        request.regex = pattern.regex;
        request.case_sensitive = pattern.case_sensitive;
        request.max_files = k;
        request.path_glob = globs[rng() % 3];
        auto search = [&](const SearcherOptions& options,
                          std::optional<PriorityProbe>& probe,
                          SearchStats* stats) {
          auto record = [&](const PriorityProbe& built) { probe = built; };
          const absl::FunctionRef<void(const PriorityProbe&)> ref(record);
          SearcherOptions with_probe = options;
          with_probe.priority_probe = &ref;
          auto response = scoped ? Search(*state, view, request, with_probe)
                                 : Search(*state, request, with_probe);
          EXPECT_TRUE(response.ok()) << response.status();
          if (response.ok() && stats != nullptr) *stats = response->stats;
        };
        std::optional<PriorityProbe> full;
        search({.priority_restriction = PriorityRestriction::kNever}, full,
               nullptr);
        if (!full.has_value()) continue;  // The bound never needed the set.
        ++probes;
        ASSERT_GE(full->keys.size(), segments);
        std::optional<PriorityProbe> serial;
        SearchStats stats;
        search({.priority_restriction = PriorityRestriction::kAlways}, serial,
               &stats);
        ASSERT_TRUE(serial.has_value());
        restricted_sources += stats.priority_restricted_sources;
        for (size_t source = 0; source < full->keys.size(); ++source) {
          SCOPED_TRACE(source);
          std::vector<RankKey> want = full->keys[source];
          if (source < segments) {
            const std::vector<RankKey> candidates = CandidateKeys(
                *state->version->segments[source].segment, request);
            std::erase_if(want, [&](const RankKey& key) {
              return !std::binary_search(candidates.begin(), candidates.end(),
                                         key);
            });
            shrunk += want.size() < full->keys[source].size();
          }
          EXPECT_EQ(serial->keys[source], want);
          EXPECT_TRUE(std::is_sorted(serial->keys[source].begin(),
                                     serial->keys[source].end()));
          // Documents also in the token list stay a subset of the keys.
          EXPECT_TRUE(std::includes(
              serial->keys[source].begin(), serial->keys[source].end(),
              serial->both[source].begin(), serial->both[source].end()));
        }
        // The same set at every thread count, task size and restriction.
        for (const PriorityRestriction mode :
             {PriorityRestriction::kNever, PriorityRestriction::kAuto,
              PriorityRestriction::kAlways}) {
          std::optional<PriorityProbe> one;
          search({.priority_restriction = mode, .priority_range_docs = 1}, one,
                 nullptr);
          ASSERT_TRUE(one.has_value());
          for (const int threads : {2, 4, 8}) {
            SearchExecutor executor(threads);
            for (const int64_t range :
                 {int64_t{1}, int64_t{5}, int64_t{4096}}) {
              std::optional<PriorityProbe> many;
              search({.priority_restriction = mode,
                      .priority_range_docs = range,
                      .executor = &executor},
                     many, nullptr);
              ASSERT_TRUE(many.has_value());
              EXPECT_EQ(many->keys, one->keys) << threads << " " << range;
              EXPECT_EQ(many->both, one->both) << threads << " " << range;
            }
          }
          if (mode == PriorityRestriction::kNever) {
            EXPECT_EQ(one->keys, full->keys);
          }
          if (mode == PriorityRestriction::kAlways) {
            EXPECT_EQ(one->keys, serial->keys);
          }
        }
      }
    }
  }
  EXPECT_GT(probes, 10);
  EXPECT_GT(shrunk, 3) << "the restriction never removed a document";
  EXPECT_GT(restricted_sources, 20);
}

// Once the heap holds K files, workers must not verify documents its worst
// file rules out. Every source document sorts before the third-party tail, so
// the tail is excluded as soon as the first grain commits.
TEST(SearchParallelTest, WorkersSkipDocumentsThePublishedWorstExcludes) {
  ferret_testing::TempDir dir;
  IndexOptions index_options;
  index_options.memtable.flush_docs = 1000;
  index_options.memtable.segment.chunk_bytes = 4096;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened,
      Index::Open(*Env::Default(), dir.Join("skip"), index_options));
  std::unique_ptr<Index> index = std::move(opened.first);
  std::string body;
  for (int j = 0; j < 400; ++j) {
    absl::StrAppend(&body, "needle padding ", j, "\n");
  }
  for (int i = 0; i < 300; ++i) {
    const std::string content = absl::StrCat(body, i);
    const std::string path =
        absl::StrCat(i < 40 ? "src/a" : "third_party/b", i, ".cc");
    ASSERT_THAT(index->Add({.hint_path = path,
                            .content = content,
                            .rank_key = ComputeRankKey(
                                path, content.size(), false, HashBlob(content)),
                            .language = DetectLanguage(path, content)}),
                absl_testing::IsOk());
  }
  ASSERT_THAT(index->Flush(), absl_testing::IsOk());
  const std::shared_ptr<const ReadState> state = index->GetReadState();
  SearchRequest request;
  request.pattern = "needle";
  request.max_files = 1;
  request.max_lines_per_file = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse expected, Search(*state, request));
  ASSERT_TRUE(expected.terminated_early);
  SearchExecutor executor(4);
  int64_t skipped = 0;
  for (int run = 0; run < 20; ++run) {
    const SearcherOptions options{.executor = &executor,
                                  .parallel_threshold_bytes = 0,
                                  .grain_chunks = 1,
                                  .grain_bytes = 1};
    FERRET_ASSERT_OK_AND_ASSIGN(SearchResponse got,
                                Search(*state, request, options));
    ExpectSameResponse(got, expected);
    EXPECT_EQ(got.stats.excluded_scans, 0);
    skipped += got.stats.worker_skipped_documents;
  }
  EXPECT_GT(skipped, 0);
}

TEST(SearchParallelTest, RejectsNonPositiveFullRunAhead) {
  ferret_testing::TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(*Env::Default(), dir.Join("bad")));
  SearcherOptions options;
  options.full_ring_slots_per_worker = 0;
  SearchRequest request;
  request.pattern = "x";
  EXPECT_EQ(
      Search(*opened.first->GetReadState(), request, options).status().code(),
      absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ferret

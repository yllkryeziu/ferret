// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/git/blob_hash.h"
#include "ferret/git/oid.h"
#include "ferret/git/path_glob.h"
#include "ferret/index/blob_registry.h"
#include "ferret/index/index.h"
#include "ferret/index/memtable.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/read_state.h"
#include "ferret/index/segment.h"
#include "ferret/search/search_types.h"
#include "ferret/search/searcher.h"
#include "ferret/testing/brute_force_search.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/testing/test_seed.h"
#include "ferret/text/language.h"
#include "ferret/workspace/path_table.h"
#include "ferret/workspace/view_state.h"

namespace ferret {
namespace {

struct IndexedContent {
  std::string text;
  RankKey rank;
  BlobId blob;
};

void ExpectFilesEqual(const SearchResponse& got, const SearchResponse& want) {
  ASSERT_EQ(got.files.size(), want.files.size());
  for (size_t i = 0; i < got.files.size(); ++i) {
    SCOPED_TRACE(i);
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

void ExpectSameResponse(const SearchResponse& got, const SearchResponse& want) {
  ExpectFilesEqual(got, want);
  EXPECT_EQ(got.num_matching_files, want.num_matching_files);
  EXPECT_EQ(got.num_matching_lines, want.num_matching_lines);
  EXPECT_EQ(got.complete, want.complete);
  EXPECT_EQ(got.terminated_early, want.terminated_early);
  EXPECT_EQ(got.verified_fraction, want.verified_fraction);
  EXPECT_EQ(got.stats.candidate_chunks, want.stats.candidate_chunks);
  // A parallel producer can filter beyond a rank stop. The count is a
  // diagnostic of work done, not part of the search result.
  if (!got.terminated_early) {
    EXPECT_EQ(got.stats.view_filtered_chunks, want.stats.view_filtered_chunks);
  }
  EXPECT_EQ(got.stats.verified_chunks, want.stats.verified_chunks);
  EXPECT_EQ(got.stats.verified_bytes, want.stats.verified_bytes);
}

// The model maps paths to the content that was published before the view edit.
using ViewModel = absl::btree_map<PathId, size_t>;

std::vector<ferret_testing::ViewCorpusFile> ViewCorpus(
    const ViewModel& model, const PathTable& paths,
    const std::vector<IndexedContent>& contents) {
  std::vector<ferret_testing::ViewCorpusFile> corpus;
  for (const auto& [path, content] : model) {
    corpus.push_back(
        {paths.path(path), contents[content].text, contents[content].rank});
  }
  return corpus;
}

TEST(SearchViewTest, GlobExpandsOnlyMatchingPathsAndUsesIngestRank) {
  ferret_testing::TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(*Env::Default(), dir.Join("index")));
  std::unique_ptr<Index> index = std::move(opened.first);
  const std::string text = "needle\nother\n";
  const std::string hint = "third_party/original.txt";
  const Oid oid = HashBlob(text);
  ASSERT_THAT(
      index->Add({.hint_path = hint,
                  .content = text,
                  .rank_key = ComputeRankKey(hint, text.size(), false, oid),
                  .language = DetectLanguage(hint, text)}),
      absl_testing::IsOk());
  PathTable paths;
  FERRET_ASSERT_OK_AND_ASSIGN(PathId one, paths.Intern("src/needle.txt"));
  FERRET_ASSERT_OK_AND_ASSIGN(PathId two, paths.Intern("docs/alias.txt"));
  ViewWriter writer;
  const BlobId blob = index->blobs().Intern(oid);
  writer.ApplyBatch({ViewEdit{one, blob}, ViewEdit{two, blob}});
  const SearchView view{writer.Snapshot(), paths};
  const auto state = index->GetReadState();
  SearchRequest request{.pattern = "needle",
                        .path_glob = "src/**",
                        .context_lines = 0,
                        .max_files = 1};
  FERRET_ASSERT_OK_AND_ASSIGN(auto filtered, Search(*state, view, request));
  ASSERT_EQ(filtered.files.size(), 1u);
  EXPECT_EQ(filtered.files.front().path, "src/needle.txt");
  EXPECT_TRUE(filtered.files.front().name_matches);
  EXPECT_EQ(filtered.files.front().tier, RankTier::kThirdParty);
  EXPECT_EQ(filtered.num_matching_files, 1);
  request.path_glob.clear();
  request.max_files = 2;
  FERRET_ASSERT_OK_AND_ASSIGN(auto expanded, Search(*state, view, request));
  ASSERT_EQ(expanded.files.size(), 2u);
  EXPECT_EQ(expanded.num_matching_files, 2);
  EXPECT_EQ(expanded.num_matching_lines, 2);
  EXPECT_EQ(expanded.files[0].lines, expanded.files[1].lines);
  EXPECT_NE(expanded.files[0].name_matches, expanded.files[1].name_matches);
  request.max_files = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(auto top_one, Search(*state, view, request));
  ASSERT_EQ(top_one.files.size(), 1u);
  EXPECT_EQ(top_one.files.front().path, "src/needle.txt");
}

TEST(SearchViewTest, InvisibleFlushedAndMemtableChunksDoNotConsumeBudget) {
  ferret_testing::TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(*Env::Default(), dir.Join("index")));
  std::unique_ptr<Index> index = std::move(opened.first);
  PathTable paths;
  FERRET_ASSERT_OK_AND_ASSIGN(PathId path, paths.Intern("src/visible.txt"));
  ViewWriter writer;
  auto add = [&](const std::string& content) {
    const std::string hint = "src/hint.txt";
    const Oid oid = HashBlob(content);
    EXPECT_THAT(index->Add({.hint_path = hint,
                            .content = content,
                            .rank_key = ComputeRankKey(hint, content.size(),
                                                       false, oid),
                            .language = DetectLanguage(hint, content)}),
                absl_testing::IsOk());
    return index->blobs().Intern(oid);
  };
  const BlobId deleted = add("needle deleted\n");
  writer.ApplyBatch({ViewEdit{path, deleted}});
  ASSERT_THAT(index->Flush(), absl_testing::IsOk());
  const BlobId invisible = add("needle memtable\n");
  const BlobId visible = add("needle visible\n");
  writer.ApplyBatch({ViewEdit{path, std::nullopt}});
  const SearchView empty{writer.Snapshot(), paths};
  SearchRequest request{.pattern = "needle",
                        .context_lines = 0,
                        .max_verify_bytes = 1,
                        .exact_counts = true};
  const auto state = index->GetReadState();
  FERRET_ASSERT_OK_AND_ASSIGN(auto absent, Search(*state, empty, request));
  EXPECT_TRUE(absent.files.empty());
  EXPECT_EQ(absent.stats.view_filtered_chunks, 3);
  EXPECT_EQ(absent.stats.candidate_chunks, 0);
  EXPECT_EQ(absent.stats.verified_bytes, 0);
  EXPECT_TRUE(absent.complete);
  EXPECT_FALSE(empty.state->Contains(invisible));
  writer.ApplyBatch({ViewEdit{path, visible}});
  const SearchView view{writer.Snapshot(), paths};
  FERRET_ASSERT_OK_AND_ASSIGN(auto result, Search(*state, view, request));
  EXPECT_EQ(result.stats.view_filtered_chunks, 2);
  EXPECT_EQ(result.stats.candidate_chunks, 1);
  EXPECT_EQ(result.stats.verified_chunks, 1);
  EXPECT_TRUE(result.complete);  // Memtables ignore the segment byte budget.
  ASSERT_EQ(result.files.size(), 1u);
  EXPECT_EQ(result.files.front().path, "src/visible.txt");
}

TEST(SearchViewTest, ByteBudgetSkipsInvisibleSegmentsButIncludesMemtables) {
  ferret_testing::TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(*Env::Default(), dir.Join("index")));
  std::unique_ptr<Index> index = std::move(opened.first);
  PathTable paths;
  FERRET_ASSERT_OK_AND_ASSIGN(PathId first, paths.Intern("src/first.txt"));
  FERRET_ASSERT_OK_AND_ASSIGN(PathId later, paths.Intern("src/later.txt"));
  FERRET_ASSERT_OK_AND_ASSIGN(PathId active, paths.Intern("src/active.txt"));
  ViewWriter writer;
  std::vector<ferret_testing::ViewCorpusFile> verified;
  auto add = [&](const std::string& hint, const std::string& text) {
    const Oid oid = HashBlob(text);
    const RankKey rank = ComputeRankKey(hint, text.size(), false, oid);
    EXPECT_THAT(index->Add({.hint_path = hint,
                            .content = text,
                            .rank_key = rank,
                            .language = DetectLanguage(hint, text)}),
                absl_testing::IsOk());
    return IndexedContent{text, rank, index->blobs().Intern(oid)};
  };
  add("src/invisible.txt", "needle invisible\n");
  const IndexedContent one = add("src/first.txt", "needle first\n");
  const IndexedContent two = add("third_party/second.txt", "needle second\n");
  ASSERT_THAT(index->Flush(), absl_testing::IsOk());
  const IndexedContent mem = add("docs/active.txt", "needle active\n");
  writer.ApplyBatch({ViewEdit{first, one.blob}, ViewEdit{later, two.blob},
                     ViewEdit{active, mem.blob}});
  const SearchView view{writer.Snapshot(), paths};
  const auto state = index->GetReadState();
  const SearchRequest request{.pattern = "needle",
                              .context_lines = 0,
                              .max_files = 10,
                              .max_verify_bytes = 1,
                              .exact_counts = true};
  FERRET_ASSERT_OK_AND_ASSIGN(auto result, Search(*state, view, request));
  verified.push_back({paths.path(first), one.text, one.rank});
  verified.push_back({paths.path(active), mem.text, mem.rank});
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto expected, ferret_testing::BruteForceViewSearch(verified, request));
  ExpectFilesEqual(result, expected);
  EXPECT_EQ(result.num_matching_files, expected.num_matching_files);
  EXPECT_EQ(result.num_matching_lines, expected.num_matching_lines);
  EXPECT_EQ(result.stats.view_filtered_chunks, 1);
  EXPECT_FALSE(result.complete);
}

TEST(SearchViewTest, LiveBlobAbsentFromOlderReadStateIsNotFound) {
  ferret_testing::TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(*Env::Default(), dir.Join("index")));
  std::unique_ptr<Index> index = std::move(opened.first);
  const auto old_state = index->GetReadState();
  const std::string text = "needle\n";
  const std::string hint = "src/ingest.txt";
  const Oid oid = HashBlob(text);
  ASSERT_THAT(
      index->Add({.hint_path = hint,
                  .content = text,
                  .rank_key = ComputeRankKey(hint, text.size(), false, oid),
                  .language = DetectLanguage(hint, text)}),
      absl_testing::IsOk());
  PathTable paths;
  FERRET_ASSERT_OK_AND_ASSIGN(PathId path, paths.Intern("src/live.txt"));
  ViewWriter writer;
  writer.ApplyBatch({ViewEdit{path, index->blobs().Intern(oid)}});
  const SearchView view{writer.Snapshot(), paths};
  const SearchRequest request{.pattern = "needle"};
  FERRET_ASSERT_OK_AND_ASSIGN(auto missing, Search(*old_state, view, request));
  EXPECT_TRUE(missing.files.empty());
  FERRET_ASSERT_OK_AND_ASSIGN(auto found,
                              Search(*index->GetReadState(), view, request));
  ASSERT_EQ(found.files.size(), 1u);
  EXPECT_EQ(found.files.front().path, "src/live.txt");
}

// View names, not the indexed hint names, determine the priority set. The
// same blob can have a matching basename in one worktree and not the other.
TEST(SearchViewTest, CachedNamesStopBeforeADeepNonpriorityDocument) {
  ferret_testing::TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(*Env::Default(), dir.Join("index")));
  std::unique_ptr<Index> index = std::move(opened.first);
  const std::string root = "int needle = 0;\n";
  for (int i = 0; i < 18; ++i) {
    const std::string hint =
        i == 0 ? "z.cc" : absl::StrCat("a/b/c/d/e/f/other", i, ".cc");
    const std::string text =
        absl::StrCat("// needle ", i, "\n", i == 0 ? root : "");
    const Oid oid = HashBlob(text);
    ASSERT_THAT(
        index->Add({.hint_path = hint,
                    .content = text,
                    .rank_key = ComputeRankKey(hint, text.size(), false, oid),
                    .language = DetectLanguage(hint, text)}),
        absl_testing::IsOk());
  }
  ASSERT_THAT(index->Flush(), absl_testing::IsOk());
  const auto state = index->GetReadState();
  PathTable paths;
  ViewWriter writers[2];
  std::vector<ViewEdit> edits[2];
  for (int i = 0; i < 18; ++i) {
    const std::string text =
        absl::StrCat("// needle ", i, "\n", i == 0 ? root : "");
    const BlobId blob = index->blobs().Intern(HashBlob(text));
    for (int w = 0; w < 2; ++w) {
      FERRET_ASSERT_OK_AND_ASSIGN(
          PathId path,
          paths.Intern(absl::StrCat(
              w == 0 ? "src/" : "test/",
              (i == 0 || (w == 0 && i == 6)) ? "needle" : "other", i, ".cc")));
      edits[w].push_back({path, blob});
    }
  }
  const SearchView first{writers[0].ApplyBatch(edits[0]), paths};
  const SearchView second{writers[1].ApplyBatch(edits[1]), paths};
  SearchRequest request{.pattern = "needle", .max_files = 1};
  for (const SearchView* view : {&first, &second}) {
    FERRET_ASSERT_OK_AND_ASSIGN(auto early, Search(*state, *view, request));
    request.exact_counts = true;
    FERRET_ASSERT_OK_AND_ASSIGN(auto exact, Search(*state, *view, request));
    request.exact_counts = false;
    ExpectFilesEqual(early, exact);
    EXPECT_TRUE(early.complete);
    EXPECT_TRUE(early.terminated_early);
    EXPECT_LT(early.stats.verified_bytes, exact.stats.verified_bytes);
    EXPECT_LT(early.stats.verified_bytes * 3, exact.stats.verified_bytes);
    // The second view lacks the extra priority path for the same blob.
    EXPECT_EQ(early.stats.priority_documents, view == &first ? 2 : 1);
  }
}

// Files one level below a root with an identifier match only mention the word
// in comments and longer identifiers. None has a matching name in the view, so
// only the code-token lists keep them from blocking the stop: their previous
// ceiling ties the root's score.
TEST(SearchViewTest, CodeTokenBoundStopsBeforeCommentsInAView) {
  const std::string root = "int needle = 0;\n";
  auto text_of = [&](int i) {
    return i == 0 ? root
                  : absl::StrCat(i % 2 == 0 ? "// needle " : "int needles", i,
                                 "\n");
  };
  auto hint_of = [](int i) {
    return i == 0 ? std::string("z.cc") : absl::StrCat("d/other", i, ".cc");
  };
  int64_t verified[2] = {};
  for (const bool write_tokens : {true, false}) {
    SCOPED_TRACE(write_tokens);
    ferret_testing::TempDir dir;
    IndexOptions options;
    options.memtable.segment.write_code_tokens = write_tokens;
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto opened, Index::Open(*Env::Default(), dir.Join("index"), options));
    std::unique_ptr<Index> index = std::move(opened.first);
    for (int i = 0; i < 20; ++i) {
      const std::string text = text_of(i);
      const std::string hint = hint_of(i);
      const Oid oid = HashBlob(text);
      ASSERT_THAT(
          index->Add({.hint_path = hint,
                      .content = text,
                      .rank_key = ComputeRankKey(hint, text.size(), false, oid),
                      .language = DetectLanguage(hint, text)}),
          absl_testing::IsOk());
    }
    ASSERT_THAT(index->Flush(), absl_testing::IsOk());
    const auto state = index->GetReadState();
    PathTable paths;
    ViewWriter writer;
    std::vector<ViewEdit> edits;
    for (int i = 0; i < 20; ++i) {
      FERRET_ASSERT_OK_AND_ASSIGN(
          PathId path, paths.Intern(absl::StrCat("src/file", i, ".cc")));
      edits.push_back({path, index->blobs().Intern(HashBlob(text_of(i)))});
    }
    // A blob outside the view does not matter to the bound, only to results.
    writer.ApplyBatch(edits);
    const SearchView view{writer.Snapshot(), paths};
    SearchRequest request{.pattern = "needle", .max_files = 1};
    FERRET_ASSERT_OK_AND_ASSIGN(auto early, Search(*state, view, request));
    request.exact_counts = true;
    FERRET_ASSERT_OK_AND_ASSIGN(auto exact, Search(*state, view, request));
    ExpectFilesEqual(early, exact);
    ASSERT_EQ(early.files.size(), 1u);
    EXPECT_EQ(early.files[0].path, "src/file0.cc");
    EXPECT_TRUE(early.complete);
    EXPECT_EQ(early.terminated_early, write_tokens);
    EXPECT_EQ(early.stats.code_token_documents, write_tokens ? 1 : 0);
    verified[write_tokens] = early.stats.verified_chunks;
    // The unscoped search agrees.
    request.exact_counts = false;
    FERRET_ASSERT_OK_AND_ASSIGN(auto unscoped, Search(*state, request));
    EXPECT_EQ(unscoped.files.size(), 1u);
    EXPECT_EQ(unscoped.terminated_early, write_tokens);
    // A view that hides the root leaves only quiet files, whose results must
    // not depend on the section either.
    ViewWriter hidden_writer;
    std::vector<ViewEdit> hidden(edits.begin() + 1, edits.end());
    hidden_writer.ApplyBatch(hidden);
    const SearchView hidden_view{hidden_writer.Snapshot(), paths};
    FERRET_ASSERT_OK_AND_ASSIGN(auto hidden_early,
                                Search(*state, hidden_view, request));
    request.exact_counts = true;
    FERRET_ASSERT_OK_AND_ASSIGN(auto hidden_exact,
                                Search(*state, hidden_view, request));
    ExpectFilesEqual(hidden_early, hidden_exact);
  }
  EXPECT_EQ(verified[1], 1);
  EXPECT_EQ(verified[0], 20);
}

// A segment without the section (written by an older build) keeps the old
// ceiling for its documents; a memtable keeps no lists, so its candidates all
// count as listed. Neither may change a result.
TEST(SearchViewTest, CodeTokenBoundWithLegacySegmentsAndMemtables) {
  auto add = [](Index& index, const std::string& hint,
                const std::string& text) {
    const Oid oid = HashBlob(text);
    EXPECT_THAT(
        index.Add({.hint_path = hint,
                   .content = text,
                   .rank_key = ComputeRankKey(hint, text.size(), false, oid),
                   .language = DetectLanguage(hint, text)}),
        absl_testing::IsOk());
  };
  for (const bool with_legacy : {true, false}) {
    for (const bool with_memtable_depth_one : {true, false}) {
      SCOPED_TRACE(absl::StrCat(with_legacy, " ", with_memtable_depth_one));
      ferret_testing::TempDir dir;
      IndexOptions options;
      options.compaction.enabled = false;
      if (with_legacy) {
        options.memtable.segment.write_code_tokens = false;
        FERRET_ASSERT_OK_AND_ASSIGN(
            auto opened,
            Index::Open(*Env::Default(), dir.Join("index"), options));
        for (int i = 0; i < 5; ++i) {
          add(*opened.first, absl::StrCat("d/legacy", i, ".cc"),
              absl::StrCat("// needle legacy ", i, "\n"));
        }
        ASSERT_THAT(opened.first->Flush(), absl_testing::IsOk());
        opened.first.reset();
      }
      options.memtable.segment.write_code_tokens = true;
      FERRET_ASSERT_OK_AND_ASSIGN(
          auto opened,
          Index::Open(*Env::Default(), dir.Join("index"), options));
      std::unique_ptr<Index> index = std::move(opened.first);
      add(*index, "z.cc", "int needle = 0;\n");
      add(*index, "d/tokenized.cc", "// needle tokenized\n");
      ASSERT_THAT(index->Flush(), absl_testing::IsOk());
      std::vector<RankKey> segment_keys;
      for (const auto& [hint, text] : {std::pair<std::string, std::string>{
               "d/tokenized.cc", "// needle tokenized\n"}}) {
        segment_keys.push_back(
            ComputeRankKey(hint, text.size(), false, HashBlob(text)));
      }
      for (int i = 0; with_legacy && i < 5; ++i) {
        const std::string text = absl::StrCat("// needle legacy ", i, "\n");
        segment_keys.push_back(
            ComputeRankKey(absl::StrCat("d/legacy", i, ".cc"), text.size(),
                           false, HashBlob(text)));
      }
      // The active memtable: one level below the root, or two. At one level
      // its document must rank after every segment document, so that the
      // stop test meets it only as a remaining candidate.
      const std::string mem_hint =
          with_memtable_depth_one ? "d/mem.cc" : "d/e/mem.cc";
      std::string mem_text;
      for (int i = 0;; ++i) {
        ASSERT_LT(i, 10000);
        mem_text = absl::StrCat("// needle memtable ", i, "\n");
        bool last = true;
        for (const RankKey& key : segment_keys) {
          last = last && key < ComputeRankKey(mem_hint, mem_text.size(), false,
                                              HashBlob(mem_text));
        }
        if (last) break;
      }
      add(*index, mem_hint, mem_text);
      const auto state = index->GetReadState();
      SearchRequest request{.pattern = "needle", .max_files = 1};
      FERRET_ASSERT_OK_AND_ASSIGN(auto early, Search(*state, request));
      request.exact_counts = true;
      FERRET_ASSERT_OK_AND_ASSIGN(auto exact, Search(*state, request));
      ExpectFilesEqual(early, exact);
      ASSERT_EQ(early.files.size(), 1u);
      EXPECT_EQ(early.files[0].path, "z.cc");
      EXPECT_TRUE(early.complete);
      // Depth-one documents tie the root under the previous ceiling, which a
      // legacy segment keeps: it blocks the stop, while the tokenized quiet
      // document at depth one is stopped by the lists alone. A memtable
      // candidate counts as listed: it blocks the stop from one level down,
      // where it ties, and not from two.
      EXPECT_EQ(early.terminated_early,
                !with_legacy && !with_memtable_depth_one);
      EXPECT_GT(early.stats.code_token_documents, 0);
      // The memtable is always verified in full.
      EXPECT_GE(early.stats.verified_chunks, 2);
    }
  }
}

TEST(SearchViewTest, SeededWorktreesAgainstExhaustiveOracleAndPipeline) {
  const uint64_t seed = ferret_testing::TestSeed(0x75e4);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  ferret_testing::TempDir dir;
  IndexOptions options;
  options.memtable.flush_docs = 1000;
  options.memtable.segment.chunk_bytes = 32;
  options.compaction.tier_min_segments = 3;
  options.compaction.tier_base_bytes = 1 << 16;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened, Index::Open(*Env::Default(), dir.Join("index"), options));
  std::unique_ptr<Index> index = std::move(opened.first);
  std::vector<IndexedContent> contents;
  for (int i = 0; i < 25; ++i) {
    const std::string hint =
        absl::StrCat(i % 6 == 0 ? "third_party/" : "src/", "ingest", i, ".txt");
    std::string text = absl::StrCat("token", i, " needle\n");
    for (int j = 0; j < 2 + i % 6; ++j) {
      absl::StrAppend(&text, j % 3 == 0 ? "other\n" : "NEEDLE\n");
    }
    const Oid oid = HashBlob(text);
    const RankKey rank = ComputeRankKey(hint, text.size(), false, oid);
    ASSERT_THAT(index->Add({.hint_path = hint,
                            .content = text,
                            .rank_key = rank,
                            .language = DetectLanguage(hint, text)}),
                absl_testing::IsOk());
    contents.push_back({text, rank, index->blobs().Intern(oid)});
    if (i == 5 || i == 11 || i == 17 || i == 20) {
      ASSERT_THAT(index->Flush(), absl_testing::IsOk());
    }
  }
  index->WaitForCompactions();
  EXPECT_GT(index->Stats().completed_compactions, 0u);
  const auto compacted = index->GetReadState();
  EXPECT_GE(compacted->version->segments.size(), 2u);
  EXPECT_GT(compacted->mem_watermark, 0u);
  PathTable paths;
  std::vector<PathId> names;
  for (int i = 0; i < 40; ++i) {
    FERRET_ASSERT_OK_AND_ASSIGN(
        PathId id,
        paths.Intern(absl::StrCat(i % 3 == 0 ? "docs/" : "src/",
                                  i % 4 == 0 ? "needle" : "file", i, ".txt")));
    names.push_back(id);
  }
  ViewWriter writers[3];
  ViewModel models[3];
  SearchExecutor executor(4);
  const SearcherOptions classify_all{.count_only_excluded = false};
  const SearcherOptions count_all{.count_only_min_bytes = 0};
  const SearcherOptions parallel{.count_only_min_bytes = 0,
                                 .executor = &executor,
                                 .parallel_threshold_bytes = 0,
                                 .grain_chunks = 1,
                                 .grain_bytes = 1,
                                 .ring_slots_per_worker = 2};
  for (int batch = 0; batch < 8; ++batch) {
    SCOPED_TRACE(batch);
    for (int w = 0; w < 3; ++w) {
      std::vector<ViewEdit> edits;
      for (int edit = 0; edit < 7; ++edit) {
        const PathId path = names[rng() % names.size()];
        const size_t content = rng() % contents.size();
        if (rng() % 5 == 0) {
          edits.push_back({path, std::nullopt});
          models[w].erase(path);
        } else {
          edits.push_back({path, contents[content].blob});
          models[w][path] = content;
        }
      }
      // Move a path while preserving its content under a different name.
      const PathId from = names[2 + batch];
      const PathId to = names[20 + batch];
      if (const auto it = models[w].find(from); it != models[w].end()) {
        const size_t content = it->second;
        edits.push_back({from, std::nullopt});
        edits.push_back({to, contents[content].blob});
        models[w].erase(it);
        models[w][to] = content;
      }
      // An alias in a second directory tests glob filtering within a blob.
      edits.push_back({names[0], contents[batch].blob});
      edits.push_back({names[1], contents[batch].blob});
      models[w][names[0]] = batch;
      models[w][names[1]] = batch;
      writers[w].ApplyBatch(edits);
    }
    for (int w = 0; w < 3; ++w) {
      SCOPED_TRACE(w);
      const SearchView view{writers[w].Snapshot(), paths};
      const auto state = index->GetReadState();
      const std::vector<ferret_testing::ViewCorpusFile> corpus =
          ViewCorpus(models[w], paths, contents);
      for (int q = 0; q < 12; ++q) {
        SCOPED_TRACE(q);
        SearchRequest request;
        request.pattern = q % 3 == 0   ? "needle"
                          : q % 3 == 1 ? "other"
                                       : "NEEDLE|other";
        request.regex = q % 3 == 2;
        request.case_sensitive = q % 2 == 0;
        request.path_glob = q % 4 == 0   ? "src/**"
                            : q % 4 == 1 ? "*needle*.txt"
                                         : "";
        request.context_lines = q % 2;
        request.max_files =
            q % 3 == 0 ? 1 : std::numeric_limits<int64_t>::max();
        request.max_lines =
            q % 4 == 0 ? 2 : std::numeric_limits<int64_t>::max();
        request.max_lines_per_file = q % 3 == 1 ? 1 : 10;
        request.exact_counts = q % 2 == 0;
        FERRET_ASSERT_OK_AND_ASSIGN(
            auto expected,
            ferret_testing::BruteForceViewSearch(corpus, request));
        FERRET_ASSERT_OK_AND_ASSIGN(
            auto inline_result, Search(*state, view, request, classify_all));
        FERRET_ASSERT_OK_AND_ASSIGN(auto counted_result,
                                    Search(*state, view, request, count_all));
        ExpectSameResponse(counted_result, inline_result);
        FERRET_ASSERT_OK_AND_ASSIGN(auto parallel_result,
                                    Search(*state, view, request, parallel));
        ExpectSameResponse(parallel_result, inline_result);
        ExpectFilesEqual(inline_result, expected);
        if (!inline_result.terminated_early) {
          EXPECT_EQ(inline_result.num_matching_files,
                    expected.num_matching_files);
          EXPECT_EQ(inline_result.num_matching_lines,
                    expected.num_matching_lines);
        }
        // Budget-limited queries have no exhaustive oracle, but both execution
        // modes must agree on every response field other than timing and
        // grains.
        request.max_verify_bytes = 40 + q % 3 * 45;
        FERRET_ASSERT_OK_AND_ASSIGN(
            auto limited, Search(*state, view, request, classify_all));
        FERRET_ASSERT_OK_AND_ASSIGN(auto counted_limited,
                                    Search(*state, view, request, count_all));
        ExpectSameResponse(counted_limited, limited);
        FERRET_ASSERT_OK_AND_ASSIGN(auto parallel_limited,
                                    Search(*state, view, request, parallel));
        ExpectSameResponse(parallel_limited, limited);
      }
      // Two-byte patterns have no trigram: segments (when their bigram lists
      // are small enough) and memtables select candidates by bigram.
      SearcherOptions none;
      none.max_bigram_posting_share = 0;
      SearcherOptions forced;
      forced.max_bigram_posting_share = 1e9;
      SearcherOptions forced_parallel = parallel;
      forced_parallel.max_bigram_posting_share = 1e9;
      const char* const bigrams[] = {"ee", "ER", "er", "ne", "t", "dl"};
      for (int q = 0; q < 12; ++q) {
        SCOPED_TRACE(absl::StrCat("bigram ", q));
        SearchRequest request;
        request.pattern = bigrams[q % 6];
        request.case_sensitive = q % 2 == 0;
        request.path_glob = q % 4 == 0 ? "src/**" : "";
        request.max_lines_per_file = 10;
        request.max_files = std::numeric_limits<int64_t>::max();
        request.exact_counts = true;
        FERRET_ASSERT_OK_AND_ASSIGN(
            auto expected,
            ferret_testing::BruteForceViewSearch(corpus, request));
        request.exact_counts = q >= 8;
        request.max_files = q >= 8 ? 1 : std::numeric_limits<int64_t>::max();
        FERRET_ASSERT_OK_AND_ASSIGN(auto scan,
                                    Search(*state, view, request, none));
        FERRET_ASSERT_OK_AND_ASSIGN(auto narrowed,
                                    Search(*state, view, request, forced));
        FERRET_ASSERT_OK_AND_ASSIGN(
            auto narrowed_parallel,
            Search(*state, view, request, forced_parallel));
        // The oracle leaves match classes at their defaults, so its scores
        // and ranks differ for partial-token matches: compare the files and
        // matching lines, and the narrowed searches with the full scan.
        if (q < 8) {
          std::set<std::pair<std::string, int64_t>> want;
          for (const FileResult& file : expected.files) {
            for (const ResultLine& line : file.lines) {
              if (line.is_match) want.emplace(file.path, line.number);
            }
          }
          std::set<std::pair<std::string, int64_t>> got;
          for (const FileResult& file : scan.files) {
            for (const ResultLine& line : file.lines) {
              if (line.is_match) got.emplace(file.path, line.number);
            }
          }
          EXPECT_EQ(got, want);
          EXPECT_EQ(scan.num_matching_files, expected.num_matching_files);
          EXPECT_EQ(scan.num_matching_lines, expected.num_matching_lines);
        }
        ExpectFilesEqual(narrowed, scan);
        ExpectFilesEqual(narrowed_parallel, scan);
        EXPECT_LE(narrowed.stats.candidate_chunks, scan.stats.candidate_chunks);
        EXPECT_EQ(narrowed.terminated_early, scan.terminated_early);
        EXPECT_EQ(narrowed.num_matching_files, scan.num_matching_files);
        EXPECT_EQ(narrowed.num_matching_lines, scan.num_matching_lines);
      }
    }
  }
}

// The priority set of a view search is exactly the documents whose blob has a
// view path that passes the glob and whose basename contains the pattern: found
// through the view's names and the segments' inverse blob tables, or by testing
// every document, at any thread count. The corpus has renamed copies (a hint
// that matches under a view name that does not, and the reverse), several paths
// per blob, globs, edits after a snapshot was taken, several segments and a
// memtable.
TEST(SearchViewTest, PrioritySetIsTheBlobsWithAMatchingViewPath) {
  const uint64_t seed = ferret_testing::TestSeed(0x6e41);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  ferret_testing::TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(*Env::Default(), dir.Join("index")));
  std::unique_ptr<Index> index = std::move(opened.first);
  constexpr int kDocs = 60;
  std::vector<Oid> oids;
  for (int i = 0; i < kDocs; ++i) {
    // Document 0 is the shallow root that fills the heap; the rest are deep
    // and sort after it, so the stop test runs and builds the priority set.
    const std::string hint =
        i == 0 ? "root.cc"
               : absl::StrCat("a/b/c/d/e/f/", i % 3 == 0 ? "needle" : "plain",
                              i, ".cc");
    const std::string text = absl::StrCat(
        i == 0 ? "int needle = 0;\n" : "// needle\n", "// document ", i, "\n");
    const Oid oid = HashBlob(text);
    oids.push_back(oid);
    ASSERT_THAT(
        index->Add({.hint_path = hint,
                    .content = text,
                    .rank_key = ComputeRankKey(hint, text.size(), false, oid),
                    .language = DetectLanguage(hint, text)}),
        absl_testing::IsOk());
    // Several segments, and the last documents stay in the memtable.
    if (i == 19 || i == 39 || i == 49) {
      ASSERT_THAT(index->Flush(), absl_testing::IsOk());
    }
  }
  const auto state = index->GetReadState();
  ASSERT_GE(state->version->segments.size(), 2u);
  ASSERT_GT(state->mem_watermark + state->frozen.size(), 0u);

  PathTable paths;
  // The documents of each source, as their blobs and RankKeys.
  auto source_keys = [&](size_t source) {
    std::vector<std::pair<Oid, RankKey>> docs;
    auto add = [&](const SegmentDoc& doc) {
      docs.push_back(
          {doc.oid,
           RankKey{.tier = doc.tier, .prior = doc.prior, .oid = doc.oid}});
    };
    const size_t segments = state->version->segments.size();
    if (source < segments) {
      const Segment& segment = *state->version->segments[source].segment;
      for (DocId d = 0; d < segment.num_docs(); ++d) add(segment.doc(d));
    } else if (source < segments + state->frozen.size()) {
      const Memtable& mem = *state->frozen[source - segments];
      for (DocId d = 0; d < mem.published_docs(); ++d) add(mem.doc(d));
    } else {
      for (DocId d = 0; d < state->mem_watermark; ++d) {
        add(state->mem->doc(d));
      }
    }
    return docs;
  };
  const char* const globs[] = {"", "src/**", "**/*.cc", "!src/**"};
  // How many documents name the pattern in each round: few (the inverse
  // table) or nearly all (testing every document).
  const int named_one_in[] = {12, 1, 5, 12, 2};
  int built = 0;
  int small_sets = 0;
  int large_sets = 0;
  SearchExecutor executor(4);
  for (int round = 0; round < 5; ++round) {
    SCOPED_TRACE(round);
    ViewWriter writer;
    auto publish = [&](int edits) {
      std::vector<ViewEdit> batch;
      for (int e = 0; e < edits; ++e) {
        const size_t doc = rng() % oids.size();
        const bool named = rng() % named_one_in[round] == 0;
        const std::string path = absl::StrCat(
            rng() % 2 == 0 ? "src/" : "lib/", rng() % 5, "/",
            named ? "Needle_" : "item_", e, "_", rng() % 1000, ".cc");
        auto id = paths.Intern(path);
        ASSERT_TRUE(id.ok());
        batch.push_back({*id, index->blobs().Intern(oids[doc])});
        // A second path for some blobs, and sometimes the same name again in
        // another directory.
        if (rng() % 4 == 0) {
          auto copy = paths.Intern(absl::StrCat("copy/", rng() % 9, "/",
                                                named ? "needle" : "other", "_",
                                                e, ".h"));
          ASSERT_TRUE(copy.ok());
          batch.push_back({*copy, batch.back().blob.value()});
        }
      }
      writer.ApplyBatch(batch);
    };
    {
      // The shallow root is always in the view and fills the heap first.
      auto root = paths.Intern("root.cc");
      ASSERT_TRUE(root.ok());
      writer.ApplyBatch({ViewEdit{*root, index->blobs().Intern(oids[0])}});
    }
    publish(30);
    const SearchView before{writer.Snapshot(), paths};
    publish(10);  // edits after `before` was published
    const SearchView after{writer.Snapshot(), paths};
    // Every path's final blob: later edits of a path replace earlier ones.
    auto named_oids = [&](const SearchView& view, absl::string_view globtext) {
      std::optional<PathGlob> glob;
      if (!globtext.empty()) {
        auto parsed = PathGlob::Parse(globtext);
        EXPECT_TRUE(parsed.ok());
        glob = *std::move(parsed);
      }
      std::set<Oid> result;
      for (size_t doc = 0; doc < oids.size(); ++doc) {
        const BlobId blob = index->blobs().Intern(oids[doc]);
        for (const PathId id : view.state->PathsFor(blob)) {
          const std::string path = paths.path(id);
          std::string base = path.substr(path.rfind('/') + 1);
          std::transform(base.begin(), base.end(), base.begin(),
                         [](unsigned char c) { return std::tolower(c); });
          if ((!glob || glob->Matches(path)) &&
              base.find("needle") != std::string::npos) {
            result.insert(oids[doc]);
          }
        }
      }
      return result;
    };
    for (const SearchView* view : {&before, &after}) {
      for (const char* glob : globs) {
        SCOPED_TRACE(glob);
        const std::set<Oid> named = named_oids(*view, glob);
        SearchRequest request{.pattern = "needle",
                              .case_sensitive = false,
                              .path_glob = glob,
                              .max_files = 1};
        std::optional<PriorityProbe> reference;
        for (const PriorityRestriction mode :
             {PriorityRestriction::kNever, PriorityRestriction::kAuto,
              PriorityRestriction::kAlways}) {
          for (const bool parallel : {false, true}) {
            std::optional<PriorityProbe> probe;
            auto record = [&](const PriorityProbe& built_set) {
              probe = built_set;
            };
            const absl::FunctionRef<void(const PriorityProbe&)> ref(record);
            SearcherOptions options{.priority_restriction = mode,
                                    .priority_probe = &ref};
            if (parallel) options.executor = &executor;
            FERRET_ASSERT_OK_AND_ASSIGN(
                auto response, Search(*state, *view, request, options));
            if (!probe.has_value()) continue;  // The stop never needed it.
            ++built;
            // Every document is a candidate, so the restricted sets equal the
            // full one: the documents whose blob has a matching view path.
            for (size_t source = 0; source < probe->keys.size(); ++source) {
              SCOPED_TRACE(source);
              std::vector<RankKey> want;
              for (const auto& [oid, key] : source_keys(source)) {
                if (named.contains(oid)) want.push_back(key);
              }
              std::sort(want.begin(), want.end());
              EXPECT_EQ(probe->keys[source], want);
              if (mode == PriorityRestriction::kNever) {
                (named.size() * 4 <= oids.size() ? small_sets : large_sets)++;
              }
            }
            if (!reference.has_value()) reference = probe;
            EXPECT_EQ(probe->keys, reference->keys);
          }
        }
      }
    }
  }
  EXPECT_GT(built, 40);
  EXPECT_GT(small_sets, 10) << "the inverse table was never used";
  EXPECT_GT(large_sets, 10) << "the full scan was never used";
}

// A document whose view path names the pattern can outrank the heap's worst
// file through the name boost, so it is not counted only: counting-only
// decides by the view's names, not by assuming every document might match (or
// none).
TEST(SearchViewTest, CountingOnlyKeepsADocumentWithAMatchingViewName) {
  ferret_testing::TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(*Env::Default(), dir.Join("index")));
  std::unique_ptr<Index> index = std::move(opened.first);
  // One directory level is 4096 prior points and the name boost 8192, so the
  // deeper document wins by its name alone and cannot without it.
  struct Doc {
    std::string hint;
    std::string text;
  };
  // The first has four matching lines (a boost of two doublings), the others
  // one. The third always has a matching view name, so the stop test cannot
  // end the search before the second.
  const std::vector<Doc> docs = {{"a.cc",
                                  "// needle one\n// needle two\n"
                                  "// needle three\n// needle four\n"},
                                 {"d/b.cc", "// needle second\n"},
                                 {"d/e/c.cc", "// needle third\n"}};
  std::vector<Oid> oids;
  for (const Doc& doc : docs) {
    const Oid oid = HashBlob(doc.text);
    oids.push_back(oid);
    ASSERT_THAT(index->Add({.hint_path = doc.hint,
                            .content = doc.text,
                            .rank_key = ComputeRankKey(
                                doc.hint, doc.text.size(), false, oid),
                            .language = DetectLanguage(doc.hint, doc.text)}),
                absl_testing::IsOk());
  }
  ASSERT_THAT(index->Flush(), absl_testing::IsOk());
  const auto state = index->GetReadState();
  PathTable paths;
  for (const bool second_is_named : {true, false}) {
    SCOPED_TRACE(second_is_named);
    ViewWriter writer;
    FERRET_ASSERT_OK_AND_ASSIGN(PathId first, paths.Intern("x/other.cc"));
    FERRET_ASSERT_OK_AND_ASSIGN(
        PathId second,
        paths.Intern(second_is_named ? "x/needle_b.cc" : "x/plain_b.cc"));
    FERRET_ASSERT_OK_AND_ASSIGN(PathId third, paths.Intern("x/needle_c.cc"));
    writer.ApplyBatch({ViewEdit{first, index->blobs().Intern(oids[0])},
                       ViewEdit{second, index->blobs().Intern(oids[1])},
                       ViewEdit{third, index->blobs().Intern(oids[2])}});
    const SearchView view{writer.Snapshot(), paths};
    SearchRequest request{.pattern = "needle", .max_files = 1};
    request.exact_counts = true;
    FERRET_ASSERT_OK_AND_ASSIGN(auto exact, Search(*state, view, request));
    request.exact_counts = false;
    for (const int64_t min_bytes : {int64_t{0}, int64_t{1} << 20}) {
      SCOPED_TRACE(min_bytes);
      SearcherOptions options;
      options.count_only_min_bytes = min_bytes;
      FERRET_ASSERT_OK_AND_ASSIGN(auto got,
                                  Search(*state, view, request, options));
      ExpectFilesEqual(got, exact);
      ASSERT_EQ(got.files.size(), 1u);
      EXPECT_EQ(got.files[0].path,
                second_is_named ? "x/needle_b.cc" : "x/other.cc");
      // With no name to earn the boost the second document is counted only.
      if (min_bytes == 0 && !second_is_named) {
        EXPECT_GE(got.stats.count_only_documents, 1);
      }
    }
  }
}

}  // namespace
}  // namespace ferret

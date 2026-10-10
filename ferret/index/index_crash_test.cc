// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/base/testing/fault_injection_env.h"
#include "ferret/git/blob_hash.h"
#include "ferret/index/blob_registry.h"
#include "ferret/index/index.h"
#include "ferret/index/manifest.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/read_state.h"
#include "ferret/index/segment.h"
#include "ferret/index/segment_writer.h"
#include "ferret/index/version_set.h"
#include "ferret/postings/ids.h"
#include "ferret/search/search_types.h"
#include "ferret/search/searcher.h"
#include "ferret/symbols/symbol_extractor.h"
#include "ferret/testing/brute_force_search.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/testing/test_seed.h"
#include "ferret/text/language.h"

namespace ferret {
namespace {

using ::ferret_testing::CorpusFile;
using ::ferret_testing::FaultInjectionEnv;
using ::ferret_testing::TempDir;

SegmentDocument CrashDocument(const CorpusFile& file) {
  return {.hint_path = file.path,
          .content = file.content,
          .rank_key = ComputeRankKey(file.path, file.content.size(), false,
                                     HashBlob(file.content))};
}

// The MANIFEST is the durability point for a flush, direct segment, or view
// edit. Add and AdjustReferences only publish process-local state.
struct CrashPrefix {
  std::vector<CorpusFile> seen;
  std::vector<CorpusFile> live;
  std::vector<ManifestView> views;
};

const ManifestView kCrashView{"/canonical/root", "/canonical/common"};

// Returns every acknowledged prefix; a failed install can nevertheless have
// committed its whole MANIFEST entry before the injected failure was reported.
std::vector<CrashPrefix> CrashWorkload(Index& index, std::mt19937_64& rng,
                                       int doc_count,
                                       std::vector<uint64_t>* completed,
                                       FaultInjectionEnv* fault = nullptr,
                                       int64_t boundary = 0) {
  std::vector<CrashPrefix> prefixes(1);
  size_t wait_index = 0;
  const auto wait = [&] {
    if (fault == nullptr) {
      index.WaitForCompactions();
      completed->push_back(index.Stats().completed_compactions);
      return true;
    }
    const uint64_t target = (*completed)[wait_index++];
    // A permanent injected I/O failure makes WaitForCompactions retry forever.
    // Poll its completion counter only until the first failing mutation.
    const absl::Time deadline = absl::Now() + absl::Milliseconds(100);
    while (index.Stats().completed_compactions < target &&
           fault->mutation_count() <= boundary && absl::Now() < deadline) {
      absl::SleepFor(absl::Milliseconds(1));
    }
    return fault->mutation_count() <= boundary;
  };
  const auto publish = [&] { prefixes.push_back(prefixes.back()); };
  if (!index.RecordView(kCrashView).ok()) return prefixes;
  // Durability point: RecordView's MANIFEST install.
  publish();
  prefixes.back().views.push_back(kCrashView);
  std::vector<CorpusFile> revisions;
  index.ResumeGc();
  for (int i = 0; i < doc_count; ++i) {
    // Suspension spans replacements, removals, and flushes rather than merely
    // toggling GC between two operations.
    if (i % 4 == 1) index.SuspendGc();
    CorpusFile file{
        absl::StrCat("revision", i, ".cc"),
        absl::StrCat("crashneedle", rng() % 1000000, " revision", i, "\n")};
    if (i > 0 && i % 3 == 1) file.path = revisions[i - 1].path;
    const SegmentDocument doc = CrashDocument(file);
    const BlobId blob = index.blobs().Intern(doc.rank_key.oid);
    if (!index.AdjustReferences({{blob, 1}}).ok()) return prefixes;
    if (i % 3 == 2) {
      if (!index
               .AddSegment(
                   [&](SegmentWriter& writer) { return writer.Add(doc); })
               .ok()) {
        return prefixes;
      }
    } else if (!index.Add(doc).ok() || !index.Flush().ok()) {
      return prefixes;
    }
    // Durability point: Flush or AddSegment installs a complete MANIFEST.
    publish();
    prefixes.back().seen.push_back(file);
    prefixes.back().live.push_back(file);
    revisions.push_back(file);
    if (i > 0 && i % 3 == 1) {
      const CorpusFile& old = revisions[i - 1];
      const BlobId obsolete =
          index.blobs().Intern(CrashDocument(old).rank_key.oid);
      if (!index.AdjustReferences({{obsolete, -1}}).ok()) return prefixes;
      std::erase_if(prefixes.back().live, [&](const CorpusFile& candidate) {
        return candidate.path == old.path;
      });
    }
    if (i % 4 == 2) index.ResumeGc();
    if (i % 3 == 2) {
      if (!wait()) return prefixes;
      // Durability point: a GC or tier compaction's MANIFEST install can
      // discard zero-reference documents and retire its input files.
      publish();
    }
  }
  index.ResumeGc();
  if (!wait()) return prefixes;
  publish();
  if (!index.ForgetView(kCrashView.canonical_root).ok()) return prefixes;
  // Durability point: ForgetView's MANIFEST install.
  publish();
  prefixes.back().views.clear();
  return prefixes;
}

std::vector<CorpusFile> SegmentContents(const ReadState& state) {
  std::vector<CorpusFile> docs;
  for (const VersionSegment& entry : state.version->segments) {
    for (DocId id = 0; id < entry.segment->num_docs(); ++id) {
      const SegmentDoc doc = entry.segment->doc(id);
      docs.push_back({std::string(doc.hint_path), std::string(doc.content)});
    }
  }
  return docs;
}

void CompareSearch(const Index& index,
                   const std::vector<CorpusFile>& expected) {
  for (const SearchRequest& request :
       {SearchRequest{.pattern = "crashneedle", .max_files = 100},
        SearchRequest{
            .pattern = "crashneedle[0-9]+", .regex = true, .max_files = 100}}) {
    auto actual = Search(*index.GetReadState(), request);
    auto oracle = ferret_testing::BruteForceSearch(expected, request);
    ASSERT_TRUE(actual.ok()) << actual.status();
    ASSERT_TRUE(oracle.ok()) << oracle.status();
    ASSERT_EQ(actual->files.size(), oracle->size());
    std::vector<std::pair<std::string, int64_t>> got;
    std::vector<std::pair<std::string, int64_t>> want;
    for (const FileResult& file : actual->files) {
      got.emplace_back(file.path, file.num_matching_lines);
    }
    for (const FileResult& file : *oracle) {
      want.emplace_back(file.path, file.num_matching_lines);
    }
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    EXPECT_EQ(got, want);
  }
}

TEST(IndexCrashTest, DurableViewProtectsBlobDuringZeroGraceCompaction) {
  TempDir temp;
  const std::string path = temp.Join("index");
  IndexOptions options;
  options.gc_grace_period = absl::ZeroDuration();
  options.compaction.tier_min_segments = 2;
  auto opened = Index::Open(*Env::Default(), path, options);
  ASSERT_TRUE(opened.ok()) << opened.status();
  std::unique_ptr<Index> index = std::move(opened->first);
  ASSERT_TRUE(index->RecordView(kCrashView).ok());
  const CorpusFile protected_file{"protected.cc", "crashneedleprotected\n"};
  const BlobId protected_blob =
      index->blobs().Intern(CrashDocument(protected_file).rank_key.oid);
  ASSERT_TRUE(index->AdjustReferences({{protected_blob, 1}}).ok());
  ASSERT_TRUE(index->Add(CrashDocument(protected_file)).ok());
  ASSERT_TRUE(index->Flush().ok());
  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(index
                    ->Add(CrashDocument({absl::StrCat("obsolete", i, ".cc"),
                                         absl::StrCat("transient ", i, "\n")}))
                    .ok());
    ASSERT_TRUE(index->Flush().ok());
  }
  index->ResumeGc();
  index->WaitForCompactions();
  EXPECT_GT(index->Stats().completed_compactions, 0u);
  CompareSearch(*index, {protected_file});
  index.reset();
  auto recovered = Index::Open(*Env::Default(), path, options);
  ASSERT_TRUE(recovered.ok()) << recovered.status();
  ASSERT_EQ(recovered->first->Views(), std::vector<ManifestView>{kCrashView});
  const BlobId restored = recovered->first->blobs().Intern(
      CrashDocument(protected_file).rank_key.oid);
  ASSERT_TRUE(recovered->first->AdjustReferences({{restored, 1}}).ok());
  recovered->first->ResumeGc();
  recovered->first->WaitForCompactions();
  CompareSearch(*recovered->first, {protected_file});
  EXPECT_TRUE(recovered->second.dropped_segments.empty());
}

TEST(IndexCrashTest, EverySmallMutationAndSeededLargeSample) {
  const uint64_t seed = ferret_testing::TestSeed(717);
  SCOPED_TRACE(seed);
  IndexOptions options;
  options.compaction.tier_min_segments = 2;
  options.gc_grace_period = absl::ZeroDuration();
  for (const int doc_count : {3, 50}) {
    SCOPED_TRACE(doc_count);
    int64_t mutations = 0;
    std::vector<CrashPrefix> all;
    std::vector<uint64_t> completed;
    {
      TempDir calibration;
      FaultInjectionEnv env(Env::Default());
      auto opened = Index::Open(env, calibration.Join("index"), options);
      ASSERT_TRUE(opened.ok()) << opened.status();
      std::mt19937_64 rng(seed);
      const int64_t before = env.mutation_count();
      all = CrashWorkload(*opened->first, rng, doc_count, &completed);
      ASSERT_TRUE(all.back().views.empty());
      ASSERT_GT(opened->first->Stats().completed_compactions, 0u);
      // Destruction drains the reaper. Include its final deletes in the
      // calibrated boundary range, not only the foreground MANIFEST writes.
      opened->first.reset();
      mutations = env.mutation_count() - before;
    }
    ASSERT_GT(mutations, 0);
    std::vector<int64_t> boundaries;
    if (doc_count == 3) {
      for (int64_t i = 0; i <= mutations; ++i) boundaries.push_back(i);
    }
    // The larger schedule samples 40 mutation boundaries, plus the first and
    // last, independently of the index format's byte count.
    std::mt19937_64 selection(seed);
    for (int i = 0; i < 40; ++i) {
      boundaries.push_back(static_cast<int64_t>(selection() % (mutations + 1)));
    }
    boundaries.push_back(0);
    boundaries.push_back(mutations);
    std::sort(boundaries.begin(), boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                     boundaries.end());
    for (const int64_t boundary : boundaries) {
      SCOPED_TRACE(
          absl::StrCat("Env mutation after workload start: ", boundary));
      TempDir temp;
      FaultInjectionEnv env(Env::Default());
      const std::string path = temp.Join("index");
      std::vector<CrashPrefix> acknowledged;
      {
        auto opened = Index::Open(env, path, options);
        ASSERT_TRUE(opened.ok()) << opened.status();
        std::mt19937_64 rng(seed);
        // Every Env mutation from here is a crash point: segment data/sync,
        // MANIFEST append/sync, directory sync, or obsolete-file cleanup.
        env.FailAfter(boundary, absl::UnavailableError("simulated power loss"));
        acknowledged = CrashWorkload(*opened->first, rng, doc_count, &completed,
                                     &env, env.mutation_count() + boundary);
      }
      ASSERT_TRUE(env.DropUnsyncedData().ok());
      IndexOptions cleanup_options = options;
      // Prefer GC rewrites to unrelated tier merges during post-crash cleanup.
      cleanup_options.compaction.tier_min_segments = 1000;
      cleanup_options.compaction.rewrite_dead_percent = 1;
      auto recovered = Index::Open(env, path, cleanup_options);
      ASSERT_TRUE(recovered.ok()) << recovered.status();
      EXPECT_TRUE(recovered->second.dropped_segments.empty());
      const std::vector<CorpusFile> docs =
          SegmentContents(*recovered->first->GetReadState());
      // The last attempted MANIFEST install may have committed despite its
      // error. No earlier acknowledged prefix may disappear, and no partial
      // segment or view edit may appear.
      for (const CorpusFile& live : acknowledged.back().live) {
        EXPECT_TRUE(std::any_of(docs.begin(), docs.end(),
                                [&](const CorpusFile& candidate) {
                                  return candidate.content == live.content;
                                }))
            << "collected live document: " << live.path;
      }
      // A GC install can finish between any two foreground installs. Dead
      // documents can still be present, but future documents cannot appear,
      // and no document with a live reference may have been collected.
      // At most one foreground install may have committed without returning.
      // Background GC installs do not introduce new document contents.
      const auto last = all.begin() + static_cast<std::ptrdiff_t>(std::min(
                                          all.size(), acknowledged.size() + 1));
      const auto match = std::find_if(
          all.begin() + static_cast<std::ptrdiff_t>(acknowledged.size() - 1),
          last, [&](const CrashPrefix& prefix) {
            if (prefix.views != recovered->first->Views()) return false;
            for (const CorpusFile& doc : docs) {
              if (std::none_of(prefix.seen.begin(), prefix.seen.end(),
                               [&](const CorpusFile& candidate) {
                                 return candidate.path == doc.path &&
                                        candidate.content == doc.content;
                               })) {
                return false;
              }
            }
            for (const CorpusFile& live : prefix.live) {
              if (std::none_of(docs.begin(), docs.end(),
                               [&](const CorpusFile& candidate) {
                                 return candidate.content == live.content;
                               })) {
                return false;
              }
            }
            return true;
          });
      ASSERT_NE(match, last);
      CompareSearch(*recovered->first, docs);
      for (const VersionSegment& segment :
           recovered->first->GetReadState()->version->segments) {
        const std::string filename =
            absl::StrFormat("%06u.fseg", segment.metadata.file_number);
        auto exists = env.FileExists(absl::StrCat(path, "/", filename));
        ASSERT_TRUE(exists.ok()) << exists.status();
        EXPECT_TRUE(*exists) << filename;
      }
      // References are process-local. Restore the durable view's references
      // before resuming GC, then remove dead revisions by zero-grace GC.
      for (const CorpusFile& file : match->live) {
        const BlobId blob =
            recovered->first->blobs().Intern(CrashDocument(file).rank_key.oid);
        ASSERT_TRUE(recovered->first->AdjustReferences({{blob, 1}}).ok());
      }
      recovered->first->ResumeGc();
      recovered->first->WaitForCompactions();
      // A merge can finish before the queued GC rewrite is chosen. A fresh
      // zero-grace scan after the merge must collect its remaining dead docs.
      for (int pass = 0; pass < 3 && recovered->first->Stats().dead_docs != 0;
           ++pass) {
        recovered->first->SuspendGc();
        recovered->first->ResumeGc();
        recovered->first->WaitForCompactions();
      }
      EXPECT_EQ(recovered->first->Stats().dead_docs, 0u)
          << "boundary=" << boundary
          << " compactions=" << recovered->first->Stats().completed_compactions
          << " failures=" << recovered->first->Stats().compaction_failures
          << " error=" << recovered->first->Stats().last_compaction_error;
      CompareSearch(*recovered->first, match->live);
      recovered->first.reset();
      auto clean = Index::Open(env, path, cleanup_options);
      ASSERT_TRUE(clean.ok()) << clean.status();
      EXPECT_TRUE(clean->second.deleted_files.empty());
      EXPECT_TRUE(clean->second.dropped_segments.empty());
    }
  }
}

TEST(IndexCrashTest, DurableSymbolsAreWholeAfterRecovery) {
  TempDir temp;
  const std::string path = temp.Join("index");
  auto opened = Index::Open(*Env::Default(), path);
  ASSERT_TRUE(opened.ok());
  std::unique_ptr<Index> index = std::move(opened->first);
  const CorpusFile file{"durable.cc", "int Durable(int x) { return x; }\n"};
  SegmentDocument doc = CrashDocument(file);
  doc.language = Language::kCpp;
  ASSERT_TRUE(index->Add(doc).ok());
  ASSERT_TRUE(index->Flush().ok());
  index.reset();
  auto recovered = Index::Open(*Env::Default(), path);
  ASSERT_TRUE(recovered.ok());
  SymbolExtractor extractor;
  auto expected = extractor.Extract(doc.language, file.content);
  ASSERT_TRUE(expected.ok());
  for (const ExtractedSymbol& symbol : *expected) {
    const SymbolLookup found =
        recovered->first->GetReadState()->FindSymbol(symbol.name, false);
    ASSERT_EQ(found.occurrences.size(), 1u);
    EXPECT_EQ(found.occurrences[0].line, symbol.line);
    EXPECT_EQ(found.occurrences[0].kind, symbol.kind);
    EXPECT_EQ(found.occurrences[0].is_definition, symbol.is_definition);
    EXPECT_TRUE(found.pending_blobs.empty());
  }
}

}  // namespace
}  // namespace ferret

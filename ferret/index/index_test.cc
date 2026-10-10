// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/index.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "re2/re2.h"

#include "ferret/base/env.h"
#include "ferret/base/mapped_file.h"
#include "ferret/base/testing/fault_injection_env.h"
#include "ferret/git/blob_hash.h"
#include "ferret/index/blob_registry.h"
#include "ferret/index/bulk_build.h"
#include "ferret/index/manifest.h"
#include "ferret/index/memtable.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/read_state.h"
#include "ferret/index/segment.h"
#include "ferret/index/segment_writer.h"
#include "ferret/index/version_set.h"
#include "ferret/postings/ids.h"
#include "ferret/regex/compile.h"
#include "ferret/search/ranking.h"
#include "ferret/search/search_types.h"
#include "ferret/search/searcher.h"
#include "ferret/testing/brute_force_search.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/testing/test_seed.h"
#include "ferret/text/language.h"
#include "ferret/text/lexer.h"
#include "ferret/text/literal_finder.h"

namespace ferret {
namespace {

using ::absl_testing::IsOk;
using ::ferret_testing::CorpusFile;
using ::ferret_testing::TempDir;

SegmentDocument Document(const CorpusFile& file) {
  return {.hint_path = file.path,
          .content = file.content,
          .rank_key = ComputeRankKey(file.path, file.content.size(), false,
                                     HashBlob(file.content)),
          .language = DetectLanguage(file.path, file.content)};
}

// BruteForceSearch deliberately omits rank fields. Compare the observable
// matches, including which lines survived the global line limit.
void Compare(const ReadState& state, const std::vector<CorpusFile>& corpus,
             SearchRequest request) {
  request.exact_counts = true;
  auto actual = Search(state, request);
  ASSERT_THAT(actual, IsOk());
  std::vector<CorpusFile> ordered = corpus;
  std::sort(ordered.begin(), ordered.end(),
            [](const CorpusFile& a, const CorpusFile& b) {
              return Document(a).rank_key < Document(b).rank_key;
            });
  auto expected = ferret_testing::BruteForceSearch(ordered, request);
  ASSERT_THAT(expected, IsOk());
  const int64_t matching_files = std::ssize(*expected);
  int64_t matching_lines = 0;
  for (const FileResult& file : *expected) {
    matching_lines += file.num_matching_lines;
  }
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto pattern, CompilePattern({.pattern = request.pattern,
                                    .literal = !request.regex,
                                    .case_sensitive = request.case_sensitive}));
  int64_t lines = 0;
  for (FileResult& file : *expected) {
    const auto entry = std::find_if(
        corpus.begin(), corpus.end(),
        [&](const CorpusFile& source) { return source.path == file.path; });
    ASSERT_NE(entry, corpus.end());
    const SegmentDocument doc = Document(*entry);
    file.tier = doc.rank_key.tier;
    file.prior = doc.rank_key.prior;
    file.oid = doc.rank_key.oid;
    const absl::string_view name =
        absl::string_view(file.path).substr(file.path.rfind('/') + 1);
    file.name_matches = RE2::PartialMatch(name, pattern->matcher());
    MatchClassifier classifier(doc.language, doc.content, LexState::kCode);
    for (ResultLine& line : file.lines) {
      if (!line.is_match) continue;
      size_t offset = 0;
      for (int64_t number = 1; number < line.number; ++number) {
        offset = doc.content.find('\n', offset) + 1;
      }
      line.match_class = classifier.Classify(offset + line.match_begin,
                                             offset + line.match_end);
    }
    file.lines.erase(std::remove_if(file.lines.begin(), file.lines.end(),
                                    [&](const ResultLine& line) {
                                      if (!line.is_match) return false;
                                      return ++lines > request.max_lines;
                                    }),
                     file.lines.end());
  }
  expected->erase(std::remove_if(expected->begin(), expected->end(),
                                 [](const FileResult& file) {
                                   return std::none_of(
                                       file.lines.begin(), file.lines.end(),
                                       [](const ResultLine& line) {
                                         return line.is_match;
                                       });
                                 }),
                  expected->end());
  RankFiles(*expected);
  if (std::ssize(*expected) > request.max_files) {
    expected->resize(static_cast<size_t>(request.max_files));
  }
  EXPECT_TRUE(actual->complete);
  EXPECT_EQ(actual->num_matching_files, matching_files) << request.pattern;
  EXPECT_EQ(actual->num_matching_lines, matching_lines) << request.pattern;
  ASSERT_EQ(actual->files.size(), expected->size()) << request.pattern;
  for (size_t i = 0; i < expected->size(); ++i) {
    SCOPED_TRACE(request.pattern);
    EXPECT_EQ(actual->files[i].path, (*expected)[i].path);
    EXPECT_EQ(actual->files[i].num_matching_lines,
              (*expected)[i].num_matching_lines);
    // Context and offsets are tested separately by SearchSegment's tests.
    std::vector<int64_t> got, want;
    for (const ResultLine& line : actual->files[i].lines) {
      if (line.is_match) got.push_back(line.number);
    }
    for (const ResultLine& line : (*expected)[i].lines) {
      if (line.is_match) want.push_back(line.number);
    }
    EXPECT_EQ(got, want);
  }
}

void CheckBlobMappings(const ReadState& state) {
  for (const VersionSegment& source : state.version->segments) {
    ASSERT_NE(source.blob_of_doc, nullptr);
    ASSERT_EQ(source.blob_of_doc->size(), source.segment->num_docs());
    for (DocId id = 0; id < source.segment->num_docs(); ++id) {
      EXPECT_EQ(state.blobs->oid((*source.blob_of_doc)[id]),
                source.segment->doc(id).oid);
    }
    // The inverse table lists every document under its blob.
    ASSERT_NE(source.docs_of_blob, nullptr);
    for (DocId id = 0; id < source.segment->num_docs(); ++id) {
      const absl::Span<const DocId> docs =
          source.docs_of_blob->Docs((*source.blob_of_doc)[id]);
      EXPECT_TRUE(std::binary_search(docs.begin(), docs.end(), id)) << id;
    }
  }
  const auto check_mem = [&](const Memtable& mem, uint32_t watermark) {
    for (DocId id = 0; id < watermark; ++id) {
      EXPECT_EQ(state.blobs->oid(mem.blob_of_doc(id)), mem.doc(id).oid);
    }
  };
  for (const auto& frozen : state.frozen) {
    check_mem(*frozen, frozen->published_docs());
  }
  check_mem(*state.mem, state.mem_watermark);
}

bool ContainsBlob(const ReadState& state, const Oid& oid) {
  for (const VersionSegment& entry : state.version->segments) {
    for (DocId doc = 0; doc < entry.segment->num_docs(); ++doc) {
      if (entry.segment->doc(doc).oid == oid) return true;
    }
  }
  for (const auto& mem : state.frozen) {
    if (mem->FindDoc(oid, mem->published_docs())) return true;
  }
  return state.mem->FindDoc(oid, state.mem_watermark).has_value();
}

void ExpectGcSearch(const Index& index, absl::string_view token) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto found,
      Search(*index.GetReadState(), {.pattern = std::string(token)}));
  EXPECT_FALSE(found.files.empty()) << token;
}

TEST(IndexTest, GcSuspendsAcrossReopenAndRejectsInvalidBatches) {
  TempDir dir;
  Env& env = *Env::Default();
  IndexOptions options;
  options.gc_grace_period = absl::ZeroDuration();
  const CorpusFile file{.path = "a.cc", .content = "uniquegctoken\n"};
  const Oid oid = HashBlob(file.content);
  {
    FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                                Index::Open(env, dir.path(), options));
    auto index = std::move(opened.first);
    ASSERT_THAT(index->Add(Document(file)), IsOk());
    ASSERT_THAT(index->Flush(), IsOk());
    EXPECT_TRUE(ContainsBlob(*index->GetReadState(), oid));
  }
  FERRET_ASSERT_OK_AND_ASSIGN(auto reopened,
                              Index::Open(env, dir.path(), options));
  auto index = std::move(reopened.first);
  const BlobId blob = index->blobs().Intern(oid);
  // The first flush after recovery must not drop unresynced views.
  ASSERT_THAT(
      index->Add(Document({.path = "b.cc", .content = "othergctoken\n"})),
      IsOk());
  ASSERT_THAT(index->Flush(), IsOk());
  EXPECT_TRUE(ContainsBlob(*index->GetReadState(), oid));
  const std::vector<Index::ReferenceDelta> invalid = {{blob, 1}, {blob, -2}};
  // An invalid batch is atomic, including its positive deltas.
  EXPECT_FALSE(index->AdjustReferences(invalid).ok());
  EXPECT_FALSE(index->AdjustReferences({{blob, -1}}).ok());
  ASSERT_THAT(index->AdjustReferences({{blob, 1}}), IsOk());
  index->ResumeGc();
  EXPECT_TRUE(ContainsBlob(*index->GetReadState(), oid));
  EXPECT_GE(index->Stats().live_docs, uint64_t{1});
  index->SuspendGc();
  ASSERT_THAT(index->AdjustReferences({{blob, -1}}), IsOk());
  EXPECT_EQ(index->Stats().dead_docs, uint64_t{0});
}

TEST(IndexTest, GcRestoresDocumentsBeforeCompactionInstall) {
  TempDir dir;
  Env& env = *Env::Default();
  auto gate = std::make_shared<absl::Notification>();
  auto reached = std::make_shared<absl::Notification>();
  IndexOptions options;
  options.gc_grace_period = absl::ZeroDuration();
  options.compaction.tier_min_segments = 20;
  options.compaction_install_gate = gate;
  options.compaction_install_reached = reached;
  const CorpusFile dead{.path = "dead.cc", .content = "resurrecttoken\n"};
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(env, dir.path(), options));
  auto index = std::move(opened.first);
  const Oid oid = HashBlob(dead.content);
  ASSERT_THAT(index->Add(Document(dead)), IsOk());
  ASSERT_THAT(index->Flush(), IsOk());
  const BlobId blob = index->blobs().Intern(oid);
  // The gate stops after the drop set is fixed but before publication.
  ASSERT_THAT(
      index->Add(Document({.path = "other.cc", .content = "othercontent\n"})),
      IsOk());
  ASSERT_THAT(index->Flush(), IsOk());
  index->ResumeGc();
  EXPECT_TRUE(ContainsBlob(*index->GetReadState(), oid));
  ExpectGcSearch(*index, dead.content.substr(0, dead.content.size() - 1));
  ASSERT_TRUE(reached->WaitForNotificationWithTimeout(absl::Seconds(10)));
  ASSERT_THAT(index->AdjustReferences({{blob, 1}}), IsOk());
  EXPECT_TRUE(ContainsBlob(*index->GetReadState(), oid));
  ExpectGcSearch(*index, dead.content.substr(0, dead.content.size() - 1));
  gate->Notify();
  index->WaitForCompactions();
  EXPECT_TRUE(ContainsBlob(*index->GetReadState(), oid));
  ExpectGcSearch(*index, dead.content.substr(0, dead.content.size() - 1));
  EXPECT_TRUE(
      index->GetReadState()
          ->mem->FindDoc(oid, index->GetReadState()->mem->published_docs())
          .has_value());
  EXPECT_EQ(index->Stats().dead_docs, uint64_t{0});
}

TEST(IndexTest, GcRestoresDocumentsBeforeFlushInstall) {
  TempDir dir;
  Env& env = *Env::Default();
  auto gate = std::make_shared<absl::Notification>();
  auto reached = std::make_shared<absl::Notification>();
  IndexOptions options;
  options.compaction.enabled = false;
  options.gc_grace_period = absl::ZeroDuration();
  options.flush_install_gate = gate;
  options.flush_install_reached = reached;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(env, dir.path(), options));
  auto index = std::move(opened.first);
  const CorpusFile file{.path = "flush.cc", .content = "flushresurrecttoken\n"};
  const Oid oid = HashBlob(file.content);
  const BlobId blob = index->blobs().Intern(oid);
  index->ResumeGc();
  ASSERT_THAT(index->Add(Document(file)), IsOk());
  absl::Status flush_status;
  std::thread flush([&] { flush_status = index->Flush(); });
  if (!reached->WaitForNotificationWithTimeout(absl::Seconds(10))) {
    gate->Notify();
    flush.join();
    FAIL() << "flush did not reach install";
  }
  ASSERT_THAT(index->AdjustReferences({{blob, 1}}), IsOk());
  EXPECT_TRUE(ContainsBlob(*index->GetReadState(), oid));
  ExpectGcSearch(*index, file.content.substr(0, file.content.size() - 1));
  gate->Notify();
  flush.join();
  ASSERT_THAT(flush_status, IsOk());
  EXPECT_TRUE(ContainsBlob(*index->GetReadState(), oid));
  ExpectGcSearch(*index, file.content.substr(0, file.content.size() - 1));
  EXPECT_TRUE(
      index->GetReadState()
          ->mem->FindDoc(oid, index->GetReadState()->mem->published_docs())
          .has_value());
}

// Blocks the MANIFEST rename inside Install, not segment output writing.
class GatedManifestEnv final : public Env {
 public:
  explicit GatedManifestEnv(Env& base) : base_(base) {}
  void Arm() { armed_.store(true); }
  absl::Notification reached;
  absl::Notification release;

  absl::StatusOr<std::unique_ptr<WritableFile>> NewWritableFile(
      absl::string_view path) override {
    return base_.NewWritableFile(path);
  }
  absl::StatusOr<std::string> ReadFile(absl::string_view path) override {
    return base_.ReadFile(path);
  }
  absl::StatusOr<std::unique_ptr<MappedFile>> NewMappedFile(
      absl::string_view path) override {
    return base_.NewMappedFile(path);
  }
  absl::StatusOr<int64_t> GetFileSize(absl::string_view path) override {
    return base_.GetFileSize(path);
  }
  absl::StatusOr<bool> FileExists(absl::string_view path) override {
    return base_.FileExists(path);
  }
  absl::StatusOr<std::vector<std::string>> GetChildren(
      absl::string_view path) override {
    return base_.GetChildren(path);
  }
  absl::Status CreateDir(absl::string_view path) override {
    return base_.CreateDir(path);
  }
  absl::Status DeleteFile(absl::string_view path) override {
    return base_.DeleteFile(path);
  }
  absl::Status RenameFile(absl::string_view from,
                          absl::string_view to) override {
    if (armed_.load() && from.ends_with("/MANIFEST.tmp")) {
      reached.Notify();
      release.WaitForNotification();
    }
    return base_.RenameFile(from, to);
  }
  absl::Status SyncDir(absl::string_view path) override {
    return base_.SyncDir(path);
  }
  absl::StatusOr<std::unique_ptr<FileLock>> LockFile(
      absl::string_view path) override {
    return base_.LockFile(path);
  }

 private:
  Env& base_;
  std::atomic<bool> armed_{false};
};

TEST(IndexTest, ManifestInstallDoesNotBlockReferenceEditsOrReaders) {
  for (bool compact : {false, true}) {
    SCOPED_TRACE(compact);
    TempDir dir;
    GatedManifestEnv env(*Env::Default());
    IndexOptions options;
    options.gc_grace_period = absl::ZeroDuration();
    options.compaction.tier_min_segments = 20;
    FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                                Index::Open(env, dir.path(), options));
    auto index = std::move(opened.first);
    const CorpusFile file{"live.cc", "manifest-live\n"};
    ASSERT_THAT(index->Add(Document(file)), IsOk());
    ASSERT_THAT(index->Flush(), IsOk());
    const BlobId blob = index->blobs().Intern(HashBlob(file.content));
    ASSERT_THAT(index->AdjustReferences({{blob, 1}}), IsOk());
    if (compact) {
      ASSERT_THAT(index->Add(Document({"dead.cc", "manifest-dead\n"})), IsOk());
      ASSERT_THAT(index->Flush(), IsOk());
      env.Arm();
      index->ResumeGc();
      // Force a rewrite while retaining the referenced input.
      ASSERT_THAT(index->AdjustReferences({{blob, -1}}), IsOk());
    } else {
      ASSERT_THAT(index->Add(Document({"new.cc", "new-manifest\n"})), IsOk());
    }
    if (!compact) env.Arm();
    absl::Status flush_status;
    std::thread flush;
    if (!compact) flush = std::thread([&] { flush_status = index->Flush(); });
    const bool reached =
        env.reached.WaitForNotificationWithTimeout(absl::Seconds(10));
    absl::Notification accessed;
    absl::Status edit_status;
    std::thread access;
    if (reached) {
      access = std::thread([&] {
        edit_status = index->AdjustReferences({{blob, 1}});
        static_cast<void>(index->GetReadState());
        accessed.Notify();
      });
    }
    const bool responsive =
        reached && accessed.WaitForNotificationWithTimeout(absl::Seconds(2));
    env.release.Notify();
    if (access.joinable()) access.join();
    if (flush.joinable()) flush.join();
    EXPECT_TRUE(reached);
    EXPECT_TRUE(responsive);
    if (reached) EXPECT_THAT(edit_status, IsOk());
    if (!compact) EXPECT_THAT(flush_status, IsOk());
    index->WaitForCompactions();
    EXPECT_TRUE(ContainsBlob(*index->GetReadState(), HashBlob(file.content)));
  }
}

TEST(IndexTest, GcRetainsDocumentsTooLargeToRestore) {
  TempDir dir;
  auto gate = std::make_shared<absl::Notification>();
  auto reached = std::make_shared<absl::Notification>();
  IndexOptions options;
  options.memtable.max_content_bytes = 2048;
  options.memtable.flush_bytes = 1024;
  options.compaction.tier_min_segments = 20;
  options.gc_grace_period = absl::ZeroDuration();
  options.compaction_install_gate = gate;
  options.compaction_install_reached = reached;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened, Index::Open(*Env::Default(), dir.path(), options));
  auto index = std::move(opened.first);
  const CorpusFile file{"big.cc", std::string(4096, 'x') + "restore-needle\n"};
  ASSERT_THAT(index->AddSegment([&](SegmentWriter& writer) {
    return writer.Add(Document(file));
  }),
              IsOk());
  ASSERT_THAT(index->AddSegment([&](SegmentWriter& writer) {
    return writer.Add(Document({"small.cc", "small-dead\n"}));
  }),
              IsOk());
  const BlobId blob = index->blobs().Intern(HashBlob(file.content));
  index->ResumeGc();
  if (!reached->WaitForNotificationWithTimeout(absl::Seconds(10))) {
    gate->Notify();
    FAIL() << "compaction never reached install";
  }
  ASSERT_THAT(index->AdjustReferences({{blob, 1}}), IsOk());
  gate->Notify();
  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  while (index->Stats().completed_compactions == 0 && absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(5));
  }
  index->WaitForCompactions();
  EXPECT_EQ(index->Stats().completed_compactions, 1u);
  EXPECT_EQ(index->Stats().compaction_failures, 0u);
  EXPECT_FALSE(ContainsBlob(*index->GetReadState(), HashBlob("small-dead\n")));
  EXPECT_TRUE(ContainsBlob(*index->GetReadState(), HashBlob(file.content)));
  ExpectGcSearch(*index, "restore-needle");
  index.reset();
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto recovered, Index::Open(*Env::Default(), dir.path(), options));
  EXPECT_TRUE(
      ContainsBlob(*recovered.first->GetReadState(), HashBlob(file.content)));
}

TEST(IndexTest, GcSkipsStaleDeadlinesAndSuspendedScans) {
  TempDir dir;
  IndexOptions options;
  options.gc_grace_period = absl::Milliseconds(80);
  options.compaction.tier_min_segments = 20;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened, Index::Open(*Env::Default(), dir.path(), options));
  auto index = std::move(opened.first);
  const CorpusFile file{"stale.cc", "stale-deadline\n"};
  const BlobId blob = index->blobs().Intern(HashBlob(file.content));
  ASSERT_THAT(index->AdjustReferences({{blob, 1}}), IsOk());
  ASSERT_THAT(index->Add(Document(file)), IsOk());
  ASSERT_THAT(index->Flush(), IsOk());
  EXPECT_EQ(index->Stats().gc_scans, 0u);
  index->ResumeGc();
  absl::SleepFor(absl::Milliseconds(110));
  EXPECT_EQ(index->Stats().gc_scans, 0u);
  ASSERT_THAT(index->AdjustReferences({{blob, -1}}), IsOk());
  ASSERT_THAT(index->AdjustReferences({{blob, 1}}), IsOk());
  absl::SleepFor(absl::Milliseconds(110));
  EXPECT_EQ(index->Stats().gc_scans, 0u);
  ASSERT_THAT(index->AdjustReferences({{blob, -1}}), IsOk());
  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  while (index->Stats().gc_scans == 0 && absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(5));
  }
  ASSERT_GT(index->Stats().gc_scans, 0u);
  index->SuspendGc();
  const uint64_t scans = index->Stats().gc_scans;
  absl::SleepFor(absl::Milliseconds(110));
  EXPECT_EQ(index->Stats().gc_scans, scans);
}

TEST(IndexTest, GcCoalescesScansAcrossDeadlines) {
  TempDir dir;
  IndexOptions options;
  // Scans run at most once per 500 ms.
  options.gc_grace_period = absl::Seconds(5);
  const absl::Duration window = options.gc_grace_period / 10;
  options.compaction.tier_min_segments = 20;
  // One live document keeps the segment below the rewrite threshold, so no
  // compaction install triggers a scan of its own.
  options.compaction.rewrite_dead_percent = 100;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened, Index::Open(*Env::Default(), dir.path(), options));
  auto index = std::move(opened.first);
  std::vector<BlobId> blobs;
  for (int i = 0; i <= 20; ++i) {
    const CorpusFile file{absl::StrCat("f", i, ".cc"),
                          absl::StrCat("coalesce", i, "\n")};
    const BlobId blob = index->blobs().Intern(HashBlob(file.content));
    ASSERT_THAT(index->AdjustReferences({{blob, 1}}), IsOk());
    ASSERT_THAT(index->Add(Document(file)), IsOk());
    blobs.push_back(blob);
  }
  ASSERT_THAT(index->Flush(), IsOk());
  index->ResumeGc();
  // Twenty view batches each zero one count, queueing twenty deadlines about
  // 30 ms apart. Without coalescing, each due deadline would scan.
  blobs.pop_back();
  const absl::Time first = absl::Now();
  for (BlobId blob : blobs) {
    ASSERT_THAT(index->AdjustReferences({{blob, -1}}), IsOk());
    absl::SleepFor(absl::Milliseconds(30));
  }
  const absl::Duration spread = absl::Now() - first;
  // Every deadline has passed, and the last coalesced scan has run.
  absl::SleepFor(options.gc_grace_period + 2 * window);
  const IndexStats stats = index->Stats();
  EXPECT_EQ(stats.dead_docs, blobs.size());
  EXPECT_GE(stats.gc_scans, 1u);
  // Scans after the first are at least one window apart and end by the last
  // deadline plus one window.
  EXPECT_LE(stats.gc_scans,
            static_cast<uint64_t>(absl::FDivDuration(spread, window)) + 2)
      << "spread " << spread;
}

TEST(IndexTest, FlushDoesNotRescanBeforeGarbageCollectionDeadline) {
  TempDir dir;
  IndexOptions options;
  options.gc_grace_period = absl::Hours(1);
  options.compaction.tier_min_segments = 20;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened, Index::Open(*Env::Default(), dir.path(), options));
  auto index = std::move(opened.first);
  index->ResumeGc();
  for (int i = 0; i < 12; ++i) {
    const CorpusFile file{absl::StrCat("pending", i, ".cc"),
                          absl::StrCat("int Pending", i, "() {}\n")};
    ASSERT_THAT(index->Add(Document(file)), IsOk());
    ASSERT_THAT(index->Flush(), IsOk());
  }
  // Each deadline will scan once it becomes eligible. Flush must not scan
  // every previously installed segment while the grace period is still open.
  EXPECT_EQ(index->Stats().gc_scans, 0u);
}

TEST(IndexTest, GcGraceDefersRewriteUntilDeadline) {
  TempDir dir;
  IndexOptions options;
  options.compaction.tier_min_segments = 20;
  options.gc_grace_period = absl::Milliseconds(200);
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened, Index::Open(*Env::Default(), dir.path(), options));
  auto index = std::move(opened.first);
  const CorpusFile file{.path = "grace.cc", .content = "gracetoken\n"};
  ASSERT_THAT(index->Add(Document(file)), IsOk());
  ASSERT_THAT(index->Flush(), IsOk());
  index->ResumeGc();
  EXPECT_TRUE(ContainsBlob(*index->GetReadState(), HashBlob(file.content)));
  // Counts describe the last scan; the deadline worker has not scanned yet.
  EXPECT_EQ(index->Stats().dead_docs, uint64_t{0});
  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  while (index->Stats().completed_compactions == 0 && absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(20));
  }
  ASSERT_GT(index->Stats().completed_compactions, 0u);
  EXPECT_FALSE(ContainsBlob(*index->GetReadState(), HashBlob(file.content)));
  EXPECT_GT(index->Stats().reclaimed_bytes, 0u);
}

TEST(IndexTest, SeededReferenceGcDifferential) {
  const uint64_t seed = ferret_testing::TestSeed(0x9a73);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  TempDir dir;
  Env& env = *Env::Default();
  IndexOptions options;
  options.gc_grace_period = absl::ZeroDuration();
  options.memtable.flush_docs = 5;
  options.compaction.rewrite_dead_percent = 1;
  options.compaction.tier_min_segments = 2;
  options.compaction.tier_base_bytes = 64;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(env, dir.path(), options));
  auto index = std::move(opened.first);
  index->ResumeGc();
  std::vector<CorpusFile> files;
  std::vector<BlobId> ids;
  std::vector<int> counts;
  for (int step = 0; step < 55; ++step) {
    SCOPED_TRACE(step);
    if (files.empty() || rng() % 3 == 0) {
      files.push_back({.path = absl::StrCat("src/", step, ".cc"),
                       .content = absl::StrCat("gcunique", step, "\n")});
      const BlobId id = index->blobs().Intern(HashBlob(files.back().content));
      ids.push_back(id);
      counts.push_back(1);
      ASSERT_THAT(index->AdjustReferences({{id, 1}}), IsOk());
      ASSERT_THAT(index->Add(Document(files.back())), IsOk());
    } else {
      const size_t i = rng() % files.size();
      if (counts[i] > 0 && rng() % 2 == 0) {
        --counts[i];
        ASSERT_THAT(index->AdjustReferences({{ids[i], -1}}), IsOk());
      } else {
        // Re-ingest if GC may have removed the last copy.
        if (counts[i] == 0) ASSERT_THAT(index->Add(Document(files[i])), IsOk());
        ++counts[i];
        ASSERT_THAT(index->AdjustReferences({{ids[i], 1}}), IsOk());
      }
    }
    if (step % 6 == 0) ASSERT_THAT(index->Flush(), IsOk());
    if (step % 9 == 0) index->WaitForCompactions();
    const auto state = index->GetReadState();
    for (size_t i = 0; i < files.size(); ++i) {
      if (counts[i] == 0) continue;
      EXPECT_TRUE(ContainsBlob(*state, HashBlob(files[i].content))) << i;
      auto found = Search(*state, {.pattern = files[i].content.substr(
                                       0, files[i].content.size() - 1)});
      ASSERT_THAT(found, IsOk());
      EXPECT_FALSE(found->files.empty()) << i;
    }
  }
  ASSERT_THAT(index->Flush(), IsOk());
  index->WaitForCompactions();
  const auto state = index->GetReadState();
  for (const VersionSegment& entry : state->version->segments) {
    for (BlobId id : *entry.blob_of_doc) {
      const auto it = std::find(ids.begin(), ids.end(), id);
      ASSERT_NE(it, ids.end());
      EXPECT_GT(counts[it - ids.begin()], 0);
    }
  }
  EXPECT_EQ(index->Stats().dead_docs, uint64_t{0});
}

TEST(IndexTest, OpenFailureReturnsAnError) {
  TempDir dir;
  Env& env = *Env::Default();
  FERRET_ASSERT_OK_AND_ASSIGN(auto first, Index::Open(env, dir.path()));
  // The first index holds the directory lock, so recovery fails before the
  // second index starts its threads; destroying it must not abort.
  EXPECT_FALSE(Index::Open(env, dir.path()).ok());
}

TEST(IndexTest, BlobIdsAcrossFlushCompactionAndReopen) {
  TempDir dir;
  Env& env = *Env::Default();
  IndexOptions options;
  options.compaction.tier_base_bytes = 64;
  options.compaction.tier_min_segments = 2;
  options.compaction.max_live_segments = 3;
  options.compaction.forced_merge_segments = 2;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(env, dir.Join("index"), options));
  auto index = std::move(opened.first);
  for (int i = 0; i < 24; ++i) {
    const CorpusFile file{absl::StrCat("file", i, ".cc"),
                          absl::StrCat("text ", i % 7, "\n")};
    ASSERT_THAT(index->Add(Document(file)), IsOk());
    CheckBlobMappings(*index->GetReadState());
    if (i % 4 == 3) {
      ASSERT_THAT(index->Flush(), IsOk());
      CheckBlobMappings(*index->GetReadState());
    }
  }
  index->WaitForCompactions();
  EXPECT_GT(index->Stats().completed_compactions, 0u);
  auto state = index->GetReadState();
  CheckBlobMappings(*state);
  EXPECT_EQ(state->blobs->size(), size_t{7});
  // Keep an old snapshot alive across reopen; its registry remains valid.
  index.reset();
  CheckBlobMappings(*state);
  FERRET_ASSERT_OK_AND_ASSIGN(auto reopened,
                              Index::Open(env, dir.Join("index"), options));
  index = std::move(reopened.first);
  CheckBlobMappings(*index->GetReadState());
  ASSERT_THAT(index->Add(Document({"new.cc", "text 0\n"})), IsOk());
  CheckBlobMappings(*index->GetReadState());
}

TEST(IndexTest, DifferentialScheduleWithRecovery) {
  const uint64_t seed = ferret_testing::TestSeed(0x73f5);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  TempDir dir;
  IndexOptions options;
  options.memtable.flush_docs = 4;
  options.memtable.segment.chunk_bytes = 16;
  options.compaction.tier_base_bytes = 64;
  options.compaction.tier_min_segments = 2;
  options.compaction.max_live_segments = 3;
  options.compaction.forced_merge_segments = 2;
  Env& env = *Env::Default();
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(env, dir.Join("index"), options));
  auto index = std::move(opened.first);
  std::vector<CorpusFile> corpus;
  std::vector<CorpusFile> durable;
  const std::vector<std::string> patterns = {"needle", "foo", "n.e[d]le",
                                             "absent", "xy"};
  const std::vector<std::string> extensions = {".cc", ".py", ".go", ".md"};
  const std::vector<std::string> snippets = {
      "int needle() { return foo; }\n",
      "def foo(): # needle\n    return 'xy'\n", "func foo() { needle := xy }\n",
      "# needle foo xy\n", "needle foo xy"};
  for (int step = 0; step < 60; ++step) {
    SCOPED_TRACE(step);
    const int operation = static_cast<int>(rng() % 10);
    if (operation == 0 || (step > 0 && step % 17 == 0)) {
      ASSERT_THAT(index->Flush(), IsOk());
      durable = corpus;
      if (step % 17 == 0) {
        index.reset();
        FERRET_ASSERT_OK_AND_ASSIGN(
            auto recovered, Index::Open(env, dir.Join("index"), options));
        index = std::move(recovered.first);
        corpus = durable;
      }
    } else {
      CorpusFile file{.path = absl::StrCat(
                          "src/", step, extensions[rng() % extensions.size()]),
                      .content = snippets[rng() % snippets.size()]};
      if (step % 2 == 0) file.content += snippets[rng() % snippets.size()];
      file.content += absl::StrCat(" step", step, "\n");
      if (step % 7 == 0 && !corpus.empty()) {
        file.content = corpus[rng() % corpus.size()].content;
      }
      ASSERT_THAT(index->Add(Document(file)), IsOk());
      const Oid oid = HashBlob(file.content);
      auto duplicate = std::find_if(corpus.begin(), corpus.end(),
                                    [&](const CorpusFile& prior) {
                                      return HashBlob(prior.content) == oid;
                                    });
      if (duplicate == corpus.end()) {
        corpus.push_back(file);
      } else if (Document(file).rank_key < Document(*duplicate).rank_key) {
        *duplicate = file;
      }
    }
    for (int query = 0; query < 3; ++query) {
      const int p = static_cast<int>(rng() % patterns.size());
      std::string query_pattern = patterns[p];
      if (!corpus.empty() && query == 2) {
        const std::string& content = corpus[rng() % corpus.size()].content;
        // Draw a whole identifier or a 3-8 byte substring of one.
        constexpr absl::string_view kIdentifierChars =
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";
        size_t begin = content.find_first_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
            static_cast<size_t>(rng() % content.size()));
        if (begin == std::string::npos) {
          begin = content.find_first_of(
              "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");
        }
        if (begin != std::string::npos) {
          const size_t end =
              std::min(content.find_first_not_of(kIdentifierChars, begin),
                       content.size());
          if (end - begin >= 3) {
            if (rng() % 2 == 0) {
              query_pattern = content.substr(begin, end - begin);
            } else {
              begin += static_cast<size_t>(rng() % (end - begin - 2));
              query_pattern = content.substr(
                  begin, std::min<size_t>(3 + rng() % 6, end - begin));
            }
          }
        }
      }
      SearchRequest request{.pattern = query_pattern,
                            .regex = query_pattern == "n.e[d]le",
                            .case_sensitive = query % 2 == 0,
                            .context_lines = 0,
                            .max_lines_per_file = 100,
                            .max_lines = 10000,
                            .max_files = 10000};
      Compare(*index->GetReadState(), corpus, request);
      request.max_files = 2;
      // With unlimited matching lines, top-K must be the prefix of the
      // exhaustively ranked response.
      Compare(*index->GetReadState(), corpus, request);
      request.max_files = 10000;
      request.max_lines = 2;
      Compare(*index->GetReadState(), corpus, request);
    }
  }
  ASSERT_THAT(index->Flush(), IsOk());
  index->WaitForCompactions();
  // With two-way forced merges, an idle index cannot exceed the cap of three.
  EXPECT_LE(index->Stats().segment_count, 3u);
  EXPECT_GT(index->Stats().completed_compactions, 0u);
}

// Compares all budget-dependent observable results, not just the top files.
void CompareSegmentLayouts(const ReadState& state, const Segment& single,
                           const SearchRequest& request) {
  FERRET_ASSERT_OK_AND_ASSIGN(const SearchResponse merged,
                              Search(state, request));
  FERRET_ASSERT_OK_AND_ASSIGN(const SearchResponse flat,
                              SearchSegment(single, request));
  EXPECT_EQ(merged.complete, flat.complete);
  EXPECT_EQ(merged.num_matching_files, flat.num_matching_files);
  EXPECT_EQ(merged.num_matching_lines, flat.num_matching_lines);
  EXPECT_EQ(merged.stats.verified_bytes, flat.stats.verified_bytes);
  EXPECT_EQ(merged.stats.candidate_chunks, flat.stats.candidate_chunks);
  EXPECT_EQ(merged.stats.verified_chunks, flat.stats.verified_chunks);
  ASSERT_EQ(merged.files.size(), flat.files.size());
  for (size_t i = 0; i < flat.files.size(); ++i) {
    EXPECT_EQ(merged.files[i].path, flat.files[i].path);
    EXPECT_EQ(merged.files[i].num_matching_lines,
              flat.files[i].num_matching_lines);
    std::vector<int64_t> merged_lines, flat_lines;
    for (const ResultLine& line : merged.files[i].lines) {
      if (line.is_match) merged_lines.push_back(line.number);
    }
    for (const ResultLine& line : flat.files[i].lines) {
      if (line.is_match) flat_lines.push_back(line.number);
    }
    EXPECT_EQ(merged_lines, flat_lines);
  }
}

// Check every observable file field, not just the winning paths.
void ExpectSameFiles(const SearchResponse& early, const SearchResponse& exact) {
  ASSERT_EQ(early.files.size(), exact.files.size());
  for (size_t i = 0; i < exact.files.size(); ++i) {
    const FileResult& a = early.files[i];
    const FileResult& b = exact.files[i];
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

TEST(IndexTest, EarlyTerminationMatchesExactCountsOnRandomLayouts) {
  const uint64_t seed = ferret_testing::TestSeed(0x9e41);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  int terminations = 0;
  for (int corpus = 0; corpus < 2; ++corpus) {
    SCOPED_TRACE(corpus);
    TempDir dir;
    Env& env = *Env::Default();
    IndexOptions options;
    options.memtable.segment.chunk_bytes = 160;
    FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                                Index::Open(env, dir.Join("random"), options));
    auto index = std::move(opened.first);
    Memtable all(options.memtable);
    const std::vector<std::string> prefixes = {"",
                                               "src/",
                                               "src/deep/more/",
                                               "tests/",
                                               "docs/",
                                               "third_party/lib/",
                                               "generated/",
                                               "src/deep/more/very/deep/"};
    const std::vector<std::string> tokens = {"AlphaNeedle", "BetaQuery",
                                             "GammaSymbol"};
    std::vector<std::string> contents;
    for (int i = 0; i < 90; ++i) {
      std::string content;
      if (i % 9 == 0 && !contents.empty()) {
        content = contents[rng() % contents.size()];
      } else {
        const std::string& token = tokens[rng() % tokens.size()];
        for (int line = 0; line < 1 + static_cast<int>(rng() % 4); ++line) {
          content += absl::StrCat((rng() % 2 == 0 ? "int " : "// "), token,
                                  " = ", tokens[rng() % tokens.size()], ";\n");
        }
        content += std::string(size_t{1} << (4 + rng() % 10), 'x');
        content += "\n";
      }
      const std::string path =
          absl::StrCat(prefixes[rng() % prefixes.size()],
                       i % 5 == 0 ? tokens[rng() % tokens.size()] : "file", i,
                       i % 13 == 0   ? ".pb.cc"
                       : i % 11 == 0 ? ".md"
                                     : ".cc");
      const CorpusFile file{.path = path, .content = content};
      contents.push_back(content);
      ASSERT_THAT(index->Add(Document(file)), IsOk());
      ASSERT_THAT(all.Add(Document(file)), IsOk());
      if (i == 29 || i == 59) ASSERT_THAT(index->Flush(), IsOk());
    }
    const auto state = index->GetReadState();
    ASSERT_EQ(state->version->segments.size(), 2u);
    ASSERT_GT(state->mem_watermark, 0u);
    const std::string single_path = dir.Join("all.fseg");
    ASSERT_THAT(all.WriteSegment(env, single_path, all.published_docs(),
                                 [](DocId) { return true; }),
                IsOk());
    FERRET_ASSERT_OK_AND_ASSIGN(auto single, Segment::Open(env, single_path));

    for (const std::string& pattern :
         {"AlphaNeedle", "BetaQuery", "GammaSymbol", "Alpha(Nee|NEE)dle"}) {
      for (bool regex : {false, true}) {
        if (!regex && absl::StrContains(pattern, '(')) continue;
        for (bool case_sensitive : {false, true}) {
          for (int k : {1, 2, 3, 10}) {
            for (int64_t max_lines : {int64_t{2}, int64_t{10000}}) {
              for (int per_file : {1, 10000}) {
                SearchRequest request{.pattern = pattern,
                                      .regex = regex,
                                      .case_sensitive = case_sensitive,
                                      .max_lines_per_file = per_file,
                                      .max_lines = max_lines,
                                      .max_files = k};
                if (rng() % 5 == 0) request.max_verify_bytes = 300;
                SCOPED_TRACE(absl::StrCat(pattern, ":", regex, ":", k, ":",
                                          max_lines, ":", per_file, ":",
                                          request.max_verify_bytes));
                for (int layout = 0; layout < 2; ++layout) {
                  request.exact_counts = false;
                  FERRET_ASSERT_OK_AND_ASSIGN(
                      auto early, layout == 0
                                      ? Search(*state, request)
                                      : SearchSegment(*single, request));
                  request.exact_counts = true;
                  FERRET_ASSERT_OK_AND_ASSIGN(
                      auto exact, layout == 0
                                      ? Search(*state, request)
                                      : SearchSegment(*single, request));
                  ExpectSameFiles(early, exact);
                  EXPECT_LE(early.num_matching_files, exact.num_matching_files);
                  EXPECT_LE(early.num_matching_lines, exact.num_matching_lines);
                  EXPECT_LE(early.stats.verified_chunks,
                            exact.stats.verified_chunks);
                  EXPECT_FALSE(exact.terminated_early);
                  if (early.terminated_early) {
                    ++terminations;
                    EXPECT_LT(early.stats.candidate_chunks,
                              exact.stats.candidate_chunks);
                    EXPECT_TRUE(early.complete);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  EXPECT_GE(terminations, 50);
}

// Every installed segment answers basename searches from its own hint paths,
// however it was installed.
void CheckBasenames(const ReadState& state) {
  for (const VersionSegment& source : state.version->segments) {
    const Segment& segment = *source.segment;
    for (const absl::string_view needle : {"", "file1", "ILE2", ".cc", "/"}) {
      for (const LiteralFinder::Mode mode :
           {LiteralFinder::Mode::kExact,
            LiteralFinder::Mode::kAsciiCaseInsensitive}) {
        std::vector<DocId> expected;
        for (DocId id = 0; id < segment.num_docs(); ++id) {
          const absl::string_view path = segment.doc(id).hint_path;
          std::string name(path.substr(path.rfind('/') + 1));
          std::string wanted(needle);
          if (mode == LiteralFinder::Mode::kAsciiCaseInsensitive) {
            absl::AsciiStrToLower(&name);
            absl::AsciiStrToLower(&wanted);
          }
          if (absl::StrContains(name, wanted)) expected.push_back(id);
        }
        std::vector<DocId> actual;
        segment.ForEachBasenameContaining(
            needle, mode, [&](DocId id) { actual.push_back(id); });
        EXPECT_EQ(actual, expected) << needle;
      }
    }
  }
}

TEST(IndexTest, BasenamesAcrossFlushCompactionAndReopen) {
  TempDir dir;
  Env& env = *Env::Default();
  IndexOptions options;
  options.compaction.tier_base_bytes = 64;
  options.compaction.tier_min_segments = 2;
  options.compaction.max_live_segments = 3;
  options.compaction.forced_merge_segments = 2;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(env, dir.Join("index"), options));
  auto index = std::move(opened.first);
  for (int i = 0; i < 24; ++i) {
    const CorpusFile file{
        absl::StrCat(i % 2 == 0 ? "" : "src/a/b/c/d/", "file", i, ".cc"),
        absl::StrCat(i % 2 == 0 ? "int file1 = " : "// file1 ", i, ";\n")};
    ASSERT_THAT(index->Add(Document(file)), IsOk());
    if (i % 4 == 3) {
      ASSERT_THAT(index->Flush(), IsOk());
      CheckBasenames(*index->GetReadState());
    }
  }
  index->WaitForCompactions();
  EXPECT_GT(index->Stats().completed_compactions, 0u);
  CheckBasenames(*index->GetReadState());
  index.reset();
  FERRET_ASSERT_OK_AND_ASSIGN(auto reopened,
                              Index::Open(env, dir.Join("index"), options));
  index = std::move(reopened.first);
  const std::shared_ptr<const ReadState> state = index->GetReadState();
  ASSERT_FALSE(state->version->segments.empty());
  CheckBasenames(*state);
  // The deep files fall below the residual ceiling, so the query collects
  // the priority documents from every segment: file1.cc and file1[0-9].cc.
  SearchRequest request{.pattern = "file1", .max_files = 1};
  FERRET_ASSERT_OK_AND_ASSIGN(const SearchResponse early,
                              Search(*state, request));
  EXPECT_TRUE(early.terminated_early);
  EXPECT_EQ(early.stats.priority_documents, 11);
  request.exact_counts = true;
  FERRET_ASSERT_OK_AND_ASSIGN(const SearchResponse exact,
                              Search(*state, request));
  ExpectSameFiles(early, exact);
}

// The name-aware bound across three segments and the live memtable, with
// duplicate Oids across sources and basenames that match at every rank
// position and in every tier. In the last two layouts, few segment basenames
// match but most memtable ones do, including copies of segment blobs at equal
// and at worse rank: mostly memtable priority documents hold back a stop, and
// a wrong stop changes which copy is reported or which file gets the last
// lines under a global cap. Inline and pipelined, the early top K equals the
// exhaustive top K in every field and the brute-force definition over the
// deduplicated corpus.
TEST(IndexTest, NameHeavyTopKMatchesOracleAcrossSources) {
  const uint64_t seed = ferret_testing::TestSeed(0x6a3e);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  const std::vector<std::string> prefixes = {
      "",       "src/",  "src/a/",           "src/a/b/c/", "src/a/b/c/d/e/f/",
      "tests/", "docs/", "third_party/lib/", "gen/"};
  const std::vector<std::string> matching_names = {"needle", "Needle", "NEEDLE",
                                                   "my_needle_x"};
  const std::vector<std::string> other_names = {"other", "needl", "plain"};
  const std::vector<std::string> extensions = {".cc", ".h", "_test.cc", ".md",
                                               ".pb.cc"};
  const std::vector<std::string> lines = {"int needle = 1;", "// needle here",
                                          "needle_call();",  "s = \"NEEDLE\";",
                                          "int other = 2;",  "x = 1;"};
  SearchExecutor executor(3);
  const SearcherOptions pipelined{.executor = &executor,
                                  .parallel_threshold_bytes = 0,
                                  .grain_chunks = 1,
                                  .grain_bytes = 1};
  const std::vector<SearchRequest> patterns = {
      {.pattern = "needle", .case_sensitive = true},
      {.pattern = "needle"},
      {.pattern = "NEEDLE"},
      {.pattern = "[Nn]eedle", .regex = true, .case_sensitive = true},
      {.pattern = "need(le|LE)", .regex = true},
      {.pattern = "(needle|other)", .regex = true},
      {.pattern = "other"}};
  // Layout 0 mixes names everywhere; layouts 1 and 2 put them in the memtable.
  for (int layout = 0; layout < 3; ++layout) {
    SCOPED_TRACE(layout);
    const bool memtable_names = layout > 0;
    TempDir dir;
    IndexOptions options;
    options.memtable.segment.chunk_bytes = 128;
    options.compaction.enabled = false;
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto opened, Index::Open(*Env::Default(), dir.Join("names"), options));
    auto index = std::move(opened.first);
    // Every added file with the directory and extension of its path, and the
    // copy of each Oid that searches report: the lowest RankKey, or the
    // earliest arrival among equal keys.
    struct AddedFile {
      CorpusFile file;
      std::string dir;
      std::string extension;
    };
    std::vector<AddedFile> added;
    std::vector<CorpusFile> reported;
    constexpr int kSegmentFiles = 90;
    const int num_files = kSegmentFiles + (memtable_names ? 60 : 30);
    for (int i = 0; i < num_files; ++i) {
      const bool in_memtable = i >= kSegmentFiles;
      const uint64_t match_percent =
          !memtable_names ? 57 : (in_memtable ? 80 : 5);
      auto name = [&] {
        return rng() % 100 < match_percent
                   ? matching_names[rng() % matching_names.size()]
                   : other_names[rng() % other_names.size()];
      };
      AddedFile file;
      if (memtable_names && in_memtable && i % 3 != 0) {
        // A segment blob under a matching basename, in the same directory
        // (the same rank) or one level deeper.
        const AddedFile& original = added[rng() % kSegmentFiles];
        file.file.content = original.file.content;
        file.dir = absl::StrCat(original.dir, rng() % 2 == 0 ? "" : "w/");
        file.extension = original.extension;
        file.file.path = absl::StrCat(
            file.dir, matching_names[rng() % matching_names.size()], i,
            file.extension);
        if (file.dir == original.dir) {
          ASSERT_EQ(Document(file.file).rank_key,
                    Document(original.file).rank_key);
        }
      } else if (i % 6 == 5) {
        // A copy of an earlier file, under its own path half of the time.
        file = added[rng() % added.size()];
        if (rng() % 2 == 0) {
          file.dir = prefixes[rng() % prefixes.size()];
          file.extension = ".cc";
          file.file.path = absl::StrCat(file.dir, name(), i, file.extension);
        }
      } else {
        for (int line = static_cast<int>(rng() % 12); line >= 0; --line) {
          absl::StrAppend(&file.file.content, lines[rng() % lines.size()],
                          "\n");
        }
        absl::StrAppend(&file.file.content,
                        std::string(rng() % 3 == 0 ? rng() % 9000 : 0, 'x'),
                        "\nend ", i, "\n");
        file.dir = prefixes[rng() % prefixes.size()];
        file.extension = extensions[rng() % extensions.size()];
        file.file.path = absl::StrCat(file.dir, name(), i, file.extension);
      }
      ASSERT_THAT(index->Add(Document(file.file)), IsOk());
      const RankKey key = Document(file.file).rank_key;
      const auto same = std::find_if(
          reported.begin(), reported.end(), [&](const CorpusFile& other) {
            return Document(other).rank_key.oid == key.oid;
          });
      if (same == reported.end()) {
        reported.push_back(file.file);
      } else if (key < Document(*same).rank_key) {
        *same = file.file;
      }
      added.push_back(std::move(file));
      if (i == 29 || i == 59 || i == kSegmentFiles - 1) {
        ASSERT_THAT(index->Flush(), IsOk());
      }
    }
    const std::shared_ptr<const ReadState> state = index->GetReadState();
    ASSERT_EQ(state->version->segments.size(), 3u);
    ASSERT_GT(state->mem_watermark, 0u);
    ASSERT_LT(reported.size(), added.size());

    int name_aware_stops = 0;
    for (const SearchRequest& pattern : patterns) {
      for (const int64_t k : {int64_t{1}, int64_t{2}, int64_t{3}, int64_t{5},
                              int64_t{10}, int64_t{1} << 40}) {
        for (const int64_t max_lines :
             {int64_t{1}, int64_t{3}, int64_t{10000}}) {
          SearchRequest request = pattern;
          request.context_lines = 0;
          request.max_lines_per_file = rng() % 2 == 0 ? 1 : 100;
          request.max_lines = max_lines;
          request.max_files = k;
          SCOPED_TRACE(absl::StrCat(request.pattern, " regex ", request.regex,
                                    " case ", request.case_sensitive, " k ", k,
                                    " lines ", request.max_lines_per_file, "/",
                                    max_lines));
          FERRET_ASSERT_OK_AND_ASSIGN(const SearchResponse early,
                                      Search(*state, request));
          FERRET_ASSERT_OK_AND_ASSIGN(const SearchResponse parallel,
                                      Search(*state, request, pipelined));
          ExpectSameFiles(parallel, early);
          EXPECT_EQ(parallel.terminated_early, early.terminated_early);
          EXPECT_EQ(parallel.stats.verified_bytes, early.stats.verified_bytes);
          EXPECT_EQ(parallel.stats.priority_documents,
                    early.stats.priority_documents);
          request.exact_counts = true;
          FERRET_ASSERT_OK_AND_ASSIGN(const SearchResponse exact,
                                      Search(*state, request));
          EXPECT_TRUE(early.complete);
          ExpectSameFiles(early, exact);
          if (!early.terminated_early) {
            EXPECT_EQ(early.num_matching_files, exact.num_matching_files);
            EXPECT_EQ(early.num_matching_lines, exact.num_matching_lines);
          } else if (early.stats.priority_documents > 0) {
            ++name_aware_stops;
          }
          Compare(*state, reported, request);
        }
      }
    }
    EXPECT_GE(name_aware_stops, 20);
  }
}

// Sixteen identifier matches: the most any file can earn besides its name.
std::string SixteenIdentifierLines() {
  std::string content;
  for (int i = 0; i < 16; ++i) content += "int needle;\n";
  return content;
}

// Deduplication keeps the segment copy a/b/x.cc of a blob, which falls below
// the residual ceiling. The memtable copy at the same rank has a basename
// that matches, so a stop at x.cc would leave the Oid unseen and report that
// copy; it keeps the full ceiling and holds back the stop.
TEST(IndexTest, NameAwareStopWaitsForMemtableCopyWithMatchingName) {
  TempDir dir;
  IndexOptions options;
  options.compaction.enabled = false;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened, Index::Open(*Env::Default(), dir.Join("index"), options));
  auto index = std::move(opened.first);
  const CorpusFile segment_copy{"a/b/x.cc", SixteenIdentifierLines()};
  const CorpusFile memtable_copy{"a/b/needle.cc", segment_copy.content};
  ASSERT_THAT(index->Add(Document({"w.cc", "int needle;\n"})), IsOk());
  ASSERT_THAT(index->Add(Document(segment_copy)), IsOk());
  ASSERT_THAT(index->Flush(), IsOk());
  ASSERT_THAT(index->Add(Document(memtable_copy)), IsOk());
  ASSERT_EQ(Document(memtable_copy).rank_key, Document(segment_copy).rank_key);
  const std::shared_ptr<const ReadState> state = index->GetReadState();
  SearchRequest request{.pattern = "needle", .max_files = 1};
  FERRET_ASSERT_OK_AND_ASSIGN(const SearchResponse early,
                              Search(*state, request));
  request.exact_counts = true;
  FERRET_ASSERT_OK_AND_ASSIGN(const SearchResponse exact,
                              Search(*state, request));
  ASSERT_EQ(exact.files.size(), 1u);
  EXPECT_EQ(exact.files[0].path, "w.cc");
  const RankKey key = Document(segment_copy).rank_key;
  EXPECT_TRUE(CannotOutrank(key, kMaxBoostWithoutName, exact.files[0]));
  EXPECT_FALSE(CannotOutrank(key, kMaxBoost, exact.files[0]));
  ExpectSameFiles(early, exact);
  EXPECT_FALSE(early.terminated_early);
  EXPECT_EQ(early.stats.priority_documents, 1);
}

// Under a global line cap, the exhaustive search spends the last lines on
// a/b/x.cc, so a later memtable file whose basename matches keeps none. A stop
// at x.cc would leave them to that file, so it may stop only when the file is
// beyond the reach of the full ceiling.
TEST(IndexTest, NameAwareStopWaitsForMemtableNameMatchUnderLineCap) {
  for (const bool reachable : {true, false}) {
    SCOPED_TRACE(reachable);
    TempDir dir;
    IndexOptions options;
    options.compaction.enabled = false;
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto opened, Index::Open(*Env::Default(), dir.Join("index"), options));
    auto index = std::move(opened.first);
    ASSERT_THAT(index->Add(Document({"w.cc", "int needle;\n"})), IsOk());
    const CorpusFile middle{"a/b/x.cc", "// needle\n// needle\n"};
    ASSERT_THAT(index->Add(Document(middle)), IsOk());
    ASSERT_THAT(index->Flush(), IsOk());
    const CorpusFile late{reachable ? "a/b/c/needle.cc" : "a/b/c/d/needle.cc",
                          SixteenIdentifierLines()};
    ASSERT_THAT(index->Add(Document(late)), IsOk());
    const std::shared_ptr<const ReadState> state = index->GetReadState();
    SearchRequest request{.pattern = "needle", .max_lines = 3, .max_files = 1};
    FERRET_ASSERT_OK_AND_ASSIGN(const SearchResponse early,
                                Search(*state, request));
    request.exact_counts = true;
    FERRET_ASSERT_OK_AND_ASSIGN(const SearchResponse exact,
                                Search(*state, request));
    ASSERT_EQ(exact.files.size(), 1u);
    EXPECT_EQ(exact.files[0].path, "w.cc");
    EXPECT_TRUE(CannotOutrank(Document(middle).rank_key, kMaxBoostWithoutName,
                              exact.files[0]));
    EXPECT_NE(CannotOutrank(Document(late).rank_key, kMaxBoost, exact.files[0]),
              reachable);
    ExpectSameFiles(early, exact);
    EXPECT_NE(early.terminated_early, reachable);
    EXPECT_EQ(early.stats.priority_documents, 1);
  }
}

TEST(IndexTest, BudgetEquivalentToSingleSegment) {
  const uint64_t seed = ferret_testing::TestSeed(0x5e9a);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  TempDir dir;
  Env& env = *Env::Default();
  IndexOptions options;
  options.memtable.segment.chunk_bytes = 24;
  // Preserve the requested layout while comparing against one segment.
  options.compaction.enabled = false;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(env, dir.Join("index"), options));
  auto index = std::move(opened.first);
  Memtable all(options.memtable);
  const int segments = 2 + static_cast<int>(rng() % 4);
  const std::vector<std::string> prefixes = {"src/", "tests/", "third_party/",
                                             "src/deep/"};
  const std::vector<std::string> tokens = {
      "AlphaNeedle", "BetaQuery", "GammaSymbol", "DeltaToken", "OmegaValue"};
  std::vector<CorpusFile> corpus;
  int next_flush = 1;
  for (int i = 0; i < 48; ++i) {
    std::string content;
    if (i % 7 == 0 && !corpus.empty()) {
      content = corpus[rng() % corpus.size()].content;
    } else {
      const int lines = i % 3 == 0 ? 1 : 2 + static_cast<int>(rng() % 5);
      for (int line = 0; line < lines; ++line) {
        content += absl::StrCat("int ", tokens[rng() % tokens.size()], " = ",
                                tokens[rng() % tokens.size()], "_", rng() % 100,
                                ";\n");
      }
    }
    corpus.push_back({.path = absl::StrCat(prefixes[rng() % prefixes.size()],
                                           "file", i, ".cc"),
                      .content = std::move(content)});
    ASSERT_THAT(index->Add(Document(corpus.back())), IsOk());
    ASSERT_THAT(all.Add(Document(corpus.back())), IsOk());
    if (i + 1 == 48 * next_flush / segments && i + 1 < 48) {
      ++next_flush;
      ASSERT_THAT(index->Flush(), IsOk());
    }
  }
  ASSERT_THAT(index->Flush(), IsOk());
  const std::shared_ptr<const ReadState> state = index->GetReadState();
  ASSERT_EQ(state->version->segments.size(), static_cast<size_t>(segments));
  ASSERT_TRUE(state->frozen.empty());
  ASSERT_EQ(state->mem_watermark, 0u);
  const std::string path = dir.Join("single.fseg");
  ASSERT_THAT(all.WriteSegment(env, path, all.published_docs(),
                               [](DocId) { return true; }),
              IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(auto single, Segment::Open(env, path));
  std::vector<std::string> patterns = {"AlphaNeedle", "BetaQuery", "absent",
                                       "int", "symbol"};
  for (int i = 0; i < 18; ++i) {
    const std::string& content = corpus[rng() % corpus.size()].content;
    const size_t token_end = content.find(' ', 4);
    const size_t start = 4 + static_cast<size_t>(rng() % (token_end - 6));
    const size_t length =
        3 +
        static_cast<size_t>(rng() % std::min<size_t>(6, token_end - start - 2));
    patterns.push_back(content.substr(start, length));
    patterns.push_back(tokens[rng() % tokens.size()]);
  }
  for (size_t i = 0; i < patterns.size(); ++i) {
    SCOPED_TRACE(patterns[i]);
    for (int64_t budget :
         {int64_t{1}, int64_t{23}, int64_t{71}, int64_t{300}, int64_t{900},
          int64_t{3000}, std::numeric_limits<int64_t>::max()}) {
      SCOPED_TRACE(budget);
      CompareSegmentLayouts(*state, *single,
                            {.pattern = patterns[i],
                             .case_sensitive = i % 3 == 0,
                             .context_lines = 0,
                             .max_lines_per_file = 100,
                             .max_lines = 10000,
                             .max_files = 10000,
                             .max_verify_bytes = budget});
    }
  }
}

TEST(IndexTest, MergesSegmentRanksAndAlwaysVerifiesMemtables) {
  TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(*Env::Default(), dir.Join("index")));
  auto index = std::move(opened.first);
  ASSERT_THAT(index->Add(Document({"third_party/low.cc", "needle low\n"})),
              IsOk());
  ASSERT_THAT(index->Flush(), IsOk());
  ASSERT_THAT(index->Add(Document({"best.cc", "needle best\n"})), IsOk());
  ASSERT_THAT(index->Flush(), IsOk());
  ASSERT_THAT(index->Add(Document({"live.cc", "needle live\n"})), IsOk());
  SearchRequest request{.pattern = "needle", .max_verify_bytes = 1};
  FERRET_ASSERT_OK_AND_ASSIGN(auto result,
                              Search(*index->GetReadState(), request));
  EXPECT_FALSE(result.complete);
  ASSERT_EQ(result.files.size(), 2u);
  EXPECT_EQ(result.files[0].path, "best.cc");
  EXPECT_EQ(result.files[1].path, "live.cc");
  EXPECT_EQ(result.num_matching_files, 2);
  ASSERT_THAT(index->Add(Document({"best.cc", "needle best\n"})), IsOk());
  request.max_verify_bytes = 1000000;
  FERRET_ASSERT_OK_AND_ASSIGN(result, Search(*index->GetReadState(), request));
  EXPECT_TRUE(result.complete);
  EXPECT_EQ(result.num_matching_files, 3);
  EXPECT_EQ(result.files.size(), 3u);
}

TEST(IndexTest, EarlyTerminationStillVerifiesMemtableCopies) {
  TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(*Env::Default(), dir.Join("index")));
  auto index = std::move(opened.first);
  ASSERT_THAT(index->Add(Document({"root.cc", "// needle top\n"})), IsOk());
  ASSERT_THAT(index->Add(Document({"third_party/low.cc", "// needle low\n"})),
              IsOk());
  ASSERT_THAT(
      index->Add(Document({"third_party/other.cc", "// needle other\n"})),
      IsOk());
  ASSERT_THAT(index->Add(Document({"third_party/more.cc", "// needle more\n"})),
              IsOk());
  ASSERT_THAT(index->Flush(), IsOk());
  ASSERT_THAT(index->Add(Document({"third_party/low.cc", "// needle low\n"})),
              IsOk());
  SearchRequest request{.pattern = "needle", .max_files = 1};
  FERRET_ASSERT_OK_AND_ASSIGN(auto early,
                              Search(*index->GetReadState(), request));
  request.exact_counts = true;
  FERRET_ASSERT_OK_AND_ASSIGN(auto exact,
                              Search(*index->GetReadState(), request));
  ExpectSameFiles(early, exact);
  EXPECT_TRUE(early.complete);
  EXPECT_TRUE(early.terminated_early);
  EXPECT_LT(early.stats.candidate_chunks, exact.stats.candidate_chunks);
  EXPECT_EQ(early.stats.verified_chunks, 2);
  EXPECT_EQ(early.num_matching_files, 2);
  EXPECT_EQ(exact.stats.verified_chunks, 4);
}

TEST(IndexTest, SkippedSegmentCopyDoesNotHideMemtableCopy) {
  TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(*Env::Default(), dir.Join("index")));
  auto index = std::move(opened.first);
  ASSERT_THAT(index->Add(Document({"top.cc", "needle top\n"})), IsOk());
  ASSERT_THAT(index->Add(Document({"third_party/low.cc", "needle low\n"})),
              IsOk());
  ASSERT_THAT(index->Flush(), IsOk());
  ASSERT_THAT(index->Add(Document({"third_party/low.cc", "needle low\n"})),
              IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto result, Search(*index->GetReadState(),
                          {.pattern = "needle", .max_verify_bytes = 1}));
  EXPECT_FALSE(result.complete);
  ASSERT_EQ(result.files.size(), 2u);
  EXPECT_EQ(result.num_matching_files, 2);
  EXPECT_EQ(result.files[1].path, "third_party/low.cc");
}

TEST(IndexTest, SnapshotOutlivesIndexAndReopenDropsUnflushedDocs) {
  TempDir dir;
  Env& env = *Env::Default();
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened, Index::Open(env, dir.Join("index")));
  auto index = std::move(opened.first);
  ASSERT_THAT(index->Add(Document({"on-disk.cc", "durabletoken\n"})), IsOk());
  ASSERT_THAT(index->Flush(), IsOk());
  ASSERT_THAT(index->Add(Document({"in-memory.py", "ephemeraltoken\n"})),
              IsOk());
  std::shared_ptr<const ReadState> snapshot = index->GetReadState();
  index.reset();
  FERRET_ASSERT_OK_AND_ASSIGN(auto old_result,
                              Search(*snapshot, {.pattern = "ephemeraltoken"}));
  ASSERT_EQ(old_result.files.size(), 1u);
  EXPECT_EQ(old_result.files[0].path, "in-memory.py");
  FERRET_ASSERT_OK_AND_ASSIGN(auto reopened,
                              Index::Open(env, dir.Join("index")));
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto missing,
      Search(*reopened.first->GetReadState(), {.pattern = "ephemeraltoken"}));
  EXPECT_TRUE(missing.files.empty());
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto durable,
      Search(*reopened.first->GetReadState(), {.pattern = "durabletoken"}));
  ASSERT_EQ(durable.files.size(), 1u);
  EXPECT_EQ(durable.files[0].path, "on-disk.cc");
  snapshot.reset();
}

TEST(IndexTest, RejectsFrozenCapAboveFour) {
  TempDir dir;
  EXPECT_THAT(Index::Open(*Env::Default(), dir.Join("index"),
                          {.max_frozen_memtables = 5}),
              ::absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(IndexTest, RejectsCompactionOptionsThatCannotMakeProgress) {
  TempDir dir;
  IndexOptions options;
  options.compaction.tier_min_segments = 1;
  EXPECT_THAT(Index::Open(*Env::Default(), dir.Join("index"), options),
              ::absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  options.compaction.tier_min_segments = 2;
  options.compaction.retry_initial_delay = absl::ZeroDuration();
  EXPECT_THAT(Index::Open(*Env::Default(), dir.Join("index"), options),
              ::absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  options.compaction.retry_initial_delay = absl::Seconds(2);
  options.compaction.retry_max_delay = absl::Seconds(1);
  EXPECT_THAT(Index::Open(*Env::Default(), dir.Join("index"), options),
              ::absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  options.compaction.enabled = false;
  EXPECT_THAT(Index::Open(*Env::Default(), dir.Join("index"), options), IsOk());
}

TEST(IndexTest, IdleAgeFlushesWithoutAnotherAdd) {
  TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened, Index::Open(*Env::Default(), dir.Join("index"),
                               {.max_memtable_age = absl::Milliseconds(20)}));
  auto index = std::move(opened.first);
  ASSERT_THAT(index->Add(Document({"a.cc", "needle\n"})), IsOk());
  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  while (index->GetReadState()->version->segments.empty() &&
         absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  const auto state = index->GetReadState();
  ASSERT_EQ(state->version->segments.size(), 1u);
  EXPECT_TRUE(state->frozen.empty());
  FERRET_ASSERT_OK_AND_ASSIGN(auto found,
                              Search(*state, {.pattern = "needle"}));
  EXPECT_EQ(found.num_matching_files, 1);
}

TEST(IndexTest, FlushFailureRetainsFrozenAndRetries) {
  TempDir dir;
  ferret_testing::FaultInjectionEnv env(Env::Default());
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened, Index::Open(env, dir.Join("index")));
  auto index = std::move(opened.first);
  const CorpusFile file{"a.cc", "unique needle\n"};
  ASSERT_THAT(index->Add(Document(file)), IsOk());
  env.FailAfter(0, absl::UnavailableError("injected flush error"));
  EXPECT_FALSE(index->Flush().ok());
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto result, Search(*index->GetReadState(), {.pattern = "needle"}));
  EXPECT_EQ(result.files.size(), 1u);
  EXPECT_FALSE(index->Add(Document({"b.cc", "other\n"})).ok());
  EXPECT_FALSE(
      index->Add(Document({"c.cc", "other again\n"}), IngestLane::kInteractive)
          .ok());
  env.ClearFault();
  ASSERT_THAT(index->Flush(), IsOk());
  index.reset();
  FERRET_ASSERT_OK_AND_ASSIGN(auto recovered,
                              Index::Open(env, dir.Join("index")));
  FERRET_ASSERT_OK_AND_ASSIGN(
      result, Search(*recovered.first->GetReadState(), {.pattern = "needle"}));
  ASSERT_EQ(result.files.size(), 1u);
  EXPECT_EQ(result.files[0].path, "a.cc");
}

TEST(IndexTest, RetryKeepsFailureStickyAcrossBothLanes) {
  TempDir dir;
  ferret_testing::FaultInjectionEnv env(Env::Default());
  auto gate = std::make_shared<absl::Notification>();
  IndexOptions options;
  options.compaction.enabled = false;
  options.memtable.flush_docs = 1;
  options.max_frozen_memtables = 1;
  options.flush_install_gate = gate;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(env, dir.Join("index"), options));
  auto index = std::move(opened.first);
  env.FailAfter(0, absl::UnavailableError("injected flush failure"));
  ASSERT_THAT(index->Add(Document({"first.cc", "marker first\n"})), IsOk());
  const absl::Status failure = index->Flush();
  EXPECT_EQ(failure.code(), absl::StatusCode::kUnavailable);
  env.ClearFault();
  absl::Status retried;
  std::thread retry([&] { retried = index->Flush(); });
  const absl::Time deadline = absl::Now() + absl::Seconds(10);
  bool writing = false;
  while (absl::Now() < deadline) {
    auto children = env.GetChildren(dir.Join("index"));
    if (children.ok() && std::any_of(children->begin(), children->end(),
                                     [](const std::string& name) {
                                       return name.ends_with(".fseg.tmp");
                                     })) {
      writing = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(1));
  }
  EXPECT_TRUE(writing);
  EXPECT_EQ(index->Add(Document({"bulk.cc", "bulk\n"})).code(), failure.code());
  EXPECT_EQ(
      index->Add(Document({"edit.cc", "edit\n"}), IngestLane::kInteractive)
          .code(),
      failure.code());
  gate->Notify();
  retry.join();
  EXPECT_THAT(retried, IsOk());
  EXPECT_THAT(
      index->Add(Document({"edit.cc", "edit\n"}), IngestLane::kInteractive),
      IsOk());
  EXPECT_THAT(index->Flush(), IsOk());
}

TEST(IndexTest, InteractiveUsesReservedSlotWhileBulkWaits) {
  TempDir dir;
  auto gate = std::make_shared<absl::Notification>();
  IndexOptions options;
  options.compaction.enabled = false;
  options.memtable.flush_docs = 1;
  options.max_frozen_memtables = 1;
  options.flush_install_gate = gate;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened, Index::Open(*Env::Default(), dir.Join("index"), options));
  auto index = std::move(opened.first);
  ASSERT_THAT(index->Add(Document({"first.cc", "marker first\n"})), IsOk());
  std::atomic<bool> bulk_done{false};
  std::atomic<bool> interactive_done{false};
  absl::Status bulk_status, interactive_status;
  std::thread bulk([&] {
    bulk_status = index->Add(Document({"bulk.cc", "marker bulk\n"}));
    bulk_done.store(true);
  });
  const absl::Time deadline = absl::Now() + absl::Seconds(10);
  while (index->Stats().bulk_ingest_waits == 0 && absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(1));
  }
  EXPECT_FALSE(bulk_done.load());
  EXPECT_THAT(index->Add(Document({"interactive.cc", "marker edit\n"}),
                         IngestLane::kInteractive),
              IsOk());
  auto snapshot = index->GetReadState();
  EXPECT_EQ(snapshot->frozen.size(), 2u);
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto visible, Search(*snapshot, {.pattern = "marker", .max_files = 10}));
  EXPECT_EQ(visible.num_matching_files, 2);
  EXPECT_EQ(snapshot->version->segments.size(), 0u);
  std::thread interactive([&] {
    interactive_status = index->Add(Document({"second.cc", "marker second\n"}),
                                    IngestLane::kInteractive);
    interactive_done.store(true);
  });
  while (index->Stats().interactive_ingest_waits == 0 &&
         absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(1));
  }
  EXPECT_FALSE(interactive_done.load());
  gate->Notify();
  bulk.join();
  interactive.join();
  EXPECT_THAT(bulk_status, IsOk());
  EXPECT_THAT(interactive_status, IsOk());
  EXPECT_THAT(index->Flush(), IsOk());
  CheckBlobMappings(*index->GetReadState());
  FERRET_ASSERT_OK_AND_ASSIGN(
      visible,
      Search(*index->GetReadState(), {.pattern = "marker", .max_files = 10}));
  EXPECT_EQ(visible.num_matching_files, 4);
}

TEST(IndexTest, DirectSegmentPublishesAtomicallyAndMapsBlobs) {
  TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(*Env::Default(), dir.Join("index")));
  auto index = std::move(opened.first);
  const CorpusFile a{"a.cc", "needle alpha\n"};
  const CorpusFile b{"b.cc", "needle beta\n"};
  std::vector<CorpusFile> files = {a, b};
  std::sort(files.begin(), files.end(),
            [](const CorpusFile& x, const CorpusFile& y) {
              return Document(x).rank_key < Document(y).rank_key;
            });
  const auto before = index->GetReadState();
  FERRET_ASSERT_OK_AND_ASSIGN(
      const SegmentStats stats, index->AddSegment([&](SegmentWriter& writer) {
        for (const CorpusFile& file : files) {
          const absl::Status status = writer.Add(Document(file));
          if (!status.ok()) return status;
        }
        return absl::OkStatus();
      }));
  EXPECT_EQ(stats.num_docs, 2u);
  EXPECT_EQ(before->version->segments.size(), 0u);
  const auto after = index->GetReadState();
  ASSERT_EQ(after->version->segments.size(), 1u);
  CheckBlobMappings(*after);
  for (DocId id = 0; id < files.size(); ++id) {
    EXPECT_EQ(after->version->segments[0].segment->doc(id).oid,
              Document(files[id]).rank_key.oid);
    EXPECT_EQ(after->version->segments[0].segment->doc(id).hint_path,
              files[id].path);
  }
  EXPECT_EQ(index->Stats().bulk_segment_bytes, stats.file_bytes);
  EXPECT_EQ(index->Stats().bulk_segments, 1u);
  Compare(*after, files, {.pattern = "needle"});
  FERRET_ASSERT_OK_AND_ASSIGN(auto empty, index->AddSegment([](SegmentWriter&) {
    return absl::OkStatus();
  }));
  EXPECT_EQ(empty.num_docs, 0u);
  EXPECT_EQ(index->Stats().bulk_segments, 1u);
  EXPECT_THAT(index
                  ->AddSegment([](SegmentWriter&) {
                    return absl::CancelledError("stopped");
                  })
                  .status(),
              absl_testing::StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(index->GetReadState()->version->segments.size(), 1u);
  index.reset();
  FERRET_ASSERT_OK_AND_ASSIGN(auto recovered,
                              Index::Open(*Env::Default(), dir.Join("index")));
  CheckBlobMappings(*recovered.first->GetReadState());
}

TEST(IndexTest, DirectSegmentAllowsDuplicateOidAcrossSources) {
  TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(*Env::Default(), dir.Join("index")));
  auto index = std::move(opened.first);
  const CorpusFile source{"source.cc", "same needle\n"};
  const CorpusFile duplicate{"copy.cc", source.content};
  ASSERT_THAT(index->Add(Document(source), IngestLane::kInteractive), IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(auto written,
                              index->AddSegment([&](SegmentWriter& writer) {
                                return writer.Add(Document(duplicate));
                              }));
  EXPECT_EQ(written.num_docs, 1u);
  auto state = index->GetReadState();
  CheckBlobMappings(*state);
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto result, Search(*state, {.pattern = "needle", .max_files = 10}));
  EXPECT_EQ(result.num_matching_files, 1);
  EXPECT_EQ(state->blobs->Find(Document(source).rank_key.oid),
            state->blobs->Find(Document(duplicate).rank_key.oid));
}

TEST(IndexTest, DirectSegmentFaultsLeavePublishedStateUnchanged) {
  TempDir dir;
  const CorpusFile file{"test.cc", "needle from segment\n"};
  const CorpusFile dead{"dead.cc", "dead after deletion\n"};
  IndexOptions options;
  options.compaction.enabled = false;
  options.gc_grace_period = absl::ZeroDuration();
  const auto build = [&](Index& index) {
    return index.AddSegment(
        [&](SegmentWriter& writer) { return writer.Add(Document(file)); });
  };
  int64_t operations = 0;
  {
    ferret_testing::FaultInjectionEnv env(Env::Default());
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto opened, Index::Open(env, dir.Join("calibration"), options));
    const int64_t before = env.mutation_count();
    ASSERT_THAT(build(*opened.first), IsOk());
    operations = env.mutation_count() - before;
  }
  ASSERT_GT(operations, 0);
  for (int64_t fail = 0; fail < operations; ++fail) {
    SCOPED_TRACE(fail);
    const std::string path = dir.Join(absl::StrCat("fault", fail));
    ferret_testing::FaultInjectionEnv env(Env::Default());
    FERRET_ASSERT_OK_AND_ASSIGN(auto opened, Index::Open(env, path, options));
    auto index = std::move(opened.first);
    ASSERT_THAT(index->Add(Document(dead)), IsOk());
    ASSERT_THAT(index->Flush(), IsOk());
    index->ResumeGc();
    ASSERT_EQ(index->Stats().dead_docs, uint64_t{1});
    env.FailAfter(fail, absl::UnavailableError("injected segment fault"));
    EXPECT_FALSE(build(*index).ok());
    EXPECT_EQ(index->GetReadState()->version->segments.size(), 1u);
    EXPECT_EQ(index->Stats().bulk_segments, 0u);
    env.ClearFault();
    index.reset();
    FERRET_ASSERT_OK_AND_ASSIGN(auto recovered,
                                Index::Open(env, path, options));
    FERRET_ASSERT_OK_AND_ASSIGN(auto children, env.GetChildren(path));
    EXPECT_TRUE(std::none_of(
        children.begin(), children.end(),
        [](const std::string& name) { return name.ends_with(".fseg.tmp"); }));
    CheckBlobMappings(*recovered.first->GetReadState());
    EXPECT_TRUE(
        ContainsBlob(*recovered.first->GetReadState(), HashBlob(dead.content)));
  }
}

TEST(IndexTest, ShutdownCancelsDirectSegmentCallback) {
  TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(*Env::Default(), dir.Join("index")));
  auto index = std::move(opened.first);
  Index* ptr = index.get();
  absl::Notification entered;
  absl::Status result;
  std::thread writer([&] {
    result = ptr->AddSegment([&](SegmentWriter&) {
                  entered.Notify();
                  while (!ptr->cancelled()) {
                    absl::SleepFor(absl::Milliseconds(1));
                  }
                  return absl::CancelledError("shutdown");
                })
                 .status();
  });
  entered.WaitForNotification();
  const absl::Time start = absl::Now();
  index.reset();
  writer.join();
  EXPECT_LT(absl::Now() - start, absl::Seconds(3));
  EXPECT_THAT(result, absl_testing::StatusIs(absl::StatusCode::kCancelled));
  FERRET_ASSERT_OK_AND_ASSIGN(auto recovered,
                              Index::Open(*Env::Default(), dir.Join("index")));
  EXPECT_EQ(recovered.first->Stats().segment_count, 0u);
}

TEST(IndexTest, DirectSegmentsInterleaveWithFlushAndCompaction) {
  TempDir dir;
  IndexOptions options;
  options.memtable.flush_docs = 2;
  options.compaction.tier_base_bytes = 64;
  options.compaction.tier_min_segments = 2;
  options.compaction.max_live_segments = 2;
  options.compaction.forced_merge_segments = 2;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened, Index::Open(*Env::Default(), dir.Join("index"), options));
  auto index = std::move(opened.first);
  std::vector<CorpusFile> files;
  files.reserve(40);
  for (int i = 0; i < 40; ++i) {
    files.push_back({absl::StrCat("source", i, ".cc"),
                     absl::StrCat("needle unique", i, "\n")});
  }
  std::atomic<int> errors{0};
  std::thread memtables([&] {
    for (int i = 0; i < 20; ++i) {
      if (!index->Add(Document(files[i]), IngestLane::kInteractive).ok()) {
        ++errors;
      }
    }
  });
  for (int batch = 0; batch < 4; ++batch) {
    std::vector<CorpusFile> ordered;
    ordered.reserve(5);
    for (int i = 20 + batch * 5; i < 25 + batch * 5; ++i) {
      ordered.push_back(files[i]);
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const CorpusFile& a, const CorpusFile& b) {
                return Document(a).rank_key < Document(b).rank_key;
              });
    auto written = index->AddSegment([&](SegmentWriter& writer) {
      for (const CorpusFile& file : ordered) {
        const absl::Status status = writer.Add(Document(file));
        if (!status.ok()) return status;
      }
      return absl::OkStatus();
    });
    if (!written.ok()) ++errors;
  }
  memtables.join();
  EXPECT_THAT(index->Flush(), IsOk());
  index->WaitForCompactions();
  EXPECT_EQ(errors.load(), 0);
  EXPECT_GT(index->Stats().completed_compactions, 0u);
  auto state = index->GetReadState();
  CheckBlobMappings(*state);
  Compare(*state, files, {.pattern = "needle", .max_files = 100});
  std::vector<std::string> paths;
  paths.reserve(files.size());
  for (const CorpusFile& file : files) {
    ASSERT_THAT(
        dir.WriteFile(absl::StrCat("worktree/", file.path), file.content),
        IsOk());
    paths.push_back(file.path);
  }
  const std::string bulk_path = dir.Join("bulk.fseg");
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto built,
      BuildSegment(*Env::Default(), dir.Join("worktree"), paths, bulk_path));
  EXPECT_EQ(built.files_indexed, static_cast<int64_t>(files.size()));
  FERRET_ASSERT_OK_AND_ASSIGN(auto reference,
                              Segment::Open(*Env::Default(), bulk_path));
  std::vector<SegmentDoc> actual;
  actual.reserve(files.size());
  for (const VersionSegment& entry : state->version->segments) {
    for (DocId id = 0; id < entry.segment->num_docs(); ++id) {
      actual.push_back(entry.segment->doc(id));
    }
  }
  std::sort(actual.begin(), actual.end(),
            [](const SegmentDoc& a, const SegmentDoc& b) {
              return a.hint_path < b.hint_path;
            });
  std::vector<SegmentDoc> expected;
  expected.reserve(reference->num_docs());
  for (DocId id = 0; id < reference->num_docs(); ++id) {
    expected.push_back(reference->doc(id));
  }
  std::sort(expected.begin(), expected.end(),
            [](const SegmentDoc& a, const SegmentDoc& b) {
              return a.hint_path < b.hint_path;
            });
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i].hint_path, expected[i].hint_path);
    EXPECT_EQ(actual[i].content, expected[i].content);
    EXPECT_EQ(actual[i].oid, expected[i].oid);
    EXPECT_EQ(actual[i].tier, expected[i].tier);
    EXPECT_EQ(actual[i].prior, expected[i].prior);
    EXPECT_EQ(actual[i].language, expected[i].language);
    EXPECT_EQ(actual[i].long_lines, expected[i].long_lines);
  }
}

TEST(IndexTest, ConcurrentLanesAndReaders) {
  TempDir dir;
  IndexOptions options;
  options.memtable.flush_docs = 3;
  options.compaction.tier_base_bytes = 64;
  options.compaction.tier_min_segments = 2;
  options.compaction.max_live_segments = 3;
  options.compaction.forced_merge_segments = 2;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened, Index::Open(*Env::Default(), dir.Join("index"), options));
  auto index = std::move(opened.first);
  std::atomic<bool> done{false};
  std::atomic<int> errors{0};
  std::thread reader([&] {
    while (!done.load()) {
      auto state = index->GetReadState();
      for (const VersionSegment& segment : state->version->segments) {
        if (segment.blob_of_doc->size() != segment.segment->num_docs()) {
          ++errors;
        }
      }
      for (const IndexedSymbol& symbol :
           state->FindSymbol("SymbolMarker", false).occurrences) {
        if (symbol.line != 1 || !symbol.is_definition) ++errors;
      }
      auto result =
          Search(*state, {.pattern = "lane_marker", .max_files = 100});
      if (!result.ok() || result->num_matching_files !=
                              static_cast<int64_t>(result->files.size())) {
        ++errors;
      }
    }
  });
  std::atomic<int> write_errors{0};
  auto ingest = [&](IngestLane lane, const char* prefix) {
    for (int i = 0; i < 30; ++i) {
      if (!index
               ->Add(
                   Document({absl::StrCat(prefix, i, ".cc"),
                             absl::StrCat("int SymbolMarker() { return 1; }\n",
                                          "lane_marker", prefix, i, "\n")}),
                   lane)
               .ok()) {
        ++write_errors;
      }
    }
  };
  std::thread bulk([&] { ingest(IngestLane::kBulk, "bulk"); });
  std::thread interactive([&] { ingest(IngestLane::kInteractive, "edit"); });
  bulk.join();
  interactive.join();
  EXPECT_THAT(index->Flush(), IsOk());
  index->WaitForCompactions();
  done.store(true);
  reader.join();
  EXPECT_EQ(write_errors.load(), 0);
  EXPECT_EQ(errors.load(), 0);
  auto state = index->GetReadState();
  CheckBlobMappings(*state);
  EXPECT_EQ(state->FindSymbol("SymbolMarker", false).occurrences.size(), 60u);
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto result,
      Search(*state, {.pattern = "lane_marker", .max_files = 100}));
  EXPECT_EQ(result.num_matching_files, 60);
}

TEST(IndexTest, ConcurrentSnapshotsSeeOnlyPublishedPrefixes) {
  TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened, Index::Open(*Env::Default(), dir.Join("index"),
                               {.memtable = {.flush_docs = 3},
                                .compaction = {.tier_base_bytes = 64,
                                               .tier_min_segments = 2,
                                               .max_live_segments = 3,
                                               .forced_merge_segments = 2}}));
  auto index = std::move(opened.first);
  std::atomic<bool> done{false};
  std::atomic<int> errors{0};
  std::vector<std::thread> readers;
  readers.reserve(4);
  for (int r = 0; r < 4; ++r) {
    readers.emplace_back([&] {
      while (!done.load(std::memory_order_acquire)) {
        auto state = index->GetReadState();
        auto result =
            Search(*state, {.pattern = "prefixmarker", .max_files = 100});
        if (!result.ok()) {
          ++errors;
          continue;
        }
        // Each snapshot includes every earlier marker or none of the later
        // ones, even while a frozen memtable is being installed.
        bool present[48] = {};
        int largest = -1;
        for (const FileResult& file : result->files) {
          const int id = std::atoi(file.path.c_str() + 3);
          if (id < 0 || id >= 48 || present[id]) {
            ++errors;
          } else {
            present[id] = true;
            largest = std::max(largest, id);
          }
        }
        for (int id = 0; id <= largest; ++id) {
          if (!present[id]) ++errors;
        }
        if (result->num_matching_files != largest + 1) ++errors;
      }
    });
  }
  for (int i = 0; i < 48; ++i) {
    const CorpusFile file{absl::StrCat("doc", i, ".cc"),
                          absl::StrCat("prefixmarker", i, "\n")};
    ASSERT_THAT(index->Add(Document(file)), IsOk());
    if (i % 7 == 0) ASSERT_THAT(index->Flush(), IsOk());
  }
  ASSERT_THAT(index->Flush(), IsOk());
  done.store(true, std::memory_order_release);
  for (auto& reader : readers) reader.join();
  index->WaitForCompactions();
  EXPECT_GT(index->Stats().completed_compactions, 0u);
  EXPECT_EQ(errors.load(), 0);
}

IndexOptions SmallCompactionOptions(bool enabled) {
  IndexOptions options;
  options.memtable.flush_docs = 1;
  options.compaction.enabled = enabled;
  options.compaction.tier_base_bytes = 64;
  options.compaction.tier_min_segments = 2;
  options.compaction.max_live_segments = 1;
  options.compaction.forced_merge_segments = 2;
  return options;
}

void PrepareSegments(Env& env, const std::string& path, int count) {
  auto opened = Index::Open(env, path, SmallCompactionOptions(false));
  ASSERT_THAT(opened, IsOk());
  auto index = std::move(opened->first);
  for (int i = 0; i < count; ++i) {
    ASSERT_THAT(index->Add(Document({absl::StrCat("file", i, ".cc"),
                                     absl::StrCat("needle", i, "\n")})),
                IsOk());
  }
  ASSERT_THAT(index->Flush(), IsOk());
  EXPECT_EQ(index->Stats().segment_count, static_cast<size_t>(count));
}

// Verifies that recovery removed every unlisted output, including temp files.
void CheckNoGcOrphans(Env& env, absl::string_view path,
                      const ReadState& state) {
  FERRET_ASSERT_OK_AND_ASSIGN(auto children, env.GetChildren(path));
  for (const std::string& name : children) {
    EXPECT_FALSE(absl::string_view(name).ends_with(".tmp")) << name;
    if (!absl::string_view(name).ends_with(".fseg")) continue;
    bool listed = false;
    for (const VersionSegment& entry : state.version->segments) {
      listed |=
          name == absl::StrFormat("%06u.fseg", entry.metadata.file_number);
    }
    EXPECT_TRUE(listed) << name;
  }
}

// Each baseline segment contains both a live and a dead blob. A flush writes
// a third mixed segment; a compaction rewrites the two baseline segments.
void PrepareGcFaultInputs(Env& env, absl::string_view path) {
  IndexOptions options;
  options.compaction.enabled = false;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened, Index::Open(env, path, options));
  for (int i = 0; i < 2; ++i) {
    ASSERT_THAT(
        opened.first->Add(Document({absl::StrCat("live", i, ".cc"),
                                    absl::StrCat("gc-live-", i, "\n")})),
        IsOk());
    ASSERT_THAT(
        opened.first->Add(Document({absl::StrCat("dead", i, ".cc"),
                                    absl::StrCat("gc-dead-", i, "\n")})),
        IsOk());
    ASSERT_THAT(opened.first->Flush(), IsOk());
  }
}

IndexOptions GcFaultOptions(bool compaction) {
  IndexOptions options;
  options.gc_grace_period = absl::ZeroDuration();
  options.compaction.enabled = compaction;
  options.compaction.tier_min_segments = 20;
  options.compaction.retry_initial_delay = absl::Seconds(10);
  options.compaction.retry_max_delay = absl::Seconds(10);
  return options;
}

// Replays each Env mutation boundary of both filtered output paths, then
// simulates a crash by discarding unsynced writes before reopening.
TEST(IndexTest, GcFlushAndCompactionRecoverAtEveryMutation) {
  for (bool compact : {false, true}) {
    SCOPED_TRACE(compact);
    int64_t mutations = 0;
    for (int64_t fail = -1; fail <= mutations; ++fail) {
      SCOPED_TRACE(fail);
      TempDir dir;
      const std::string path = dir.Join("index");
      PrepareGcFaultInputs(*Env::Default(), path);
      ferret_testing::FaultInjectionEnv env(Env::Default());
      IndexOptions options = GcFaultOptions(compact);
      // Start workers only after reference resync and fault installation.
      options.compaction.enabled = false;
      FERRET_ASSERT_OK_AND_ASSIGN(auto opened, Index::Open(env, path, options));
      auto index = std::move(opened.first);
      for (int i = 0; i < 2; ++i) {
        const BlobId id =
            index->blobs().Intern(HashBlob(absl::StrCat("gc-live-", i, "\n")));
        ASSERT_THAT(index->AdjustReferences({{id, 1}}), IsOk());
      }
      if (compact) {
        // Suspend GC while allowing the compaction worker to start only
        // after both reference counts have been installed.
        index.reset();
        options = GcFaultOptions(true);
        FERRET_ASSERT_OK_AND_ASSIGN(auto compaction_opened,
                                    Index::Open(env, path, options));
        index = std::move(compaction_opened.first);
        for (int i = 0; i < 2; ++i) {
          const BlobId id = index->blobs().Intern(
              HashBlob(absl::StrCat("gc-live-", i, "\n")));
          ASSERT_THAT(index->AdjustReferences({{id, 1}}), IsOk());
        }
      } else {
        ASSERT_THAT(index->Add(Document({"live-extra.cc", "gc-live-extra\n"})),
                    IsOk());
        ASSERT_THAT(index->Add(Document({"dead-extra.cc", "gc-dead-extra\n"})),
                    IsOk());
        const BlobId id = index->blobs().Intern(HashBlob("gc-live-extra\n"));
        ASSERT_THAT(index->AdjustReferences({{id, 1}}), IsOk());
      }
      index->ResumeGc();
      const int64_t start = env.mutation_count();
      if (fail >= 0) {
        env.FailAfter(fail, absl::UnavailableError("simulated GC crash"));
      }
      if (compact) {
        const absl::Time deadline = absl::Now() + absl::Seconds(10);
        while (index->Stats().compaction_failures == 0 &&
               index->Stats().completed_compactions == 0 &&
               absl::Now() < deadline) {
          absl::SleepFor(absl::Milliseconds(5));
        }
        ASSERT_TRUE(index->Stats().compaction_failures != 0 ||
                    index->Stats().completed_compactions != 0);
      } else {
        static_cast<void>(index->Flush());
      }
      if (fail == -1) {
        mutations = env.mutation_count() - start;
        ASSERT_GT(mutations, 0);
        // The successful operation really did discard the dead documents.
        if (compact) {
          EXPECT_FALSE(
              ContainsBlob(*index->GetReadState(), HashBlob("gc-dead-0\n")));
        } else {
          EXPECT_FALSE(ContainsBlob(*index->GetReadState(),
                                    HashBlob("gc-dead-extra\n")));
        }
      }
      env.ClearFault();
      index.reset();
      ASSERT_THAT(env.DropUnsyncedData(), IsOk());
      FERRET_ASSERT_OK_AND_ASSIGN(
          auto recovered, Index::Open(env, path, GcFaultOptions(false)));
      const auto state = recovered.first->GetReadState();
      for (int i = 0; i < 2; ++i) {
        EXPECT_TRUE(
            ContainsBlob(*state, HashBlob(absl::StrCat("gc-live-", i, "\n"))));
      }
      CheckBlobMappings(*state);
      CheckNoGcOrphans(env, path, *state);
    }
  }
}

TEST(IndexTest, FailedMergeAndInstallRetryWithoutLosingDocuments) {
  // Calibrate the mutation count for one merge to inject a failure in the
  // MANIFEST write, rather than in the segment writer.
  TempDir calibration;
  ferret_testing::FaultInjectionEnv calibration_env(Env::Default());
  PrepareSegments(calibration_env, calibration.Join("index"), 2);
  const int64_t before = calibration_env.mutation_count();
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto calibrated, Index::Open(calibration_env, calibration.Join("index"),
                                   SmallCompactionOptions(true)));
  // Keep the retired inputs mapped so reaper deletions do not contribute to
  // the measured number of merge and install mutations.
  auto old_snapshot = calibrated.first->GetReadState();
  calibrated.first->WaitForCompactions();
  ASSERT_EQ(calibrated.first->Stats().completed_compactions, 1u);
  const int64_t merge_mutations = calibration_env.mutation_count() - before;
  calibrated.first.reset();
  old_snapshot.reset();
  ASSERT_GT(merge_mutations, 8);

  for (int64_t fail_after : {int64_t{1}, merge_mutations - 6}) {
    SCOPED_TRACE(fail_after);
    TempDir dir;
    const std::string path = dir.Join("index");
    ferret_testing::FaultInjectionEnv env(Env::Default());
    PrepareSegments(env, path, 2);
    env.FailAfter(fail_after,
                  absl::UnavailableError("injected compaction error"));
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto opened, Index::Open(env, path, SmallCompactionOptions(true)));
    auto index = std::move(opened.first);
    const absl::Time deadline = absl::Now() + absl::Seconds(5);
    while (index->Stats().compaction_failures == 0 && absl::Now() < deadline) {
      absl::SleepFor(absl::Milliseconds(10));
    }
    EXPECT_GT(index->Stats().compaction_failures, 0u);
    EXPECT_EQ(index->Stats().segment_count, 2u);
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto before_retry,
        Search(*index->GetReadState(), {.pattern = "needle"}));
    EXPECT_EQ(before_retry.num_matching_files, 2);
    env.ClearFault();
    index->WaitForCompactions();
    EXPECT_EQ(index->Stats().completed_compactions, 1u);
    EXPECT_EQ(index->Stats().segment_count, 1u);
    index.reset();
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto recovered, Index::Open(env, path, SmallCompactionOptions(false)));
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto result,
        Search(*recovered.first->GetReadState(), {.pattern = "needle"}));
    EXPECT_EQ(result.num_matching_files, 2);
    recovered.first.reset();
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto clean, Index::Open(env, path, SmallCompactionOptions(false)));
    EXPECT_TRUE(clean.second.deleted_files.empty());
  }
}

// Waits for the background lane without depending on a particular merge speed.
void WaitForCompactionFailures(Index& index, uint64_t count) {
  const absl::Time deadline = absl::Now() + absl::Seconds(10);
  while (index.Stats().compaction_failures < count && absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(5));
  }
  ASSERT_GE(index.Stats().compaction_failures, count);
}

TEST(IndexTest, FailedCompactionBacksOff) {
  TempDir dir;
  ferret_testing::FaultInjectionEnv env(Env::Default());
  const std::string path = dir.Join("index");
  PrepareSegments(env, path, 2);
  env.FailAfter(1, absl::UnavailableError("injected merge error"));
  IndexOptions options = SmallCompactionOptions(true);
  options.compaction.retry_initial_delay = absl::Milliseconds(10);
  options.compaction.retry_max_delay = absl::Milliseconds(40);
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened, Index::Open(env, path, options));
  auto index = std::move(opened.first);
  for (uint64_t attempt = 1; attempt <= 3; ++attempt) {
    WaitForCompactionFailures(*index, attempt);
    const IndexStats observed = index->Stats();
    const absl::Duration expected = std::min(
        options.compaction.retry_initial_delay *
            (1 << std::min<uint64_t>(observed.compaction_failures - 1, 2)),
        options.compaction.retry_max_delay);
    EXPECT_EQ(observed.last_compaction_retry_delay, expected);
    if (attempt == 3) {
      EXPECT_EQ(observed.last_compaction_retry_delay,
                options.compaction.retry_max_delay);
    }
  }
  env.ClearFault();
  index->WaitForCompactions();
  EXPECT_EQ(index->Stats().completed_compactions, 1u);
  EXPECT_TRUE(index->Stats().last_compaction_error.ok());
}

TEST(IndexTest, ReopenRequiredDisablesCompactionButNotFlushOrSearch) {
  // Keep retired segments mapped during calibration so the last mutation of
  // the merge is the directory sync after the MANIFEST rename.
  TempDir calibration;
  ferret_testing::FaultInjectionEnv calibration_env(Env::Default());
  PrepareSegments(calibration_env, calibration.Join("index"), 2);
  const int64_t before = calibration_env.mutation_count();
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto calibrated, Index::Open(calibration_env, calibration.Join("index"),
                                   SmallCompactionOptions(true)));
  auto old_snapshot = calibrated.first->GetReadState();
  calibrated.first->WaitForCompactions();
  ASSERT_EQ(calibrated.first->Stats().completed_compactions, 1u);
  const int64_t merge_mutations = calibration_env.mutation_count() - before;
  calibrated.first.reset();
  old_snapshot.reset();
  ASSERT_GT(merge_mutations, 8);

  TempDir dir;
  ferret_testing::FaultInjectionEnv env(Env::Default());
  const std::string path = dir.Join("index");
  PrepareSegments(env, path, 2);
  const absl::Status injected =
      absl::UnavailableError("injected final directory sync error");
  env.FailAfter(merge_mutations - 1, injected);
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened, Index::Open(env, path, SmallCompactionOptions(true)));
  auto index = std::move(opened.first);
  WaitForCompactionFailures(*index, 1);
  const IndexStats failed = index->Stats();
  EXPECT_TRUE(failed.compaction_disabled);
  EXPECT_EQ(failed.compaction_failures, 1u);
  EXPECT_EQ(failed.last_compaction_error, injected);
  index->WaitForCompactions();
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto before_reopen,
      Search(*index->GetReadState(), {.pattern = "needle"}));
  EXPECT_EQ(before_reopen.num_matching_files, 2);

  env.ClearFault();
  ASSERT_THAT(index->Add(Document({"later.cc", "needle later\n"})), IsOk());
  const absl::Status flush_error =
      absl::FailedPreconditionError("reopen required");
  EXPECT_EQ(index->Flush(), flush_error);
  EXPECT_EQ(index->Add(Document({"another.cc", "needle again\n"})),
            flush_error);
  EXPECT_EQ(index->Stats().compaction_failures, 1u);
  index.reset();
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto recovered, Index::Open(env, path, SmallCompactionOptions(false)));
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto result,
      Search(*recovered.first->GetReadState(), {.pattern = "needle"}));
  EXPECT_EQ(result.num_matching_files, 2);
  recovered.first.reset();
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto clean, Index::Open(env, path, SmallCompactionOptions(false)));
  EXPECT_TRUE(clean.second.deleted_files.empty());
}

TEST(IndexTest, ShutdownCancelsMergeAndRemovesPartialOutput) {
  TempDir dir;
  const std::string path = dir.Join("index");
  // A large enough input keeps the merge busy long enough to observe its
  // temporary file without relying on a fixed sleep.
  {
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto opened,
        Index::Open(*Env::Default(), path, SmallCompactionOptions(false)));
    auto index = std::move(opened.first);
    for (int i = 0; i < 400; ++i) {
      ASSERT_THAT(
          index->Add(Document(
              {absl::StrCat("bulk", i, ".cc"),
               absl::StrCat("needle", i, " ", std::string(4096, 'x'), "\n")})),
          IsOk());
      if (i == 199) ASSERT_THAT(index->Flush(), IsOk());
    }
    ASSERT_THAT(index->Flush(), IsOk());
  }
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened,
      Index::Open(*Env::Default(), path, SmallCompactionOptions(true)));
  auto index = std::move(opened.first);
  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  bool observed = false;
  while (absl::Now() < deadline) {
    FERRET_ASSERT_OK_AND_ASSIGN(auto files, Env::Default()->GetChildren(path));
    observed = std::any_of(
        files.begin(), files.end(),
        [](const std::string& file) { return file.ends_with(".fseg.tmp"); });
    if (observed) break;
    absl::SleepFor(absl::Milliseconds(1));
  }
  ASSERT_TRUE(observed);
  EXPECT_EQ(index->Stats().completed_compactions, 0u);
  const absl::Time start = absl::Now();
  index.reset();
  EXPECT_LT(absl::Now() - start, absl::Seconds(3));
  FERRET_ASSERT_OK_AND_ASSIGN(auto files, Env::Default()->GetChildren(path));
  EXPECT_TRUE(std::none_of(
      files.begin(), files.end(),
      [](const std::string& file) { return file.ends_with(".fseg.tmp"); }));
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto recovered,
      Index::Open(*Env::Default(), path, SmallCompactionOptions(false)));
  FERRET_ASSERT_OK_AND_ASSIGN(auto result,
                              Search(*recovered.first->GetReadState(),
                                     {.pattern = "needle", .max_files = 500}));
  EXPECT_EQ(result.num_matching_files, 400);
}

TEST(IndexTest, SyntheticCorpusCompactionSmoke) {
  TempDir dir;
  IndexOptions options = SmallCompactionOptions(true);
  options.memtable.flush_docs = 3;
  options.compaction.max_live_segments = 3;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened, Index::Open(*Env::Default(), dir.Join("index"), options));
  auto index = std::move(opened.first);
  for (int i = 0; i < 120; ++i) {
    ASSERT_THAT(
        index->Add(Document(
            {absl::StrCat("file", i, ".cc"),
             absl::StrCat("needle", i, " ", std::string(150, 'x'), "\n")})),
        IsOk());
  }
  ASSERT_THAT(index->Flush(), IsOk());
  index->WaitForCompactions();
  const IndexStats stats = index->Stats();
  EXPECT_LE(stats.segment_count, 3u);
  EXPECT_GT(stats.flush_bytes, 0u);
  EXPECT_GT(stats.compaction_bytes, 0u);
  RecordProperty("compaction_bytes", absl::StrCat(stats.compaction_bytes));
  RecordProperty("flush_bytes", absl::StrCat(stats.flush_bytes));
  RecordProperty("final_segments", absl::StrCat(stats.segment_count));
}

TEST(IndexTest, ViewEditsSurviveEveryManifestCrashBoundary) {
  const ManifestView view{"/canonical/tree", "/canonical/common"};
  // Manifest-only installs mutate, in order: create MANIFEST.tmp, append,
  // sync its contents, close, rename over MANIFEST, and sync the directory.
  // Calibrate the count rather than assuming how many writes the Env makes.
  for (bool remove : {false, true}) {
    TempDir calibration;
    const std::string path = calibration.Join("index");
    IndexOptions options;
    options.compaction.enabled = false;
    {
      FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                                  Index::Open(*Env::Default(), path, options));
      if (remove) ASSERT_THAT(opened.first->RecordView(view), IsOk());
    }
    ferret_testing::FaultInjectionEnv counter(Env::Default());
    int64_t steps;
    {
      FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                                  Index::Open(counter, path, options));
      const int64_t before = counter.mutation_count();
      ASSERT_THAT(remove ? opened.first->ForgetView(view.canonical_root)
                         : opened.first->RecordView(view),
                  IsOk());
      steps = counter.mutation_count() - before;
    }
    ASSERT_GT(steps, 0);
    for (int64_t boundary = 0; boundary <= steps; ++boundary) {
      SCOPED_TRACE(absl::StrCat(remove ? "forget" : "record",
                                " after mutation ", boundary));
      TempDir dir;
      const std::string index_dir = dir.Join("index");
      {
        FERRET_ASSERT_OK_AND_ASSIGN(
            auto opened, Index::Open(*Env::Default(), index_dir, options));
        if (remove) ASSERT_THAT(opened.first->RecordView(view), IsOk());
      }
      ferret_testing::FaultInjectionEnv fault(Env::Default());
      {
        FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                                    Index::Open(fault, index_dir, options));
        fault.FailAfter(boundary, absl::UnavailableError("simulated crash"));
        if (remove) {
          opened.first->ForgetView(view.canonical_root).IgnoreError();
        } else {
          opened.first->RecordView(view).IgnoreError();
        }
      }
      ASSERT_THAT(fault.DropUnsyncedData(), IsOk());
      FERRET_ASSERT_OK_AND_ASSIGN(
          auto recovered, Index::Open(*Env::Default(), index_dir, options));
      const std::vector<ManifestView> views = recovered.first->Views();
      EXPECT_TRUE(views.empty() || views == std::vector<ManifestView>{view});
    }
  }
}

TEST(IndexTest, ViewRecordAndFlushCrashAtEveryMutation) {
  const ManifestView view{"/canonical/tree", "/canonical/common"};
  IndexOptions options;
  options.compaction.enabled = false;
  for (bool flush_first : {false, true}) {
    // Calibrate both install boundaries on this Env and segment format.
    TempDir calibration;
    ferret_testing::FaultInjectionEnv counter(Env::Default());
    int64_t first_steps;
    int64_t total_steps;
    {
      FERRET_ASSERT_OK_AND_ASSIGN(
          auto opened,
          Index::Open(counter, calibration.Join("index"), options));
      Index& index = *opened.first;
      ASSERT_THAT(index.Add(Document({"file.cc", "crashneedle\n"})), IsOk());
      const auto first = [&] {
        return flush_first ? index.Flush() : index.RecordView(view);
      };
      const auto second = [&] {
        return flush_first ? index.RecordView(view) : index.Flush();
      };
      const int64_t start = counter.mutation_count();
      ASSERT_THAT(first(), IsOk());
      first_steps = counter.mutation_count() - start;
      ASSERT_THAT(second(), IsOk());
      total_steps = counter.mutation_count() - start;
    }
    ASSERT_GT(first_steps, 0);
    ASSERT_GT(total_steps, first_steps);
    for (int64_t boundary = 0; boundary <= total_steps; ++boundary) {
      SCOPED_TRACE(
          absl::StrCat(flush_first ? "flush then record" : "record then flush",
                       " mutation ", boundary));
      TempDir dir;
      const std::string path = dir.Join("index");
      ferret_testing::FaultInjectionEnv fault(Env::Default());
      {
        FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                                    Index::Open(fault, path, options));
        Index& index = *opened.first;
        ASSERT_THAT(index.Add(Document({"file.cc", "crashneedle\n"})), IsOk());
        fault.FailAfter(boundary, absl::UnavailableError("simulated crash"));
        const auto first = [&] {
          return flush_first ? index.Flush() : index.RecordView(view);
        };
        const auto second = [&] {
          return flush_first ? index.RecordView(view) : index.Flush();
        };
        if (first().ok()) second().IgnoreError();
      }
      ASSERT_THAT(fault.DropUnsyncedData(), IsOk());
      FERRET_ASSERT_OK_AND_ASSIGN(auto recovered,
                                  Index::Open(*Env::Default(), path, options));
      // The first MANIFEST rename is not durable before its directory sync.
      // The second sync publishes the whole sequence, not half an edit.
      const bool first_durable = boundary >= first_steps;
      const bool second_durable = boundary >= total_steps;
      EXPECT_EQ(recovered.first->Views(),
                std::vector<ManifestView>(
                    flush_first ? second_durable : first_durable, view));
      EXPECT_EQ(recovered.first->GetReadState()->version->segments.size(),
                size_t{flush_first ? first_durable : second_durable});
    }
  }
}

TEST(IndexTest, ViewEditAndFlushSerializeWithoutLosingEither) {
  TempDir dir;
  IndexOptions options;
  options.compaction.enabled = false;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened, Index::Open(*Env::Default(), dir.Join("index"), options));
  Index& index = *opened.first;
  ASSERT_THAT(index.Add(Document({"file.cc", "durableneedle\n"})), IsOk());
  const ManifestView view{"/canonical/tree", "/canonical/common"};
  absl::Status installed;
  std::thread record([&] { installed = index.RecordView(view); });
  ASSERT_THAT(index.Flush(), IsOk());
  record.join();
  ASSERT_THAT(installed, IsOk());
  EXPECT_EQ(index.Views(), std::vector<ManifestView>{view});
  opened.first.reset();
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto recovered, Index::Open(*Env::Default(), dir.Join("index"), options));
  EXPECT_EQ(recovered.first->Views(), std::vector<ManifestView>{view});
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto result,
      Search(*recovered.first->GetReadState(), {.pattern = "durableneedle"}));
  EXPECT_EQ(result.num_matching_files, 1);
}

TEST(IndexTest, SymbolOidDedupChoosesBestRankedCopy) {
  TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              Index::Open(*Env::Default(), dir.path()));
  auto index = std::move(opened.first);
  const std::string content = "int BestRank() { return 3; }\n";
  CorpusFile low{"third_party/low.cc", content};
  CorpusFile high{"src/high.cc", content};
  ASSERT_THAT(index
                  ->AddSegment([&](SegmentWriter& writer) {
                    return writer.Add(Document(low));
                  })
                  .status(),
              IsOk());
  ASSERT_THAT(index->Add(Document(high)), IsOk());
  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  while (index->Stats().pending_symbols != 0 && absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(1));
  }
  EXPECT_EQ(index->Stats().pending_symbols, 0u);
  const SymbolLookup found =
      index->GetReadState()->FindSymbol("BestRank", false);
  ASSERT_EQ(found.occurrences.size(), 1u);
  EXPECT_EQ(found.occurrences[0].doc.hint_path, "src/high.cc");
}

TEST(IndexTest, SymbolExtractionFailureAttachesEmptyAndPersists) {
  TempDir dir;
  IndexOptions options;
  auto counter = std::make_shared<std::atomic<uint64_t>>(0);
  options.symbol_extractions = counter;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened, Index::Open(*Env::Default(), dir.path(), options));
  auto index = std::move(opened.first);
  // Larger than the extractor's 4 MiB limit, but below the memtable's
  // configured hard content limit. Text indexing is unaffected.
  const std::string content = absl::StrCat(
      "int OversizeSymbol() {}\n", std::string(size_t{4} * 1024 * 1024, 'a'));
  ASSERT_THAT(index->Add(Document({"oversize.cc", content})), IsOk());
  ASSERT_THAT(index->Flush(), IsOk());
  EXPECT_EQ(index->Stats().failed_symbols, 1u);
  EXPECT_EQ(index->Stats().pending_symbols, 0u);
  EXPECT_EQ(counter->load(), 1u);
  const auto state = index->GetReadState();
  ASSERT_EQ(state->version->segments.size(), 1u);
  EXPECT_TRUE(state->version->segments[0].segment->doc(0).symbols_failed);
  EXPECT_TRUE(state->FindSymbol("OversizeSymbol", false).occurrences.empty());
  index.reset();
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto reopened, Index::Open(*Env::Default(), dir.path(), options));
  EXPECT_EQ(reopened.first->Stats().failed_symbols, 1u);
  EXPECT_EQ(counter->load(), 1u);
}

TEST(IndexTest, FlushWaitsForPendingSymbols) {
  TempDir dir;
  IndexOptions options;
  auto gate = std::make_shared<absl::Notification>();
  auto reached = std::make_shared<absl::Notification>();
  options.symbol_extraction_gate = gate;
  options.symbol_extraction_reached = reached;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened, Index::Open(*Env::Default(), dir.path(), options));
  auto index = std::move(opened.first);
  absl::Cleanup release([gate] {
    if (!gate->HasBeenNotified()) gate->Notify();
  });
  ASSERT_THAT(index->Add(Document({"pending.cc", "int FlushSymbol() {}\n"})),
              IsOk());
  ASSERT_TRUE(reached->WaitForNotificationWithTimeout(absl::Seconds(5)));
  std::atomic<bool> finished{false};
  absl::Status flush_status;
  std::thread flusher([&] {
    flush_status = index->Flush();
    finished.store(true);
  });
  absl::SleepFor(absl::Milliseconds(50));
  EXPECT_FALSE(finished.load());
  gate->Notify();
  flusher.join();
  EXPECT_THAT(flush_status, IsOk());
  const SymbolLookup found =
      index->GetReadState()->FindSymbol("FlushSymbol", false);
  ASSERT_EQ(found.occurrences.size(), 1u);
  EXPECT_TRUE(found.pending_blobs.empty());
}

TEST(IndexTest, ShutdownCancelsBlockedSymbolExtraction) {
  TempDir dir;
  IndexOptions options;
  options.memtable.flush_docs = 1;
  auto gate = std::make_shared<absl::Notification>();
  auto reached = std::make_shared<absl::Notification>();
  auto done = std::make_shared<absl::Notification>();
  options.symbol_extraction_gate = gate;
  options.symbol_extraction_reached = reached;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened, Index::Open(*Env::Default(), dir.path(), options));
  auto index = std::move(opened.first);
  ASSERT_THAT(index->Add(Document({"blocked.cc", "int BlockedSymbol() {}\n"})),
              IsOk());
  ASSERT_TRUE(reached->WaitForNotificationWithTimeout(absl::Seconds(5)));
  std::thread shutdown([index = std::move(index), done]() mutable {
    index.reset();
    done->Notify();
  });
  const bool stopped = done->WaitForNotificationWithTimeout(absl::Seconds(1));
  // Always release the gate before joining, even when the assertion fails.
  gate->Notify();
  shutdown.join();
  EXPECT_TRUE(stopped);
}

TEST(IndexTest, SymbolsAttachAfterTextIsSearchableAndSurviveRestart) {
  TempDir dir;
  IndexOptions options;
  auto gate = std::make_shared<absl::Notification>();
  auto reached = std::make_shared<absl::Notification>();
  auto counter = std::make_shared<std::atomic<uint64_t>>(0);
  options.symbol_extractions = counter;
  options.symbol_extraction_gate = gate;
  options.symbol_extraction_reached = reached;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto opened, Index::Open(*Env::Default(), dir.path(), options));
  auto index = std::move(opened.first);
  ASSERT_THAT(index->Add(Document({"file.cc", "int Lookup() { return 7; }\n"})),
              IsOk());
  ASSERT_TRUE(reached->WaitForNotificationWithTimeout(absl::Seconds(5)));
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto text, Search(*index->GetReadState(), {.pattern = "Lookup"}));
  EXPECT_EQ(text.num_matching_files, 1);
  const SymbolLookup before =
      index->GetReadState()->FindSymbol("Lookup", false);
  EXPECT_TRUE(before.occurrences.empty());
  EXPECT_EQ(before.pending_blobs.size(), 1u);
  EXPECT_EQ(index->Stats().pending_symbols, 1u);
  gate->Notify();
  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  while (index->Stats().pending_symbols != 0 && absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(1));
  }
  EXPECT_EQ(index->Stats().pending_symbols, 0u);
  EXPECT_EQ(counter->load(), 1u);
  const SymbolLookup attached =
      index->GetReadState()->FindSymbol("Lookup", false);
  ASSERT_EQ(attached.occurrences.size(), 1u);
  EXPECT_EQ(attached.occurrences[0].line, 1u);
  EXPECT_TRUE(attached.occurrences[0].is_definition);
  ASSERT_THAT(index->Flush(), IsOk());
  index.reset();
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto recovered, Index::Open(*Env::Default(), dir.path(), options));
  EXPECT_EQ(recovered.first->Stats().pending_symbols, 0u);
  EXPECT_EQ(counter->load(), 1u);
  const SymbolLookup durable =
      recovered.first->GetReadState()->FindSymbol("Lookup", false);
  ASSERT_EQ(durable.occurrences.size(), 1u);
  EXPECT_EQ(durable.occurrences[0].line, 1u);
}

}  // namespace
}  // namespace ferret

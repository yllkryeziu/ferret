// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/bulk_build.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/git/blob_hash.h"
#include "ferret/git/testing/git_repo_builder.h"
#include "ferret/index/prepared_document.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/segment.h"
#include "ferret/index/segment_writer.h"
#include "ferret/postings/ids.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/text/eligibility.h"
#include "ferret/text/language.h"
#include "ferret/text/trigram_extractor.h"

namespace ferret {
namespace {

using ::absl_testing::StatusIs;
using ::ferret_testing::GitRepoBuilder;
using ::ferret_testing::TempDir;
using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

std::vector<std::string> HintPaths(const Segment& segment) {
  std::vector<std::string> paths;
  paths.reserve(segment.num_docs());
  for (DocId d = 0; d < segment.num_docs(); ++d) {
    paths.emplace_back(segment.doc(d).hint_path);
  }
  return paths;
}

class BulkBuildTest : public ::testing::Test {
 protected:
  void SetUp() override {
    FERRET_ASSERT_OK_AND_ASSIGN(repo_, GitRepoBuilder::Init(temp_dir_, "repo"));
  }

  std::string SegmentPath() const { return temp_dir_.Join("out.fseg"); }

  TempDir temp_dir_;
  std::optional<GitRepoBuilder> repo_;

  GitRepoBuilder& repo() {
    CHECK(repo_.has_value());
    return *repo_;
  }
};

TEST_F(BulkBuildTest, IndexesEligibleWorktreeFilesInRankOrder) {
  ABSL_ASSERT_OK(repo().WriteFile(".gitignore", "*.log\n"));
  ABSL_ASSERT_OK(repo().WriteFile("src/net/socket.cc", "int Connect();\n"));
  ABSL_ASSERT_OK(repo().WriteFile("src/net/socket_test.cc", "TEST(A, B) {}\n"));
  ABSL_ASSERT_OK(repo().WriteFile("third_party/z/z.c", "int z;\n"));
  ABSL_ASSERT_OK(repo().WriteFile("main.cc", "int main() {}\n"));
  ABSL_ASSERT_OK(repo().WriteFile("empty.txt", ""));
  ABSL_ASSERT_OK(repo().WriteFile("image.bin", std::string("\x89PNG\0\0", 6)));
  ABSL_ASSERT_OK(repo().WriteFile("big.txt", std::string(2000, 'x')));
  ABSL_ASSERT_OK(repo().WriteFile("gone.txt", "deleted after commit\n"));
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().Commit("initial"));
  ABSL_ASSERT_OK(repo().DeleteFile("gone.txt"));
  ABSL_ASSERT_OK(repo().WriteFile("untracked.py", "print(1)\n"));
  ABSL_ASSERT_OK(repo().WriteFile("debug.log", "ignored\n"));

  BulkBuildOptions options;
  options.eligibility.max_file_bytes = 1000;
  options.num_threads = 3;
  FERRET_ASSERT_OK_AND_ASSIGN(
      const BulkBuildStats stats,
      BuildSegmentFromWorktree(*Env::Default(), repo().root(), SegmentPath(),
                               options));
  EXPECT_EQ(stats.files_listed, 10);
  EXPECT_EQ(stats.files_indexed, 6);
  EXPECT_EQ(stats.files_skipped[static_cast<size_t>(SkipReason::kEmpty)], 1);
  EXPECT_EQ(stats.files_skipped[static_cast<size_t>(SkipReason::kContainsNul)],
            1);
  EXPECT_EQ(stats.files_skipped[static_cast<size_t>(SkipReason::kTooLarge)], 1);
  EXPECT_EQ(stats.files_unreadable, 1);
  EXPECT_EQ(stats.segment.num_docs, 6u);

  FERRET_ASSERT_OK_AND_ASSIGN(auto segment,
                              Segment::Open(*Env::Default(), SegmentPath()));
  // Top-level source files first (equal priors, so their order is by Oid),
  // then deeper source, the test, and third-party code.
  const std::vector<std::string> paths = HintPaths(*segment);
  ASSERT_EQ(paths.size(), 6u);
  EXPECT_THAT(std::vector<std::string>(paths.begin(), paths.begin() + 3),
              UnorderedElementsAre(".gitignore", "main.cc", "untracked.py"));
  EXPECT_THAT(std::vector<std::string>(paths.begin() + 3, paths.end()),
              ElementsAre("src/net/socket.cc", "src/net/socket_test.cc",
                          "third_party/z/z.c"));
  for (DocId d = 0; d + 1 < segment->num_docs(); ++d) {
    const SegmentDoc a = segment->doc(d);
    const SegmentDoc b = segment->doc(d + 1);
    EXPECT_LT((RankKey{a.tier, a.prior, a.oid}),
              (RankKey{b.tier, b.prior, b.oid}));
  }
  const auto main = std::find(paths.begin(), paths.end(), "main.cc");
  const SegmentDoc main_doc =
      segment->doc(static_cast<DocId>(main - paths.begin()));
  EXPECT_EQ(main_doc.content, "int main() {}\n");
  EXPECT_EQ(main_doc.oid, HashBlob("int main() {}\n"));
  EXPECT_EQ(main_doc.language, Language::kCpp);
  EXPECT_EQ(segment->doc(4).tier, RankTier::kTest);
  EXPECT_EQ(segment->doc(5).tier, RankTier::kThirdParty);
  int64_t bytes = 0;
  for (DocId d = 0; d < segment->num_docs(); ++d) {
    bytes += static_cast<int64_t>(segment->doc(d).content.size());
  }
  EXPECT_EQ(stats.bytes_indexed, bytes);
}

// Copies of one blob at several paths each get a document, ordered by path
// when their rank keys tie.
TEST_F(BulkBuildTest, DuplicateBlobsAreSeparateDocuments) {
  ABSL_ASSERT_OK(repo().WriteFile("b/LICENSE", "same\n"));
  ABSL_ASSERT_OK(repo().WriteFile("a/LICENSE", "same\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      const BulkBuildStats stats,
      BuildSegmentFromWorktree(*Env::Default(), repo().root(), SegmentPath()));
  EXPECT_EQ(stats.files_indexed, 2);
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment,
                              Segment::Open(*Env::Default(), SegmentPath()));
  EXPECT_THAT(HintPaths(*segment), ElementsAre("a/LICENSE", "b/LICENSE"));
  EXPECT_EQ(segment->doc(0).oid, segment->doc(1).oid);
}

TEST_F(BulkBuildTest, ExplicitPathsToleratesMissingFiles) {
  ABSL_ASSERT_OK(repo().WriteFile("a.txt", "alpha\n"));
  const std::vector<std::string> paths = {"a.txt", "missing.txt"};
  FERRET_ASSERT_OK_AND_ASSIGN(
      const BulkBuildStats stats,
      BuildSegment(*Env::Default(), repo().root(), paths, SegmentPath()));
  EXPECT_EQ(stats.files_listed, 2);
  EXPECT_EQ(stats.files_indexed, 1);
  EXPECT_EQ(stats.files_unreadable, 1);
}

TEST_F(BulkBuildTest, PreparedDocumentsProduceIdenticalSegmentBytes) {
  const std::vector<std::string> paths = {"z.cc", "a.py", "readme.md"};
  const std::vector<std::string> bytes = {"int example() {}\n", "print(42)\n",
                                          "# Example\n"};
  for (size_t i = 0; i < paths.size(); ++i) {
    ABSL_ASSERT_OK(repo().WriteFile(paths[i], bytes[i]));
  }
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto built,
      BuildSegment(*Env::Default(), repo().root(), paths, SegmentPath()));
  TrigramExtractor extractor;
  struct RankedFile {
    size_t index;
    PreparedDocument prepared;
  };
  std::vector<RankedFile> ordered;
  ordered.reserve(paths.size());
  for (size_t i = 0; i < paths.size(); ++i) {
    ordered.push_back({i, PrepareDocument(paths[i], bytes[i], {}, extractor)});
  }
  std::sort(ordered.begin(), ordered.end(),
            [&](const RankedFile& a, const RankedFile& b) {
              return std::tie(a.prepared.rank_key, paths[a.index]) <
                     std::tie(b.prepared.rank_key, paths[b.index]);
            });
  const std::string direct_path = temp_dir_.Join("prepared.fseg");
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto writer, SegmentWriter::Create(*Env::Default(), direct_path));
  for (const RankedFile& file : ordered) {
    ABSL_ASSERT_OK(
        writer->Add(file.prepared.Bind(paths[file.index], bytes[file.index])));
  }
  FERRET_ASSERT_OK_AND_ASSIGN(auto written, writer->Finish());
  EXPECT_EQ(written.file_bytes, built.segment.file_bytes);
  FERRET_ASSERT_OK_AND_ASSIGN(auto bulk_bytes,
                              Env::Default()->ReadFile(SegmentPath()));
  FERRET_ASSERT_OK_AND_ASSIGN(auto direct_bytes,
                              Env::Default()->ReadFile(direct_path));
  EXPECT_EQ(bulk_bytes, direct_bytes);
}

TEST_F(BulkBuildTest, SingleThreaded) {
  ABSL_ASSERT_OK(repo().WriteFile("a.txt", "alpha\n"));
  BulkBuildOptions options;
  options.num_threads = 1;
  FERRET_ASSERT_OK_AND_ASSIGN(
      const BulkBuildStats stats,
      BuildSegmentFromWorktree(*Env::Default(), repo().root(), SegmentPath(),
                               options));
  EXPECT_EQ(stats.files_indexed, 1);
}

TEST_F(BulkBuildTest, Errors) {
  BulkBuildOptions options;
  options.num_threads = 0;
  EXPECT_THAT(
      BuildSegment(*Env::Default(), repo().root(), {}, SegmentPath(), options)
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument));
  // A failure after the scan has started the symbol workers returns with
  // them joined: a directory where the segment goes fails its creation.
  ABSL_ASSERT_OK(repo().WriteFile("src/a.cc", "int main() { return 0; }\n"));
  ABSL_ASSERT_OK(temp_dir_.MakeDirs("seg_is_a_directory"));
  for (const int threads : {1, 4}) {
    options.num_threads = threads;
    EXPECT_FALSE(BuildSegment(*Env::Default(), repo().root(), {"src/a.cc"},
                              temp_dir_.Join("seg_is_a_directory"), options)
                     .ok());
  }
  ABSL_ASSERT_OK(temp_dir_.MakeDirs("not_a_repo"));
  EXPECT_THAT(BuildSegmentFromWorktree(
                  *Env::Default(), temp_dir_.Join("not_a_repo"), SegmentPath())
                  .status(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace ferret

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/worktree_scanner.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/git/blob_hash.h"
#include "ferret/git/testing/git_repo_builder.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"

namespace ferret {
namespace {

using ::absl_testing::StatusIs;
using ::ferret_testing::GitRepoBuilder;
using ::ferret_testing::TempDir;

class WorktreeScannerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ferret_testing::IsolateGitEnvironment(temp_.Join("home"));
    FERRET_ASSERT_OK_AND_ASSIGN(repo_, GitRepoBuilder::Init(temp_, "repo"));
  }

  absl::StatusOr<WorktreeScan> Scan() {
    return ScanWorktree(repo().root(), {.verify_hint_fraction = 1.0});
  }

  const ScannedWorktreeFile* Find(const WorktreeScan& scan,
                                  absl::string_view path) const {
    for (const auto& file : scan.files) {
      if (file.path == path) return &file;
    }
    return nullptr;
  }

  TempDir temp_;
  std::optional<GitRepoBuilder> repo_;

  GitRepoBuilder& repo() {
    CHECK(repo_.has_value());
    return *repo_;
  }
};

TEST_F(WorktreeScannerTest, CleanFilesAreHintedAndMatchTheirBytes) {
  ABSL_ASSERT_OK(repo().WriteFile("a", "hello\n"));
  ABSL_ASSERT_OK(repo().WriteFile("dir/b", "hi\n"));
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().Commit("initial"));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeScan scan, Scan());
  ASSERT_EQ(scan.report.hinted, 2);
  EXPECT_EQ(scan.report.files, 2);
  for (const auto& file : scan.files) {
    ASSERT_TRUE(file.skip_hint);
    ASSERT_TRUE(file.size);
    ASSERT_TRUE(file.mtime);
    FERRET_ASSERT_OK_AND_ASSIGN(
        std::string bytes, Env::Default()->ReadFile(repo().Path(file.path)));
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access): Asserted above.
    EXPECT_EQ(*file.skip_hint, HashBlob(bytes));
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access): Asserted above.
    EXPECT_EQ(*file.size, static_cast<int64_t>(bytes.size()));
  }
}

TEST_F(WorktreeScannerTest, DirtyDeletedUntrackedAndUnmergedPathsMustRead) {
  ABSL_ASSERT_OK(repo().WriteFile("dirty", "a"));
  ABSL_ASSERT_OK(repo().WriteFile("deleted", "a"));
  ABSL_ASSERT_OK(repo().WriteFile("conflict", "base"));
  ABSL_ASSERT_OK(repo().WriteFile(".gitignore", "*.log\n"));
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().Commit("base"));
  ABSL_ASSERT_OK(repo().Git({"branch", "other"}));
  ABSL_ASSERT_OK(repo().WriteFile("conflict", "ours"));
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().Commit("ours"));
  ABSL_ASSERT_OK(repo().Git({"switch", "other"}));
  ABSL_ASSERT_OK(repo().WriteFile("conflict", "theirs"));
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().Commit("theirs"));
  ABSL_ASSERT_OK(repo().RunGit({"merge", "--no-edit", "main"}));
  ABSL_ASSERT_OK(repo().WriteFile("dirty", "new"));
  ABSL_ASSERT_OK(repo().DeleteFile("deleted"));
  ABSL_ASSERT_OK(repo().WriteFile("untracked", "hi"));
  ABSL_ASSERT_OK(repo().WriteFile("ignored.log", "no"));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeScan scan, Scan());
  ASSERT_NE(Find(scan, "dirty"), nullptr);
  EXPECT_EQ(Find(scan, "dirty")->read_reason, ScanReadReason::kDirty);
  EXPECT_EQ(Find(scan, "deleted")->read_reason, ScanReadReason::kDirty);
  EXPECT_FALSE(Find(scan, "deleted")->size);
  EXPECT_EQ(Find(scan, "conflict")->read_reason, ScanReadReason::kUnmerged);
  EXPECT_EQ(Find(scan, "untracked")->read_reason, ScanReadReason::kUntracked);
  EXPECT_EQ(Find(scan, "ignored.log"), nullptr);
  EXPECT_EQ(scan.report.unmerged, 1);
  EXPECT_EQ(scan.report.untracked, 1);
}

TEST_F(WorktreeScannerTest, SparseAssumeUnchangedSymlinkAndGitlink) {
  ABSL_ASSERT_OK(repo().WriteFile("sparse", "a"));
  ABSL_ASSERT_OK(repo().WriteFile("assumed", "a"));
  ABSL_ASSERT_OK(repo().Symlink("assumed", "link"));
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().AddGitlink("vendor/sub",
                                   "0123456789abcdef0123456789abcdef01234567"));
  ABSL_ASSERT_OK(repo().Commit("base"));
  ABSL_ASSERT_OK(repo().SetSkipWorktree("sparse"));
  ABSL_ASSERT_OK(repo().SetAssumeUnchanged("assumed"));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeScan scan, Scan());
  EXPECT_EQ(Find(scan, "assumed")->read_reason,
            ScanReadReason::kAssumeUnchanged);
  EXPECT_EQ(Find(scan, "sparse"), nullptr);
  EXPECT_EQ(Find(scan, "link"), nullptr);
  EXPECT_EQ(Find(scan, "vendor/sub"), nullptr);
  EXPECT_EQ(scan.report.skip_worktree, 1);
  EXPECT_EQ(scan.report.symlink, 1);
  EXPECT_EQ(scan.report.gitlink, 1);
  EXPECT_EQ(scan.nested_repositories, std::vector<std::string>{"vendor/sub"});
}

TEST_F(WorktreeScannerTest, NestedRepositoriesAreReportedNotEntered) {
  ABSL_ASSERT_OK(repo().WriteFile("top.cc", "a"));
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().AddGitlink("vendor/sub",
                                   "0123456789abcdef0123456789abcdef01234567"));
  ABSL_ASSERT_OK(repo().Commit("base"));
  // Files inside a gitlink's directory belong to the submodule.
  ABSL_ASSERT_OK(repo().WriteFile("vendor/sub/inside.cc", "b"));
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder nested,
                              GitRepoBuilder::Init(temp_, "repo/lib/nested"));
  ABSL_ASSERT_OK(nested.WriteFile("inner.cc", "c"));
  ABSL_ASSERT_OK(repo().WriteFile("lib/untracked.cc", "d"));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeScan scan, Scan());
  EXPECT_EQ(scan.nested_repositories,
            (std::vector<std::string>{"lib/nested", "vendor/sub"}));
  std::vector<std::string> paths;
  paths.reserve(scan.files.size());
  for (const ScannedWorktreeFile& file : scan.files) paths.push_back(file.path);
  EXPECT_EQ(paths, (std::vector<std::string>{"lib/untracked.cc", "top.cc"}));
}

TEST_F(WorktreeScannerTest, GlobalCrlfDisablesHints) {
  ABSL_ASSERT_OK(repo().WriteFile("plain", "hi\n"));
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().Commit("base"));
  ABSL_ASSERT_OK(repo().SetConfig("core.autocrlf", "true"));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeScan scan, Scan());
  EXPECT_EQ(Find(scan, "plain")->read_reason, ScanReadReason::kConversion);
  EXPECT_EQ(scan.report.hinted, 0);
}

TEST_F(WorktreeScannerTest, AttributesDisableOnlyMatchingHints) {
  for (const char* path : {"plain", "crlf", "lfs", "id", "encoded"}) {
    ABSL_ASSERT_OK(repo().WriteFile(path, "hello\n"));
  }
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().Commit("base"));
  ABSL_ASSERT_OK(
      repo().WriteFile(".gitattributes",
                       "crlf eol=crlf\nlfs filter=lfs\nid ident\nencoded "
                       "working-tree-encoding=UTF-16\n"));
  ABSL_ASSERT_OK(repo().Add({".gitattributes"}));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeScan scan, Scan());
  for (const char* path : {"crlf", "lfs", "id", "encoded"}) {
    EXPECT_EQ(Find(scan, path)->read_reason, ScanReadReason::kConversion)
        << path;
  }
  EXPECT_TRUE(Find(scan, "plain")->skip_hint);
  EXPECT_TRUE(Find(scan, ".gitattributes")->skip_hint);
  EXPECT_EQ(scan.report.conversion, 4);
}

TEST_F(WorktreeScannerTest, NativeCheckoutConfigAndNestedAttributes) {
  ABSL_ASSERT_OK(repo().WriteFile("src/plain", "a\n"));
  ABSL_ASSERT_OK(repo().WriteFile("src/converted", "b\n"));
  ABSL_ASSERT_OK(repo().WriteFile("other/converted", "c\n"));
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().Commit("base"));
  ABSL_ASSERT_OK(repo().WriteFile("src/.gitattributes", "converted ident\n"));
  ABSL_ASSERT_OK(repo().Add({"src/.gitattributes"}));
  ABSL_ASSERT_OK(repo().SetConfig("core.autocrlf", "input"));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeScan scan, Scan());
  EXPECT_EQ(Find(scan, "src/converted")->read_reason,
            ScanReadReason::kConversion);
  EXPECT_TRUE(Find(scan, "src/plain")->skip_hint);
  EXPECT_TRUE(Find(scan, "other/converted")->skip_hint);
  ABSL_ASSERT_OK(repo().SetConfig("core.eol", "crlf"));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeScan crlf, Scan());
  EXPECT_EQ(crlf.report.hinted, 0);
}

TEST_F(WorktreeScannerTest, AttributeMacroAndInfoAttributesDisableHints) {
  ABSL_ASSERT_OK(repo().WriteFile("macro", "hello\n"));
  ABSL_ASSERT_OK(repo().WriteFile("info", "hello\n"));
  ABSL_ASSERT_OK(repo().WriteFile("plain", "hello\n"));
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().Commit("base"));
  ABSL_ASSERT_OK(
      repo().WriteFile(".gitattributes", "[attr]custom ident\nmacro custom\n"));
  ABSL_ASSERT_OK(repo().Add({".gitattributes"}));
  ABSL_ASSERT_OK(repo().WriteFile(".git/info/attributes", "info filter=lfs\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeScan scan, Scan());
  EXPECT_EQ(Find(scan, "macro")->read_reason, ScanReadReason::kConversion);
  EXPECT_EQ(Find(scan, "info")->read_reason, ScanReadReason::kConversion);
  EXPECT_TRUE(Find(scan, "plain")->skip_hint);
}

TEST_F(WorktreeScannerTest, GlobalAttributesSelectOnlyMatchingPaths) {
  for (const char* path : {"a.txt", "b.txt", "binary.bin", "plain"}) {
    ABSL_ASSERT_OK(repo().WriteFile(path, "hello\n"));
  }
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().Commit("base"));
  ABSL_ASSERT_OK(
      temp_.WriteFile("home/.config/git/attributes", "*.txt eol=crlf\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeScan scan, Scan());
  for (const char* path : {"a.txt", "b.txt"}) {
    EXPECT_EQ(Find(scan, path)->read_reason, ScanReadReason::kConversion);
  }
  for (const char* path : {"binary.bin", "plain"}) {
    EXPECT_TRUE(Find(scan, path)->skip_hint);
  }
  EXPECT_EQ(scan.report.attribute_candidates, 2);
  EXPECT_EQ(scan.report.hinted, 2);
}

TEST_F(WorktreeScannerTest, ConfiguredGlobalAttributesSelectMatchingPaths) {
  ABSL_ASSERT_OK(repo().WriteFile("data.bin", "hello\n"));
  ABSL_ASSERT_OK(repo().WriteFile("plain", "hello\n"));
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().Commit("base"));
  const std::string attributes = temp_.Join("home/custom-attributes");
  ABSL_ASSERT_OK(
      temp_.WriteFile("home/custom-attributes", "*.bin filter=lfs\n"));
  ABSL_ASSERT_OK(repo().SetConfig("core.attributesFile", attributes));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeScan scan, Scan());
  EXPECT_EQ(Find(scan, "data.bin")->read_reason, ScanReadReason::kConversion);
  EXPECT_TRUE(Find(scan, "plain")->skip_hint);
  EXPECT_EQ(scan.report.attribute_candidates, 1);
}

TEST_F(WorktreeScannerTest, NestedUncertainAttributesStayInSubtree) {
  for (const char* path : {"src/a", "src/b", "outside/a"}) {
    ABSL_ASSERT_OK(repo().WriteFile(path, "hello\n"));
  }
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().Commit("base"));
  ABSL_ASSERT_OK(
      repo().WriteFile("src/.gitattributes", "[attr]custom ident\na ident\n"));
  ABSL_ASSERT_OK(repo().Add({"src/.gitattributes"}));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeScan scan, Scan());
  EXPECT_EQ(scan.report.attribute_candidates, 3);
  EXPECT_EQ(scan.report.conversion, 1);
  EXPECT_TRUE(Find(scan, "outside/a")->skip_hint);
}

TEST_F(WorktreeScannerTest, DiskNonregularIsNotAnIndexSymlink) {
  ABSL_ASSERT_OK(repo().WriteFile("file", "hello"));
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().Commit("base"));
  ABSL_ASSERT_OK(repo().Rename("file", "other"));
  ABSL_ASSERT_OK(repo().Symlink("other", "file"));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeScan scan, Scan());
  EXPECT_EQ(Find(scan, "file"), nullptr);
  EXPECT_EQ(scan.report.not_regular, 1);
  EXPECT_EQ(scan.report.symlink, 0);
}

TEST_F(WorktreeScannerTest, DiskDirectoryIsNotAnIndexSymlink) {
  ABSL_ASSERT_OK(repo().WriteFile("file", "hello"));
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().Commit("base"));
  ABSL_ASSERT_OK(repo().Rename("file", "other"));
  ABSL_ASSERT_OK(temp_.MakeDirs("repo/file"));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeScan scan, Scan());
  EXPECT_EQ(Find(scan, "file"), nullptr);
  EXPECT_EQ(scan.report.not_regular, 1);
  EXPECT_EQ(scan.report.symlink, 0);
}

TEST_F(WorktreeScannerTest, CanonicalAndTrailingSlashRootsKeepHints) {
  ABSL_ASSERT_OK(repo().WriteFile("case", "hello"));
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().Commit("base"));
  struct stat a, b;
  if (stat(repo().Path("case").c_str(), &a) != 0 ||
      stat(repo().Path("CASE").c_str(), &b) != 0 || a.st_ino != b.st_ino) {
    GTEST_SKIP() << "temp volume is case-sensitive";
  }
  const std::string alias = temp_.Join("alias");
  ASSERT_EQ(symlink(repo().root().c_str(), alias.c_str()), 0);
  for (const std::string& root :
       {alias, absl::StrCat(alias, "/"), temp_.Join("REPO")}) {
    FERRET_ASSERT_OK_AND_ASSIGN(
        WorktreeScan scan, ScanWorktree(root, {.verify_hint_fraction = 1.0}));
    EXPECT_TRUE(Find(scan, "case")->skip_hint);
    EXPECT_EQ(scan.report.hinted, 1);
    EXPECT_EQ(scan.report.dirty, 0);
  }
}

TEST_F(WorktreeScannerTest, InvalidOptionsAreRejected) {
  EXPECT_THAT(ScanWorktree(repo().root(), {.verify_hint_fraction = 2.0}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ScanWorktree(repo().root(), {.lstat_threads = 0}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(WorktreeScannerTest, RejectsSha256Repositories) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      GitRepoBuilder sha,
      GitRepoBuilder::Init(temp_, "sha", {.object_format = "sha256"}));
  EXPECT_THAT(ScanWorktree(sha.root(), {.verify_hint_fraction = 1.0}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(WorktreeScannerTest, CaseMismatchIsADeletionOnInsensitiveVolumes) {
  ABSL_ASSERT_OK(repo().WriteFile("case", "one"));
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().Commit("base"));
  struct stat a, b;
  if (stat(repo().Path("case").c_str(), &a) != 0 ||
      stat(repo().Path("CASE").c_str(), &b) != 0 || a.st_ino != b.st_ino) {
    GTEST_SKIP() << "temp volume is case-sensitive";
  }
  ABSL_ASSERT_OK(repo().Rename("case", "CASE"));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeScan scan, Scan());
  const ScannedWorktreeFile* file = Find(scan, "case");
  ASSERT_NE(file, nullptr);
  EXPECT_FALSE(file->size);
  EXPECT_FALSE(file->skip_hint);
  EXPECT_EQ(file->read_reason, ScanReadReason::kDirty);
}

TEST_F(WorktreeScannerTest,
       CaseCollisionsKeepDiskSpellingOnInsensitiveVolumes) {
  ABSL_ASSERT_OK(repo().WriteFile("case", "one"));
  ABSL_ASSERT_OK(repo().Add());
  ABSL_ASSERT_OK(repo().Commit("base"));
  struct stat a, b;
  if (stat(repo().Path("case").c_str(), &a) != 0 ||
      stat(repo().Path("CASE").c_str(), &b) != 0 || a.st_ino != b.st_ino) {
    GTEST_SKIP() << "temp volume is case-sensitive";
  }
  ABSL_ASSERT_OK(
      repo().Git({"update-index", "--add", "--cacheinfo",
                  absl::StrCat("100644,", HashBlob("two").ToHex(), ",CASE")}));
  FERRET_ASSERT_OK_AND_ASSIGN(WorktreeScan scan, Scan());
  EXPECT_EQ(scan.report.case_collisions, 1);
  EXPECT_NE(Find(scan, "case"), nullptr);
  EXPECT_EQ(Find(scan, "CASE"), nullptr);
}

}  // namespace
}  // namespace ferret

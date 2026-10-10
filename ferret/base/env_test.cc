// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/base/env.h"

#include <fcntl.h>
#include <limits.h>
#include <sys/fcntl.h>
#include <sys/syslimits.h>
#include <unistd.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/mapped_file.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"

namespace ferret {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

class EnvTest : public ::testing::Test {
 protected:
  Env& env_ = *Env::Default();
  ferret_testing::TempDir dir_;
};

TEST_F(EnvTest, DefaultIsASingleton) {
  EXPECT_EQ(Env::Default(), Env::Default());
}

TEST_F(EnvTest, WritableFileAppendsOverwritesAndSyncs) {
  const std::string path = dir_.Join("f");
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WritableFile> file,
                              env_.NewWritableFile(path));
  EXPECT_EQ(file->size(), 0);
  ABSL_ASSERT_OK(file->Append("hello "));
  ABSL_ASSERT_OK(file->Append("world"));
  EXPECT_EQ(file->size(), 11);
  ABSL_ASSERT_OK(file->WriteAt(0, "HELLO"));
  EXPECT_EQ(file->size(), 11);
  ABSL_ASSERT_OK(file->Sync());
  ABSL_ASSERT_OK(file->FullSync());
  ABSL_ASSERT_OK(file->Close());
  EXPECT_THAT(env_.ReadFile(path), IsOkAndHolds("HELLO world"));
}

TEST_F(EnvTest, WriteAtRejectsBytesPastTheEnd) {
  const std::string path = dir_.Join("f");
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WritableFile> file,
                              env_.NewWritableFile(path));
  ABSL_ASSERT_OK(file->Append("0123456789"));
  EXPECT_THAT(file->WriteAt(8, "xyz"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr(absl::StrCat("pwrite(", path, ")"))));
  EXPECT_THAT(file->WriteAt(-1, "x"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(file->WriteAt(10, ""), IsOk());
  EXPECT_THAT(file->WriteAt(7, "xyz"), IsOk());
  EXPECT_EQ(file->size(), 10);
  ABSL_ASSERT_OK(file->Close());
  EXPECT_THAT(env_.ReadFile(path), IsOkAndHolds("0123456xyz"));
}

TEST_F(EnvTest, ClosedFileRejectsEveryCall) {
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WritableFile> file,
                              env_.NewWritableFile(dir_.Join("f")));
  ABSL_ASSERT_OK(file->Append("x"));
  ABSL_ASSERT_OK(file->Close());
  const auto closed = StatusIs(absl::StatusCode::kFailedPrecondition,
                               HasSubstr("already closed"));
  EXPECT_THAT(file->Append("y"), closed);
  EXPECT_THAT(file->WriteAt(0, "y"), closed);
  EXPECT_THAT(file->Sync(), closed);
  EXPECT_THAT(file->FullSync(), closed);
  EXPECT_THAT(file->Close(), closed);
}

TEST_F(EnvTest, NewWritableFileTruncates) {
  ABSL_ASSERT_OK(dir_.WriteFile("f", "old content"));
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WritableFile> file,
                              env_.NewWritableFile(dir_.Join("f")));
  ABSL_ASSERT_OK(file->Append("new"));
  ABSL_ASSERT_OK(file->Close());
  EXPECT_THAT(env_.ReadFile(dir_.Join("f")), IsOkAndHolds("new"));
}

TEST_F(EnvTest, NewWritableFileInMissingDirectoryIsNotFound) {
  const std::string path = dir_.Join("no/such/f");
  EXPECT_THAT(env_.NewWritableFile(path),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr(absl::StrCat("open(", path, ")"))));
}

TEST_F(EnvTest, ReadFileReadsEmptyAndLargeFiles) {
  ABSL_ASSERT_OK(dir_.WriteFile("empty", ""));
  EXPECT_THAT(env_.ReadFile(dir_.Join("empty")), IsOkAndHolds(""));

  std::string large;
  for (int i = 0; large.size() < (size_t{3} << 20); ++i) {
    absl::StrAppend(&large, i, ",");
  }
  ABSL_ASSERT_OK(dir_.WriteFile("large", large));
  EXPECT_THAT(env_.ReadFile(dir_.Join("large")), IsOkAndHolds(large));
}

TEST_F(EnvTest, ReadFileIfAtMostReadsOnlyFilesWithinTheLimit) {
  ABSL_ASSERT_OK(dir_.WriteFile("ten", "0123456789"));
  ABSL_ASSERT_OK(dir_.WriteFile("empty", ""));
  const std::string ten = dir_.Join("ten");
  EXPECT_THAT(env_.ReadFileIfAtMost(ten, 10),
              IsOkAndHolds(std::optional<std::string>("0123456789")));
  EXPECT_THAT(env_.ReadFileIfAtMost(ten, 9),
              IsOkAndHolds(std::optional<std::string>()));
  EXPECT_THAT(env_.ReadFileIfAtMost(dir_.Join("empty"), 0),
              IsOkAndHolds(std::optional<std::string>("")));
  EXPECT_THAT(env_.ReadFileIfAtMost(dir_.Join("missing"), 10),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(EnvTest, ReadFileOfMissingFileNamesThePath) {
  const std::string path = dir_.Join("missing");
  EXPECT_THAT(
      env_.ReadFile(path),
      StatusIs(absl::StatusCode::kNotFound,
               HasSubstr(absl::StrCat("open(", path, "): No such file"))));
}

TEST_F(EnvTest, NewMappedFileMapsTheContent) {
  ABSL_ASSERT_OK(dir_.WriteFile("f", "mapped"));
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MappedFile> file,
                              env_.NewMappedFile(dir_.Join("f")));
  EXPECT_EQ(file->data(), "mapped");
}

TEST_F(EnvTest, RenameAtomicallyReplacesTheTarget) {
  ABSL_ASSERT_OK(dir_.WriteFile("MANIFEST", "old"));
  ABSL_ASSERT_OK(dir_.WriteFile("MANIFEST.tmp", "new"));
  ABSL_ASSERT_OK(
      env_.RenameFile(dir_.Join("MANIFEST.tmp"), dir_.Join("MANIFEST")));
  EXPECT_THAT(env_.ReadFile(dir_.Join("MANIFEST")), IsOkAndHolds("new"));
  EXPECT_THAT(env_.FileExists(dir_.Join("MANIFEST.tmp")), IsOkAndHolds(false));
}

TEST_F(EnvTest, RenameOfMissingFileNamesBothPaths) {
  const std::string from = dir_.Join("a");
  const std::string to = dir_.Join("b");
  EXPECT_THAT(
      env_.RenameFile(from, to),
      StatusIs(absl::StatusCode::kNotFound,
               HasSubstr(absl::StrCat("rename(", from, ", ", to, ")"))));
}

TEST_F(EnvTest, GetChildrenListsSortedNames) {
  ABSL_ASSERT_OK(dir_.WriteFile("b", ""));
  ABSL_ASSERT_OK(dir_.WriteFile("a", ""));
  ABSL_ASSERT_OK(dir_.WriteFile("d/nested", ""));
  ABSL_ASSERT_OK(dir_.WriteFile("C", ""));
  EXPECT_THAT(env_.GetChildren(dir_.path()),
              IsOkAndHolds(ElementsAre("C", "a", "b", "d")));
  EXPECT_THAT(env_.GetChildren(dir_.Join("d")),
              IsOkAndHolds(ElementsAre("nested")));
}

TEST_F(EnvTest, GetChildrenFailsOnMissingDirectoryAndOnFiles) {
  const std::string missing = dir_.Join("missing");
  EXPECT_THAT(env_.GetChildren(missing),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr(missing)));
  ABSL_ASSERT_OK(dir_.WriteFile("f", ""));
  EXPECT_THAT(env_.GetChildren(dir_.Join("f")),
              StatusIs(::testing::_, HasSubstr(dir_.Join("f"))));
}

TEST_F(EnvTest, DeleteFileRemovesTheEntry) {
  const std::string path = dir_.Join("f");
  ABSL_ASSERT_OK(dir_.WriteFile("f", "x"));
  ABSL_ASSERT_OK(env_.DeleteFile(path));
  EXPECT_THAT(env_.FileExists(path), IsOkAndHolds(false));
  EXPECT_THAT(env_.DeleteFile(path),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr(absl::StrCat("unlink(", path, ")"))));
}

TEST_F(EnvTest, FileExistsAndGetFileSize) {
  ABSL_ASSERT_OK(dir_.WriteFile("f", "12345"));
  EXPECT_THAT(env_.FileExists(dir_.Join("f")), IsOkAndHolds(true));
  EXPECT_THAT(env_.FileExists(dir_.path()), IsOkAndHolds(true));
  EXPECT_THAT(env_.FileExists(dir_.Join("missing")), IsOkAndHolds(false));
  EXPECT_THAT(env_.FileExists(dir_.Join("f/child")), IsOkAndHolds(false));
  EXPECT_THAT(env_.GetFileSize(dir_.Join("f")), IsOkAndHolds(5));
  EXPECT_THAT(env_.GetFileSize(dir_.Join("missing")),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("stat(")));
}

TEST_F(EnvTest, CreateDirCreatesParentsAndIsIdempotent) {
  const std::string path = dir_.Join("a/b/c");
  ABSL_ASSERT_OK(env_.CreateDir(path));
  EXPECT_THAT(env_.FileExists(path), IsOkAndHolds(true));
  EXPECT_THAT(env_.CreateDir(path), IsOk());
  EXPECT_THAT(env_.CreateDir(dir_.Join("a/b/c/")), IsOk());
  EXPECT_THAT(env_.CreateDir(dir_.path()), IsOk());
}

TEST_F(EnvTest, CreateDirFailsWhereAFileIsInTheWay) {
  ABSL_ASSERT_OK(dir_.WriteFile("f", ""));
  const std::string file = dir_.Join("f");
  EXPECT_THAT(env_.CreateDir(file),
              StatusIs(absl::StatusCode::kAlreadyExists,
                       HasSubstr(absl::StrCat("mkdir(", file, ")"))));
  EXPECT_THAT(env_.CreateDir(dir_.Join("f/sub")),
              StatusIs(::testing::_, HasSubstr("mkdir(")));
}

TEST_F(EnvTest, SyncDirSyncsExistingDirectoriesOnly) {
  EXPECT_THAT(env_.SyncDir(dir_.path()), IsOk());
  const std::string missing = dir_.Join("missing");
  EXPECT_THAT(env_.SyncDir(missing),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr(absl::StrCat("open(", missing, ")"))));
}

TEST_F(EnvTest, LockFileExcludesASecondLockInTheSameProcess) {
  const std::string path = dir_.Join("LOCK");
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<FileLock> lock,
                              env_.LockFile(path));
  EXPECT_THAT(env_.FileExists(path), IsOkAndHolds(true));
  // A second open file description conflicts even within one process.
  EXPECT_THAT(env_.LockFile(path),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr(absl::StrCat("flock(", path, ")"))));
  lock.reset();
  EXPECT_THAT(env_.LockFile(path), IsOk());
}

TEST_F(EnvTest, LockFileInMissingDirectoryIsNotFound) {
  EXPECT_THAT(env_.LockFile(dir_.Join("missing/LOCK")),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("open(")));
}

#if defined(F_GETPATH)
// Returns the descriptors of this process that refer to `path`.
std::vector<int> DescriptorsOf(const std::string& path) {
  std::vector<int> fds;
  for (int fd = 0; fd < getdtablesize(); ++fd) {
    char fd_path[PATH_MAX];
    if (fcntl(fd, F_GETPATH, fd_path) == 0 && fd_path == path) {
      fds.push_back(fd);
    }
  }
  return fds;
}

TEST_F(EnvTest, DescriptorsAreCloseOnExec) {
  const std::string file_path = dir_.Join("f");
  const std::string lock_path = dir_.Join("LOCK");
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WritableFile> file,
                              env_.NewWritableFile(file_path));
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<FileLock> lock,
                              env_.LockFile(lock_path));
  for (const std::string& path : {file_path, lock_path}) {
    const std::vector<int> fds = DescriptorsOf(path);
    ASSERT_EQ(fds.size(), 1u) << path;
    EXPECT_NE(fcntl(fds[0], F_GETFD) & FD_CLOEXEC, 0) << path;
  }
  ABSL_ASSERT_OK(file->Close());
  lock.reset();
  EXPECT_THAT(DescriptorsOf(file_path), ElementsAre());
  EXPECT_THAT(DescriptorsOf(lock_path), ElementsAre());
}
#endif

}  // namespace
}  // namespace ferret

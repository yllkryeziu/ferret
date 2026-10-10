// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/base/mapped_file.h"

#include <fcntl.h>
#include <limits.h>
#include <sys/fcntl.h>
#include <sys/syslimits.h>
#include <unistd.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/system_info.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"

namespace ferret {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;

constexpr MappedFileAdvice kAllAdvice[] = {
    MappedFileAdvice::kWillNeed, MappedFileAdvice::kRandom,
    MappedFileAdvice::kSequential, MappedFileAdvice::kDontNeed};

// Content spanning a few pages, with a partial last page.
std::string MultiPageContent() {
  std::string content;
  for (int i = 0; static_cast<int64_t>(content.size()) < 3 * PageSize() + 100;
       ++i) {
    absl::StrAppend(&content, "line ", i, "\n");
  }
  return content;
}

TEST(MappedFileTest, MapsTheWholeFile) {
  ferret_testing::TempDir dir;
  const std::string content = MultiPageContent();
  ABSL_ASSERT_OK(dir.WriteFile("f", content));
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MappedFile> file,
                              MappedFile::Open(dir.Join("f")));
  EXPECT_EQ(file->data(), content);
}

TEST(MappedFileTest, EmptyFileHasEmptyData) {
  ferret_testing::TempDir dir;
  ABSL_ASSERT_OK(dir.WriteFile("empty", ""));
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MappedFile> file,
                              MappedFile::Open(dir.Join("empty")));
  EXPECT_TRUE(file->data().empty());
  EXPECT_THAT(file->Advise(0, 0, MappedFileAdvice::kWillNeed), IsOk());
  EXPECT_THAT(file->Advise(0, 1, MappedFileAdvice::kWillNeed),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MappedFileTest, MissingFileIsNotFound) {
  ferret_testing::TempDir dir;
  const std::string path = dir.Join("missing");
  EXPECT_THAT(MappedFile::Open(path),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr(absl::StrCat("open(", path, ")"))));
}

TEST(MappedFileTest, DirectoryIsNotMappable) {
  ferret_testing::TempDir dir;
  EXPECT_THAT(MappedFile::Open(dir.path()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("not a regular file")));
}

TEST(MappedFileTest, MappingSurvivesDeletionWithoutADescriptor) {
  ferret_testing::TempDir dir;
  const std::string content = MultiPageContent();
  const std::string path = dir.Join("f");
  ABSL_ASSERT_OK(dir.WriteFile("f", content));
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MappedFile> file,
                              MappedFile::Open(path));
#if defined(F_GETPATH)
  for (int fd = 0; fd < getdtablesize(); ++fd) {
    char fd_path[PATH_MAX];
    if (fcntl(fd, F_GETPATH, fd_path) == 0) {
      EXPECT_NE(fd_path, path) << "descriptor " << fd << " is still open";
    }
  }
#endif
  ASSERT_EQ(unlink(path.c_str()), 0);
  EXPECT_EQ(file->data(), content);
}

TEST(MappedFileTest, AdviseWidensUnalignedRangesToPages) {
  ferret_testing::TempDir dir;
  const std::string content = MultiPageContent();
  ABSL_ASSERT_OK(dir.WriteFile("f", content));
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MappedFile> file,
                              MappedFile::Open(dir.Join("f")));
  const int64_t size = static_cast<int64_t>(content.size());
  const int64_t page = PageSize();
  for (const MappedFileAdvice advice : kAllAdvice) {
    SCOPED_TRACE(static_cast<int>(advice));
    EXPECT_THAT(file->Advise(0, size, advice), IsOk());
    EXPECT_THAT(file->Advise(1, size - 2, advice), IsOk());
    EXPECT_THAT(file->Advise(page - 1, 2, advice), IsOk());
    EXPECT_THAT(file->Advise(size - 1, 1, advice), IsOk());
    EXPECT_THAT(file->Advise(size, 0, advice), IsOk());
  }
  // Advice never changes the bytes, even after kDontNeed.
  EXPECT_EQ(file->data(), content);
}

TEST(MappedFileTest, AdviseRejectsRangesOutsideTheFile) {
  ferret_testing::TempDir dir;
  const std::string path = dir.Join("f");
  ABSL_ASSERT_OK(dir.WriteFile("f", "0123456789"));
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MappedFile> file,
                              MappedFile::Open(path));
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  constexpr struct {
    int64_t offset;
    int64_t length;
  } kBadRanges[] = {{-1, 1}, {0, -1},   {11, 0},  {5, 6},
                    {0, 11}, {1, kMax}, {kMax, 1}};
  for (const auto& range : kBadRanges) {
    EXPECT_THAT(
        file->Advise(range.offset, range.length, MappedFileAdvice::kWillNeed),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 AllOf(HasSubstr(path), HasSubstr("file size 10"))))
        << "offset " << range.offset << " length " << range.length;
  }
}

TEST(MappedFileDeathTest, ViewFaultsAfterItsFileIsDestroyed) {
#if defined(NDEBUG) && !defined(ABSL_HAVE_ADDRESS_SANITIZER)
  GTEST_SKIP() << "the stale-view trap exists only in debug and ASan builds";
#else
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ferret_testing::TempDir dir;
  ABSL_ASSERT_OK(dir.WriteFile("f", "stale"));
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MappedFile> file,
                              MappedFile::Open(dir.Join("f")));
  const char* const stale = file->data().data();
  file.reset();
  EXPECT_DEATH(
      {
        const volatile char byte = *stale;
        static_cast<void>(byte);
      },
      "");
#endif
}

}  // namespace
}  // namespace ferret

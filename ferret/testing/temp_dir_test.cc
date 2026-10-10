// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/testing/temp_dir.h"

#include <_stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/syslimits.h>
#include <unistd.h>

#include <cstddef>
#include <cstdlib>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/match.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ferret {
namespace {

using ::absl_testing::IsOk;
using ::ferret_testing::TempDir;

bool IsDirectory(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool Exists(const std::string& path) {
  struct stat st;
  return lstat(path.c_str(), &st) == 0;
}

std::string ReadAll(const std::string& path) {
  FILE* const file = fopen(path.c_str(), "rb");
  if (file == nullptr) return "<missing>";
  std::string content;
  char buffer[256];
  size_t n = 0;
  while ((n = fread(buffer, 1, sizeof(buffer), file)) > 0) {
    content.append(buffer, n);
  }
  fclose(file);
  return content;
}

TEST(TempDirTest, CreatesAFreshCanonicalDirectoryUnderTestTmpdir) {
  const TempDir dir;
  EXPECT_TRUE(IsDirectory(dir.path()));
  char resolved[PATH_MAX];
  ASSERT_NE(realpath(dir.path().c_str(), resolved), nullptr);
  EXPECT_EQ(dir.path(), resolved);
  EXPECT_FALSE(absl::EndsWith(dir.path(), "/"));

  const char* test_tmpdir = std::getenv("TEST_TMPDIR");
  ASSERT_NE(test_tmpdir, nullptr) << "run under bazel test";
  ASSERT_NE(realpath(test_tmpdir, resolved), nullptr);
  EXPECT_TRUE(absl::StartsWith(dir.path(), std::string(resolved) + "/"));
}

TEST(TempDirTest, EachInstanceGetsItsOwnDirectory) {
  const TempDir a;
  const TempDir b;
  EXPECT_NE(a.path(), b.path());
}

TEST(TempDirTest, JoinAppendsARelativePath) {
  const TempDir dir;
  EXPECT_EQ(dir.Join("a/b"), dir.path() + "/a/b");
}

TEST(TempDirTest, WriteFileCreatesParentsAndReplacesContent) {
  const TempDir dir;
  ASSERT_THAT(dir.WriteFile("x/y/z.txt", "first"), IsOk());
  EXPECT_TRUE(IsDirectory(dir.Join("x/y")));
  EXPECT_EQ(ReadAll(dir.Join("x/y/z.txt")), "first");
  ASSERT_THAT(dir.WriteFile("x/y/z.txt", "2nd"), IsOk());
  EXPECT_EQ(ReadAll(dir.Join("x/y/z.txt")), "2nd");
}

TEST(TempDirTest, MakeDirsIsIdempotent) {
  const TempDir dir;
  ASSERT_THAT(dir.MakeDirs("p/q/r"), IsOk());
  EXPECT_TRUE(IsDirectory(dir.Join("p/q/r")));
  EXPECT_THAT(dir.MakeDirs("p/q/r"), IsOk());
}

TEST(TempDirTest, MakeDirsFailsWhereAFileIsInTheWay) {
  const TempDir dir;
  ASSERT_THAT(dir.WriteFile("f", ""), IsOk());
  EXPECT_FALSE(dir.MakeDirs("f/sub").ok());
}

TEST(TempDirTest, DestructorRemovesEverythingWithoutFollowingSymlinks) {
  const TempDir outside;
  ASSERT_THAT(outside.WriteFile("keep", "kept"), IsOk());
  std::string path;
  {
    const TempDir dir;
    path = dir.path();
    ASSERT_THAT(dir.WriteFile("a/b/c", "x"), IsOk());
    ASSERT_THAT(dir.MakeDirs("empty/nested"), IsOk());
    ASSERT_EQ(symlink(outside.path().c_str(), dir.Join("link").c_str()), 0);
  }
  EXPECT_FALSE(Exists(path));
  EXPECT_EQ(ReadAll(outside.Join("keep")), "kept");
}

}  // namespace
}  // namespace ferret

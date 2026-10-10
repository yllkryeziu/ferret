// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/cli/command_util.h"

#include <_stdlib.h>

#include <cstdint>
#include <cstdlib>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"

namespace ferret {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::MatchesRegex;

class DefaultIndexPathTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ABSL_ASSERT_OK(temp_.MakeDirs("work/repo"));
    ::setenv("XDG_CACHE_HOME", temp_.Join("cache").c_str(), 1);
    ::setenv("HOME", temp_.Join("home").c_str(), 1);
  }

  ferret_testing::TempDir temp_;
};

TEST_F(DefaultIndexPathTest, UsesXdgCacheHome) {
  EXPECT_THAT(DefaultIndexPath(temp_.Join("work/repo")),
              IsOkAndHolds(MatchesRegex(temp_.Join("cache") +
                                        "/ferret/repo-[0-9a-f]{12}\\.fseg")));
}

TEST_F(DefaultIndexPathTest, FallsBackToHome) {
  ::unsetenv("XDG_CACHE_HOME");
  EXPECT_THAT(
      DefaultIndexPath(temp_.Join("work/repo")),
      IsOkAndHolds(MatchesRegex(temp_.Join("home") +
                                "/.cache/ferret/repo-[0-9a-f]{12}\\.fseg")));
  ::unsetenv("HOME");
  EXPECT_THAT(DefaultIndexPath(temp_.Join("work/repo")),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(DefaultIndexPathTest, OnePathPerWorktree) {
  ABSL_ASSERT_OK(temp_.MakeDirs("work/other/repo"));
  FERRET_ASSERT_OK_AND_ASSIGN(const std::string path,
                              DefaultIndexPath(temp_.Join("work/repo")));
  EXPECT_THAT(DefaultIndexPath(temp_.Join("work/./other/../repo/")),
              IsOkAndHolds(path));
  EXPECT_THAT(DefaultIndexPath(temp_.Join("work/other/repo")),
              IsOkAndHolds(testing::Ne(path)));
}

TEST_F(DefaultIndexPathTest, MissingWorktree) {
  EXPECT_THAT(DefaultIndexPath(temp_.Join("absent")),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(ResolveIndexPathTest, ExplicitPathWins) {
  EXPECT_THAT(ResolveIndexPath("/does/not/exist", "/tmp/x.fseg"),
              IsOkAndHolds("/tmp/x.fseg"));
}

TEST(OpenIndexTest, MissingIndexSuggestsIndexing) {
  ferret_testing::TempDir temp;
  EXPECT_THAT(OpenIndex(temp.Join("none.fseg")),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("ferret index")));
}

TEST(HumanBytesTest, Units) {
  EXPECT_EQ(HumanBytes(0), "0 B");
  EXPECT_EQ(HumanBytes(1023), "1023 B");
  EXPECT_EQ(HumanBytes(1024), "1.0 KiB");
  EXPECT_EQ(HumanBytes(1536), "1.5 KiB");
  EXPECT_EQ(HumanBytes(int64_t{5} << 30), "5.0 GiB");
  EXPECT_EQ(HumanBytes(int64_t{3} << 50), "3072.0 TiB");
}

}  // namespace
}  // namespace ferret

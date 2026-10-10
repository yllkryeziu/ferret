// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/path_glob.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ferret {
namespace {

using ::absl_testing::StatusIs;

bool Matches(const std::string& glob, const std::string& path) {
  absl::StatusOr<PathGlob> parsed = PathGlob::Parse(glob);
  EXPECT_TRUE(parsed.ok()) << glob;
  return parsed.ok() && parsed->Matches(path);
}

TEST(PathGlobTest, NameGlobsMatchAtAnyDepth) {
  EXPECT_TRUE(Matches("*.h", "a.h"));
  EXPECT_TRUE(Matches("*.h", "src/net/a.h"));
  EXPECT_FALSE(Matches("*.h", "src/a.hh"));
  EXPECT_TRUE(Matches("BUILD", "src/BUILD"));
  EXPECT_FALSE(Matches("BUILD", "src/BUILD.bazel"));
}

TEST(PathGlobTest, NameGlobsSelectDirectories) {
  EXPECT_TRUE(Matches("testdata", "a/testdata/x/y.txt"));
  EXPECT_TRUE(Matches("third_party/", "third_party/zlib/z.c"));
  EXPECT_TRUE(Matches("third_party/", "chrome/third_party/x.cc"));
  // A trailing '/' never matches a file.
  EXPECT_FALSE(Matches("notes/", "notes"));
}

TEST(PathGlobTest, SlashGlobsAreAnchoredAtTheRoot) {
  EXPECT_TRUE(Matches("src/*.cc", "src/a.cc"));
  EXPECT_FALSE(Matches("src/*.cc", "src/net/a.cc"));
  EXPECT_FALSE(Matches("src/*.cc", "x/src/a.cc"));
  EXPECT_TRUE(Matches("src/**/*.cc", "src/net/a.cc"));
  EXPECT_TRUE(Matches("src/**/*.cc", "src/a.cc"));
  EXPECT_TRUE(Matches("/docs/*.md", "docs/a.md"));
  EXPECT_TRUE(Matches("src/net", "src/net/a.cc"));
  EXPECT_FALSE(Matches("src/net", "src/network/a.cc"));
}

TEST(PathGlobTest, BangInverts) {
  EXPECT_FALSE(Matches("!*_test.cc", "a/b_test.cc"));
  EXPECT_TRUE(Matches("!*_test.cc", "a/b.cc"));
  EXPECT_FALSE(Matches("!third_party", "third_party/x.c"));
}

TEST(PathGlobTest, CaseSensitive) { EXPECT_FALSE(Matches("*.H", "a.h")); }

TEST(PathGlobTest, RejectsEmptyGlobs) {
  EXPECT_THAT(PathGlob::Parse("").status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(PathGlob::Parse("!").status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(PathGlob::Parse(std::string("a\0b", 3)).status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace ferret

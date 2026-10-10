// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "bench/ferret/file_list.h"

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/git/testing/git_repo_builder.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/text/eligibility.h"

namespace ferret_bench {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

TEST(ParseFileListTest, ReadsOnePathPerLine) {
  EXPECT_THAT(ParseFileList("a.c\nsrc/b.h\n"),
              IsOkAndHolds(ElementsAre("a.c", "src/b.h")));
  EXPECT_THAT(ParseFileList("a.c"), IsOkAndHolds(ElementsAre("a.c")));
}

TEST(ParseFileListTest, RejectsPathsTheEnginesCouldResolveDifferently) {
  const auto error = [](absl::string_view text, absl::string_view message) {
    EXPECT_THAT(
        ParseFileList(text),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr(message)))
        << text;
  };
  error("", "empty file list");
  error("a.c\n\nb.c\n", "line 2: empty path");
  error("a.c\r\n", "CR or NUL");
  error(std::string("a\0.c", 4), "CR or NUL");
  error("/etc/passwd", "not a clean relative path");
  error("../a.c", "not a clean relative path");
  error("src//a.c", "not a clean relative path");
  error("./a.c", "not a clean relative path");
  error("src/", "not a clean relative path");
  error("a.c\nb.c\na.c\n", "line 3: \"a.c\" duplicates line 1");
}

TEST(FormatFileListTest, RoundTrips) {
  const std::vector<std::string> paths = {"a.c", "src/b.h"};
  EXPECT_EQ(FormatFileList(paths), "a.c\nsrc/b.h\n");
  EXPECT_THAT(ParseFileList(FormatFileList(paths)), IsOkAndHolds(paths));
}

TEST(FilterByPathGlobTest, AppliesFerretsOwnGlobSemantics) {
  const std::vector<std::string> paths = {
      "Documentation/networking/index.rst",
      "Documentation/networking/dsa/dsa.rst",
      "Documentation/admin/index.rst",
      "net/core/dev.c",
      "net/core/dev_test.c",
      "include/net/tcp.h",
  };
  EXPECT_THAT(
      FilterByPathGlob(paths, "Documentation/networking/*"),
      IsOkAndHolds(ElementsAre("Documentation/networking/index.rst",
                               "Documentation/networking/dsa/dsa.rst")));
  EXPECT_THAT(FilterByPathGlob(paths, "*.h"),
              IsOkAndHolds(ElementsAre("include/net/tcp.h")));
  EXPECT_THAT(
      FilterByPathGlob(paths, "net/**/*.c"),
      IsOkAndHolds(ElementsAre("net/core/dev.c", "net/core/dev_test.c")));
  EXPECT_THAT(FilterByPathGlob(paths, "!*_test.c"),
              IsOkAndHolds(ElementsAre("Documentation/networking/index.rst",
                                       "Documentation/networking/dsa/dsa.rst",
                                       "Documentation/admin/index.rst",
                                       "net/core/dev.c", "include/net/tcp.h")));
  EXPECT_THAT(FilterByPathGlob(paths, "nothing/*"), IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(FilterByPathGlob(paths, ""),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ListEligibleFilesTest, AppliesThePolicy) {
  const ferret_testing::TempDir temp_dir;
  FERRET_ASSERT_OK_AND_ASSIGN(
      const ferret_testing::GitRepoBuilder repo,
      ferret_testing::GitRepoBuilder::Init(temp_dir, "repo"));
  ABSL_ASSERT_OK(
      repo.WriteFile("src/main.c", "int main(void) { return 0; }\n"));
  ABSL_ASSERT_OK(repo.WriteFile("a.bin", std::string("\x7f"
                                                     "ELF\0\0",
                                                     6)));
  ABSL_ASSERT_OK(repo.WriteFile("tiny", "x"));
  ABSL_ASSERT_OK(repo.WriteFile("empty", ""));
  ABSL_ASSERT_OK(repo.Add());
  ABSL_ASSERT_OK(repo.Commit("files"));

  const ferret::EligibilityOptions zoekt = {
      .policy = ferret::EligibilityPolicy::kZoekt};
  FERRET_ASSERT_OK_AND_ASSIGN(
      const EligibleFiles files,
      ListEligibleFiles(*ferret::Env::Default(), repo.root(), zoekt));
  EXPECT_THAT(files.paths, ElementsAre("src/main.c"));
  EXPECT_EQ(files.files_listed, 4);
  const auto skipped = [&](ferret::SkipReason reason) {
    return files.files_skipped[static_cast<size_t>(reason)];
  };
  EXPECT_EQ(skipped(ferret::SkipReason::kContainsNul), 1);
  EXPECT_EQ(skipped(ferret::SkipReason::kTooSmall), 1);
  EXPECT_EQ(skipped(ferret::SkipReason::kEmpty), 1);
}

}  // namespace
}  // namespace ferret_bench

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/cli/commands.h"

#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/git/testing/git_repo_builder.h"
#include "ferret/search/formatter.h"
#include "ferret/search/search_types.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"

namespace ferret {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::ferret_testing::GitRepoBuilder;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::StartsWith;

class CommandsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    FERRET_ASSERT_OK_AND_ASSIGN(repo_, GitRepoBuilder::Init(temp_, "repo"));
    ABSL_ASSERT_OK(repo().WriteFile("src/net/socket.cc",
                                    "int Connect(int fd) {\n"
                                    "  return connect(fd);\n"
                                    "}\n"));
    ABSL_ASSERT_OK(repo().WriteFile("src/net/socket_test.cc",
                                    "TEST(Socket, Connect) {}\n"));
    ABSL_ASSERT_OK(repo().WriteFile("README.md", "Connect to the server.\n"));
    ABSL_ASSERT_OK(repo().Add());
  }

  std::string IndexPath() const { return temp_.Join("index/repo.fseg"); }

  std::string Index() {
    std::ostringstream out;
    const absl::Status status = RunIndexCommand(
        {.root = repo().root(), .index_path = IndexPath()}, out);
    EXPECT_TRUE(status.ok()) << status;
    return out.str();
  }

  // Runs a search and returns its output, or the error.
  absl::StatusOr<std::string> Search(SearchRequest request,
                                     OutputShape shape = OutputShape::kContext,
                                     bool* matched = nullptr) {
    std::ostringstream out;
    std::ostringstream err;
    absl::StatusOr<bool> result =
        RunSearchCommand({.index_path = IndexPath(),
                          .request = std::move(request),
                          .format = {.shape = shape}},
                         out, err);
    if (!result.ok()) return result.status();
    if (matched != nullptr) *matched = *result;
    return out.str();
  }

  ferret_testing::TempDir temp_;
  std::optional<GitRepoBuilder> repo_;

  GitRepoBuilder& repo() {
    CHECK(repo_.has_value());
    return *repo_;
  }
};

TEST_F(CommandsTest, IndexThenSearch) {
  EXPECT_THAT(Index(), StartsWith("Indexed 3 of 3 files"));
  bool matched = false;
  FERRET_ASSERT_OK_AND_ASSIGN(
      const std::string out,
      Search({.pattern = "connect"}, OutputShape::kContext, &matched));
  EXPECT_TRUE(matched);
  EXPECT_THAT(out, StartsWith("4 matching lines in 3 files\n"));
  // Source outranks its test.
  EXPECT_LT(out.find("src/net/socket.cc\n"),
            out.find("src/net/socket_test.cc\n"));
  EXPECT_THAT(
      out,
      HasSubstr("1:int Connect(int fd) {\n2:  return connect(fd);\n3-}\n"));
}

TEST_F(CommandsTest, NoMatch) {
  Index();
  bool matched = true;
  EXPECT_THAT(
      Search({.pattern = "disconnect"}, OutputShape::kContext, &matched),
      IsOkAndHolds("0 matching lines in 0 files\n"));
  EXPECT_FALSE(matched);
}

TEST_F(CommandsTest, ReindexReplacesTheIndex) {
  Index();
  ABSL_ASSERT_OK(repo().WriteFile("src/new.cc", "void Reconnect();\n"));
  Index();
  EXPECT_THAT(Search({.pattern = "reconnect"}, OutputShape::kLocations),
              IsOkAndHolds("src/new.cc:1\n"));
  FERRET_ASSERT_OK_AND_ASSIGN(std::vector<std::string> files,
                              Env::Default()->GetChildren(temp_.Join("index")));
  EXPECT_THAT(files, testing::ElementsAre("repo.fseg"));
}

TEST_F(CommandsTest, SearchWithoutIndex) {
  EXPECT_THAT(Search({.pattern = "connect"}),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("ferret index")));
}

TEST_F(CommandsTest, SearchRejectsBadPattern) {
  Index();
  EXPECT_THAT(Search({.pattern = "(", .regex = true}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(CommandsTest, Plan) {
  std::ostringstream out;
  ABSL_ASSERT_OK(RunPlanCommand({.pattern = {.pattern = "Connect",
                                             .literal = true,
                                             .case_sensitive = false}},
                                out));
  EXPECT_THAT(out.str(), StartsWith("pattern       \"Connect\" (literal, "
                                    "case-insensitive)\nquery         "));
  EXPECT_THAT(out.str(), HasSubstr("literal       \"connect\"\n"));
  EXPECT_THAT(out.str(), Not(HasSubstr("candidates")));
}

TEST_F(CommandsTest, PlanExplainCountsCandidates) {
  Index();
  std::ostringstream out;
  ABSL_ASSERT_OK(RunPlanCommand({.pattern = {.pattern = "connect",
                                             .literal = true,
                                             .case_sensitive = false},
                                 .explain = true,
                                 .index_path = IndexPath()},
                                out));
  EXPECT_THAT(out.str(), HasSubstr("index         3 docs, 3 chunks\n"));
  EXPECT_THAT(out.str(), HasSubstr("  \"con\"  3 chunks\n"));
  EXPECT_THAT(out.str(), HasSubstr("candidates    3 chunks in 3 files ("));
}

TEST_F(CommandsTest, PlanRejectsBadPattern) {
  std::ostringstream out;
  EXPECT_THAT(RunPlanCommand({.pattern = {.pattern = "a(b"}}, out),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace ferret

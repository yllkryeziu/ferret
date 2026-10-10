// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Runs the ferret binary: flags, exit statuses, and the default index path.

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/subprocess.h"
#include "ferret/git/testing/git_repo_builder.h"
#include "ferret/rpc/daemon.h"
#include "ferret/rpc/daemon_test_guard.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"

namespace ferret {
namespace {

using ::ferret_testing::GitRepoBuilder;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::testing::StartsWith;

// The binary, relative to the test's runfiles directory.
constexpr char kFerret[] = "ferret/cli/ferret";

std::vector<std::string> Lines(const std::string& text) {
  return absl::StrSplit(text, '\n', absl::SkipEmpty());
}

class FerretCliTest : public ::testing::Test {
 protected:
  void SetUp() override {
    FERRET_ASSERT_OK_AND_ASSIGN(repo_, GitRepoBuilder::Init(temp_, "repo"));
    std::string seven;
    for (int i = 1; i <= 7; ++i) absl::StrAppend(&seven, "needle ", i, "\n");
    ABSL_ASSERT_OK(repo().WriteFile("b.cc", seven));
    ABSL_ASSERT_OK(repo().WriteFile("a.h", "int needle;\n"));
    ABSL_ASSERT_OK(repo().Add());
  }

  void TearDown() override {
    daemon_.Stop(kFerret, temp_.Join("cache/ferret"));
  }

  SubprocessResult Run(std::vector<std::string> args) {
    args.insert(args.begin(), kFerret);
    absl::StatusOr<SubprocessResult> result =
        RunSubprocess(args, {.env_overrides = {absl::StrCat(
                                 "XDG_CACHE_HOME=", temp_.Join("cache"))}});
    EXPECT_TRUE(result.ok()) << result.status();
    return result.ok() ? *std::move(result) : SubprocessResult{.exit_code = -1};
  }

  DaemonTestGuard daemon_;
  ferret_testing::TempDir temp_;
  std::optional<GitRepoBuilder> repo_;

  GitRepoBuilder& repo() {
    CHECK(repo_.has_value());
    return *repo_;
  }
};

TEST_F(FerretCliTest, RipgrepRouterWithoutSymbolStore) {
  for (int i = 0; i < 25; ++i) {
    ABSL_ASSERT_OK(repo().WriteFile(absl::StrCat("calls/file", i, ".cc"),
                                    absl::StrCat("// needle ", i, "\n")));
  }
  ABSL_ASSERT_OK(repo().Add());
  const SubprocessResult result =
      Run({"search", "--backend=ripgrep", "--rg_path=/opt/homebrew/bin/rg",
           absl::StrCat("--root=", repo().root()), "--router", "needle"});
  EXPECT_EQ(result.exit_code, 0) << result.stderr_data;
  EXPECT_THAT(result.stdout_data, StartsWith("Structural route:"));
  EXPECT_THAT(result.stdout_data, HasSubstr("0 definitions"));
}

TEST_F(FerretCliTest, RouterRequiresContextAtStartup) {
  for (const std::string& command : {"search", "mcp"}) {
    const SubprocessResult result =
        Run({command, "--router", "--shape=raw", repo().root()});
    EXPECT_EQ(result.exit_code, 2);
    EXPECT_THAT(result.stderr_data,
                HasSubstr("--router requires --shape=context"));
  }
}

TEST_F(FerretCliTest, UsageErrors) {
  for (const std::vector<std::string>& args :
       std::vector<std::vector<std::string>>{
           {}, {"frobnicate"}, {"search"}, {"search", "a", "b"}}) {
    const SubprocessResult result = Run(args);
    EXPECT_EQ(result.exit_code, 2);
    EXPECT_THAT(result.stderr_data, HasSubstr("Usage:"));
    EXPECT_THAT(result.stdout_data, IsEmpty());
  }
}

TEST_F(FerretCliTest, McpWithoutDaemonReturnsErrorsAndKeepsSession) {
  const std::string input =
      "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\","
      "\"params\":{\"protocolVersion\":\"2025-11-25\","
      "\"capabilities\":{},\"clientInfo\":{\"name\":\"test\","
      "\"version\":\"1\"}}}\n"
      "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\","
      "\"params\":{\"name\":\"search\",\"arguments\":{"
      "\"pattern\":\"needle\"}}}\n"
      "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\","
      "\"params\":{\"name\":\"search\",\"arguments\":{"
      "\"pattern\":\"needle\"}}}\n";
  auto result = RunSubprocess(
      {kFerret, "mcp", repo().root(), "--spawn_daemon=false"},
      {.env_overrides = {absl::StrCat("XDG_CACHE_HOME=", temp_.Join("cache"))},
       .stdin_data = input});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->exit_code, 0) << result->stderr_data;
  EXPECT_THAT(result->stdout_data, HasSubstr("\"id\":3"));
  EXPECT_THAT(result->stdout_data, HasSubstr("\"code\":-32603"));
  auto socket = DaemonSocketPath(repo().root());
  ASSERT_TRUE(socket.ok());
  // F_OK is declared by the platform unistd headers.
  // NOLINTNEXTLINE(misc-include-cleaner)
  EXPECT_NE(access(socket->c_str(), F_OK), 0);
}

TEST_F(FerretCliTest, McpUsesDaemonByDefault) {
  const std::string index = temp_.Join("cache/ferret");
  daemon_.Watch(repo().root());
  const std::string input =
      "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\","
      "\"params\":{\"protocolVersion\":\"2025-11-25\","
      "\"capabilities\":{},\"clientInfo\":{\"name\":\"test\","
      "\"version\":\"1\"}}}\n"
      "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\","
      "\"params\":{\"name\":\"search\",\"arguments\":{"
      "\"pattern\":\"needle\"}}}\n";
  const std::vector<std::string> args = {kFerret, "mcp", repo().root(),
                                         absl::StrCat("--index_dir=", index)};
  auto result = RunSubprocess(
      args,
      {.env_overrides = {absl::StrCat("XDG_CACHE_HOME=", temp_.Join("cache"))},
       .stdin_data = input});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->exit_code, 0) << result->stderr_data;
  EXPECT_THAT(result->stdout_data, HasSubstr("\"matching_files\":2"));
  auto socket = DaemonSocketPath(repo().root());
  ASSERT_TRUE(socket.ok()) << socket.status();
  struct stat st;
  ASSERT_EQ(lstat(socket->c_str(), &st), 0);
  EXPECT_TRUE(S_ISSOCK(st.st_mode));
  const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(fd, 0);
  sockaddr_un address = {};
  address.sun_family = AF_UNIX;
  socket->copy(address.sun_path, socket->size());
  EXPECT_EQ(connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)),
            0);
  EXPECT_TRUE(CheckDaemonPeer(fd).ok());
  close(fd);
}

TEST_F(FerretCliTest, IndexAndSearchWithTheDefaultIndex) {
  const SubprocessResult index = Run({"index", repo().root()});
  ASSERT_EQ(index.exit_code, 0) << index.stderr_data;
  EXPECT_THAT(index.stdout_data, StartsWith("Indexed 2 of 2 files"));
  EXPECT_THAT(index.stdout_data, HasSubstr(temp_.Join("cache/ferret/repo-")));

  const std::string root = absl::StrCat("--root=", repo().root());
  const SubprocessResult found = Run({"search", root, "needle"});
  EXPECT_EQ(found.exit_code, 0) << found.stderr_data;
  EXPECT_THAT(found.stdout_data, StartsWith("8 matching lines in 2 files\n"));

  const SubprocessResult missing = Run({"search", root, "haystack"});
  EXPECT_EQ(missing.exit_code, 1);
  EXPECT_EQ(missing.stdout_data, "0 matching lines in 0 files\n");

  const SubprocessResult bad_shape = Run({"search", root, "--shape=json", "x"});
  EXPECT_EQ(bad_shape.exit_code, 2);
  EXPECT_THAT(bad_shape.stderr_data, HasSubstr("unknown output shape"));

  const SubprocessResult stats = Run({"search", root, "--stats", "needle"});
  EXPECT_THAT(stats.stderr_data, StartsWith("candidates 2 chunks"));
}

TEST_F(FerretCliTest, RawShowsEveryMatchInPathOrder) {
  ASSERT_EQ(Run({"index", repo().root()}).exit_code, 0);
  const std::string root = absl::StrCat("--root=", repo().root());
  const SubprocessResult raw = Run({"search", root, "--shape=raw", "needle"});
  const std::vector<std::string> lines = Lines(raw.stdout_data);
  ASSERT_THAT(lines, SizeIs(8));
  EXPECT_EQ(lines[0], "a.h:1:int needle;");
  EXPECT_EQ(lines[7], "b.cc:7:needle 7");
  // Given explicitly, the cap applies to raw output too.
  const SubprocessResult capped =
      Run({"search", root, "--shape=raw", "--max_lines_per_file=1", "needle"});
  EXPECT_THAT(Lines(capped.stdout_data),
              ElementsAre("a.h:1:int needle;", "b.cc:1:needle 1",
                          "[truncated: 6 more lines in 1 file]"));
  // The line shape keeps SearchRequest's default of five lines per file.
  const SubprocessResult line = Run({"search", root, "--shape=line", "needle"});
  EXPECT_THAT(Lines(line.stdout_data), SizeIs(7));
}

TEST_F(FerretCliTest, MaxLinesCapsRawOutputUnlessRaised) {
  // More matching lines than SearchRequest's default cap of 10,000.
  constexpr int kLines = 10'500;
  std::string many;
  for (int i = 1; i <= kLines; ++i) absl::StrAppend(&many, "needle ", i, "\n");
  ABSL_ASSERT_OK(repo().WriteFile("many.txt", many));
  ABSL_ASSERT_OK(repo().Add());
  ASSERT_EQ(Run({"index", repo().root()}).exit_code, 0);
  const std::string root = absl::StrCat("--root=", repo().root());
  const std::string path = "--path=many.txt";
  // The default is unchanged: the cap applies, and the output says so.
  const SubprocessResult capped = Run({"search", root, path, "--shape=raw",
                                       "--max_bytes=1000000000", "needle"});
  const std::vector<std::string> capped_lines = Lines(capped.stdout_data);
  ASSERT_THAT(capped_lines, SizeIs(10'001));
  EXPECT_THAT(capped_lines.back(), HasSubstr("truncated"));
  // A larger --max_lines removes the cap.
  const SubprocessResult uncapped =
      Run({"search", root, path, "--shape=raw", "--max_bytes=1000000000",
           "--max_lines=9223372036854775807", "needle"});
  const std::vector<std::string> lines = Lines(uncapped.stdout_data);
  ASSERT_THAT(lines, SizeIs(kLines));
  EXPECT_EQ(lines.back(),
            absl::StrCat("many.txt:", kLines, ":needle ", kLines));
  EXPECT_EQ(Run({"search", root, "--max_lines=-1", "needle"}).exit_code, 2);
}

TEST_F(FerretCliTest, Plan) {
  const SubprocessResult plan = Run({"plan", "--regex", "--", "-foo+bar"});
  EXPECT_EQ(plan.exit_code, 0) << plan.stderr_data;
  EXPECT_THAT(plan.stdout_data,
              StartsWith("pattern       \"-foo+bar\" (regex"));
  EXPECT_EQ(Run({"plan", "--regex", "a(b"}).exit_code, 2);
}

}  // namespace
}  // namespace ferret

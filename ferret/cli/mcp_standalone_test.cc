// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstddef>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

#include "ferret/git/testing/git_repo_builder.h"
#include "ferret/rpc/framing.h"
#include "ferret/rpc/json_codec.h"
#include "ferret/testing/temp_dir.h"

extern char** environ;

namespace ferret {
namespace {

TEST(McpStandaloneTest, FindsEditsWithoutExplicitNotify) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok()) << repo.status();
  ASSERT_TRUE(repo->WriteFile("file.cc", "before\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("initial").ok());
  int input[2];
  int output[2];
  ASSERT_EQ(pipe(input), 0);
  ASSERT_EQ(pipe(output), 0);
  posix_spawn_file_actions_t actions;
  ASSERT_EQ(posix_spawn_file_actions_init(&actions), 0);
  ASSERT_EQ(posix_spawn_file_actions_adddup2(&actions, input[0], STDIN_FILENO),
            0);
  ASSERT_EQ(
      posix_spawn_file_actions_adddup2(&actions, output[1], STDOUT_FILENO), 0);
  ASSERT_EQ(posix_spawn_file_actions_addclose(&actions, input[1]), 0);
  ASSERT_EQ(posix_spawn_file_actions_addclose(&actions, output[0]), 0);
  const std::string root = repo->root();
  const std::string dir_flag = absl::StrCat("--index_dir=", temp.Join("index"));
  const char* argv[] = {"ferret/cli/ferret", "mcp",
                        "--standalone",      root.c_str(),
                        dir_flag.c_str(),    nullptr};
  pid_t pid;
  ASSERT_EQ(posix_spawn(&pid, argv[0], &actions, nullptr,
                        const_cast<char* const*>(argv), environ),
            0);
  posix_spawn_file_actions_destroy(&actions);
  close(input[0]);
  close(output[1]);
  RpcFrameWriter writer(input[1]);
  RpcFrameReader reader(output[0]);
  using Json = nlohmann::json;
  auto call = [&](int id, const std::string& method, Json params) -> Json {
    EXPECT_TRUE(writer
                    .Write(SerializeJson({{"jsonrpc", "2.0"},
                                          {"id", id},
                                          {"method", method},
                                          {"params", std::move(params)}}))
                    .ok());
    auto frame = reader.Read();
    EXPECT_TRUE(frame.ok()) << frame.status();
    if (!frame.ok()) return {};
    auto parsed = ParseJson(*frame);
    EXPECT_TRUE(parsed.ok()) << parsed.status();
    return parsed.ok() ? *parsed : Json::object();
  };
  EXPECT_EQ(call(1, "initialize",
                 {{"protocolVersion", "2025-11-25"},
                  {"capabilities", Json::object()},
                  {"clientInfo",
                   {{"name", "test"},
                    {"version", "1"}}}})["result"]["protocolVersion"],
            "2025-11-25");
  Json list = call(2, "tools/list", Json::object());
  ASSERT_EQ(list["result"]["tools"].size(), size_t{3});
  EXPECT_EQ(list["result"]["tools"][0]["name"], "search");
  EXPECT_EQ(list["result"]["tools"][1]["name"], "find_symbol");
  EXPECT_EQ(list["result"]["tools"][2]["name"], "index_status");
  Json status = call(3, "tools/call", {{"name", "index_status"}});
  EXPECT_EQ(status["result"]["structuredContent"]["worktrees"][0]["file_count"],
            1);
  ASSERT_TRUE(repo->WriteFile("file.cc", "after edit needle\n").ok());
  Json edited = call(
      4, "tools/call",
      {{"name", "search"}, {"arguments", {{"pattern", "after edit needle"}}}});
  EXPECT_EQ(edited["result"]["structuredContent"]["matching_files"], 1);
  EXPECT_EQ(edited["result"]["structuredContent"]["stale"], false);
  ASSERT_TRUE(repo->DeleteFile("file.cc").ok());
  Json deleted = call(
      5, "tools/call",
      {{"name", "search"}, {"arguments", {{"pattern", "after edit needle"}}}});
  EXPECT_EQ(deleted["result"]["structuredContent"]["matching_files"], 0);
  EXPECT_EQ(deleted["result"]["structuredContent"]["stale"], false);
  ASSERT_TRUE(repo->WriteFile("a.cc", "duplicate needle\n").ok());
  ASSERT_TRUE(repo->WriteFile("b.cc", "duplicate needle\n").ok());
  Json synced = call(
      6, "tools/call",
      {{"name", "search"}, {"arguments", {{"pattern", "duplicate needle"}}}});
  EXPECT_EQ(synced["result"]["structuredContent"]["matching_files"], 2);
  Json duplicates = call(7, "tools/call", {{"name", "index_status"}});
  const Json& view = duplicates["result"]["structuredContent"]["worktrees"][0];
  EXPECT_EQ(view["file_count"], 2);
  EXPECT_EQ(view["blob_count"], 1);
  close(input[1]);
  close(output[0]);
  int exit_status = 0;
  ASSERT_EQ(waitpid(pid, &exit_status, 0), pid);
  EXPECT_TRUE(WIFEXITED(exit_status));
  EXPECT_EQ(WEXITSTATUS(exit_status), 0);
}

}  // namespace
}  // namespace ferret

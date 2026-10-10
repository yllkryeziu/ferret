// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

#include "ferret/base/env.h"
#include "ferret/base/subprocess.h"
#include "ferret/git/testing/git_repo_builder.h"
#include "ferret/rpc/json_codec.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"

namespace ferret_bench {
namespace {

using ::ferret_testing::GitRepoBuilder;
using ::testing::HasSubstr;

constexpr char kBinary[] = "bench/ferret/ferret_bench";

class FerretBenchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ferret_testing::IsolateGitEnvironment(temp_.Join("home"));
    FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                                GitRepoBuilder::Init(temp_, "repo"));
    ABSL_ASSERT_OK(repo.WriteFile("a.c", "needle\nneedle\n"));
    ABSL_ASSERT_OK(repo.WriteFile("b.c", "other\n"));
    ABSL_ASSERT_OK(repo.Add());
    repo_root_ = repo.root();
  }

  ferret::SubprocessResult Run(std::vector<std::string> args) {
    args.insert(args.begin(), kBinary);
    absl::StatusOr<ferret::SubprocessResult> result =
        ferret::RunSubprocess(args);
    EXPECT_TRUE(result.ok()) << result.status();
    return result.ok() ? *std::move(result)
                       : ferret::SubprocessResult{.exit_code = -1};
  }

  ferret_testing::TempDir temp_;
  std::string repo_root_;
};

TEST_F(FerretBenchTest, FilesBuildAndBothSearchModes) {
  const std::string list = temp_.Join("paths.txt");
  const std::string index = temp_.Join("index.fseg");
  const ferret::SubprocessResult files =
      Run({"files", absl::StrCat("--root=", repo_root_),
           absl::StrCat("--out=", list)});
  ASSERT_EQ(files.exit_code, 0) << files.stderr_data;
  FERRET_ASSERT_OK_AND_ASSIGN(const nlohmann::json file_report,
                              ferret::ParseJson(files.stdout_data));
  EXPECT_EQ(file_report["files_eligible"], 2);
  FERRET_ASSERT_OK_AND_ASSIGN(const std::string listed,
                              ferret::Env::Default()->ReadFile(list));
  EXPECT_EQ(listed, "a.c\nb.c\n");

  const ferret::SubprocessResult build =
      Run({"build", absl::StrCat("--root=", repo_root_),
           absl::StrCat("--files=", list), absl::StrCat("--out=", index),
           "--threads=2"});
  ASSERT_EQ(build.exit_code, 0) << build.stderr_data;
  FERRET_ASSERT_OK_AND_ASSIGN(const nlohmann::json report,
                              ferret::ParseJson(build.stdout_data));
  EXPECT_EQ(report["engine_version"], "fseg-v2");
  EXPECT_EQ(report["files_indexed"], 2);
  EXPECT_EQ(report["sections"]["content_bytes"], 20);
  EXPECT_EQ(report["options"]["threads"], 2);

  ABSL_ASSERT_OK(temp_.WriteFile(
      "queries.jsonl",
      R"({"id":"q","class":"identifier","kind":"literal","pattern":"needle","case_sensitive":true})"
      "\n"));
  const std::string queries = temp_.Join("queries.jsonl");
  for (const std::string& mode : {"topk", "exhaustive"}) {
    const ferret::SubprocessResult search =
        Run({"search", absl::StrCat("--index=", index),
             absl::StrCat("--queries=", queries), absl::StrCat("--mode=", mode),
             "--runs=2", "--warmup=0", "--full_results"});
    ASSERT_EQ(search.exit_code, 0) << search.stderr_data;
    FERRET_ASSERT_OK_AND_ASSIGN(const nlohmann::json record,
                                ferret::ParseJson(search.stdout_data));
    EXPECT_EQ(record["result_count"], 2);
    EXPECT_EQ(record["results"].size(), size_t{2});
    EXPECT_EQ(record["latencies_ns"].size(), size_t{2});
    EXPECT_TRUE(record["digest_stable"]);
    EXPECT_EQ(record.contains("k"), mode == "topk");
    const ferret::SubprocessResult scoped =
        Run({"search", absl::StrCat("--index=", index),
             absl::StrCat("--queries=", queries), absl::StrCat("--mode=", mode),
             "--runs=2", "--warmup=0", "--full_results", "--view"});
    ASSERT_EQ(scoped.exit_code, 0) << scoped.stderr_data;
    FERRET_ASSERT_OK_AND_ASSIGN(const nlohmann::json scoped_record,
                                ferret::ParseJson(scoped.stdout_data));
    EXPECT_EQ(scoped_record["result_digest"], record["result_digest"]);
    EXPECT_EQ(scoped_record["result_count"], record["result_count"]);
    EXPECT_TRUE(scoped_record["digest_stable"]);
    EXPECT_GT(scoped_record["cold_view_names_ns"], 0);
    const ferret::SubprocessResult edited =
        Run({"search", absl::StrCat("--index=", index),
             absl::StrCat("--queries=", queries), absl::StrCat("--mode=", mode),
             "--runs=2", "--warmup=0", "--view_edit"});
    ASSERT_EQ(edited.exit_code, 0) << edited.stderr_data;
    FERRET_ASSERT_OK_AND_ASSIGN(const nlohmann::json edited_record,
                                ferret::ParseJson(edited.stdout_data));
    EXPECT_EQ(edited_record["result_digest"], record["result_digest"]);
    EXPECT_TRUE(edited_record["digest_stable"]);
    EXPECT_GT(edited_record["view_rebuild_ns"], 0);
  }
}

TEST_F(FerretBenchTest, FilesKeepsOnlyThePathsAGlobAdmits) {
  const std::string list = temp_.Join("scoped.txt");
  const ferret::SubprocessResult scoped =
      Run({"files", absl::StrCat("--root=", repo_root_),
           absl::StrCat("--out=", list), "--path_glob=a.*"});
  ASSERT_EQ(scoped.exit_code, 0) << scoped.stderr_data;
  FERRET_ASSERT_OK_AND_ASSIGN(const nlohmann::json report,
                              ferret::ParseJson(scoped.stdout_data));
  EXPECT_EQ(report["files_eligible"], 2);
  EXPECT_EQ(report["files_written"], 1);
  EXPECT_EQ(report["path_glob"], "a.*");
  FERRET_ASSERT_OK_AND_ASSIGN(const std::string listed,
                              ferret::Env::Default()->ReadFile(list));
  EXPECT_EQ(listed, "a.c\n");
  EXPECT_EQ(Run({"files", absl::StrCat("--root=", repo_root_),
                 absl::StrCat("--out=", list), "--path_glob=!"})
                .exit_code,
            1);
}

TEST_F(FerretBenchTest, FilesFromListEqualsFilesFromRootForEveryGlob) {
  FERRET_ASSERT_OK_AND_ASSIGN(GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp_, "nested"));
  for (const char* path : {"a.c", "src/b.c", "src/deep/c.h", "docs/d.rst",
                           "docs/sub/e.rst", "Makefile"}) {
    ABSL_ASSERT_OK(repo.WriteFile(path, "needle\n"));
  }
  ABSL_ASSERT_OK(repo.Add());
  const std::string full = temp_.Join("full.txt");
  ASSERT_EQ(Run({"files", absl::StrCat("--root=", repo.root()),
                 absl::StrCat("--out=", full)})
                .exit_code,
            0);
  int index = 0;
  for (const char* glob : {"*.c", "src/*", "src/**", "!*.rst", "docs/*",
                           "nothing/*", "Makefile"}) {
    const std::string from_root = temp_.Join(absl::StrCat("root", index));
    const std::string from_list = temp_.Join(absl::StrCat("list", index));
    ++index;
    ASSERT_EQ(Run({"files", absl::StrCat("--root=", repo.root()),
                   absl::StrCat("--out=", from_root),
                   absl::StrCat("--path_glob=", glob)})
                  .exit_code,
              0)
        << glob;
    const ferret::SubprocessResult filtered =
        Run({"files", absl::StrCat("--from_list=", full),
             absl::StrCat("--out=", from_list),
             absl::StrCat("--path_glob=", glob)});
    ASSERT_EQ(filtered.exit_code, 0) << glob << filtered.stderr_data;
    FERRET_ASSERT_OK_AND_ASSIGN(const std::string expected,
                                ferret::Env::Default()->ReadFile(from_root));
    FERRET_ASSERT_OK_AND_ASSIGN(const std::string actual,
                                ferret::Env::Default()->ReadFile(from_list));
    EXPECT_EQ(actual, expected) << glob;
  }
  // The listing is not read from the tree: a path that is not there passes.
  ABSL_ASSERT_OK(temp_.WriteFile("ghost.txt", "ghost/x.c\nother.c\n"));
  ASSERT_EQ(Run({"files", absl::StrCat("--from_list=", temp_.Join("ghost.txt")),
                 absl::StrCat("--out=", temp_.Join("ghost.out")),
                 "--path_glob=ghost/*"})
                .exit_code,
            0);
  FERRET_ASSERT_OK_AND_ASSIGN(
      const std::string ghost,
      ferret::Env::Default()->ReadFile(temp_.Join("ghost.out")));
  EXPECT_EQ(ghost, "ghost/x.c\n");
  // The glob is required, --root is exclusive, and a bad glob fails.
  EXPECT_EQ(Run({"files", absl::StrCat("--from_list=", full),
                 absl::StrCat("--out=", temp_.Join("x"))})
                .exit_code,
            2);
  EXPECT_EQ(Run({"files", absl::StrCat("--from_list=", full),
                 absl::StrCat("--root=", repo.root()),
                 absl::StrCat("--out=", temp_.Join("x")), "--path_glob=*.c"})
                .exit_code,
            2);
  EXPECT_EQ(Run({"files", absl::StrCat("--from_list=", full),
                 absl::StrCat("--out=", temp_.Join("x")), "--path_glob=!"})
                .exit_code,
            1);
}

TEST_F(FerretBenchTest, ReportsSkipsAndDistinguishesUsageFromFailures) {
  EXPECT_EQ(Run({}).exit_code, 2);
  EXPECT_EQ(Run({"files"}).exit_code, 2);
  EXPECT_EQ(Run({"files", "--unknown_flag"}).exit_code, 2);
  EXPECT_EQ(Run({"build", "--threads=oops"}).exit_code, 2);
  EXPECT_EQ(Run({"files", "--root"}).exit_code, 2);
  EXPECT_EQ(Run({"search", "--full_results=not_bool"}).exit_code, 2);
  const ferret::SubprocessResult agent =
      Run({"files", absl::StrCat("--root=", repo_root_),
           absl::StrCat("--out=", temp_.Join("agent.txt")), "--policy=agent"});
  ASSERT_EQ(agent.exit_code, 0) << agent.stderr_data;
  FERRET_ASSERT_OK_AND_ASSIGN(const nlohmann::json agent_report,
                              ferret::ParseJson(agent.stdout_data));
  EXPECT_EQ(agent_report["policy"], "agent");
  const std::string list = temp_.Join("paths.txt");
  ABSL_ASSERT_OK(temp_.WriteFile("paths.txt", "a.c\nb.c\nmissing.c\n"));
  const ferret::SubprocessResult build =
      Run({"build", absl::StrCat("--root=", repo_root_),
           absl::StrCat("--files=", list),
           absl::StrCat("--out=", temp_.Join("skip.fseg"))});
  EXPECT_EQ(build.exit_code, 1) << build.stderr_data;
  FERRET_ASSERT_OK_AND_ASSIGN(const nlohmann::json report,
                              ferret::ParseJson(build.stdout_data));
  EXPECT_EQ(report["files_indexed"], 2);
  EXPECT_EQ(report["skipped"]["unreadable"], 1);
  EXPECT_EQ(Run({"build", absl::StrCat("--root=", repo_root_),
                 absl::StrCat("--files=", list),
                 absl::StrCat("--out=", temp_.Join("bad.fseg")), "--threads=0"})
                .exit_code,
            2);
  EXPECT_THAT(build.stderr_data, HasSubstr("skipped or unreadable"));
  ABSL_ASSERT_OK(temp_.WriteFile("bad.txt", "../outside\n"));
  const ferret::SubprocessResult bad_list =
      Run({"build", absl::StrCat("--root=", repo_root_),
           absl::StrCat("--files=", temp_.Join("bad.txt")),
           absl::StrCat("--out=", temp_.Join("unused.fseg"))});
  EXPECT_EQ(bad_list.exit_code, 1);
  EXPECT_TRUE(bad_list.stdout_data.empty());
  ABSL_ASSERT_OK(temp_.WriteFile("bad.jsonl", "{\n"));
  const ferret::SubprocessResult bad_queries =
      Run({"search", absl::StrCat("--index=", temp_.Join("skip.fseg")),
           absl::StrCat("--queries=", temp_.Join("bad.jsonl"))});
  EXPECT_EQ(bad_queries.exit_code, 1);
  EXPECT_TRUE(bad_queries.stdout_data.empty());
}

}  // namespace
}  // namespace ferret_bench

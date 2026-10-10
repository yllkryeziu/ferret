// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/ignore.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/git/testing/git_repo_builder.h"
#include "ferret/git/worktree_files.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/testing/test_seed.h"

namespace ferret {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::ferret_testing::GitRepoBuilder;
using ::ferret_testing::TempDir;

TEST(IgnoreMatcherTest, ParsesGitignoreSyntax) {
  std::map<std::string, std::string> files = {
      {"/tree/.gitignore",
       "# comment\n\n\\#hash\n\\!bang\nspace   \nkept\\ \n"
       "*.log\n!keep.log\n/docs/*.md\na/b\ndir/\n"
       "q?.txt\n[ab].c\n[!a].d\n[^b].e\n[1-3].f\n"
       "**/deep.cc\nroot/**\na/**/tail.cc\ncrlf\r\n"}};
  int reads = 0;
  IgnoreMatcher matcher({.root = "/tree"},
                        [&](absl::string_view path)
                            -> absl::StatusOr<std::optional<std::string>> {
                          ++reads;
                          auto it = files.find(std::string(path));
                          if (it == files.end()) return std::nullopt;
                          return it->second;
                        });
  const std::vector<std::pair<std::string, bool>> cases = {
      {"#hash", true},
      {"!bang", true},
      {"space", true},
      {"space   ", false},
      {"kept ", true},
      {"kept", false},
      {"a.log", true},
      {"sub/a.log", true},
      {"keep.log", false},
      {"docs/a.md", true},
      {"sub/docs/a.md", false},
      {"a/b", true},
      {"sub/a/b", false},
      {"dir", false},
      {"dir/x", true},
      {"qa.txt", true},
      {"qaa.txt", false},
      {"b.c", true},
      {"c.d", true},
      {"a.d", false},
      {"a.e", true},
      {"b.e", false},
      {"2.f", true},
      {"crlf", true},
      {"deep.cc", true},
      {"sub/deep.cc", true},
      {"root/x/y", true},
      {"a/tail.cc", true},
      {"a/x/y/tail.cc", true}};
  for (const auto& [path, expected] : cases) {
    SCOPED_TRACE(path);
    EXPECT_THAT(matcher.IsIgnored(path, false), IsOkAndHolds(expected));
  }
  EXPECT_EQ(reads, 9);  // Root and each distinct traversed directory.
}

TEST(IgnoreMatcherTest, PrecedenceAndExcludedParents) {
  std::map<std::string, std::string> files = {
      {"/tree/.gitignore", "*.tmp\n!ok.tmp\nclosed/\n!closed/a\n"},
      {"/tree/sub/.gitignore", "!x.tmp\n"},
      {"/tree/sub/.ferretignore", "x.tmp\n"},
      {"/tree/closed/.gitignore", "!a\n"},
      {"/info", "*.tmp\n"},
      {"/global", "*.tmp\n"}};
  int reads = 0;
  IgnoreMatcher matcher({.root = "/tree",
                         .info_exclude_path = "/info",
                         .global_excludes_path = "/global",
                         .default_excludes = {"*.tmp"},
                         .ferretignore = true},
                        [&](absl::string_view path)
                            -> absl::StatusOr<std::optional<std::string>> {
                          ++reads;
                          auto it = files.find(std::string(path));
                          if (it == files.end()) return std::nullopt;
                          return it->second;
                        });
  EXPECT_THAT(matcher.IsIgnored("ok.tmp", false), IsOkAndHolds(false));
  EXPECT_THAT(matcher.IsIgnored("sub/x.tmp", false), IsOkAndHolds(true));
  EXPECT_THAT(matcher.IsIgnored("closed/a", false), IsOkAndHolds(true));
  files["/tree/sub/.ferretignore"] = "!x.tmp\n";
  EXPECT_THAT(matcher.IsIgnored("sub/x.tmp", false), IsOkAndHolds(true));
  matcher.Invalidate("sub");
  EXPECT_THAT(matcher.IsIgnored("sub/x.tmp", false), IsOkAndHolds(false));
  EXPECT_GT(reads, 0);
}

TEST(IgnoreMatcherTest, CaseInsensitiveAndSourcePriority) {
  std::map<std::string, std::string> files = {{"/tree/.gitignore", "!*.LOG\n"},
                                              {"/info", "*.log\n"},
                                              {"/global", "*.log\n"}};
  IgnoreMatcher matcher({.root = "/tree",
                         .info_exclude_path = "/info",
                         .global_excludes_path = "/global",
                         .ignore_case = true},
                        [&](absl::string_view path)
                            -> absl::StatusOr<std::optional<std::string>> {
                          auto it = files.find(std::string(path));
                          if (it == files.end()) return std::nullopt;
                          return it->second;
                        });
  EXPECT_THAT(matcher.IsIgnored("sub/FILE.log", false), IsOkAndHolds(false));
}

TEST(IgnoreMatcherTest, SourcesOverrideInDirectoryOrder) {
  std::map<std::string, std::string> files = {
      {"/tree/.gitignore", "!c.tmp\n!d.tmp\n"},
      {"/tree/.ferretignore", "d.tmp\n"},
      {"/tree/sub/.gitignore", "!d.tmp\n"},
      {"/info", "b.tmp\nc.tmp\nd.tmp\n"},
      {"/global", "!a.tmp\n!b.tmp\n!c.tmp\n!d.tmp\n"}};
  IgnoreMatcher matcher(
      {.root = "/tree",
       .info_exclude_path = "/info",
       .global_excludes_path = "/global",
       .default_excludes = {"a.tmp", "b.tmp", "c.tmp", "d.tmp"},
       .ferretignore = true},
      [&](absl::string_view path)
          -> absl::StatusOr<std::optional<std::string>> {
        auto it = files.find(std::string(path));
        if (it == files.end()) return std::nullopt;
        return it->second;
      });
  EXPECT_THAT(matcher.IsIgnored("a.tmp", false), IsOkAndHolds(false));
  EXPECT_THAT(matcher.IsIgnored("b.tmp", false), IsOkAndHolds(true));
  EXPECT_THAT(matcher.IsIgnored("c.tmp", false), IsOkAndHolds(false));
  EXPECT_THAT(matcher.IsIgnored("d.tmp", false), IsOkAndHolds(true));
  EXPECT_THAT(matcher.IsIgnored("sub/d.tmp", false), IsOkAndHolds(false));
  files["/info"] = "!b.tmp\n";
  files["/global"] = "a.tmp\n";
  matcher.Invalidate("");
  EXPECT_THAT(matcher.IsIgnored("a.tmp", false), IsOkAndHolds(true));
  EXPECT_THAT(matcher.IsIgnored("b.tmp", false), IsOkAndHolds(false));
}

TEST(IgnoreMatcherTest, DoesNotFollowSymlinkedPerDirectoryIgnoreFiles) {
  TempDir temp;
  FERRET_ASSERT_OK_AND_ASSIGN(auto repo, GitRepoBuilder::Init(temp, "repo"));
  ABSL_ASSERT_OK(repo.WriteFile("rules", "*.log\n"));
  ABSL_ASSERT_OK(repo.Symlink("rules", ".gitignore"));
  ABSL_ASSERT_OK(repo.WriteFile("a.log", "x"));
  IgnoreMatcher matcher({.root = repo.root()});
  EXPECT_THAT(matcher.IsIgnored("a.log", false), IsOkAndHolds(false));
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto result, repo.RunGit({"check-ignore", "--no-index", "a.log"}));
  EXPECT_EQ(result.exit_code, 1);
  EXPECT_TRUE(result.stdout_data.empty());
}

TEST(IgnoreMatcherTest, ReaderFailureDoesNotCacheIncompleteRules) {
  int calls = 0;
  IgnoreMatcher matcher(
      {.root = "/tree"},
      [&](absl::string_view) -> absl::StatusOr<std::optional<std::string>> {
        if (++calls == 1) return absl::UnavailableError("retry");
        return std::string("*.log\n");
      });
  EXPECT_EQ(matcher.IsIgnored("a.log", false).status().code(),
            absl::StatusCode::kUnavailable);
  EXPECT_THAT(matcher.IsIgnored("a.log", false), IsOkAndHolds(true));
  EXPECT_EQ(calls, 2);
}

// The same small vocabulary drives both paths and patterns so that generated
// rules frequently match, rather than mostly testing unrelated random bytes.
constexpr std::array<absl::string_view, 16> kNames = {
    "a",     "b",  "A",  "ab",    "x.c",    "X.C", ".log", "a.log",
    "A.LOG", "#a", "!a", "space", "space ", "a ",  "c",    "ba"};
constexpr std::array<absl::string_view, 10> kSegments = {
    "a", "b", "A", "ab", "x.c", "*", "?", "[a-c]", "[!a]", "[^b]"};

template <typename T, size_t N>
T Pick(std::mt19937_64& rng, const std::array<T, N>& choices) {
  return choices[rng() % choices.size()];
}

std::string GeneratePattern(std::mt19937_64& rng) {
  // Each ** position is selected independently, including the zero-directory
  // cases of leading **/ and middle /**/.
  const int segments = 1 + static_cast<int>(rng() % 3);
  std::string pattern;
  for (int i = 0; i < segments; ++i) {
    if (i != 0) pattern += "/";
    if (rng() % 5 == 0) {
      pattern += "**";
    } else {
      pattern += Pick(rng, kSegments);
    }
  }
  if (rng() % 5 == 0) pattern.insert(0, "/");
  if (rng() % 4 == 0) pattern.insert(0, "!");
  if (rng() % 6 == 0) pattern += "/";
  return pattern;
}

std::string GenerateRule(std::mt19937_64& rng) {
  switch (rng() % 16) {
    case 0:
      return "";
    case 1:
      return "# a comment";
    case 2:
      return "\\#a";
    case 3:
      return "\\!a";
    case 4:
      return "space  ";
    case 5:
      return "space\\ ";
    case 6:
      return "a\\ ";
    default:
      return GeneratePattern(rng);
  }
}

struct GeneratedTree {
  std::vector<std::string> dirs = {""};
  std::string deepest_dir;
  std::vector<std::string> paths;
  std::map<std::string, std::string> sources;
};

std::string Child(absl::string_view dir, absl::string_view name) {
  return dir.empty() ? std::string(name) : absl::StrCat(dir, "/", name);
}

// Generate a branching tree with a guaranteed four-deep spine. Siblings are
// distinct even on case-insensitive filesystems used by macOS.
GeneratedTree GenerateTree(std::mt19937_64& rng) {
  GeneratedTree tree;
  std::string spine;
  for (int depth = 0; depth < 4; ++depth) {
    const std::string name = std::string(Pick(rng, kNames));
    spine = Child(spine, name);
    tree.dirs.push_back(spine);
    tree.deepest_dir = spine;
    if (rng() % 3 == 0) {
      std::string branch;
      do {
        branch = std::string(Pick(rng, kNames));
      } while (absl::EqualsIgnoreCase(branch, name));
      const size_t slash = spine.rfind('/');
      tree.dirs.push_back(Child(slash == std::string::npos
                                    ? ""
                                    : absl::string_view(spine).substr(0, slash),
                                branch));
    }
  }
  for (const std::string& dir : tree.dirs) {
    std::set<std::string> children;
    for (const std::string& child : tree.dirs) {
      const size_t slash = child.rfind('/');
      if (!child.empty() &&
          (slash == std::string::npos ? "" : child.substr(0, slash)) == dir) {
        children.insert(child.substr(slash + 1));
      }
    }
    std::set<std::string> used;
    for (const std::string& child : children) {
      used.insert(absl::AsciiStrToLower(child));
    }
    for (int i = 0; i < 7; ++i) {
      std::string name(Pick(rng, kNames));
      if (rng() % 4 == 0) name += std::string(Pick(rng, kNames));
      if (used.insert(absl::AsciiStrToLower(name)).second &&
          name != ".gitignore") {
        tree.paths.push_back(Child(dir, name));
      }
    }
  }
  std::sort(tree.paths.begin(), tree.paths.end());
  tree.paths.erase(std::unique(tree.paths.begin(), tree.paths.end()),
                   tree.paths.end());
  return tree;
}

std::string GenerateRules(std::mt19937_64& rng, int count) {
  std::string text;
  for (int i = 0; i < count; ++i) {
    absl::StrAppend(&text, GenerateRule(rng), "\n");
  }
  return text;
}

// check-ignore without --verbose emits only excluded paths; a negated rule
// does not produce a record. --no-index also checks ignore files themselves.
TEST(IgnoreMatcherTest, DifferentialAgainstGit) {
  const uint64_t seed = ferret_testing::TestSeed(0x617acb);
  std::mt19937_64 rng(seed);
  for (bool ignore_case : {false, true}) {
    for (int iteration = 0; iteration < 150; ++iteration) {
      SCOPED_TRACE(absl::StrCat("seed=", seed, " case=", ignore_case,
                                " tree=", iteration));
      TempDir temp;
      FERRET_ASSERT_OK_AND_ASSIGN(auto repo,
                                  GitRepoBuilder::Init(temp, "repo"));
      ABSL_ASSERT_OK(
          repo.SetConfig("core.ignoreCase", ignore_case ? "true" : "false"));
      GeneratedTree tree = GenerateTree(rng);
      for (const std::string& dir : tree.dirs) {
        if (!dir.empty()) tree.paths.push_back(dir);
        if (!dir.empty() && rng() % 4 == 0) continue;
        const std::string path = Child(dir, ".gitignore");
        tree.sources[path] = GenerateRules(rng, 8);
        if (dir.empty()) {
          // Exercise all three ** positions and a negation beneath an
          // excluded directory in every generated tree.
          const std::string& top = tree.dirs[1];
          absl::StrAppend(&tree.sources[path], "**/", Pick(rng, kNames), "\n",
                          top, "/**/", Pick(rng, kNames), "\n", top, "/**\n",
                          top, "/\n");
          const std::string deep_prefix = tree.deepest_dir + "/";
          for (const std::string& file : tree.paths) {
            if (file.starts_with(deep_prefix)) {
              absl::StrAppend(&tree.sources[path], "!", file, "\n");
              break;
            }
          }
        }
        ABSL_ASSERT_OK(repo.WriteFile(path, tree.sources[path]));
        ABSL_ASSERT_OK(
            temp.WriteFile(Child("listing", path), tree.sources[path]));
      }
      tree.sources[".git/info/exclude"] = GenerateRules(rng, 8);
      ABSL_ASSERT_OK(repo.WriteFile(".git/info/exclude",
                                    tree.sources[".git/info/exclude"]));
      std::string input;
      for (const std::string& path : tree.paths) {
        if (std::find(tree.dirs.begin(), tree.dirs.end(), path) ==
            tree.dirs.end()) {
          ABSL_ASSERT_OK(repo.WriteFile(path, "x"));
          ABSL_ASSERT_OK(temp.WriteFile(Child("listing", path), "x"));
        }
        absl::StrAppend(&input, path);
        input.push_back('\0');
      }
      SCOPED_TRACE(::testing::PrintToString(tree.dirs));
      SCOPED_TRACE(absl::StrCat("tree files and directories:\n",
                                absl::StrJoin(tree.paths, "\n")));
      SCOPED_TRACE(::testing::PrintToString(tree.sources));
      FERRET_ASSERT_OK_AND_ASSIGN(
          auto result,
          repo.RunGit({"check-ignore", "--no-index", "-z", "--stdin"}, input));
      ASSERT_TRUE(result.exit_code == 0 || result.exit_code == 1)
          << result.stderr_data;
      std::set<std::string> ignored;
      for (absl::string_view path :
           absl::StrSplit(result.stdout_data, '\0', absl::SkipEmpty())) {
        ignored.emplace(path);
      }
      IgnoreMatcher matcher(
          {.root = repo.root(),
           .info_exclude_path = repo.Path(".git/info/exclude"),
           .ignore_case = ignore_case});
      for (const std::string& path : tree.paths) {
        SCOPED_TRACE(absl::StrCat("path=", path));
        FERRET_ASSERT_OK_AND_ASSIGN(
            bool actual_ignored,
            matcher.IsIgnored(
                path, std::find(tree.dirs.begin(), tree.dirs.end(), path) !=
                          tree.dirs.end()));
        ASSERT_EQ(actual_ignored, ignored.contains(path));
      }
      FERRET_ASSERT_OK_AND_ASSIGN(
          auto git_files,
          repo.Git({"ls-files", "--others", "--exclude-standard", "-z"}));
      std::vector<std::string> expected;
      for (absl::string_view path :
           absl::StrSplit(git_files, '\0', absl::SkipEmpty())) {
        expected.emplace_back(path);
      }
      std::sort(expected.begin(), expected.end());
      FERRET_ASSERT_OK_AND_ASSIGN(
          auto actual_files,
          ListDirectoryFiles(
              temp.Join("listing"),
              {.ignore_case = ignore_case,
               .info_exclude_path = repo.Path(".git/info/exclude")}));
      auto difference = std::mismatch(actual_files.begin(), actual_files.end(),
                                      expected.begin(), expected.end());
      if (difference.first != actual_files.end()) {
        SCOPED_TRACE(absl::StrCat("listing path=", *difference.first));
        EXPECT_EQ(actual_files, expected);
      } else if (difference.second != expected.end()) {
        SCOPED_TRACE(absl::StrCat("missing path=", *difference.second));
        EXPECT_EQ(actual_files, expected);
      }
    }
  }
}

}  // namespace
}  // namespace ferret

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/engine/engine.h"
#include "ferret/engine/ripgrep_backend.h"
#include "ferret/git/blob_hash.h"
#include "ferret/git/testing/git_repo_builder.h"
#include "ferret/search/formatter.h"
#include "ferret/search/search_types.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/workspace/workspace.h"

namespace ferret {
namespace {

std::string RgPath() {
  const char* directory = getenv("TEST_SRCDIR");
  if (directory == nullptr) return "";
#if defined(__APPLE__)
  return absl::StrCat(directory, "/+http_archive+ripgrep_macos/rg");
#else
  return absl::StrCat(directory, "/+http_archive+ripgrep_linux/rg");
#endif
}

SearchRequest Exhaustive(absl::string_view pattern) {
  SearchRequest request;
  request.pattern = std::string(pattern);
  request.max_files = std::numeric_limits<int64_t>::max();
  request.max_lines = std::numeric_limits<int64_t>::max();
  request.max_lines_per_file = std::numeric_limits<int>::max();
  return request;
}

TEST(BackendEquivalenceTest, LiveFilesAndRawOutput) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok()) << repo.status();
  ASSERT_TRUE(
      repo->WriteFile("a.cc", "needle one\r\nother\nneedle again\n").ok());
  ASSERT_TRUE(repo->WriteFile(".hidden.cc", "needle two\n").ok());
  ASSERT_TRUE(repo->WriteFile("nested/ignored.cc", "needle tracked\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("base").ok());
  ASSERT_TRUE(
      repo->WriteFile(".gitignore", "nested/*.cc\nnot-visible.cc\n").ok());
  ASSERT_TRUE(repo->WriteFile("not-visible.cc", "needle ignored\n").ok());
  ASSERT_TRUE(repo->WriteFile("new.cc", "needle untracked\n").ok());
  Engine engine(*Env::Default(), temp.Join("index"));
  RipgrepBackend rg(RgPath());
  ASSERT_TRUE(engine.Register(repo->root()).ok());
  ASSERT_TRUE(rg.Register(repo->root()).ok());
  EXPECT_TRUE(engine.Register(repo->root()).ok());
  EXPECT_TRUE(rg.Register(repo->root()).ok());
  EXPECT_EQ(engine.Roots(), rg.Roots());
  EXPECT_TRUE(rg.Register(repo->root()).ok());
  EXPECT_EQ(rg.Roots(), std::vector<std::string>{repo->root()});
  SearchRequest request = Exhaustive("needle");
  const SearchConsistency sync{.mode = SearchConsistencyMode::kSync,
                               .timeout = absl::Seconds(10)};
  auto indexed = engine.Search(repo->root(), request, sync);
  auto scanned = rg.Search(repo->root(), request, sync);
  ASSERT_TRUE(indexed.ok()) << indexed.status();
  ASSERT_TRUE(scanned.ok()) << scanned.status();
  const FormatOptions options{.shape = OutputShape::kRaw, .max_bytes = 100000};
  EXPECT_EQ(FormatSearchResponse(indexed->response, options),
            FormatSearchResponse(scanned->response, options));
  EXPECT_EQ(indexed->response.num_matching_lines,
            scanned->response.num_matching_lines);
  EXPECT_EQ(indexed->response.num_matching_files,
            scanned->response.num_matching_files);
  ASSERT_EQ(indexed->response.files.size(), scanned->response.files.size());
  for (size_t i = 0; i < indexed->response.files.size(); ++i) {
    const FileResult& a = indexed->response.files[i];
    const FileResult& b = scanned->response.files[i];
    EXPECT_EQ(a.path, b.path);
    EXPECT_EQ(a.oid, b.oid);
    EXPECT_EQ(a.tier, b.tier);
    EXPECT_EQ(a.prior, b.prior);
    EXPECT_EQ(a.name_matches, b.name_matches);
    EXPECT_EQ(a.score, b.score);
    EXPECT_EQ(a.num_matching_lines, b.num_matching_lines);
    EXPECT_EQ(a.lines, b.lines);
  }
  SearchRequest capped = request;
  capped.max_lines_per_file = 1;
  auto capped_index = engine.Search(repo->root(), capped, sync);
  auto capped_rg = rg.Search(repo->root(), capped, sync);
  ASSERT_TRUE(capped_index.ok()) << capped_index.status();
  ASSERT_TRUE(capped_rg.ok()) << capped_rg.status();
  EXPECT_EQ(capped_index->response.num_matching_lines,
            capped_rg->response.num_matching_lines);
  EXPECT_EQ(FormatSearchResponse(capped_index->response, options),
            FormatSearchResponse(capped_rg->response, options));
}

// The generator uses only ASCII literals, character classes, alternation,
// repetition, and anchors. Unicode \w and \b are excluded because RE2 and
// Rust regex disagree on Unicode word boundaries; lookaround and backrefs
// are excluded because neither engine implements both with the same syntax.
TEST(BackendEquivalenceTest, SeededEditsEligibilityAndGlobs) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok()) << repo.status();
  ASSERT_TRUE(repo->WriteFile("edited.cc", "before\n").ok());
  ASSERT_TRUE(repo->WriteFile("deleted.cc", "needle deleted\n").ok());
  ASSERT_TRUE(repo->WriteFile("old.cc", "needle renamed\n").ok());
  ASSERT_TRUE(
      repo->WriteFile("ignore/kept.cc", "needle tracked ignored\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("base").ok());
  ASSERT_TRUE(
      repo->WriteFile(".gitignore", "ignore/*\n!ignore/visible.cc\n").ok());
  ASSERT_TRUE(repo->WriteFile("ignore/visible.cc", "needle visible\n").ok());
  ASSERT_TRUE(
      repo->WriteFile("ignore/missed.cc", "needle untracked ignored\n").ok());
  ASSERT_TRUE(repo->WriteFile("edited.cc", "needle modified\r\n").ok());
  ASSERT_TRUE(repo->DeleteFile("deleted.cc").ok());
  ASSERT_TRUE(repo->Rename("old.cc", "renamed.cc").ok());
  ASSERT_TRUE(
      repo->WriteFile("added.cc", "needle added\n\377needle bytes\n").ok());
  ASSERT_TRUE(repo->WriteFile("long.cc", absl::StrCat(std::string(1100, 'x'),
                                                      "needle tail\n"))
                  .ok());
  ASSERT_TRUE(
      repo->WriteFile("large.cc",
                      absl::StrCat("needle\n", std::string(1 << 20, 'x')))
          .ok());
  ASSERT_TRUE(repo->WriteFile("nul.cc", std::string("needle\0", 7)).ok());
  ASSERT_TRUE(repo->WriteFile("case.cc", "Needle\nNEEDLE\n").ok());
  std::mt19937 bytes(34915);
  std::string trigrams = "needle random\n";
  for (int i = 0; i < 100000; ++i) {
    trigrams.push_back(static_cast<char>(33 + bytes() % 94));
  }
  ASSERT_TRUE(repo->WriteFile("trigrams.cc", trigrams).ok());
  ASSERT_TRUE(repo->WriteFile("unicode.cc", "\303\204\n\303\244\n").ok());
  ASSERT_TRUE(repo->WriteFile("bom.cc", "\357\273\277needle bom\n").ok());
  Engine engine(*Env::Default(), temp.Join("index"));
  RipgrepBackend rg(RgPath());
  ASSERT_TRUE(engine.Register(repo->root()).ok());
  ASSERT_TRUE(rg.Register(repo->root()).ok());
  const SearchConsistency sync{.mode = SearchConsistencyMode::kSync,
                               .timeout = absl::Seconds(10)};
  unsigned seed = 1;
  if (const char* value = getenv("FERRET_TEST_SEED")) {
    seed = static_cast<unsigned>(strtoul(value, nullptr, 10));
  }
  std::mt19937 random(seed);
  const std::vector<std::string> patterns = {"needle",  "Needle",    "nee.*e",
                                             "^needle", "[Nn]eedle", "missing"};
  for (int i = 0; i < 30; ++i) {
    SearchRequest request = Exhaustive(patterns[random() % patterns.size()]);
    request.regex = (random() % 2) != 0;
    request.case_sensitive = (random() % 2) != 0;
    request.context_lines = static_cast<int>(random() % 4);
    if (random() % 3 == 0) request.path_glob = "*.cc";
    SCOPED_TRACE(absl::StrCat(
        "seed=", seed, " pattern=", request.pattern, " regex=", request.regex,
        " case=", request.case_sensitive, " context=", request.context_lines));
    auto indexed = engine.Search(repo->root(), request, sync);
    auto scanned = rg.Search(repo->root(), request, sync);
    ASSERT_TRUE(indexed.ok()) << indexed.status();
    ASSERT_TRUE(scanned.ok()) << scanned.status();
    const FormatOptions raw{.shape = OutputShape::kRaw, .max_bytes = 100000};
    EXPECT_EQ(FormatSearchResponse(indexed->response, raw),
              FormatSearchResponse(scanned->response, raw));
    EXPECT_EQ(indexed->response.num_matching_lines,
              scanned->response.num_matching_lines);
    EXPECT_EQ(indexed->response.num_matching_files,
              scanned->response.num_matching_files);
    if (indexed->response.complete) {
      const FormatOptions ranked{.shape = OutputShape::kContext,
                                 .max_bytes = 100000};
      EXPECT_EQ(FormatSearchResponse(indexed->response, ranked),
                FormatSearchResponse(scanned->response, ranked));
    }
  }
  SearchRequest unicode = Exhaustive("\303\244");
  unicode.case_sensitive = false;
  auto unicode_index = engine.Search(repo->root(), unicode, sync);
  auto unicode_rg = rg.Search(repo->root(), unicode, sync);
  ASSERT_TRUE(unicode_index.ok()) << unicode_index.status();
  ASSERT_TRUE(unicode_rg.ok()) << unicode_rg.status();
  EXPECT_EQ(
      FormatSearchResponse(unicode_index->response,
                           {.shape = OutputShape::kRaw}),
      FormatSearchResponse(unicode_rg->response, {.shape = OutputShape::kRaw}));
  EXPECT_EQ(unicode_index->response.num_matching_lines, 2);
  SearchRequest skipped = Exhaustive("needle");
  auto skipped_rg = rg.Search(repo->root(), skipped, sync);
  ASSERT_TRUE(skipped_rg.ok()) << skipped_rg.status();
  const std::string skipped_raw =
      FormatSearchResponse(skipped_rg->response, {.shape = OutputShape::kRaw});
  EXPECT_EQ(skipped_raw.find("large.cc:"), std::string::npos);
  EXPECT_EQ(skipped_raw.find("nul.cc:"), std::string::npos);
  EXPECT_EQ(skipped_raw.find("trigrams.cc:"), std::string::npos);
}

// Checks both the complete response and the no-oid/no-lexer raw hint.
void CompareNow(Engine& index, RipgrepBackend& backend, absl::string_view root,
                absl::string_view expected) {
  SearchRequest request = Exhaustive("needle");
  request.context_lines = 0;
  const SearchConsistency sync{.mode = SearchConsistencyMode::kSync,
                               .timeout = absl::Seconds(20)};
  auto indexed = index.Search(root, request, sync);
  auto scanned = backend.Search(root, request, sync);
  ASSERT_TRUE(indexed.ok()) << indexed.status();
  ASSERT_TRUE(scanned.ok()) << scanned.status();
  const FormatOptions raw{.shape = OutputShape::kRaw, .max_bytes = 100000};
  const std::string index_text = FormatSearchResponse(indexed->response, raw);
  EXPECT_EQ(index_text, expected);
  EXPECT_EQ(FormatSearchResponse(scanned->response, raw), index_text);
  EXPECT_EQ(scanned->response.num_matching_files,
            indexed->response.num_matching_files);
  EXPECT_EQ(scanned->response.num_matching_lines,
            indexed->response.num_matching_lines);
  ASSERT_EQ(scanned->response.files.size(), indexed->response.files.size());
  for (size_t i = 0; i < indexed->response.files.size(); ++i) {
    const FileResult& a = indexed->response.files[i];
    const FileResult& b = scanned->response.files[i];
    EXPECT_EQ(a.path, b.path);
    EXPECT_EQ(a.oid, b.oid);
    EXPECT_EQ(a.tier, b.tier);
    EXPECT_EQ(a.prior, b.prior);
    EXPECT_EQ(a.name_matches, b.name_matches);
    EXPECT_EQ(a.score, b.score);
    EXPECT_EQ(a.lines, b.lines);
  }
  request.text_only = true;
  auto fast = backend.Search(root, request, sync);
  ASSERT_TRUE(fast.ok()) << fast.status();
  EXPECT_EQ(FormatSearchResponse(fast->response, raw), index_text);
}

TEST(BackendEquivalenceTest, GitAndRipgrepIgnoreSources) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok());
  ASSERT_TRUE(repo->WriteFile("tracked.cc", "needle tracked\n").ok());
  ASSERT_TRUE(repo->WriteFile("nested/tracked.cc", "needle nested\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("base").ok());
  ASSERT_TRUE(
      repo->WriteFile(
              "nested/.gitignore",
              "tracked.cc\nignored/\n!ignored/yes.cc\nallow/*\n!allow/yes.cc\n")
          .ok());
  ASSERT_TRUE(
      repo->WriteFile("nested/ignored/yes.cc", "needle excluded\n").ok());
  ASSERT_TRUE(repo->WriteFile("nested/allow/yes.cc", "needle allowed\n").ok());
  ASSERT_TRUE(repo->WriteFile("nested/allow/no.cc", "needle excluded\n").ok());
  ASSERT_TRUE(repo->WriteFile(".ignore", "tracked.cc\n").ok());
  ASSERT_TRUE(repo->WriteFile(".rgignore", "nested/tracked.cc\n").ok());
  ASSERT_TRUE(temp.WriteFile("global-ignore", "global.cc\n").ok());
  ASSERT_TRUE(
      repo->SetConfig("core.excludesFile", temp.Join("global-ignore")).ok());
  ASSERT_TRUE(repo->WriteFile("global.cc", "needle excluded\n").ok());
  ASSERT_TRUE(repo->WriteFile("untracked.cc", "needle untracked\n").ok());
  Engine engine(*Env::Default(), temp.Join("index"));
  RipgrepBackend backend(RgPath());
  ASSERT_TRUE(engine.Register(repo->root()).ok());
  ASSERT_TRUE(backend.Register(repo->root()).ok());
  CompareNow(engine, backend, repo->root(),
             "nested/allow/yes.cc:1:needle allowed\n"
             "nested/tracked.cc:1:needle nested\n"
             "tracked.cc:1:needle tracked\n"
             "untracked.cc:1:needle untracked\n");
}

TEST(BackendEquivalenceTest, LinkedWorktreeUsesCommonExclude) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok());
  ASSERT_TRUE(repo->WriteFile("tracked.cc", "needle tracked\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("base").ok());
  auto linked = repo->AddWorktree("linked");
  ASSERT_TRUE(linked.ok()) << linked.status();
  ASSERT_TRUE(temp.WriteFile("repo/.git/info/exclude", "hidden.cc\n").ok());
  ASSERT_TRUE(linked->WriteFile("hidden.cc", "needle hidden\n").ok());
  Engine engine(*Env::Default(), temp.Join("index"));
  RipgrepBackend backend(RgPath());
  ASSERT_TRUE(engine.Register(linked->root()).ok());
  ASSERT_TRUE(backend.Register(linked->root()).ok());
  CompareNow(engine, backend, linked->root(), "tracked.cc:1:needle tracked\n");
  ASSERT_TRUE(temp.WriteFile("repo/.git/info/exclude", "other.cc\n").ok());
  CompareNow(engine, backend, linked->root(),
             "hidden.cc:1:needle hidden\ntracked.cc:1:needle tracked\n");
}

// Two tracked paths that differ only in case share one file on a
// case-insensitive volume. Both backends return that file once, under the
// spelling in its directory entry, with the bytes on disk, not the blob of
// the other path.
TEST(BackendEquivalenceTest, CaseCollidingPathsReturnTheDiskFileOnce) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok());
  ASSERT_TRUE(repo->WriteFile("d/xt_a.h", "needle disk\nit line\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("base").ok());
  struct stat lower, upper;
  if (stat(repo->Path("d/xt_a.h").c_str(), &lower) != 0 ||
      stat(repo->Path("d/XT_A.h").c_str(), &upper) != 0 ||
      lower.st_ino != upper.st_ino) {
    GTEST_SKIP() << "temp volume is case-sensitive";
  }
  const std::string other = "/* longer header */\nneedle other blob\n";
  ASSERT_TRUE(
      repo->Git({"update-index", "--add", "--cacheinfo",
                 absl::StrCat("100644,", HashBlob(other).ToHex(), ",d/XT_A.h")})
          .ok());
  Engine engine(*Env::Default(), temp.Join("index"));
  RipgrepBackend backend(RgPath());
  ASSERT_TRUE(engine.Register(repo->root()).ok());
  ASSERT_TRUE(backend.Register(repo->root()).ok());
  CompareNow(engine, backend, repo->root(), "d/xt_a.h:1:needle disk\n");
  EXPECT_EQ(backend.IndexStatus()->front().skipped_paths, 0);
}

TEST(BackendEquivalenceTest, SyncReloadsCoreExcludesFile) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok());
  ASSERT_TRUE(repo->WriteFile("tracked.cc", "needle tracked\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("base").ok());
  ASSERT_TRUE(
      repo->Git({"config", "core.excludesFile", temp.Join("global-ignore")})
          .ok());
  ASSERT_TRUE(temp.WriteFile("global-ignore", "hidden.cc\n").ok());
  ASSERT_TRUE(repo->WriteFile("hidden.cc", "needle hidden\n").ok());
  Engine engine(*Env::Default(), temp.Join("index"));
  RipgrepBackend backend(RgPath());
  ASSERT_TRUE(engine.Register(repo->root()).ok());
  ASSERT_TRUE(backend.Register(repo->root()).ok());
  CompareNow(engine, backend, repo->root(), "tracked.cc:1:needle tracked\n");
  ASSERT_TRUE(temp.WriteFile("global-ignore", "other.cc\n").ok());
  CompareNow(engine, backend, repo->root(),
             "hidden.cc:1:needle hidden\ntracked.cc:1:needle tracked\n");
}

TEST(BackendEquivalenceTest, NonRegularAndNestedRepositoriesAreSkipped) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok());
  ASSERT_TRUE(repo->WriteFile("tracked.cc", "needle tracked\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("base").ok());
  ASSERT_TRUE(repo->Symlink("tracked.cc", "link.cc").ok());
  ASSERT_TRUE(repo->Symlink(".", "linkdir").ok());
  auto head = repo->Git({"rev-parse", "HEAD"});
  ASSERT_TRUE(head.ok());
  head->pop_back();
  ASSERT_TRUE(repo->AddGitlink("submodule", *head).ok());
  ASSERT_TRUE(repo->WriteFile("submodule/file.cc", "needle submodule\n").ok());
  auto nested = ferret_testing::GitRepoBuilder::Init(temp, "repo/nested");
  ASSERT_TRUE(nested.ok()) << nested.status();
  ASSERT_TRUE(nested->WriteFile("inside.cc", "needle nested\n").ok());
  Engine engine(*Env::Default(), temp.Join("index"));
  RipgrepBackend backend(RgPath());
  ASSERT_TRUE(engine.Register(repo->root()).ok());
  ASSERT_TRUE(backend.Register(repo->root()).ok());
  CompareNow(engine, backend, repo->root(), "tracked.cc:1:needle tracked\n");
}

TEST(BackendEquivalenceTest, CachedPathsAndEligibilityInvalidateImmediately) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok());
  ASSERT_TRUE(repo->WriteFile("base.cc", "needle base\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("base").ok());
  Engine engine(*Env::Default(), temp.Join("index"));
  RipgrepBackend backend(RgPath());
  ASSERT_TRUE(engine.Register(repo->root()).ok());
  ASSERT_TRUE(backend.Register(repo->root()).ok());
  CompareNow(engine, backend, repo->root(), "base.cc:1:needle base\n");
  const std::string path = repo->Path("base.cc");
  struct stat before = {};
  ASSERT_EQ(stat(path.c_str(), &before), 0);
  ASSERT_TRUE(repo->WriteFile("base.cc", "needle edit\n").ok());
  timespec times[2];
#if defined(__APPLE__)
  times[0] = before.st_atimespec;
  times[1] = before.st_mtimespec;
#else
  times[0] = before.st_atim;
  times[1] = before.st_mtim;
#endif
  // AT_FDCWD is declared in the platform fcntl headers.
  // NOLINTNEXTLINE(misc-include-cleaner)
  ASSERT_EQ(utimensat(AT_FDCWD, path.c_str(), times, 0), 0);
  CompareNow(engine, backend, repo->root(), "base.cc:1:needle edit\n");
  ASSERT_TRUE(
      repo->WriteFile("base.cc", absl::string_view("needle\0skip\n", 12)).ok());
  CompareNow(engine, backend, repo->root(), "");
  ASSERT_TRUE(repo->WriteFile("new.cc", "needle new\n").ok());
  CompareNow(engine, backend, repo->root(), "new.cc:1:needle new\n");
  ASSERT_TRUE(repo->Add({"new.cc"}).ok());
  CompareNow(engine, backend, repo->root(), "new.cc:1:needle new\n");
  ASSERT_TRUE(repo->WriteFile(".gitignore", "new.cc\n").ok());
  CompareNow(engine, backend, repo->root(), "new.cc:1:needle new\n");
  ASSERT_TRUE(repo->WriteFile(".git/info/exclude", "untracked.cc\n").ok());
  ASSERT_TRUE(repo->WriteFile("untracked.cc", "needle hidden\n").ok());
  CompareNow(engine, backend, repo->root(), "new.cc:1:needle new\n");
  ASSERT_TRUE(repo->DeleteFile("new.cc").ok());
  CompareNow(engine, backend, repo->root(), "");
}

TEST(BackendEquivalenceTest, PlainDirectoryHasNoGitIndex) {
  ferret_testing::TempDir temp;
  ASSERT_TRUE(temp.MakeDirs("plain").ok());
  ASSERT_TRUE(temp.WriteFile("plain/.hidden.cc", "needle\n").ok());
  RipgrepBackend rg(RgPath());
  ASSERT_TRUE(rg.Register(temp.Join("plain")).ok());
  auto result = rg.Search(temp.Join("plain"), Exhaustive("needle"), {});
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->response.files.size(), size_t{1});
  EXPECT_EQ(result->response.files.front().path, ".hidden.cc");
}

TEST(BackendEquivalenceTest, MillionPlainMatchesRetainOnlyKeptLines) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok());
  ASSERT_TRUE(repo->WriteFile("a.cc", "needle\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("base").ok());
  const std::string script =
      "#!/usr/bin/env python3\n"
      "import sys\n"
      "out = sys.stdout.buffer\n"
      "out.write((b'./a.cc\\0' + b'1:0:needle\\n') * 1000000)\n";
  ASSERT_TRUE(temp.WriteFile("fake", script).ok());
  ASSERT_EQ(chmod(temp.Join("fake").c_str(), 0700), 0);
  RipgrepBackend rg(temp.Join("fake"));
  ASSERT_TRUE(rg.Register(repo->root()).ok());
  SearchRequest request = Exhaustive("needle");
  request.max_lines_per_file = 1;
  request.max_lines = 1;
  auto found = rg.Search(repo->root(), request, {});
  ASSERT_TRUE(found.ok()) << found.status();
  EXPECT_EQ(found->response.num_matching_lines, 1000000);
  ASSERT_EQ(found->response.files.size(), size_t{1});
  EXPECT_EQ(found->response.files.front().lines.size(), size_t{1});
}

TEST(BackendEquivalenceTest, ConcurrentSearchesShareFileAndPathCaches) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok());
  ASSERT_TRUE(repo->WriteFile("nested/a.cc", "needle one\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("base").ok());
  RipgrepBackend backend(RgPath());
  ASSERT_TRUE(backend.Register(repo->root()).ok());
  SearchRequest request = Exhaustive("needle");
  request.context_lines = 0;
  request.text_only = true;
  std::vector<std::thread> threads;
  std::vector<absl::StatusOr<WorkspaceSearchResult>> results(8);
  threads.reserve(results.size());
  for (auto& result : results) {
    threads.emplace_back([&backend, &repo, &request, &result] {
      result = backend.Search(repo->root(), request, {});
    });
  }
  for (auto& thread : threads) thread.join();
  for (const auto& result : results) {
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(
        FormatSearchResponse(result->response, {.shape = OutputShape::kRaw}),
        "nested/a.cc:1:needle one\n");
  }
}

TEST(BackendEquivalenceTest, RawUsesBytesSearchedByRgNotALaterRead) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok());
  ASSERT_TRUE(repo->WriteFile("a.cc", "needle new\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("base").ok());
  ASSERT_TRUE(
      temp.WriteFile(
              "fake",
              "#!/bin/sh\nprintf '%b%b' './a.cc\\000' '1:0:needle old\\n'\n")
          .ok());
  ASSERT_EQ(chmod(temp.Join("fake").c_str(), 0700), 0);
  RipgrepBackend backend(temp.Join("fake"));
  ASSERT_TRUE(backend.Register(repo->root()).ok());
  SearchRequest request = Exhaustive("needle");
  request.text_only = true;
  request.context_lines = 0;
  auto result = backend.Search(repo->root(), request, {});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(
      FormatSearchResponse(result->response, {.shape = OutputShape::kRaw}),
      "a.cc:1:needle old\n");
}

// Writes an executable that prints `output` (printf %b escapes apply) in
// place of rg.
std::string FakeRg(ferret_testing::TempDir& temp, absl::string_view output) {
  EXPECT_TRUE(temp.WriteFile("fake", absl::StrCat("#!/bin/sh\nprintf '%b' '",
                                                  output, "'\n"))
                  .ok());
  EXPECT_EQ(chmod(temp.Join("fake").c_str(), 0700), 0);
  return temp.Join("fake");
}

// One file whose bytes disagree with rg's line leaves out only that file,
// even after some of its lines agreed and were kept.
TEST(BackendEquivalenceTest, FileDisagreeingWithRgIsDroppedNotFatal) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok());
  ASSERT_TRUE(repo->WriteFile("a.cc", "needle x\nneedle y\n").ok());
  ASSERT_TRUE(repo->WriteFile("b.cc", "needle b\nneedle c\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("base").ok());
  RipgrepBackend backend(FakeRg(temp,
                                "./a.cc\\00001:0:needle x\\n"
                                "./a.cc\\00002:9:needle CHANGED\\n"
                                "./b.cc\\00001:0:needle b\\n"
                                "./b.cc\\00002:9:needle c\\n"));
  ASSERT_TRUE(backend.Register(repo->root()).ok());
  SearchRequest request = Exhaustive("needle");
  request.context_lines = 0;
  // The budget of two lines is spent on a.cc's first line until the file is
  // dropped; b.cc needs both.
  request.max_lines = 2;
  auto result = backend.Search(repo->root(), request, {});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(
      FormatSearchResponse(result->response, {.shape = OutputShape::kRaw}),
      "b.cc:1:needle b\nb.cc:2:needle c\n");
  EXPECT_EQ(result->response.num_matching_files, 1);
  EXPECT_EQ(result->response.num_matching_lines, 2);
  EXPECT_EQ(result->response.stats.inconsistent_files, 1);
  EXPECT_EQ(backend.IndexStatus()->front().skipped_paths, 1);
}

// A stream that is not rg's grammar is a lost search, not a bad file.
TEST(BackendEquivalenceTest, MalformedRgStreamIsDataLoss) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok());
  ASSERT_TRUE(repo->WriteFile("a.cc", "needle a\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("base").ok());
  SearchRequest request = Exhaustive("needle");
  request.context_lines = 0;
  for (const char* output :
       {"./a.cc\\00001:0:needle a\\n./a.cc\\00002:0:needle",
        "./a.cc\\0000no header here\\n"}) {
    RipgrepBackend backend(FakeRg(temp, output));
    ASSERT_TRUE(backend.Register(repo->root()).ok());
    auto result = backend.Search(repo->root(), request, {});
    ASSERT_FALSE(result.ok()) << output;
    EXPECT_EQ(result.status().code(), absl::StatusCode::kDataLoss) << output;
  }
}

TEST(BackendEquivalenceTest, RgRegexFailureIsInvalidArgument) {
  ferret_testing::TempDir temp;
  ferret_testing::IsolateGitEnvironment(temp.Join("home"));
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok());
  ASSERT_TRUE(
      temp.WriteFile("error",
                     "#!/bin/sh\necho 'regex parse error' >&2\nexit 2\n")
          .ok());
  ASSERT_EQ(chmod(temp.Join("error").c_str(), 0700), 0);
  RipgrepBackend rg(temp.Join("error"));
  ASSERT_TRUE(rg.Register(repo->root()).ok());
  const auto result = rg.Search(repo->root(), Exhaustive("needle"), {});
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(BackendEquivalenceTest, CrashingRgReturnsStatus) {
  ferret_testing::TempDir temp;
  ASSERT_TRUE(temp.MakeDirs("repo").ok());
  ASSERT_TRUE(temp.WriteFile("crash", "#!/bin/sh\nkill -9 $$\n").ok());
  ASSERT_EQ(chmod(temp.Join("crash").c_str(), 0700), 0);
  RipgrepBackend rg(temp.Join("crash"));
  ASSERT_TRUE(rg.Register(temp.Join("repo")).ok());
  const auto result = rg.Search(temp.Join("repo"), Exhaustive("needle"), {});
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInternal);
}

TEST(BackendEquivalenceTest, MissingBinaryAndInvalidPattern) {
  ferret_testing::TempDir temp;
  ASSERT_TRUE(temp.MakeDirs("repo").ok());
  RipgrepBackend backend(temp.Join("absent"));
  ASSERT_TRUE(backend.Register(temp.Join("repo")).ok());
  SearchRequest request = Exhaustive("needle");
  EXPECT_FALSE(backend.Search(temp.Join("repo"), request, {}).ok());
  request.pattern = "[";
  request.regex = true;
  EXPECT_EQ(backend.Search(temp.Join("repo"), request, {}).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(backend.IndexStatus()->size(), size_t{1});
  EXPECT_EQ(backend.IndexStatus()->front().file_count, 0);
}

}  // namespace
}  // namespace ferret

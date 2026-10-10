// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/ls_files.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/git/oid.h"
#include "ferret/testing/status_matchers.h"

namespace ferret {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

constexpr absl::string_view kOidA = "ce013625030ba8dba906f756967f9e9ca394464a";
constexpr absl::string_view kOidB = "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391";

std::string Record(absl::string_view tag, absl::string_view mode,
                   absl::string_view oid, absl::string_view stage,
                   absl::string_view path) {
  return absl::StrCat(tag, " ", mode, " ", oid, " ", stage, "\t", path,
                      absl::string_view("\0", 1));
}

TEST(LsFilesTest, EmptyOutputHasNoEntries) {
  EXPECT_THAT(ParseLsFiles(""), absl_testing::IsOkAndHolds(IsEmpty()));
}

TEST(LsFilesTest, ParsesEveryTag) {
  const std::string output =
      Record("H", "100644", kOidA, "0", "a.txt") +
      Record("h", "100755", kOidB, "0", "bin/run") +
      Record("S", "100644", kOidA, "0", "sparse/x") +
      Record("M", "100644", kOidA, "1", "conflict") +
      Record("m", "100644", kOidB, "3", "conflict") +
      Record("H", "160000", kOidA, "0", "third_party/sub") +
      Record("H", "120000", kOidB, "0", "link");
  FERRET_ASSERT_OK_AND_ASSIGN(std::vector<LsFilesEntry> entries,
                              ParseLsFiles(output));
  const Oid a = *Oid::FromHex(kOidA);
  const Oid b = *Oid::FromHex(kOidB);
  ASSERT_EQ(entries.size(), 7u);
  EXPECT_EQ(entries[0].mode, 0100644u);
  EXPECT_EQ(entries[0].oid, a);
  EXPECT_EQ(entries[0].stage, 0);
  EXPECT_FALSE(entries[0].skip_worktree);
  EXPECT_FALSE(entries[0].assume_unchanged);
  EXPECT_EQ(entries[0].path, "a.txt");
  EXPECT_EQ(entries[1].mode, 0100755u);
  EXPECT_EQ(entries[1].oid, b);
  EXPECT_TRUE(entries[1].assume_unchanged);
  EXPECT_TRUE(entries[2].skip_worktree);
  EXPECT_FALSE(entries[2].assume_unchanged);
  EXPECT_EQ(entries[3].stage, 1);
  EXPECT_EQ(entries[4].stage, 3);
  EXPECT_TRUE(entries[4].assume_unchanged);
  EXPECT_EQ(entries[5].mode, 0160000u);
  EXPECT_EQ(entries[6].mode, 0120000u);
}

TEST(LsFilesTest, PathsAreVerbatim) {
  const std::string output =
      Record("H", "100644", kOidA, "0", "with space/tab\there") +
      Record("H", "100644", kOidA, "0", "new\nline") +
      Record("H", "100644", kOidA, "0", "caf\xC3\xA9/\xE2\x84\xAA.txt") +
      Record("H", "100644", kOidA, "0", "\"quoted\"\\back");
  EXPECT_THAT(ParseLsFiles(output),
              absl_testing::IsOkAndHolds(ElementsAre(
                  Field(&LsFilesEntry::path, "with space/tab\there"),
                  Field(&LsFilesEntry::path, "new\nline"),
                  Field(&LsFilesEntry::path, "caf\xC3\xA9/\xE2\x84\xAA.txt"),
                  Field(&LsFilesEntry::path, "\"quoted\"\\back"))));
}

TEST(LsFilesTest, RejectsMissingTerminator) {
  std::string output = Record("H", "100644", kOidA, "0", "a");
  output.pop_back();
  EXPECT_THAT(ParseLsFiles(output), StatusIs(absl::StatusCode::kInvalidArgument,
                                             HasSubstr("not NUL-terminated")));
}

TEST(LsFilesTest, RejectsMalformedRecords) {
  const struct {
    std::string record;
    absl::string_view problem;
  } k_cases[] = {
      {Record("X", "100644", kOidA, "0", "a"), "unknown tag"},
      {Record("H", "100644", kOidA, "2", "a"), "bad stage"},
      {Record("S", "100644", kOidA, "1", "a"), "bad stage"},
      {Record("M", "100644", kOidA, "0", "a"), "bad stage"},
      {Record("H", "100644", kOidA, "4", "a"), "bad stage"},
      {Record("H", "100694", kOidA, "0", "a"), "bad mode"},
      {Record("H", "100644", std::string(40, 'g'), "0", "a"), "bad object id"},
      // A SHA-256 object id shifts every later field.
      {Record("H", "100644", std::string(64, 'a'), "0", "a"),
       "bad field layout"},
      {Record("H", "10644", kOidA, "0", "a"), "bad field layout"},
      {std::string("H 100644\0", 9), "bad field layout"},
      {Record("H", "100644", kOidA, "0", ""), "bad path"},
      {Record("H", "100644", kOidA, "0", "../escape"), "bad path"},
      {Record("H", "100644", kOidA, "0", "/abs"), "bad path"},
      {Record("H", "100644", kOidA, "0", ".git/config"), "bad path"},
      {Record("H", "100644", kOidA, "0", "dir/"), "bad path"},
  };
  for (const auto& c : k_cases) {
    SCOPED_TRACE(c.record);
    const std::string output =
        Record("H", "100644", kOidA, "0", "fine") + c.record;
    EXPECT_THAT(ParseLsFiles(output),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         AllOf(HasSubstr("record 2"), HasSubstr(c.problem))));
  }
}

TEST(LsFilesTest, MutatedOutputNeverCrashes) {
  const std::string valid = Record("H", "100644", kOidA, "0", "src/a b.cc") +
                            Record("s", "100755", kOidB, "0", "x") +
                            Record("M", "100644", kOidA, "2", "y/z");
  constexpr uint64_t kSeed = 7;
  SCOPED_TRACE(testing::Message() << "seed " << kSeed);
  std::mt19937_64 rng(kSeed);
  constexpr absl::string_view kAlphabet("\0\t /.Hhm0123456789abcdefg", 25);
  int parsed = 0;
  for (int i = 0; i < 20000; ++i) {
    std::string input = valid;
    const int edits = 1 + static_cast<int>(rng() % 4);
    for (int e = 0; e < edits; ++e) {
      const size_t pos = static_cast<size_t>(rng() % (input.size() + 1));
      const char c = kAlphabet[rng() % kAlphabet.size()];
      switch (rng() % 3) {
        case 0:
          if (pos < input.size()) input[pos] = c;
          break;
        case 1:
          input.insert(pos, 1, c);
          break;
        default:
          input.resize(pos);
          break;
      }
    }
    const absl::StatusOr<std::vector<LsFilesEntry>> result =
        ParseLsFiles(input);
    if (result.ok()) {
      ++parsed;
    } else {
      ASSERT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument)
          << result.status();
    }
  }
  // Both outcomes must be common, or the mutations explore nothing.
  EXPECT_GT(parsed, 1000);
  EXPECT_LT(parsed, 19000);
}

}  // namespace
}  // namespace ferret

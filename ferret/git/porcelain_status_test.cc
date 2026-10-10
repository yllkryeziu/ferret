// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/porcelain_status.h"

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

#include "ferret/testing/status_matchers.h"

namespace ferret {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

constexpr absl::string_view kOid = "61780798228d17af2d34fce4cfbdf35556832472";
constexpr absl::string_view kNull = "0000000000000000000000000000000000000000";

std::string Nul(absl::string_view record) {
  return absl::StrCat(record, absl::string_view("\0", 1));
}

std::string Changed(absl::string_view xy, absl::string_view worktree_mode,
                    absl::string_view path) {
  return Nul(absl::StrCat("1 ", xy, " N... 100644 100644 ", worktree_mode, " ",
                          kOid, " ", kOid, " ", path));
}

TEST(PorcelainStatusTest, EmptyOutputHasNoEntries) {
  EXPECT_THAT(ParsePorcelainStatus(""), IsOkAndHolds(IsEmpty()));
}

TEST(PorcelainStatusTest, ParsesChangedRecords) {
  const std::string output = Changed(".M", "100644", "b.txt") +
                             Changed(".T", "120000", "link") +
                             Changed("M.", "100755", "staged only") +
                             Changed(".D", "000000", "sub/gone.txt") +
                             Nul(absl::StrCat("1 .A N... 000000 000000 100644 ",
                                              kNull, " ", kNull, " ita.txt")) +
                             Nul(absl::StrCat("1 AD S... 000000 160000 000000 ",
                                              kNull, " ", kOid, " gl"));
  FERRET_ASSERT_OK_AND_ASSIGN(std::vector<PorcelainStatusEntry> entries,
                              ParsePorcelainStatus(output));
  ASSERT_EQ(entries.size(), 6u);
  EXPECT_EQ(entries[0].kind, PorcelainStatusKind::kChanged);
  EXPECT_EQ(entries[0].index_status, '.');
  EXPECT_EQ(entries[0].worktree_status, 'M');
  EXPECT_EQ(entries[0].worktree_mode, 0100644u);
  EXPECT_EQ(entries[0].path, "b.txt");
  EXPECT_EQ(entries[1].worktree_status, 'T');
  EXPECT_EQ(entries[1].worktree_mode, 0120000u);
  EXPECT_EQ(entries[2].index_status, 'M');
  EXPECT_EQ(entries[2].worktree_mode, 0100755u);
  EXPECT_EQ(entries[2].path, "staged only");
  EXPECT_EQ(entries[3].worktree_status, 'D');
  EXPECT_EQ(entries[3].worktree_mode, 0u);
  EXPECT_EQ(entries[4].worktree_status, 'A');
  EXPECT_EQ(entries[5].index_status, 'A');
  EXPECT_EQ(entries[5].worktree_status, 'D');
}

TEST(PorcelainStatusTest, ParsesUnmergedRecords) {
  const std::string output =
      Nul(absl::StrCat("u UU N... 100644 100644 100644 100755 ", kOid, " ",
                       kOid, " ", kOid, " conflict with spaces.txt"));
  FERRET_ASSERT_OK_AND_ASSIGN(std::vector<PorcelainStatusEntry> entries,
                              ParsePorcelainStatus(output));
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0].kind, PorcelainStatusKind::kUnmerged);
  EXPECT_EQ(entries[0].index_status, 'U');
  EXPECT_EQ(entries[0].worktree_status, 'U');
  EXPECT_EQ(entries[0].worktree_mode, 0100755u);
  EXPECT_EQ(entries[0].path, "conflict with spaces.txt");
}

TEST(PorcelainStatusTest, ParsesUntrackedAndIgnoredRecords) {
  const std::string output = Nul("? new\nline.txt") + Nul("? nested/") +
                             Nul("! build/out.o") + Nul("! out/");
  EXPECT_THAT(ParsePorcelainStatus(output),
              IsOkAndHolds(ElementsAre(
                  AllOf(Field(&PorcelainStatusEntry::kind,
                              PorcelainStatusKind::kUntracked),
                        Field(&PorcelainStatusEntry::path, "new\nline.txt"),
                        Field(&PorcelainStatusEntry::worktree_mode, 0u)),
                  AllOf(Field(&PorcelainStatusEntry::kind,
                              PorcelainStatusKind::kUntracked),
                        Field(&PorcelainStatusEntry::path, "nested/")),
                  AllOf(Field(&PorcelainStatusEntry::kind,
                              PorcelainStatusKind::kIgnored),
                        Field(&PorcelainStatusEntry::path, "build/out.o")),
                  AllOf(Field(&PorcelainStatusEntry::kind,
                              PorcelainStatusKind::kIgnored),
                        Field(&PorcelainStatusEntry::path, "out/")))));
}

TEST(PorcelainStatusTest, SkipsHeaders) {
  const std::string output =
      Nul("# branch.oid (initial)") + Nul("# branch.head main") + Nul("? a");
  EXPECT_THAT(
      ParsePorcelainStatus(output),
      IsOkAndHolds(ElementsAre(Field(&PorcelainStatusEntry::path, "a"))));
}

TEST(PorcelainStatusTest, RejectsMissingTerminator) {
  EXPECT_THAT(ParsePorcelainStatus("? a"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("not NUL-terminated")));
}

TEST(PorcelainStatusTest, RejectsMalformedRecords) {
  const struct {
    std::string record;
    absl::string_view problem;
  } k_cases[] = {
      {Nul(absl::StrCat("2 R. N... 100644 100644 100644 ", kOid, " ", kOid,
                        " R100 new")) +
           Nul("old"),
       "rename record"},
      {Nul("x a"), "bad record type"},
      {Nul("?"), "bad record type"},
      {Nul("?a"), "bad record type"},
      {Changed(".X", "100644", "a"), "bad status code"},
      {Changed(".", "100644", "a"), "bad status code"},
      {Nul(absl::StrCat("1 .M Q... 100644 100644 100644 ", kOid, " ", kOid,
                        " a")),
       "bad status code"},
      {Changed(".M", "10064x", "a"), "bad mode"},
      {Nul(absl::StrCat("1 .M N... 100644 100644 100644 ", kOid, " xyz a")),
       "bad object id"},
      {Nul("1 .M N... 100644"), "too few fields"},
      {Changed(".M", "100644", "../a"), "bad path"},
      {Changed(".M", "100644", "dir/"), "bad path"},
      {Changed(".M", "100644", ""), "bad path"},
      {Nul("? "), "bad path"},
      {Nul("? /abs"), "bad path"},
      {Nul("? .git/"), "bad path"},
      {Nul("! a//b"), "bad path"},
  };
  for (const auto& c : k_cases) {
    SCOPED_TRACE(c.record);
    EXPECT_THAT(ParsePorcelainStatus(Nul("? fine") + c.record),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         AllOf(HasSubstr("record 2"), HasSubstr(c.problem))));
  }
}

TEST(PorcelainStatusTest, MutatedOutputNeverCrashes) {
  const std::string valid =
      Changed(".M", "100644", "src/a b.cc") +
      Nul(absl::StrCat("u UU N... 100644 100644 100644 100644 ", kOid, " ",
                       kOid, " ", kOid, " c")) +
      Nul("? new/") + Nul("! x.o");
  constexpr uint64_t kSeed = 11;
  SCOPED_TRACE(testing::Message() << "seed " << kSeed);
  std::mt19937_64 rng(kSeed);
  constexpr absl::string_view kAlphabet("\0 /.1u?!#MTADU0123456789abcdefN", 31);
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
    const absl::StatusOr<std::vector<PorcelainStatusEntry>> result =
        ParsePorcelainStatus(input);
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

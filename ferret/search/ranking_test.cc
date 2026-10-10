// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/search/ranking.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/index/rank_key.h"
#include "ferret/search/search_types.h"
#include "ferret/text/lexer.h"

namespace ferret {
namespace {

using ::testing::ElementsAre;

FileResult File(std::string path, RankTier tier, uint16_t prior,
                int64_t matching_lines = 1,
                MatchClass match_class = MatchClass::kPartialToken,
                bool name_matches = false) {
  FileResult file;
  file.path = std::move(path);
  file.tier = tier;
  file.prior = prior;
  file.num_matching_lines = matching_lines;
  file.name_matches = name_matches;
  file.lines.push_back(ResultLine{
      .number = 1, .text = "x", .is_match = true, .match_class = match_class});
  return file;
}

std::vector<std::string> Paths(const std::vector<FileResult>& files) {
  std::vector<std::string> paths;
  paths.reserve(files.size());
  for (const FileResult& file : files) paths.push_back(file.path);
  return paths;
}

TEST(ScoreFileTest, Boosts) {
  EXPECT_EQ(ScoreFile(File("a", RankTier::kSource, 100)), 100);
  EXPECT_EQ(ScoreFile(File("a", RankTier::kSource, 100, 1,
                           MatchClass::kCodeIdentifier)),
            100 + kIdentifierBoost);
  EXPECT_EQ(ScoreFile(File("a", RankTier::kSource, 100, 1, MatchClass::kComment,
                           /*name_matches=*/true)),
            100 + kNameBoost);
  EXPECT_EQ(ScoreFile(File("a", RankTier::kSource, 0, 2)),
            kMatchCountBoostStep);
  EXPECT_EQ(ScoreFile(File("a", RankTier::kSource, 0, 7)),
            2 * kMatchCountBoostStep);
  EXPECT_EQ(ScoreFile(File("a", RankTier::kSource, 0, 1'000'000)),
            kMaxMatchCountBoost);
}

TEST(ScoreFileTest, ContextLinesDoNotCountAsIdentifierMatches) {
  FileResult file = File("a", RankTier::kSource, 0);
  file.lines.push_back(ResultLine{.number = 2,
                                  .text = "y",
                                  .is_match = false,
                                  .match_class = MatchClass::kCodeIdentifier});
  EXPECT_EQ(ScoreFile(file), 0);
}

TEST(RankFilesTest, StrictBoundAndTier) {
  FileResult worst = File("worst.cc", RankTier::kSource, 50000);
  worst.score = ScoreFile(worst);
  for (const int64_t max_boost : {kMaxBoost, kMaxBoostWithoutName}) {
    SCOPED_TRACE(max_boost);
    EXPECT_FALSE(CannotOutrank(
        RankKey{.tier = RankTier::kSource,
                .prior = static_cast<uint16_t>(worst.score - max_boost)},
        max_boost, worst));
    EXPECT_TRUE(CannotOutrank(
        RankKey{.tier = RankTier::kSource,
                .prior = static_cast<uint16_t>(worst.score - max_boost - 1)},
        max_boost, worst));
    EXPECT_TRUE(CannotOutrank(RankKey{.tier = RankTier::kTest, .prior = 65535},
                              max_boost, worst));
    EXPECT_FALSE(CannotOutrank(RankKey{.tier = RankTier::kSource, .prior = 0},
                               max_boost, File("x", RankTier::kTest, 0)));
  }
  EXPECT_EQ(kMaxBoostWithoutName, kIdentifierBoost + kMaxMatchCountBoost);
}

TEST(RankFilesTest, TierThenScoreThenPath) {
  std::vector<FileResult> files = {
      File("test_high", RankTier::kTest, 65535, 100,
           MatchClass::kCodeIdentifier, true),
      File("src_b", RankTier::kSource, 1000),
      File("src_a", RankTier::kSource, 1000),
      File("src_boosted", RankTier::kSource, 500, 1,
           MatchClass::kCodeIdentifier),
  };
  RankFiles(files);
  EXPECT_THAT(Paths(files),
              ElementsAre("src_boosted", "src_a", "src_b", "test_high"));
  EXPECT_EQ(files[0].score, 500 + kIdentifierBoost);
  for (size_t i = 1; i < files.size(); ++i) {
    EXPECT_TRUE(RanksBefore(files[i - 1], files[i]));
    EXPECT_FALSE(RanksBefore(files[i], files[i - 1]));
  }
  EXPECT_FALSE(RanksBefore(files[1], files[1]));
}

}  // namespace
}  // namespace ferret

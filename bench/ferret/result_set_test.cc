// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "bench/ferret/result_set.h"

#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/search/search_types.h"

namespace ferret_bench {
namespace {

using ::testing::ElementsAre;

ferret::ResultLine Line(int64_t number, bool is_match) {
  return {.number = number, .is_match = is_match};
}

TEST(MatchedLinesTest, KeepsDistinctMatchesInOrder) {
  ferret::SearchResponse response;
  response.files.push_back(
      {.path = "a/b.c",
       .lines = {Line(3, true), Line(4, false), Line(12, true)}});
  response.files.push_back({.path = "a.c", .lines = {Line(1, true)}});
  response.files.push_back({.path = "a/b.c", .lines = {Line(12, true)}});
  // Paths sort by bytes ('.' < '/'); lines sort numerically (3 < 12).
  EXPECT_THAT(MatchedLines(response),
              ElementsAre(LineRef{"a.c", 1}, LineRef{"a/b.c", 3},
                          LineRef{"a/b.c", 12}));
}

// The vectors are pinned in bench/zoekt/results_test.go too, so both
// harnesses encode result sets identically.
TEST(LineSetDigestTest, MatchesTheZoektHarness) {
  const std::vector<LineRef> lines = {{"a.c", 1}, {"a/b.c", 3}, {"a/b.c", 12}};
  EXPECT_EQ(LineSetDigest(lines),
            "240b57ffcb3e75c91fb996a0f7e1c3ecb3609b68aabbdb12ff6a292d0e4b4adf");
  EXPECT_EQ(LineSetDigest({}),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

}  // namespace
}  // namespace ferret_bench

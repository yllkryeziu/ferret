// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/thread_pool.h"
#include "ferret/index/internal/code_tokens.h"
#include "ferret/postings/ids.h"
#include "ferret/testing/test_seed.h"
#include "ferret/text/language.h"
#include "ferret/text/lexer.h"

namespace ferret::segment_internal {
namespace {

constexpr uint32_t kDocs = 300;
constexpr const char* kWords[] = {"alpha", "Beta", "gamma_1", "DELTA", "eps",
                                  "zeta",  "ETA",  "theta",   "iota",  "kappa"};

std::vector<std::string> Keys(absl::string_view text) {
  std::vector<std::string> keys;
  std::string scratch;
  ForEachCodeTokenKey(Language::kCpp, text, LexState::kCode, scratch,
                      [&](absl::string_view key) { keys.emplace_back(key); });
  return keys;
}

TEST(CodeTokensTest, KeysAreFoldedAndFiltered) {
  EXPECT_THAT(Keys("FooBar baz_Q 12345 9lives"),
              ::testing::IsSupersetOf({"foobar", "baz_q"}));
  for (const std::string& key : Keys("FooBar baz_Q 12345 9lives $x é")) {
    EXPECT_FALSE(key.empty());
    EXPECT_FALSE(key.front() >= '0' && key.front() <= '9') << key;
    for (const char c : key) {
      EXPECT_TRUE(static_cast<uint8_t>(c) < 0x80 && c != '$') << key;
      EXPECT_FALSE(c >= 'A' && c <= 'Z') << key;
    }
  }
  // Longer than the stored limit.
  EXPECT_THAT(Keys(std::string(100, 'a')), ::testing::IsEmpty());
}

// The same documents indexed in one map in rank order, or spread over maps
// under provisional ids with a permutation, give the same section.
TEST(CodeTokensTest, SectionIsIndependentOfHowTheMapsWereFilled) {
  std::mt19937_64 rng(ferret_testing::TestSeed(9));
  std::vector<std::string> texts(kDocs);
  for (std::string& text : texts) {
    const int n = static_cast<int>(rng() % 12);
    for (int i = 0; i < n; ++i) {
      absl::StrAppend(&text, kWords[rng() % std::size(kWords)],
                      rng() % 7 == 0 ? "x" : "", rng() % 40, " ");
    }
  }
  // Final id of provisional doc p.
  std::vector<DocId> final_of(kDocs);
  std::iota(final_of.begin(), final_of.end(), 0);
  std::shuffle(final_of.begin(), final_of.end(), rng);
  std::vector<std::string> by_final(kDocs);
  for (uint32_t p = 0; p < kDocs; ++p) by_final[final_of[p]] = texts[p];

  std::string scratch;
  CodeTokenMap expected_map;
  for (DocId d = 0; d < kDocs; ++d) {
    AddCodeTokens(Language::kCpp, by_final[d], LexState::kCode, d, scratch,
                  expected_map);
  }
  const absl::StatusOr<std::string> expected = BuildCodeTokensSection(
      absl::MakeConstSpan(&expected_map, 1), {}, nullptr, 1);
  ASSERT_THAT(expected.status(), absl_testing::IsOk());

  // Provisional ids ascend within each map, as they do within a worker.
  std::vector<CodeTokenMap> maps(4);
  for (uint32_t p = 0; p < kDocs; ++p) {
    AddCodeTokens(Language::kCpp, texts[p], LexState::kCode, p, scratch,
                  maps[rng() % maps.size()]);
  }
  // The maps were filled in provisional order, so each is ascending.
  ThreadPool pool(3, ThreadQos::kUtility);
  for (const int threads : {1, 4}) {
    const absl::StatusOr<std::string> actual = BuildCodeTokensSection(
        maps, final_of, threads > 1 ? &pool : nullptr, threads);
    ASSERT_THAT(actual.status(), absl_testing::IsOk());
    EXPECT_TRUE(*actual == *expected) << threads;
  }
}

}  // namespace
}  // namespace ferret::segment_internal

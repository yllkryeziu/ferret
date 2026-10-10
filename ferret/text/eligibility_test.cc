// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/eligibility.h"

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "ferret/text/trigram_extractor.h"

namespace ferret {
namespace {

constexpr EligibilityOptions kAgent = {.policy = EligibilityPolicy::kAgent};
constexpr EligibilityOptions kZoekt = {.policy = EligibilityPolicy::kZoekt};

SkipReason Check(absl::string_view content, const EligibilityOptions& options) {
  TrigramExtractor extractor;
  return CheckEligibility(content, options, extractor).skip_reason;
}

EligibilityOptions WithTrigramLimit(EligibilityOptions options, int limit) {
  options.max_distinct_trigrams = limit;
  return options;
}

class EligibilityPolicyTest
    : public ::testing::TestWithParam<EligibilityPolicy> {
 protected:
  EligibilityOptions options() const { return {.policy = GetParam()}; }
};

TEST_P(EligibilityPolicyTest, OrdinarySourceIsEligible) {
  EXPECT_EQ(Check("int main() { return 0; }\n", options()), SkipReason::kNone);
}

TEST_P(EligibilityPolicyTest, EmptyFileIsSkipped) {
  EXPECT_EQ(Check("", options()), SkipReason::kEmpty);
}

TEST_P(EligibilityPolicyTest, SizeLimitIsInclusive) {
  EligibilityOptions small = options();
  small.max_file_bytes = 8;
  EXPECT_EQ(Check("12345678", small), SkipReason::kNone);
  EXPECT_EQ(Check("123456789", small), SkipReason::kTooLarge);
}

TEST_P(EligibilityPolicyTest, NulAnywhereIsSkipped) {
  EXPECT_EQ(Check(absl::string_view("\0abc", 4), options()),
            SkipReason::kContainsNul);
  EXPECT_EQ(Check(absl::string_view("abc\0", 4), options()),
            SkipReason::kContainsNul);
}

TEST_P(EligibilityPolicyTest, TrigramLimitIsInclusive) {
  // "abcdefgh" has six trigrams under either definition.
  EXPECT_EQ(Check("abcdefgh", WithTrigramLimit(options(), 6)),
            SkipReason::kNone);
  EXPECT_EQ(Check("abcdefgh", WithTrigramLimit(options(), 5)),
            SkipReason::kTooManyTrigrams);
}

TEST_P(EligibilityPolicyTest, TooLargeTakesPrecedence) {
  EligibilityOptions small = options();
  small.max_file_bytes = 2;
  EXPECT_EQ(Check(absl::string_view("a\0b", 3), small), SkipReason::kTooLarge);
}

TEST_P(EligibilityPolicyTest, FlagsLinesLongerThanTheLimit) {
  EligibilityOptions short_lines = options();
  short_lines.long_line_bytes = 4;
  TrigramExtractor extractor;
  // The '\n' does not count toward the line length.
  EXPECT_FALSE(
      CheckEligibility("abcd\nefgh\n", short_lines, extractor).long_lines);
  EXPECT_TRUE(
      CheckEligibility("abcd\nefghi\n", short_lines, extractor).long_lines);
  // An unterminated last line counts.
  EXPECT_TRUE(
      CheckEligibility("abcd\nefghi", short_lines, extractor).long_lines);
}

INSTANTIATE_TEST_SUITE_P(BothPolicies, EligibilityPolicyTest,
                         ::testing::Values(EligibilityPolicy::kAgent,
                                           EligibilityPolicy::kZoekt));

TEST(EligibilityTest, DefaultsMatchTheDesign) {
  const EligibilityOptions defaults;
  EXPECT_EQ(defaults.policy, EligibilityPolicy::kAgent);
  EXPECT_EQ(defaults.max_file_bytes, 1 << 20);
  EXPECT_EQ(defaults.max_distinct_trigrams, 20000);
  EXPECT_EQ(defaults.long_line_bytes, 1000);
}

TEST(EligibilityTest, OnlyZoektSkipsOneAndTwoByteFiles) {
  EXPECT_EQ(Check("a", kZoekt), SkipReason::kTooSmall);
  EXPECT_EQ(Check("a\n", kZoekt), SkipReason::kTooSmall);
  EXPECT_EQ(Check("a\n", kAgent), SkipReason::kNone);
  EXPECT_EQ(Check("ab\n", kZoekt), SkipReason::kNone);
}

// For text within the trigram limit, only the size decides.
TEST_P(EligibilityPolicyTest, SizeChecksAgreeWithContentChecks) {
  EligibilityOptions small = options();
  small.max_file_bytes = 8;
  for (const absl::string_view content :
       {"", "a", "ab", "abc", "12345678", "123456789"}) {
    EXPECT_EQ(SkipReasonForSize(static_cast<int64_t>(content.size()), small),
              Check(content, small))
        << content;
  }
}

TEST(EligibilityTest, AgentFoldsCaseAndZoektDoesNot) {
  // Folded: {"aaa"}. Case-sensitive: {"aaa", "aaA", "aAA", "AAA"}.
  constexpr absl::string_view kContent = "aaaaAAAA";
  EXPECT_EQ(Check(kContent, WithTrigramLimit(kAgent, 2)), SkipReason::kNone);
  EXPECT_EQ(Check(kContent, WithTrigramLimit(kZoekt, 2)),
            SkipReason::kTooManyTrigrams);
}

TEST(EligibilityTest, OnlyZoektCountsTrigramsWithNewlines) {
  // Without '\n': {"abc", "def"}. With it, three more.
  constexpr absl::string_view kContent = "abc\ndef";
  EXPECT_EQ(Check(kContent, WithTrigramLimit(kAgent, 2)), SkipReason::kNone);
  EXPECT_EQ(Check(kContent, WithTrigramLimit(kZoekt, 2)),
            SkipReason::kTooManyTrigrams);
}

TEST(EligibilityTest, ZoektCountsRunesAndMergesIllFormedBytes) {
  // Four distinct ill-formed bytes are four U+FFFD runes to Zoekt, one
  // trigram, but two distinct byte trigrams to Ferret.
  constexpr absl::string_view kIllFormed = "\xFF\xFE\xFD\xFC";
  EXPECT_EQ(Check(kIllFormed, WithTrigramLimit(kZoekt, 1)), SkipReason::kNone);
  EXPECT_EQ(Check(kIllFormed, WithTrigramLimit(kAgent, 1)),
            SkipReason::kTooManyTrigrams);
  // Three equal two-byte runes are one rune trigram but two byte trigrams.
  constexpr absl::string_view kRunes = "\xC3\xA9\xC3\xA9\xC3\xA9";
  EXPECT_EQ(Check(kRunes, WithTrigramLimit(kZoekt, 1)), SkipReason::kNone);
  EXPECT_EQ(Check(kRunes, WithTrigramLimit(kAgent, 1)),
            SkipReason::kTooManyTrigrams);
}

TEST(EligibilityTest, SkipReasonNamesAreDistinct) {
  absl::flat_hash_set<absl::string_view> names;
  for (int i = 0; i < kNumSkipReasons; ++i) {
    const absl::string_view name = SkipReasonName(static_cast<SkipReason>(i));
    EXPECT_NE(name, "unknown");
    EXPECT_TRUE(names.insert(name).second) << name;
  }
  EXPECT_EQ(SkipReasonName(SkipReason::kTooManyTrigrams), "too_many_trigrams");
}

}  // namespace
}  // namespace ferret

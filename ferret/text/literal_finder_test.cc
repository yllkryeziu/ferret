// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/literal_finder.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "ferret/text/fold.h"
#include "ferret/text/internal/literal_finder_kernels.h"
#include "ferret/text/testing/exact_buffer.h"

namespace ferret {
namespace {

using ::ferret_testing::ExactBuffer;
using Mode = LiteralFinder::Mode;

constexpr size_t kNpos = absl::string_view::npos;

std::string FoldAscii(absl::string_view text) {
  std::string folded(text);
  for (char& c : folded) {
    c = static_cast<char>(FoldAsciiByte(static_cast<uint8_t>(c)));
  }
  return folded;
}

// An oracle independent of the kernels: std::string_view::find, after folding
// both sides in case-insensitive mode.
size_t ReferenceFind(absl::string_view haystack, absl::string_view needle,
                     Mode mode, size_t from) {
  if (mode == Mode::kExact) return haystack.find(needle, from);
  return absl::string_view(FoldAscii(haystack)).find(FoldAscii(needle), from);
}

TEST(LiteralFinderTest, FindsFirstOccurrenceAtOrAfterFrom) {
  const LiteralFinder finder("abc", Mode::kExact);
  EXPECT_EQ(finder.Find("xxabcxabc"), 2u);
  EXPECT_EQ(finder.Find("xxabcxabc", 2), 2u);
  EXPECT_EQ(finder.Find("xxabcxabc", 3), 6u);
  EXPECT_EQ(finder.Find("xxabcxabc", 7), kNpos);
  EXPECT_EQ(finder.Find("xxabcxabc", 100), kNpos);
  EXPECT_EQ(finder.Find("ab"), kNpos);
  EXPECT_EQ(finder.Find(""), kNpos);
}

TEST(LiteralFinderTest, EmptyNeedleMatchesLikeStringViewFind) {
  const LiteralFinder finder("", Mode::kExact);
  EXPECT_EQ(finder.Find("abc"), 0u);
  EXPECT_EQ(finder.Find("abc", 3), 3u);
  EXPECT_EQ(finder.Find("abc", 4), kNpos);
  EXPECT_EQ(finder.Find(""), 0u);
}

TEST(LiteralFinderTest, ExactModeIsCaseSensitive) {
  const LiteralFinder finder("WebContents", Mode::kExact);
  EXPECT_EQ(finder.Find("webcontents WEBCONTENTS WebContents"), 24u);
}

TEST(LiteralFinderTest, CaseInsensitiveModeFoldsAsciiLettersOnly) {
  const LiteralFinder finder("WebContents", Mode::kAsciiCaseInsensitive);
  EXPECT_EQ(finder.Find("x WEBcontents"), 2u);
  // ORing 0x20 must not make '@' equal '`', '[' equal '{', or 0xC9 equal 0xE9.
  EXPECT_EQ(LiteralFinder("@", Mode::kAsciiCaseInsensitive).Find("`"), kNpos);
  EXPECT_EQ(LiteralFinder("[x", Mode::kAsciiCaseInsensitive).Find("{x"), kNpos);
  EXPECT_EQ(LiteralFinder("\xC3\xA9", Mode::kAsciiCaseInsensitive)
                .Find("\xC3\x89\xC3\xA9"),
            2u);
}

TEST(LiteralFinderTest, CaseInsensitiveModeDoesNotFoldKelvinSign) {
  // Documented limitation: U+212A matches 'k' under RE2's (?i), not here.
  EXPECT_EQ(LiteralFinder("kelvin", Mode::kAsciiCaseInsensitive)
                .Find("\xE2\x84\xAA"
                      "elvin"),
            kNpos);
}

TEST(LiteralFinderTest, FilterHitsAreVerifiedInFull) {
  // The needle consists of 'q' and 'z' only, so a third of these offsets
  // pass the two-byte filter, but only the copy planted at the end matches.
  std::string haystack;
  for (int i = 0; i < 100; ++i) haystack += "qzqzqzq.";
  haystack += "qzqzqzqz";
  EXPECT_EQ(LiteralFinder("qzqzqzqz", Mode::kExact).Find(haystack),
            haystack.size() - 8);
}

// Checks one case against both oracles. The haystack is copied to an
// exactly-sized buffer so that ASan catches any read past its end.
void ExpectAgreesWithOracles(absl::string_view haystack,
                             absl::string_view needle, Mode mode, size_t from) {
  const ExactBuffer buffer(haystack);
  const LiteralFinder finder(needle, mode);
  const size_t expected = ReferenceFind(haystack, needle, mode, from);
  ASSERT_EQ(finder.Find(buffer.view(), from), expected)
      << "needle '" << needle << "' from " << from << " in '" << haystack
      << "'";
  const std::string folded =
      mode == Mode::kExact ? std::string(needle) : FoldAscii(needle);
  const text_internal::LiteralNeedle scalar_needle = {
      .bytes = folded, .fold_ascii = mode != Mode::kExact};
  ASSERT_EQ(
      text_internal::FindLiteralScalar(scalar_needle, buffer.view(), from),
      expected);
}

class LiteralFinderDifferentialTest : public ::testing::TestWithParam<Mode> {};

TEST_P(LiteralFinderDifferentialTest, PlantedNeedleAtEveryBoundary) {
  constexpr uint64_t kSeed = 20260923;
  SCOPED_TRACE(kSeed);
  std::mt19937_64 rng(kSeed);
  // Case variants, bytes that differ from letters only in bit 0x20, and high
  // bytes make partial matches and false filter hits common.
  constexpr absl::string_view kAlphabet = "abcABC@`[{\x80\xA0\xC1\xE1\xFF\n ";
  auto random_text = [&](size_t size) {
    std::string text(size, '\0');
    for (char& c : text) c = kAlphabet[rng() % kAlphabet.size()];
    return text;
  };
  for (size_t length = 1; length <= 64; ++length) {
    const std::string needle = random_text(length);
    // Haystacks from shorter than one vector block to several 64-byte loops.
    for (const size_t size : {length, length + 7, length + 15, length + 16,
                              length + 63, length + 64, length + 200}) {
      std::vector<size_t> positions = {0, 1, 15, 16, 17, 63, 64};
      for (size_t tail = 0; tail < 16; ++tail) {
        positions.push_back(size - length - std::min(tail, size - length));
      }
      for (const size_t position : positions) {
        if (position + length > size) continue;
        std::string haystack = random_text(size);
        haystack.replace(position, length, needle);
        if (GetParam() == Mode::kAsciiCaseInsensitive && rng() % 2 == 0) {
          // Plant a case variant of the needle instead.
          for (size_t i = 0; i < length; ++i) {
            const char c = haystack[position + i];
            if (c >= 'a' && c <= 'z') {
              haystack[position + i] = static_cast<char>(c - ('a' - 'A'));
            }
          }
        }
        for (const size_t from : {size_t{0}, position, position + 1,
                                  static_cast<size_t>(rng() % (size + 2))}) {
          ASSERT_NO_FATAL_FAILURE(
              ExpectAgreesWithOracles(haystack, needle, GetParam(), from));
        }
      }
    }
  }
}

TEST_P(LiteralFinderDifferentialTest, RandomHaystacksWithoutPlanting) {
  constexpr uint64_t kSeed = 99;
  SCOPED_TRACE(kSeed);
  std::mt19937_64 rng(kSeed);
  // A tiny alphabet so that short needles occur by chance at random offsets.
  constexpr absl::string_view kAlphabet = "aAb\xE1";
  for (int iteration = 0; iteration < 3000; ++iteration) {
    std::string needle(1 + rng() % 4, '\0');
    for (char& c : needle) c = kAlphabet[rng() % kAlphabet.size()];
    std::string haystack(rng() % 300, '\0');
    for (char& c : haystack) c = kAlphabet[rng() % kAlphabet.size()];
    ASSERT_NO_FATAL_FAILURE(ExpectAgreesWithOracles(
        haystack, needle, GetParam(), rng() % (haystack.size() + 1)));
  }
}

INSTANTIATE_TEST_SUITE_P(BothModes, LiteralFinderDifferentialTest,
                         ::testing::Values(Mode::kExact,
                                           Mode::kAsciiCaseInsensitive));

}  // namespace
}  // namespace ferret

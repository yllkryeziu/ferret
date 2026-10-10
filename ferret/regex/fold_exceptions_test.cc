// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Checks the premise of the fold transducer F (ferret/text/fold.h) against RE2
// itself: U+212A and U+017F are the only non-ASCII runes that (?i) matches
// to an ASCII letter. If a future RE2 or Unicode version adds another, F must
// learn it or case-insensitive plans become unsound.

#include <cstddef>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "re2/re2.h"
#include "util/utf.h"

#include "ferret/text/fold.h"

namespace ferret {
namespace {

using ::testing::ElementsAre;

std::string EncodeRune(re2::Rune rune) {
  char buffer[re2::UTFmax];
  return std::string(buffer,
                     static_cast<size_t>(re2::runetochar(buffer, &rune)));
}

TEST(FoldExceptionsTest, OnlyKelvinAndLongSFoldToAsciiLetters) {
  const RE2 ascii_letter("(?i)[a-z]");
  ASSERT_TRUE(ascii_letter.ok());
  std::vector<re2::Rune> exceptions;
  for (re2::Rune rune = 0x80; rune <= re2::Runemax; ++rune) {
    if (rune >= 0xD800 && rune <= 0xDFFF) continue;  // Surrogates.
    if (RE2::FullMatch(EncodeRune(rune), ascii_letter)) {
      exceptions.push_back(rune);
    }
  }
  EXPECT_THAT(exceptions, ElementsAre(0x017F, 0x212A));
}

TEST(FoldExceptionsTest, FoldMapsExceptionsToTheirLetters) {
  EXPECT_TRUE(RE2::FullMatch(EncodeRune(0x212A), "(?i)k"));
  EXPECT_TRUE(RE2::FullMatch(EncodeRune(0x017F), "(?i)s"));
  EXPECT_EQ(Fold(EncodeRune(0x212A)), "k");
  EXPECT_EQ(Fold(EncodeRune(0x017F)), "s");
}

}  // namespace
}  // namespace ferret

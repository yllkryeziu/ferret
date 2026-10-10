// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/trigram.h"

#include <string>

#include "gtest/gtest.h"

namespace ferret {
namespace {

TEST(TrigramTest, PacksBigEndianIntoLow24Bits) {
  EXPECT_EQ(MakeTrigram('a', 'b', 'c'), 0x616263u);
  EXPECT_EQ(MakeTrigram('a', 'b', 'c') & ~kTrigramMask, 0u);
}

TEST(TrigramTest, HighBytesAreNotSignExtended) {
  // KELVIN SIGN. A naive (s[0] << 16) | ... over signed char yields
  // 0xffffffaa here.
  const std::string kelvin = "\xE2\x84\xAA";
  EXPECT_EQ(TrigramAt(kelvin, 0), 0xE284AAu);
}

TEST(TrigramTest, ByteAccessorRoundTrips) {
  const Trigram t = MakeTrigram(0x01, 0x80, 0xFF);
  EXPECT_EQ(TrigramByte(t, 0), 0x01);
  EXPECT_EQ(TrigramByte(t, 1), 0x80);
  EXPECT_EQ(TrigramByte(t, 2), 0xFF);
}

TEST(TrigramTest, DetectsNewlineInAnyPosition) {
  EXPECT_TRUE(TrigramHasNewline(MakeTrigram('\n', 'a', 'b')));
  EXPECT_TRUE(TrigramHasNewline(MakeTrigram('a', '\n', 'b')));
  EXPECT_TRUE(TrigramHasNewline(MakeTrigram('a', 'b', '\n')));
  EXPECT_FALSE(TrigramHasNewline(MakeTrigram('a', 'b', 'c')));
}

TEST(TrigramTest, ToStringEscapesNonPrintableBytes) {
  EXPECT_EQ(TrigramToString(MakeTrigram('a', 'b', 'c')), "\"abc\"");
  EXPECT_EQ(TrigramToString(MakeTrigram('a', '\n', 0xE2)), "\"a\\n\\xe2\"");
}

}  // namespace
}  // namespace ferret

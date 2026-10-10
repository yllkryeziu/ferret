// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/fold.h"

#include <string>

#include "gtest/gtest.h"

namespace ferret {
namespace {

TEST(FoldTest, LowercasesAsciiOnly) {
  EXPECT_EQ(Fold("Hello, WORLD_42"), "hello, world_42");
  EXPECT_EQ(Fold("@[`{"), "@[`{");  // Neighbors of A-Z and a-z.
}

TEST(FoldTest, FoldsKelvinSignAndLongS) {
  EXPECT_EQ(Fold("\xE2\x84\xAA"
                 "elvin"),
            "kelvin");
  EXPECT_EQ(Fold("cla\xC5\xBF\xC5\xBF"), "class");
}

TEST(FoldTest, LeavesOtherNonAsciiUnchanged) {
  // U+00C9 LATIN CAPITAL LETTER E WITH ACUTE is not in an ASCII orbit.
  EXPECT_EQ(Fold("\xC3\x89t\xC3\xA9"), "\xC3\x89t\xC3\xA9");
  // U+212B ANGSTROM SIGN shares E2 84 but is not folded.
  EXPECT_EQ(Fold("\xE2\x84\xAB"), "\xE2\x84\xAB");
}

TEST(FoldTest, TruncatedSequencesAreLeftAlone) {
  EXPECT_EQ(Fold("\xE2\x84"), "\xE2\x84");
  EXPECT_EQ(Fold("\xC5"), "\xC5");
  EXPECT_EQ(Fold("A\xE2"), "a\xE2");
}

TEST(FoldTest, AppendsToExistingOutput) {
  std::string out = "x";
  AppendFolded("YZ", out);
  EXPECT_EQ(out, "xyz");
}

TEST(FoldTest, IsIdempotent) {
  const std::string text =
      "Mixed \xE2\x84\xAA and \xC5\xBF, plus \xC3\x89 and \xFF bytes";
  EXPECT_EQ(Fold(Fold(text)), Fold(text));
}

}  // namespace
}  // namespace ferret

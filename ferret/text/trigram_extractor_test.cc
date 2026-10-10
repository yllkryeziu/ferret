// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/trigram_extractor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/text/fold.h"
#include "ferret/text/trigram.h"

namespace ferret {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;

std::vector<Trigram> ExtractAll(absl::string_view text) {
  TrigramExtractor extractor;
  std::vector<Trigram> out;
  extractor.Extract(text, out);
  return out;
}

// The definition from trigram_extractor.h, written as plainly as possible.
absl::flat_hash_set<Trigram> ReferenceTrigrams(absl::string_view text) {
  const std::string folded = Fold(text);
  absl::flat_hash_set<Trigram> set;
  for (size_t i = 0; i + 3 <= folded.size(); ++i) {
    if (!absl::StrContains(folded.substr(i, 3), '\n')) {
      set.insert(TrigramAt(folded, i));
    }
  }
  return set;
}

TEST(TrigramExtractorTest, EmitsDistinctTrigramsInFirstSeenOrder) {
  EXPECT_THAT(ExtractAll("abcabc"), ElementsAre(MakeTrigram('a', 'b', 'c'),
                                                MakeTrigram('b', 'c', 'a'),
                                                MakeTrigram('c', 'a', 'b')));
}

TEST(TrigramExtractorTest, ShortTextHasNoTrigrams) {
  EXPECT_THAT(ExtractAll(""), IsEmpty());
  EXPECT_THAT(ExtractAll("ab"), IsEmpty());
}

TEST(TrigramExtractorTest, SkipsWindowsContainingNewline) {
  EXPECT_THAT(ExtractAll("ab\ncd"), IsEmpty());
  EXPECT_THAT(ExtractAll("abc\ndef"), ElementsAre(MakeTrigram('a', 'b', 'c'),
                                                  MakeTrigram('d', 'e', 'f')));
}

TEST(TrigramExtractorTest, FoldsBeforeExtracting) {
  EXPECT_THAT(ExtractAll("ABC"), ElementsAre(MakeTrigram('a', 'b', 'c')));
  // "\xE2\x84\xAAey" is KELVIN SIGN + "ey", which folds to "key".
  EXPECT_THAT(ExtractAll("\xE2\x84\xAA"
                         "ey"),
              ElementsAre(MakeTrigram('k', 'e', 'y')));
}

TEST(TrigramExtractorTest, BitmapIsClearedBetweenCalls) {
  TrigramExtractor extractor;
  std::vector<Trigram> first, second;
  extractor.Extract("hello", first);
  extractor.Extract("hello", second);
  EXPECT_EQ(first, second);
}

TEST(TrigramExtractorTest, CountDistinctStopsAfterLimit) {
  TrigramExtractor extractor;
  EXPECT_EQ(extractor.CountDistinct("abcdefgh", 100), 6u);
  EXPECT_EQ(extractor.CountDistinct("abcdefgh", 3), 4u);
  // The early exit must still clear the bitmap.
  EXPECT_EQ(extractor.CountDistinct("abcdefgh", 100), 6u);
}

TEST(TrigramExtractorTest, MatchesReferenceOnRandomBytes) {
  std::mt19937 rng(42);
  // A small alphabet with newlines, fold targets, and high bytes makes
  // collisions and boundary cases common.
  constexpr char kAlphabet[] = "aAkK\n\xE2\x84\xAA\xC5\xBFsS\xFF";
  std::uniform_int_distribution<size_t> pick(0, sizeof(kAlphabet) - 2);
  std::uniform_int_distribution<size_t> length(0, 300);
  TrigramExtractor extractor;
  for (int iteration = 0; iteration < 2000; ++iteration) {
    std::string text(length(rng), '\0');
    for (char& c : text) c = kAlphabet[pick(rng)];
    std::vector<Trigram> out;
    extractor.Extract(text, out);
    const absl::flat_hash_set<Trigram> expected = ReferenceTrigrams(text);
    ASSERT_EQ(out.size(), expected.size()) << "text: " << text;
    EXPECT_THAT(out,
                UnorderedElementsAreArray(expected.begin(), expected.end()));
  }
}

std::vector<Trigram> ShortLines(absl::string_view text) {
  TrigramExtractor extractor;
  std::vector<Trigram> out;
  extractor.ExtractShortLines(text, out);
  // Raw bigrams (third byte 0) are tested separately.
  out.erase(std::remove_if(out.begin(), out.end(),
                           [](Trigram t) { return (t & 0xFF) == 0; }),
            out.end());
  return out;
}

TEST(ShortLineTest, EmitsLinesOfExactlyTwoFoldedBytes) {
  EXPECT_THAT(
      ShortLines("};\nx\nabc\n})\n};\n\n"),
      ElementsAre(MakeTrigram('}', ';', '\n'), MakeTrigram('}', ')', '\n')));
  // The last line needs no newline, and the text may be one line.
  EXPECT_THAT(ShortLines("ab"), ElementsAre(MakeTrigram('a', 'b', '\n')));
  EXPECT_THAT(ShortLines("x\nab"), ElementsAre(MakeTrigram('a', 'b', '\n')));
  EXPECT_THAT(ShortLines(""), IsEmpty());
  EXPECT_THAT(ShortLines("a"), IsEmpty());
  EXPECT_THAT(ShortLines("\n\n\n"), IsEmpty());
  // "a\r" is two bytes; "ab\r" is three, so it has a real trigram instead.
  EXPECT_THAT(ShortLines("a\r\nab\r\n"),
              ElementsAre(MakeTrigram('a', '\r', '\n')));
}

TEST(ShortLineTest, UsesTheFoldedImage) {
  EXPECT_THAT(ShortLines("AB\nab\nAb\n"),
              ElementsAre(MakeTrigram('a', 'b', '\n')));
  // KELVIN SIGN + 'a' folds to "ka"; KELVIN SIGN alone and LONG S alone fold
  // to one byte; LONG S + KELVIN SIGN folds to "sk".
  EXPECT_THAT(
      ShortLines("\xE2\x84\xAA"
                 "a\n\xE2\x84\xAA\n\xC5\xBF\n\xC5\xBF\xE2\x84\xAA\n"),
      ElementsAre(MakeTrigram('k', 'a', '\n'), MakeTrigram('s', 'k', '\n')));
  // Two UTF-8 bytes of one rune are a two-byte image; so are two folded
  // runes of up to six bytes.
  EXPECT_THAT(
      ShortLines("\xC3\xA9\n\xE2\x84\xAA\xE2\x84\xAA\n"),
      ElementsAre(MakeTrigram(0xC3, 0xA9, '\n'), MakeTrigram('k', 'k', '\n')));
  // Three folded bytes are not short.
  EXPECT_THAT(ShortLines("\xE2\x84\xAA\xE2\x84\xAA\xE2\x84\xAA\n"), IsEmpty());
}

TEST(ShortLineTest, EmitsRawBigramsWithACapital) {
  TrigramExtractor extractor;
  std::vector<Trigram> out;
  extractor.ExtractShortLines("xIFy AB\nIF\nab if\n(F", out);
  std::vector<Trigram> expected = {
      MakeTrigram('x', 'I', 0),    MakeTrigram('I', 'F', 0),
      MakeTrigram('F', 'y', 0),    MakeTrigram('A', 'B', 0),
      MakeTrigram(' ', 'A', 0),    MakeTrigram('(', 'F', 0),
      MakeTrigram('i', 'f', '\n'), MakeTrigram('(', 'f', '\n')};
  std::sort(out.begin(), out.end());
  std::sort(expected.begin(), expected.end());
  EXPECT_EQ(out, expected);
  // Calls are independent: the dedupe state is cleared.
  out.clear();
  extractor.ExtractShortLines("IF", out);
  EXPECT_THAT(out, UnorderedElementsAre(MakeTrigram('i', 'f', '\n'),
                                        MakeTrigram('I', 'F', 0)));
}

TEST(ShortLineTest, CountDistinctIgnoresThem) {
  TrigramExtractor extractor;
  EXPECT_EQ(extractor.CountDistinct("ab\ncd\n};\n", 100), 0u);
  std::vector<Trigram> out;
  extractor.Extract("ab\ncd\n};\n", out);
  EXPECT_THAT(out, IsEmpty());
  // The bitmap is clear between calls and after repeated lines.
  EXPECT_THAT(ShortLines("ab\nab\nab\n"), SizeIs(1));
  EXPECT_THAT(ShortLines("ab\nab\nab\n"), SizeIs(1));
}

// The claim behind TrigramsCoveringBigram(): a line that contains the folded
// bigram has a trigram in the cover that Extract() or ExtractShortLines()
// reports for its text, on any text and at any chunk boundary.
TEST(ShortLineTest, CoverNamesEveryTextWithTheBigramInALine) {
  std::mt19937 rng(7);
  constexpr char kAlphabet[] = "abkK\n\n\r\xE2\x84\xAA\xC5\xBFsS\xFF";
  std::uniform_int_distribution<size_t> pick(0, sizeof(kAlphabet) - 2);
  std::uniform_int_distribution<size_t> length(0, 12);
  TrigramExtractor extractor;
  int with_bigram = 0;
  for (int iteration = 0; iteration < 20000; ++iteration) {
    std::string text(length(rng), '\0');
    for (char& c : text) c = kAlphabet[pick(rng)];
    const std::string folded = Fold(text);
    std::vector<Trigram> indexed;
    extractor.Extract(text, indexed);
    extractor.ExtractShortLines(text, indexed);
    const absl::flat_hash_set<Trigram> have(indexed.begin(), indexed.end());
    for (const char* bigram : {"ab", "kk", "sk", "aa", "\xFF\xFF", "ks"}) {
      const bool in_a_line = [&] {
        for (absl::string_view line : absl::StrSplit(folded, '\n')) {
          if (absl::StrContains(line, bigram)) return true;
        }
        return false;
      }();
      bool covered = false;
      for (const Trigram t :
           TrigramsCoveringBigram(static_cast<uint8_t>(bigram[0]),
                                  static_cast<uint8_t>(bigram[1]))) {
        covered |= have.contains(t);
      }
      if (in_a_line) {
        ++with_bigram;
        ASSERT_TRUE(covered) << "text " << absl::CEscape(text) << " bigram "
                             << absl::CEscape(bigram);
      }
    }
  }
  EXPECT_GT(with_bigram, 1000);
}

TEST(ShortLineTest, CoverIsSortedAndSkipsFoldedAndNewlineBytes) {
  const std::vector<Trigram> cover = TrigramsCoveringBigram('a', 'b');
  EXPECT_TRUE(std::is_sorted(cover.begin(), cover.end()));
  EXPECT_EQ(std::adjacent_find(cover.begin(), cover.end()), cover.end());
  EXPECT_TRUE(std::binary_search(cover.begin(), cover.end(),
                                 MakeTrigram('a', 'b', '\n')));
  EXPECT_TRUE(std::binary_search(cover.begin(), cover.end(),
                                 MakeTrigram('a', 'b', 'c')));
  EXPECT_TRUE(std::binary_search(cover.begin(), cover.end(),
                                 MakeTrigram(0xFF, 'a', 'b')));
  EXPECT_FALSE(std::binary_search(cover.begin(), cover.end(),
                                  MakeTrigram('a', 'b', 'C')));
  EXPECT_FALSE(std::binary_search(cover.begin(), cover.end(),
                                  MakeTrigram('\n', 'a', 'b')));
  // "aa": abX and Xab coincide for X == 'a'.
  EXPECT_EQ(TrigramsCoveringBigram('a', 'a').size(), 458u);
  EXPECT_THAT(TrigramsCoveringBigram('a', '\n'), IsEmpty());
  EXPECT_THAT(TrigramsCoveringBigram('\n', 'a'), IsEmpty());
}

}  // namespace
}  // namespace ferret

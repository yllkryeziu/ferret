// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/utf8.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <random>
#include <string>

#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ferret {
namespace {

using ::testing::HasSubstr;
using ::testing::StartsWith;

constexpr absl::string_view kReplacement = "\xEF\xBF\xBD";
constexpr absl::string_view kEllipsis = "\xE2\x80\xA6";

std::string Sanitize(absl::string_view text) {
  std::string out;
  AppendSanitizedUtf8(text, out);
  return out;
}

std::string DisplayLine(absl::string_view line, size_t match_begin,
                        size_t match_end, size_t max_bytes) {
  std::string out;
  AppendDisplayLine(line, match_begin, match_end, max_bytes, out);
  return out;
}

// Encodes a code point with the textbook bit layout, without validation.
std::string Encode(uint32_t c) {
  std::string out;
  if (c < 0x80) {
    out.push_back(static_cast<char>(c));
  } else if (c < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (c >> 6)));
    out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
  } else if (c < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (c >> 12)));
    out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (c >> 18)));
    out.push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
  }
  return out;
}

// An oracle built from the definition rather than Table 3-7: a byte string
// is one well-formed sequence iff it is the shortest encoding of a Unicode
// scalar value (not a surrogate, at most U+10FFFF).
std::optional<uint32_t> ReferenceDecode(absl::string_view bytes) {
  if (bytes.empty() || bytes.size() > 4) return std::nullopt;
  const uint8_t lead = static_cast<uint8_t>(bytes[0]);
  constexpr uint8_t kLeadMask[] = {0x7F, 0x1F, 0x0F, 0x07};
  uint32_t c = lead & kLeadMask[bytes.size() - 1];
  for (size_t i = 1; i < bytes.size(); ++i) {
    c = (c << 6) | (static_cast<uint8_t>(bytes[i]) & 0x3F);
  }
  if (c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF)) return std::nullopt;
  if (Encode(c) != bytes) return std::nullopt;
  return c;
}

// True if `bytes` splits into sequences that ReferenceDecode accepts.
bool ReferenceIsValid(absl::string_view bytes) {
  if (bytes.empty()) return true;
  for (size_t length = 1; length <= std::min<size_t>(4, bytes.size());
       ++length) {
    if (ReferenceDecode(bytes.substr(0, length)).has_value() &&
        ReferenceIsValid(bytes.substr(length))) {
      return true;
    }
  }
  return false;
}

TEST(DecodeUtf8Test, DecodesEachLengthAndItsBoundaries) {
  for (const uint32_t c : {0x00u, 0x7Fu, 0x80u, 0x7FFu, 0x800u, 0xD7FFu,
                           0xE000u, 0xFFFFu, 0x10000u, 0x10FFFFu}) {
    const std::string bytes = Encode(c);
    const Utf8Sequence sequence = DecodeUtf8(bytes + "tail");
    EXPECT_EQ(sequence.code_point, static_cast<int32_t>(c)) << c;
    EXPECT_EQ(sequence.length, bytes.size()) << c;
  }
}

TEST(DecodeUtf8Test, ReportsMaximalSubpartOfIllFormedSequences) {
  struct Case {
    absl::string_view bytes;
    size_t length;
  };
  constexpr Case kCases[] = {
      {"\x80", 1},              // Lone continuation byte.
      {"\xC0\xAF", 1},          // C0 and C1 only start overlong forms.
      {"\xE0\x80\x80", 1},      // Overlong: E0 needs A0..BF next.
      {"\xED\xA0\x80", 1},      // Surrogate U+D800: ED needs 80..9F next.
      {"\xF4\x90\x80\x80", 1},  // Above U+10FFFF: F4 needs 80..8F next.
      {"\xF5\x80\x80\x80", 1},  // F5..FF never occur.
      {"\xE2\x84", 2},          // Truncated at the end of the text.
      {"\xE2\x84x", 2},         // Interrupted by ASCII.
      {"\xF0\x90\x80\xC0", 3},  // Interrupted by a lead byte.
  };
  for (const Case& c : kCases) {
    const Utf8Sequence sequence = DecodeUtf8(c.bytes);
    EXPECT_EQ(sequence.code_point, -1) << absl::CHexEscape(c.bytes);
    EXPECT_EQ(sequence.length, c.length) << absl::CHexEscape(c.bytes);
  }
}

TEST(IsValidUtf8Test, AgreesWithTheDefinitionOnShortStrings) {
  // Every string of the given length whose first byte is in [first, 0xFF].
  auto check_all = [](size_t length, uint8_t first) {
    std::string bytes(length, '\0');
    const uint32_t end = 1u << (8 * length);
    for (uint32_t n = uint32_t{first} << (8 * (length - 1)); n < end; ++n) {
      for (size_t i = 0; i < length; ++i) {
        bytes[i] = static_cast<char>(n >> (8 * (length - 1 - i)));
      }
      ASSERT_EQ(IsValidUtf8(bytes), ReferenceIsValid(bytes))
          << absl::CHexEscape(bytes);
    }
  };
  check_all(1, 0x00);
  check_all(2, 0x00);
  // Three-byte strings led by the three- and four-byte lead bytes.
  check_all(3, 0xE0);
}

TEST(IsValidUtf8Test, FourByteSequences) {
  EXPECT_TRUE(IsValidUtf8("\xF0\x90\x80\x80"));
  EXPECT_TRUE(IsValidUtf8("\xF4\x8F\xBF\xBF"));
  EXPECT_FALSE(IsValidUtf8("\xF0\x8F\xBF\xBF"));  // Overlong U+FFFF.
  EXPECT_FALSE(IsValidUtf8("\xF4\x90\x80\x80"));  // U+110000.
  constexpr uint64_t kSeed = 3;
  SCOPED_TRACE(kSeed);
  std::mt19937_64 rng(kSeed);
  for (int i = 0; i < 200000; ++i) {
    std::string bytes(4, '\0');
    // Leads from F0 to F4 and bytes near the continuation range.
    bytes[0] = static_cast<char>(0xF0 + rng() % 5);
    for (size_t j = 1; j < 4; ++j) {
      bytes[j] = static_cast<char>(0x70 + rng() % 0x60);
    }
    ASSERT_EQ(IsValidUtf8(bytes), ReferenceIsValid(bytes))
        << absl::CHexEscape(bytes);
  }
}

TEST(IsValidUtf8Test, FindsErrorsAfterLongAsciiRuns) {
  EXPECT_TRUE(IsValidUtf8(""));
  EXPECT_TRUE(IsValidUtf8(std::string(1000, 'a') + "\xC3\xA9"));
  EXPECT_FALSE(IsValidUtf8(std::string(1000, 'a') + "\xC3"));
  EXPECT_FALSE(IsValidUtf8(std::string(1000, 'a') + "\xFF" + "bcdefgh"));
}

TEST(AppendSanitizedUtf8Test, ReplacesEachMaximalSubpart) {
  // The example of Unicode 15.0 §3.9, Table 3-8.
  EXPECT_EQ(Sanitize("\x61\xF1\x80\x80\xE1\x80\xC2\x62\x80\x63\x80\xBF\x64"),
            absl::StrCat("a", kReplacement, kReplacement, kReplacement, "b",
                         kReplacement, "c", kReplacement, kReplacement, "d"));
  EXPECT_EQ(Sanitize("\xED\xA0\x80"),
            absl::StrCat(kReplacement, kReplacement, kReplacement));
  EXPECT_EQ(Sanitize("ok\xE2\x84"), absl::StrCat("ok", kReplacement));
}

TEST(AppendSanitizedUtf8Test, KeepsValidTextAndAppends) {
  std::string out = "prefix:";
  AppendSanitizedUtf8("caf\xC3\xA9 \xE2\x84\xAA", out);
  EXPECT_EQ(out, "prefix:caf\xC3\xA9 \xE2\x84\xAA");
}

TEST(AppendSanitizedUtf8Test, OutputIsValidAndIdempotentOnRandomBytes) {
  constexpr uint64_t kSeed = 11;
  SCOPED_TRACE(kSeed);
  std::mt19937_64 rng(kSeed);
  for (int i = 0; i < 20000; ++i) {
    std::string bytes(rng() % 40, '\0');
    for (char& c : bytes) {
      // Half ASCII, half high bytes, so valid sequences occur too.
      c = static_cast<char>(rng() % 2 == 0 ? rng() % 0x80
                                           : 0x80 + rng() % 0x80);
    }
    const std::string sanitized = Sanitize(bytes);
    ASSERT_TRUE(IsValidUtf8(sanitized)) << absl::CHexEscape(bytes);
    ASSERT_EQ(Sanitize(sanitized), sanitized);
    if (IsValidUtf8(bytes)) ASSERT_EQ(sanitized, bytes);
  }
}

TEST(AppendDisplayLineTest,
     ShortLineIsSanitizedAndLosesTrailingCarriageReturn) {
  EXPECT_EQ(DisplayLine("int x;\r", 4, 5, 256), "int x;");
  EXPECT_EQ(DisplayLine("a\xFF"
                        "b",
                        0, 1, 256),
            absl::StrCat("a", kReplacement, "b"));
  EXPECT_EQ(DisplayLine("", 0, 0, 256), "");
}

TEST(AppendDisplayLineTest, LongLineIsCutAroundTheMatch) {
  const std::string line =
      std::string(1000, 'a') + "NEEDLE" + std::string(1000, 'b');
  const std::string shown = DisplayLine(line, 1000, 1006, 64);
  EXPECT_EQ(shown.size(), 64u);
  EXPECT_EQ(shown, absl::StrCat(kEllipsis, std::string(26, 'a'), "NEEDLE",
                                std::string(26, 'b'), kEllipsis));
}

TEST(AppendDisplayLineTest, MatchNearAnEndGivesTheRestToTheOtherSide) {
  const std::string line = "NEEDLE" + std::string(1000, 'b');
  EXPECT_EQ(DisplayLine(line, 0, 6, 32),
            absl::StrCat("NEEDLE", std::string(23, 'b'), kEllipsis));
  const std::string tail = std::string(1000, 'a') + "NEEDLE";
  EXPECT_EQ(DisplayLine(tail, 1000, 1006, 32),
            absl::StrCat(kEllipsis, std::string(23, 'a'), "NEEDLE"));
}

TEST(AppendDisplayLineTest, OverlongMatchShowsItsStart) {
  const std::string line =
      std::string(100, 'a') + std::string(100, 'M') + std::string(100, 'b');
  EXPECT_EQ(DisplayLine(line, 100, 200, 16),
            absl::StrCat(kEllipsis, std::string(10, 'M'), kEllipsis));
}

TEST(AppendDisplayLineTest, EmptyMatchAtZeroKeepsThePrefix) {
  EXPECT_EQ(DisplayLine(std::string(300, 'x'), 0, 0, 16),
            absl::StrCat(std::string(13, 'x'), kEllipsis));
}

TEST(AppendDisplayLineTest, NeverSplitsASequence) {
  // Two-byte runes, and a budget that leaves an odd number of bytes.
  std::string line;
  for (int i = 0; i < 200; ++i) line += "\xC3\xA9";
  const std::string shown = DisplayLine(line, 200, 202, 17);
  EXPECT_TRUE(IsValidUtf8(shown));
  EXPECT_LE(shown.size(), 17u);
  EXPECT_THAT(shown, StartsWith(kEllipsis));
}

TEST(AppendDisplayLineTest, ReplacementCharactersCountTowardTheBudget) {
  // 90 invalid bytes display as 270 bytes, over a budget of 256.
  const std::string line(90, '\xFF');
  const std::string shown = DisplayLine(line, 45, 46, 256);
  EXPECT_LE(shown.size(), 256u);
  EXPECT_TRUE(IsValidUtf8(shown));
  EXPECT_THAT(shown, HasSubstr(kEllipsis));
}

TEST(AppendDisplayLineTest, RandomLinesStayWithinBudgetAndKeepTheMatch) {
  constexpr uint64_t kSeed = 5;
  SCOPED_TRACE(kSeed);
  std::mt19937_64 rng(kSeed);
  constexpr absl::string_view kPieces[] = {"a",
                                           "b",
                                           " ",
                                           "\xC3\xA9",
                                           "\xE2\x84\xAA",
                                           "\xF0\x9F\x98\x80",
                                           "\xFF",
                                           "\x80",
                                           "\xE2\x84",
                                           "\r"};
  for (int i = 0; i < 20000; ++i) {
    std::string line;
    const size_t pieces = rng() % 200;
    for (size_t p = 0; p < pieces; ++p) {
      line += kPieces[rng() % std::size(kPieces)];
    }
    const size_t max_bytes = 6 + rng() % 120;
    const size_t match_begin = line.empty() ? 0 : rng() % line.size();
    const size_t match_end =
        match_begin +
        rng() % (std::min<size_t>(line.size() - match_begin, 8) + 1);
    const std::string shown =
        DisplayLine(line, match_begin, match_end, max_bytes);
    SCOPED_TRACE(absl::StrCat(absl::CHexEscape(line), " [", match_begin, ", ",
                              match_end, ") max ", max_bytes));
    ASSERT_LE(shown.size(), max_bytes);
    ASSERT_TRUE(IsValidUtf8(shown));
    // A match whose sanitized form fits between two ellipses is shown whole.
    // Only matches on sequence boundaries are checked, since the window
    // widens a match that starts or ends inside a sequence.
    const absl::string_view match =
        absl::string_view(line).substr(match_begin, match_end - match_begin);
    const std::string sanitized_match = Sanitize(match);
    const bool on_boundaries =
        Sanitize(absl::string_view(line).substr(0, match_begin)) +
            sanitized_match +
            Sanitize(absl::string_view(line).substr(match_end)) ==
        Sanitize(line);
    if (on_boundaries && !absl::StrContains(match, '\r') &&
        sanitized_match.size() + 2 * kEllipsis.size() <= max_bytes) {
      ASSERT_THAT(shown, HasSubstr(sanitized_match));
    }
  }
}

}  // namespace
}  // namespace ferret

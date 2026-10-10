// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/search/verifier.h"

#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "re2/re2.h"

#include "ferret/regex/compile.h"
#include "ferret/regex/testing/regex_gen.h"
#include "ferret/testing/test_seed.h"

namespace ferret {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

std::vector<LineMatch> Verify(const CompiledPattern& pattern,
                              absl::string_view text) {
  LineVerifier verifier(pattern);
  std::vector<LineMatch> out;
  verifier.ForEachMatchingLine(text, [&](const LineMatch& match) {
    out.push_back(match);
    return true;
  });
  return out;
}

// The definition: split the text into lines, where a final '\n' ends the
// last line, and match each line on its own.
std::vector<LineMatch> Reference(const CompiledPattern& pattern,
                                 absl::string_view text) {
  std::vector<LineMatch> out;
  size_t begin = 0;
  while (begin < text.size()) {
    size_t end = text.find('\n', begin);
    if (end == absl::string_view::npos) end = text.size();
    const absl::string_view line = text.substr(begin, end - begin);
    absl::string_view match;
    if (pattern.matcher().Match(line, 0, line.size(), RE2::UNANCHORED, &match,
                                1)) {
      const size_t match_begin =
          begin + static_cast<size_t>(match.data() - line.data());
      out.push_back({begin, end, match_begin, match_begin + match.size()});
    }
    begin = end + 1;
  }
  return out;
}

std::unique_ptr<CompiledPattern> Compile(const std::string& pattern,
                                         bool literal = false,
                                         bool case_sensitive = true) {
  absl::StatusOr<std::unique_ptr<CompiledPattern>> compiled =
      CompilePattern({.pattern = pattern,
                      .literal = literal,
                      .case_sensitive = case_sensitive});
  EXPECT_TRUE(compiled.ok()) << pattern << ": " << compiled.status();
  return compiled.ok() ? *std::move(compiled) : nullptr;
}

TEST(LineVerifierTest, ReportsEachMatchingLineOnce) {
  const auto pattern = Compile("foo");
  ASSERT_NE(pattern, nullptr);
  EXPECT_THAT(Verify(*pattern, "foo foo\nbar\nxfoo"),
              ElementsAre(LineMatch{0, 7, 0, 3}, LineMatch{12, 16, 13, 16}));
}

TEST(LineVerifierTest, NoPhantomLineAfterFinalNewline) {
  const auto empty_line = Compile("^$");
  ASSERT_NE(empty_line, nullptr);
  EXPECT_THAT(Verify(*empty_line, "a\n\nb\n"),
              ElementsAre(LineMatch{2, 2, 2, 2}));
  EXPECT_THAT(Verify(*empty_line, "\n"), ElementsAre(LineMatch{0, 0, 0, 0}));
  EXPECT_THAT(Verify(*empty_line, "a\n"), IsEmpty());
}

TEST(LineVerifierTest, TextAnchorsMeanLineBoundaries) {
  for (const std::string p : {"\\Afoo", "foo\\z", "(?-m:^foo$)"}) {
    const auto pattern = Compile(p);
    ASSERT_NE(pattern, nullptr);
    EXPECT_TRUE(pattern->has_text_anchor()) << p;
    EXPECT_THAT(Verify(*pattern, "x\nfoo\nfoo"),
                ElementsAre(LineMatch{2, 5, 2, 5}, LineMatch{6, 9, 6, 9}))
        << p;
  }
}

TEST(LineVerifierTest, LiteralPathMatchesTheLineOnItsOwn) {
  // "\Afoo" has the required literal "foo", so the literal path runs.
  const auto pattern = Compile("\\Afoobar");
  ASSERT_NE(pattern, nullptr);
  ASSERT_FALSE(pattern->required_literal().empty());
  EXPECT_THAT(Verify(*pattern, "xfoobar\nfoobar\n"),
              ElementsAre(LineMatch{8, 14, 8, 14}));
}

TEST(LineVerifierTest, ShortLiteralPrefilterMatchesTheReference) {
  const char* patterns[] = {"a",  "ab", "AB",    ";\\(",  "x[0-9]",
                            "^a", "a$", "\\bab", "(?i)k", "ks"};
  const std::string text = std::string("ab\nxab\nAB\nA;(\n;(x\nx5\n\nabab\n") +
                           "a\n\xc3\xa9"
                           "a\nb\n\xe2\x84\xaa\n";
  for (const char* p : patterns) {
    for (const bool case_sensitive : {true, false}) {
      SCOPED_TRACE(std::string(p) + (case_sensitive ? " cs" : " ci"));
      const auto pattern = Compile(p, /*literal=*/false, case_sensitive);
      ASSERT_NE(pattern, nullptr);
      EXPECT_THAT(Verify(*pattern, text),
                  ElementsAreArray(Reference(*pattern, text)));
    }
  }
  const auto pair = Compile("ab");
  ASSERT_NE(pair, nullptr);
  EXPECT_TRUE(pair->required_literal().empty());
  EXPECT_EQ(pair->scan_literal(), "ab");
}

// A plain literal needs no RE2: the finder's hit is the match. Every literal
// over a small alphabet, in both cases, against the definition.
TEST(LineVerifierTest, ExactLiteralMatchesTheReference) {
  std::mt19937_64 rng(ferret_testing::TestSeed(37));
  constexpr char kAlphabet[] = "abAB;( \xC3\xA9\xC3\x89\n";
  int exact = 0;
  for (int iteration = 0; iteration < 4000; ++iteration) {
    std::string literal;
    const int length = 1 + static_cast<int>(rng() % 3);
    while (static_cast<int>(literal.size()) < length) {
      const char c = kAlphabet[rng() % (sizeof(kAlphabet) - 1)];
      if (c != '\n') literal.push_back(c);
    }
    const bool case_sensitive = rng() % 2 == 0;
    absl::StatusOr<std::unique_ptr<CompiledPattern>> pattern =
        CompilePattern({.pattern = literal,
                        .literal = true,
                        .case_sensitive = case_sensitive});
    if (!pattern.ok()) continue;
    exact += (*pattern)->exact_literal() != nullptr;
    std::string text;
    const int size = static_cast<int>(rng() % 40);
    while (static_cast<int>(text.size()) < size) {
      text.push_back(kAlphabet[rng() % (sizeof(kAlphabet) - 1)]);
    }
    EXPECT_THAT(Verify(**pattern, text),
                ElementsAreArray(Reference(**pattern, text)))
        << absl::CEscape(literal) << " in " << absl::CEscape(text) << " cs "
        << case_sensitive;
  }
  EXPECT_GT(exact, 1000);
}

TEST(LineVerifierTest, CaseInsensitiveLiteral) {
  const auto pattern = Compile("Connect", /*literal=*/true,
                               /*case_sensitive=*/false);
  ASSERT_NE(pattern, nullptr);
  EXPECT_THAT(Verify(*pattern, "CONNECT()\nconnection\nnope"),
              ElementsAre(LineMatch{0, 9, 0, 7}, LineMatch{10, 20, 10, 17}));
}

TEST(LineVerifierTest, StopsWhenAsked) {
  const auto pattern = Compile("a");
  ASSERT_NE(pattern, nullptr);
  LineVerifier verifier(*pattern);
  int calls = 0;
  verifier.ForEachMatchingLine("a\na\na\n",
                               [&](const LineMatch&) { return ++calls < 2; });
  EXPECT_EQ(calls, 2);
}

// All three scanning strategies against the definition, on the adversarial
// generator the planner's soundness test uses.
TEST(LineVerifierTest, RandomPatternsMatchReference) {
  std::mt19937_64 rng(ferret_testing::TestSeed(31));
  int compiled = 0;
  int with_literal = 0;
  int with_anchor = 0;
  for (int iteration = 0; iteration < 3000; ++iteration) {
    const std::string pattern_text = ferret_testing::RandomPattern(rng);
    const bool case_sensitive = rng() % 2 == 0;
    absl::StatusOr<std::unique_ptr<CompiledPattern>> pattern = CompilePattern(
        {.pattern = pattern_text, .case_sensitive = case_sensitive});
    if (!pattern.ok()) continue;
    ++compiled;
    with_literal += !(*pattern)->required_literal().empty();
    with_anchor += (*pattern)->has_text_anchor();
    for (int t = 0; t < 4; ++t) {
      std::string text;
      for (int piece = 0; piece < 1 + static_cast<int>(rng() % 4); ++piece) {
        absl::StrAppend(&text, ferret_testing::RandomText(rng),
                        rng() % 2 == 0 ? "\n" : "");
      }
      ASSERT_THAT(Verify(**pattern, text),
                  ElementsAreArray(Reference(**pattern, text)))
          << "pattern " << pattern_text << " text \"" << absl::CEscape(text)
          << "\"";
    }
  }
  EXPECT_GT(compiled, 1000);
  EXPECT_GT(with_literal, 100);
  EXPECT_GT(with_anchor, 10);
}

}  // namespace
}  // namespace ferret

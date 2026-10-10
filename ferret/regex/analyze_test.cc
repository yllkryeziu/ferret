// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/regex/analyze.h"

#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "re2/re2.h"
#include "re2/regexp.h"

#include "ferret/query/trigram_query.h"
#include "ferret/regex/compile.h"

namespace ferret {
namespace {

struct GoldenPlan {
  const char* pattern;
  const char* query;
  const char* required_literal;
};

// Each plan was checked by hand against the analyzer's rules. Strings are
// shown C-escaped, as TrigramQuery::ToString() prints them.
constexpr GoldenPlan kGoldenPlans[] = {
    // Literals: every trigram, folded; shorter than a trigram means All.
    {"hello", R"("ell" & "hel" & "llo")", "hello"},
    {"he", "All", ""},
    {"Hello", R"("ell" & "hel" & "llo")", "hello"},
    // (?i)k is the class {K, k, U+212A}, which folds to {"k"}: same plan, but
    // no required 'k' because the text may spell it as U+212A.
    {"(?i)Kelvin", R"("elv" & "kel" & "lvi" & "vin")", "elvin"},
    {"(?i)ask", R"("ask")", ""},
    {"a\\x{212A}c", R"("akc")", ""},
    // Concatenation and alternation.
    {"foo.*bar", R"("bar" & "foo")", "foo"},
    {"foo|bar", R"("bar" | "foo")", ""},
    {"a[bc]d", R"("abd" | "acd")", ""},
    {"(?:ab|cd)(?:ef|gh)",
     R"(("abe" & "bef") | ("abg" & "bgh") | ("cde" & "def") | ("cdg" & "dgh"))",
     ""},
    // Common trigrams are factored out of an OR.
    {"(foo|foobar)baz", R"("baz" & "foo" & "oba" & "oob")", "foo"},
    {"(a|b|c|d|e|f|g|h)xyz",
     R"("xyz" & ("axy" | "bxy" | "cxy" | "dxy" | "exy" | "fxy" | "gxy" | "hxy"))",
     "xyz"},
    // Repetition.
    {"x{0}abc", R"("abc")", "abc"},
    {"(abc){2}", R"("abc" & "bca" & "cab")", "abcabc"},
    {"(abc){2,5}", R"("abc")", "abc"},
    {"(abc)+", R"("abc")", "abc"},
    {"(abc)*d", "All", ""},
    {"a+bcd", R"("abc" & "bcd")", "bcd"},
    {"abc?", "All", ""},
    // Assertions match "".
    {"\\bfoo\\b", R"("foo")", "foo"},
    {"^foo$", R"("foo")", "foo"},
    // Classes too large to enumerate match any character.
    {"[^x]yz", "All", ""},
    {"\\p{Greek}\\p{Greek}\\p{Greek}", "All", ""},
    {"[a-z]+", "All", ""},
    // \s excludes '\n' under NeverNL.
    {"func\\s+Name\\(",
     R"("ame" & "fun" & "me(" & "nam" & "unc" & ("nc\t" | "nc\x0c" | "nc\r" | "nc ") & ("\tna" | "\x0cna" | "\rna" | " na"))",
     "name("},
    {"Name\\w*Error", R"("ame" & "err" & "nam" & "ror" & "rro")", "error"},
    // UTF-8: trigrams straddle runes; é/É are a class F does not fold.
    {"r\xC3\xA9sum\xC3\xA9",
     R"("m\xc3\xa9" & "r\xc3\xa9" & "sum" & "um\xc3" & "\xa9su" & "\xc3\xa9s")",
     "r\xC3\xA9sum\xC3\xA9"},
    {"\\Qa.b\\E", R"("a.b")", "a.b"},
    {"abc|", "All", ""},
};

TEST(TrigramAnalyzerTest, GoldenPlans) {
  for (const GoldenPlan& golden : kGoldenPlans) {
    SCOPED_TRACE(golden.pattern);
    absl::StatusOr<std::unique_ptr<CompiledPattern>> compiled =
        CompilePattern({.pattern = golden.pattern});
    ASSERT_TRUE(compiled.ok()) << compiled.status();
    EXPECT_EQ((*compiled)->query().ToString(), golden.query);
    EXPECT_EQ((*compiled)->required_literal(), golden.required_literal);
    EXPECT_TRUE((*compiled)->plan_complete());
  }
}

TEST(TrigramAnalyzerTest, WideOrsAreWeakened) {
  absl::StatusOr<std::unique_ptr<CompiledPattern>> compiled =
      CompilePattern({.pattern = "[0-9a-f]xyz"}, {.or_fanout = 4});
  ASSERT_TRUE(compiled.ok());
  EXPECT_EQ((*compiled)->query().ToString(), R"("xyz")");
}

TEST(TrigramAnalyzerTest, VisitBudgetYieldsAllAndIncomplete) {
  // Far more nodes than the visit budget, parsed directly because
  // CompilePattern() caps pattern size well below it.
  std::string pattern;
  for (int i = 0; i < 60000; ++i) absl::StrAppend(&pattern, "[ab]c?");
  re2::RegexpStatus status;
  re2::Regexp* re = re2::Regexp::Parse(pattern, re2::Regexp::LikePerl, &status);
  ASSERT_NE(re, nullptr) << status.Text();
  TrigramAnalyzer analyzer((RegexPlanOptions()));
  const RegexPlan plan = analyzer.Analyze(re);
  re->Decref();
  EXPECT_FALSE(plan.complete);
  EXPECT_EQ(plan.query, TrigramQuery::All());
  EXPECT_EQ(plan.required_literal, "");
}

bool MatchesNothingParsed(const char* pattern) {
  re2::RegexpStatus status;
  re2::Regexp* re =
      re2::Regexp::Parse(pattern,
                         static_cast<re2::Regexp::ParseFlags>(
                             re2::Regexp::LikePerl | re2::Regexp::NeverNL),
                         &status);
  CHECK(re != nullptr) << status.Text();
  const bool result = MatchesNothing(re);
  re->Decref();
  return result;
}

TEST(MatchesNothingTest, DetectsEmptyLanguages) {
  EXPECT_TRUE(MatchesNothingParsed("\\n"));
  EXPECT_TRUE(MatchesNothingParsed("a\\nb"));
  EXPECT_TRUE(MatchesNothingParsed("[\\n]"));
  EXPECT_TRUE(MatchesNothingParsed("(?:\\n)+"));
  EXPECT_TRUE(MatchesNothingParsed("(?:\\n){2,}"));
  EXPECT_FALSE(MatchesNothingParsed("a|\\n"));
  EXPECT_FALSE(MatchesNothingParsed("\\n*"));
  EXPECT_FALSE(MatchesNothingParsed("\\n?"));
  EXPECT_FALSE(MatchesNothingParsed("(?:\\n){0}"));
  EXPECT_FALSE(MatchesNothingParsed("^$"));
}

}  // namespace
}  // namespace ferret

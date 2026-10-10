// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/regex/compile.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "re2/re2.h"

#include "ferret/query/trigram_query.h"

namespace ferret {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

std::unique_ptr<CompiledPattern> MustCompile(const PatternSpec& spec) {
  absl::StatusOr<std::unique_ptr<CompiledPattern>> compiled =
      CompilePattern(spec);
  CHECK_OK(compiled.status()) << spec.pattern;
  return *std::move(compiled);
}

TEST(CompilePatternTest, RejectsEmptyPattern) {
  EXPECT_THAT(CompilePattern({.pattern = ""}),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("empty")));
}

TEST(CompilePatternTest, RejectsPatternsOverTheSizeLimit) {
  EXPECT_THAT(
      CompilePattern({.pattern = std::string(kMaxPatternBytes + 1, 'a')}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("limit")));
  EXPECT_TRUE(
      CompilePattern({.pattern = std::string(kMaxPatternBytes, 'a')}).ok());
}

TEST(CompilePatternTest, RejectsSyntaxErrors) {
  EXPECT_THAT(CompilePattern({.pattern = "a(b"}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("invalid pattern")));
}

TEST(CompilePatternTest, RejectsAnyByte) {
  EXPECT_THAT(CompilePattern({.pattern = "a\\Cb"}),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("\\C")));
}

TEST(CompilePatternTest, RejectsPatternsThatNeedANewline) {
  for (const char* pattern :
       {"\\n", "foo\\nbar", "[\\n]", "a(?:\\n|\\r\\n)b"}) {
    SCOPED_TRACE(pattern);
    EXPECT_THAT(
        CompilePattern({.pattern = pattern}),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("newline")));
  }
  EXPECT_THAT(
      CompilePattern({.pattern = "a\nb", .literal = true}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("newline")));
}

TEST(CompilePatternTest, AcceptsPatternsWhereNewlinesAreOptional) {
  // \s and negated classes contain '\n' only without NeverNL; an alternative
  // or an optional newline leaves a line-local match.
  for (const char* pattern :
       {"foo\\s+bar", "a[^x]b", "a|\\n", "a\\n?b", "(?s)a.b"}) {
    SCOPED_TRACE(pattern);
    EXPECT_TRUE(CompilePattern({.pattern = pattern}).ok());
  }
}

TEST(CompilePatternTest, EmptyLanguageIsNotAnError) {
  const std::unique_ptr<CompiledPattern> compiled =
      MustCompile({.pattern = "[^\\x00-\\x{10FFFF}]"});
  EXPECT_EQ(compiled->query(), TrigramQuery::None());
}

TEST(CompilePatternTest, MatchesWithinALineOnly) {
  const std::unique_ptr<CompiledPattern> compiled =
      MustCompile({.pattern = "foo\\s+bar"});
  EXPECT_TRUE(RE2::PartialMatch("x foo \t bar y", compiled->matcher()));
  EXPECT_FALSE(RE2::PartialMatch("foo\nbar", compiled->matcher()));
  const std::unique_ptr<CompiledPattern> dot =
      MustCompile({.pattern = "(?s)a.b"});
  EXPECT_FALSE(RE2::PartialMatch("a\nb", dot->matcher()));
}

TEST(CompilePatternTest, AnchorsMatchAtLineBoundaries) {
  const std::unique_ptr<CompiledPattern> compiled =
      MustCompile({.pattern = "^foo$"});
  EXPECT_TRUE(RE2::PartialMatch("bar\nfoo\nbaz", compiled->matcher()));
}

TEST(CompilePatternTest, LiteralModeQuotesMetacharacters) {
  const std::unique_ptr<CompiledPattern> compiled =
      MustCompile({.pattern = "a.b(", .literal = true});
  EXPECT_TRUE(RE2::PartialMatch("xa.b(y", compiled->matcher()));
  EXPECT_FALSE(RE2::PartialMatch("axb(", compiled->matcher()));
  EXPECT_EQ(compiled->query().ToString(), "\".b(\" & \"a.b\"");
  EXPECT_EQ(compiled->required_literal(), "a.b(");
}

TEST(CompilePatternTest, CaseInsensitiveMatchesFoldOrbits) {
  const std::unique_ptr<CompiledPattern> compiled =
      MustCompile({.pattern = "Kelvin", .case_sensitive = false});
  EXPECT_TRUE(RE2::PartialMatch("KELVIN", compiled->matcher()));
  EXPECT_TRUE(
      RE2::PartialMatch("\xE2\x84\xAA"
                        "elvin",
                        compiled->matcher()));
  // The plan is the same as for the case-sensitive pattern: the index folds.
  EXPECT_EQ(compiled->query(), MustCompile({.pattern = "Kelvin"})->query());
  // U+212A folds to 'k' but is not ASCII, so 'k' cannot be required.
  EXPECT_EQ(compiled->required_literal(), "elvin");
}

TEST(CompilePatternTest, ScanLiteralIsAlsoSetForShortLiterals) {
  struct Case {
    const char* pattern;
    bool case_sensitive;
    const char* required;
    const char* scan;
  };
  const Case cases[] = {
      {"ab", true, "", "ab"},
      {"a", true, "", "a"},
      {"AB", true, "", "ab"},
      {"abc", true, "abc", "abc"},
      {"a.b", true, "", "a"},
      {"x[0-9]", true, "", "x"},
      {"(a|b)", true, "", ""},
      {"a*", true, "", ""},
      {".", true, "", ""},
      // U+212A and U+017F fold to 'k' and 's' only under (?i).
      {"k", false, "", ""},
      {"s", false, "", ""},
      {"ks", false, "", ""},
      {"ak", false, "", "a"},
      {"k", true, "", "k"},
      {"ks", true, "", "ks"},
      {"\xC3\xA9", true, "", "\xC3\xA9"},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.pattern);
    const std::unique_ptr<CompiledPattern> compiled =
        MustCompile({.pattern = c.pattern, .case_sensitive = c.case_sensitive});
    EXPECT_EQ(compiled->required_literal(), c.required);
    EXPECT_EQ(compiled->scan_literal(), c.scan);
  }
}

TEST(CompilePatternTest, ScanAlternativesNameEachBranchOfAnAlternation) {
  struct Case {
    const char* pattern;
    bool case_sensitive;
    std::vector<std::string> alternatives;
  };
  const Case cases[] = {
      {"(?:WindowHandler|ConvertSettingsToDictionary)",
       true,
       {"convertsettingstodictionary", "windowhandler"}},
      {"foo|bar|bazz", true, {"bar", "bazz", "foo"}},
      {"(a|b)", true, {"a", "b"}},
      // A longer single literal wins; the alternatives are only a fallback.
      {"(?:foo|bar)xyz", true, {}},
      // A branch's longest required string may be short.
      {"foo|b.r", true, {"b", "foo"}},
      // One branch without a required string: nothing is required.
      {"foo|[a-z]+", true, {}},
      {"foo|", true, {}},
      // An optional branch.
      {"(?:foo)?bar", true, {}},
      // Branches may be regexes with their own required string.
      {"fo+bar|ba[rz]+qux", true, {"bar", "qux"}},
      // More strings than kMaxScanAlternatives.
      {"a1|b2|c3|d4|e5|f6|g7|h8|i9", true, {}},
      // U+212A and U+017F fold to 'k' and 's' only under (?i): a branch whose
      // longest k/s-free run is empty makes the set empty, and the others
      // use their runs.
      {"ks|foo", false, {}},
      {"kast|foo", false, {"a", "foo"}},
      {"kast|foo", true, {"foo", "kast"}},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.pattern);
    const std::unique_ptr<CompiledPattern> compiled =
        MustCompile({.pattern = c.pattern, .case_sensitive = c.case_sensitive});
    EXPECT_EQ(compiled->scan_alternatives(), c.alternatives);
    if (!c.alternatives.empty()) EXPECT_TRUE(compiled->scan_literal().empty());
  }
}

TEST(CompilePatternTest, ExactLiteralOnlyForPlainLiteralsAByteSearchDecides) {
  struct Case {
    const char* pattern;
    bool literal;
    bool case_sensitive;
    bool exact;
  };
  const Case cases[] = {
      {"foo", true, true, true},        {"Foo", true, true, true},
      {"a.b(", true, true, true},       {"\xC3\xA9", true, true, true},
      {"foo", false, true, false},  // a regular expression
      {"Foo", true, false, true},   // ASCII, no k or s
      {"foo", true, false, true},       {"bar", true, false, true},
      {"key", true, false, false},       // U+212A matches (?i)k
      {"Sx", true, false, false},        // U+017F matches (?i)s
      {"\xC3\xA9", true, false, false},  // (?i) folds U+00C9 and U+00E9
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(absl::StrCat(c.pattern, " literal ", c.literal, " cs ",
                              c.case_sensitive));
    const std::unique_ptr<CompiledPattern> compiled =
        MustCompile({.pattern = c.pattern,
                     .literal = c.literal,
                     .case_sensitive = c.case_sensitive});
    EXPECT_EQ(compiled->exact_literal() != nullptr, c.exact);
  }
}

TEST(CompilePatternTest, MatchesWithinAgreesWithRe2) {
  const std::string texts[] = {
      "",
      "foo",
      "FOO",
      "xFoOx",
      "Foo",
      "fo",
      "a.b(",
      "axb(",
      "\xC3\xA9",
      "\xC3\x89",
      "\xE2\x84\xAA"  // NOLINT(bugprone-suspicious-missing-comma)
      "ey",
      "KEY",
      "key",
      "sx",
      "\xC5\xBF"
      "x",
      "SX"};
  const std::string patterns[] = {"foo", "Foo", "a.b(",    "key",
                                  "Sx",  "x",   "\xC3\xA9"};
  for (const std::string& pattern : patterns) {
    for (const bool literal : {false, true}) {
      for (const bool case_sensitive : {false, true}) {
        const absl::StatusOr<std::unique_ptr<CompiledPattern>> compiled =
            CompilePattern({.pattern = pattern,
                            .literal = literal,
                            .case_sensitive = case_sensitive});
        if (!compiled.ok()) continue;
        for (const std::string& text : texts) {
          SCOPED_TRACE(
              absl::StrCat(pattern, "|", text, "|", literal, case_sensitive));
          EXPECT_EQ((*compiled)->MatchesWithin(text),
                    RE2::PartialMatch(text, (*compiled)->matcher()));
        }
      }
    }
  }
}

TEST(CompilePatternTest, ReportsTextAnchors) {
  const auto has_text_anchor = [](const std::string& pattern, bool literal) {
    absl::StatusOr<std::unique_ptr<CompiledPattern>> compiled =
        CompilePattern({.pattern = pattern, .literal = literal});
    EXPECT_TRUE(compiled.ok()) << pattern;
    return compiled.ok() && (*compiled)->has_text_anchor();
  };
  EXPECT_FALSE(has_text_anchor("^foo$", false));
  EXPECT_FALSE(has_text_anchor("\\bfoo\\b", false));
  EXPECT_TRUE(has_text_anchor("\\Afoo", false));
  EXPECT_TRUE(has_text_anchor("foo\\z", false));
  EXPECT_TRUE(has_text_anchor("(?-m:^foo)", false));
  EXPECT_FALSE(has_text_anchor("\\Afoo", true));
}

TEST(CompilePatternTest, NewMatcherIsEquivalent) {
  const std::unique_ptr<CompiledPattern> compiled =
      MustCompile({.pattern = "(?i)ab+c", .case_sensitive = true});
  const std::unique_ptr<RE2> copy = compiled->NewMatcher();
  ASSERT_TRUE(copy->ok());
  for (const char* text : {"ABBC", "ac", "xabcx", "a\nbc"}) {
    EXPECT_EQ(RE2::PartialMatch(text, *copy),
              RE2::PartialMatch(text, compiled->matcher()))
        << text;
  }
}

}  // namespace
}  // namespace ferret

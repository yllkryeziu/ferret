// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/internal/wildmatch.h"

#include <string>

#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

namespace ferret {
namespace {

using ::ferret::git_internal::Wildmatch;

struct WildmatchVector {
  absl::string_view text;
  absl::string_view pattern;
  bool pathname;           // wildmatch(pattern, text, WM_PATHNAME)
  bool pathname_casefold;  // ... WM_PATHNAME | WM_CASEFOLD
  bool plain;              // ... 0
  bool plain_casefold;     // ... WM_CASEFOLD
};

// Every single-pattern vector of git's t/t3070-wildmatch.sh (v2.49.0), in
// order. The four booleans are the expected results of test-tool wildmatch,
// iwildmatch, pathmatch, and ipathmatch.
constexpr WildmatchVector kGitVectors[] = {
    {"foo", "foo", true, true, true, true},
    {"foo", "bar", false, false, false, false},
    {"", "", true, true, true, true},
    {"foo", "???", true, true, true, true},
    {"foo", "??", false, false, false, false},
    {"foo", "*", true, true, true, true},
    {"foo", "f*", true, true, true, true},
    {"foo", "*f", false, false, false, false},
    {"foo", "*foo*", true, true, true, true},
    {"foobar", "*ob*a*r*", true, true, true, true},
    {"aaaaaaabababab", "*ab", true, true, true, true},
    {"foo*", R"(foo\*)", true, true, true, true},
    {"foobar", R"(foo\*bar)", false, false, false, false},
    {R"(f\oo)", R"(f\\oo)", true, true, true, true},
    {"ball", "*[al]?", true, true, true, true},
    {"ten", "[ten]", false, false, false, false},
    {"ten", "**[!te]", true, true, true, true},
    {"ten", "**[!ten]", false, false, false, false},
    {"ten", "t[a-g]n", true, true, true, true},
    {"ten", "t[!a-g]n", false, false, false, false},
    {"ton", "t[!a-g]n", true, true, true, true},
    {"ton", "t[^a-g]n", true, true, true, true},
    {"a]b", "a[]]b", true, true, true, true},
    {"a-b", "a[]-]b", true, true, true, true},
    {"a]b", "a[]-]b", true, true, true, true},
    {"aab", "a[]-]b", false, false, false, false},
    {"aab", "a[]a-]b", true, true, true, true},
    {"]", "]", true, true, true, true},
    {"foo/baz/bar", "foo*bar", false, false, true, true},
    {"foo/baz/bar", "foo**bar", false, false, true, true},
    {"foobazbar", "foo**bar", true, true, true, true},
    {"foo/baz/bar", "foo/**/bar", true, true, true, true},
    {"foo/baz/bar", "foo/**/**/bar", true, true, false, false},
    {"foo/b/a/z/bar", "foo/**/bar", true, true, true, true},
    {"foo/b/a/z/bar", "foo/**/**/bar", true, true, true, true},
    {"foo/bar", "foo/**/bar", true, true, false, false},
    {"foo/bar", "foo/**/**/bar", true, true, false, false},
    {"foo/bar", "foo?bar", false, false, true, true},
    {"foo/bar", "foo[/]bar", false, false, true, true},
    {"foo/bar", "foo[^a-z]bar", false, false, true, true},
    {"foo/bar", "f[^eiu][^eiu][^eiu][^eiu][^eiu]r", false, false, true, true},
    {"foo-bar", "f[^eiu][^eiu][^eiu][^eiu][^eiu]r", true, true, true, true},
    {"foo", "**/foo", true, true, false, false},
    {"XXX/foo", "**/foo", true, true, true, true},
    {"bar/baz/foo", "**/foo", true, true, true, true},
    {"bar/baz/foo", "*/foo", false, false, true, true},
    {"foo/bar/baz", "**/bar*", false, false, true, true},
    {"deep/foo/bar/baz", "**/bar/*", true, true, true, true},
    {"deep/foo/bar/baz/", "**/bar/*", false, false, true, true},
    {"deep/foo/bar/baz/", "**/bar/**", true, true, true, true},
    {"deep/foo/bar", "**/bar/*", false, false, false, false},
    {"deep/foo/bar/", "**/bar/**", true, true, true, true},
    {"foo/bar/baz", "**/bar**", false, false, true, true},
    {"foo/bar/baz/x", "*/bar/**", true, true, true, true},
    {"deep/foo/bar/baz/x", "*/bar/**", false, false, true, true},
    {"deep/foo/bar/baz/x", "**/bar/*/*", true, true, true, true},
    {"acrt", "a[c-c]st", false, false, false, false},
    {"acrt", "a[c-c]rt", true, true, true, true},
    {"]", "[!]-]", false, false, false, false},
    {"a", "[!]-]", true, true, true, true},
    {"", R"(\)", false, false, false, false},
    {R"(\)", R"(\)", false, false, false, false},
    {R"(XXX/\)", R"(*/\)", false, false, false, false},
    {R"(XXX/\)", R"(*/\\)", true, true, true, true},
    {"foo", "foo", true, true, true, true},
    {"@foo", "@foo", true, true, true, true},
    {"foo", "@foo", false, false, false, false},
    {"[ab]", R"(\[ab])", true, true, true, true},
    {"[ab]", "[[]ab]", true, true, true, true},
    {"[ab]", "[[:]ab]", true, true, true, true},
    {"[ab]", "[[::]ab]", false, false, false, false},
    {"[ab]", "[[:digit]ab]", true, true, true, true},
    {"[ab]", R"([\[:]ab])", true, true, true, true},
    {"?a?b", R"(\??\?b)", true, true, true, true},
    {"abc", R"(\a\b\c)", true, true, true, true},
    {"foo", "", false, false, false, false},
    {"foo/bar/baz/to", "**/t[o]", true, true, true, true},
    {"a1B", "[[:alpha:]][[:digit:]][[:upper:]]", true, true, true, true},
    {"a", "[[:digit:][:upper:][:space:]]", false, true, false, true},
    {"A", "[[:digit:][:upper:][:space:]]", true, true, true, true},
    {"1", "[[:digit:][:upper:][:space:]]", true, true, true, true},
    {"1", "[[:digit:][:upper:][:spaci:]]", false, false, false, false},
    {" ", "[[:digit:][:upper:][:space:]]", true, true, true, true},
    {".", "[[:digit:][:upper:][:space:]]", false, false, false, false},
    {".", "[[:digit:][:punct:][:space:]]", true, true, true, true},
    {"5", "[[:xdigit:]]", true, true, true, true},
    {"f", "[[:xdigit:]]", true, true, true, true},
    {"D", "[[:xdigit:]]", true, true, true, true},
    {"_",
     "[[:alnum:][:alpha:][:blank:][:cntrl:][:digit:][:graph:][:lower:][:print:]"
     "[:punct:][:space:][:upper:][:xdigit:]]",
     true, true, true, true},
    {".",
     "[^[:alnum:][:alpha:][:blank:][:cntrl:][:digit:][:lower:][:space:][:upper:"
     "][:xdigit:]]",
     true, true, true, true},
    {"5", "[a-c[:digit:]x-z]", true, true, true, true},
    {"b", "[a-c[:digit:]x-z]", true, true, true, true},
    {"y", "[a-c[:digit:]x-z]", true, true, true, true},
    {"q", "[a-c[:digit:]x-z]", false, false, false, false},
    {"]", R"([\\-^])", true, true, true, true},
    {"[", R"([\\-^])", false, false, false, false},
    {"-", R"([\-_])", true, true, true, true},
    {"]", R"([\]])", true, true, true, true},
    {R"(\])", R"([\]])", false, false, false, false},
    {R"(\)", R"([\]])", false, false, false, false},
    {"ab", "a[]b", false, false, false, false},
    {"a[]b", "a[]b", false, false, false, false},
    {"ab[", "ab[", false, false, false, false},
    {"ab", "[!", false, false, false, false},
    {"ab", "[-", false, false, false, false},
    {"-", "[-]", true, true, true, true},
    {"-", "[a-", false, false, false, false},
    {"-", "[!a-", false, false, false, false},
    {"-", "[--A]", true, true, true, true},
    {"5", "[--A]", true, true, true, true},
    {" ", "[ --]", true, true, true, true},
    {"$", "[ --]", true, true, true, true},
    {"-", "[ --]", true, true, true, true},
    {"0", "[ --]", false, false, false, false},
    {"-", "[---]", true, true, true, true},
    {"-", "[------]", true, true, true, true},
    {"j", "[a-e-n]", false, false, false, false},
    {"-", "[a-e-n]", true, true, true, true},
    {"a", "[!------]", true, true, true, true},
    {"[", "[]-a]", false, false, false, false},
    {"^", "[]-a]", true, true, true, true},
    {"^", "[!]-a]", false, false, false, false},
    {"[", "[!]-a]", true, true, true, true},
    {"^", "[a^bc]", true, true, true, true},
    {"-b]", "[a-]b]", true, true, true, true},
    {R"(\)", R"([\])", false, false, false, false},
    {R"(\)", R"([\\])", true, true, true, true},
    {R"(\)", R"([!\\])", false, false, false, false},
    {"G", R"([A-\\])", true, true, true, true},
    {"aaabbb", "b*a", false, false, false, false},
    {"aabcaa", "*ba*", false, false, false, false},
    {",", "[,]", true, true, true, true},
    {",", R"([\\,])", true, true, true, true},
    {R"(\)", R"([\\,])", true, true, true, true},
    {"-", "[,-.]", true, true, true, true},
    {"+", "[,-.]", false, false, false, false},
    {"-.]", "[,-.]", false, false, false, false},
    {"2", R"([\1-\3])", true, true, true, true},
    {"3", R"([\1-\3])", true, true, true, true},
    {"4", R"([\1-\3])", false, false, false, false},
    {R"(\)", R"([[-\]])", true, true, true, true},
    {"[", R"([[-\]])", true, true, true, true},
    {"]", R"([[-\]])", true, true, true, true},
    {"-", R"([[-\]])", false, false, false, false},
    {"-adobe-courier-bold-o-normal--12-120-75-75-m-70-iso8859-1",
     "-*-*-*-*-*-*-12-*-*-*-m-*-*-*", true, true, true, true},
    {"-adobe-courier-bold-o-normal--12-120-75-75-X-70-iso8859-1",
     "-*-*-*-*-*-*-12-*-*-*-m-*-*-*", false, false, false, false},
    {"-adobe-courier-bold-o-normal--12-120-75-75-/-70-iso8859-1",
     "-*-*-*-*-*-*-12-*-*-*-m-*-*-*", false, false, false, false},
    {"XXX/adobe/courier/bold/o/normal//12/120/75/75/m/70/iso8859/1",
     "XXX/*/*/*/*/*/*/12/*/*/*/m/*/*/*", true, true, true, true},
    {"XXX/adobe/courier/bold/o/normal//12/120/75/75/X/70/iso8859/1",
     "XXX/*/*/*/*/*/*/12/*/*/*/m/*/*/*", false, false, false, false},
    {"abcd/abcdefg/abcdefghijk/abcdefghijklmnop.txt", "**/*a*b*g*n*t", true,
     true, true, true},
    {"abcd/abcdefg/abcdefghijk/abcdefghijklmnop.txtz", "**/*a*b*g*n*t", false,
     false, false, false},
    {"foo", "*/*/*", false, false, false, false},
    {"foo/bar", "*/*/*", false, false, false, false},
    {"foo/bba/arr", "*/*/*", true, true, true, true},
    {"foo/bb/aa/rr", "*/*/*", false, false, true, true},
    {"foo/bb/aa/rr", "**/**/**", true, true, true, true},
    {"abcXdefXghi", "*X*i", true, true, true, true},
    {"ab/cXd/efXg/hi", "*X*i", false, false, true, true},
    {"ab/cXd/efXg/hi", "*/*X*/*/*i", true, true, true, true},
    {"ab/cXd/efXg/hi", "**/*X*/**/*i", true, true, true, true},
    {"foo", "fo", false, false, false, false},
    {"foo/bar", "foo/bar", true, true, true, true},
    {"foo/bar", "foo/*", true, true, true, true},
    {"foo/bba/arr", "foo/*", false, false, true, true},
    {"foo/bba/arr", "foo/**", true, true, true, true},
    {"foo/bba/arr", "foo*", false, false, true, true},
    {"foo/bba/arr", "foo**", false, false, true, true},
    {"foo/bba/arr", "foo/*arr", false, false, true, true},
    {"foo/bba/arr", "foo/**arr", false, false, true, true},
    {"foo/bba/arr", "foo/*z", false, false, false, false},
    {"foo/bba/arr", "foo/**z", false, false, false, false},
    {"foo/bar", "foo?bar", false, false, true, true},
    {"foo/bar", "foo[/]bar", false, false, true, true},
    {"foo/bar", "foo[^a-z]bar", false, false, true, true},
    {"ab/cXd/efXg/hi", "*Xg*i", false, false, true, true},
    {"a", "[A-Z]", false, true, false, true},
    {"A", "[A-Z]", true, true, true, true},
    {"A", "[a-z]", false, true, false, true},
    {"a", "[a-z]", true, true, true, true},
    {"a", "[[:upper:]]", false, true, false, true},
    {"A", "[[:upper:]]", true, true, true, true},
    {"A", "[[:lower:]]", false, true, false, true},
    {"a", "[[:lower:]]", true, true, true, true},
    {"A", "[B-Za]", false, true, false, true},
    {"a", "[B-Za]", true, true, true, true},
    {"A", "[B-a]", false, true, false, true},
    {"a", "[B-a]", true, true, true, true},
    {"z", "[Z-y]", false, true, false, true},
    {"Z", "[Z-y]", true, true, true, true},
};

TEST(WildmatchTest, MatchesGitTestSuite) {
  for (const WildmatchVector& v : kGitVectors) {
    SCOPED_TRACE(testing::Message() << "text \"" << v.text << "\" pattern \""
                                    << v.pattern << "\"");
    EXPECT_EQ(Wildmatch(v.pattern, v.text, {.pathname = true}), v.pathname);
    EXPECT_EQ(
        Wildmatch(v.pattern, v.text, {.pathname = true, .casefold = true}),
        v.pathname_casefold);
    EXPECT_EQ(Wildmatch(v.pattern, v.text, {}), v.plain);
    EXPECT_EQ(Wildmatch(v.pattern, v.text, {.casefold = true}),
              v.plain_casefold);
  }
}

TEST(WildmatchTest, CasefoldDoesNotApplyInsideBracketsOrEscapes) {
  // Git lowercases the text byte but not bracket members or escaped
  // characters, so these fail case-insensitively. .gitignore matching
  // inherits the quirk, and so must we.
  EXPECT_FALSE(Wildmatch("[A]", "A", {.casefold = true}));
  EXPECT_TRUE(Wildmatch("[A]", "A", {}));
  EXPECT_FALSE(Wildmatch(R"(\A)", "A", {.casefold = true}));
  EXPECT_TRUE(Wildmatch("[a]", "A", {.casefold = true}));
}

TEST(WildmatchTest, DoubleStarIsSpecialOnlyAsAWholeComponent) {
  EXPECT_TRUE(Wildmatch("a/**/b", "a/b", {.pathname = true}));
  EXPECT_TRUE(Wildmatch("a/**/b", "a/x/y/b", {.pathname = true}));
  EXPECT_FALSE(Wildmatch("a**/b", "ax/y/b", {.pathname = true}));
  EXPECT_TRUE(Wildmatch("a**/b", "ax/b", {.pathname = true}));
  EXPECT_TRUE(Wildmatch("**", "a/b/c", {.pathname = true}));
  EXPECT_FALSE(Wildmatch("*", "a/b", {.pathname = true}));
}

TEST(WildmatchTest, AdversarialStarsFinishQuickly) {
  // Without git's abort codes this is exponential in the number of stars.
  const std::string text(4000, 'a');
  std::string pattern;
  for (int i = 0; i < 400; ++i) pattern += "*a";
  pattern += "*b";
  EXPECT_FALSE(Wildmatch(pattern, text, {.pathname = true}));
  EXPECT_FALSE(Wildmatch(pattern, text, {}));
  EXPECT_TRUE(Wildmatch(pattern, text + "b", {}));
}

}  // namespace
}  // namespace ferret

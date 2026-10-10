// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/lexer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "ferret/testing/test_seed.h"
#include "ferret/text/chunker.h"
#include "ferret/text/language.h"

namespace ferret {
namespace {

absl::string_view Name(MatchClass match_class) {
  switch (match_class) {
    case MatchClass::kCodeIdentifier:
      return "kCodeIdentifier";
    case MatchClass::kPartialToken:
      return "kPartialToken";
    case MatchClass::kComment:
      return "kComment";
    case MatchClass::kString:
      return "kString";
  }
  return "invalid";
}

absl::string_view Name(LexState state) {
  switch (state) {
    case LexState::kCode:
      return "kCode";
    case LexState::kBlockComment:
      return "kBlockComment";
    case LexState::kLineComment:
      return "kLineComment";
    case LexState::kDoubleQuotedString:
      return "kDoubleQuotedString";
    case LexState::kSingleQuotedString:
      return "kSingleQuotedString";
    case LexState::kTripleDoubleQuotedString:
      return "kTripleDoubleQuotedString";
    case LexState::kTripleSingleQuotedString:
      return "kTripleSingleQuotedString";
    case LexState::kBacktickString:
      return "kBacktickString";
    case LexState::kTemplateSubstitution:
      return "kTemplateSubstitution";
    case LexState::kCppRawString:
      return "kCppRawString";
    case LexState::kHereDocument:
      return "kHereDocument";
    case LexState::kRustRawString0:
      return "kRustRawString0";
    case LexState::kRustRawString1:
      return "kRustRawString1";
    case LexState::kRustRawString2:
      return "kRustRawString2";
    case LexState::kRustRawString3:
      return "kRustRawString3";
  }
  return "invalid";
}

constexpr MatchClass kCode = MatchClass::kCodeIdentifier;
constexpr MatchClass kPartial = MatchClass::kPartialToken;
constexpr MatchClass kComment = MatchClass::kComment;
constexpr MatchClass kString = MatchClass::kString;

struct ClassifyCase {
  absl::string_view text;
  // Classified at its first occurrence in `text`, or at the one numbered
  // `occurrence` from 0.
  absl::string_view match;
  MatchClass expected;
  LexState entry = LexState::kCode;
  int occurrence = 0;
};

void ExpectClasses(Language language, const std::vector<ClassifyCase>& cases) {
  for (const ClassifyCase& c : cases) {
    size_t begin = c.text.find(c.match);
    for (int i = 0; i < c.occurrence && begin != absl::string_view::npos; ++i) {
      begin = c.text.find(c.match, begin + 1);
    }
    ASSERT_NE(begin, absl::string_view::npos) << c.match;
    EXPECT_EQ(Name(ClassifyMatch(language, c.text, c.entry, begin,
                                 begin + c.match.size())),
              Name(c.expected))
        << LanguageName(language) << ": \"" << c.match << "\" #" << c.occurrence
        << " in \"" << absl::CEscape(c.text) << "\" from " << Name(c.entry);
  }
}

TEST(LexStateTest, PersistedValuesAreStable) {
  EXPECT_EQ(static_cast<int>(LexState::kCode), 0);
  EXPECT_EQ(static_cast<int>(LexState::kBlockComment), 1);
  EXPECT_EQ(static_cast<int>(LexState::kLineComment), 2);
  EXPECT_EQ(static_cast<int>(LexState::kDoubleQuotedString), 3);
  EXPECT_EQ(static_cast<int>(LexState::kSingleQuotedString), 4);
  EXPECT_EQ(static_cast<int>(LexState::kTripleDoubleQuotedString), 5);
  EXPECT_EQ(static_cast<int>(LexState::kTripleSingleQuotedString), 6);
  EXPECT_EQ(static_cast<int>(LexState::kBacktickString), 7);
  EXPECT_EQ(static_cast<int>(LexState::kTemplateSubstitution), 8);
  EXPECT_EQ(static_cast<int>(LexState::kCppRawString), 9);
  EXPECT_EQ(static_cast<int>(LexState::kHereDocument), 10);
  EXPECT_EQ(static_cast<int>(LexState::kRustRawString0), 11);
  EXPECT_EQ(static_cast<int>(LexState::kRustRawString1), 12);
  EXPECT_EQ(static_cast<int>(LexState::kRustRawString2), 13);
  EXPECT_EQ(static_cast<int>(LexState::kRustRawString3), 14);
  // The index packs a state into four bits.
  EXPECT_LT(static_cast<int>(LexState::kRustRawString3), 16);
}

TEST(ClassifyMatchTest, IdentifierBoundaries) {
  ExpectClasses(Language::kC,
                {
                    {"int foo = 1;", "foo", kCode},
                    {"a.foo(b)", "foo", kCode},
                    {"foo", "foo", kCode},
                    {"int foobar;", "foo", kPartial},
                    {"int barfoo;", "foo", kPartial},
                    {"x = foo_1;", "foo", kPartial},
                    {"x = foo2;", "foo", kPartial},
                    {"x = 2foo;", "foo", kPartial},
                    // Non-ASCII bytes are identifier bytes.
                    {"int foo\xC3\xA9;", "foo", kPartial},
                    // A match may span tokens; only its ends matter.
                    {"a = foo(bar);", "foo(bar", kCode},
                    {"a = foo(bar);", "oo(ba", kPartial},
                    // The ends of the text are boundaries.
                    {"foo\nbar", "foo\nbar", kCode},
                    // '$' is an identifier byte only in Java and JavaScript.
                    {"x = $foo;", "foo", kCode},
                });
  ExpectClasses(Language::kJava, {{"int $foo = 1;", "foo", kPartial}});
  ExpectClasses(Language::kJavaScript, {{"let foo$ = 1;", "foo", kPartial}});
}

TEST(ClassifyMatchTest, C) {
  ExpectClasses(
      Language::kC,
      {
          {"x = 1; // foo", "foo", kComment},
          {"/* foo */ int x;", "foo", kComment},
          {"/* a */ int foo;", "foo", kCode},
          {"/* a\n * foo\n */", "foo", kComment},
          {"/*/ foo */", "foo", kComment},
          {"s = \"foo\";", "foo", kString},
          {"s = \"a\\\"foo\";", "foo", kString},
          {"s = \"a\\\\\"; foo", "foo", kCode},
          {"s = \"a\"; foo", "foo", kCode},
          {"c = 'f';", "f", kString},
          {"c = '\\''; foo", "foo", kCode},
          {"c = '\"'; foo", "foo", kCode},
          // An unterminated literal ends at the end of its line.
          {"s = \"abc\nfoo", "foo", kCode},
          // A backslash continues literals and // comments past a line.
          {"s = \"a\\\nfoo\";", "foo", kString},
          {"// a \\\nfoo", "foo", kComment},
          {"// a \\\r\nfoo", "foo", kComment},
          {"// a \\ b\nfoo", "foo", kCode},
          // C23 digit separators do not open character literals.
          {"n = 1'000; foo();", "foo", kCode},
          {"n = 0x1'f; foo();", "foo", kCode},
          {"c = L'x'; foo();", "foo", kCode},
          {"#include \"foo.h\"", "foo", kString},
          {"#define foo 1", "foo", kCode},
          {"x = a / foo;", "foo", kCode},
          {"s = \"// foo\";", "foo", kString},
          {"// \"foo\"", "foo", kComment},
      });
}

TEST(ClassifyMatchTest, Cpp) {
  const std::vector<ClassifyCase> cases = {
      {"auto s = R\"(foo)\";", "foo", kString},
      {"auto s = R\"x(a)\" foo )x\"; bar", "foo", kString},
      {"auto s = R\"x(a)\" foo )x\"; bar", "bar", kCode},
      {"auto s = R\"(a\nfoo\n)\"; bar", "foo", kString},
      {"auto s = R\"(a\nfoo\n)\"; bar", "bar", kCode},
      {"auto s = R\"(\\)\"; foo", "foo", kCode},
      {"auto s = u8R\"(foo)\";", "foo", kString},
      {"auto s = LR\"(a)\"; foo", "foo", kCode},
      {"auto s = R\"(a\" foo )\";", "foo", kString},
      // Not a raw string: the prefix is part of a longer identifier.
      {"auto s = FOOR\"(a\" foo )\";", "foo", kCode},
      {"int n = 1'000'000; foo();", "foo", kCode},
      {"template <int N> foo<'a'>", "foo", kCode},
  };
  ExpectClasses(Language::kCpp, cases);
  ExpectClasses(Language::kObjectiveCpp, cases);
  ExpectClasses(Language::kObjectiveCpp,
                {{"NSString* s = @\"foo\";", "foo", kString}});
}

TEST(ClassifyMatchTest, Python) {
  ExpectClasses(Language::kPython,
                {
                    {"x = 1  # foo", "foo", kComment},
                    {"# foo", "foo", kComment},
                    {"s = 'foo'", "foo", kString},
                    {"s = \"foo\"", "foo", kString},
                    {"s = \"\"\"a\nfoo\n\"\"\" + bar", "foo", kString},
                    {"s = \"\"\"a\nfoo\n\"\"\" + bar", "bar", kCode},
                    {"s = '''a \"\"\" foo''' + bar", "foo", kString},
                    {"s = '''a \"\"\" foo''' + bar", "bar", kCode},
                    {"s = \"\" + foo", "foo", kCode},
                    {"s = r'\\d' + foo", "foo", kCode},
                    {"s = f\"{foo}\"", "foo", kString},
                    {"s = 'a\\\nfoo'", "foo", kString},
                    {"s = 'a\\\r\nfoo'", "foo", kString},
                    {"s = 'abc\nfoo", "foo", kCode},
                    {"s = '#'; foo", "foo", kCode},
                    {"\"\"\"# foo\"\"\"", "foo", kString},
                    {"x = a // foo", "foo", kCode},
                });
}

TEST(ClassifyMatchTest, Go) {
  ExpectClasses(Language::kGo, {
                                   {"s := `a\nfoo`", "foo", kString},
                                   {"s := `a\\`; foo", "foo", kCode},
                                   {"s := \"`\"; foo", "foo", kCode},
                                   {"r := 'f'", "f", kString},
                                   {"r := '\\''; foo", "foo", kCode},
                                   {"x := 1 // foo", "foo", kComment},
                                   {"/* foo */", "foo", kComment},
                                   {"x := 1 # foo", "foo", kCode},
                               });
}

TEST(ClassifyMatchTest, Java) {
  ExpectClasses(Language::kJava,
                {
                    {"String s = \"\"\"\n    foo \"quoted\"\n    \"\"\"; bar",
                     "foo", kString},
                    {"String s = \"\"\"\n    foo \"quoted\"\n    \"\"\"; bar",
                     "bar", kCode},
                    {"/** foo */", "foo", kComment},
                    {"char c = '\"'; foo", "foo", kCode},
                    {"int n = 1_000; foo();", "foo", kCode},
                    {"s = \"\"; foo", "foo", kCode},
                });
}

TEST(ClassifyMatchTest, JavaScriptAndTypeScript) {
  const std::vector<ClassifyCase> cases = {
      {"const r = /foo/g;", "foo", kString},
      {"f(/foo/)", "foo", kString},
      {"x = /[/]foo/", "foo", kString},
      {"x = /a\\/foo/", "foo", kString},
      {"x = /a/; foo", "foo", kCode},
      {"x = a / foo / b", "foo", kCode},
      {"x = (a) / foo / b", "foo", kCode},
      {"x = \"a\" / foo / b", "foo", kCode},
      // A regular expression ends at the end of its line.
      {"x = /a\nfoo", "foo", kCode},
      {"const t = `a ${foo} b`;", "foo", kCode},
      {"const t = `a ${foo} b`;", "b", kString},
      {"const t = `a ${x} foo`;", "foo", kString},
      {"const t = `a ${x} b`; foo", "foo", kCode},
      {"`${ {a: 1}.foo }`", "foo", kCode},
      {"`${ {a: 1} } foo`", "foo", kString},
      {"`a ${ \"}\" } foo`", "foo", kString},
      {"`a ${ `foo` } b`", "foo", kString},
      {"`a ${ `b` } foo`", "foo", kString},
      {"`a ${ `b ${foo}` } c`", "foo", kCode},
      {"`a ${ /* } */ b } foo`", "foo", kString},
      {"`a\nfoo`", "foo", kString},
      {"`a\\` foo`", "foo", kString},
      {"`$foo`", "foo", kString},
      {"x = 1; // foo", "foo", kComment},
      {"x = '/*'; foo", "foo", kCode},
      {"let s = 'abc\nfoo", "foo", kCode},
  };
  ExpectClasses(Language::kJavaScript, cases);
  ExpectClasses(Language::kTypeScript, cases);
}

TEST(ClassifyMatchTest, Rust) {
  ExpectClasses(Language::kRust,
                {
                    {"let s = r#\"a \" foo\"#; bar", "foo", kString},
                    {"let s = r#\"a \" foo\"#; bar", "bar", kCode},
                    {"let s = r##\"a\"# foo\"##; bar", "foo", kString},
                    {"let s = r##\"a\"# foo\"##; bar", "bar", kCode},
                    {"let s = r\"\\\"; foo", "foo", kCode},
                    {"let s = br\"foo\";", "foo", kString},
                    {"let s = cr#\"foo\"#;", "foo", kString},
                    {"let s = \"a\nfoo\";", "foo", kString},
                    {"let s = \"a\\\"foo\";", "foo", kString},
                    {"/* a /* b */ foo */ bar", "foo", kComment},
                    {"/* a /* b */ foo */ bar", "bar", kCode},
                    {"fn f<'a>(x: &'a str) -> foo", "foo", kCode},
                    {"let x: &'static foo;", "foo", kCode},
                    {"'outer: loop { break 'outer; } foo", "foo", kCode},
                    {"let c = 'f';", "f", kString},
                    {"let c = '\\''; foo", "foo", kCode},
                    {"let c = '\\u{1F600}'; foo", "foo", kCode},
                    {"let c = '\xC3\xA9'; foo", "foo", kCode},
                    {"let b = b'x'; foo", "foo", kCode},
                    {"x = 1; // foo", "foo", kComment},
                    {"#[derive(foo)]", "foo", kCode},
                });
}

TEST(ClassifyMatchTest, Shell) {
  ExpectClasses(
      Language::kShell,
      {
          {"echo foo # bar", "bar", kComment},
          {"echo foo # bar", "foo", kCode},
          {"# foo", "foo", kComment},
          {"echo x;# foo", "foo", kComment},
          {"echo $#foo", "foo", kCode},
          {"echo a#foo", "foo", kCode},
          {"echo ${#foo}", "foo", kCode},
          {"echo 'foo'", "foo", kString},
          {"echo 'a\\' foo", "foo", kCode},
          {"echo 'a\nfoo'", "foo", kString},
          {"echo \"a\\\"foo\"", "foo", kString},
          {"echo \"a\nfoo\"", "foo", kString},
          {"echo \\\"foo", "foo", kCode},
          {"echo \\# foo", "foo", kCode},
          {"echo \"$(foo)\"", "foo", kString},
          {"cat <<EOF\nfoo\nEOF\nbar", "foo", kString},
          {"cat <<EOF\nfoo\nEOF\nbar", "bar", kCode},
          {"cat <<EOF\nfoo\nEOF\nbar", "EOF", kCode, LexState::kCode, 1},
          {"cat <<EOF\nEOF x\nfoo\nEOF\nbar", "foo", kString},
          {"cat <<EOF | grep foo\nbar\nEOF", "foo", kCode},
          {"cat <<EOF | grep x\nfoo\nEOF", "foo", kString},
          {"cat <<-'END'\n\tfoo\n\tEND\nbar", "foo", kString},
          {"cat <<-'END'\n\tfoo\n\tEND\nbar", "bar", kCode},
          {"cat << \"EOF\"\n'foo\nEOF\nbar", "foo", kString},
          {"cat << \"EOF\"\n'foo\nEOF\nbar", "bar", kCode},
          {"cat <<\\EOF\nfoo\nEOF\nbar", "bar", kCode},
          {"cat <<EOF\r\nfoo\r\nEOF\r\nbar", "bar", kCode},
          // A here-string and a shift are not here-documents.
          {"cat <<< foo\nbar", "bar", kCode},
          {"echo $((1<<2))\nfoo", "foo", kCode},
      });
}

TEST(ClassifyMatchTest, OtherLanguagesAreAllCode) {
  for (const Language language : {Language::kUnknown, Language::kMarkup}) {
    ExpectClasses(language, {
                                {"<!-- foo -->", "foo", kCode},
                                {"\"foo\"", "foo", kCode},
                                {"# foo", "foo", kCode},
                                {"// foo", "foo", kCode},
                                {"foobar", "foo", kPartial},
                            });
  }
}

TEST(ClassifyMatchTest, ResumesFromTheEntryState) {
  ExpectClasses(
      Language::kC,
      {
          {"foo */ bar", "foo", kComment, LexState::kBlockComment},
          {"foo */ bar", "bar", kCode, LexState::kBlockComment},
          {"foo \\\nbar\nbaz", "bar", kComment, LexState::kLineComment},
          {"foo \\\nbar\nbaz", "baz", kCode, LexState::kLineComment},
          {"foo\" bar", "foo", kString, LexState::kDoubleQuotedString},
          {"foo\" bar", "bar", kCode, LexState::kDoubleQuotedString},
          {"foo' bar", "bar", kCode, LexState::kSingleQuotedString},
      });
  ExpectClasses(
      Language::kCpp,
      {
          {"foo)delim\" bar", "foo", kString, LexState::kCppRawString},
          {"foo)delim\" bar", "bar", kCode, LexState::kCppRawString},
          // Without the delimiter, the first plausible terminator ends it.
          {"a)b\" foo )delim\"", "foo", kCode, LexState::kCppRawString},
          {"a) \" foo )delim\"", "foo", kString, LexState::kCppRawString},
      });
  ExpectClasses(
      Language::kPython,
      {
          {"foo\"\"\" bar", "foo", kString,
           LexState::kTripleDoubleQuotedString},
          {"foo\"\"\" bar", "bar", kCode, LexState::kTripleDoubleQuotedString},
          {"foo''' bar", "bar", kCode, LexState::kTripleSingleQuotedString},
      });
  ExpectClasses(Language::kGo,
                {
                    {"foo` bar", "foo", kString, LexState::kBacktickString},
                    {"foo` bar", "bar", kCode, LexState::kBacktickString},
                });
  ExpectClasses(
      Language::kJavaScript,
      {
          {"foo ${bar} baz` qux", "foo", kString, LexState::kBacktickString},
          {"foo ${bar} baz` qux", "bar", kCode, LexState::kBacktickString},
          {"foo ${bar} baz` qux", "baz", kString, LexState::kBacktickString},
          {"foo ${bar} baz` qux", "qux", kCode, LexState::kBacktickString},
          {"foo } bar` baz", "foo", kCode, LexState::kTemplateSubstitution},
          {"foo } bar` baz", "bar", kString, LexState::kTemplateSubstitution},
          {"foo } bar` baz", "baz", kCode, LexState::kTemplateSubstitution},
          {"{a} } foo` bar", "foo", kString, LexState::kTemplateSubstitution},
      });
  ExpectClasses(
      Language::kRust,
      {
          {"foo\"# x\"## bar", "x", kString, LexState::kRustRawString2},
          {"foo\"# x\"## bar", "bar", kCode, LexState::kRustRawString2},
          {"foo\" bar", "bar", kCode, LexState::kRustRawString0},
          {"foo\"# bar", "bar", kCode, LexState::kRustRawString1},
          {"foo\"### bar", "bar", kCode, LexState::kRustRawString3},
          {"foo */ bar", "bar", kCode, LexState::kBlockComment},
      });
  ExpectClasses(
      Language::kShell,
      {
          {"foo bar\nEOF\nbaz", "foo", kString, LexState::kHereDocument},
          {"foo bar\nEOF\nbaz", "EOF", kCode, LexState::kHereDocument},
          {"foo bar\nEOF\nbaz", "baz", kCode, LexState::kHereDocument},
          // Without the word, the first line of one word ends the body.
          {"foo bar\n\tdone\nbaz", "done", kCode, LexState::kHereDocument},
          {"foo\" bar", "bar", kCode, LexState::kDoubleQuotedString},
          {"foo' bar", "foo", kString, LexState::kSingleQuotedString},
          {"foo' bar", "bar", kCode, LexState::kSingleQuotedString},
      });
  // A state from a newer index is read as code.
  ExpectClasses(Language::kC,
                {{"foo */", "foo", kCode, static_cast<LexState>(200)}});
}

TEST(ClassifyMatchTest, EmptyMatchAtTheEndTakesTheExitState) {
  EXPECT_EQ(ClassifyMatch(Language::kC, "/* abc", LexState::kCode, 6, 6),
            kComment);
  EXPECT_EQ(ClassifyMatch(Language::kC, "abc", LexState::kCode, 3, 3), kCode);
  EXPECT_EQ(ClassifyMatch(Language::kC, "", LexState::kBlockComment, 0, 0),
            kComment);
  EXPECT_EQ(ClassifyMatch(Language::kRust, "", LexState::kRustRawString1, 0, 0),
            kString);
  EXPECT_EQ(ClassifyMatch(Language::kJavaScript, "`${", LexState::kCode, 3, 3),
            kCode);
}

struct AdvanceCase {
  Language language;
  absl::string_view text;
  LexState expected;
  LexState entry = LexState::kCode;
};

TEST(AdvanceLexStateTest, ReturnsTheConstructOpenAtTheEnd) {
  const AdvanceCase cases[] = {
      {Language::kC, "", LexState::kCode},
      {Language::kC, "int x;\n", LexState::kCode},
      {Language::kC, "/* a\n", LexState::kBlockComment},
      {Language::kC, "/* a */\n", LexState::kCode},
      {Language::kC, "// a\n", LexState::kCode},
      {Language::kC, "// a \\\n", LexState::kLineComment},
      {Language::kC, "// a \\\r\n", LexState::kLineComment},
      {Language::kC, "\"a\\\n", LexState::kDoubleQuotedString},
      {Language::kC, "'a\\\n", LexState::kSingleQuotedString},
      {Language::kC, "\"a\n", LexState::kCode},
      {Language::kC, "R\"(a\n", LexState::kCode},
      {Language::kCpp, "R\"x(a\n", LexState::kCppRawString},
      {Language::kCpp, "R\"x(a)\"\n", LexState::kCppRawString},
      {Language::kCpp, "R\"x(a)x\"\n", LexState::kCode},
      {Language::kObjectiveCpp, "/* a\n", LexState::kBlockComment},
      {Language::kPython, "\"\"\"a\n", LexState::kTripleDoubleQuotedString},
      {Language::kPython, "'''a\n", LexState::kTripleSingleQuotedString},
      {Language::kPython, "'a\\\n", LexState::kSingleQuotedString},
      {Language::kPython, "/* a\n", LexState::kCode},
      {Language::kGo, "`a\n", LexState::kBacktickString},
      {Language::kGo, "\"a\n", LexState::kCode},
      {Language::kJava, "s = \"\"\"\n", LexState::kTripleDoubleQuotedString},
      {Language::kJava, "`a\n", LexState::kCode},
      {Language::kJavaScript, "`a\n", LexState::kBacktickString},
      {Language::kJavaScript, "`a ${b\n", LexState::kTemplateSubstitution},
      {Language::kJavaScript, "`a ${ {b\n", LexState::kTemplateSubstitution},
      {Language::kJavaScript, "`a ${b} c` + `${\n",
       LexState::kTemplateSubstitution},
      {Language::kJavaScript, "`a ${b}\n", LexState::kBacktickString},
      {Language::kJavaScript, "`${ `\n", LexState::kBacktickString},
      {Language::kJavaScript, "x = /a\n", LexState::kCode},
      {Language::kTypeScript, "/* a\n", LexState::kBlockComment},
      {Language::kRust, "\"a\n", LexState::kDoubleQuotedString},
      {Language::kRust, "r\"a\n", LexState::kRustRawString0},
      {Language::kRust, "r#\"a\n", LexState::kRustRawString1},
      {Language::kRust, "br##\"a\n", LexState::kRustRawString2},
      {Language::kRust, "r###\"a\n", LexState::kRustRawString3},
      {Language::kRust, "r####\"a\n", LexState::kRustRawString3},
      {Language::kRust, "r#\"a\"\n", LexState::kRustRawString1},
      {Language::kRust, "/* /* */\n", LexState::kBlockComment},
      {Language::kRust, "/* /* */ */\n", LexState::kCode},
      {Language::kRust, "'a\n", LexState::kCode},
      {Language::kShell, "cat <<EOF\n", LexState::kHereDocument},
      {Language::kShell, "cat <<EOF\nbody\n", LexState::kHereDocument},
      {Language::kShell, "cat <<EOF\nbody\nEOF\n", LexState::kCode},
      {Language::kShell, "cat <<EOF\nbody\nEOF", LexState::kCode},
      {Language::kShell, "echo 'a\n", LexState::kSingleQuotedString},
      {Language::kShell, "echo \"a\n", LexState::kDoubleQuotedString},
      {Language::kShell, "echo \\'a\n", LexState::kCode},
      {Language::kMarkup, "/* \"a\n", LexState::kCode},
      {Language::kUnknown, "`a\n", LexState::kCode},
      // Entry states carry through text that does not close them.
      {Language::kC, "", LexState::kBlockComment, LexState::kBlockComment},
      {Language::kC, "abc\n", LexState::kBlockComment, LexState::kBlockComment},
      {Language::kC, "a */\n", LexState::kCode, LexState::kBlockComment},
      {Language::kPython, "x\"\"\" \"\"\"\n",
       LexState::kTripleDoubleQuotedString,
       LexState::kTripleDoubleQuotedString},
      {Language::kJavaScript, "a\n", LexState::kTemplateSubstitution,
       LexState::kTemplateSubstitution},
      {Language::kJavaScript, "a}\n", LexState::kBacktickString,
       LexState::kTemplateSubstitution},
      {Language::kJavaScript, "a}`\n", LexState::kCode,
       LexState::kTemplateSubstitution},
      {Language::kShell, "a b\n", LexState::kHereDocument,
       LexState::kHereDocument},
      {Language::kShell, "a b\nEND\n", LexState::kCode,
       LexState::kHereDocument},
      {Language::kC, "", LexState::kCode, static_cast<LexState>(200)},
  };
  for (const AdvanceCase& c : cases) {
    EXPECT_EQ(Name(AdvanceLexState(c.language, c.text, c.entry)),
              Name(c.expected))
        << LanguageName(c.language) << ": \"" << absl::CEscape(c.text)
        << "\" from " << Name(c.entry);
  }
}

// The class of every byte of `text` as a one-byte match: 'c' for code, 'p'
// for a partial token, '/' for a comment, and 's' for a string.
std::string ByteClasses(Language language, absl::string_view text,
                        LexState entry) {
  std::string classes;
  for (size_t i = 0; i < text.size(); ++i) {
    switch (ClassifyMatch(language, text, entry, i, i + 1)) {
      case MatchClass::kCodeIdentifier:
        classes += 'c';
        break;
      case MatchClass::kPartialToken:
        classes += 'p';
        break;
      case MatchClass::kComment:
        classes += '/';
        break;
      case MatchClass::kString:
        classes += 's';
        break;
    }
  }
  return classes;
}

// Interleaves each line of `text` with the classes of its bytes, so that a
// test failure shows where they differ.
std::string Annotate(absl::string_view text, absl::string_view classes) {
  std::string annotated;
  size_t begin = 0;
  while (begin < text.size()) {
    size_t end = text.find('\n', begin);
    end = end == absl::string_view::npos ? text.size() : end + 1;
    absl::StrAppend(&annotated, "\n  ",
                    absl::CEscape(text.substr(begin, end - begin)), "\n  ",
                    classes.substr(begin, end - begin));
    begin = end;
  }
  return annotated;
}

// Checks that lexing `text` in chunks, each from the exit state of the
// chunks before it, classifies every byte as lexing it whole does.
void ExpectChunkingPreservesClasses(Language language, absl::string_view text) {
  const std::string whole = ByteClasses(language, text, LexState::kCode);
  const LexState whole_exit = AdvanceLexState(language, text, LexState::kCode);
  for (const int64_t chunk_bytes : {1, 24, 80}) {
    std::string chunked;
    LexState state = LexState::kCode;
    for (const ChunkSpan& span : SplitIntoChunks(text, chunk_bytes)) {
      const absl::string_view chunk = text.substr(
          static_cast<size_t>(span.offset), static_cast<size_t>(span.size));
      chunked += ByteClasses(language, chunk, state);
      state = AdvanceLexState(language, chunk, state);
    }
    EXPECT_EQ(Annotate(text, chunked), Annotate(text, whole))
        << LanguageName(language) << " in chunks of " << chunk_bytes;
    EXPECT_EQ(Name(state), Name(whole_exit))
        << LanguageName(language) << " in chunks of " << chunk_bytes;
  }
}

TEST(ChunkContinuityTest, MultiLineConstructs) {
  ExpectChunkingPreservesClasses(Language::kC, R"c(/* A block comment
   spanning lines */
int foo = 1; // trailing \
   spliced comment
const char* s = "string \
continued";
char c = '\'';
int n = 1'000'000;
#define STR "x" /* one */ /* two
*/
)c");
  ExpectChunkingPreservesClasses(Language::kCpp, R"cc(auto raw = R"delim(
  raw ) "string" (
)delim";
                                                      auto empty = R"(a
b)";
                                                      int after;
                                                      std::string s =
                                                          u8"text"; /* c
                                                                     */
                                                      int x;
  )cc");
  ExpectChunkingPreservesClasses(Language::kPython, R"py(def f(x):
    """Docstring with 'quotes'
    and # not a comment
    """
    s = 'a\
b'  # comment
    t = '''x
''' + "y"
    return x
)py");
  ExpectChunkingPreservesClasses(Language::kGo, R"go(s := `raw
string "with" quotes`
r := '`'
/* block
comment */ x := "a\"b"
)go");
  ExpectChunkingPreservesClasses(Language::kJava, R"java(String block = """
    text "block"
    """;
/** Javadoc
 * comment */
char c = '"';
int $x = 1;
)java");
  for (const Language language :
       {Language::kJavaScript, Language::kTypeScript}) {
    ExpectChunkingPreservesClasses(language, R"js(const t = `line one
${value + `inner`} line two
done`;
const u = `${
  x + {a: 1}.a
}`;
const r = /[/"]+/g;
const d = a / b / c;
/* multi
   line */ const s = "x";
)js");
  }
  ExpectChunkingPreservesClasses(Language::kRust, R"rs(let raw = r#"a "quoted"
string"#;
let s = "multi
line";
/* outer /* nested */ still */ let c = 'x';
fn f<'a>(x: &'a str) {}
let b = br##"x"#
"##;
)rs");
  ExpectChunkingPreservesClasses(Language::kShell, R"sh(echo "multi
line" # comment
echo 'single
quoted'
cat <<EOF
body line one
body line two
EOF
echo $# \# done
)sh");
  ExpectChunkingPreservesClasses(Language::kC,
                                 "/* crlf\r\n */ x; // a \\\r\n b\r\n");
}

// Random sequences of tokens, including unbalanced delimiters, for each
// language. The vocabularies leave out the constructs that lexer.h lists as
// resuming inexactly: C++ raw strings, Rust nested comments (only whole
// single-line comments appear), shell here-documents, JavaScript template
// substitutions, and a JavaScript `/` that could start a line.
struct Vocabulary {
  Language language;
  std::vector<absl::string_view> tokens;
};

std::vector<Vocabulary> Vocabularies() {
  const std::vector<absl::string_view> common = {
      "foo", "x1", "42", "(", ")", "{", "}", ";", "=", "+", ",", ".", "a_b"};
  std::vector<Vocabulary> vocabularies = {
      {Language::kC,
       {"/*", "*/", "//", "\"", "'", "\\", "\\\"", "\\'", "1'000", "'a'",
        "\"s\"", "#include", "#", "u8'", "L\"", "0x1'f", "a / b"}},
      {Language::kCpp,
       {"/*", "*/", "//", "\"", "'", "\\", "\\\"", "1'000", "'a'", "R", "\"s\"",
        "#", "a / b"}},
      {Language::kObjectiveCpp,
       {"/*", "*/", "//", "\"", "'", "\\", "@\"", "@", "'a'"}},
      {Language::kPython,
       {"#", "\"", "'", "\"\"\"", "'''", "\\", "r'", "f\"", "b'", "\\\"", "\\'",
        "\"\"", "''", "a // b"}},
      {Language::kGo,
       {"//", "/*", "*/", "\"", "'", "`", "\\", "\\\"", "'\\''", "\\`"}},
      {Language::kJava,
       {"//", "/*", "*/", "\"", "'", "\"\"\"", "\\", "\\\"", "$x", "a$", "\"\"",
        "'\"'"}},
      {Language::kJavaScript,
       {"//", "/*", "*/", "\"", "'", "`", "\\", "\\`", "= /re[/]x/g",
        "x = /a\\/b/", "a / b", "*/ c", "/* d"}},
      {Language::kTypeScript,
       {"//", "/*", "*/", "\"", "'", "`", "\\", "\\\"", "(/[\"]/)", "a / b"}},
      {Language::kRust,
       {"//",    "/* c */", "\"",       "'",          "'a'",  "'a",
        "'\\''", "'\\n'",   "r\"",      "r#\"",       "\"#",  "r##\"",
        "\"##",  "r###\"",  "\"###",    "br\"",       "b'x'", "\\",
        "\\\"",  "'static", "\xC3\xA9", "'\xC3\xA9'", "#[x]"}},
      {Language::kShell,
       {"#", "a#b", "$#", "'", "\"", "\\", "\\\"", "\\'", "\\#", "`", "$(",
        "${x}", "${#x}", "<", ">", "|", "&&", "echo"}},
      {Language::kMarkup, {"<!--", "-->", "\"", "'", "#", "//"}},
  };
  for (Vocabulary& vocabulary : vocabularies) {
    vocabulary.tokens.insert(vocabulary.tokens.end(), common.begin(),
                             common.end());
  }
  return vocabularies;
}

size_t Draw(std::mt19937_64& rng, size_t bound) {
  return static_cast<size_t>(rng() % bound);
}

TEST(ChunkContinuityTest, RandomTokenSequences) {
  std::mt19937_64 rng(20260923);
  for (const Vocabulary& vocabulary : Vocabularies()) {
    for (int file = 0; file < 60; ++file) {
      std::string text;
      const size_t lines = 1 + Draw(rng, 14);
      for (size_t line = 0; line < lines; ++line) {
        const size_t tokens = Draw(rng, 9);
        for (size_t token = 0; token < tokens; ++token) {
          if (token > 0) text += ' ';
          text += vocabulary.tokens[Draw(rng, vocabulary.tokens.size())];
        }
        text += Draw(rng, 8) == 0 ? "\r\n" : "\n";
      }
      ExpectChunkingPreservesClasses(vocabulary.language, text);
      if (::testing::Test::HasFailure()) return;
    }
  }
}

// One classifier used for many sorted matches agrees with a fresh one per
// match (ClassifyMatch, whose results the tests above pin), so reusing its
// scanner state never changes a result.
TEST(MatchClassifierTest, RandomSortedMatchesAgreeWithRestarting) {
  const uint64_t seed = ferret_testing::TestSeed(20261003);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  std::vector<Vocabulary> vocabularies = Vocabularies();
  vocabularies.push_back({Language::kUnknown, {"foo", "/*", "\n", "'", "#"}});
  for (const Vocabulary& vocabulary : vocabularies) {
    for (int sample = 0; sample < 80; ++sample) {
      std::string text;
      const size_t count = Draw(rng, 80);
      for (size_t i = 0; i < count; ++i) {
        text += vocabulary.tokens[Draw(rng, vocabulary.tokens.size())];
        if (Draw(rng, 4) == 0) text += '\n';
      }
      // Include both valid open states and unknown states from newer indexes.
      const LexState entry = static_cast<LexState>(Draw(rng, 18));
      SCOPED_TRACE(absl::StrCat(LanguageName(vocabulary.language), " sample=",
                                sample, " entry=", static_cast<int>(entry)));
      std::vector<std::pair<size_t, size_t>> matches = {
          {0, 0},
          {text.size(), text.size()},
          {text.size(), text.size()},
          {0, 0}};
      for (int i = 0; i < 80; ++i) {
        const size_t begin = Draw(rng, text.size() + 1);
        const size_t end = begin + Draw(rng, text.size() - begin + 1);
        matches.emplace_back(begin, end);
      }
      std::stable_sort(matches.begin(), matches.end());
      MatchClassifier classifier(vocabulary.language, text, entry);
      for (const auto& [begin, end] : matches) {
        EXPECT_EQ(classifier.Classify(begin, end),
                  ClassifyMatch(vocabulary.language, text, entry, begin, end))
            << "match [" << begin << ", " << end << ") in "
            << absl::CEscape(text);
      }
    }
  }
}

// The code-token index relies on this: a match made of identifier bytes is
// kCodeIdentifier exactly when it is a maximal run that ForEachCodeIdentifier
// reports, so a document holds a code-identifier match of a query only if the
// query is one of its runs.
TEST(ForEachCodeIdentifierTest, ReportsExactlyTheCodeIdentifierMatches) {
  const uint64_t seed = ferret_testing::TestSeed(20261004);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  std::vector<Vocabulary> vocabularies = Vocabularies();
  vocabularies.push_back({Language::kUnknown, {"foo", "/*", "\n", "'", "#"}});
  auto is_plain_identifier_byte = [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_' ||
           static_cast<uint8_t>(c) >= 0x80;
  };
  int64_t runs_checked = 0;
  for (const Vocabulary& vocabulary : vocabularies) {
    for (int sample = 0; sample < 60; ++sample) {
      std::string text;
      const size_t count = Draw(rng, 60);
      for (size_t i = 0; i < count; ++i) {
        text += vocabulary.tokens[Draw(rng, vocabulary.tokens.size())];
        text += Draw(rng, 3) == 0 ? "\n" : " ";
      }
      const LexState entry = static_cast<LexState>(Draw(rng, 18));
      SCOPED_TRACE(absl::StrCat(LanguageName(vocabulary.language),
                                " entry=", static_cast<int>(entry), " ",
                                absl::CEscape(text)));
      std::set<std::pair<size_t, size_t>> reported;
      size_t previous_end = 0;
      const LexState exit = ForEachCodeIdentifier(
          vocabulary.language, text, entry, [&](size_t begin, size_t end) {
            EXPECT_LT(begin, end);
            EXPECT_GE(begin, previous_end);
            previous_end = end;
            // Runs of plain identifier bytes only: the others, with '$', are
            // not matches of any query.
            if (std::all_of(text.begin() + static_cast<ptrdiff_t>(begin),
                            text.begin() + static_cast<ptrdiff_t>(end),
                            is_plain_identifier_byte)) {
              reported.emplace(begin, end);
            }
          });
      EXPECT_EQ(exit, AdvanceLexState(vocabulary.language, text, entry));
      std::set<std::pair<size_t, size_t>> expected;
      MatchClassifier classifier(vocabulary.language, text, entry);
      for (size_t begin = 0; begin < text.size(); ++begin) {
        for (size_t end = begin + 1;
             end <= text.size() && is_plain_identifier_byte(text[end - 1]);
             ++end) {
          if (!is_plain_identifier_byte(text[begin])) break;
          if (classifier.Classify(begin, end) == MatchClass::kCodeIdentifier) {
            expected.emplace(begin, end);
          }
        }
      }
      EXPECT_EQ(reported, expected);
      runs_checked += std::ssize(expected);
    }
  }
  EXPECT_GT(runs_checked, 1000);
}

}  // namespace
}  // namespace ferret

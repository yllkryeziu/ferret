// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/search/formatter.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/search/search_types.h"
#include "ferret/testing/test_seed.h"
#include "ferret/text/utf8.h"

namespace ferret {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

ResultLine Match(int64_t number, std::string text, size_t begin, size_t end) {
  return ResultLine{.number = number,
                    .text = std::move(text),
                    .is_match = true,
                    .match_begin = begin,
                    .match_end = end};
}

ResultLine Context(int64_t number, std::string text) {
  return ResultLine{.number = number, .text = std::move(text)};
}

// Seven matching lines in three files: src/b.cc keeps two of its four, a.h
// its one, and the third file keeps none.
SearchResponse Sample() {
  SearchResponse response;
  response.num_matching_files = 3;
  response.num_matching_lines = 7;
  response.files.push_back(
      FileResult{.path = "src/b.cc",
                 .num_matching_lines = 4,
                 .lines = {Context(1, "int x;"), Match(2, "foo();", 0, 3),
                           Context(3, "}"), Match(9, "  foo(1);", 2, 5)}});
  response.files.push_back(
      FileResult{.path = "a.h",
                 .num_matching_lines = 1,
                 .lines = {Match(5, "void foo();", 5, 8)}});
  return response;
}

std::string Format(const SearchResponse& response, OutputShape shape,
                   size_t max_bytes = 8192) {
  return FormatSearchResponse(response,
                              {.shape = shape, .max_bytes = max_bytes});
}

TEST(FormatterTest, Context) {
  EXPECT_EQ(Format(Sample(), OutputShape::kContext),
            "7 matching lines in 3 files\n"
            "\n"
            "src/b.cc\n"
            "1-int x;\n"
            "2:foo();\n"
            "3-}\n"
            "--\n"
            "9:  foo(1);\n"
            "+2 more matching lines\n"
            "\n"
            "a.h\n"
            "5:void foo();\n"
            "[2 more matching lines in 1 file not shown; narrow the search "
            "with a path glob or a more specific pattern]\n");
}

TEST(FormatterTest, Line) {
  EXPECT_EQ(Format(Sample(), OutputShape::kLine),
            "src/b.cc:2:foo();\n"
            "src/b.cc:9:  foo(1);\n"
            "a.h:5:void foo();\n"
            "[4 more matching lines in 2 files not shown; narrow the search "
            "with a path glob or a more specific pattern]\n");
}

TEST(FormatterTest, RawIsInPathOrder) {
  EXPECT_EQ(Format(Sample(), OutputShape::kRaw),
            "a.h:5:void foo();\n"
            "src/b.cc:2:foo();\n"
            "src/b.cc:9:  foo(1);\n"
            "[truncated: 4 more lines in 2 files]\n");
}

TEST(FormatterTest, Locations) {
  EXPECT_EQ(Format(Sample(), OutputShape::kLocations),
            "src/b.cc:2\n"
            "src/b.cc:9\n"
            "a.h:5\n"
            "[4 more matching lines in 2 files not shown; narrow the search "
            "with a path glob or a more specific pattern]\n");
}

TEST(FormatterTest, NoTrailerWhenEverythingIsShown) {
  SearchResponse response;
  response.num_matching_files = 1;
  response.num_matching_lines = 1;
  response.files.push_back(FileResult{.path = "a.h",
                                      .num_matching_lines = 1,
                                      .lines = {Match(5, "foo", 0, 3)}});
  EXPECT_EQ(Format(response, OutputShape::kContext),
            "1 matching line in 1 file\n\na.h\n5:foo\n");
  EXPECT_EQ(Format(response, OutputShape::kRaw), "a.h:5:foo\n");
}

TEST(FormatterTest, Empty) {
  const SearchResponse empty;
  EXPECT_EQ(Format(empty, OutputShape::kContext),
            "0 matching lines in 0 files\n");
  EXPECT_EQ(Format(empty, OutputShape::kRaw), "");
  EXPECT_EQ(Format(empty, OutputShape::kLine), "");
  EXPECT_EQ(Format(empty, OutputShape::kLocations), "");
}

TEST(FormatterTest, StopsAtLineBoundaryWithinBudget) {
  // Room for the first line only.
  EXPECT_EQ(Format(Sample(), OutputShape::kLine, /*max_bytes=*/20),
            "src/b.cc:2:foo();\n"
            "[6 more matching lines in 3 files not shown; narrow the search "
            "with a path glob or a more specific pattern]\n");
}

TEST(FormatterTest, NothingFollowsTheFirstLineThatDoesNotFit) {
  // "src/b.cc:9:  foo(1);" does not fit after the first line, but the
  // shorter "a.h:5:void foo();" would; it must not skip ahead.
  EXPECT_EQ(Format(Sample(), OutputShape::kLine, /*max_bytes=*/36),
            "src/b.cc:2:foo();\n"
            "[6 more matching lines in 3 files not shown; narrow the search "
            "with a path glob or a more specific pattern]\n");
}

TEST(FormatterTest, NeverPrintsAPathAlone) {
  // The header fits, and the blank line and path would too, but not with
  // the file's first line.
  const std::string header = "7 matching lines in 3 files\n";
  const std::string out =
      Format(Sample(), OutputShape::kContext, header.size() + 1 + 9 + 3);
  EXPECT_EQ(out, header +
                     "[7 more matching lines in 3 files not shown; narrow the "
                     "search with a path glob or a more specific pattern]\n");
}

TEST(FormatterTest, Incomplete) {
  SearchResponse response = Sample();
  response.complete = false;
  response.verified_fraction = 0.25;
  EXPECT_THAT(Format(response, OutputShape::kContext),
              HasSubstr("7 matching lines in 3 files (incomplete: verified "
                        "25% of candidate chunks)\n"));
  EXPECT_THAT(Format(response, OutputShape::kLine),
              HasSubstr("[incomplete: the search verified 25% of candidate "
                        "chunks; counts cover only those]\n"));
}

TEST(FormatterTest, SanitizesAndCutsLines) {
  SearchResponse response;
  response.num_matching_files = 1;
  response.num_matching_lines = 1;
  std::string long_line(1000, 'x');
  long_line.replace(500, 3, "foo");
  response.files.push_back(
      FileResult{.path = "bad\xff.cc",
                 .num_matching_lines = 1,
                 .lines = {Match(1, long_line, 500, 503)}});
  const std::string out = FormatSearchResponse(
      response, {.shape = OutputShape::kLine, .max_line_bytes = 64});
  EXPECT_TRUE(IsValidUtf8(out));
  EXPECT_THAT(out, HasSubstr("bad\xEF\xBF\xBD.cc:1:\xE2\x80\xA6"));
  EXPECT_THAT(out, HasSubstr("foo"));
  const std::string prefix = "bad\xEF\xBF\xBD.cc:1:";
  EXPECT_LE(out.size(), prefix.size() + 64 + 1);
}

TEST(FormatterTest, ParsesShapeNames) {
  for (const OutputShape shape :
       {OutputShape::kRaw, OutputShape::kLine, OutputShape::kContext,
        OutputShape::kLocations}) {
    EXPECT_THAT(ParseOutputShape(OutputShapeName(shape)), IsOkAndHolds(shape));
  }
  EXPECT_THAT(ParseOutputShape("json"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

std::string RandomBytes(std::mt19937_64& rng, size_t max_size) {
  static constexpr char kAlphabet[] = "ab \t\r\xc3\xa9\xe2\x80\xa6\xff\xf0";
  std::string out(rng() % (max_size + 1), '\0');
  for (char& c : out) c = kAlphabet[rng() % (sizeof(kAlphabet) - 1)];
  return out;
}

// On arbitrary bytes, every shape yields valid UTF-8 that stays within the
// budget before the trailer.
TEST(FormatterTest, RandomResponsesAreValidAndBudgeted) {
  std::mt19937_64 rng(ferret_testing::TestSeed(53));
  for (int iteration = 0; iteration < 500; ++iteration) {
    SearchResponse response;
    const int num_files = static_cast<int>(rng() % 5);
    for (int f = 0; f < num_files; ++f) {
      FileResult file{.path = RandomBytes(rng, 12)};
      int64_t number = 0;
      for (int l = 0; l < static_cast<int>(rng() % 6); ++l) {
        number += 1 + static_cast<int64_t>(rng() % 3);
        std::string text = RandomBytes(rng, 400);
        const size_t begin = rng() % (text.size() + 1);
        const size_t end = begin + rng() % (text.size() - begin + 1);
        ResultLine line = rng() % 2 == 0 ? Match(number, text, begin, end)
                                         : Context(number, text);
        file.num_matching_lines += line.is_match;
        file.lines.push_back(std::move(line));
      }
      file.num_matching_lines += static_cast<int64_t>(rng() % 3);
      response.num_matching_lines += file.num_matching_lines;
      response.num_matching_files += file.num_matching_lines > 0;
      response.files.push_back(std::move(file));
    }
    response.num_matching_files += static_cast<int64_t>(rng() % 2);
    response.num_matching_lines += static_cast<int64_t>(rng() % 2);
    const size_t max_bytes = rng() % 2000;
    for (const OutputShape shape :
         {OutputShape::kRaw, OutputShape::kLine, OutputShape::kContext,
          OutputShape::kLocations}) {
      const std::string out = Format(response, shape, max_bytes);
      ASSERT_TRUE(IsValidUtf8(out));
      // The body ends where the trailer's first line, which starts with
      // '[', begins; no body line starts with '[' here.
      const size_t trailer = out.starts_with('[') ? 0 : out.find("\n[");
      const size_t body =
          trailer == std::string::npos ? out.size() : trailer + (trailer > 0);
      EXPECT_LE(body, max_bytes) << OutputShapeName(shape);
    }
  }
}

}  // namespace
}  // namespace ferret

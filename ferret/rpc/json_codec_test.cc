// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/rpc/json_codec.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"

#include "ferret/testing/status_matchers.h"

namespace ferret {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::Optional;
using Json = nlohmann::json;

constexpr uint64_t kSeed = 20260728;

std::string Nested(int depth) {
  return std::string(static_cast<size_t>(depth), '[') +
         std::string(static_cast<size_t>(depth), ']');
}

TEST(ParseJsonTest, ParsesOneValue) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      Json value, ParseJson(R"({"a": [1, -2, 3.5, "x", true, null]})"));
  EXPECT_EQ(value, Json::parse(R"({"a": [1, -2, 3.5, "x", true, null]})"));
}

TEST(ParseJsonTest, SyntaxErrorNamesByteOffset) {
  EXPECT_THAT(ParseJson(R"({"a": 1,})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("invalid json at byte 9")));
}

TEST(ParseJsonTest, RejectsWhatRfc8259Rejects) {
  for (const absl::string_view text :
       {"", "   ", "{} {}", "[1] x", "// c\n1", "{'a': 1}", "[1,]", "01",
        "\"\\x\"", "\"\xff\"", "\"\xc3\"", "NaN"}) {
    SCOPED_TRACE(text);
    EXPECT_THAT(ParseJson(text), StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(ParseJsonTest, QuotesAtMostAShortPrefixOfTheBadToken) {
  const std::string text = absl::StrCat("\"", std::string(1000, 'a'), "\x01\"");
  const absl::StatusOr<Json> value = ParseJson(text);
  ASSERT_THAT(value, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_LT(value.status().message().size(), 100u);
}

TEST(ParseJsonTest, EnforcesMaxBytes) {
  EXPECT_THAT(ParseJson("[1, 2]", {.max_bytes = 6}), IsOk());
  EXPECT_THAT(ParseJson("[1, 2] ", {.max_bytes = 6}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("7 bytes exceeds the limit of 6")));
}

TEST(ParseJsonTest, EnforcesMaxDepth) {
  EXPECT_THAT(ParseJson(Nested(3), {.max_depth = 3}), IsOk());
  EXPECT_THAT(ParseJson(R"({"a": [{}]})", {.max_depth = 3}), IsOk());
  EXPECT_THAT(ParseJson(Nested(4), {.max_depth = 3}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("nesting exceeds the limit of 3")));
  EXPECT_THAT(ParseJson(R"({"a": [{"b": []}]})", {.max_depth = 3}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ParseJsonTest, SurvivesPathologicalNesting) {
  EXPECT_THAT(ParseJson(std::string(1'000'000, '[')),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("nesting exceeds the limit of 64")));
}

// The deepest nesting of arrays and objects in `text`, which must be valid
// JSON: brackets inside strings do not count.
int ReferenceDepth(absl::string_view text) {
  int depth = 0;
  int max_depth = 0;
  bool in_string = false;
  for (size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (in_string) {
      if (c == '\\') {
        ++i;
      } else if (c == '"') {
        in_string = false;
      }
    } else if (c == '"') {
      in_string = true;
    } else if (c == '[' || c == '{') {
      if (++depth > max_depth) max_depth = depth;
    } else if (c == ']' || c == '}') {
      --depth;
    }
  }
  return max_depth;
}

// Returns a random value nested at least `min_depth - depth` levels deep:
// until then, every value is a non-empty array or object.
Json RandomValue(std::mt19937_64& rng, int depth, int min_depth = 0) {
  const uint64_t kind = depth < min_depth ? 5 + rng() % 2
                        : depth >= 12     ? rng() % 4
                                          : rng() % 7;
  // The one child that carries the minimum depth.
  const uint64_t min_children = depth < min_depth ? 1 : 0;
  switch (kind) {
    case 0:
      return nullptr;
    case 1:
      return rng() % 2 == 0;
    case 2:
      return static_cast<int64_t>(rng() % 2001) - 1000;
    case 3: {
      std::string text;
      for (uint64_t n = rng() % 6; n > 0; --n) {
        text.push_back("ab\"\\/[]{}\n\t"[rng() % 11]);
      }
      return text;
    }
    case 4:
      return static_cast<double>(rng() % 1000) / 8;
    case 5: {
      Json array = Json::array();
      for (uint64_t n = min_children + rng() % 4; n > 0; --n) {
        array.push_back(RandomValue(rng, depth + 1, min_depth));
      }
      return array;
    }
    default: {
      Json object = Json::object();
      for (uint64_t n = min_children + rng() % 4; n > 0; --n) {
        object[absl::StrCat("k", rng() % 8)] =
            RandomValue(rng, depth + 1, min_depth);
      }
      return object;
    }
  }
}

// Random documents, some mutated into invalid ones, must be accepted exactly
// when nlohmann's own validator accepts them and they nest no deeper than
// the limit, and must then parse to what nlohmann's parser produces.
TEST(ParseJsonTest, AgreesWithNlohmannAndReferenceDepth) {
  SCOPED_TRACE(absl::StrCat("seed ", kSeed));
  std::mt19937_64 rng(kSeed);
  constexpr int kMaxDepth = 4;
  constexpr absl::string_view kMutations = "[]{},:\"\\ 0a\xff\xc3";
  int accepted = 0;
  int too_deep = 0;
  for (int iteration = 0; iteration < 20'000; ++iteration) {
    // A third of the documents nest deeper than kMaxDepth by construction.
    const int min_depth = static_cast<int>(rng() % 3) == 0 ? kMaxDepth + 1 : 0;
    std::string text = SerializeJson(RandomValue(rng, 0, min_depth));
    for (uint64_t n = rng() % 3; n > 0 && !text.empty(); --n) {
      const size_t at = static_cast<size_t>(rng() % text.size());
      const char c = kMutations[rng() % kMutations.size()];
      switch (rng() % 3) {
        case 0:
          text[at] = c;
          break;
        case 1:
          text.insert(at, 1, c);
          break;
        default:
          text.erase(at, 1);
      }
    }
    SCOPED_TRACE(text);
    const bool valid = Json::accept(text);
    const bool shallow = valid && ReferenceDepth(text) <= kMaxDepth;
    const absl::StatusOr<Json> value =
        ParseJson(text, {.max_depth = kMaxDepth});
    ASSERT_EQ(value.ok(), shallow) << value.status();
    if (valid && !shallow) ++too_deep;
    if (!shallow) continue;
    ++accepted;
    EXPECT_EQ(*value, Json::parse(text, nullptr, /*allow_exceptions=*/false));
  }
  // Guard against a generator that exercises only one side.
  EXPECT_GT(accepted, 5'000);
  EXPECT_GT(too_deep, 500);
}

TEST(SerializeJsonTest, IsCompact) {
  EXPECT_EQ(SerializeJson(Json::parse(R"({"b": [1, "x"], "a": null})")),
            R"({"a":null,"b":[1,"x"]})");
}

TEST(SerializeJsonTest, EscapesLineBreaksAndControlCharacters) {
  EXPECT_EQ(SerializeJson(Json("a\nb\r\x01")), R"("a\nb\r\u0001")");
}

TEST(SerializeJsonTest, KeepsValidUtf8) {
  EXPECT_EQ(SerializeJson(Json("K\u00e9lvin \u212a")),
            "\"K\u00e9lvin \u212a\"");
}

TEST(SerializeJsonTest, ReplacesInvalidUtf8InValuesAndKeys) {
  Json object = Json::object();
  object["bad\xff"] = "\xc3(";
  EXPECT_EQ(SerializeJson(object), "{\"bad\xef\xbf\xbd\":\"\xef\xbf\xbd(\"}");
}

TEST(SerializeJsonTest, RandomBytesYieldOneParsableLine) {
  SCOPED_TRACE(absl::StrCat("seed ", kSeed));
  std::mt19937_64 rng(kSeed);
  for (int iteration = 0; iteration < 2'000; ++iteration) {
    std::string bytes;
    for (uint64_t n = rng() % 32; n > 0; --n) {
      bytes.push_back(static_cast<char>(rng() % 256));
    }
    Json object = Json::object();
    object[bytes] = bytes;
    const std::string line = SerializeJson(object);
    SCOPED_TRACE(line);
    EXPECT_EQ(line.find_first_of("\n\r"), std::string::npos);
    EXPECT_THAT(ParseJson(line), IsOk());
  }
}

TEST(JsonViewTest, ReadsTypedMembers) {
  const Json doc = Json::parse(
      R"({"s": "x", "i": -7, "u": 7, "b": true, "o": {}, "a": [1]})");
  const JsonView view(doc);
  EXPECT_THAT(view.GetString("s"), IsOkAndHolds("x"));
  EXPECT_THAT(view.GetInt64("i"), IsOkAndHolds(-7));
  EXPECT_THAT(view.GetInt64("u"), IsOkAndHolds(7));
  EXPECT_THAT(view.GetUint64("u"), IsOkAndHolds(7u));
  EXPECT_THAT(view.GetBool("b"), IsOkAndHolds(true));
  FERRET_ASSERT_OK_AND_ASSIGN(const JsonView object, view.GetObject("o"));
  EXPECT_EQ(object.path(), "/o");
  FERRET_ASSERT_OK_AND_ASSIGN(const JsonView array, view.GetArray("a"));
  EXPECT_EQ(array.path(), "/a");
  EXPECT_EQ(array.size(), 1u);
}

TEST(JsonViewTest, TypeMismatchNamesPathAndTypes) {
  const Json doc = Json::parse(R"({"params": {"arguments": {"query": 1}}})");
  FERRET_ASSERT_OK_AND_ASSIGN(const JsonView params,
                              JsonView(doc).GetObject("params"));
  FERRET_ASSERT_OK_AND_ASSIGN(const JsonView arguments,
                              params.GetObject("arguments"));
  EXPECT_THAT(
      arguments.GetString("query"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "/params/arguments/query: expected string, got integer"));
  EXPECT_THAT(params.GetArray("arguments"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "/params/arguments: expected array, got object"));
}

TEST(JsonViewTest, ReadingMembersOfANonObjectFails) {
  const Json doc = Json::parse("[1]");
  EXPECT_THAT(JsonView(doc).GetString("a"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "(root): expected object, got array"));
  EXPECT_THAT(JsonView(doc).GetOptionalString("a"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "(root): expected object, got array"));
}

TEST(JsonViewTest, RequiredMemberMustBePresentAndNotNull) {
  const Json doc = Json::parse(R"({"n": null})");
  EXPECT_THAT(JsonView(doc).GetString("n"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "/n: missing required member"));
  EXPECT_THAT(JsonView(doc).GetBool("absent"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "/absent: missing required member"));
}

TEST(JsonViewTest, OptionalMemberTreatsNullAsAbsent) {
  const Json doc = Json::parse(R"({"n": null, "s": "x", "i": "7"})");
  const JsonView view(doc);
  EXPECT_THAT(view.GetOptionalString("n"), IsOkAndHolds(std::nullopt));
  EXPECT_THAT(view.GetOptionalString("absent"), IsOkAndHolds(std::nullopt));
  EXPECT_THAT(view.GetOptionalString("s"),
              IsOkAndHolds(Optional(std::string("x"))));
  EXPECT_THAT(view.GetOptionalInt64("i"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "/i: expected integer, got string"));
  EXPECT_THAT(view.GetOptionalObject("n"), IsOkAndHolds(Eq(std::nullopt)));
  EXPECT_THAT(view.GetOptionalArray("absent"), IsOkAndHolds(Eq(std::nullopt)));
  EXPECT_THAT(view.GetOptionalBool("absent"), IsOkAndHolds(std::nullopt));
  EXPECT_THAT(view.GetOptionalUint64("absent"), IsOkAndHolds(std::nullopt));
}

TEST(JsonViewTest, FindReturnsNullMembers) {
  const Json doc = Json::parse(R"({"n": null})");
  const std::optional<JsonView> member = JsonView(doc).Find("n");
  ASSERT_TRUE(member.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): Asserted above.
  EXPECT_TRUE(member->value().is_null());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): Asserted above.
  EXPECT_EQ(member->path(), "/n");
  EXPECT_FALSE(JsonView(doc).Find("absent").has_value());
  const Json number(1);
  EXPECT_FALSE(JsonView(number).Find("n").has_value());
}

TEST(JsonViewTest, IntegersMustBeIntegralAndInRange) {
  const Json doc = Json::parse(R"({
    "max": 9223372036854775807, "over": 9223372036854775808,
    "min": -9223372036854775808, "neg": -1, "float": 1.0,
    "umax": 18446744073709551615})");
  const JsonView view(doc);
  EXPECT_THAT(view.GetInt64("max"),
              IsOkAndHolds(std::numeric_limits<int64_t>::max()));
  EXPECT_THAT(view.GetInt64("min"),
              IsOkAndHolds(std::numeric_limits<int64_t>::min()));
  EXPECT_THAT(view.GetInt64("over"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "/over: integer 9223372036854775808 is out of range "
                       "for int64"));
  EXPECT_THAT(view.GetUint64("over"), IsOkAndHolds(uint64_t{1} << 63));
  EXPECT_THAT(view.GetUint64("umax"),
              IsOkAndHolds(std::numeric_limits<uint64_t>::max()));
  EXPECT_THAT(view.GetUint64("neg"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "/neg: integer -1 is out of range for uint64"));
  EXPECT_THAT(view.GetInt64("float"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "/float: expected integer, got number"));
}

TEST(JsonViewTest, EscapesKeysInPaths) {
  const Json doc = Json::parse(R"({"a/b~c": 1})");
  EXPECT_THAT(JsonView(doc, "/params").GetString("a/b~c"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "/params/a~1b~0c: expected string, got integer"));
}

TEST(JsonViewTest, ElementsCarryTheirIndex) {
  const Json doc = Json::parse(R"({"items": ["a", 2]})");
  FERRET_ASSERT_OK_AND_ASSIGN(const JsonView items,
                              JsonView(doc).GetArray("items"));
  ASSERT_EQ(items.size(), 2u);
  EXPECT_THAT(items.Element(0).AsString(), IsOkAndHolds("a"));
  EXPECT_THAT(items.Element(1).AsString(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "/items/1: expected string, got integer"));
}

TEST(JsonViewTest, SizeIsZeroForNonArrays) {
  for (const absl::string_view text : {"null", "1", "\"abc\"", R"({"a": 1})"}) {
    SCOPED_TRACE(text);
    const Json value = Json::parse(text);
    EXPECT_EQ(JsonView(value).size(), 0u);
  }
}

TEST(JsonViewTest, TypeErrorDescribesUnions) {
  const Json value(true);
  EXPECT_THAT(JsonView(value, "/id").TypeError("integer or string"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "/id: expected integer or string, got boolean"));
  EXPECT_THAT(JsonView(value).AsObject(), Not(IsOk()));
}

}  // namespace
}  // namespace ferret

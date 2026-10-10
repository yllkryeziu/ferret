// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "bench/ferret/query_set.h"

#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ferret_bench {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::SizeIs;

TEST(ParseQuerySetTest, ReadsLiteralsAndRegexes) {
  const absl::StatusOr<std::vector<QuerySpec>> specs = ParseQuerySet(
      R"({"id": "q1", "class": "identifier", "kind": "literal", )"
      R"("pattern": "sched_entity", "case_sensitive": true, "df": 31})"
      "\n"
      R"({"id": "q2", "class": "regex", "kind": "regex", )"
      R"("pattern": "foo.*bar", "case_sensitive": false})"
      "\n");
  ASSERT_THAT(specs, IsOk());
  ASSERT_THAT(*specs, SizeIs(2));
  EXPECT_EQ((*specs)[0].id, "q1");
  EXPECT_EQ((*specs)[0].query_class, "identifier");
  EXPECT_EQ((*specs)[0].pattern.pattern, "sched_entity");
  EXPECT_TRUE((*specs)[0].pattern.literal);
  EXPECT_TRUE((*specs)[0].pattern.case_sensitive);
  EXPECT_FALSE((*specs)[1].pattern.literal);
  EXPECT_FALSE((*specs)[1].pattern.case_sensitive);
}

TEST(ParseQuerySetTest, RejectsInvalidQueries) {
  const auto error = [](absl::string_view text, absl::string_view message) {
    EXPECT_THAT(
        ParseQuerySet(text),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr(message)))
        << text;
  };
  error("", "no queries");
  error("{", "line 1:");
  error(R"({"id": "", "class": "c", "kind": "literal", "pattern": "x", )"
        R"("case_sensitive": true})",
        "empty id");
  error(R"({"id": "q", "class": "c", "kind": "glob", "pattern": "x", )"
        R"("case_sensitive": true})",
        "neither");
  error(R"({"id": "q", "class": "c", "kind": "literal", "pattern": ""})",
        "case_sensitive");
  error(R"({"id": "q", "class": "c", "kind": "literal", "pattern": "", )"
        R"("case_sensitive": true})",
        "empty pattern");
  error(R"({"id": "q", "class": "c", "kind": "literal", "pattern": "x", )"
        R"("case_sensitive": true})"
        "\n"
        R"({"id": "q", "class": "c", "kind": "literal", "pattern": "y", )"
        R"("case_sensitive": true})",
        "line 2: id \"q\" duplicates line 1");
}

}  // namespace
}  // namespace ferret_bench

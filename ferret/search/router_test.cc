// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
#include "ferret/search/router.h"

#include <cstdint>
#include <string>

#include "gtest/gtest.h"

namespace ferret {
namespace {
TEST(RouterTest, DecisionBoundaries) {
  struct Case {
    bool identifier;
    int64_t files;
    double precision;
    bool definitions;
    SearchRoute want;
  };
  for (const Case& test : {
           Case{false, 21, 0.0, true, SearchRoute::kText},
           Case{true, 20, 0.0, true, SearchRoute::kText},
           Case{true, 21, 0.699, false, SearchRoute::kStructural},
           Case{true, 21, 0.7, false, SearchRoute::kText},
           Case{true, 21, 1.0, true, SearchRoute::kStructural},
           Case{true, 21, 0.7, true, SearchRoute::kStructural},
       }) {
    EXPECT_EQ(DecideRoute(test.identifier, test.files, test.precision,
                          test.definitions),
              test.want);
  }
  EXPECT_EQ(
      DecideRoute(true, 5, 0.5, false, {.route_min_files = 4, .theta = 0.5}),
      SearchRoute::kText);
  EXPECT_EQ(
      DecideRoute(true, 5, 0.49, false, {.route_min_files = 4, .theta = 0.5}),
      SearchRoute::kStructural);
}

TEST(RouterTest, IdentifierLiteralOnly) {
  for (const std::string& literal : {"foo", "_a", "A_9", "\xcf\x80"}) {
    EXPECT_TRUE(IsIdentifierQuery({.pattern = literal}));
    EXPECT_FALSE(IsIdentifierQuery({.pattern = literal, .regex = true}));
  }
  for (const std::string& literal :
       {"", "9a", "foo.bar", "a-b", "a b", "\xff"}) {
    EXPECT_FALSE(IsIdentifierQuery({.pattern = literal}));
  }
}
}  // namespace
}  // namespace ferret

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/rpc/symbol_codec.h"

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

#include "ferret/rpc/json_codec.h"
#include "ferret/symbols/symbol_kind.h"
#include "ferret/workspace/workspace.h"

namespace ferret {
namespace {

TEST(SymbolCodecTest, RoundTripsByteStringsAndKinds) {
  FindSymbolRequest request{
      .name = "\xffName",
      .case_insensitive = true,
      .kinds = {SymbolKind::kFunction, SymbolKind::kTypeAlias},
      .definitions_only = true,
      .path_glob = "*.cc",
      .max_results = 17};
  auto decoded =
      DecodeFindSymbolRequest(JsonView(EncodeFindSymbolRequest(request)));
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(decoded->name, request.name);
  EXPECT_EQ(decoded->kinds, request.kinds);
  EXPECT_EQ(decoded->path_glob, request.path_glob);
  EXPECT_EQ(decoded->max_results, request.max_results);
  FindSymbolResponse response{
      .occurrences = {{"a\xff.cc", 42, SymbolKind::kFunction, true, "a\xfe"}},
      .total_count = 12,
      .pending_files = 3,
      .stale = true,
      .generation = 5};
  auto result =
      DecodeFindSymbolResponse(JsonView(EncodeFindSymbolResponse(response)));
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->occurrences, response.occurrences);
  EXPECT_EQ(result->pending_files, 3);
  EXPECT_TRUE(result->stale);
}

TEST(SymbolCodecTest, RejectsInvalidRecordRanges) {
  auto request = EncodeFindSymbolRequest({.name = "valid"});
  request["name_hex"] = "oops";
  EXPECT_EQ(DecodeFindSymbolRequest(JsonView(request)).status().code(),
            absl::StatusCode::kInvalidArgument);
  request = EncodeFindSymbolRequest({.name = "valid"});
  request["kinds"] = nlohmann::json::array({100});
  EXPECT_EQ(DecodeFindSymbolRequest(JsonView(request)).status().code(),
            absl::StatusCode::kInvalidArgument);
  auto response = EncodeFindSymbolResponse(
      {.occurrences = {{"a.cc", 1, SymbolKind::kFunction, true, "int f();"}},
       .total_count = 1});
  response["occurrences"][0]["line"] = 0;
  EXPECT_EQ(DecodeFindSymbolResponse(JsonView(response)).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ferret

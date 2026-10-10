// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/rpc/json_rpc.h"

#include <cstdint>
#include <string>
#include <variant>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "nlohmann/json_fwd.hpp"

#include "ferret/rpc/json_codec.h"

namespace ferret {
namespace {

using Json = nlohmann::json;

TEST(JsonRpcTest, RoundTripsRequestsNotificationsAndResponses) {
  const JsonRpcRequest request{
      .id = std::string("7"), .method = "search", .params = {{"q", "hi"}}};
  const JsonRpcNotification notification{.method = "changed"};
  const JsonRpcResponse response{.id = int64_t{7}, .result = {1, 2}};
  const auto decoded_request =
      DecodeJsonRpcMessage(*ParseJson(SerializeJsonRpcMessage(request)));
  ASSERT_TRUE(decoded_request.ok()) << decoded_request.status();
  ASSERT_TRUE(std::holds_alternative<JsonRpcRequest>(*decoded_request));
  EXPECT_EQ(std::get<JsonRpcRequest>(*decoded_request).id,
            JsonRpcId(std::string("7")));
  EXPECT_EQ(std::get<JsonRpcRequest>(*decoded_request).params, request.params);
  const auto decoded_notification =
      DecodeJsonRpcMessage(*ParseJson(SerializeJsonRpcMessage(notification)));
  ASSERT_TRUE(decoded_notification.ok());
  EXPECT_TRUE(
      std::holds_alternative<JsonRpcNotification>(*decoded_notification));
  const auto decoded_response =
      DecodeJsonRpcMessage(*ParseJson(SerializeJsonRpcMessage(response)));
  ASSERT_TRUE(decoded_response.ok());
  ASSERT_TRUE(std::holds_alternative<JsonRpcResponse>(*decoded_response));
  EXPECT_EQ(std::get<JsonRpcResponse>(*decoded_response).id, response.id);
  EXPECT_EQ(std::get<JsonRpcResponse>(*decoded_response).result,
            response.result);
}

TEST(JsonRpcTest, ErrorsRetainCodesDataAndNullableId) {
  for (int code :
       {kJsonRpcParseError, kJsonRpcInvalidRequest, kJsonRpcMethodNotFound,
        kJsonRpcInvalidParams, kJsonRpcInternalError}) {
    const JsonRpcResponse response{
        .error = JsonRpcError{
            .code = code, .message = "failed", .data = {{"detail", "why"}}}};
    const auto decoded =
        DecodeJsonRpcMessage(*ParseJson(SerializeJsonRpcMessage(response)));
    ASSERT_TRUE(decoded.ok()) << decoded.status();
    const auto& error = std::get<JsonRpcResponse>(*decoded);
    EXPECT_FALSE(error.id.has_value());
    ASSERT_TRUE(error.error.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access): Asserted above.
    EXPECT_EQ(error.error->code, code);
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access): Asserted above.
    EXPECT_EQ(error.error->message, "failed");
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access): Asserted above.
    EXPECT_EQ(error.error->data, response.error->data);
  }
}

TEST(JsonRpcTest, RejectsInvalidEnvelopesAndIds) {
  for (const std::string& text : {
           R"json([{"jsonrpc":"2.0","method":"x"}])json",
           R"json({"jsonrpc":"1.0","id":1,"method":"x"})json",
           R"json({"jsonrpc":"2.0","id":null,"method":"x"})json",
           R"json({"jsonrpc":"2.0","id":1.5,"method":"x"})json",
           R"json({"jsonrpc":"2.0","id":true,"method":"x"})json",
           R"json({"jsonrpc":"2.0","id":{},"method":"x"})json",
           R"json({"jsonrpc":"2.0","id":1,"method":"x","params":1})json",
           R"json({"jsonrpc":"2.0","id":1,"result":0,"error":{}})json",
           R"json({"jsonrpc":"2.0","id":1,"method":"x","result":0})json",
           R"json({"jsonrpc":"2.0","id":1,"method":"x","error":{}})json",
           R"json({"jsonrpc":"2.0","id":1})json",
           R"json({"jsonrpc":"2.0","id":1,"error":{"code":"x"}})json",
       }) {
    SCOPED_TRACE(text);
    const auto document = ParseJson(text);
    ASSERT_TRUE(document.ok());
    EXPECT_EQ(DecodeJsonRpcMessage(*document).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
  EXPECT_EQ(ParseJson("{").status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(JsonRpcIdToString(JsonRpcId(int64_t{7})), "7");
  EXPECT_EQ(JsonRpcIdToString(JsonRpcId(std::string("a\n"))), "\"a\\n\"");
}

}  // namespace
}  // namespace ferret

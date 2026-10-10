// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The JSON-RPC 2.0 message model (https://www.jsonrpc.org/specification)
// shared by the MCP server and the daemon RPC.
//
// Decoding validates the envelope and hands params, results, and error data
// on as JSON, which each method reads through JsonView. Serialization goes
// through SerializeJson, so every message fits on one line. The model is
// stricter than JSON-RPC 2.0 in the two ways MCP requires
// (https://modelcontextprotocol.io/specification/2026-07-28/basic/index):
// a request id is an integer or a string, never null or fractional, and a
// batch (a JSON array of messages) is not a message.

#ifndef FERRET_RPC_JSON_RPC_H_
#define FERRET_RPC_JSON_RPC_H_

#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include "absl/status/statusor.h"
#include "nlohmann/json.hpp"

#include "ferret/rpc/json_codec.h"

namespace ferret {

// The error codes JSON-RPC 2.0 §5.1 defines.
inline constexpr int kJsonRpcParseError = -32700;
inline constexpr int kJsonRpcInvalidRequest = -32600;
inline constexpr int kJsonRpcMethodNotFound = -32601;
inline constexpr int kJsonRpcInvalidParams = -32602;
inline constexpr int kJsonRpcInternalError = -32603;

// A request id. 7 and "7" are different ids.
using JsonRpcId = std::variant<int64_t, std::string>;

struct JsonRpcError {
  int code = 0;
  std::string message;
  // Additional information; null means the error has none.
  nlohmann::json data;
};

// A call that expects a response.
struct JsonRpcRequest {
  JsonRpcId id;
  std::string method;
  // An object or an array; null means the request has no params.
  nlohmann::json params;
};

// A call that must not be answered.
struct JsonRpcNotification {
  std::string method;
  // An object or an array; null means the notification has no params.
  nlohmann::json params;
};

// The answer to a request: a result, or an error if `error` is set.
struct JsonRpcResponse {
  // The id of the request answered. Absent only in an error response to a
  // message whose id could not be read, and then serialized as null.
  std::optional<JsonRpcId> id;
  // Any JSON value. Ignored when `error` is set.
  nlohmann::json result;
  std::optional<JsonRpcError> error;
};

using JsonRpcMessage =
    std::variant<JsonRpcRequest, JsonRpcNotification, JsonRpcResponse>;

// Reads a request id: an integer that fits int64_t, or a string. Returns
// InvalidArgument naming id.path() otherwise.
absl::StatusOr<JsonRpcId> DecodeJsonRpcId(const JsonView& id);

// Validates `message` as a JSON-RPC 2.0 request, notification, or response
// and copies its payload out. Returns InvalidArgument, with a message naming
// the offending member, for anything else; a server answers that with
// kJsonRpcInvalidRequest, addressed to the message's id when DecodeJsonRpcId
// accepts it (JSON-RPC 2.0 §5). A null "params", or a null "data" in an
// error, is treated as absent. Thread-safe.
absl::StatusOr<JsonRpcMessage> DecodeJsonRpcMessage(
    const nlohmann::json& message);

// Serialize one message as a single line without the trailing '\n', with
// members in the conventional order (jsonrpc, id, method, params; or jsonrpc,
// id, result or error). Members that are null in the struct are omitted,
// except a response's id. Thread-safe.
std::string SerializeJsonRpcMessage(const JsonRpcRequest& request);
std::string SerializeJsonRpcMessage(const JsonRpcNotification& notification);
std::string SerializeJsonRpcMessage(const JsonRpcResponse& response);

// Renders `id` for logs: 7, or "seven" quoted and C-escaped.
std::string JsonRpcIdToString(const JsonRpcId& id);

}  // namespace ferret

#endif  // FERRET_RPC_JSON_RPC_H_

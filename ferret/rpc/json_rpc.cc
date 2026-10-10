// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/rpc/json_rpc.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"

#include "ferret/rpc/json_codec.h"

namespace ferret {
namespace {

using Json = nlohmann::json;

Json IdToJson(const JsonRpcId& id) {
  return std::visit([](const auto& value) { return Json(value); }, id);
}

// Reads "params", which JSON-RPC 2.0 §4.2 requires to be structured.
absl::StatusOr<Json> DecodeParams(const JsonView& message) {
  const std::optional<JsonView> params = message.Find("params");
  if (!params.has_value() || params->value().is_null()) return Json();
  if (!params->value().is_object() && !params->value().is_array()) {
    return params->TypeError("object or array");
  }
  return params->value();
}

absl::StatusOr<JsonRpcResponse> DecodeResponse(const JsonView& message) {
  const std::optional<JsonView> result = message.Find("result");
  const std::optional<JsonView> error = message.Find("error");
  if (result.has_value() && error.has_value()) {
    return absl::InvalidArgumentError("response has both result and error");
  }
  if (!result.has_value() && !error.has_value()) {
    return absl::InvalidArgumentError(
        "message has none of method, result, and error");
  }
  const std::optional<JsonView> id = message.Find("id");
  if (!id.has_value()) {
    return absl::InvalidArgumentError("/id: missing required member");
  }
  JsonRpcResponse response;
  // Only an error response may carry a null id (JSON-RPC 2.0 §5).
  if (!id->value().is_null() || result.has_value()) {
    ABSL_ASSIGN_OR_RETURN(JsonRpcId response_id, DecodeJsonRpcId(*id));
    response.id = std::move(response_id);
  }
  if (result.has_value()) {
    response.result = result->value();
    return response;
  }
  ABSL_ASSIGN_OR_RETURN(const JsonView error_object, error->AsObject());
  ABSL_ASSIGN_OR_RETURN(const int64_t code, error_object.GetInt64("code"));
  if (code < std::numeric_limits<int>::min() ||
      code > std::numeric_limits<int>::max()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "/error/code: integer ", code, " is out of range for int"));
  }
  ABSL_ASSIGN_OR_RETURN(std::string error_message,
                        error_object.GetString("message"));
  const std::optional<JsonView> data = error_object.Find("data");
  response.error = JsonRpcError{
      .code = static_cast<int>(code),
      .message = std::move(error_message),
      .data = data.has_value() ? data->value() : Json(),
  };
  return response;
}

void AppendParams(const Json& params, std::string& out) {
  if (!params.is_null()) {
    absl::StrAppend(&out, ",\"params\":", SerializeJson(params));
  }
}

}  // namespace

absl::StatusOr<JsonRpcId> DecodeJsonRpcId(const JsonView& id) {
  if (id.value().is_string()) {
    ABSL_ASSIGN_OR_RETURN(std::string value, id.AsString());
    return JsonRpcId(std::move(value));
  }
  if (id.value().is_number_integer()) {
    ABSL_ASSIGN_OR_RETURN(const int64_t value, id.AsInt64());
    return JsonRpcId(value);
  }
  return id.TypeError("integer or string");
}

absl::StatusOr<JsonRpcMessage> DecodeJsonRpcMessage(const Json& message) {
  if (message.is_array()) {
    return absl::InvalidArgumentError("batch requests are not supported");
  }
  const JsonView root(message);
  ABSL_ASSIGN_OR_RETURN(const std::string version, root.GetString("jsonrpc"));
  if (version != "2.0") {
    return absl::InvalidArgumentError("/jsonrpc: expected \"2.0\"");
  }
  if (!root.Find("method").has_value()) return DecodeResponse(root);
  if (root.Find("result").has_value() || root.Find("error").has_value()) {
    return absl::InvalidArgumentError(
        "request has both method and result or error");
  }

  ABSL_ASSIGN_OR_RETURN(std::string method, root.GetString("method"));
  ABSL_ASSIGN_OR_RETURN(Json params, DecodeParams(root));
  const std::optional<JsonView> id = root.Find("id");
  if (!id.has_value()) {
    return JsonRpcNotification{.method = std::move(method),
                               .params = std::move(params)};
  }
  ABSL_ASSIGN_OR_RETURN(JsonRpcId request_id, DecodeJsonRpcId(*id));
  return JsonRpcRequest{.id = std::move(request_id),
                        .method = std::move(method),
                        .params = std::move(params)};
}

std::string SerializeJsonRpcMessage(const JsonRpcRequest& request) {
  std::string out = absl::StrCat(
      "{\"jsonrpc\":\"2.0\",\"id\":", SerializeJson(IdToJson(request.id)),
      ",\"method\":", SerializeJson(Json(request.method)));
  AppendParams(request.params, out);
  out.push_back('}');
  return out;
}

std::string SerializeJsonRpcMessage(const JsonRpcNotification& notification) {
  std::string out = absl::StrCat("{\"jsonrpc\":\"2.0\",\"method\":",
                                 SerializeJson(Json(notification.method)));
  AppendParams(notification.params, out);
  out.push_back('}');
  return out;
}

std::string SerializeJsonRpcMessage(const JsonRpcResponse& response) {
  std::string out = absl::StrCat(
      "{\"jsonrpc\":\"2.0\",\"id\":",
      response.id.has_value() ? SerializeJson(IdToJson(*response.id)) : "null");
  if (response.error.has_value()) {
    Json error = {{"code", response.error->code},
                  {"message", response.error->message}};
    if (!response.error->data.is_null()) error["data"] = response.error->data;
    absl::StrAppend(&out, ",\"error\":", SerializeJson(error));
  } else {
    absl::StrAppend(&out, ",\"result\":", SerializeJson(response.result));
  }
  out.push_back('}');
  return out;
}

std::string JsonRpcIdToString(const JsonRpcId& id) {
  if (const int64_t* number = std::get_if<int64_t>(&id)) {
    return absl::StrCat(*number);
  }
  return absl::StrCat("\"", absl::CHexEscape(*std::get_if<std::string>(&id)),
                      "\"");
}

}  // namespace ferret

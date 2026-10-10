// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/mcp/mcp_server.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"

#include "ferret/mcp/mcp_tool.h"
#include "ferret/mcp/protocol.h"
#include "ferret/rpc/framing.h"
#include "ferret/rpc/json_codec.h"
#include "ferret/rpc/json_rpc.h"
#include "ferret/text/utf8.h"

namespace ferret {
namespace {

using Json = nlohmann::json;

// Modern tools/list and server/discover results are cacheable
// (https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/caching).
// The tool set never changes while the process runs; five minutes bounds how
// long a client keeps a stale list across a server upgrade. The results
// describe this user's server, so shared caches must not keep them.
constexpr int64_t kListTtlMs = 300'000;
constexpr absl::string_view kCacheScope = "private";

// structuredContent and outputSchema first appear in this revision.
constexpr absl::string_view kFirstStructuredContentVersion = "2025-06-18";

bool IsModernVersion(absl::string_view version) {
  return absl::c_linear_search(kMcpModernVersions, version);
}

bool IsLegacyVersion(absl::string_view version) {
  return absl::c_linear_search(kMcpLegacyVersions, version);
}

// `version` is a revision this server serves; revisions order as strings.
bool SupportsStructuredContent(absl::string_view version) {
  return version >= kFirstStructuredContentVersion;
}

JsonRpcError InvalidParams(const absl::Status& status) {
  return {.code = kJsonRpcInvalidParams,
          .message = absl::StrCat("Invalid params: ", status.message())};
}

// Only stateless revisions may be named in _meta. Legacy revisions are
// negotiated through initialize, not advertised for stateless requests.
Json SupportedVersions() {
  Json versions = Json::array();
  for (const absl::string_view version : kMcpModernVersions) {
    versions.push_back(version);
  }
  return versions;
}

JsonRpcError UnsupportedProtocolVersion(absl::string_view requested) {
  return {
      .code = kMcpUnsupportedProtocolVersion,
      .message = "Unsupported protocol version",
      .data = {{"supported", SupportedVersions()}, {"requested", requested}}};
}

// The id of a message that failed validation, if it has a readable one, so
// that the error reaches the right caller (JSON-RPC 2.0 §5).
std::optional<JsonRpcId> ReadableId(const Json& message) {
  const std::optional<JsonView> id = JsonView(message).Find("id");
  if (!id.has_value()) return std::nullopt;
  absl::StatusOr<JsonRpcId> decoded = DecodeJsonRpcId(*id);
  if (!decoded.ok()) return std::nullopt;
  return *std::move(decoded);
}

// A tools/call from the moment it is accepted until it is answered or
// dropped.
struct McpCall {
  JsonRpcId id;
  const McpTool* tool = nullptr;
  Json arguments;
  // The revision that shapes the response.
  std::string protocol_version;
  absl::Notification cancellation;
  absl::Time accepted_at;
};

// The state of one Serve call. HandleFrame, HandleOversizedFrame, and the
// destructor run on the reader thread; calls run on the workers.
class McpSession {
 public:
  McpSession(const McpToolRegistry& tools, const McpServerOptions& options,
             RpcFrameWriter& writer);
  McpSession(const McpSession&) = delete;
  McpSession& operator=(const McpSession&) = delete;
  // Lets pending calls finish, cancelling those still pending after
  // shutdown_grace, and joins the workers.
  ~McpSession();

  void HandleFrame(absl::string_view frame);
  void HandleOversizedFrame(const absl::Status& status);

 private:
  void HandleRequest(JsonRpcRequest request, absl::Time decoded_at);
  void HandleInitialize(const JsonRpcId& id, const JsonView& params);
  void HandleModernRequest(const JsonRpcRequest& request,
                           const JsonView& params, const JsonView& meta,
                           absl::Time decoded_at);
  void HandleLegacyRequest(const JsonRpcRequest& request,
                           const JsonView& params, absl::Time decoded_at);
  void ListTools(const JsonRpcId& id, const JsonView& params,
                 absl::string_view version);
  void StartCall(const JsonRpcId& id, const JsonView& params,
                 absl::string_view version, absl::Time decoded_at)
      ABSL_LOCKS_EXCLUDED(mu_);
  void HandleNotification(const JsonRpcNotification& notification);
  void HandleCancelled(const Json& params) ABSL_LOCKS_EXCLUDED(mu_);

  void WorkLoop() ABSL_LOCKS_EXCLUDED(mu_);
  void RunCall(McpCall& call) ABSL_LOCKS_EXCLUDED(mu_);
  JsonRpcResponse CallResponse(
      const McpCall& call, const absl::StatusOr<McpToolResult>& result) const;

  // Every Notify happens under mu_, which makes checking first safe.
  void CancelLocked(McpCall& call) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (!call.cancellation.HasBeenNotified()) call.cancellation.Notify();
  }
  bool WorkAvailableLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return !queue_.empty() || finishing_;
  }
  bool IdleLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return calls_.empty();
  }

  // Adds what every modern result carries: resultType and serverInfo.
  void AddModernFields(Json& result) const;
  Json ServerInfo() const;
  void Reply(const JsonRpcId& id, Json result);
  void ReplyError(std::optional<JsonRpcId> id, JsonRpcError error);
  void Send(const JsonRpcResponse& response);

  const McpToolRegistry& tools_;
  const McpServerOptions& options_;
  RpcFrameWriter& writer_;
  // The revision initialize negotiated. Reader thread only.
  std::optional<std::string> legacy_version_;

  absl::Mutex mu_;
  // Calls queued or running, by id. A client may not reuse their ids until
  // they are answered.
  absl::flat_hash_map<JsonRpcId, std::unique_ptr<McpCall>> calls_
      ABSL_GUARDED_BY(mu_);
  // The calls in calls_ that no worker has started, oldest first.
  std::deque<McpCall*> queue_ ABSL_GUARDED_BY(mu_);
  // Set once the input has ended; workers exit when the queue is empty.
  bool finishing_ ABSL_GUARDED_BY(mu_) = false;
  // Written only by the constructor and the destructor.
  std::vector<std::thread> workers_;
};

McpSession::McpSession(const McpToolRegistry& tools,
                       const McpServerOptions& options, RpcFrameWriter& writer)
    : tools_(tools), options_(options), writer_(writer) {
  workers_.reserve(static_cast<size_t>(options_.max_concurrent_calls));
  for (int i = 0; i < options_.max_concurrent_calls; ++i) {
    workers_.emplace_back(&McpSession::WorkLoop, this);
  }
}

McpSession::~McpSession() {
  // Once the output has broken, nobody can receive a result.
  const absl::Duration grace =
      writer_.status().ok() ? options_.shutdown_grace : absl::ZeroDuration();
  size_t cancelled = 0;
  {
    absl::MutexLock lock(mu_);
    finishing_ = true;
    if (!mu_.AwaitWithTimeout(absl::Condition(this, &McpSession::IdleLocked),
                              grace)) {
      for (auto& [id, call] : calls_) CancelLocked(*call);
      cancelled = calls_.size();
    }
  }
  if (cancelled > 0) {
    LOG(WARNING) << "cancelled " << cancelled << " tool calls still pending "
                 << grace << " after the session ended";
  }
  // A worker exits once the queue is empty, after its current call.
  for (std::thread& worker : workers_) worker.join();
}

void McpSession::HandleFrame(absl::string_view frame) {
  absl::StatusOr<Json> document =
      ParseJson(frame, {.max_bytes = options_.max_message_bytes,
                        .max_depth = options_.max_json_depth});
  if (!document.ok()) {
    ReplyError(std::nullopt,
               {.code = kJsonRpcParseError,
                .message = absl::StrCat("Parse error: ",
                                        document.status().message())});
    return;
  }
  absl::StatusOr<JsonRpcMessage> message = DecodeJsonRpcMessage(*document);
  if (!message.ok()) {
    ReplyError(ReadableId(*document),
               {.code = kJsonRpcInvalidRequest,
                .message = absl::StrCat("Invalid request: ",
                                        message.status().message())});
    return;
  }
  if (JsonRpcRequest* request = std::get_if<JsonRpcRequest>(&*message)) {
    HandleRequest(std::move(*request), absl::Now());
  } else if (const JsonRpcNotification* notification =
                 std::get_if<JsonRpcNotification>(&*message)) {
    HandleNotification(*notification);
  } else {
    // This server sends no requests, so it expects no responses.
    LOG(WARNING) << "ignoring a JSON-RPC response from the client";
  }
}

void McpSession::HandleOversizedFrame(const absl::Status& status) {
  ReplyError(std::nullopt,
             {.code = kJsonRpcInvalidRequest,
              .message = absl::StrCat("Invalid request: ", status.message())});
}

void McpSession::HandleRequest(JsonRpcRequest request, absl::Time decoded_at) {
  // MCP params are an object, and optional.
  if (request.params.is_null()) request.params = Json::object();
  const JsonView params(request.params, "/params");
  if (!request.params.is_object()) {
    ReplyError(request.id, InvalidParams(params.TypeError("object")));
    return;
  }
  if (request.method == kMcpMethodInitialize) {
    HandleInitialize(request.id, params);
    return;
  }
  // A request is modern exactly when its _meta names a protocol version;
  // legacy requests may carry _meta too, for example with a progressToken.
  const absl::StatusOr<std::optional<JsonView>> meta =
      params.GetOptionalObject("_meta");
  if (!meta.ok()) {
    ReplyError(request.id, InvalidParams(meta.status()));
    return;
  }
  if (meta->has_value() && (*meta)->Find(kMcpMetaProtocolVersion).has_value()) {
    HandleModernRequest(request, params, **meta, decoded_at);
  } else {
    HandleLegacyRequest(request, params, decoded_at);
  }
}

void McpSession::HandleInitialize(const JsonRpcId& id, const JsonView& params) {
  if (legacy_version_.has_value()) {
    ReplyError(id, {.code = kJsonRpcInvalidRequest,
                    .message = "Invalid request: initialize was already "
                               "called in this session"});
    return;
  }
  const absl::StatusOr<std::string> requested =
      params.GetString("protocolVersion");
  absl::Status status = requested.status();
  if (status.ok()) status = params.GetOptionalObject("capabilities").status();
  if (status.ok()) status = params.GetOptionalObject("clientInfo").status();
  if (!status.ok()) {
    ReplyError(id, InvalidParams(status));
    return;
  }
  // Echo the client's version if it is one we serve, and otherwise offer our
  // newest (2025-11-25 basic/lifecycle, "Version Negotiation").
  legacy_version_ = IsLegacyVersion(*requested)
                        ? *requested
                        : std::string(kMcpLegacyVersions.front());
  Reply(id, {{"protocolVersion", *legacy_version_},
             {"capabilities", {{"tools", Json::object()}}},
             {"serverInfo", ServerInfo()}});
}

void McpSession::HandleModernRequest(const JsonRpcRequest& request,
                                     const JsonView& params,
                                     const JsonView& meta,
                                     absl::Time decoded_at) {
  const absl::StatusOr<std::string> version =
      meta.GetString(kMcpMetaProtocolVersion);
  if (!version.ok()) {
    ReplyError(request.id, InvalidParams(version.status()));
    return;
  }
  if (!IsModernVersion(*version)) {
    ReplyError(request.id, UnsupportedProtocolVersion(*version));
    return;
  }
  // The other per-request fields: clientCapabilities is required and
  // clientInfo optional (2026-07-28 basic/index, "_meta").
  absl::Status status = meta.GetObject(kMcpMetaClientCapabilities).status();
  if (status.ok()) status = meta.GetOptionalObject(kMcpMetaClientInfo).status();
  if (!status.ok()) {
    ReplyError(request.id, InvalidParams(status));
    return;
  }
  if (request.method == kMcpMethodDiscover) {
    Json result = {{"supportedVersions", SupportedVersions()},
                   {"capabilities", {{"tools", Json::object()}}},
                   {"ttlMs", kListTtlMs},
                   {"cacheScope", kCacheScope}};
    AddModernFields(result);
    Reply(request.id, std::move(result));
  } else if (request.method == kMcpMethodToolsList) {
    ListTools(request.id, params, *version);
  } else if (request.method == kMcpMethodToolsCall) {
    StartCall(request.id, params, *version, decoded_at);
  } else {
    ReplyError(request.id,
               {.code = kJsonRpcMethodNotFound,
                .message = absl::StrCat("Method not found: ", request.method)});
  }
}

void McpSession::HandleLegacyRequest(const JsonRpcRequest& request,
                                     const JsonView& params,
                                     absl::Time decoded_at) {
  const absl::string_view method = request.method;
  if (method == kMcpMethodPing) {
    Reply(request.id, Json::object());
    return;
  }
  if (method != kMcpMethodToolsList && method != kMcpMethodToolsCall &&
      method != kMcpMethodDiscover) {
    ReplyError(request.id,
               {.code = kJsonRpcMethodNotFound,
                .message = absl::StrCat("Method not found: ", method)});
    return;
  }
  // Without a negotiated legacy revision the request must be modern, and a
  // modern request names its version.
  if (method == kMcpMethodDiscover || !legacy_version_.has_value()) {
    ReplyError(
        request.id,
        {.code = kJsonRpcInvalidParams,
         .message = absl::StrCat(
             "Invalid params: _meta must name \"", kMcpMetaProtocolVersion,
             "\"; legacy clients must send initialize first")});
    return;
  }
  if (method == kMcpMethodToolsList) {
    ListTools(request.id, params, *legacy_version_);
  } else {
    StartCall(request.id, params, *legacy_version_, decoded_at);
  }
}

void McpSession::ListTools(const JsonRpcId& id, const JsonView& params,
                           absl::string_view version) {
  const absl::StatusOr<std::optional<std::string>> cursor =
      params.GetOptionalString("cursor");
  if (!cursor.ok()) {
    ReplyError(id, InvalidParams(cursor.status()));
    return;
  }
  // Every tool fits on the first page, so this server never issues a cursor
  // (2026-07-28 server/utilities/pagination: invalid cursors get -32602).
  if (cursor->has_value()) {
    ReplyError(id,
               {.code = kJsonRpcInvalidParams,
                .message = "Invalid params: /params/cursor: unknown cursor"});
    return;
  }
  Json tools = Json::array();
  for (const std::unique_ptr<McpTool>& tool : tools_.tools()) {
    Json entry = {{"name", tool->name()},
                  {"description", tool->description()},
                  {"inputSchema", tool->input_schema()}};
    Json output_schema = tool->output_schema();
    if (!output_schema.is_null() && SupportsStructuredContent(version)) {
      entry["outputSchema"] = std::move(output_schema);
    }
    tools.push_back(std::move(entry));
  }
  Json result = {{"tools", std::move(tools)}};
  if (IsModernVersion(version)) {
    result["ttlMs"] = kListTtlMs;
    result["cacheScope"] = kCacheScope;
    AddModernFields(result);
  }
  Reply(id, std::move(result));
}

void McpSession::StartCall(const JsonRpcId& id, const JsonView& params,
                           absl::string_view version, absl::Time decoded_at) {
  const absl::StatusOr<std::string> name = params.GetString("name");
  if (!name.ok()) {
    ReplyError(id, InvalidParams(name.status()));
    return;
  }
  const McpTool* tool = tools_.Find(*name);
  if (tool == nullptr) {
    ReplyError(id, {.code = kJsonRpcInvalidParams,
                    .message = absl::StrCat("Unknown tool: ", *name)});
    return;
  }
  const absl::StatusOr<std::optional<JsonView>> arguments =
      params.GetOptionalObject("arguments");
  if (!arguments.ok()) {
    ReplyError(id, InvalidParams(arguments.status()));
    return;
  }
  auto call = std::make_unique<McpCall>();
  call->id = id;
  call->tool = tool;
  call->arguments =
      arguments->has_value() ? (*arguments)->value() : Json::object();
  call->protocol_version = std::string(version);
  call->accepted_at = decoded_at;

  std::optional<JsonRpcError> rejection;
  {
    absl::MutexLock lock(mu_);
    if (calls_.contains(id)) {
      rejection = {
          .code = kJsonRpcInvalidRequest,
          .message = absl::StrCat("Invalid request: id ", JsonRpcIdToString(id),
                                  " belongs to a pending call")};
    } else if (calls_.size() >=
               static_cast<size_t>(options_.max_pending_calls)) {
      rejection = {.code = kJsonRpcInternalError,
                   .message = absl::StrCat("Internal error: ", calls_.size(),
                                           " tool calls are already pending")};
    } else {
      queue_.push_back(call.get());
      calls_.emplace(id, std::move(call));
    }
  }
  if (rejection.has_value()) ReplyError(id, *std::move(rejection));
}

void McpSession::HandleNotification(const JsonRpcNotification& notification) {
  if (notification.method == kMcpNotificationCancelled) {
    HandleCancelled(notification.params);
    return;
  }
  // notifications/initialized ends the legacy handshake and needs no action.
  // Other notifications are not ours to answer (JSON-RPC 2.0 §4.1).
  if (notification.method != kMcpNotificationInitialized) {
    VLOG(1) << "ignoring notification " << notification.method;
  }
}

void McpSession::HandleCancelled(const Json& params) {
  // Malformed cancellations are ignored, and so are unknown ids: the call may
  // have been answered already (2026-07-28 basic/patterns/cancellation).
  const JsonView view(params, "/params");
  const std::optional<JsonView> request_id = view.Find("requestId");
  const absl::StatusOr<JsonRpcId> id =
      request_id.has_value()
          ? DecodeJsonRpcId(*request_id)
          : absl::InvalidArgumentError("/params/requestId: missing");
  if (!id.ok()) {
    LOG(WARNING) << "ignoring a malformed notifications/cancelled: "
                 << id.status();
    return;
  }
  const absl::StatusOr<std::optional<std::string>> reason =
      view.GetOptionalString("reason");
  LOG(INFO) << "client cancelled request " << JsonRpcIdToString(*id)
            << (reason.ok() && reason->has_value()
                    ? absl::StrCat(": ", **reason)
                    : "");
  absl::MutexLock lock(mu_);
  const auto it = calls_.find(*id);
  if (it != calls_.end()) CancelLocked(*it->second);
}

void McpSession::WorkLoop() {
  while (true) {
    McpCall* call = nullptr;
    {
      absl::MutexLock lock(mu_);
      mu_.Await(absl::Condition(this, &McpSession::WorkAvailableLocked));
      if (queue_.empty()) return;
      call = queue_.front();
      queue_.pop_front();
    }
    RunCall(*call);
  }
}

void McpSession::RunCall(McpCall& call) {
  // A call cancelled while it was queued never starts.
  absl::StatusOr<McpToolResult> result = absl::CancelledError();
  if (!call.cancellation.HasBeenNotified()) {
    result = call.tool->Call(JsonView(call.arguments),
                             McpToolContext(call.cancellation));
  }
  std::unique_ptr<McpCall> owned;
  bool cancelled = false;
  {
    absl::MutexLock lock(mu_);
    // Deciding under mu_ orders this against HandleCancelled: a cancellation
    // either finds the call here and suppresses the response, or comes later
    // and finds nothing.
    auto node = calls_.extract(call.id);
    DCHECK(!node.empty());
    owned = std::move(node.mapped());
    cancelled = call.cancellation.HasBeenNotified();
  }
  if (cancelled) {
    VLOG(1) << "dropping the result of cancelled request "
            << JsonRpcIdToString(call.id);
    return;
  }
  const std::string encoded =
      SerializeJsonRpcMessage(CallResponse(call, result));
  if (options_.log_call_latency) {
    LOG(INFO) << "mcp_call_latency_us tool=" << call.tool->name() << " us="
              << absl::ToInt64Microseconds(absl::Now() - call.accepted_at);
  }
  const absl::Status status = writer_.Write(encoded);
  if (!status.ok()) VLOG(1) << "dropping a response: " << status;
}

JsonRpcResponse McpSession::CallResponse(
    const McpCall& call, const absl::StatusOr<McpToolResult>& result) const {
  const absl::string_view name = call.tool->name();
  if (!result.ok()) {
    if (absl::IsInvalidArgument(result.status())) {
      std::string message;
      AppendSanitizedUtf8(result.status().message(), message);
      Json body = {
          {"content", Json::array({{{"type", "text"}, {"text", message}}})},
          {"isError", true}};
      if (IsModernVersion(call.protocol_version)) AddModernFields(body);
      return {.id = call.id, .result = std::move(body)};
    }
    LOG(ERROR) << "tool " << name << " failed: " << result.status();
    return {.id = call.id,
            .error = JsonRpcError{
                .code = kJsonRpcInternalError,
                .message = absl::StrCat("Internal error in tool ", name, ": ",
                                        result.status().message())}};
  }
  // The registry admits only object output schemas, which legacy revisions
  // require.
  if (!result->structured.is_null() && !result->structured.is_object()) {
    LOG(ERROR) << "tool " << name << " returned non-object structured content";
    return {.id = call.id,
            .error = JsonRpcError{
                .code = kJsonRpcInternalError,
                .message = absl::StrCat("Internal error in tool ", name)}};
  }
  Json text_block = {{"type", "text"}, {"text", result->text}};
  Json body = {{"content", Json::array({std::move(text_block)})},
               {"isError", result->is_error}};
  if (!result->structured.is_null() &&
      SupportsStructuredContent(call.protocol_version)) {
    body["structuredContent"] = result->structured;
  }
  if (IsModernVersion(call.protocol_version)) AddModernFields(body);
  return {.id = call.id, .result = std::move(body)};
}

void McpSession::AddModernFields(Json& result) const {
  result["resultType"] = "complete";
  result["_meta"] = {{kMcpMetaServerInfo, ServerInfo()}};
}

Json McpSession::ServerInfo() const {
  return {{"name", options_.server_name}, {"version", options_.server_version}};
}

void McpSession::Reply(const JsonRpcId& id, Json result) {
  Send({.id = id, .result = std::move(result)});
}

void McpSession::ReplyError(std::optional<JsonRpcId> id, JsonRpcError error) {
  Send({.id = std::move(id), .error = std::move(error)});
}

void McpSession::Send(const JsonRpcResponse& response) {
  // The writer remembers the first failure, and Serve reports it.
  const absl::Status status = writer_.Write(SerializeJsonRpcMessage(response));
  if (!status.ok()) VLOG(1) << "dropping a response: " << status;
}

}  // namespace

McpServer::McpServer(const McpToolRegistry& tools, McpServerOptions options)
    : tools_(tools), options_(std::move(options)) {
  CHECK_GE(options_.max_concurrent_calls, 1);
  CHECK_GE(options_.max_pending_calls, 1);
}

absl::Status McpServer::Serve(int in_fd, int out_fd) const {
  RpcFrameWriter writer(out_fd);
  absl::Status read_status;
  {
    McpSession session(tools_, options_, writer);
    RpcFrameReader reader(in_fd,
                          {.max_frame_bytes = options_.max_message_bytes});
    // Reading on after the output has broken would only start calls whose
    // results nobody can receive.
    while (writer.status().ok()) {
      absl::StatusOr<std::string> frame = reader.Read();
      if (frame.ok()) {
        session.HandleFrame(*frame);
        continue;
      }
      if (absl::IsResourceExhausted(frame.status())) {
        session.HandleOversizedFrame(frame.status());
        continue;
      }
      if (absl::IsDataLoss(frame.status())) {
        LOG(WARNING) << "dropping a truncated final message: "
                     << frame.status();
      } else if (!absl::IsOutOfRange(frame.status())) {
        read_status = frame.status();
      }
      break;
    }
  }  // Lets pending calls finish and joins the workers.
  if (!read_status.ok()) return read_status;
  return writer.status();
}

}  // namespace ferret

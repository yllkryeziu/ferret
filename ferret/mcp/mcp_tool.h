// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// McpTool, the interface behind each tool the MCP server exposes, and
// McpToolRegistry, the fixed, ordered set of tools a server serves.
//
// A tool publishes a name, a description, and JSON Schemas for its arguments
// and its structured result, and runs calls. It never sees the protocol: the
// server shapes the result for the client's revision and maps a tool's
// status to JSON-RPC errors
// (https://modelcontextprotocol.io/specification/2026-07-28/server/tools).
// Experiment arms swap the implementation behind a tool while its name,
// description, and schemas stay byte-identical.

#ifndef FERRET_MCP_MCP_TOOL_H_
#define FERRET_MCP_MCP_TOOL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "nlohmann/json.hpp"

#include "ferret/rpc/json_codec.h"

namespace ferret {

// What a tool sees of the call it is running. Valid only during the call.
// Thread-safe.
class McpToolContext final {
 public:
  explicit McpToolContext(
      const absl::Notification& cancellation ABSL_ATTRIBUTE_LIFETIME_BOUND)
      : cancellation_(cancellation) {}
  McpToolContext(const McpToolContext&) = delete;
  McpToolContext& operator=(const McpToolContext&) = delete;

  // True once the client has cancelled the call, or the server has given up
  // on it while shutting down. Nobody reads the result of a cancelled call,
  // so a tool should check this between units of work and return promptly.
  bool cancelled() const { return cancellation_.HasBeenNotified(); }

  // Notified when the call is cancelled, for tools that wait: they can use
  // WaitForNotificationWithTimeout() instead of sleeping.
  const absl::Notification& cancellation() const { return cancellation_; }

 private:
  const absl::Notification& cancellation_;
};

// The outcome of a call that ran to completion.
struct McpToolResult {
  // The text of the result's single text content block. When `structured`
  // is set this should carry the same information, because clients that
  // ignore structured content show only the text.
  std::string text;
  // The result's structuredContent; null means none. It must conform to the
  // tool's output_schema(). Legacy clients accept only objects.
  nlohmann::json structured;
  // A failure the model can act on, such as an invalid regular expression.
  // The client shows it to the model as a result with isError: true.
  bool is_error = false;
};

// Implementations must be thread-safe: the server runs concurrent calls on
// several threads.
class McpTool {
 public:
  McpTool() = default;
  McpTool(const McpTool&) = delete;
  McpTool& operator=(const McpTool&) = delete;
  virtual ~McpTool() = default;

  // Unique within a registry; 1 to 128 characters from [A-Za-z0-9_.-].
  // Valid while *this is alive.
  virtual absl::string_view name() const = 0;
  // Tells the model when and how to use the tool. Valid while *this is
  // alive.
  virtual absl::string_view description() const = 0;
  // A JSON Schema whose root has "type": "object".
  virtual nlohmann::json input_schema() const = 0;
  // A JSON Schema whose root has "type": "object", or null if the tool
  // returns no structured content.
  virtual nlohmann::json output_schema() const = 0;

  // Runs one call on `arguments`, a JSON object. Returns:
  //   * a result, with is_error set for failures the model can correct;
  //   * InvalidArgument if the arguments do not satisfy input_schema(), which
  //     the client receives as JSON-RPC error -32602 (the JsonView readers
  //     produce suitable messages);
  //   * any other status for a server fault, sent as -32603.
  // The result of a cancelled call is discarded, whatever it is.
  virtual absl::StatusOr<McpToolResult> Call(
      JsonView arguments, const McpToolContext& context) const = 0;
};

// The tools one server exposes, in the order tools/list reports them. MCP
// asks servers for a deterministic order, which lets clients cache the list
// and keeps LLM prompt caches warm. Immutable, hence thread-safe.
class McpToolRegistry final {
 public:
  // Takes ownership of `tools`. Returns InvalidArgument if a name is invalid
  // or repeated, or a schema is not an object schema.
  static absl::StatusOr<McpToolRegistry> FromTools(
      std::vector<std::unique_ptr<McpTool>> tools);

  McpToolRegistry(const McpToolRegistry&) = delete;
  McpToolRegistry& operator=(const McpToolRegistry&) = delete;
  McpToolRegistry(McpToolRegistry&&) = default;
  McpToolRegistry& operator=(McpToolRegistry&&) = default;

  // In registration order. Valid while *this is alive.
  absl::Span<const std::unique_ptr<McpTool>> tools() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return tools_;
  }

  // Returns the tool called `name`, or null. O(number of tools), which is a
  // handful. The tool lives as long as *this.
  const McpTool* absl_nullable Find(absl::string_view name) const;

 private:
  explicit McpToolRegistry(std::vector<std::unique_ptr<McpTool>> tools)
      : tools_(std::move(tools)) {}

  std::vector<std::unique_ptr<McpTool>> tools_;
};

}  // namespace ferret

#endif  // FERRET_MCP_MCP_TOOL_H_

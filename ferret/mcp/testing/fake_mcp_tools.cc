// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/mcp/testing/fake_mcp_tools.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "nlohmann/json_fwd.hpp"

#include "ferret/mcp/mcp_tool.h"
#include "ferret/rpc/json_codec.h"

namespace ferret_testing {

using ::ferret::JsonView;
using ::ferret::McpToolContext;
using ::ferret::McpToolResult;
using Json = nlohmann::json;

absl::string_view EchoMcpTool::description() const {
  return "Returns its message.";
}

Json EchoMcpTool::input_schema() const {
  return {{"type", "object"},
          {"properties", {{"message", {{"type", "string"}}}}},
          {"required", {"message"}}};
}

Json EchoMcpTool::output_schema() const { return input_schema(); }

absl::StatusOr<McpToolResult> EchoMcpTool::Call(
    JsonView arguments, const McpToolContext& /*context*/) const {
  ABSL_ASSIGN_OR_RETURN(std::string message, arguments.GetString("message"));
  return McpToolResult{.text = message,
                       .structured = {{"message", std::move(message)}}};
}

absl::string_view BlockingMcpTool::description() const {
  return "Waits until the test releases it or the call is cancelled.";
}

Json BlockingMcpTool::input_schema() const { return {{"type", "object"}}; }

Json BlockingMcpTool::output_schema() const {
  return {{"type", "object"},
          {"properties", {{"released", {{"type", "boolean"}}}}},
          {"required", {"released"}}};
}

absl::StatusOr<McpToolResult> BlockingMcpTool::Call(
    JsonView /*arguments*/, const McpToolContext& context) const {
  {
    absl::MutexLock lock(mu_);
    ++started_calls_;
    peak_running_calls_ = std::max(peak_running_calls_, ++running_calls_);
  }
  // Release and cancellation are separate notifications: wait on one and
  // poll the other.
  bool cancelled = false;
  while (!cancelled && !released_.HasBeenNotified()) {
    cancelled = context.cancellation().WaitForNotificationWithTimeout(
        absl::Milliseconds(1));
  }
  absl::MutexLock lock(mu_);
  --running_calls_;
  if (cancelled) {
    ++cancelled_calls_;
    return absl::CancelledError("block: cancelled");
  }
  return McpToolResult{.text = "released", .structured = {{"released", true}}};
}

void BlockingMcpTool::WaitForStartedCalls(int count) const {
  absl::MutexLock lock(mu_);
  const auto enough = [this, count]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return started_calls_ >= count;
  };
  mu_.Await(absl::Condition(&enough));
}

int BlockingMcpTool::cancelled_calls() const {
  absl::MutexLock lock(mu_);
  return cancelled_calls_;
}

int BlockingMcpTool::peak_running_calls() const {
  absl::MutexLock lock(mu_);
  return peak_running_calls_;
}

absl::string_view FailingMcpTool::description() const {
  return "Fails: as a tool error, or as a server fault if internal is true.";
}

Json FailingMcpTool::input_schema() const {
  return {{"type", "object"},
          {"properties", {{"internal", {{"type", "boolean"}}}}}};
}

absl::StatusOr<McpToolResult> FailingMcpTool::Call(
    JsonView arguments, const McpToolContext& /*context*/) const {
  ABSL_ASSIGN_OR_RETURN(const std::optional<bool> internal,
                        arguments.GetOptionalBool("internal"));
  if (internal.value_or(false)) {
    return absl::InternalError("fail: failed internally, as asked");
  }
  return McpToolResult{.text = "fail: failed, as asked", .is_error = true};
}

}  // namespace ferret_testing

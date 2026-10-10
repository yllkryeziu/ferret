// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Fake MCP tools that exercise every path of the server: a tool that
// succeeds with structured content, one that blocks until the test releases
// it or the client cancels the call, and one that fails either as a tool
// error the model sees or as a server fault.

#ifndef FERRET_MCP_TESTING_FAKE_MCP_TOOLS_H_
#define FERRET_MCP_TESTING_FAKE_MCP_TOOLS_H_

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "nlohmann/json.hpp"

#include "ferret/mcp/mcp_tool.h"
#include "ferret/rpc/json_codec.h"

namespace ferret_testing {

// "echo": returns its required string argument "message" as the text and as
// {"message": ...}. Thread-safe.
class EchoMcpTool final : public ferret::McpTool {
 public:
  absl::string_view name() const override { return "echo"; }
  absl::string_view description() const override;
  nlohmann::json input_schema() const override;
  nlohmann::json output_schema() const override;
  absl::StatusOr<ferret::McpToolResult> Call(
      ferret::JsonView arguments,
      const ferret::McpToolContext& context) const override;
};

// "block": each call waits until Release() is called, then returns
// {"released": true}, or until the call is cancelled, then returns
// CancelledError. Thread-safe.
class BlockingMcpTool final : public ferret::McpTool {
 public:
  absl::string_view name() const override { return "block"; }
  absl::string_view description() const override;
  nlohmann::json input_schema() const override;
  nlohmann::json output_schema() const override;
  absl::StatusOr<ferret::McpToolResult> Call(
      ferret::JsonView arguments,
      const ferret::McpToolContext& context) const override;

  // Lets every current and future call finish. Call at most once.
  void Release() { released_.Notify(); }
  // Blocks until at least `count` calls have started.
  void WaitForStartedCalls(int count) const ABSL_LOCKS_EXCLUDED(mu_);
  // The number of calls that returned because they were cancelled.
  int cancelled_calls() const ABSL_LOCKS_EXCLUDED(mu_);
  // The largest number of calls that were ever running at once.
  int peak_running_calls() const ABSL_LOCKS_EXCLUDED(mu_);

 private:
  absl::Notification released_;
  // Call() is const, so the counters it updates are mutable.
  mutable absl::Mutex mu_;
  mutable int started_calls_ ABSL_GUARDED_BY(mu_) = 0;
  mutable int running_calls_ ABSL_GUARDED_BY(mu_) = 0;
  mutable int peak_running_calls_ ABSL_GUARDED_BY(mu_) = 0;
  mutable int cancelled_calls_ ABSL_GUARDED_BY(mu_) = 0;
};

// "fail": returns a result with is_error set, or, when its optional boolean
// argument "internal" is true, InternalError. Publishes no output schema.
// Thread-safe.
class FailingMcpTool final : public ferret::McpTool {
 public:
  absl::string_view name() const override { return "fail"; }
  absl::string_view description() const override;
  nlohmann::json input_schema() const override;
  nlohmann::json output_schema() const override { return nullptr; }
  absl::StatusOr<ferret::McpToolResult> Call(
      ferret::JsonView arguments,
      const ferret::McpToolContext& context) const override;
};

}  // namespace ferret_testing

#endif  // FERRET_MCP_TESTING_FAKE_MCP_TOOLS_H_

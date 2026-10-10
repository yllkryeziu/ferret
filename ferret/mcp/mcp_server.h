// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// McpServer: a dual-era Model Context Protocol server over newline-delimited
// JSON-RPC. `ferret mcp` serves it on stdin and stdout.
//
// The era of each request follows
// https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning:
//
//   * A request whose params._meta names
//   io.modelcontextprotocol/protocolVersion
//     is modern and is served statelessly: server/discover, tools/list, and
//     tools/call. A version other than those in kMcpModernVersions gets
//     UnsupportedProtocolVersionError (-32022), and a request without the
//     required clientCapabilities gets -32602.
//   * initialize negotiates a legacy revision for the rest of the session
//     (https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle),
//     after which tools/list and tools/call without modern _meta get that
//     revision's result shapes. ping is answered on any request without
//     modern _meta, because legacy clients may ping before initialize
//     completes.
//   * Both eras may interleave on one session.
//
// The thread that calls Serve reads every message and answers everything
// except tools/call, which runs on worker threads, at most
// max_concurrent_calls at a time. The reader therefore keeps reading while
// tools run, and notifications/cancelled reaches a running call through its
// McpToolContext. A cancelled call gets no response, as the stdio transport
// requires
// (https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio).
// Responses are written as whole lines, in completion order.
//
// All input is untrusted: message length, JSON nesting, and the number of
// pending calls are bounded, and every field is read through JsonView. The
// output carries nothing but protocol messages; diagnostics go to the log.

#ifndef FERRET_MCP_MCP_SERVER_H_
#define FERRET_MCP_MCP_SERVER_H_

#include <cstdint>
#include <string>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/time/time.h"

#include "ferret/mcp/mcp_tool.h"

namespace ferret {

struct McpServerOptions {
  // Reported as serverInfo by initialize and in every modern result's _meta.
  std::string server_name = "ferret";
  std::string server_version = "0.1.0";
  // Longest accepted message. MCP requests carry tool arguments and a little
  // metadata, so a longer line is a broken or hostile client. It gets -32600,
  // and the session continues with the next line.
  int64_t max_message_bytes = int64_t{1} << 20;
  // Deepest accepted JSON nesting in a message.
  int max_json_depth = 64;
  // tools/call requests that run at once. More wait in a queue.
  int max_concurrent_calls = 4;
  // tools/call requests that may be queued or running. More are rejected
  // with -32603.
  int max_pending_calls = 64;
  // Logs decoded tools/call-to-encoded-response latency to stderr for
  // measurements. Disabled by default.
  bool log_call_latency = false;
  // How long calls still pending at the end of the input may run before
  // they are cancelled.
  absl::Duration shutdown_grace = absl::Seconds(10);
};

// Immutable, hence thread-safe: concurrent Serve calls run independent
// sessions.
class McpServer final {
 public:
  // Borrows `tools`, which must outlive the server. Requires
  // options.max_concurrent_calls >= 1 and options.max_pending_calls >= 1.
  McpServer(const McpToolRegistry& tools ABSL_ATTRIBUTE_LIFETIME_BOUND,
            McpServerOptions options);
  McpServer(const McpServer&) = delete;
  McpServer& operator=(const McpServer&) = delete;

  // Serves one session. Reads messages from `in_fd` until the end of input,
  // then lets pending calls finish (cancelling any still pending after
  // shutdown_grace), and returns. Responses go to `out_fd`. Neither
  // descriptor is closed. Returns OK after a clean end of input (a truncated
  // final line is logged and dropped), the read error if reading fails, or
  // the write error if the output broke.
  absl::Status Serve(int in_fd, int out_fd) const;

 private:
  const McpToolRegistry& tools_;
  const McpServerOptions options_;
};

}  // namespace ferret

#endif  // FERRET_MCP_MCP_SERVER_H_

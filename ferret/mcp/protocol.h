// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Model Context Protocol constants: the revisions Ferret speaks, and the
// method names, _meta keys, and error codes it uses.
//
// MCP has two eras. Modern revisions (2026-07-28 and later) are stateless:
// every request names its protocol version in _meta
// (https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning).
// Legacy revisions (2025-11-25 and earlier) negotiate a version once, with
// an initialize handshake
// (https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle).
// Ferret is dual-era: it serves both, so that agents on either kind of client
// see the same tools.

#ifndef FERRET_MCP_PROTOCOL_H_
#define FERRET_MCP_PROTOCOL_H_

#include <array>

#include "absl/strings/string_view.h"

namespace ferret {

// Newest first. Revision strings are dates, so they also compare in time
// order as strings.
inline constexpr std::array<absl::string_view, 1> kMcpModernVersions = {
    "2026-07-28"};
inline constexpr std::array<absl::string_view, 3> kMcpLegacyVersions = {
    "2025-11-25", "2025-06-18", "2025-03-26"};

// Requests.
inline constexpr absl::string_view kMcpMethodInitialize =
    "initialize";                                            // Legacy.
inline constexpr absl::string_view kMcpMethodPing = "ping";  // Legacy.
inline constexpr absl::string_view kMcpMethodDiscover =
    "server/discover";  // Modern.
inline constexpr absl::string_view kMcpMethodToolsList = "tools/list";
inline constexpr absl::string_view kMcpMethodToolsCall = "tools/call";

// Notifications.
inline constexpr absl::string_view kMcpNotificationInitialized =
    "notifications/initialized";  // Legacy.
inline constexpr absl::string_view kMcpNotificationCancelled =
    "notifications/cancelled";

// Per-request and per-result _meta keys of modern revisions
// (https://modelcontextprotocol.io/specification/2026-07-28/basic/index).
inline constexpr absl::string_view kMcpMetaProtocolVersion =
    "io.modelcontextprotocol/protocolVersion";
inline constexpr absl::string_view kMcpMetaClientCapabilities =
    "io.modelcontextprotocol/clientCapabilities";
inline constexpr absl::string_view kMcpMetaClientInfo =
    "io.modelcontextprotocol/clientInfo";
inline constexpr absl::string_view kMcpMetaServerInfo =
    "io.modelcontextprotocol/serverInfo";

// UnsupportedProtocolVersionError: a modern request named a version this
// server does not serve. Its data lists {"supported": [...], "requested"}.
inline constexpr int kMcpUnsupportedProtocolVersion = -32022;

}  // namespace ferret

#endif  // FERRET_MCP_PROTOCOL_H_

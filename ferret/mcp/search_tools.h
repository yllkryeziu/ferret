// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FERRET_MCP_SEARCH_TOOLS_H_
#define FERRET_MCP_SEARCH_TOOLS_H_

#include <memory>
#include <vector>

#include "ferret/engine/engine.h"
#include "ferret/mcp/mcp_tool.h"
#include "ferret/search/formatter.h"

namespace ferret {

// Borrows the backend. Both tools are immutable and safe for concurrent calls.
std::vector<std::unique_ptr<McpTool>> SearchTools(
    const SearchBackend& backend,
    OutputShape default_shape = OutputShape::kContext,
    SearchConsistencyMode consistency = kServingConsistency,
    bool fixed_shape = false, bool router = false);

}  // namespace ferret
#endif  // FERRET_MCP_SEARCH_TOOLS_H_

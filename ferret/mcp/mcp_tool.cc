// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/mcp/mcp_tool.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "nlohmann/json_fwd.hpp"

#include "ferret/rpc/json_codec.h"

namespace ferret {
namespace {

// The tool name rules of
// https://modelcontextprotocol.io/specification/2026-07-28/server/tools.
constexpr size_t kMaxToolNameBytes = 128;

bool IsValidToolName(absl::string_view name) {
  return !name.empty() && name.size() <= kMaxToolNameBytes &&
         absl::c_all_of(name, [](char c) {
           return absl::ascii_isalnum(static_cast<unsigned char>(c)) ||
                  c == '_' || c == '-' || c == '.';
         });
}

// Legacy revisions require object schemas at the root of both schemas
// (https://modelcontextprotocol.io/specification/2025-11-25/schema#tool).
absl::Status CheckObjectSchema(absl::string_view tool_name,
                               absl::string_view which,
                               const nlohmann::json& schema) {
  const absl::StatusOr<std::string> type = JsonView(schema).GetString("type");
  if (!type.ok() || *type != "object") {
    return absl::InvalidArgumentError(
        absl::StrCat("tool ", tool_name, ": ", which,
                     " must be a JSON Schema with \"type\": \"object\""));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<McpToolRegistry> McpToolRegistry::FromTools(
    std::vector<std::unique_ptr<McpTool>> tools) {
  absl::flat_hash_set<absl::string_view> names;
  for (const std::unique_ptr<McpTool>& tool : tools) {
    CHECK(tool != nullptr);
    const absl::string_view name = tool->name();
    if (!IsValidToolName(name)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "invalid MCP tool name \"", absl::CHexEscape(name), "\""));
    }
    if (!names.insert(name).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("duplicate MCP tool name \"", name, "\""));
    }
    ABSL_RETURN_IF_ERROR(
        CheckObjectSchema(name, "input schema", tool->input_schema()));
    const nlohmann::json output_schema = tool->output_schema();
    if (!output_schema.is_null()) {
      ABSL_RETURN_IF_ERROR(
          CheckObjectSchema(name, "output schema", output_schema));
    }
  }
  return McpToolRegistry(std::move(tools));
}

const McpTool* absl_nullable McpToolRegistry::Find(
    absl::string_view name) const {
  const auto it = absl::c_find_if(
      tools_, [name](const auto& tool) { return tool->name() == name; });
  return it == tools_.end() ? nullptr : it->get();
}

}  // namespace ferret

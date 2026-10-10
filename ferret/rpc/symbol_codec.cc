// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/rpc/symbol_codec.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "nlohmann/json.hpp"

#include "ferret/rpc/json_codec.h"
#include "ferret/symbols/symbol_kind.h"
#include "ferret/workspace/workspace.h"

namespace ferret {
namespace {

absl::StatusOr<std::string> ReadHex(const JsonView& wire,
                                    absl::string_view field) {
  ABSL_ASSIGN_OR_RETURN(std::string hex, wire.GetString(field));
  if (hex.size() % 2 != 0) return absl::InvalidArgumentError("odd hex length");
  std::string bytes;
  if (!absl::HexStringToBytes(hex, &bytes)) {
    return absl::InvalidArgumentError("invalid hex string");
  }
  return bytes;
}

absl::StatusOr<SymbolKind> ReadKind(const JsonView& wire) {
  ABSL_ASSIGN_OR_RETURN(int64_t kind, wire.AsInt64());
  if (kind < 0 || kind > static_cast<int64_t>(SymbolKind::kVariable)) {
    return absl::InvalidArgumentError("invalid symbol kind");
  }
  return static_cast<SymbolKind>(kind);
}

}  // namespace

nlohmann::json EncodeFindSymbolRequest(const FindSymbolRequest& request) {
  nlohmann::json kinds = nlohmann::json::array();
  for (SymbolKind kind : request.kinds) kinds.push_back(static_cast<int>(kind));
  return {{"name_hex", absl::BytesToHexString(request.name)},
          {"case_insensitive", request.case_insensitive},
          {"kinds", std::move(kinds)},
          {"definitions_only", request.definitions_only},
          {"path_glob_hex", absl::BytesToHexString(request.path_glob)},
          {"max_results", request.max_results}};
}

absl::StatusOr<FindSymbolRequest> DecodeFindSymbolRequest(
    const JsonView& wire) {
  FindSymbolRequest request;
  ABSL_ASSIGN_OR_RETURN(request.name, ReadHex(wire, "name_hex"));
  ABSL_ASSIGN_OR_RETURN(request.case_insensitive,
                        wire.GetBool("case_insensitive"));
  ABSL_ASSIGN_OR_RETURN(JsonView kinds, wire.GetArray("kinds"));
  for (size_t i = 0; i < kinds.size(); ++i) {
    ABSL_ASSIGN_OR_RETURN(SymbolKind kind, ReadKind(kinds.Element(i)));
    request.kinds.push_back(kind);
  }
  ABSL_ASSIGN_OR_RETURN(request.definitions_only,
                        wire.GetBool("definitions_only"));
  ABSL_ASSIGN_OR_RETURN(request.path_glob, ReadHex(wire, "path_glob_hex"));
  ABSL_ASSIGN_OR_RETURN(request.max_results, wire.GetInt64("max_results"));
  return request;
}

nlohmann::json EncodeFindSymbolResponse(const FindSymbolResponse& response) {
  nlohmann::json occurrences = nlohmann::json::array();
  for (const SymbolLocation& item : response.occurrences) {
    occurrences.push_back(
        {{"path_hex", absl::BytesToHexString(item.path)},
         {"line", item.line},
         {"kind", static_cast<int>(item.kind)},
         {"is_definition", item.is_definition},
         {"source_line_hex", absl::BytesToHexString(item.source_line)}});
  }
  return {{"occurrences", std::move(occurrences)},
          {"total_count", response.total_count},
          {"pending_files", response.pending_files},
          {"stale", response.stale},
          {"generation", response.generation}};
}

absl::StatusOr<FindSymbolResponse> DecodeFindSymbolResponse(
    const JsonView& wire) {
  FindSymbolResponse response;
  ABSL_ASSIGN_OR_RETURN(JsonView occurrences, wire.GetArray("occurrences"));
  for (size_t i = 0; i < occurrences.size(); ++i) {
    JsonView item = occurrences.Element(i);
    SymbolLocation location;
    ABSL_ASSIGN_OR_RETURN(location.path, ReadHex(item, "path_hex"));
    ABSL_ASSIGN_OR_RETURN(uint64_t line, item.GetUint64("line"));
    if (line == 0 || line > UINT32_MAX) {
      return absl::InvalidArgumentError("invalid symbol line");
    }
    location.line = static_cast<uint32_t>(line);
    ABSL_ASSIGN_OR_RETURN(int64_t kind, item.GetInt64("kind"));
    if (kind < 0 || kind > static_cast<int64_t>(SymbolKind::kVariable)) {
      return absl::InvalidArgumentError("invalid symbol kind");
    }
    location.kind = static_cast<SymbolKind>(kind);
    ABSL_ASSIGN_OR_RETURN(location.is_definition,
                          item.GetBool("is_definition"));
    ABSL_ASSIGN_OR_RETURN(location.source_line,
                          ReadHex(item, "source_line_hex"));
    response.occurrences.push_back(std::move(location));
  }
  ABSL_ASSIGN_OR_RETURN(response.total_count, wire.GetInt64("total_count"));
  ABSL_ASSIGN_OR_RETURN(response.pending_files, wire.GetInt64("pending_files"));
  ABSL_ASSIGN_OR_RETURN(response.stale, wire.GetBool("stale"));
  ABSL_ASSIGN_OR_RETURN(response.generation, wire.GetUint64("generation"));
  if (response.total_count <
          static_cast<int64_t>(response.occurrences.size()) ||
      response.pending_files < 0) {
    return absl::InvalidArgumentError("invalid symbol counts");
  }
  return response;
}

}  // namespace ferret

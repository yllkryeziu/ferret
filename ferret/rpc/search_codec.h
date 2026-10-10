// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// JSON wire codecs for the daemon's search and status values. Byte strings
// use hex rather than JSON strings so invalid UTF-8 survives a round trip.
#ifndef FERRET_RPC_SEARCH_CODEC_H_
#define FERRET_RPC_SEARCH_CODEC_H_

#include <vector>

#include "absl/status/statusor.h"
#include "nlohmann/json.hpp"

#include "ferret/engine/engine.h"
#include "ferret/rpc/json_codec.h"

namespace ferret {

nlohmann::json EncodeSearchRequest(const SearchRequest& value);
absl::StatusOr<SearchRequest> DecodeSearchRequest(const JsonView& value);
nlohmann::json EncodeSearchConsistency(const SearchConsistency& value);
absl::StatusOr<SearchConsistency> DecodeSearchConsistency(
    const JsonView& value);
nlohmann::json EncodeWorkspaceSearchResult(const WorkspaceSearchResult& value);
absl::StatusOr<WorkspaceSearchResult> DecodeWorkspaceSearchResult(
    const JsonView& value);
nlohmann::json EncodeWorktreeIndexStatus(const WorktreeIndexStatus& value);
absl::StatusOr<WorktreeIndexStatus> DecodeWorktreeIndexStatus(
    const JsonView& value);

}  // namespace ferret
#endif  // FERRET_RPC_SEARCH_CODEC_H_

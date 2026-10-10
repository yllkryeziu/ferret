// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#ifndef FERRET_RPC_SYMBOL_CODEC_H_
#define FERRET_RPC_SYMBOL_CODEC_H_

#include "absl/status/statusor.h"
#include "nlohmann/json.hpp"

#include "ferret/rpc/json_codec.h"
#include "ferret/workspace/workspace.h"

namespace ferret {

nlohmann::json EncodeFindSymbolRequest(const FindSymbolRequest& request);
absl::StatusOr<FindSymbolRequest> DecodeFindSymbolRequest(const JsonView& wire);
nlohmann::json EncodeFindSymbolResponse(const FindSymbolResponse& response);
absl::StatusOr<FindSymbolResponse> DecodeFindSymbolResponse(
    const JsonView& wire);

}  // namespace ferret
#endif  // FERRET_RPC_SYMBOL_CODEC_H_

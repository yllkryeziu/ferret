// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FERRET_SEARCH_STRUCTURAL_RESPONSE_H_
#define FERRET_SEARCH_STRUCTURAL_RESPONSE_H_

#include <string>

#include "ferret/search/formatter.h"
#include "ferret/search/search_types.h"
#include "ferret/workspace/workspace.h"

namespace ferret {

// Formats definitions and already verified text results. Ranking never opens
// another file; the header reports when unverified files may change it.
std::string FormatStructuralResponse(const SearchResponse& search,
                                     const FindSymbolResponse& definitions,
                                     const SearchRequest& request,
                                     const FormatOptions& options);

}  // namespace ferret
#endif  // FERRET_SEARCH_STRUCTURAL_RESPONSE_H_

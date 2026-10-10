// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FERRET_SEARCH_ROUTER_H_
#define FERRET_SEARCH_ROUTER_H_

#include <cstdint>

#include "absl/strings/string_view.h"

#include "ferret/search/search_types.h"

namespace ferret {

enum class SearchRoute { kText, kStructural };

struct RouterOptions {
  int64_t route_min_files = 20;
  double theta = 0.7;
};

// Regexes are never identifier-like. The literal must be a single UTF-8
// identifier token; other literals always take the text route.
bool IsIdentifierQuery(const SearchRequest& request);

// Chooses the response without touching an index or a search backend.
SearchRoute DecideRoute(bool identifier_like, int64_t matching_files,
                        double token_precision, bool definitions_exist,
                        const RouterOptions& options = {});

}  // namespace ferret
#endif  // FERRET_SEARCH_ROUTER_H_

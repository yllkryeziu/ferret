// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
#include "ferret/search/router.h"

#include <cstdint>

#include "ferret/search/search_types.h"
#include "ferret/text/utf8.h"

namespace ferret {
namespace {
bool IdentifierStart(unsigned char ch) {
  return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_' ||
         ch >= 0x80;
}
bool IdentifierContinue(unsigned char ch) {
  return IdentifierStart(ch) || (ch >= '0' && ch <= '9');
}
}  // namespace

bool IsIdentifierQuery(const SearchRequest& request) {
  if (request.regex || request.pattern.empty() ||
      !IdentifierStart(request.pattern.front())) {
    return false;
  }
  for (unsigned char ch : request.pattern) {
    if (!IdentifierContinue(ch)) {
      return false;
    }
  }
  return IsValidUtf8(request.pattern);
}

SearchRoute DecideRoute(bool identifier_like, int64_t matching_files,
                        double token_precision, bool definitions_exist,
                        const RouterOptions& options) {
  return identifier_like && matching_files > options.route_min_files &&
                 (token_precision < options.theta || definitions_exist)
             ? SearchRoute::kStructural
             : SearchRoute::kText;
}

}  // namespace ferret

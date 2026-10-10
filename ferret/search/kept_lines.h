// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
#ifndef FERRET_SEARCH_KEPT_LINES_H_
#define FERRET_SEARCH_KEPT_LINES_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "ferret/search/search_types.h"

namespace ferret {

// A matching line kept for the response.
struct KeptLine {
  int64_t number = 0;
  // The line is [begin, end) in its document's content.
  size_t begin = 0;
  size_t end = 0;
  // The first match, relative to `begin`.
  size_t match_begin = 0;
  size_t match_end = 0;
  MatchClass match_class = MatchClass::kCodeIdentifier;
};

std::vector<ResultLine> WithContext(absl::string_view content,
                                    int64_t num_lines,
                                    absl::Span<const KeptLine> kept,
                                    int64_t context_lines);

}  // namespace ferret

#endif  // FERRET_SEARCH_KEPT_LINES_H_

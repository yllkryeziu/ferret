// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "absl/strings/string_view.h"

#include "ferret/text/internal/line_counter_kernels.h"

namespace ferret::text_internal {

int64_t CountNewlinesScalar(absl::string_view text) {
  int64_t count = 0;
  for (const char c : text) {
    if (c == '\n') ++count;
  }
  return count;
}

}  // namespace ferret::text_internal

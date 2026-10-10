// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"

#include "ferret/text/internal/literal_finder_kernels.h"

namespace ferret::text_internal {

size_t FindLiteralScalar(const LiteralNeedle& needle,
                         absl::string_view haystack, size_t from) {
  const size_t length = needle.bytes.size();
  DCHECK_GT(length, 0u);
  if (from > haystack.size() || haystack.size() - from < length) {
    return absl::string_view::npos;
  }
  for (size_t offset = from; offset <= haystack.size() - length; ++offset) {
    const char* const candidate = haystack.data() + offset;
    if (needle.fold_ascii ? LiteralMatchesAt<true>(needle.bytes, candidate)
                          : LiteralMatchesAt<false>(needle.bytes, candidate)) {
      return offset;
    }
  }
  return absl::string_view::npos;
}

}  // namespace ferret::text_internal

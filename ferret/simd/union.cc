// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/simd/union.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "absl/base/nullability.h"
#include "absl/types/span.h"

namespace ferret {

size_t Union(absl::Span<const uint32_t> a, absl::Span<const uint32_t> b,
             uint32_t* absl_nonnull out) {
#if defined(__aarch64__)
  return UnionNeon(a, b, out);
#else
  return UnionScalar(a, b, out);
#endif
}

size_t UnionScalar(absl::Span<const uint32_t> a, absl::Span<const uint32_t> b,
                   uint32_t* absl_nonnull out) {
  size_t i = 0;
  size_t j = 0;
  size_t count = 0;
  while (i < a.size() && j < b.size()) {
    const uint32_t x = a[i];
    const uint32_t y = b[j];
    out[count++] = std::min(x, y);
    i += x <= y;
    j += y <= x;
  }
  // At most one input has elements left.
  std::copy(a.begin() + i, a.end(), out + count);
  count += a.size() - i;
  std::copy(b.begin() + j, b.end(), out + count);
  count += b.size() - j;
  return count;
}

}  // namespace ferret

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/simd/intersect.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/types/span.h"

#include "ferret/base/ablation.h"
#include "ferret/simd/dispatch.h"

namespace ferret {

size_t Intersect(absl::Span<const uint32_t> a, absl::Span<const uint32_t> b,
                 uint32_t* absl_nonnull out) {
  if (IntersectShouldGallop(a.size(), b.size())) {
    return IntersectGalloping(a, b, out);
  }
#if defined(__aarch64__)
  if (AblationOn(Ablation::scalar_intersect)) [[unlikely]] {
    ++ThisThreadAblationCounts().scalar_merges;
    return IntersectScalar(a, b, out);
  }
  return IntersectNeon(a, b, out);
#else
  return IntersectScalar(a, b, out);
#endif
}

size_t IntersectScalar(absl::Span<const uint32_t> a,
                       absl::Span<const uint32_t> b,
                       uint32_t* absl_nonnull out) {
  // Stores only below min(|a|, |b|): the count reaches that bound only once
  // one input is exhausted. IntersectNeon relies on this for its tail.
  size_t i = 0;
  size_t j = 0;
  size_t count = 0;
  while (i < a.size() && j < b.size()) {
    const uint32_t x = a[i];
    const uint32_t y = b[j];
    out[count] = x;
    count += x == y;
    i += x <= y;
    j += y <= x;
  }
  return count;
}

size_t IntersectGalloping(absl::Span<const uint32_t> a,
                          absl::Span<const uint32_t> b,
                          uint32_t* absl_nonnull out) {
  if (a.size() > b.size()) std::swap(a, b);
  size_t count = 0;
  // Every element of b[0, lo) is smaller than the next element of `a`.
  size_t lo = 0;
  for (const uint32_t x : a) {
    // Probe b[lo], b[lo + 1], b[lo + 3], b[lo + 7], ... until one is >= x,
    // then binary search the last gap.
    size_t hi = lo;
    size_t step = 1;
    while (hi < b.size() && b[hi] < x) {
      lo = hi + 1;
      hi += step;
      step *= 2;
    }
    hi = std::min(hi, b.size());
    lo = static_cast<size_t>(
        std::lower_bound(b.begin() + lo, b.begin() + hi, x) - b.begin());
    if (lo == b.size()) break;
    out[count] = x;
    count += b[lo] == x;
  }
  return count;
}

}  // namespace ferret

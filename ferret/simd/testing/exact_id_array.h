// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Exactly sized id buffers for kernel tests.
//
// The kernel contract bounds every read by the input sizes and every write by
// the output capacity (ferret/simd/intersect.h). AddressSanitizer checks such
// bounds only at the edge of a heap allocation, and a std::vector may have
// spare capacity past its size, so kernel tests keep inputs and outputs in
// ExactIdArrays instead.

#ifndef FERRET_SIMD_TESTING_EXACT_ID_ARRAY_H_
#define FERRET_SIMD_TESTING_EXACT_ID_ARRAY_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/types/span.h"

namespace ferret_testing {

// A heap allocation of exactly size() ids. Thread-compatible.
class ExactIdArray final {
 public:
  // `size` zeros.
  explicit ExactIdArray(size_t size);
  // A copy of `ids`.
  explicit ExactIdArray(absl::Span<const uint32_t> ids);

  ExactIdArray(const ExactIdArray&) = delete;
  ExactIdArray& operator=(const ExactIdArray&) = delete;
  // Both leave `other` empty, where defaulted moves would leave it with a null
  // pointer and a nonzero size.
  ExactIdArray(ExactIdArray&& other) noexcept;
  ExactIdArray& operator=(ExactIdArray&& other) noexcept;

  size_t size() const { return size_; }

  // Valid while *this is alive.
  uint32_t* data() ABSL_ATTRIBUTE_LIFETIME_BOUND { return ids_.get(); }
  absl::Span<const uint32_t> span() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return absl::MakeConstSpan(ids_.get(), size_);
  }

 private:
  std::unique_ptr<uint32_t[]> ids_;
  size_t size_;
};

// Returns every subset of {0, 1, ..., universe_size - 1}; element m holds the
// ids of the set bits of m. Requires universe_size <= 20.
std::vector<ExactIdArray> AllSubsets(int universe_size);

}  // namespace ferret_testing

#endif  // FERRET_SIMD_TESTING_EXACT_ID_ARRAY_H_

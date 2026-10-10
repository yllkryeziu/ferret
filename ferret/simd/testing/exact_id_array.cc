// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/simd/testing/exact_id_array.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"

namespace ferret_testing {

ExactIdArray::ExactIdArray(size_t size)
    : ids_(std::make_unique<uint32_t[]>(size)), size_(size) {}

ExactIdArray::ExactIdArray(absl::Span<const uint32_t> ids)
    : ExactIdArray(ids.size()) {
  std::copy(ids.begin(), ids.end(), ids_.get());
}

ExactIdArray::ExactIdArray(ExactIdArray&& other) noexcept
    : ids_(std::move(other.ids_)), size_(std::exchange(other.size_, 0)) {}

ExactIdArray& ExactIdArray::operator=(ExactIdArray&& other) noexcept {
  ids_ = std::move(other.ids_);
  size_ = std::exchange(other.size_, 0);
  return *this;
}

std::vector<ExactIdArray> AllSubsets(int universe_size) {
  CHECK_GE(universe_size, 0);
  CHECK_LE(universe_size, 20);
  const uint32_t num_subsets = uint32_t{1} << universe_size;
  std::vector<ExactIdArray> subsets;
  subsets.reserve(num_subsets);
  std::vector<uint32_t> ids;
  for (uint32_t mask = 0; mask < num_subsets; ++mask) {
    ids.clear();
    for (int id = 0; id < universe_size; ++id) {
      if ((mask & (uint32_t{1} << id)) != 0) {
        ids.push_back(static_cast<uint32_t>(id));
      }
    }
    subsets.emplace_back(ids);
  }
  return subsets;
}

}  // namespace ferret_testing

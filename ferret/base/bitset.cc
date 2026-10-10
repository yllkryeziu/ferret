// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/base/bitset.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>

#include "absl/functional/function_ref.h"

namespace ferret {

void Bitset::Resize(uint32_t size) {
  size_ = size;
  words_.resize(WordCount(size), 0);
  ClearTail();
}

int64_t Bitset::Count() const {
  int64_t count = 0;
  for (const uint64_t word : words_) count += std::popcount(word);
  return count;
}

void Bitset::ForEachSetBit(absl::FunctionRef<void(uint32_t)> fn) const {
  for (size_t i = 0; i < words_.size(); ++i) {
    const uint32_t base = static_cast<uint32_t>(i * 64);
    for (uint64_t word = words_[i]; word != 0; word &= word - 1) {
      fn(base + static_cast<uint32_t>(std::countr_zero(word)));
    }
  }
}

void Bitset::IntersectWith(const Bitset& other) {
  const size_t common = std::min(words_.size(), other.words_.size());
  for (size_t i = 0; i < common; ++i) words_[i] &= other.words_[i];
  for (size_t i = common; i < words_.size(); ++i) words_[i] = 0;
}

void Bitset::UnionWith(const Bitset& other) {
  const size_t common = std::min(words_.size(), other.words_.size());
  for (size_t i = 0; i < common; ++i) words_[i] |= other.words_[i];
  ClearTail();
}

void Bitset::ClearTail() {
  if (size_ % 64 != 0) words_.back() &= Bit(size_) - 1;
}

}  // namespace ferret

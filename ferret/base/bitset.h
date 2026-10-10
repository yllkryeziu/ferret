// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Bitset: a dense set of uint32_t ids.
//
// Path-glob caches use it over PathIds. Worktree views do not: their
// liveness lives in ViewState's chunked copy-on-write directory, which also
// lists a blob's paths.
//
// Invariant: bits at positions >= size() are zero, so Count() and equality
// work on whole words.

#ifndef FERRET_BASE_BITSET_H_
#define FERRET_BASE_BITSET_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/log/check.h"

namespace ferret {

// A regular value type. Thread-compatible.
class Bitset final {
 public:
  // A set over no ids.
  Bitset() = default;
  // A set over ids [0, size), all clear.
  explicit Bitset(uint32_t size) : size_(size), words_(WordCount(size)) {}

  Bitset(const Bitset&) = default;
  Bitset& operator=(const Bitset&) = default;
  Bitset(Bitset&&) = default;
  Bitset& operator=(Bitset&&) = default;

  // The id bound: the set holds ids in [0, size()).
  uint32_t size() const { return size_; }

  // Changes the id bound. Growing adds clear ids; shrinking drops the ids at
  // or above `size`, so growing back later exposes clear bits.
  void Resize(uint32_t size);

  // Set, Clear and Test require id < size() (DCHECKed). O(1).
  void Set(uint32_t id) {
    DCHECK_LT(id, size_);
    words_[id >> 6] |= Bit(id);
  }
  void Clear(uint32_t id) {
    DCHECK_LT(id, size_);
    words_[id >> 6] &= ~Bit(id);
  }
  bool Test(uint32_t id) const {
    DCHECK_LT(id, size_);
    return (words_[id >> 6] & Bit(id)) != 0;
  }

  // Returns the number of ids in the set. O(size() / 64).
  int64_t Count() const;

  // Calls `fn(id)` for every id in the set, in increasing order.
  // O(size() / 64 + Count()).
  void ForEachSetBit(absl::FunctionRef<void(uint32_t)> fn) const;

  // Keeps only the ids that are also in `other`. `other` may have a different
  // size(); it holds no ids at or above its own size(). size() is unchanged.
  void IntersectWith(const Bitset& other);

  // Adds the ids of `other` that are below size(); size() is unchanged.
  void UnionWith(const Bitset& other);

  // Equal iff both have the same size() and the same ids.
  friend bool operator==(const Bitset&, const Bitset&) = default;

 private:
  static size_t WordCount(uint32_t size) {
    return (static_cast<size_t>(size) + 63) / 64;
  }
  static uint64_t Bit(uint32_t id) { return uint64_t{1} << (id & 63); }

  // Restores the invariant after words_ may have gained bits >= size_.
  void ClearTail();

  uint32_t size_ = 0;
  std::vector<uint64_t> words_;
};

}  // namespace ferret

#endif  // FERRET_BASE_BITSET_H_

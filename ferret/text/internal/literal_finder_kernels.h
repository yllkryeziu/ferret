// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The search kernels behind LiteralFinder (literal_finder.h).
//
// Package-private so that tests and benchmarks can compare the NEON kernel
// with the scalar reference. Both have LiteralFinder::Find's contract for a
// non-empty needle.

#ifndef FERRET_TEXT_INTERNAL_LITERAL_FINDER_KERNELS_H_
#define FERRET_TEXT_INTERNAL_LITERAL_FINDER_KERNELS_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "absl/strings/string_view.h"

#include "ferret/text/fold.h"

namespace ferret::text_internal {

// A needle as LiteralFinder prepares it.
struct LiteralNeedle {
  // Non-empty. Already ASCII-folded if fold_ascii is set.
  absl::string_view bytes;
  // Compare haystack bytes through FoldAsciiByte.
  bool fold_ascii = false;
  // Offsets into `bytes` of the two bytes the vector filter compares. Both
  // are less than bytes.size(); they are equal only for a one-byte needle.
  size_t rare1_offset = 0;
  size_t rare2_offset = 0;
};

// Returns true if `needle` occurs at `candidate`, which must have at least
// needle.size() readable bytes.
template <bool kFoldAscii>
bool LiteralMatchesAt(absl::string_view needle, const char* candidate) {
  if constexpr (kFoldAscii) {
    for (size_t i = 0; i < needle.size(); ++i) {
      if (FoldAsciiByte(static_cast<uint8_t>(candidate[i])) !=
          static_cast<uint8_t>(needle[i])) {
        return false;
      }
    }
    return true;
  } else {
    return std::memcmp(candidate, needle.data(), needle.size()) == 0;
  }
}

// The portable reference: tries every offset in turn.
[[nodiscard]] size_t FindLiteralScalar(const LiteralNeedle& needle,
                                       absl::string_view haystack, size_t from);

#if defined(__aarch64__)
// Filters 64 candidate offsets per iteration on the two rare bytes.
[[nodiscard]] size_t FindLiteralNeon(const LiteralNeedle& needle,
                                     absl::string_view haystack, size_t from);
#endif

}  // namespace ferret::text_internal

#endif  // FERRET_TEXT_INTERNAL_LITERAL_FINDER_KERNELS_H_

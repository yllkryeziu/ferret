// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// LiteralFinder: the substring search behind the verifier's literal prefilter.
//
// When a pattern has a required literal, the verifier finds the literal,
// expands each hit to its line, and runs RE2 on that line only, so this search
// sets the verification rate. macOS memmem runs at 1.1-1.8 GB/s on source
// code, which would put p99 out of reach.
//
// Algorithm: the SIMD filter of W. Muła, "SIMD-friendly algorithms for
// substring searching" (2016), keyed on rare bytes as in Rust's memchr crate.
// The two needle bytes that are rarest in source code, per a byte-frequency
// table measured on Linux and Chromium, are compared at their offsets against
// 16 candidate positions per NEON instruction, and only positions where both
// match are compared in full. On source code few positions survive, so the
// scan runs near load bandwidth. The worst case, a haystack that keeps
// matching both filter bytes, is O(|haystack| * |needle|).
//
// Case-insensitive mode is ASCII-only: it compares bytes after mapping A-Z to
// a-z. RE2's (?i) also matches U+212A KELVIN SIGN for 'k' and U+017F LATIN
// SMALL LETTER LONG S for 's' (see fold.h), so a line can match a (?i) pattern
// while containing no ASCII-folded occurrence of a literal with 'k' or 's' in
// it. A caller that uses this mode as a prefilter for a case-insensitive
// pattern must therefore choose a needle without 'k' and 's'.

#ifndef FERRET_TEXT_LITERAL_FINDER_H_
#define FERRET_TEXT_LITERAL_FINDER_H_

#include <cstddef>
#include <string>

#include "absl/strings/string_view.h"

namespace ferret {

// Searches for one fixed needle. Immutable after construction. Thread-safe.
class LiteralFinder final {
 public:
  enum class Mode {
    // Bytes must be equal.
    kExact,
    // A-Z equals a-z; every other byte, including every non-ASCII byte, must
    // be equal.
    kAsciiCaseInsensitive,
  };

  // `needle` may be empty and may contain any bytes. O(|needle|).
  LiteralFinder(absl::string_view needle, Mode mode);

  LiteralFinder(const LiteralFinder&) = default;
  LiteralFinder& operator=(const LiteralFinder&) = default;
  LiteralFinder(LiteralFinder&&) = default;
  LiteralFinder& operator=(LiteralFinder&&) = default;

  // Returns the offset of the first occurrence of the needle in `haystack`
  // that starts at or after `from`, or absl::string_view::npos if there is
  // none. Like absl::string_view::find, an empty needle occurs at every offset
  // up to haystack.size(). Never reads outside `haystack`.
  size_t Find(absl::string_view haystack, size_t from = 0) const;

  // The needle's length.
  size_t size() const { return needle_.size(); }

 private:
  // ASCII-folded in kAsciiCaseInsensitive mode.
  std::string needle_;
  Mode mode_;
  // Offsets into needle_ of the two bytes the vector filter compares.
  size_t rare1_offset_ = 0;
  size_t rare2_offset_ = 0;
};

}  // namespace ferret

#endif  // FERRET_TEXT_LITERAL_FINDER_H_

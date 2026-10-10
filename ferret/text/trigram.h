// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The Trigram type and the one routine allowed to pack bytes into it.
//
// A trigram is three consecutive bytes of *folded* text (see fold.h), packed
// big-endian into the low 24 bits of a uint32_t: (b0 << 16) | (b1 << 8) | b2.
// Every producer of trigrams (the index-side extractor and the query-side
// planner) must pack through MakeTrigram()/TrigramAt(). On arm64 `char` is
// signed, so packing a byte >= 0x80 straight from a `char` sign-extends it
// and silently produces a different trigram; that would be a false negative,
// not a crash.

#ifndef FERRET_TEXT_TRIGRAM_H_
#define FERRET_TEXT_TRIGRAM_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"

namespace ferret {

// A 24-bit packed byte trigram. The top 8 bits are always zero, which lets
// on-disk formats borrow them for flags.
using Trigram = uint32_t;

inline constexpr Trigram kTrigramMask = 0x00FFFFFF;

// Packs three bytes. This and TrigramAt() are the only packing routines.
constexpr Trigram MakeTrigram(uint8_t b0, uint8_t b1, uint8_t b2) {
  return (Trigram{b0} << 16) | (Trigram{b1} << 8) | Trigram{b2};
}

// Returns the trigram starting at byte `pos` of `text`.
// Requires pos + 3 <= text.size().
inline Trigram TrigramAt(absl::string_view text, size_t pos) {
  DCHECK_LE(pos + 3, text.size());
  return MakeTrigram(static_cast<uint8_t>(text[pos]),
                     static_cast<uint8_t>(text[pos + 1]),
                     static_cast<uint8_t>(text[pos + 2]));
}

// Returns the i-th byte (0, 1 or 2) of `trigram`.
constexpr uint8_t TrigramByte(Trigram trigram, int i) {
  return static_cast<uint8_t>(trigram >> (8 * (2 - i)));
}

// Returns true if any byte of `trigram` is '\n'. Such trigrams are never
// indexed because search is line-oriented.
constexpr bool TrigramHasNewline(Trigram trigram) {
  return TrigramByte(trigram, 0) == '\n' || TrigramByte(trigram, 1) == '\n' ||
         TrigramByte(trigram, 2) == '\n';
}

// Returns a C-escaped, quoted rendering such as "abc" or "a\xe2\x84" for
// logs, test failures and `ferret plan --explain`.
std::string TrigramToString(Trigram trigram);

}  // namespace ferret

#endif  // FERRET_TEXT_TRIGRAM_H_

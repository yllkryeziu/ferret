// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The case-folding transducer F shared by the indexer and the query planner.
//
// F maps A-Z to a-z, the UTF-8 sequence E2 84 AA (U+212A KELVIN SIGN) to 'k',
// and C5 BF (U+017F LATIN SMALL LETTER LONG S) to 's'; every other byte is
// unchanged. Those two runes are the only non-ASCII members of any ASCII
// letter's Unicode simple-fold orbit, which is what RE2's (?i) uses, so
// folding them at index time keeps (?i) planning both sound and cheap. F is
// not length-preserving, which is harmless because trigram sets carry no
// positions; content is always stored unfolded.
//
// Soundness relies on this property: F only rewrites whole valid UTF-8
// sequences, and literal material in a pattern is always a sequence of whole
// runes, so for any occurrence of e in t, F(e) is a substring of F(t).
//
// Changing F changes every trigram in every segment: bump the segment format
// version if you ever do.

#ifndef FERRET_TEXT_FOLD_H_
#define FERRET_TEXT_FOLD_H_

#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"

namespace ferret {

// Folds one ASCII byte (A-Z -> a-z). Identity for every other byte.
constexpr uint8_t FoldAsciiByte(uint8_t b) {
  return (b >= 'A' && b <= 'Z') ? static_cast<uint8_t>(b + ('a' - 'A')) : b;
}

// Appends F(in) to `out`.
void AppendFolded(absl::string_view in, std::string& out);

// Returns F(in).
std::string Fold(absl::string_view in);

}  // namespace ferret

#endif  // FERRET_TEXT_FOLD_H_

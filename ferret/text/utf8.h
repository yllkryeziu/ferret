// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// UTF-8 decoding, sanitizing, and display truncation for result shaping.
//
// Ferret indexes and verifies raw bytes, so content need not be UTF-8:
// Chromium has 792 indexable files that are not. Everything that leaves the
// process as JSON must be valid UTF-8, though (nlohmann's strict dump() aborts
// under -fno-exceptions), so the formatter sanitizes at that boundary and cuts
// long lines down to a window around the match.
//
// Well-formedness follows Unicode 15.0 §3.9, Table 3-7: no overlong forms, no
// surrogates, nothing above U+10FFFF. Each maximal subpart of an ill-formed
// sequence (the longest prefix of some well-formed sequence, or else a single
// byte) becomes one U+FFFD, the practice recommended by §3.9 and required by
// the WHATWG Encoding Standard. Every function here is pure and thread-safe.

#ifndef FERRET_TEXT_UTF8_H_
#define FERRET_TEXT_UTF8_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"

namespace ferret {

// The result of decoding the sequence at the start of a text.
struct Utf8Sequence {
  // The code point, or -1 if the sequence is ill-formed.
  int32_t code_point = -1;
  // The bytes it spans, in [1, 4]: a whole well-formed sequence, or the
  // maximal subpart of an ill-formed one.
  size_t length = 1;
};

// Decodes the sequence that starts at text[0]. Requires !text.empty().
Utf8Sequence DecodeUtf8(absl::string_view text);

// Returns true if `text` is well-formed UTF-8. O(|text|).
bool IsValidUtf8(absl::string_view text);

// Appends `text` to `out` with each maximal subpart of an ill-formed sequence
// replaced by U+FFFD. The result is `text` itself if it is valid. O(|text|).
void AppendSanitizedUtf8(absl::string_view text, std::string& out);

// Appends one line of a search result for display: the line without a
// trailing '\r', sanitized as by AppendSanitizedUtf8, in at most `max_bytes`
// bytes. A line that does not fit is cut on sequence boundaries to a window
// around the match text[match_begin, match_end), balanced so the match sits
// in the middle, and "…" (U+2026) marks each cut end; both markers count
// toward `max_bytes`. If the match itself does not fit, the window starts at
// the match. Pass an empty match at 0 to keep the start of a line. Requires
// match_begin <= match_end <= line.size() and max_bytes >= 6.
// O(min(|line|, max_bytes)).
void AppendDisplayLine(absl::string_view line, size_t match_begin,
                       size_t match_end, size_t max_bytes, std::string& out);

}  // namespace ferret

#endif  // FERRET_TEXT_UTF8_H_

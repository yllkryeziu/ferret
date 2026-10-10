// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Newline counting and line-boundary helpers for the chunker and the verifier.
//
// Search is line-oriented. A reported match's line number is its chunk's
// first_line plus the number of '\n' bytes between the chunk start and the
// match, and the verifier runs RE2 on the line around each literal hit, so both
// operations sit on the query path. CountNewlines is a NEON kernel on arm64;
// line_counter_scalar.cc holds the portable reference.
//
// A line is a maximal run of bytes without '\n'. The '\n' that terminates a
// line is not part of its text. Every function here is pure and thread-safe.

#ifndef FERRET_TEXT_LINE_COUNTER_H_
#define FERRET_TEXT_LINE_COUNTER_H_

#include <cstddef>
#include <cstdint>

#include "absl/strings/string_view.h"

namespace ferret {

// Returns the number of '\n' bytes in `text`. O(|text|).
[[nodiscard]] int64_t CountNewlines(absl::string_view text);

// Returns the offset of the first byte of the line containing offset `pos`:
// one past the last '\n' before `pos`, or 0 if there is none.
// Requires pos <= text.size(). O(pos - result).
size_t LineBegin(absl::string_view text, size_t pos);

// Returns the offset of the '\n' that terminates the line containing offset
// `pos`, or text.size() if that line is unterminated. Returns `pos` itself if
// text[pos] is '\n'. Requires pos <= text.size(). O(result - pos).
size_t LineEnd(absl::string_view text, size_t pos);

}  // namespace ferret

#endif  // FERRET_TEXT_LINE_COUNTER_H_

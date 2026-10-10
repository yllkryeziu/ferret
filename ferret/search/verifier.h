// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Finds the lines of a text that contain a match of a compiled pattern. The
// text starts at a line boundary, like a chunk.
//
// A line matches when RE2 finds a match in the line on its own, as grep
// decides; ^, $, \A and \z all mean the line's start and end. How lines are
// found only affects speed:
//   * A plain literal pattern needs no RE2 at all: the vectorized search
//     finds the match itself (CompiledPattern::exact_literal()).
//   * With a required literal, a vectorized case-insensitive scan finds each
//     line that contains it, and RE2 runs on that line only. The literal is
//     a sound filter because every match contains it
//     (docs/regex_analysis.md).
//   * Otherwise RE2 scans the text for the next match and resumes after its
//     line. With the pattern's multi-line ^ and $, that finds the same lines,
//     unless the pattern has a text anchor; then every line is matched on
//     its own.

#ifndef FERRET_SEARCH_VERIFIER_H_
#define FERRET_SEARCH_VERIFIER_H_

#include <cstddef>
#include <memory>
#include <optional>

#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "re2/re2.h"

#include "ferret/regex/compile.h"
#include "ferret/text/literal_finder.h"

namespace ferret {

// A line with a match, as byte offsets into the searched text.
struct LineMatch {
  // The line is [line_begin, line_end); text[line_end] is '\n' unless the
  // line is the last and unterminated.
  size_t line_begin = 0;
  size_t line_end = 0;
  // The first match in the line.
  size_t match_begin = 0;
  size_t match_end = 0;

  friend bool operator==(const LineMatch&, const LineMatch&) = default;
};

// Owns a matcher copy, since RE2's DFA cache is per object and contended if
// shared. Thread-compatible: one verifier per thread.
class LineVerifier final {
 public:
  // `pattern` must outlive the verifier.
  explicit LineVerifier(const CompiledPattern& pattern);

  LineVerifier(const LineVerifier&) = delete;
  LineVerifier& operator=(const LineVerifier&) = delete;

  // Calls `fn` for each line of `text` that contains a match, in
  // increasing byte offset order, until `fn` returns false. `text` must start
  // at a line boundary. A '\n' at the very end of `text` ends its last line; it
  // does not start an empty one, and an empty text has no lines. O(|text|) plus
  // RE2 time on candidate lines.
  void ForEachMatchingLine(absl::string_view text,
                           absl::FunctionRef<bool(const LineMatch&)> fn);

 private:
  // Matches text[line_begin, line_end) on its own.
  std::optional<LineMatch> MatchLine(absl::string_view text, size_t line_begin,
                                     size_t line_end) const;
  void ScanExact(const LiteralFinder& literal, absl::string_view text,
                 absl::FunctionRef<bool(const LineMatch&)> fn) const;
  void ScanWithLiteral(const LiteralFinder& literal, absl::string_view text,
                       absl::FunctionRef<bool(const LineMatch&)> fn);
  void ScanLineByLine(absl::string_view text,
                      absl::FunctionRef<bool(const LineMatch&)> fn);
  void ScanWithRe2(absl::string_view text,
                   absl::FunctionRef<bool(const LineMatch&)> fn);

  const std::unique_ptr<RE2> matcher_;
  const bool match_lines_separately_;
  // The pattern's exact literal, when it has one: a hit is the match.
  const LiteralFinder* exact_ = nullptr;
  // Set when the pattern has a required literal.
  std::optional<LiteralFinder> literal_;
};

}  // namespace ferret

#endif  // FERRET_SEARCH_VERIFIER_H_

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/search/verifier.h"

#include <cstddef>
#include <memory>
#include <optional>

#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "re2/re2.h"

#include "ferret/regex/compile.h"
#include "ferret/text/line_counter.h"
#include "ferret/text/literal_finder.h"

namespace ferret {

LineVerifier::LineVerifier(const CompiledPattern& pattern)
    : matcher_(pattern.NewMatcher()),
      match_lines_separately_(pattern.has_text_anchor()),
      exact_(pattern.exact_literal()) {
  if (exact_ == nullptr && !pattern.scan_literal().empty()) {
    literal_.emplace(pattern.scan_literal(),
                     LiteralFinder::Mode::kAsciiCaseInsensitive);
  }
}

void LineVerifier::ForEachMatchingLine(
    absl::string_view text, absl::FunctionRef<bool(const LineMatch&)> fn) {
  if (text.empty()) return;
  if (exact_ != nullptr) {
    ScanExact(*exact_, text, fn);
  } else if (literal_.has_value()) {
    ScanWithLiteral(*literal_, text, fn);
  } else if (match_lines_separately_) {
    ScanLineByLine(text, fn);
  } else {
    ScanWithRe2(text, fn);
  }
}

std::optional<LineMatch> LineVerifier::MatchLine(absl::string_view text,
                                                 size_t line_begin,
                                                 size_t line_end) const {
  const absl::string_view line = text.substr(line_begin, line_end - line_begin);
  absl::string_view match;
  if (!matcher_->Match(line, 0, line.size(), RE2::UNANCHORED, &match, 1)) {
    return std::nullopt;
  }
  const size_t match_begin =
      line_begin + static_cast<size_t>(match.data() - line.data());
  return LineMatch{.line_begin = line_begin,
                   .line_end = line_end,
                   .match_begin = match_begin,
                   .match_end = match_begin + match.size()};
}

void LineVerifier::ScanExact(
    const LiteralFinder& literal, absl::string_view text,
    absl::FunctionRef<bool(const LineMatch&)> fn) const {
  size_t from = 0;
  while (from < text.size()) {
    const size_t hit = literal.Find(text, from);
    if (hit == absl::string_view::npos) return;
    // The needle has no '\n', so the hit lies within one line, and it is the
    // first hit in the line because the scan resumes at the next line.
    const size_t line_end = LineEnd(text, hit);
    if (!fn(LineMatch{.line_begin = LineBegin(text, hit),
                      .line_end = line_end,
                      .match_begin = hit,
                      .match_end = hit + literal.size()})) {
      return;
    }
    from = line_end + 1;
  }
}

void LineVerifier::ScanWithLiteral(
    const LiteralFinder& literal, absl::string_view text,
    absl::FunctionRef<bool(const LineMatch&)> fn) {
  size_t from = 0;
  while (from < text.size()) {
    const size_t hit = literal.Find(text, from);
    if (hit == absl::string_view::npos) return;
    const size_t line_end = LineEnd(text, hit);
    const std::optional<LineMatch> match =
        MatchLine(text, LineBegin(text, hit), line_end);
    if (match.has_value()) {
      if (!fn(*match)) return;
    }
    from = line_end + 1;
  }
}

void LineVerifier::ScanLineByLine(
    absl::string_view text, absl::FunctionRef<bool(const LineMatch&)> fn) {
  for (size_t line_begin = 0; line_begin < text.size();) {
    const size_t line_end = LineEnd(text, line_begin);
    const std::optional<LineMatch> match =
        MatchLine(text, line_begin, line_end);
    if (match.has_value()) {
      if (!fn(*match)) return;
    }
    line_begin = line_end + 1;
  }
}

void LineVerifier::ScanWithRe2(absl::string_view text,
                               absl::FunctionRef<bool(const LineMatch&)> fn) {
  size_t from = 0;
  while (from <= text.size()) {
    // RE2 sees the whole text as context, and without text anchors in the
    // pattern, ^, $ and \b at a line boundary judge the neighboring '\n'
    // exactly as they judge the edge of an isolated line.
    absl::string_view match;
    if (!matcher_->Match(text, from, text.size(), RE2::UNANCHORED, &match, 1)) {
      return;
    }
    const auto match_begin = static_cast<size_t>(match.data() - text.data());
    // An empty match after a final '\n' is on a line that does not exist.
    if (match_begin == text.size() && !text.empty() && text.back() == '\n') {
      return;
    }
    const size_t line_end = LineEnd(text, match_begin);
    if (!fn(LineMatch{.line_begin = LineBegin(text, match_begin),
                      .line_end = line_end,
                      .match_begin = match_begin,
                      .match_end = match_begin + match.size()})) {
      return;
    }
    from = line_end + 1;
  }
}

}  // namespace ferret

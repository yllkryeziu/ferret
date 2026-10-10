// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// CompilePattern: the one way a search pattern becomes a matcher and a plan.
//
// The planner, the verifier, and the tests all compile through this function,
// so the RE2 program that verifies candidates and the parse tree that the
// trigram analysis walks come from the same RE2 object and the same options.
// Search is line-oriented: patterns compile with (?m) and NeverNL, so no
// match ever contains '\n', which is what lets the index skip every trigram
// that contains one.

#ifndef FERRET_REGEX_COMPILE_H_
#define FERRET_REGEX_COMPILE_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "re2/re2.h"

#include "ferret/query/trigram_query.h"
#include "ferret/regex/analyze.h"
#include "ferret/text/literal_finder.h"

namespace ferret {

// Patterns longer than this are rejected: RE2 compile time and plan size grow
// with the pattern, and no agent query needs more.
inline constexpr size_t kMaxPatternBytes = 4096;

struct PatternSpec {
  std::string pattern;
  // Match `pattern` as a literal string rather than a regular expression.
  bool literal = false;
  bool case_sensitive = true;
};

// A compiled pattern. Immutable after construction; thread-safe. RE2's DFA
// cache is per object and contended when shared, so each verification worker
// should use its own NewMatcher().
class CompiledPattern final {
 public:
  CompiledPattern(const CompiledPattern&) = delete;
  CompiledPattern& operator=(const CompiledPattern&) = delete;

  const PatternSpec& spec() const { return spec_; }

  // The shared matcher. Matches only within a line.
  const RE2& matcher() const { return *matcher_; }

  // Returns a new matcher equivalent to matcher().
  std::unique_ptr<RE2> NewMatcher() const;

  // Every document with a line that matches satisfies this query. None() if
  // the pattern matches no string at all (for example "[^\x00-\x{10FFFF}]").
  const TrigramQuery& query() const { return plan_.query; }

  // Empty, or a string of at least 3 bytes without A-Z that every match
  // contains after lowercasing its ASCII letters; see RegexPlan.
  const std::string& required_literal() const { return plan_.required_literal; }

  // Empty, or a string of at least 1 byte without A-Z that every match
  // contains after lowercasing its ASCII letters; see RegexPlan. The same as
  // required_literal() when that is set, but also set for 1- and 2-byte
  // literals, which the trigram index cannot use.
  const std::string& scan_literal() const { return plan_.scan_literal; }

  // Empty, or between 2 and kMaxScanAlternatives strings that are set only
  // when scan_literal() is empty: every match contains at least one of them
  // after lowercasing its ASCII letters; see RegexPlan. The basename scan
  // searches for each and unions the hits.
  const std::vector<std::string>& scan_alternatives() const {
    return plan_.scan_alternatives;
  }

  // Set when the pattern is a plain literal that a byte search decides
  // exactly: a literal-mode pattern that is case-sensitive, or case-insensitive
  // and ASCII without 'k' or 's' (whose case-insensitive matches include
  // U+212A and U+017F). A line matches if and only if the finder finds the
  // pattern in it, the same as RE2 finds a match, so a verifier or a name
  // test may skip RE2. Null otherwise. Thread-safe.
  const LiteralFinder* exact_literal() const {
    return exact_literal_.has_value() ? &*exact_literal_ : nullptr;
  }

  // Whether `text` contains a match, as RE2::PartialMatch(text, matcher())
  // decides, using exact_literal() when there is one. Thread-safe.
  bool MatchesWithin(absl::string_view text) const;

  // False if the analysis gave up on a pathological parse tree; query() is
  // then All().
  bool plan_complete() const { return plan_.complete; }

  // True if the pattern asserts the start or end of the text: \A, \z, or ^
  // or $ with multi-line mode turned off by (?-m). Search is line-oriented,
  // so these mean the start and end of a line, and a verifier must then
  // match each line on its own rather than scan a chunk.
  bool has_text_anchor() const { return has_text_anchor_; }

 private:
  friend absl::StatusOr<std::unique_ptr<CompiledPattern>> CompilePattern(
      const PatternSpec& spec, const RegexPlanOptions& options);

  CompiledPattern(PatternSpec spec, std::unique_ptr<RE2> matcher,
                  RegexPlan plan, bool has_text_anchor,
                  std::optional<LiteralFinder> exact_literal)
      : spec_(std::move(spec)),
        matcher_(std::move(matcher)),
        plan_(std::move(plan)),
        has_text_anchor_(has_text_anchor),
        exact_literal_(std::move(exact_literal)) {}

  const PatternSpec spec_;
  const std::unique_ptr<RE2> matcher_;
  const RegexPlan plan_;
  const bool has_text_anchor_;
  const std::optional<LiteralFinder> exact_literal_;
};

// Compiles `spec`. Returns InvalidArgument for an empty pattern, a pattern
// longer than kMaxPatternBytes, an RE2 syntax error, \C (which can split a
// UTF-8 sequence and so defeat folding), and a pattern that can only match
// across a newline, such as "foo\nbar" or "[\n]": NeverNL would turn it into
// a pattern that silently matches nothing.
absl::StatusOr<std::unique_ptr<CompiledPattern>> CompilePattern(
    const PatternSpec& spec, const RegexPlanOptions& options = {});

}  // namespace ferret

#endif  // FERRET_REGEX_COMPILE_H_

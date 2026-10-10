// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/regex/compile.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "re2/re2.h"
#include "re2/regexp.h"

#include "ferret/regex/analyze.h"
#include "ferret/text/literal_finder.h"

namespace ferret {
namespace {

// RE2's default is 8 MiB; large alternations of identifiers need more before
// the DFA falls back to the NFA.
constexpr int64_t kMaxRe2Memory = int64_t{64} << 20;

RE2::Options MakeOptions(const PatternSpec& spec, bool never_nl) {
  RE2::Options options;
  options.set_encoding(RE2::Options::EncodingUTF8);
  options.set_posix_syntax(false);
  options.set_longest_match(false);
  options.set_log_errors(false);
  options.set_max_mem(kMaxRe2Memory);
  options.set_never_nl(never_nl);
  options.set_dot_nl(false);
  options.set_never_capture(true);
  options.set_literal(spec.literal);
  options.set_case_sensitive(spec.case_sensitive);
  return options;
}

std::string PatternText(const PatternSpec& spec) {
  return spec.literal ? spec.pattern : absl::StrCat("(?m)", spec.pattern);
}

// Returns whether the tree contains a node whose op is one of `ops`.
// Iterative because RE2 accepts nesting deep enough to overflow a small
// thread stack.
bool ContainsOp(re2::Regexp* re, std::initializer_list<re2::RegexpOp> ops) {
  std::vector<re2::Regexp*> stack = {re};
  while (!stack.empty()) {
    re2::Regexp* node = stack.back();
    stack.pop_back();
    if (absl::c_linear_search(ops, node->op())) return true;
    re2::Regexp** subs = node->sub();
    for (int i = 0; i < node->nsub(); ++i) stack.push_back(subs[i]);
  }
  return false;
}

// NeverNL rewrites every newline in the pattern to "no match". If that
// leaves a pattern that matches nothing although the same pattern without
// NeverNL matches something, every match needed a newline: a user error in a
// line-oriented search, not a query with no results.
absl::Status CheckNotNewlineOnly(const PatternSpec& spec, const RE2& matcher) {
  if (!MatchesNothing(matcher.Regexp())) return absl::OkStatus();
  const RE2 without_never_nl(PatternText(spec),
                             MakeOptions(spec, /*never_nl=*/false));
  if (!without_never_nl.ok() || MatchesNothing(without_never_nl.Regexp())) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      "pattern can only match across a newline; search is line-oriented");
}

// A literal-mode pattern that a byte search decides exactly (see
// CompiledPattern::exact_literal()).
std::optional<LiteralFinder> ExactLiteral(const PatternSpec& spec) {
  if (!spec.literal || spec.pattern.empty()) return std::nullopt;
  if (spec.case_sensitive) {
    return LiteralFinder(spec.pattern, LiteralFinder::Mode::kExact);
  }
  for (const char c : spec.pattern) {
    // (?i) folds the other case of an ASCII letter and, for 'k' and 's',
    // U+212A and U+017F; non-ASCII letters fold too.
    if (static_cast<unsigned char>(c) >= 0x80 || c == 'k' || c == 'K' ||
        c == 's' || c == 'S') {
      return std::nullopt;
    }
  }
  return LiteralFinder(spec.pattern,
                       LiteralFinder::Mode::kAsciiCaseInsensitive);
}

}  // namespace

bool CompiledPattern::MatchesWithin(absl::string_view text) const {
  if (exact_literal_.has_value()) {
    return exact_literal_->Find(text) != absl::string_view::npos;
  }
  return RE2::PartialMatch(text, *matcher_);
}

std::unique_ptr<RE2> CompiledPattern::NewMatcher() const {
  return std::make_unique<RE2>(PatternText(spec_),
                               MakeOptions(spec_, /*never_nl=*/true));
}

absl::StatusOr<std::unique_ptr<CompiledPattern>> CompilePattern(
    const PatternSpec& spec, const RegexPlanOptions& options) {
  if (spec.pattern.empty()) {
    return absl::InvalidArgumentError("empty pattern");
  }
  if (spec.pattern.size() > kMaxPatternBytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("pattern is ", spec.pattern.size(),
                     " bytes; the limit is ", kMaxPatternBytes));
  }
  auto matcher = std::make_unique<RE2>(PatternText(spec),
                                       MakeOptions(spec, /*never_nl=*/true));
  if (!matcher->ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("invalid pattern: ", matcher->error()));
  }
  if (ContainsOp(matcher->Regexp(), {re2::kRegexpAnyByte})) {
    return absl::InvalidArgumentError(
        "\\C (any byte) is not supported: it can match part of a UTF-8 "
        "sequence");
  }
  ABSL_RETURN_IF_ERROR(CheckNotNewlineOnly(spec, *matcher));
  TrigramAnalyzer analyzer(options);
  RegexPlan plan = analyzer.Analyze(matcher->Regexp());
  const bool has_text_anchor = ContainsOp(
      matcher->Regexp(), {re2::kRegexpBeginText, re2::kRegexpEndText});
  return std::unique_ptr<CompiledPattern>(
      new CompiledPattern(spec, std::move(matcher), std::move(plan),
                          has_text_anchor, ExactLiteral(spec)));
}

}  // namespace ferret

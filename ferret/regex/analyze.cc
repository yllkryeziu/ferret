// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/regex/analyze.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "re2/regexp.h"
#include "util/utf.h"

#include "ferret/query/trigram_query.h"
#include "ferret/text/fold.h"
#include "ferret/text/trigram.h"

namespace ferret {
namespace {

using StringSet = std::vector<std::string>;

// RE2's parse trees have at most a few thousand nodes for the 4 KiB patterns
// CompilePattern() accepts; the budget only guards direct callers.
constexpr int kMaxVisits = 100000;

// Concat adds the trigrams of x.suffix × y.prefix only up to this many
// strings.
constexpr size_t kMaxCrossProduct = 64;

// The two non-ASCII runes that F folds to an ASCII letter.
constexpr re2::Rune kKelvinSign = 0x212A;
constexpr re2::Rune kLongS = 0x017F;

bool IsNonAsciiKs(re2::Rune rune) {
  return rune == kKelvinSign || rune == kLongS;
}

// Appends F(U(rune)), where U is RE2's own UTF-8 encoder. It encodes
// surrogates as three bytes, exactly as the compiled program matches them.
void AppendFoldedRune(re2::Rune rune, std::string& out) {
  char utf8[re2::UTFmax];
  const int length = re2::runetochar(utf8, &rune);
  AppendFolded(absl::string_view(utf8, static_cast<size_t>(length)), out);
}

bool SuffixLess(const std::string& a, const std::string& b) {
  return std::lexicographical_compare(
      a.rbegin(), a.rend(), b.rbegin(), b.rend(), [](char x, char y) {
        return static_cast<unsigned char>(x) < static_cast<unsigned char>(y);
      });
}

// Sorts (by reversed string for suffix sets) and removes duplicates.
void Clean(StringSet& set, bool is_suffix) {
  if (is_suffix) {
    std::sort(set.begin(), set.end(), SuffixLess);
  } else {
    std::sort(set.begin(), set.end());
  }
  set.erase(std::unique(set.begin(), set.end()), set.end());
}

StringSet Union(StringSet a, const StringSet& b) {
  a.insert(a.end(), b.begin(), b.end());
  Clean(a, /*is_suffix=*/false);
  return a;
}

StringSet Cross(const StringSet& a, const StringSet& b) {
  StringSet product;
  product.reserve(a.size() * b.size());
  for (const std::string& x : a) {
    for (const std::string& y : b) product.push_back(x + y);
  }
  Clean(product, /*is_suffix=*/false);
  return product;
}

// MinLen of the empty set is 0, so an empty set never yields trigrams.
size_t MinLen(const StringSet& set) {
  size_t min = set.empty() ? 0 : set.front().size();
  for (const std::string& s : set) min = std::min(min, s.size());
  return min;
}

// q AND (OR over s in set of (AND of the trigrams of s)). Adds nothing when
// some string is shorter than a trigram, since it then guarantees none.
TrigramQuery AndTrigrams(TrigramQuery query, const StringSet& set) {
  if (MinLen(set) < 3) return query;
  TrigramQuery any = TrigramQuery::None();
  std::vector<Trigram> trigrams;
  for (const std::string& s : set) {
    trigrams.clear();
    for (size_t i = 0; i + 3 <= s.size(); ++i) {
      const Trigram trigram = TrigramAt(s, i);
      // Trigrams with '\n' are never indexed, so requiring one would drop
      // every document; treating it as All() is sound. NeverNL keeps them
      // out of RE2's tree in the first place.
      DCHECK(!TrigramHasNewline(trigram))
          << "newline in planned string " << TrigramToString(trigram);
      if (!TrigramHasNewline(trigram)) trigrams.push_back(trigram);
    }
    any = TrigramQuery::Or(std::move(any), TrigramQuery::AllOf(trigrams));
  }
  return TrigramQuery::And(std::move(query), std::move(any));
}

void AddExact(RegexInfo& info) {
  if (info.exact.has_value()) {
    info.match = AndTrigrams(std::move(info.match), *info.exact);
  }
}

void KeepLonger(std::string& kept, absl::string_view candidate) {
  if (candidate.size() > kept.size()) kept = std::string(candidate);
}

// The size of the shortest string of `set`: 0 for an empty set (no
// information). A set with a longer shortest string selects fewer positions.
size_t ShortestLen(const std::vector<std::string>& set) { return MinLen(set); }

// Keeps in `kept` whichever of the two required sets is more selective:
// longer shortest string, then fewer strings.
void KeepBetter(std::vector<std::string>& kept,
                const std::vector<std::string>& candidate) {
  const size_t kept_len = ShortestLen(kept);
  const size_t candidate_len = ShortestLen(candidate);
  if (candidate_len > kept_len ||
      (candidate_len == kept_len && candidate_len > 0 &&
       candidate.size() < kept.size())) {
    kept = candidate;
  }
}

// The union of two required sets, or nothing when either is unknown or the
// result is too large: a match of x|y matches x or y, and so contains a
// string of the set of x or of y.
std::vector<std::string> UnionRequired(const std::vector<std::string>& x,
                                       const std::vector<std::string>& y) {
  if (x.empty() || y.empty()) return {};
  std::vector<std::string> both = x;
  both.insert(both.end(), y.begin(), y.end());
  std::sort(both.begin(), both.end());
  both.erase(std::unique(both.begin(), both.end()), both.end());
  if (both.size() > kMaxScanAlternatives) return {};
  return both;
}

// The strings of `exact` as a required set: every match is one of them.
// Nothing when one is empty or there are too many.
void NoteRequiredAny(RegexInfo& info) {
  if (!info.exact.has_value() || info.exact->empty() ||
      info.exact->size() > kMaxScanAlternatives) {
    return;
  }
  const std::vector<std::string>& exact = *info.exact;
  if (MinLen(exact) > 0) KeepBetter(info.required_any, exact);
  // Without 'k' and 's': the longest run of each string.
  std::vector<std::string> runs;
  for (const std::string& s : exact) {
    std::string longest;
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); ++i) {
      if (i == s.size() || s[i] == 'k' || s[i] == 's') {
        KeepLonger(longest, absl::string_view(s).substr(start, i - start));
        start = i + 1;
      }
    }
    if (longest.empty()) return;
    runs.push_back(std::move(longest));
  }
  std::sort(runs.begin(), runs.end());
  runs.erase(std::unique(runs.begin(), runs.end()), runs.end());
  KeepBetter(info.required_any_ks_free, runs);
}

// A singleton exact set {s} means F(m) == s for every match m, so s is
// required, and so is each of its runs without 'k' or 's'.
void NoteRequired(RegexInfo& info) {
  NoteRequiredAny(info);
  if (!info.exact.has_value() || info.exact->size() != 1) return;
  const std::string& s = info.exact->front();
  KeepLonger(info.required, s);
  size_t start = 0;
  for (size_t i = 0; i <= s.size(); ++i) {
    if (i == s.size() || s[i] == 'k' || s[i] == 's') {
      KeepLonger(info.required_ks_free,
                 absl::string_view(s).substr(start, i - start));
      start = i + 1;
    }
  }
}

RegexInfo NoMatch() {
  RegexInfo info;
  info.match = TrigramQuery::None();
  return info;
}

RegexInfo EmptyString() {
  RegexInfo info;
  info.can_empty = true;
  info.exact = StringSet{""};
  return info;
}

RegexInfo AnyChar() {
  RegexInfo info;
  info.prefix = {""};
  info.suffix = {""};
  return info;
}

RegexInfo AnyMatch() {
  RegexInfo info = AnyChar();
  info.can_empty = true;
  return info;
}

// Computes language emptiness bottom-up: a child argument is true when the
// child matches nothing.
class EmptinessWalker final : public re2::Regexp::Walker<bool> {
 public:
  bool PostVisit(re2::Regexp* re, bool /*parent_arg*/, bool /*pre_arg*/,
                 bool* child_args, int nchild_args) override {
    const auto children = [&] {
      return absl::MakeConstSpan(child_args, static_cast<size_t>(nchild_args));
    };
    switch (re->op()) {
      case re2::kRegexpNoMatch:
        return true;
      case re2::kRegexpCharClass:
        return re->cc()->empty();
      case re2::kRegexpConcat:
        return absl::c_any_of(children(), [](bool empty) { return empty; });
      case re2::kRegexpAlternate:
        return absl::c_all_of(children(), [](bool empty) { return empty; });
      case re2::kRegexpPlus:
      case re2::kRegexpCapture:
        return child_args[0];
      case re2::kRegexpRepeat:
        return re->min() > 0 && child_args[0];
      default:
        // Leaves that match a string, assertions, and Star and Quest, which
        // match "".
        return false;
    }
  }

  bool ShortVisit(re2::Regexp* /*re*/, bool /*parent_arg*/) override {
    // Only reached when the visit budget runs out: claim nothing.
    return false;
  }
};

}  // namespace

bool MatchesNothing(re2::Regexp* re) {
  return EmptinessWalker().WalkExponential(re, false, kMaxVisits);
}

TrigramAnalyzer::TrigramAnalyzer(const RegexPlanOptions& options)
    : options_(options) {
  DCHECK_GE(options_.max_exact, 0);
  DCHECK_GE(options_.max_set, 1);
  DCHECK_GE(options_.class_limit, 0);
  DCHECK_GE(options_.or_fanout, 1);
}

RegexPlan TrigramAnalyzer::Analyze(re2::Regexp* re) {
  non_ascii_ks_ = false;
  budget_exceeded_ = false;
  RegexInfo info = WalkExponential(re, RegexInfo(), kMaxVisits);
  RegexPlan plan;
  if (budget_exceeded_) {
    plan.complete = false;
    return plan;
  }
  Simplify(info, /*force=*/true);
  AddExact(info);
  plan.query = WeakenExpensiveOrs(std::move(info.match), options_.or_fanout);
  // An ASCII case-insensitive search finds 'k' and 's' only where the text
  // spells them in ASCII, so they may not be required when the pattern can
  // match U+212A or U+017F instead.
  std::string& literal = non_ascii_ks_ ? info.required_ks_free : info.required;
  plan.scan_literal = literal;
  if (literal.empty()) {
    const std::vector<std::string>& any =
        non_ascii_ks_ ? info.required_any_ks_free : info.required_any;
    if (any.size() >= 2 && any.size() <= kMaxScanAlternatives) {
      plan.scan_alternatives = any;
    }
  }
  if (literal.size() >= 3) plan.required_literal = std::move(literal);
  return plan;
}

RegexInfo TrigramAnalyzer::ShortVisit(re2::Regexp* /*re*/,
                                      RegexInfo /*parent_arg*/) {
  // The walk calls this only once the budget is spent; Analyze() then
  // discards the result.
  budget_exceeded_ = true;
  return AnyMatch();
}

RegexInfo TrigramAnalyzer::PostVisit(re2::Regexp* re, RegexInfo /*parent_arg*/,
                                     RegexInfo /*pre_arg*/,
                                     RegexInfo* child_args, int nchild_args) {
  switch (re->op()) {
    case re2::kRegexpNoMatch:
      return NoMatch();

    case re2::kRegexpEmptyMatch:
    case re2::kRegexpBeginLine:
    case re2::kRegexpEndLine:
    case re2::kRegexpBeginText:
    case re2::kRegexpEndText:
    case re2::kRegexpWordBoundary:
    case re2::kRegexpNoWordBoundary:
      return EmptyString();

    case re2::kRegexpLiteral: {
      // RE2 keeps FoldCase only on runes whose orbit is {X, x} (both fold to
      // x) or trivial; larger orbits arrive as CharClass nodes.
      if (IsNonAsciiKs(re->rune())) non_ascii_ks_ = true;
      std::string s;
      AppendFoldedRune(re->rune(), s);
      return Leaf({std::move(s)});
    }

    case re2::kRegexpLiteralString: {
      std::string s;
      for (int i = 0; i < re->nrunes(); ++i) {
        if (IsNonAsciiKs(re->runes()[i])) non_ascii_ks_ = true;
        AppendFoldedRune(re->runes()[i], s);
      }
      return Leaf({std::move(s)});
    }

    case re2::kRegexpCharClass: {
      re2::CharClass* cc = re->cc();
      if (cc->empty()) return NoMatch();
      if (cc->size() > options_.class_limit) return AnyChar();
      if (cc->Contains(kKelvinSign) || cc->Contains(kLongS)) {
        non_ascii_ks_ = true;
      }
      StringSet exact;
      exact.reserve(static_cast<size_t>(cc->size()));
      for (const re2::RuneRange range : *cc) {
        for (re2::Rune rune = range.lo; rune <= range.hi; ++rune) {
          AppendFoldedRune(rune, exact.emplace_back());
        }
      }
      return Leaf(std::move(exact));
    }

    case re2::kRegexpAnyChar:
    case re2::kRegexpAnyByte:
      return AnyChar();

    case re2::kRegexpCapture:
      return std::move(child_args[0]);

    case re2::kRegexpStar:
      return AnyMatch();

    case re2::kRegexpQuest:
      return Alternate(std::move(child_args[0]), EmptyString());

    case re2::kRegexpPlus:
      return Plus(std::move(child_args[0]));

    case re2::kRegexpRepeat:
      return Repeat(std::move(child_args[0]), re->min(), re->max());

    case re2::kRegexpConcat: {
      if (nchild_args == 0) return EmptyString();
      RegexInfo info = std::move(child_args[0]);
      for (int i = 1; i < nchild_args; ++i) {
        info = Concat(std::move(info), std::move(child_args[i]));
      }
      return info;
    }

    case re2::kRegexpAlternate: {
      if (nchild_args == 0) return NoMatch();
      RegexInfo info = std::move(child_args[0]);
      for (int i = 1; i < nchild_args; ++i) {
        info = Alternate(std::move(info), std::move(child_args[i]));
      }
      return info;
    }

    case re2::kRegexpHaveMatch:
      // Only RE2::Set builds these; any string may follow.
      return AnyMatch();
  }
  DCHECK(false) << "unknown RE2 op " << static_cast<int>(re->op());
  return AnyMatch();
}

RegexInfo TrigramAnalyzer::Leaf(StringSet exact) const {
  RegexInfo info;
  Clean(exact, /*is_suffix=*/false);
  info.exact = std::move(exact);
  NoteRequired(info);
  Simplify(info, /*force=*/false);
  return info;
}

RegexInfo TrigramAnalyzer::Concat(RegexInfo x, RegexInfo y) const {
  RegexInfo xy;
  xy.match = TrigramQuery::And(std::move(x.match), std::move(y.match));
  if (x.exact.has_value() && y.exact.has_value()) {
    xy.exact = Cross(*x.exact, *y.exact);
  } else {
    if (x.exact.has_value()) {
      xy.prefix = Cross(*x.exact, y.prefix);
    } else {
      xy.prefix = x.can_empty ? Union(x.prefix, y.prefix) : x.prefix;
    }
    if (y.exact.has_value()) {
      xy.suffix = Cross(x.suffix, *y.exact);
    } else {
      xy.suffix = y.can_empty ? Union(y.suffix, x.suffix) : y.suffix;
    }
  }
  // A string of x.suffix × y.prefix straddles the boundary, so its trigrams
  // are in neither side's match yet.
  if (!x.exact.has_value() && !y.exact.has_value() &&
      x.suffix.size() * y.prefix.size() <= kMaxCrossProduct &&
      MinLen(x.suffix) + MinLen(y.prefix) >= 3) {
    xy.match = AndTrigrams(std::move(xy.match), Cross(x.suffix, y.prefix));
  }
  xy.can_empty = x.can_empty && y.can_empty;
  xy.required = std::move(x.required);
  KeepLonger(xy.required, y.required);
  xy.required_ks_free = std::move(x.required_ks_free);
  KeepLonger(xy.required_ks_free, y.required_ks_free);
  xy.required_any = std::move(x.required_any);
  KeepBetter(xy.required_any, y.required_any);
  xy.required_any_ks_free = std::move(x.required_any_ks_free);
  KeepBetter(xy.required_any_ks_free, y.required_any_ks_free);
  NoteRequired(xy);
  Simplify(xy, /*force=*/false);
  return xy;
}

RegexInfo TrigramAnalyzer::Alternate(RegexInfo x, RegexInfo y) const {
  RegexInfo xy;
  if (x.exact.has_value() && y.exact.has_value()) {
    xy.exact = Union(*x.exact, *y.exact);
  } else if (x.exact.has_value()) {
    xy.prefix = Union(*x.exact, y.prefix);
    xy.suffix = Union(*x.exact, y.suffix);
    AddExact(x);
  } else if (y.exact.has_value()) {
    xy.prefix = Union(x.prefix, *y.exact);
    xy.suffix = Union(x.suffix, *y.exact);
    AddExact(y);
  } else {
    xy.prefix = Union(x.prefix, y.prefix);
    xy.suffix = Union(x.suffix, y.suffix);
  }
  xy.can_empty = x.can_empty || y.can_empty;
  xy.match = TrigramQuery::Or(std::move(x.match), std::move(y.match));
  xy.required_any = UnionRequired(x.required_any, y.required_any);
  xy.required_any_ks_free =
      UnionRequired(x.required_any_ks_free, y.required_any_ks_free);
  NoteRequired(xy);
  Simplify(xy, /*force=*/false);
  return xy;
}

RegexInfo TrigramAnalyzer::Plus(RegexInfo x) const {
  // x+ starts with an x and ends with an x, but is no longer exact.
  if (x.exact.has_value()) {
    x.prefix = *x.exact;
    x.suffix = *std::move(x.exact);
    x.exact.reset();
  }
  Simplify(x, /*force=*/false);
  return x;
}

RegexInfo TrigramAnalyzer::Repeat(RegexInfo x, int min, int max) const {
  // x{0} stays in RE2's tree and matches only "".
  if (max == 0) return EmptyString();
  if (min == 0) return AnyMatch();
  if (min == max && min <= 3) {
    RegexInfo info = x;
    for (int i = 1; i < min; ++i) info = Concat(std::move(info), x);
    return info;
  }
  return Plus(std::move(x));
}

void TrigramAnalyzer::Simplify(RegexInfo& info, bool force) const {
  if (info.exact.has_value()) {
    StringSet& exact = *info.exact;
    Clean(exact, /*is_suffix=*/false);
    const size_t min_len = MinLen(exact);
    if (exact.size() > static_cast<size_t>(options_.max_exact) ||
        min_len >= 4 || (force && min_len >= 3)) {
      // Record the exact strings' trigrams, then keep only their ends.
      AddExact(info);
      for (const std::string& s : exact) {
        if (s.size() < 3) {
          info.prefix.push_back(s);
          info.suffix.push_back(s);
        } else {
          info.prefix.push_back(s.substr(0, 2));
          info.suffix.push_back(s.substr(s.size() - 2));
        }
      }
      info.exact.reset();
    }
  }
  if (!info.exact.has_value()) {
    SimplifySet(info, info.prefix, /*is_suffix=*/false);
    SimplifySet(info, info.suffix, /*is_suffix=*/true);
  }
}

void TrigramAnalyzer::SimplifySet(RegexInfo& info, StringSet& set,
                                  bool is_suffix) const {
  Clean(set, is_suffix);
  info.match = AndTrigrams(std::move(info.match), set);
  // The trigrams are recorded, so strings can shrink to 2 bytes, and further
  // while the set is too large.
  for (size_t n = 3;
       n == 3 || set.size() > static_cast<size_t>(options_.max_set); --n) {
    DCHECK_GE(n, 1u);
    for (std::string& s : set) {
      if (s.size() >= n) {
        s = is_suffix ? s.substr(s.size() - (n - 1)) : s.substr(0, n - 1);
      }
    }
    Clean(set, is_suffix);
  }
  // A string that extends another adds no information: drop it.
  size_t kept = 0;
  for (size_t i = 0; i < set.size(); ++i) {
    if (kept > 0 && (is_suffix ? absl::EndsWith(set[i], set[kept - 1])
                               : absl::StartsWith(set[i], set[kept - 1]))) {
      continue;
    }
    if (kept != i) set[kept] = std::move(set[i]);
    ++kept;
  }
  set.resize(kept);
}

}  // namespace ferret

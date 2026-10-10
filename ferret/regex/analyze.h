// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Trigram analysis of RE2's parse tree.
//
// TrigramAnalyzer computes, bottom-up over the exact re2::Regexp that RE2
// compiled, facts about the folded images F(s) (ferret/text/fold.h) of the
// strings s each node can match, and turns them into a TrigramQuery that
// every document containing a match satisfies. Because the planner walks
// RE2's own tree, it cannot disagree with the matcher about syntax, escapes,
// fold orbits, or repeat limits. The algebra is a port of codesearch's
// regexp.go [1] over F-images; docs/regex_analysis.md proves it sound.
//
// The walk is non-recursive (RE2 accepts nesting 5,000 levels deep) and
// visits at most 100,000 nodes; a pattern that exceeds the budget plans to
// All(). Planning costs O(nodes * set sizes), which the options bound.
//
// [1] https://github.com/google/codesearch/blob/master/index/regexp.go

#ifndef FERRET_REGEX_ANALYZE_H_
#define FERRET_REGEX_ANALYZE_H_

#include <optional>
#include <string>
#include <vector>

#include "re2/regexp.h"
// walker-inl.h is RE2-internal and was not written for -Wextra: its default
// Visit hooks leave parameters unused.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "re2/walker-inl.h"
#pragma GCC diagnostic pop
#include "ferret/query/trigram_query.h"

namespace ferret {

// Limits that trade plan precision for plan size. The defaults start from
// Cox's values; the planner ablation compares (7, 20), (16, 32), (32, 64).
struct RegexPlanOptions {
  // kMaxExact: an exact set with more strings is flushed into prefix and
  // suffix sets. Must be >= 0.
  int max_exact = 7;
  // kMaxSet: prefix and suffix sets are truncated until they have at most
  // this many strings. Must be >= 1.
  int max_set = 20;
  // A character class with more runes than this matches "any character".
  // Must be >= 0.
  int class_limit = 128;
  // ORs with more elements than this become All() (WeakenExpensiveOrs).
  // Must be >= 1.
  int or_fanout = 32;
};

// What the analysis knows about the F-images of the strings a node matches.
// For every string s in L(node):
//   * s == "" implies can_empty;
//   * if exact is set, F(s) is in *exact;
//   * otherwise some p in prefix is a prefix of F(s), and some q in suffix is
//     a suffix of F(s);
//   * the set of trigrams of F(s) satisfies match;
//   * required and required_ks_free are substrings of F(s).
// Moreover, can_empty implies that "" is in *exact, or in both prefix and
// suffix.
struct RegexInfo {
  bool can_empty = false;
  std::optional<std::vector<std::string>> exact;
  std::vector<std::string> prefix;
  std::vector<std::string> suffix;
  TrigramQuery match;
  // The longest required string found so far, and the longest one without the
  // bytes 'k' and 's' (see RegexPlan::required_literal).
  std::string required;
  std::string required_ks_free;
  // Empty (nothing known), or non-empty strings, at most kMaxAlternatives of
  // them, such that every string s that the node matches contains at least
  // one of them once s's ASCII letters are lowercased (the F-image rule of
  // `required`). A set of one string is `required` when that is the longest.
  // The second set uses only strings without 'k' and 's'.
  std::vector<std::string> required_any;
  std::vector<std::string> required_any_ks_free;
};

struct RegexPlan {
  // Every document with a line that the regexp matches satisfies this query.
  TrigramQuery query;
  // Empty, or at least 3 bytes with no A-Z such that every string s that the
  // regexp matches contains it once s's ASCII letters are lowercased. It lets
  // the verifier find candidate lines with an ASCII case-insensitive search.
  std::string required_literal;
  // False if the walk exceeded its visit budget; query is then All().
  bool complete = true;
  // Like required_literal, but of any length: empty, or at least 1 byte with
  // no A-Z that every matched string contains once its ASCII letters are
  // lowercased. It equals required_literal whenever that is set. Trigram
  // planning needs 3 bytes; the basename scan and the verifier's prefilter
  // do not.
  std::string scan_literal;
  // Empty, or between 2 and kMaxAlternatives strings, none empty and none
  // with A-Z, such that every string s that the regexp matches contains at
  // least one of them once s's ASCII letters are lowercased. Set only when
  // scan_literal is empty: an alternation such as "foo|bar" has no single
  // required string, but each branch has one. The 'k' and 's' rule of
  // required_literal applies to every string.
  std::vector<std::string> scan_alternatives;
};

// The most strings in RegexPlan::scan_alternatives. Each costs a pass over
// the packed basenames when the basename scan cannot be limited to the
// candidate documents.
inline constexpr size_t kMaxScanAlternatives = 8;

// Not final: analyze_soundness_test derives seeded-bug variants that override
// PostVisit() for one op, so the mutation guard needs no hook in this code.
// Not thread-safe; use one analyzer per thread.
class TrigramAnalyzer : public re2::Regexp::Walker<RegexInfo> {
 public:
  // Requires options within the documented ranges.
  explicit TrigramAnalyzer(const RegexPlanOptions& options);
  TrigramAnalyzer(const TrigramAnalyzer&) = delete;
  TrigramAnalyzer& operator=(const TrigramAnalyzer&) = delete;
  ~TrigramAnalyzer() override = default;

  // Plans `re`, which must come from a UTF-8 parse. Planning is sound for any
  // such tree; CompilePattern() additionally guarantees NeverNL, so no
  // required trigram contains '\n' (one that did would be dropped, which is
  // sound because such trigrams are never indexed, and DCHECK-fails).
  RegexPlan Analyze(re2::Regexp* re);

 protected:
  RegexInfo PostVisit(re2::Regexp* re, RegexInfo parent_arg, RegexInfo pre_arg,
                      RegexInfo* child_args, int nchild_args) override;
  RegexInfo ShortVisit(re2::Regexp* re, RegexInfo parent_arg) final;

  const RegexPlanOptions& options() const { return options_; }

  // The analysis rules, for PostVisit() overrides.
  //
  // A leaf whose F-images are exactly `exact`.
  RegexInfo Leaf(std::vector<std::string> exact) const;
  // The info of xy, and of x|y.
  RegexInfo Concat(RegexInfo x, RegexInfo y) const;
  RegexInfo Alternate(RegexInfo x, RegexInfo y) const;

 private:
  RegexInfo Plus(RegexInfo x) const;
  RegexInfo Repeat(RegexInfo x, int min, int max) const;

  // Bounds the sets, moving what they lose into match first.
  void Simplify(RegexInfo& info, bool force) const;
  void SimplifySet(RegexInfo& info, std::vector<std::string>& set,
                   bool is_suffix) const;

  RegexPlanOptions options_;
  // Whether some literal or small class contains U+212A or U+017F, whose
  // F-images 'k' and 's' are not their ASCII-lowercased bytes.
  bool non_ascii_ks_ = false;
  bool budget_exceeded_ = false;
};

// Returns whether `re` matches no string at all, for example "[\n]" under
// NeverNL. Assertions such as ^ and \b count as matchable. Iterative, so any
// nesting depth RE2 accepts is safe.
bool MatchesNothing(re2::Regexp* re);

}  // namespace ferret

#endif  // FERRET_REGEX_ANALYZE_H_

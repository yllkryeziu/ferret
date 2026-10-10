// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// TrigramQuery: the monotone boolean formula over trigrams that a pattern's
// plan compiles to.
//
// A document can contain a match only if its trigram set satisfies the
// pattern's query, so the index returns a superset of the matching documents
// and RE2 removes the rest. Queries have no NOT, so they are monotone: adding
// trigrams to a document never turns a query from true to false, and
// replacing any subquery by All() only weakens the query. The planner's caps
// and WeakenExpensiveOrs() rely on exactly that.
//
// Every value is in canonical form, maintained by the factories:
//   * kAll and kNone have no trigrams and no subqueries;
//   * kAnd and kOr have sorted, unique trigrams (stored inline, as codesearch
//     does), at least two elements (trigrams plus subqueries), and every
//     subquery is a kAnd or kOr of the opposite op;
//   * the one exception is a single trigram, which is a kAnd with one trigram
//     and no subqueries.
// The factories port codesearch's andOr [1]: single-element nodes are
// unwrapped, a node that provably implies its sibling absorbs it, nodes of the
// same op are merged, and trigrams common to two siblings are factored out,
// (abc & X) | (abc & Y) -> abc & (X | Y), recursively. Canonicalization is
// idempotent: rebuilding a canonical query through the factories returns an
// equal query (query_algebra_test checks this).
//
// [1] https://github.com/google/codesearch/blob/master/index/regexp.go

#ifndef FERRET_QUERY_TRIGRAM_QUERY_H_
#define FERRET_QUERY_TRIGRAM_QUERY_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/types/span.h"

#include "ferret/text/trigram.h"

namespace ferret {

// A regular value type. Thread-compatible.
class TrigramQuery final {
 public:
  enum class Op : uint8_t {
    kAll,   // Every document.
    kNone,  // No document.
    kAnd,   // Every trigram and every subquery.
    kOr,    // Some trigram or some subquery.
  };

  // All().
  TrigramQuery() = default;

  TrigramQuery(const TrigramQuery&) = default;
  TrigramQuery& operator=(const TrigramQuery&) = default;
  TrigramQuery(TrigramQuery&&) = default;
  TrigramQuery& operator=(TrigramQuery&&) = default;

  static TrigramQuery All();
  static TrigramQuery None();

  // The conjunction of `trigrams`, in any order and possibly with duplicates.
  // All() if `trigrams` is empty.
  static TrigramQuery AllOf(absl::Span<const Trigram> trigrams);

  // The disjunction of `trigrams`, in any order and possibly with
  // duplicates. None() if `trigrams` is empty.
  static TrigramQuery AnyOf(absl::Span<const Trigram> trigrams);

  // q AND r and q OR r, in canonical form. Cost is polynomial in the sizes of
  // q and r; plans stay small because the planner caps its string sets.
  static TrigramQuery And(TrigramQuery q, TrigramQuery r);
  static TrigramQuery Or(TrigramQuery q, TrigramQuery r);

  Op op() const { return op_; }

  // Sorted and unique. Empty for kAll and kNone.
  absl::Span<const Trigram> trigrams() const { return trigrams_; }

  // Subqueries, each a kAnd or kOr of the opposite op. Empty for kAll and
  // kNone.
  absl::Span<const TrigramQuery> subs() const { return subs_; }

  // Returns true only if every trigram set that satisfies *this provably
  // satisfies `r`. False negatives are allowed; a false positive would let the
  // factories absorb a subquery that still matters and drop documents.
  bool Implies(const TrigramQuery& r) const;

  // Returns whether a document whose trigram set is described by
  // `has_trigram` satisfies the query. This is the reference semantics that
  // the posting-list evaluator must agree with.
  bool Evaluate(absl::FunctionRef<bool(Trigram)> has_trigram) const;

  // Renders the query for `ferret plan --explain` and test failures, e.g.
  //   "bla" & "lac" & ("ack" | "ac\xe2")
  // AND binds tighter than OR, and subqueries are always parenthesized. The
  // constants render as All and None.
  std::string ToString() const;

  // Structural equality. Semantically equivalent queries that were built in a
  // different order may compare unequal.
  friend bool operator==(const TrigramQuery&, const TrigramQuery&) = default;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const TrigramQuery& query) {
    sink.Append(query.ToString());
  }

 private:
  TrigramQuery(Op op, std::vector<Trigram> trigrams,
               std::vector<TrigramQuery> subs);

  friend TrigramQuery WeakenExpensiveOrs(TrigramQuery query, int max_or_fanout);

  // The node `op` over sorted, unique `trigrams` alone, normalized: All() or
  // None() when empty, a single-trigram kAnd when there is one.
  static TrigramQuery FromTrigrams(Op op, std::vector<Trigram> trigrams);

  // Canonicalizes the node `op` over `trigrams` and `subs` by inserting the
  // subqueries one at a time into FromTrigrams(op, trigrams). This is the
  // definition of canonical form: a canonical query is a fixpoint of it.
  static TrigramQuery Rebuild(Op op, std::vector<Trigram> trigrams,
                              std::vector<TrigramQuery> subs);

  static TrigramQuery AndOr(TrigramQuery q, TrigramQuery r, Op op);

  // Implements WeakenExpensiveOrs(). Returns true if `query` changed.
  static bool WeakenInPlace(TrigramQuery& query, size_t max_or_fanout);

  bool IsSingleTrigram() const {
    return op_ == Op::kAnd && trigrams_.size() == 1 && subs_.empty();
  }

  void AppendTo(std::string& out) const;

  Op op_ = Op::kAll;
  std::vector<Trigram> trigrams_;
  std::vector<TrigramQuery> subs_;
};

// Replaces every OR with more than `max_or_fanout` elements (trigrams plus
// subqueries) by All() and re-canonicalizes. Sound because queries are
// monotone; it bounds the number of posting lists one OR unions. Requires
// max_or_fanout >= 1.
TrigramQuery WeakenExpensiveOrs(TrigramQuery query, int max_or_fanout);

}  // namespace ferret

#endif  // FERRET_QUERY_TRIGRAM_QUERY_H_

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/query/trigram_query.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/types/span.h"

#include "ferret/text/trigram.h"

namespace ferret {
namespace {

using Op = TrigramQuery::Op;

Op OtherOp(Op op) {
  DCHECK(op == Op::kAnd || op == Op::kOr);
  return op == Op::kAnd ? Op::kOr : Op::kAnd;
}

// Both arguments are sorted.
bool IsSubset(absl::Span<const Trigram> sub, absl::Span<const Trigram> set) {
  return std::includes(set.begin(), set.end(), sub.begin(), sub.end());
}

// Both arguments are sorted.
bool Intersects(absl::Span<const Trigram> a, absl::Span<const Trigram> b) {
  size_t i = 0;
  size_t j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] < b[j]) {
      ++i;
    } else if (b[j] < a[i]) {
      ++j;
    } else {
      return true;
    }
  }
  return false;
}

// Returns true if the conjunction of the sorted `trigrams` provably implies
// `query` (codesearch's trigramsImply).
bool TrigramsImply(absl::Span<const Trigram> trigrams,
                   const TrigramQuery& query) {
  switch (query.op()) {
    case Op::kAll:
      return true;
    case Op::kNone:
      return false;
    case Op::kAnd:
      return IsSubset(query.trigrams(), trigrams) &&
             std::all_of(query.subs().begin(), query.subs().end(),
                         [trigrams](const TrigramQuery& sub) {
                           return TrigramsImply(trigrams, sub);
                         });
    case Op::kOr:
      return Intersects(trigrams, query.trigrams()) ||
             std::any_of(query.subs().begin(), query.subs().end(),
                         [trigrams](const TrigramQuery& sub) {
                           return TrigramsImply(trigrams, sub);
                         });
  }
  return false;
}

}  // namespace

TrigramQuery::TrigramQuery(Op op, std::vector<Trigram> trigrams,
                           std::vector<TrigramQuery> subs)
    : op_(op), trigrams_(std::move(trigrams)), subs_(std::move(subs)) {}

TrigramQuery TrigramQuery::All() { return TrigramQuery(); }

TrigramQuery TrigramQuery::None() { return TrigramQuery(Op::kNone, {}, {}); }

TrigramQuery TrigramQuery::AllOf(absl::Span<const Trigram> trigrams) {
  std::vector<Trigram> sorted(trigrams.begin(), trigrams.end());
  std::sort(sorted.begin(), sorted.end());
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
  return FromTrigrams(Op::kAnd, std::move(sorted));
}

TrigramQuery TrigramQuery::AnyOf(absl::Span<const Trigram> trigrams) {
  std::vector<Trigram> sorted(trigrams.begin(), trigrams.end());
  std::sort(sorted.begin(), sorted.end());
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
  return FromTrigrams(Op::kOr, std::move(sorted));
}

TrigramQuery TrigramQuery::And(TrigramQuery q, TrigramQuery r) {
  return AndOr(std::move(q), std::move(r), Op::kAnd);
}

TrigramQuery TrigramQuery::Or(TrigramQuery q, TrigramQuery r) {
  return AndOr(std::move(q), std::move(r), Op::kOr);
}

TrigramQuery TrigramQuery::FromTrigrams(Op op, std::vector<Trigram> trigrams) {
  if (trigrams.empty()) return op == Op::kAnd ? All() : None();
  // A single trigram means the same under either op; kAnd is its one
  // canonical spelling.
  if (trigrams.size() == 1) op = Op::kAnd;
  return TrigramQuery(op, std::move(trigrams), {});
}

TrigramQuery TrigramQuery::Rebuild(Op op, std::vector<Trigram> trigrams,
                                   std::vector<TrigramQuery> subs) {
  TrigramQuery result = FromTrigrams(op, std::move(trigrams));
  for (TrigramQuery& sub : subs) {
    result = AndOr(std::move(result), std::move(sub), op);
  }
  return result;
}

TrigramQuery TrigramQuery::AndOr(TrigramQuery q, TrigramQuery r, Op op) {
  // Absorption: if q => r, then q & r == q and q | r == r. This also folds
  // the constants, since None implies everything and everything implies All.
  if (q.Implies(r)) return op == Op::kAnd ? std::move(q) : std::move(r);
  if (r.Implies(q)) return op == Op::kAnd ? std::move(r) : std::move(q);
  DCHECK(q.op_ == Op::kAnd || q.op_ == Op::kOr);
  DCHECK(r.op_ == Op::kAnd || r.op_ == Op::kOr);

  // A node "joins" `op` if it can take r's elements directly: it has op
  // `op`, or it is a single trigram, which means the same under either op.
  const bool q_joins = q.op_ == op || q.IsSingleTrigram();
  const bool r_joins = r.op_ == op || r.IsSingleTrigram();
  if (q_joins && r_joins) {
    std::vector<Trigram> trigrams;
    std::set_union(q.trigrams_.begin(), q.trigrams_.end(), r.trigrams_.begin(),
                   r.trigrams_.end(), std::back_inserter(trigrams));
    std::vector<TrigramQuery> subs = std::move(q.subs_);
    subs.insert(subs.end(), std::make_move_iterator(r.subs_.begin()),
                std::make_move_iterator(r.subs_.end()));
    // The merged trigrams may now absorb subqueries of either side.
    return Rebuild(op, std::move(trigrams), std::move(subs));
  }
  if (q_joins || r_joins) {
    // Exactly one side joins. The other is a node of the other op with at
    // least two elements, and absorption above ruled out redundancy, so it
    // simply becomes a subquery.
    if (!q_joins) std::swap(q, r);
    q.op_ = op;
    q.subs_.push_back(std::move(r));
    return q;
  }

  // Both are nodes of the other op: an OR of ANDs or an AND of ORs. Factor
  // out their common trigrams, since
  //   (C & X) | (C & Y) == C & (X | Y)  and  (C | X) & (C | Y) == C | (X & Y).
  std::vector<Trigram> common;
  std::set_intersection(q.trigrams_.begin(), q.trigrams_.end(),
                        r.trigrams_.begin(), r.trigrams_.end(),
                        std::back_inserter(common));
  if (common.empty()) {
    return TrigramQuery(op, {}, {std::move(q), std::move(r)});
  }
  const Op other = OtherOp(op);
  const auto remove_common = [&common, other](TrigramQuery node) {
    std::vector<Trigram> rest;
    std::set_difference(node.trigrams_.begin(), node.trigrams_.end(),
                        common.begin(), common.end(), std::back_inserter(rest));
    return Rebuild(other, std::move(rest), std::move(node.subs_));
  };
  TrigramQuery rest =
      AndOr(remove_common(std::move(q)), remove_common(std::move(r)), op);
  return AndOr(FromTrigrams(other, std::move(common)), std::move(rest), other);
}

bool TrigramQuery::Implies(const TrigramQuery& r) const {
  if (op_ == Op::kNone || r.op_ == Op::kAll) return true;
  if (op_ == Op::kAll || r.op_ == Op::kNone) return false;
  // An AND implies whatever its trigrams alone imply.
  if (op_ == Op::kAnd) return TrigramsImply(trigrams_, r);
  // An OR of trigrams implies an OR that contains all of them.
  return r.op_ == Op::kOr && subs_.empty() && IsSubset(trigrams_, r.trigrams_);
}

bool TrigramQuery::Evaluate(
    absl::FunctionRef<bool(Trigram)> has_trigram) const {
  switch (op_) {
    case Op::kAll:
      return true;
    case Op::kNone:
      return false;
    case Op::kAnd:
      return std::all_of(trigrams_.begin(), trigrams_.end(), has_trigram) &&
             std::all_of(subs_.begin(), subs_.end(),
                         [has_trigram](const TrigramQuery& sub) {
                           return sub.Evaluate(has_trigram);
                         });
    case Op::kOr:
      return std::any_of(trigrams_.begin(), trigrams_.end(), has_trigram) ||
             std::any_of(subs_.begin(), subs_.end(),
                         [has_trigram](const TrigramQuery& sub) {
                           return sub.Evaluate(has_trigram);
                         });
  }
  return false;
}

std::string TrigramQuery::ToString() const {
  std::string out;
  AppendTo(out);
  return out;
}

void TrigramQuery::AppendTo(std::string& out) const {
  switch (op_) {
    case Op::kAll:
      out.append("All");
      return;
    case Op::kNone:
      out.append("None");
      return;
    case Op::kAnd:
    case Op::kOr:
      break;
  }
  const char* const separator = op_ == Op::kAnd ? " & " : " | ";
  bool first = true;
  for (const Trigram trigram : trigrams_) {
    if (!first) out.append(separator);
    first = false;
    out.append(TrigramToString(trigram));
  }
  for (const TrigramQuery& sub : subs_) {
    if (!first) out.append(separator);
    first = false;
    out.push_back('(');
    sub.AppendTo(out);
    out.push_back(')');
  }
}

bool TrigramQuery::WeakenInPlace(TrigramQuery& query, size_t max_or_fanout) {
  if (query.op_ == Op::kOr &&
      query.trigrams_.size() + query.subs_.size() > max_or_fanout) {
    query = All();
    return true;
  }
  bool changed = false;
  for (TrigramQuery& sub : query.subs_) {
    if (WeakenInPlace(sub, max_or_fanout)) changed = true;
  }
  if (!changed) return false;
  // A weakened subquery may merge into this node and make it too wide in
  // turn. Each round removes at least one trigram occurrence and rebuilding
  // never adds one, so this terminates.
  query =
      Rebuild(query.op_, std::move(query.trigrams_), std::move(query.subs_));
  WeakenInPlace(query, max_or_fanout);
  return true;
}

TrigramQuery WeakenExpensiveOrs(TrigramQuery query, int max_or_fanout) {
  DCHECK_GE(max_or_fanout, 1);
  TrigramQuery::WeakenInPlace(query, static_cast<size_t>(max_or_fanout));
  return query;
}

}  // namespace ferret

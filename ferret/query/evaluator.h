// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Evaluates a TrigramQuery over posting lists: finds the chunks in a window
// of chunk ids whose trigram sets satisfy the query.
//
// A TrigramQueryEvaluator is built once per query and index reader. It looks
// up the posting list of every trigram in the query, drops the parts that
// are provably empty, and orders the operands of each AND by estimated
// result size. It then evaluates any number of windows, reusing its buffers.
//
// Evaluation is exact: the result is the set of ids c in the window for
// which query.Evaluate() is true when "trigram t is present" means that c is
// in t's posting list. The strategy only affects cost:
//
//   * AND evaluates its first operand over the window and restricts each
//     later one to the candidates so far, stopping as soon as none are left.
//     A restricted trigram is intersected with IntersectWithPostingList,
//     which probes the skip table when the candidates are few.
//   * OR evaluates every operand under the OR's own restriction and returns
//     the union (UnionAll). Under an AND, no list is decoded beyond what the
//     candidates need.
//
// The template is instantiated for the two list types that list_ops.h
// supports: PostingListView (segments) and absl::Span<const ChunkId>
// (memtables and tests).

#ifndef FERRET_QUERY_EVALUATOR_H_
#define FERRET_QUERY_EVALUATOR_H_

#include <cstddef>
#include <optional>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/types/span.h"

#include "ferret/postings/ids.h"
#include "ferret/postings/list_ops.h"
#include "ferret/postings/posting_list.h"
#include "ferret/query/trigram_query.h"
#include "ferret/text/trigram.h"

namespace ferret {

// Thread-compatible: one evaluator per thread. Movable, not copyable.
template <typename List>
class TrigramQueryEvaluator final {
 public:
  // Looks up the posting list of each trigram in `query` with `find_list`,
  // which returns an empty list for a trigram that no chunk contains. The
  // lists must outlive the evaluator; `find_list` is not retained.
  TrigramQueryEvaluator(const TrigramQuery& query,
                        absl::FunctionRef<List(Trigram)> find_list);

  TrigramQueryEvaluator(TrigramQueryEvaluator&&) = default;
  TrigramQueryEvaluator& operator=(TrigramQueryEvaluator&&) = default;
  TrigramQueryEvaluator(const TrigramQueryEvaluator&) = delete;
  TrigramQueryEvaluator& operator=(const TrigramQueryEvaluator&) = delete;

  // Writes the ids in `window` that satisfy the query to `out`, in increasing
  // order. When the query is All, or reduces to All, that is every id in the
  // window, so callers keep windows within their id range.
  void Evaluate(ChunkRange window, ChunkIdBuffer& out);

  // An upper bound on the number of ids that satisfy the query, over all
  // windows: 0 when the query provably matches nothing, and SIZE_MAX when it
  // is All.
  size_t estimated_size() const { return root_.estimated_size; }

  // Returns true if no window can have a match, so evaluation can be skipped.
  bool matches_nothing() const { return root_.kind == Node::Kind::kNone; }

 private:
  struct Node {
    enum class Kind { kAll, kNone, kList, kAnd, kOr };

    Kind kind = Kind::kAll;
    // The posting list of a kList node.
    List list;
    // The operands of a kAnd or kOr node, at least two of them; a kAnd's are
    // in increasing order of estimated_size.
    std::vector<Node> operands;
    // An upper bound on the number of ids the node matches.
    size_t estimated_size = 0;
  };

  static Node Build(const TrigramQuery& query,
                    absl::FunctionRef<List(Trigram)> find_list);
  static Node MakeAnd(std::vector<Node> operands);
  static Node MakeOr(std::vector<Node> operands);

  // Writes the ids of `node` in `window` to `out`, or only those among
  // `restriction` if it is set; `restriction` lies in `window` and does not
  // point into `out`.
  void Run(const Node& node, ChunkRange window,
           std::optional<absl::Span<const ChunkId>> restriction,
           ChunkIdBuffer& out);
  void RunAnd(const Node& node, ChunkRange window,
              std::optional<absl::Span<const ChunkId>> restriction,
              ChunkIdBuffer& out);
  void RunOr(const Node& node, ChunkRange window,
             std::optional<absl::Span<const ChunkId>> restriction,
             ChunkIdBuffer& out);

  // Buffers for intermediate results, which nested operators take and give
  // back in stack order.
  ChunkIdBuffer TakeBuffer();
  void ReturnBuffer(ChunkIdBuffer buffer);

  Node root_;
  std::vector<ChunkIdBuffer> free_buffers_;
  // Working storage for one list operation at a time.
  ChunkIdBuffer scratch_;
};

extern template class TrigramQueryEvaluator<PostingListView>;
extern template class TrigramQueryEvaluator<absl::Span<const ChunkId>>;

}  // namespace ferret

#endif  // FERRET_QUERY_EVALUATOR_H_

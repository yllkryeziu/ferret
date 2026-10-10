// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/query/evaluator.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/types/span.h"

#include "ferret/postings/ids.h"
#include "ferret/postings/list_ops.h"
#include "ferret/postings/posting_list.h"
#include "ferret/query/trigram_query.h"
#include "ferret/text/trigram.h"

namespace ferret {
namespace {

constexpr size_t kUnbounded = std::numeric_limits<size_t>::max();

size_t SaturatingAdd(size_t a, size_t b) {
  return a > kUnbounded - b ? kUnbounded : a + b;
}

void FillRange(ChunkRange range, ChunkIdBuffer& out) {
  ChunkId* const ids = out.ResetForOverwrite(range.size());
  std::iota(ids, ids + range.size(), range.first);
  out.set_size(range.size());
}

}  // namespace

template <typename List>
TrigramQueryEvaluator<List>::TrigramQueryEvaluator(
    const TrigramQuery& query, absl::FunctionRef<List(Trigram)> find_list)
    : root_(Build(query, find_list)) {}

template <typename List>
void TrigramQueryEvaluator<List>::Evaluate(ChunkRange window,
                                           ChunkIdBuffer& out) {
  Run(root_, window, std::nullopt, out);
}

template <typename List>
auto TrigramQueryEvaluator<List>::Build(
    const TrigramQuery& query, absl::FunctionRef<List(Trigram)> find_list)
    -> Node {
  Node node;
  switch (query.op()) {
    case TrigramQuery::Op::kAll:
      node.kind = Node::Kind::kAll;
      node.estimated_size = kUnbounded;
      return node;
    case TrigramQuery::Op::kNone:
      node.kind = Node::Kind::kNone;
      return node;
    case TrigramQuery::Op::kAnd:
    case TrigramQuery::Op::kOr:
      break;
  }
  std::vector<Node> operands;
  operands.reserve(query.trigrams().size() + query.subs().size());
  for (const Trigram trigram : query.trigrams()) {
    Node& leaf = operands.emplace_back();
    leaf.list = find_list(trigram);
    leaf.estimated_size = leaf.list.size();
    // A trigram that no chunk contains is None, which the factories below
    // propagate.
    leaf.kind =
        leaf.estimated_size == 0 ? Node::Kind::kNone : Node::Kind::kList;
  }
  for (const TrigramQuery& sub : query.subs()) {
    operands.push_back(Build(sub, find_list));
  }
  return query.op() == TrigramQuery::Op::kAnd ? MakeAnd(std::move(operands))
                                              : MakeOr(std::move(operands));
}

// Every rewrite below preserves the node's meaning exactly: All is the
// identity of AND and absorbs OR, None absorbs AND and is the identity of OR,
// and nested operators of the same kind are associative. Nesting arises when
// dropping None operands reduces an OR to its only AND operand, or the
// reverse.
template <typename List>
auto TrigramQueryEvaluator<List>::MakeAnd(std::vector<Node> operands) -> Node {
  std::vector<Node> flat;
  for (Node& operand : operands) {
    switch (operand.kind) {
      case Node::Kind::kAll:
        break;
      case Node::Kind::kNone:
        return std::move(operand);
      case Node::Kind::kAnd:
        std::move(operand.operands.begin(), operand.operands.end(),
                  std::back_inserter(flat));
        break;
      case Node::Kind::kList:
      case Node::Kind::kOr:
        flat.push_back(std::move(operand));
        break;
    }
  }
  if (flat.empty()) {
    return Node{.kind = Node::Kind::kAll, .estimated_size = kUnbounded};
  }
  if (flat.size() == 1) return std::move(flat.front());
  std::stable_sort(flat.begin(), flat.end(), [](const Node& a, const Node& b) {
    return a.estimated_size < b.estimated_size;
  });
  Node node{.kind = Node::Kind::kAnd,
            .estimated_size = flat.front().estimated_size};
  node.operands = std::move(flat);
  return node;
}

template <typename List>
auto TrigramQueryEvaluator<List>::MakeOr(std::vector<Node> operands) -> Node {
  std::vector<Node> flat;
  for (Node& operand : operands) {
    switch (operand.kind) {
      case Node::Kind::kAll:
        return std::move(operand);
      case Node::Kind::kNone:
        break;
      case Node::Kind::kOr:
        std::move(operand.operands.begin(), operand.operands.end(),
                  std::back_inserter(flat));
        break;
      case Node::Kind::kList:
      case Node::Kind::kAnd:
        flat.push_back(std::move(operand));
        break;
    }
  }
  if (flat.empty()) return Node{.kind = Node::Kind::kNone};
  if (flat.size() == 1) return std::move(flat.front());
  size_t estimated_size = 0;
  for (const Node& operand : flat) {
    estimated_size = SaturatingAdd(estimated_size, operand.estimated_size);
  }
  Node node{.kind = Node::Kind::kOr, .estimated_size = estimated_size};
  node.operands = std::move(flat);
  return node;
}

template <typename List>
void TrigramQueryEvaluator<List>::Run(
    const Node& node, ChunkRange window,
    std::optional<absl::Span<const ChunkId>> restriction, ChunkIdBuffer& out) {
  switch (node.kind) {
    case Node::Kind::kAll:
      if (restriction.has_value()) {
        out.Assign(*restriction);
      } else {
        FillRange(window, out);
      }
      return;
    case Node::Kind::kNone:
      out.clear();
      return;
    case Node::Kind::kList:
      if (restriction.has_value()) {
        IntersectWithPostingList(*restriction, node.list, scratch_, out);
      } else {
        DecodeRange(node.list, window, out);
      }
      return;
    case Node::Kind::kAnd:
      RunAnd(node, window, restriction, out);
      return;
    case Node::Kind::kOr:
      RunOr(node, window, restriction, out);
      return;
  }
}

template <typename List>
void TrigramQueryEvaluator<List>::RunAnd(
    const Node& node, ChunkRange window,
    std::optional<absl::Span<const ChunkId>> restriction, ChunkIdBuffer& out) {
  // Each operand reads the previous result and writes the other buffer.
  ChunkIdBuffer other = TakeBuffer();
  ChunkIdBuffer* target = &out;
  ChunkIdBuffer* result = &other;
  for (const Node& operand : node.operands) {
    Run(operand, window, restriction, *target);
    restriction = target->ids();
    std::swap(target, result);
    if (result->empty()) break;
  }
  if (result != &out) swap(out, other);
  ReturnBuffer(std::move(other));
}

template <typename List>
void TrigramQueryEvaluator<List>::RunOr(
    const Node& node, ChunkRange window,
    std::optional<absl::Span<const ChunkId>> restriction, ChunkIdBuffer& out) {
  absl::InlinedVector<ChunkIdBuffer, 8> results;
  absl::InlinedVector<absl::Span<const ChunkId>, 8> lists;
  bool matched_all_candidates = false;
  for (const Node& operand : node.operands) {
    ChunkIdBuffer& result = results.emplace_back(TakeBuffer());
    Run(operand, window, restriction, result);
    lists.push_back(result.ids());
    // A restricted result is a subset of the restriction, so one of equal
    // size is the whole restriction, and so is the union.
    if (restriction.has_value() && result.size() == restriction->size()) {
      matched_all_candidates = true;
      break;
    }
  }
  if (matched_all_candidates) {
    out.Assign(*restriction);
  } else {
    UnionAll(lists, scratch_, out);
  }
  for (ChunkIdBuffer& result : results) ReturnBuffer(std::move(result));
}

template <typename List>
ChunkIdBuffer TrigramQueryEvaluator<List>::TakeBuffer() {
  if (free_buffers_.empty()) return ChunkIdBuffer();
  ChunkIdBuffer buffer = std::move(free_buffers_.back());
  free_buffers_.pop_back();
  return buffer;
}

template <typename List>
void TrigramQueryEvaluator<List>::ReturnBuffer(ChunkIdBuffer buffer) {
  free_buffers_.push_back(std::move(buffer));
}

template class TrigramQueryEvaluator<PostingListView>;
template class TrigramQueryEvaluator<absl::Span<const ChunkId>>;

}  // namespace ferret

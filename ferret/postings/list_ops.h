// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The posting-list operations that query evaluation is built from: decoding
// the part of a list inside a window, intersecting a candidate set with a
// list, and the union of several id lists. They run the kernels of
// ferret/simd on ChunkIdBuffers, which carry the output slack those kernels
// write into.
//
// Each operation on a posting list has two overloads with the same shape, so
// the evaluator is one template instantiated for both: one takes an encoded
// list (PostingListView, from a segment), the other a plain array
// (absl::Span<const ChunkId>, from the memtable and from tests). Every list
// and candidate set is strictly increasing, and so is every result. No output
// buffer may overlap an input.

#ifndef FERRET_POSTINGS_LIST_OPS_H_
#define FERRET_POSTINGS_LIST_OPS_H_

#include <cstddef>
#include <memory>

#include "absl/base/nullability.h"
#include "absl/types/span.h"

#include "ferret/postings/ids.h"
#include "ferret/postings/posting_list.h"

namespace ferret {

// A reusable array of ChunkIds that the kernels can write into directly: its
// storage always extends kSimdSlack elements past the requested capacity, and
// it never initializes elements, so resetting a scratch buffer costs nothing
// once it has grown. Storage is kept across resets and only ever grows.
// Movable, not copyable. Thread-compatible.
class ChunkIdBuffer final {
 public:
  ChunkIdBuffer() = default;
  ChunkIdBuffer(ChunkIdBuffer&& other) noexcept;
  ChunkIdBuffer& operator=(ChunkIdBuffer&& other) noexcept;
  ChunkIdBuffer(const ChunkIdBuffer&) = delete;
  ChunkIdBuffer& operator=(const ChunkIdBuffer&) = delete;
  ~ChunkIdBuffer() = default;

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  absl::Span<const ChunkId> ids() const { return {data_.get(), size_}; }

  // Empties the buffer and returns storage for `capacity` ids plus
  // kSimdSlack, with unspecified contents. After writing, pass the number of
  // ids written to set_size(). Invalidates earlier ids() spans.
  ChunkId* absl_nonnull ResetForOverwrite(size_t capacity);

  // Sets size() after ResetForOverwrite(capacity) has been called and the
  // first `size` elements written. Requires size <= capacity.
  void set_size(size_t size);

  void clear() { size_ = 0; }

  // Replaces the contents with a copy of `ids`, which must not point into
  // this buffer.
  void Assign(absl::Span<const ChunkId> ids);

  friend void swap(ChunkIdBuffer& a, ChunkIdBuffer& b) noexcept;

 private:
  std::unique_ptr<ChunkId[]> data_;
  // Usable elements, excluding the kSimdSlack that follow them.
  size_t capacity_ = 0;
  size_t size_ = 0;
};

// Writes the ids of `list` that lie in `range` to `out`. For an encoded list
// only the blocks that overlap `range` are decoded.
void DecodeRange(const PostingListView& list, ChunkRange range,
                 ChunkIdBuffer& out);
void DecodeRange(absl::Span<const ChunkId> list, ChunkRange range,
                 ChunkIdBuffer& out);

// Writes the ids of `candidates` that `list` contains to `out`, using
// `scratch` as working storage; the contents of `scratch` are unspecified
// afterwards. `candidates` may point into neither buffer.
//
// For an encoded list with at least twice as many blocks as there are
// candidates, each candidate is probed through the skip table
// (IntersectBySkipProbe), so most of the list is never decoded. Otherwise the
// part of the list between the first and the last candidate is decoded into
// `scratch` and intersected with Intersect(). An array is intersected
// directly and does not use `scratch`.
void IntersectWithPostingList(absl::Span<const ChunkId> candidates,
                              const PostingListView& list,
                              ChunkIdBuffer& scratch, ChunkIdBuffer& out);
void IntersectWithPostingList(absl::Span<const ChunkId> candidates,
                              absl::Span<const ChunkId> list,
                              ChunkIdBuffer& scratch, ChunkIdBuffer& out);

// Writes the union of `lists` to `out`, using `scratch` as working storage;
// the contents of `scratch` are unspecified afterwards. No list may point
// into either buffer.
//
// Two lists are merged with Union(). Three or more are set in a bitmap over
// the ids they span when it has at most 64 bits per input id, so building and
// scanning it costs O(total ids); sparser inputs are merged pairwise with
// Union(), shortest first.
void UnionAll(absl::Span<const absl::Span<const ChunkId>> lists,
              ChunkIdBuffer& scratch, ChunkIdBuffer& out);

}  // namespace ferret

#endif  // FERRET_POSTINGS_LIST_OPS_H_

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/postings/list_ops.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/types/span.h"

#include "ferret/base/ablation.h"
#include "ferret/base/stage_clock.h"
#include "ferret/postings/ids.h"
#include "ferret/postings/posting_list.h"
#include "ferret/simd/dispatch.h"
#include "ferret/simd/intersect.h"
#include "ferret/simd/union.h"

namespace ferret {
namespace {

using IdSpan = absl::Span<const ChunkId>;

// Returns the part of `list` inside [first, last].
IdSpan Slice(IdSpan list, ChunkId first, ChunkId last) {
  const auto begin = std::lower_bound(list.begin(), list.end(), first);
  const auto end = std::upper_bound(begin, list.end(), last);
  return IdSpan(begin, static_cast<size_t>(end - begin));
}

void UnionTwo(IdSpan a, IdSpan b, ChunkIdBuffer& out) {
  ChunkId* const ids = out.ResetForOverwrite(a.size() + b.size());
  out.set_size(Union(a, b, ids));
}

// Sets the bit of every id of `lists` in a bitmap over [first, last] that
// lives in `scratch`, then writes the set bits to `out` in increasing order.
void UnionByBitmap(absl::Span<const IdSpan> lists, ChunkId first, ChunkId last,
                   size_t total, ChunkIdBuffer& scratch, ChunkIdBuffer& out) {
  const size_t num_words = (size_t{last} - first) / 32 + 1;
  uint32_t* const words = scratch.ResetForOverwrite(num_words);
  std::memset(words, 0, num_words * sizeof(uint32_t));
  for (const IdSpan list : lists) {
    for (const ChunkId id : list) {
      const uint32_t offset = id - first;
      words[offset / 32] |= uint32_t{1} << (offset % 32);
    }
  }
  ChunkId* const ids = out.ResetForOverwrite(total);
  size_t count = 0;
  for (size_t i = 0; i < num_words; ++i) {
    // Ids fit in 32 bits, so the base of every word does too.
    const auto base = static_cast<ChunkId>(first + 32 * i);
    for (uint32_t word = words[i]; word != 0; word &= word - 1) {
      ids[count++] = base + static_cast<ChunkId>(std::countr_zero(word));
    }
  }
  out.set_size(count);
}

// Merges `lists`, sorted by increasing size, into an accumulator that
// alternates between `out` and `scratch`, so each merge reads one buffer and
// writes the other.
void UnionPairwise(absl::Span<const IdSpan> lists, ChunkIdBuffer& scratch,
                   ChunkIdBuffer& out) {
  DCHECK_GE(lists.size(), size_t{2});
  ChunkIdBuffer* accumulator = &out;
  ChunkIdBuffer* other = &scratch;
  UnionTwo(lists[0], lists[1], *accumulator);
  for (size_t i = 2; i < lists.size(); ++i) {
    UnionTwo(accumulator->ids(), lists[i], *other);
    std::swap(accumulator, other);
  }
  if (accumulator != &out) swap(out, scratch);
}

}  // namespace

ChunkIdBuffer::ChunkIdBuffer(ChunkIdBuffer&& other) noexcept
    : data_(std::move(other.data_)),
      capacity_(std::exchange(other.capacity_, 0)),
      size_(std::exchange(other.size_, 0)) {}

ChunkIdBuffer& ChunkIdBuffer::operator=(ChunkIdBuffer&& other) noexcept {
  data_ = std::move(other.data_);
  capacity_ = std::exchange(other.capacity_, 0);
  size_ = std::exchange(other.size_, 0);
  return *this;
}

ChunkId* ChunkIdBuffer::ResetForOverwrite(size_t capacity) {
  size_ = 0;
  if (data_ == nullptr || capacity > capacity_) {
    // Doubling keeps a buffer that is reset with slowly growing sizes from
    // reallocating on every reset.
    capacity_ = std::max(capacity, 2 * capacity_);
    data_ = std::make_unique_for_overwrite<ChunkId[]>(capacity_ + kSimdSlack);
  }
  return data_.get();
}

void ChunkIdBuffer::set_size(size_t size) {
  DCHECK_LE(size, capacity_);
  size_ = size;
}

void ChunkIdBuffer::Assign(absl::Span<const ChunkId> ids) {
  ChunkId* const out = ResetForOverwrite(ids.size());
  if (!ids.empty()) std::memcpy(out, ids.data(), ids.size() * sizeof(ChunkId));
  size_ = ids.size();
}

void swap(ChunkIdBuffer& a, ChunkIdBuffer& b) noexcept {
  using std::swap;
  swap(a.data_, b.data_);
  swap(a.capacity_, b.capacity_);
  swap(a.size_, b.size_);
}

void DecodeRange(const PostingListView& list, ChunkRange range,
                 ChunkIdBuffer& out) {
  const StageTimer timer(&StageClock::decode_ns);
  ChunkId* const ids =
      out.ResetForOverwrite(std::min(list.size(), range.size()));
  out.set_size(list.DecodeRange(range.first, range.last, ids));
}

void DecodeRange(absl::Span<const ChunkId> list, ChunkRange range,
                 ChunkIdBuffer& out) {
  out.Assign(Slice(list, range.first, range.last));
}

void IntersectWithPostingList(absl::Span<const ChunkId> candidates,
                              const PostingListView& list,
                              ChunkIdBuffer& scratch, ChunkIdBuffer& out) {
  if (candidates.empty() || list.size() == 0) {
    out.clear();
    return;
  }
  const bool probe_pays =
      candidates.size() < static_cast<size_t>(list.num_blocks()) / 2;
  if (probe_pays && AblationOn(Ablation::no_skip_probe)) [[unlikely]] {
    ++ThisThreadAblationCounts().unprobed_intersections;
  }
  if (probe_pays && !AblationOn(Ablation::no_skip_probe)) {
    // The probe decodes the blocks it needs itself; it counts as intersecting.
    const StageTimer timer(&StageClock::intersect_ns);
    ChunkId* const ids = out.ResetForOverwrite(candidates.size());
    out.set_size(IntersectBySkipProbe(candidates, list, ids));
    return;
  }
  DecodeRange(list, ChunkRange{candidates.front(), candidates.back()}, scratch);
  const StageTimer timer(&StageClock::intersect_ns);
  ChunkId* const ids =
      out.ResetForOverwrite(std::min(candidates.size(), scratch.size()));
  out.set_size(Intersect(candidates, scratch.ids(), ids));
}

void IntersectWithPostingList(absl::Span<const ChunkId> candidates,
                              absl::Span<const ChunkId> list,
                              ChunkIdBuffer& /*scratch*/, ChunkIdBuffer& out) {
  if (candidates.empty()) {
    out.clear();
    return;
  }
  const StageTimer timer(&StageClock::intersect_ns);
  const IdSpan slice = Slice(list, candidates.front(), candidates.back());
  ChunkId* const ids =
      out.ResetForOverwrite(std::min(candidates.size(), slice.size()));
  out.set_size(Intersect(candidates, slice, ids));
}

void UnionAll(absl::Span<const absl::Span<const ChunkId>> lists,
              ChunkIdBuffer& scratch, ChunkIdBuffer& out) {
  const StageTimer timer(&StageClock::intersect_ns);
  absl::InlinedVector<IdSpan, 16> nonempty;
  size_t total = 0;
  for (const IdSpan list : lists) {
    if (list.empty()) continue;
    nonempty.push_back(list);
    total += list.size();
  }
  switch (nonempty.size()) {
    case 0:
      out.clear();
      return;
    case 1:
      out.Assign(nonempty[0]);
      return;
    case 2:
      UnionTwo(nonempty[0], nonempty[1], out);
      return;
    default:
      break;
  }
  ChunkId first = nonempty[0].front();
  ChunkId last = nonempty[0].back();
  for (const IdSpan list : nonempty) {
    first = std::min(first, list.front());
    last = std::max(last, list.back());
  }
  if ((size_t{last} - first) / 64 < total) {
    UnionByBitmap(nonempty, first, last, total, scratch, out);
    return;
  }
  std::sort(nonempty.begin(), nonempty.end(),
            [](IdSpan a, IdSpan b) { return a.size() < b.size(); });
  UnionPairwise(nonempty, scratch, out);
}

}  // namespace ferret

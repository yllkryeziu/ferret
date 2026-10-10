// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/postings/posting_list.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "ferret/postings/bitpack.h"
#include "ferret/postings/ids.h"
#include "ferret/postings/internal/posting_list_format.h"
#include "ferret/postings/varint.h"

namespace ferret {
namespace {

using postings_internal::BitWidth;
using postings_internal::kBeforeFirstId;
using postings_internal::LoadSkipEntry;
using postings_internal::Offset16;
using postings_internal::PayloadRegionOffset;
using postings_internal::SkipEntry;

constexpr int64_t kMaxChunkId = std::numeric_limits<ChunkId>::max();

template <typename... Args>
absl::Status PostingListError(const Args&... args) {
  return absl::DataLossError(absl::StrCat("posting list: ", args...));
}

// Returns true if `span`, the sum of gap - 1 over a block, is one that the
// encoder can produce at `bit_width`: each gap - 1 is below 2^bit_width, and
// the width is the smallest that holds the largest of them.
bool BitWidthMatchesSpan(int bit_width, int64_t span) {
  if (bit_width == 0) return span == 0;
  return (int64_t{1} << (bit_width - 1)) <= span &&
         span <= kPostingBlockSize * ((int64_t{1} << bit_width) - 1);
}

// The widest block for which ending at the right id proves a decoded block
// well formed. Its 128 gaps sum to some S in [128, 2^32], and the true sum,
// last - base, lies in [128, 2^32] too, so if the two agree modulo 2^32 they
// are equal: the running id never wrapped around. A wider block can wrap and
// still end at its last id, so it is checked id by id; valid lists have few
// such blocks, as each spans at least 2^25 ids.
constexpr int kMaxBitWidthCheckedByLastId = 25;

// Returns true if `ids`, a full block of `bit_width` decoded after `base` (-1
// for the first block), is strictly increasing from above `base` up to `last`,
// as every block the encoder writes is.
bool IsWellFormedBlock(int64_t base, ChunkId last, int bit_width,
                       const ChunkId* ids) {
  if (ids[kPostingBlockSize - 1] != last) return false;
  if (bit_width <= kMaxBitWidthCheckedByLastId) return true;
  int64_t previous = base;
  for (int i = 0; i < kPostingBlockSize; ++i) {
    if (ids[i] <= previous) return false;
    previous = ids[i];
  }
  return true;
}

}  // namespace

absl::StatusOr<PostingListView> PostingListView::FromEncoded(
    absl::string_view bytes) {
  absl::string_view input = bytes;
  const std::optional<uint32_t> count = ConsumeVarint32(input);
  if (!count.has_value()) return PostingListError("truncated count");
  if (IsInlinedPostingList(*count)) {
    return PostingListError("count ", *count, " belongs in the dictionary");
  }
  if (*count > kMaxPostingListSize) {
    return PostingListError("count ", *count, " exceeds the limit of ",
                            kMaxPostingListSize);
  }
  PostingListView view;
  view.size_ = *count;
  // The last id validated so far; -1 before the first.
  int64_t previous = -1;
  const int num_full_blocks = view.num_full_blocks();
  if (num_full_blocks > 0) {
    const auto available = static_cast<int64_t>(bytes.size());
    const int64_t count_bytes = available - static_cast<int64_t>(input.size());
    const int64_t payload_offset =
        PayloadRegionOffset(count_bytes, num_full_blocks);
    if (payload_offset > available) {
      return PostingListError("skip table ends at byte ", payload_offset,
                              " beyond the ", available, " available");
    }
    const int64_t skip_table_offset =
        payload_offset -
        num_full_blocks * static_cast<int64_t>(sizeof(SkipEntry));
    for (int64_t i = count_bytes; i < skip_table_offset; ++i) {
      if (bytes[static_cast<size_t>(i)] != '\0') {
        return PostingListError("nonzero padding at byte ", i);
      }
    }
    const auto* data = reinterpret_cast<const uint8_t*>(bytes.data());
    view.skip_table_ = data + skip_table_offset;
    view.payload_ = data + payload_offset;
    int64_t offset16 = 0;
    for (int block = 0; block < num_full_blocks; ++block) {
      const SkipEntry entry = LoadSkipEntry(view.skip_table_, block);
      const int bit_width = BitWidth(entry);
      if (bit_width > kMaxBitPackWidth) {
        return PostingListError("block ", block, " bit width ", bit_width,
                                " exceeds ", kMaxBitPackWidth);
      }
      if (Offset16(entry) != offset16) {
        return PostingListError("block ", block, " payload offset ",
                                Offset16(entry), " is not the expected ",
                                offset16, " (16-byte units)");
      }
      const int64_t span =
          int64_t{entry.last_chunk} - previous - kPostingBlockSize;
      if (span < 0) {
        return PostingListError("block ", block, " last id ", entry.last_chunk,
                                " is less than 128 above the previous ",
                                previous);
      }
      if (!BitWidthMatchesSpan(bit_width, span)) {
        return PostingListError("block ", block, " bit width ", bit_width,
                                " does not match its id span ", span);
      }
      offset16 += bit_width;
      previous = entry.last_chunk;
    }
    const int64_t tail_offset =
        payload_offset + offset16 * kPostingPayloadAlignment;
    if (tail_offset > available) {
      return PostingListError("block payloads end at byte ", tail_offset,
                              " beyond the ", available, " available");
    }
    input = bytes.substr(static_cast<size_t>(tail_offset));
  }
  const absl::string_view tail = input;
  const uint32_t tail_size = view.size_ % kPostingBlockSize;
  for (uint32_t i = 0; i < tail_size; ++i) {
    const std::optional<uint32_t> gap = ConsumeVarint32(input);
    if (!gap.has_value()) {
      return PostingListError("truncated gap ", i, " of ", tail_size);
    }
    previous += int64_t{*gap} + 1;
    if (previous > kMaxChunkId) {
      return PostingListError("gap ", i, " of ", tail_size,
                              " overflows 32-bit ids");
    }
  }
  view.tail_ = tail.substr(0, tail.size() - input.size());
  view.last_ = static_cast<ChunkId>(previous);
  return view;
}

absl::StatusOr<PostingListView> PostingListView::FromInlined(
    absl::Span<const ChunkId> ids) {
  if (!IsInlinedPostingList(ids.size())) {
    return PostingListError(ids.size(), " ids cannot be inlined");
  }
  for (size_t i = 1; i < ids.size(); ++i) {
    if (ids[i] <= ids[i - 1]) {
      return PostingListError("inlined ids ", ids[i - 1], " and ", ids[i],
                              " are not increasing");
    }
  }
  PostingListView view;
  view.size_ = static_cast<uint32_t>(ids.size());
  std::copy(ids.begin(), ids.end(), view.inlined_.begin());
  if (!ids.empty()) view.last_ = ids.back();
  return view;
}

int PostingListView::num_blocks() const {
  return static_cast<int>((size_ + kPostingBlockSize - 1) / kPostingBlockSize);
}

int PostingListView::num_full_blocks() const {
  return static_cast<int>(size_ / kPostingBlockSize);
}

ChunkId PostingListView::block_last(int block) const {
  DCHECK(0 <= block && block < num_blocks()) << block;
  return block < num_full_blocks()
             ? LoadSkipEntry(skip_table_, block).last_chunk
             : last_;
}

int PostingListView::FindBlock(ChunkId target, int first_block) const {
  const int end = num_blocks();
  DCHECK(0 <= first_block && first_block <= end) << first_block;
  // Gallop to a bracket [low, high) whose end, if not `end`, is a block at
  // or past `target`, then binary search it.
  int low = first_block;
  int high = first_block;
  for (int step = 1; high < end && block_last(high) < target; step *= 2) {
    low = high + 1;
    high = first_block + step;
  }
  high = std::min(high, end);
  while (low < high) {
    const int middle = low + (high - low) / 2;
    if (block_last(middle) < target) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

int PostingListView::DecodeBlock(int block, ChunkId* absl_nonnull out) const {
  DCHECK(0 <= block && block < num_blocks()) << block;
  const ChunkId base = block == 0 ? kBeforeFirstId : block_last(block - 1);
  if (block >= num_full_blocks()) return DecodeShortBlock(base, out);
  const SkipEntry entry = LoadSkipEntry(skip_table_, block);
  const int bit_width = BitWidth(entry);
  const size_t offset = size_t{Offset16(entry)} * kPostingPayloadAlignment;
  UnpackDeltaBlock(DeltaCoding::kD1, base, payload_ + offset, bit_width, out);
  if (!IsWellFormedBlock(block == 0 ? -1 : int64_t{base}, entry.last_chunk,
                         bit_width, out)) [[unlikely]] {
    // Only a corrupt payload gets here, one that FromEncoded cannot detect.
    // Its ids may be unordered or outside the block, which would break
    // callers that rely on sorted input or index by id, so the block becomes
    // the 128 ids that end at its last id, which FromEncoded validated.
    std::iota(out, out + kPostingBlockSize,
              entry.last_chunk - ChunkId{kPostingBlockSize - 1});
  }
  return kPostingBlockSize;
}

int PostingListView::DecodeShortBlock(ChunkId base,
                                      ChunkId* absl_nonnull out) const {
  if (IsInlinedPostingList(size_)) {
    std::copy_n(inlined_.begin(), size_, out);
    return static_cast<int>(size_);
  }
  const int count = static_cast<int>(size_ % kPostingBlockSize);
  absl::string_view input = tail_;
  ChunkId id = base;
  for (int i = 0; i < count; ++i) {
    // FromEncoded validated every gap.
    const std::optional<uint32_t> gap = ConsumeVarint32(input);
    DCHECK(gap.has_value());
    id += *gap + 1;
    out[i] = id;
  }
  return count;
}

void PostingListView::DecodeAll(ChunkId* absl_nonnull out) const {
  const int end = num_blocks();
  for (int block = 0; block < end; ++block) {
    out += DecodeBlock(block, out);
  }
}

size_t PostingListView::DecodeRange(ChunkId first, ChunkId last,
                                    ChunkId* absl_nonnull out) const {
  DCHECK_LE(first, last);
  ChunkId* const begin = out;
  const int end = num_blocks();
  for (int block = FindBlock(first); block < end; ++block) {
    // Block `block` holds ids in [lowest, block_last(block)].
    const ChunkId lowest = block == 0 ? 0 : block_last(block - 1) + 1;
    if (lowest > last) break;
    if (first <= lowest && block_last(block) <= last) {
      out += DecodeBlock(block, out);
      continue;
    }
    ChunkId ids[kPostingBlockSize];
    const int count = DecodeBlock(block, ids);
    // The block is sorted, so the ids in range are ids[below, not_above).
    int below = 0;
    int not_above = 0;
    for (int i = 0; i < count; ++i) {
      below += ids[i] < first ? 1 : 0;
      not_above += ids[i] <= last ? 1 : 0;
    }
    out = std::copy(ids + below, ids + not_above, out);
  }
  return static_cast<size_t>(out - begin);
}

size_t IntersectBySkipProbe(absl::Span<const ChunkId> candidates,
                            const PostingListView& list,
                            ChunkId* absl_nonnull out) {
  DCHECK(std::adjacent_find(candidates.begin(), candidates.end(),
                            std::greater_equal<ChunkId>()) == candidates.end())
      << "candidates are not strictly increasing";
  const int num_blocks = list.num_blocks();
  ChunkId ids[kPostingBlockSize];
  int decoded_block = -1;
  int decoded_size = 0;
  int block = 0;
  size_t count = 0;
  for (const ChunkId candidate : candidates) {
    // Every id before `block` is below the previous candidate, so below this
    // one too, as FindBlock requires.
    block = list.FindBlock(candidate, block);
    if (block == num_blocks) break;
    if (block != decoded_block) {
      decoded_size = list.DecodeBlock(block, ids);
      decoded_block = block;
    }
    if (std::binary_search(ids, ids + decoded_size, candidate)) {
      out[count++] = candidate;
    }
  }
  return count;
}

PostingCursor::PostingCursor(PostingListView list)
    : list_(list), num_blocks_(list_.num_blocks()) {
  if (num_blocks_ > 0) LoadBlock(0);
}

ChunkId PostingCursor::value() const {
  DCHECK(!done());
  return ids_[static_cast<size_t>(position_)];
}

void PostingCursor::Next() {
  DCHECK(!done());
  if (++position_ < block_size_) return;
  if (block_ + 1 < num_blocks_) {
    LoadBlock(block_ + 1);
  } else {
    block_ = num_blocks_;
  }
}

bool PostingCursor::SeekGE(ChunkId target) {
  if (done()) return false;
  if (list_.block_last(block_) < target) {
    const int block = list_.FindBlock(target, block_ + 1);
    if (block == num_blocks_) {
      block_ = num_blocks_;
      return false;
    }
    LoadBlock(block);
  }
  // The block is sorted and ends at block_last(block_) >= target, so the
  // first id >= target is in it, just after the ids below `target`; counting
  // those without branches beats a search.
  int below = 0;
  for (int i = position_; i < block_size_; ++i) {
    below += ids_[static_cast<size_t>(i)] < target ? 1 : 0;
  }
  position_ += below;
  DCHECK_LT(position_, block_size_);
  return true;
}

void PostingCursor::LoadBlock(int block) {
  block_ = block;
  block_size_ = list_.DecodeBlock(block, ids_.data());
  position_ = 0;
}

}  // namespace ferret

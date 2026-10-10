// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Read access to posting lists: PostingListView validates and decodes one
// list, a strictly increasing sequence of ChunkIds, and PostingCursor walks
// it with skip-table seeks. posting_list_encoder.h writes the format.
//
// Format. Integers are little-endian. The gap of an id is its distance from
// the previous id; the first id's predecessor is taken to be -1, so its
// stored gap - 1 is the id itself.
//
//   varint count                    count > kMaxInlinedPostings
//   if count < 128:
//     varint (gap - 1), count times
//   else, with B = count / 128 full blocks:
//     zero bytes, 0 to 15 of them, so that the payload region starts at a
//         multiple of kPostingPayloadAlignment
//     SkipEntry[B] { u32 last_chunk; u32 offset16 << 8 | bit_width }
//     payload region: block b holds ids 128b .. 128b + 127, bit-packed as
//         DeltaCoding::kD1 deltas from the previous id (bitpack.h) in
//         16 * bit_width bytes at byte 16 * offset16 of the region, where
//         offset16 is the sum of the preceding blocks' widths
//     varint (gap - 1), count % 128 times: the tail, after the last payload
//
// A list of at most kMaxInlinedPostings ids has no encoding; the segment
// dictionary stores its ids and FromInlined views them.
//
// Blocks. A view presents a list as num_blocks() = ceil(size() / 128) blocks
// of kPostingBlockSize ids, of which only the last may be shorter: the tail,
// or the whole list when size() < 128. block_last() is the skip table
// extended by that short block, so FindBlock() locates any id by binary
// search and every block decodes independently.
//
// Validation. FromEncoded checks, once, everything the varints, the padding
// and the skip table can reveal, in O(num_blocks() + kPostingBlockSize) time.
// After that no accessor fails, and none reads outside the bytes given to
// FromEncoded, whatever those bytes are; in particular no padding is needed
// after a list. A corrupt block payload can only be detected by decoding it,
// so DecodeBlock checks each block against its skip entry (one comparison for
// all but the rare blocks wider than 25 bits) and turns a block that fails
// into the 128 ids that end at its last id. Every accessor therefore sees a
// strictly increasing list whose ids lie in their blocks' skip-table bounds,
// even for arbitrary bytes: corruption can make ids wrong, but never
// unordered or larger than the validated block_last(num_blocks() - 1). A
// caller that indexes a table by id checks that one id against the table's
// size. `ferret fsck` finds payload corruption through the section CRC.
//
// Alignment. Block payloads start at multiples of kPostingPayloadAlignment
// (16) bytes from the start of the encoded list, and skip entries at
// multiples of 8, so a list placed at a 16-byte-aligned address gets aligned
// NEON loads. Alignment only affects speed: decoding is correct at any
// address.

#ifndef FERRET_POSTINGS_POSTING_LIST_H_
#define FERRET_POSTINGS_POSTING_LIST_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "ferret/postings/ids.h"

namespace ferret {

// Ids per block. Every block but the last of a list has exactly this many.
inline constexpr int kPostingBlockSize = 128;

// Lists of at most this many ids are stored in their dictionary entry instead
// of being encoded.
inline constexpr size_t kMaxInlinedPostings = 2;

// The longest list the format can describe. Block offsets have 24 bits in
// 16-byte units, and a block takes at most 32 units.
inline constexpr size_t kMaxPostingListSize = size_t{1} << 26;

// Alignment of block payloads relative to the start of an encoded list.
inline constexpr int kPostingPayloadAlignment = 16;

// Returns true if a list of `size` ids belongs inline in its dictionary entry
// (read back with PostingListView::FromInlined), false if it is encoded with
// AppendEncodedPostingList (read back with PostingListView::FromEncoded).
constexpr bool IsInlinedPostingList(size_t size) {
  return size <= kMaxInlinedPostings;
}

// A validated, non-owning view of one posting list. Cheap to copy: it holds
// pointers into the encoded bytes and never allocates. A default-constructed
// view is the empty list. Thread-compatible.
class PostingListView final {
 public:
  PostingListView() = default;
  PostingListView(const PostingListView&) = default;
  PostingListView& operator=(const PostingListView&) = default;

  // Validates the encoded list at the front of `bytes` and returns a view of
  // it. `bytes` may extend past the end of the list, for example to the end
  // of the postings section; only the bytes the encoding describes are read.
  // Returns DataLoss if the list is truncated or inconsistent: a truncated or
  // oversized varint, a count that must be inlined or exceeds
  // kMaxPostingListSize, ids beyond UINT32_MAX, nonzero padding, a skip table
  // or payload that does not fit in `bytes`, a bit width above 32, an offset
  // that is not the sum of the preceding widths, or a skip entry that is not
  // at least 128 above its predecessor or whose width does not match its id
  // span. The view is valid while `bytes` is alive.
  // O(num_blocks() + kPostingBlockSize).
  static absl::StatusOr<PostingListView> FromEncoded(
      absl::string_view bytes ABSL_ATTRIBUTE_LIFETIME_BOUND);

  // Returns a view of an inlined list, copying its ids. Returns DataLoss
  // unless ids.size() <= kMaxInlinedPostings and the ids strictly increase.
  static absl::StatusOr<PostingListView> FromInlined(
      absl::Span<const ChunkId> ids);

  // The number of ids.
  size_t size() const { return size_; }

  // ceil(size() / kPostingBlockSize).
  int num_blocks() const;

  // Returns the largest id of block `block`: the skip entry's last_chunk, or
  // the list's last id for the short last block. Strictly increasing in
  // `block`. Requires 0 <= block < num_blocks(). O(1).
  ChunkId block_last(int block) const;

  // Returns the first block at or after `first_block` whose block_last() is
  // at least `target`, or num_blocks() if there is none. This is the block
  // that would hold the first id >= target, provided the ids before
  // `first_block` are all below `target`. Requires
  // 0 <= first_block <= num_blocks(). Gallops from `first_block`, so it costs
  // O(log(result - first_block)) skip-table probes.
  int FindBlock(ChunkId target, int first_block = 0) const;

  // Writes the ids of block `block` to `out` and returns their number:
  // kPostingBlockSize, or fewer for the last block. `out` needs room for
  // kPostingBlockSize ids. Requires 0 <= block < num_blocks().
  [[nodiscard]] int DecodeBlock(int block, ChunkId* absl_nonnull out) const;

  // Writes all size() ids to `out`, which needs room for exactly that many.
  void DecodeAll(ChunkId* absl_nonnull out) const;

  // Writes the ids in [first, last], in increasing order, to `out` and returns
  // their number. `out` needs room for min(size(), last - first + 1) ids;
  // nothing past the returned count is written.
  // Decodes only the blocks overlapping [first, last], found through the skip
  // table, which is what windowed evaluation over skip-block-aligned windows
  // relies on. Requires first <= last.
  [[nodiscard]] size_t DecodeRange(ChunkId first, ChunkId last,
                                   ChunkId* absl_nonnull out) const;

 private:
  int num_full_blocks() const;
  // Decodes the short last block from `base`, the last id of the previous
  // block.
  int DecodeShortBlock(ChunkId base, ChunkId* absl_nonnull out) const;

  // The skip table and the payload region of a blocked list.
  const uint8_t* absl_nullable skip_table_ = nullptr;
  const uint8_t* absl_nullable payload_ = nullptr;
  // The varint gaps of an encoded short last block.
  absl::string_view tail_;
  uint32_t size_ = 0;
  // The largest id, if size_ > 0.
  ChunkId last_ = 0;
  // The ids of an inlined list, which has no bytes to point into.
  std::array<ChunkId, kMaxInlinedPostings> inlined_ = {};
};

// Writes the candidates that `list` contains to `out`, in increasing order,
// and returns their number. Requires `candidates` to be strictly increasing
// (DCHECKed); `out` needs room for candidates.size() ids.
//
// This is the skip-table membership probe that intersects a few candidates
// with a long list without decoding all of it; IntersectWithPostingList
// (list_ops.h) uses it when candidates.size() < list.num_blocks() / 2. Each
// candidate costs a gallop over the skip table from the previous candidate's
// block, O(log(blocks skipped)), plus a binary search in its block, and each
// block landed in is decoded once.
[[nodiscard]] size_t IntersectBySkipProbe(absl::Span<const ChunkId> candidates,
                                          const PostingListView& list,
                                          ChunkId* absl_nonnull out);

// Forward iteration over a PostingListView with skip-table seeks. The cursor
// decodes one block at a time into an internal buffer, so it touches only the
// blocks it lands in.
//
// Valid while the list's bytes are alive. Thread-compatible.
class PostingCursor final {
 public:
  // Positions the cursor at the first id of `list`, or at the end if `list`
  // is empty.
  explicit PostingCursor(PostingListView list);
  PostingCursor(const PostingCursor&) = default;
  PostingCursor& operator=(const PostingCursor&) = default;

  // True once the cursor has moved past the last id.
  bool done() const { return block_ == num_blocks_; }

  // The current id. Requires !done().
  ChunkId value() const;

  // Moves to the next id. Requires !done().
  void Next();

  // Moves to the first id >= target, or to the end if there is none, and
  // returns !done(). Never moves backward: if value() >= target already, the
  // cursor stays put.
  bool SeekGE(ChunkId target);

 private:
  void LoadBlock(int block);

  PostingListView list_;
  int num_blocks_;
  int block_ = 0;
  int block_size_ = 0;
  int position_ = 0;
  std::array<ChunkId, kPostingBlockSize> ids_ = {};
};

}  // namespace ferret

#endif  // FERRET_POSTINGS_POSTING_LIST_H_

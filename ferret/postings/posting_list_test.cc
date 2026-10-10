// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/postings/posting_list.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/postings/ids.h"
#include "ferret/postings/posting_list_encoder.h"
#include "ferret/postings/testing/random_posting_lists.h"
#include "ferret/postings/varint.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/test_seed.h"

namespace ferret {
namespace {

using ::absl_testing::StatusIs;
using ::ferret_testing::RandomPostingList;
using ::ferret_testing::UniformBelow;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

constexpr ChunkId kMaxId = std::numeric_limits<ChunkId>::max();

std::string Encode(absl::Span<const ChunkId> ids) {
  std::string out;
  AppendEncodedPostingList(ids, out);
  return out;
}

// A copy of some bytes in a heap block of exactly their size, so that ASan
// reports any read past the end.
class HeapBytes final {
 public:
  explicit HeapBytes(absl::string_view bytes)
      : data_(std::make_unique<char[]>(bytes.size())), size_(bytes.size()) {
    std::memcpy(data_.get(), bytes.data(), bytes.size());
  }
  HeapBytes(const HeapBytes&) = delete;
  HeapBytes& operator=(const HeapBytes&) = delete;

  absl::string_view view() const {
    return absl::string_view(data_.get(), size_);
  }

 private:
  std::unique_ptr<char[]> data_;
  size_t size_;
};

// The decoders below write into heap blocks of exactly the documented
// capacity, so that ASan reports any write past it.

std::vector<ChunkId> DecodeAll(const PostingListView& view) {
  auto out = std::make_unique<ChunkId[]>(view.size());
  view.DecodeAll(out.get());
  return std::vector<ChunkId>(out.get(), out.get() + view.size());
}

std::vector<ChunkId> DecodeRange(const PostingListView& view, ChunkId first,
                                 ChunkId last) {
  const auto capacity = static_cast<size_t>(
      std::min<uint64_t>(view.size(), uint64_t{last} - first + 1));
  auto out = std::make_unique<ChunkId[]>(capacity);
  const size_t count = view.DecodeRange(first, last, out.get());
  EXPECT_LE(count, capacity);
  return std::vector<ChunkId>(out.get(), out.get() + std::min(count, capacity));
}

std::vector<ChunkId> SkipProbe(absl::Span<const ChunkId> candidates,
                               const PostingListView& view) {
  auto out = std::make_unique<ChunkId[]>(candidates.size());
  const size_t count = IntersectBySkipProbe(candidates, view, out.get());
  EXPECT_LE(count, candidates.size());
  return std::vector<ChunkId>(out.get(),
                              out.get() + std::min(count, candidates.size()));
}

bool IsStrictlyIncreasing(absl::Span<const ChunkId> ids) {
  return std::adjacent_find(ids.begin(), ids.end(),
                            std::greater_equal<ChunkId>()) == ids.end();
}

std::vector<ChunkId> ReferenceRange(const std::vector<ChunkId>& ids,
                                    ChunkId first, ChunkId last) {
  std::vector<ChunkId> in_range;
  std::copy_if(ids.begin(), ids.end(), std::back_inserter(in_range),
               [&](ChunkId id) { return first <= id && id <= last; });
  return in_range;
}

// Returns a probe near `ids`: an id, a neighbor of one, or anything at all.
ChunkId RandomTarget(std::mt19937_64& rng, const std::vector<ChunkId>& ids) {
  const uint64_t kind = UniformBelow(rng, 5);
  if (ids.empty() || kind == 4) {
    const uint64_t extreme = UniformBelow(rng, 4);
    if (extreme == 0) return 0;
    if (extreme == 1) return kMaxId;
    return static_cast<ChunkId>(rng());
  }
  const ChunkId id = ids[UniformBelow(rng, ids.size())];
  if (kind == 0) return id - 1;
  if (kind == 1) return id + 1;
  return id;
}

// Checks every accessor of `view` against `ids`, the list it must hold.
void ExpectViewHolds(const PostingListView& view,
                     const std::vector<ChunkId>& ids, std::mt19937_64& rng) {
  ASSERT_EQ(view.size(), ids.size());
  const int num_blocks = static_cast<int>((ids.size() + kPostingBlockSize - 1) /
                                          kPostingBlockSize);
  ASSERT_EQ(view.num_blocks(), num_blocks);
  for (int block = 0; block < num_blocks; ++block) {
    const auto begin = static_cast<size_t>(block) * kPostingBlockSize;
    const size_t end = std::min(begin + kPostingBlockSize, ids.size());
    ASSERT_EQ(view.block_last(block), ids[end - 1]) << "block " << block;
    auto out = std::make_unique<ChunkId[]>(kPostingBlockSize);
    ASSERT_EQ(view.DecodeBlock(block, out.get()),
              static_cast<int>(end - begin));
    ASSERT_TRUE(std::equal(ids.begin() + static_cast<ptrdiff_t>(begin),
                           ids.begin() + static_cast<ptrdiff_t>(end),
                           out.get()))
        << "block " << block;
  }
  ASSERT_EQ(DecodeAll(view), ids);

  for (int probe = 0; probe < 8; ++probe) {
    const ChunkId target = RandomTarget(rng, ids);
    const auto first_block = static_cast<int>(
        UniformBelow(rng, static_cast<uint64_t>(num_blocks) + 1));
    int expected = first_block;
    while (expected < num_blocks && view.block_last(expected) < target) {
      ++expected;
    }
    ASSERT_EQ(view.FindBlock(target, first_block), expected)
        << "target " << target << " from block " << first_block;
  }

  for (int window = 0; window < 8; ++window) {
    ChunkId first = RandomTarget(rng, ids);
    ChunkId last = RandomTarget(rng, ids);
    if (first > last) std::swap(first, last);
    ASSERT_EQ(DecodeRange(view, first, last), ReferenceRange(ids, first, last))
        << "window [" << first << ", " << last << "]";
  }
  // Windows aligned to skip blocks, as windowed evaluation uses them.
  if (num_blocks > 0) {
    const auto block =
        static_cast<int>(UniformBelow(rng, static_cast<uint64_t>(num_blocks)));
    const ChunkId first = block == 0 ? 0 : view.block_last(block - 1) + 1;
    const ChunkId last = view.block_last(block);
    ASSERT_EQ(DecodeRange(view, first, last), ReferenceRange(ids, first, last));
  }
  ASSERT_EQ(DecodeRange(view, 0, kMaxId), ids);

  // Next() visits every id; SeekGE() agrees with std::lower_bound, clamped
  // so that the cursor never moves backward.
  PostingCursor walker(view);
  for (const ChunkId id : ids) {
    ASSERT_FALSE(walker.done());
    ASSERT_EQ(walker.value(), id);
    walker.Next();
  }
  ASSERT_TRUE(walker.done());

  std::vector<ChunkId> targets(16);
  for (ChunkId& target : targets) target = RandomTarget(rng, ids);
  std::sort(targets.begin(), targets.end());
  PostingCursor cursor(view);
  size_t expected = 0;
  for (const ChunkId target : targets) {
    const auto lower = static_cast<size_t>(
        std::lower_bound(ids.begin(), ids.end(), target) - ids.begin());
    expected = std::max(expected, lower);
    ASSERT_EQ(cursor.SeekGE(target), expected < ids.size())
        << "target " << target;
    ASSERT_EQ(cursor.done(), expected == ids.size());
    if (cursor.done()) break;
    ASSERT_EQ(cursor.value(), ids[expected]) << "target " << target;
    if (UniformBelow(rng, 3) == 0) {
      cursor.Next();
      ++expected;
      ASSERT_EQ(cursor.done(), expected == ids.size());
      if (cursor.done()) break;
      ASSERT_EQ(cursor.value(), ids[expected]);
    }
  }

  // The skip-table probe agrees with a merge.
  std::vector<ChunkId> candidates(UniformBelow(rng, 24));
  for (ChunkId& candidate : candidates) candidate = RandomTarget(rng, ids);
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  std::vector<ChunkId> members;
  std::set_intersection(ids.begin(), ids.end(), candidates.begin(),
                        candidates.end(), std::back_inserter(members));
  ASSERT_EQ(SkipProbe(candidates, view), members);
}

// Encodes or inlines `ids`, views them through an exact-size copy, and checks
// the view.
void ExpectRoundTrip(const std::vector<ChunkId>& ids, std::mt19937_64& rng) {
  if (IsInlinedPostingList(ids.size())) {
    FERRET_ASSERT_OK_AND_ASSIGN(const PostingListView view,
                                PostingListView::FromInlined(ids));
    ExpectViewHolds(view, ids, rng);
    return;
  }
  const HeapBytes bytes(Encode(ids));
  FERRET_ASSERT_OK_AND_ASSIGN(const PostingListView view,
                              PostingListView::FromEncoded(bytes.view()));
  ExpectViewHolds(view, ids, rng);
}

TEST(PostingListViewTest, DefaultIsEmpty) {
  const PostingListView view;
  EXPECT_EQ(view.size(), 0u);
  EXPECT_EQ(view.num_blocks(), 0);
  EXPECT_EQ(view.FindBlock(0), 0);
  EXPECT_THAT(DecodeRange(view, 0, kMaxId), IsEmpty());
  EXPECT_TRUE(PostingCursor(view).done());
}

TEST(PostingListViewTest, InlinedListsRoundTrip) {
  const uint64_t seed = ferret_testing::TestSeed(0x5eed1000);
  SCOPED_TRACE(testing::Message() << "seed " << seed);
  std::mt19937_64 rng(seed);
  for (const std::vector<ChunkId>& ids : std::vector<std::vector<ChunkId>>{
           {}, {0}, {kMaxId}, {7, 8}, {0, kMaxId}}) {
    SCOPED_TRACE(testing::PrintToString(ids));
    ExpectRoundTrip(ids, rng);
  }
}

TEST(PostingListViewTest, FromInlinedRejectsWhatCannotBeInlined) {
  EXPECT_THAT(PostingListView::FromInlined(std::vector<ChunkId>{1, 2, 3}),
              StatusIs(absl::StatusCode::kDataLoss,
                       HasSubstr("3 ids cannot be inlined")));
  EXPECT_THAT(
      PostingListView::FromInlined(std::vector<ChunkId>{5, 5}),
      StatusIs(absl::StatusCode::kDataLoss, HasSubstr("are not increasing")));
  EXPECT_THAT(
      PostingListView::FromInlined(std::vector<ChunkId>{6, 5}),
      StatusIs(absl::StatusCode::kDataLoss, HasSubstr("are not increasing")));
}

TEST(PostingListViewTest, RoundTripsEverySmallSizeAtEveryGapWidth) {
  const uint64_t seed = ferret_testing::TestSeed(0x5eed1001);
  SCOPED_TRACE(testing::Message() << "seed " << seed);
  std::mt19937_64 rng(seed);
  for (size_t size = 3; size <= 2 * kPostingBlockSize + 3; ++size) {
    for (int gap_bits = 0; gap_bits <= 32; ++gap_bits) {
      SCOPED_TRACE(testing::Message()
                   << "size " << size << " gap_bits " << gap_bits);
      const uint64_t first_bound =
          std::min(uint64_t{1} << gap_bits, uint64_t{kMaxId} - size + 2);
      const auto first = static_cast<ChunkId>(UniformBelow(rng, first_bound));
      ExpectRoundTrip(RandomPostingList(rng, size, gap_bits, first), rng);
    }
  }
}

TEST(PostingListViewTest, RoundTripsBlockBoundarySizesAtManyDensities) {
  const uint64_t seed = ferret_testing::TestSeed(0x5eed1002);
  SCOPED_TRACE(testing::Message() << "seed " << seed);
  std::mt19937_64 rng(seed);
  for (const size_t size : {127, 128, 129, 255, 256, 257, 1000, 4096 + 17}) {
    for (const int gap_bits : {0, 1, 2, 3, 5, 8, 12, 16, 20, 24, 28, 32}) {
      SCOPED_TRACE(testing::Message()
                   << "size " << size << " gap_bits " << gap_bits);
      ExpectRoundTrip(RandomPostingList(rng, size, gap_bits, 0), rng);
      ExpectRoundTrip(
          RandomPostingList(rng, size, gap_bits,
                            static_cast<ChunkId>(UniformBelow(rng, 1 << 20))),
          rng);
    }
  }
}

TEST(PostingListViewTest, RoundTripsIdsNearUint32Max) {
  const uint64_t seed = ferret_testing::TestSeed(0x5eed1003);
  SCOPED_TRACE(testing::Message() << "seed " << seed);
  std::mt19937_64 rng(seed);
  for (const size_t size : {3, 127, 128, 129, 256, 300}) {
    SCOPED_TRACE(testing::Message() << "size " << size);
    // Consecutive ids ending at UINT32_MAX: the first block packs at width 32
    // for its first id and 0 elsewhere.
    ExpectRoundTrip(RandomPostingList(rng, size, 0,
                                      static_cast<ChunkId>(kMaxId - size + 1)),
                    rng);
    // Sparse ids in the top 2^20.
    ExpectRoundTrip(
        RandomPostingList(rng, size, 12, kMaxId - (ChunkId{1} << 20)), rng);
  }
  // A width-32 gap in the middle of a block, then a tail near the top.
  std::vector<ChunkId> ids = RandomPostingList(rng, 100, 0, 0);
  for (const ChunkId id : RandomPostingList(rng, 200, 3, 0x80000100)) {
    ids.push_back(id);
  }
  ExpectRoundTrip(ids, rng);
}

TEST(PostingListViewTest, RoundTripsMixedDensityRuns) {
  const uint64_t seed = ferret_testing::TestSeed(0x5eed1004);
  SCOPED_TRACE(testing::Message() << "seed " << seed);
  std::mt19937_64 rng(seed);
  for (int iteration = 0; iteration < 100; ++iteration) {
    std::vector<ChunkId> ids;
    ChunkId next = static_cast<ChunkId>(UniformBelow(rng, 1000));
    const auto runs = static_cast<int>(1 + UniformBelow(rng, 12));
    for (int run = 0; run < runs; ++run) {
      const size_t size = 1 + UniformBelow(rng, 400);
      const auto gap_bits = static_cast<int>(UniformBelow(rng, 14));
      for (const ChunkId id : RandomPostingList(rng, size, gap_bits, next)) {
        ids.push_back(id);
      }
      next = ids.back() + 1 + static_cast<ChunkId>(UniformBelow(rng, 1 << 16));
    }
    if (IsInlinedPostingList(ids.size())) continue;
    ExpectRoundTrip(ids, rng);
  }
}

TEST(PostingListViewTest, ReadsOnlyTheEncodingAtTheFrontOfItsInput) {
  const uint64_t seed = ferret_testing::TestSeed(0x5eed1005);
  SCOPED_TRACE(testing::Message() << "seed " << seed);
  std::mt19937_64 rng(seed);
  for (const size_t size : {3, 100, 128, 200, 256}) {
    const std::vector<ChunkId> ids = RandomPostingList(rng, size, 6, 3);
    const std::string encoded = Encode(ids);
    // Trailing bytes, such as the next list in the section, are ignored.
    const HeapBytes followed(encoded + "\x01\x02\x03");
    FERRET_ASSERT_OK_AND_ASSIGN(const PostingListView view,
                                PostingListView::FromEncoded(followed.view()));
    EXPECT_EQ(DecodeAll(view), ids);
    // Every strict prefix is truncated.
    for (size_t length = 0; length < encoded.size(); ++length) {
      const HeapBytes prefix(absl::string_view(encoded).substr(0, length));
      EXPECT_THAT(PostingListView::FromEncoded(prefix.view()),
                  StatusIs(absl::StatusCode::kDataLoss))
          << "size " << size << " length " << length;
    }
  }
}

// A valid 130-id list: one block of width 1 (see
// posting_list_encoder_test.cc) and a 2-id tail. Its skip entry sits at bytes
// [8, 16) and its payload at [16, 32).
std::string BlockAndTail() {
  std::vector<ChunkId> ids;
  for (ChunkId id = 0; id <= 128; ++id) {
    if (id != 5) ids.push_back(id);
  }
  ids.push_back(130);
  ids.push_back(131);
  return Encode(ids);
}

absl::Status FromEncodedStatus(absl::string_view bytes) {
  const HeapBytes heap(bytes);
  return PostingListView::FromEncoded(heap.view()).status();
}

std::string Varints(std::initializer_list<uint32_t> values) {
  std::string out;
  for (const uint32_t value : values) AppendVarint32(value, out);
  return out;
}

TEST(PostingListViewTest, RejectsMalformedCounts) {
  EXPECT_THAT(FromEncodedStatus(""), StatusIs(absl::StatusCode::kDataLoss,
                                              HasSubstr("truncated count")));
  EXPECT_THAT(FromEncodedStatus(Varints({2, 1, 1})),
              StatusIs(absl::StatusCode::kDataLoss,
                       HasSubstr("count 2 belongs in the dictionary")));
  EXPECT_THAT(
      FromEncodedStatus(Varints({(uint32_t{1} << 26) + 1})),
      StatusIs(absl::StatusCode::kDataLoss, HasSubstr("exceeds the limit")));
}

TEST(PostingListViewTest, RejectsMalformedGaps) {
  EXPECT_THAT(
      FromEncodedStatus(Varints({3, 5, 0})),
      StatusIs(absl::StatusCode::kDataLoss, HasSubstr("truncated gap 2 of 3")));
  EXPECT_THAT(FromEncodedStatus(Varints({3, kMaxId - 1, 0, 0})),
              StatusIs(absl::StatusCode::kDataLoss,
                       HasSubstr("gap 2 of 3 overflows 32-bit ids")));
  std::string tail_overflow = BlockAndTail();
  tail_overflow.replace(32, 1, Varints({kMaxId}));
  EXPECT_THAT(FromEncodedStatus(tail_overflow),
              StatusIs(absl::StatusCode::kDataLoss,
                       HasSubstr("gap 0 of 2 overflows 32-bit ids")));
}

TEST(PostingListViewTest, RejectsInconsistentSkipTables) {
  const std::string valid = BlockAndTail();
  ASSERT_TRUE(PostingListView::FromEncoded(valid).ok());
  struct Corruption {
    size_t offset;
    char byte;
    absl::string_view message;
  };
  for (const Corruption& corruption : {
           Corruption{12, '\x21', "bit width 33 exceeds 32"},
           Corruption{12, '\xFF', "bit width 255 exceeds 32"},
           Corruption{13, '\x01', "payload offset 1 is not the expected 0"},
           Corruption{8, '\x7E', "last id 126 is less than 128 above"},
           Corruption{12, '\x02', "bit width 2 does not match its id span 1"},
           Corruption{12, '\x00', "bit width 0 does not match its id span 1"},
           Corruption{2, '\x01', "nonzero padding at byte 2"},
           Corruption{7, '\x80', "nonzero padding at byte 7"},
       }) {
    std::string bytes = valid;
    bytes[corruption.offset] = corruption.byte;
    EXPECT_THAT(
        FromEncodedStatus(bytes),
        StatusIs(absl::StatusCode::kDataLoss, HasSubstr(corruption.message)));
  }
  EXPECT_THAT(FromEncodedStatus(absl::string_view(valid).substr(0, 15)),
              StatusIs(absl::StatusCode::kDataLoss,
                       HasSubstr("skip table ends at byte 16")));
  EXPECT_THAT(FromEncodedStatus(absl::string_view(valid).substr(0, 31)),
              StatusIs(absl::StatusCode::kDataLoss,
                       HasSubstr("block payloads end at byte 32")));
  EXPECT_THAT(
      FromEncodedStatus(absl::string_view(valid).substr(0, 33)),
      StatusIs(absl::StatusCode::kDataLoss, HasSubstr("truncated gap 1 of 2")));
}

// A valid 257-id list: block 0 (0..128 without 5, width 1), block 1
// (129..256, width 0) and a one-id tail, 258. Its skip entries sit at bytes
// [16, 32), block 0's payload at [32, 48), and the tail at byte 48.
std::vector<ChunkId> TwoBlocksAndTail() {
  std::vector<ChunkId> ids;
  for (ChunkId id = 0; id <= 256; ++id) {
    if (id != 5) ids.push_back(id);
  }
  ids.push_back(258);
  return ids;
}

TEST(PostingListViewTest, CorruptPayloadStillYieldsAWellFormedList) {
  const uint64_t seed = ferret_testing::TestSeed(0x5eed1008);
  SCOPED_TRACE(testing::Message() << "seed " << seed);
  std::mt19937_64 rng(seed);
  const std::vector<ChunkId> ids = TwoBlocksAndTail();
  const std::string valid = Encode(ids);
  ASSERT_EQ(valid.size(), 49u);
  // The skip table cannot reveal these flips: the first makes block 0 end at
  // 129, past its last id, and the second at 127, short of it.
  for (const auto [offset, mask] : {std::pair<size_t, char>{32, '\x01'},
                                    std::pair<size_t, char>{36, '\x02'}}) {
    SCOPED_TRACE(testing::Message() << "flip at byte " << offset);
    std::string bytes = valid;
    bytes[offset] = static_cast<char>(bytes[offset] ^ mask);
    const HeapBytes heap(bytes);
    FERRET_ASSERT_OK_AND_ASSIGN(const PostingListView view,
                                PostingListView::FromEncoded(heap.view()));
    const std::vector<ChunkId> decoded = DecodeAll(view);
    ASSERT_TRUE(IsStrictlyIncreasing(decoded));
    ExpectViewHolds(view, decoded, rng);
    EXPECT_EQ(DecodeRange(view, 0, 128).back(), 128u);
    // The damage stays in block 0.
    EXPECT_TRUE(std::equal(decoded.begin() + kPostingBlockSize, decoded.end(),
                           ids.begin() + kPostingBlockSize));
  }
}

TEST(PostingListViewTest, CorruptWideBlockThatWrapsToItsLastIdIsCaught) {
  // Ids 2^31 .. 2^31 + 127: one block at width 32, whose payload words are
  // its deltas in order, 2^31 and then zeros, at bytes [16, 528).
  std::vector<ChunkId> ids;
  for (ChunkId id = 0x80000000; id < 0x80000080; ++id) ids.push_back(id);
  std::string bytes = Encode(ids);
  ASSERT_EQ(bytes.size(), 16u + 512u);
  // Deltas 1 and 2 become 2^32 - 1 and 1: the running sum wraps once and
  // still ends at the skip entry's last id, but ids 0 and 1 collide.
  const uint32_t delta1 = 0xFFFFFFFF;
  const uint32_t delta2 = 1;
  std::memcpy(&bytes[16 + 4], &delta1, sizeof(delta1));
  std::memcpy(&bytes[16 + 8], &delta2, sizeof(delta2));
  const HeapBytes heap(bytes);
  FERRET_ASSERT_OK_AND_ASSIGN(const PostingListView view,
                              PostingListView::FromEncoded(heap.view()));
  const std::vector<ChunkId> decoded = DecodeAll(view);
  EXPECT_TRUE(IsStrictlyIncreasing(decoded));
  EXPECT_EQ(decoded.back(), 0x8000007Fu);
}

// Whatever FromEncoded accepts must behave as a valid list. Payload
// corruption can go undetected, so the ids may be wrong, but DecodeAll must
// return a strictly increasing list and every other accessor must agree with
// it, without reading or writing out of bounds (checked by ASan through
// exact-size buffers).
void ExpectUntrustedBytesAreSafe(absl::string_view bytes,
                                 std::mt19937_64& rng) {
  const HeapBytes heap(bytes);
  const absl::StatusOr<PostingListView> view =
      PostingListView::FromEncoded(heap.view());
  if (!view.ok()) {
    ASSERT_EQ(view.status().code(), absl::StatusCode::kDataLoss);
    return;
  }
  const std::vector<ChunkId> ids = DecodeAll(*view);
  ASSERT_TRUE(IsStrictlyIncreasing(ids));
  ExpectViewHolds(*view, ids, rng);
}

TEST(PostingListViewTest, UntrustedBytesAreRejectedOrDecodedSafely) {
  const uint64_t seed = ferret_testing::TestSeed(0x5eed1006);
  SCOPED_TRACE(testing::Message() << "seed " << seed);
  std::mt19937_64 rng(seed);
  std::vector<std::string> valid;
  for (const size_t size : {3, 50, 127, 128, 129, 300, 700}) {
    for (const int gap_bits : {0, 2, 9, 24, 25, 32}) {
      valid.push_back(Encode(RandomPostingList(rng, size, gap_bits, 1)));
    }
  }
  for (int iteration = 0; iteration < 20000; ++iteration) {
    std::string bytes;
    switch (UniformBelow(rng, 4)) {
      case 0: {  // A plausible count followed by random bytes.
        AppendVarint32(static_cast<uint32_t>(3 + UniformBelow(rng, 400)),
                       bytes);
        const size_t size = UniformBelow(rng, 200);
        for (size_t i = 0; i < size; ++i) {
          bytes.push_back(static_cast<char>(rng()));
        }
        break;
      }
      case 1: {  // Bit flips.
        bytes = valid[UniformBelow(rng, valid.size())];
        const auto flips = static_cast<int>(1 + UniformBelow(rng, 3));
        for (int flip = 0; flip < flips; ++flip) {
          char& byte = bytes[UniformBelow(rng, bytes.size())];
          byte = static_cast<char>(byte ^ (1 << UniformBelow(rng, 8)));
        }
        break;
      }
      case 2:  // Truncation.
        bytes = valid[UniformBelow(rng, valid.size())];
        bytes.resize(UniformBelow(rng, bytes.size()));
        break;
      default:  // A random byte anywhere, including the skip table.
        bytes = valid[UniformBelow(rng, valid.size())];
        bytes[UniformBelow(rng, std::min<size_t>(bytes.size(), 64))] =
            static_cast<char>(rng());
        break;
    }
    ExpectUntrustedBytesAreSafe(bytes, rng);
    if (testing::Test::HasFatalFailure()) {
      FAIL() << "iteration " << iteration;
    }
  }
}

TEST(PostingCursorTest, SeekToPastEndIsDone) {
  const std::vector<ChunkId> ids = {3, 9, 200};
  const HeapBytes bytes(Encode(ids));
  FERRET_ASSERT_OK_AND_ASSIGN(const PostingListView view,
                              PostingListView::FromEncoded(bytes.view()));
  PostingCursor cursor(view);
  EXPECT_FALSE(cursor.SeekGE(201));
  EXPECT_TRUE(cursor.done());
  EXPECT_FALSE(cursor.SeekGE(0));
}

TEST(PostingCursorTest, SeekNeverMovesBackward) {
  std::vector<ChunkId> ids;
  for (ChunkId id = 10; id < 1000; id += 3) ids.push_back(id);
  const HeapBytes bytes(Encode(ids));
  FERRET_ASSERT_OK_AND_ASSIGN(const PostingListView view,
                              PostingListView::FromEncoded(bytes.view()));
  PostingCursor cursor(view);
  ASSERT_TRUE(cursor.SeekGE(500));
  EXPECT_EQ(cursor.value(), 502u);
  ASSERT_TRUE(cursor.SeekGE(11));
  EXPECT_EQ(cursor.value(), 502u);
  ASSERT_TRUE(cursor.SeekGE(502));
  EXPECT_EQ(cursor.value(), 502u);
}

TEST(IntersectBySkipProbeTest, MatchesAMergeForSparseCandidates) {
  const uint64_t seed = ferret_testing::TestSeed(0x5eed1007);
  SCOPED_TRACE(testing::Message() << "seed " << seed);
  std::mt19937_64 rng(seed);
  for (int iteration = 0; iteration < 50; ++iteration) {
    const std::vector<ChunkId> ids =
        RandomPostingList(rng, 3 + UniformBelow(rng, 5000),
                          static_cast<int>(UniformBelow(rng, 8)), 0);
    std::vector<ChunkId> candidates =
        RandomPostingList(rng, 1 + UniformBelow(rng, 40), 12, 0);
    for (int i = 0; i < 10; ++i) candidates.push_back(RandomTarget(rng, ids));
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
    const HeapBytes bytes(Encode(ids));
    FERRET_ASSERT_OK_AND_ASSIGN(const PostingListView view,
                                PostingListView::FromEncoded(bytes.view()));
    std::vector<ChunkId> expected;
    std::set_intersection(ids.begin(), ids.end(), candidates.begin(),
                          candidates.end(), std::back_inserter(expected));
    EXPECT_EQ(SkipProbe(candidates, view), expected);
  }
}

TEST(IntersectBySkipProbeTest, EmptyInputsGiveNothing) {
  const HeapBytes bytes(Encode(TwoBlocksAndTail()));
  FERRET_ASSERT_OK_AND_ASSIGN(const PostingListView view,
                              PostingListView::FromEncoded(bytes.view()));
  EXPECT_THAT(SkipProbe({}, view), IsEmpty());
  EXPECT_THAT(SkipProbe(std::vector<ChunkId>{5, 257, 259, kMaxId}, view),
              IsEmpty());
  EXPECT_THAT(SkipProbe(std::vector<ChunkId>{0, 1, kMaxId}, PostingListView()),
              IsEmpty());
}

TEST(IntersectBySkipProbeTest, FindsBlockBoundaryIds) {
  const HeapBytes bytes(Encode(TwoBlocksAndTail()));
  FERRET_ASSERT_OK_AND_ASSIGN(const PostingListView view,
                              PostingListView::FromEncoded(bytes.view()));
  EXPECT_THAT(
      SkipProbe(std::vector<ChunkId>{0, 5, 128, 129, 256, 257, 258}, view),
      ElementsAre(0u, 128u, 129u, 256u, 258u));
}

TEST(PostingCursorTest, WalksInlinedList) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      const PostingListView view,
      PostingListView::FromInlined(std::vector<ChunkId>{4, kMaxId}));
  std::vector<ChunkId> walked;
  for (PostingCursor cursor(view); !cursor.done(); cursor.Next()) {
    walked.push_back(cursor.value());
  }
  EXPECT_THAT(walked, ElementsAre(4u, kMaxId));
}

}  // namespace
}  // namespace ferret

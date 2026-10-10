// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Golden encodings: these pin the on-disk format of posting_list.h byte by
// byte. posting_list_test.cc covers decoding and round trips.

#include "ferret/postings/posting_list_encoder.h"

#include <cstddef>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "ferret/postings/ids.h"
#include "ferret/postings/posting_list.h"

namespace ferret {
namespace {

std::string Encode(const std::vector<ChunkId>& ids) {
  std::string out;
  AppendEncodedPostingList(ids, out);
  return out;
}

std::vector<ChunkId> Range(ChunkId first, ChunkId end) {
  std::vector<ChunkId> ids;
  for (ChunkId id = first; id < end; ++id) ids.push_back(id);
  return ids;
}

// 0..127 except that every id from index 5 on is one higher, ending at 128:
// one block whose only nonzero gap - 1 is at index 5 (see bitpack_test.cc).
std::vector<ChunkId> BlockWithOneGap() {
  std::vector<ChunkId> ids = Range(0, 5);
  for (ChunkId id = 6; id <= 128; ++id) ids.push_back(id);
  return ids;
}

std::string Zeros(size_t n) { return std::string(n, '\0'); }

TEST(PostingListEncoderTest, OnlyListsOfThreeOrMoreAreEncoded) {
  EXPECT_TRUE(IsInlinedPostingList(0));
  EXPECT_TRUE(IsInlinedPostingList(1));
  EXPECT_TRUE(IsInlinedPostingList(2));
  EXPECT_FALSE(IsInlinedPostingList(3));
}

TEST(PostingListEncoderTest, ShortListIsCountThenGapsMinusOne) {
  // The first id's predecessor is -1, so its gap - 1 is the id itself.
  EXPECT_EQ(Encode({5, 6, 10}), std::string("\x03\x05\x00\x03", 4));
  EXPECT_EQ(Encode({0, 1, 2}), std::string("\x03\x00\x00\x00", 4));
  EXPECT_EQ(Encode({200, 0xFFFFFFFE, 0xFFFFFFFF}),
            std::string("\x03\xC8\x01\xB5\xFE\xFF\xFF\x0F\x00", 9));
}

TEST(PostingListEncoderTest, ConsecutiveBlockPacksAtWidthZero) {
  // Count 128 takes 2 bytes; padding to byte 8 puts the 8-byte skip table
  // right before the payload region at byte 16, which holds nothing.
  EXPECT_EQ(Encode(Range(0, 128)),
            absl::StrCat("\x80\x01", Zeros(6),  // count 128, padding
                         "\x7F", Zeros(3),      // last_chunk = 127
                         Zeros(4)));            // offset 0, width 0
}

TEST(PostingListEncoderTest, BlockThenVarintTail) {
  std::vector<ChunkId> ids = BlockWithOneGap();
  ids.push_back(130);
  ids.push_back(131);
  EXPECT_EQ(Encode(ids),
            absl::StrCat("\x82\x01", Zeros(6),         // count 130, padding
                         "\x80", Zeros(3),             // last_chunk = 128
                         "\x01", Zeros(3),             // offset 0, width 1
                         Zeros(4), "\x02", Zeros(11),  // payload (bitpack.h)
                         "\x01", Zeros(1)));           // tail gaps - 1
}

TEST(PostingListEncoderTest, OffsetsAreRunningSumsOfWidths) {
  std::vector<ChunkId> ids = BlockWithOneGap();
  for (ChunkId id = 129; id <= 256; ++id) ids.push_back(id);
  // Two skip entries: padding to byte 16, payloads from byte 32.
  EXPECT_EQ(Encode(ids),
            absl::StrCat("\x80\x02", Zeros(14),               // count 256
                         "\x80", Zeros(3), "\x01", Zeros(3),  // 128; 0, 1
                         std::string("\x00\x01\x00\x00", 4),  // 256
                         std::string("\x00\x01\x00\x00", 4),  // 1, 0
                         Zeros(4), "\x02", Zeros(11)));       // block 0
}

TEST(PostingListEncoderTest, EncodingIsPositionIndependent) {
  const std::vector<ChunkId> ids = Range(1000, 1300);
  std::string out = "prefix";
  AppendEncodedPostingList(ids, out);
  EXPECT_EQ(out, "prefix" + Encode(ids));
}

TEST(PostingListEncoderDeathTest, InlinedListIsAProgrammingError) {
  std::string out;
  EXPECT_DEATH(AppendEncodedPostingList(std::vector<ChunkId>{1, 2}, out),
               "belongs in its dictionary entry");
}

}  // namespace
}  // namespace ferret

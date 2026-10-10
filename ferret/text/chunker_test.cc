// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/chunker.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ferret {
namespace {

using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::IsEmpty;

std::string Describe(const std::vector<ChunkSpan>& chunks) {
  std::string out;
  for (const ChunkSpan& chunk : chunks) {
    absl::StrAppend(&out, "{", chunk.offset, ", ", chunk.size, ", ",
                    chunk.first_line, "} ");
  }
  return out;
}

// The rule of chunker.h restated as a single pass over the bytes: a chunk ends
// after the first '\n' at an offset of at least start + c, or at the end of the
// content.
std::vector<ChunkSpan> ReferenceChunks(absl::string_view content, int64_t c) {
  std::vector<ChunkSpan> chunks;
  int64_t start = 0;
  int64_t line = 1;
  int64_t newlines = 0;
  const int64_t size = static_cast<int64_t>(content.size());
  for (int64_t i = 0; i < size; ++i) {
    const bool newline = content[static_cast<size_t>(i)] == '\n';
    if (newline) ++newlines;
    if ((newline && i >= start + c) || i == size - 1) {
      chunks.push_back(
          {.offset = start, .size = i + 1 - start, .first_line = line});
      start = i + 1;
      line += newlines;
      newlines = 0;
    }
  }
  return chunks;
}

TEST(SplitIntoChunksTest, EmptyContentHasNoChunks) {
  EXPECT_THAT(SplitIntoChunks("", 8), IsEmpty());
}

TEST(SplitIntoChunksTest, ContentUpToChunkBytesIsOneChunk) {
  EXPECT_THAT(SplitIntoChunks("a\nb\nc", 8), ElementsAre(FieldsAre(0, 5, 1)));
  EXPECT_THAT(SplitIntoChunks("1234567\n", 8), ElementsAre(FieldsAre(0, 8, 1)));
}

TEST(SplitIntoChunksTest, SplitsAtFirstNewlineAtOrAfterChunkBytes) {
  // With chunk_bytes = 4, the first chunk may end after the '\n' at offset 4
  // or later; the one at offset 3 is too early.
  EXPECT_THAT(SplitIntoChunks("abc\ndefg\nhij\nk", 4),
              ElementsAre(FieldsAre(0, 9, 1), FieldsAre(9, 5, 3)));
  EXPECT_THAT(
      SplitIntoChunks("abcd\nefgh\nijkl\n", 4),
      ElementsAre(FieldsAre(0, 5, 1), FieldsAre(5, 5, 2), FieldsAre(10, 5, 3)));
}

TEST(SplitIntoChunksTest, OverlongLineIsOneOversizeChunk) {
  const std::string content = "short\n" + std::string(100, 'x') + "\nend\n";
  EXPECT_THAT(SplitIntoChunks(content, 4),
              ElementsAre(FieldsAre(0, 6, 1), FieldsAre(6, 101, 2),
                          FieldsAre(107, 4, 3)));
  EXPECT_THAT(SplitIntoChunks(std::string(100, 'x'), 4),
              ElementsAre(FieldsAre(0, 100, 1)));
}

TEST(SplitIntoChunksTest, CountsBlankLines) {
  EXPECT_THAT(
      SplitIntoChunks("\n\n\n\n\n\n", 1),
      ElementsAre(FieldsAre(0, 2, 1), FieldsAre(2, 2, 3), FieldsAre(4, 2, 5)));
}

TEST(SplitIntoChunksTest, MatchesReferenceOnRandomContent) {
  constexpr uint64_t kSeed = 17;
  SCOPED_TRACE(kSeed);
  std::mt19937_64 rng(kSeed);
  for (int iteration = 0; iteration < 5000; ++iteration) {
    std::string content(rng() % 400, 'a');
    // Lines of random length, including empty and overlong ones.
    const uint64_t newline_odds = 1 + rng() % 60;
    for (char& c : content) {
      if (rng() % newline_odds == 0) c = '\n';
    }
    const int64_t chunk_bytes = 1 + static_cast<int64_t>(rng() % 64);
    const std::vector<ChunkSpan> chunks = SplitIntoChunks(content, chunk_bytes);
    const std::vector<ChunkSpan> expected =
        ReferenceChunks(content, chunk_bytes);
    ASSERT_TRUE(chunks == expected)
        << Describe(chunks) << "!= " << Describe(expected) << "for chunk_bytes "
        << chunk_bytes << " and content '" << content << "'";
  }
}

}  // namespace
}  // namespace ferret

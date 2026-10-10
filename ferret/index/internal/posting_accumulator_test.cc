// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/internal/posting_accumulator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/postings/ids.h"
#include "ferret/testing/test_seed.h"
#include "ferret/text/trigram.h"

namespace ferret::segment_internal {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Pair;

using Lists = std::map<Trigram, std::vector<ChunkId>>;

Lists Collect(PostingAccumulator& accumulator) {
  Lists lists;
  Trigram previous = 0;
  bool first = true;
  EXPECT_THAT(accumulator.ForEachList([&](Trigram trigram,
                                          absl::Span<const ChunkId> ids) {
    EXPECT_TRUE(first || trigram > previous) << "trigrams out of order";
    first = false;
    previous = trigram;
    lists[trigram].assign(ids.begin(), ids.end());
    return absl::OkStatus();
  }),
              IsOk());
  return lists;
}

TEST(PostingAccumulatorTest, Empty) {
  PostingAccumulator accumulator(16);
  EXPECT_EQ(accumulator.num_postings(), 0u);
  EXPECT_TRUE(Collect(accumulator).empty());
}

TEST(PostingAccumulatorTest, SmallExample) {
  PostingAccumulator accumulator(1000);
  accumulator.AddChunk(0, std::vector<Trigram>{30, 10});
  accumulator.AddChunk(4, std::vector<Trigram>{});
  accumulator.AddChunk(5, std::vector<Trigram>{10, 20, kTrigramMask});
  EXPECT_EQ(accumulator.num_postings(), 5u);
  EXPECT_THAT(Collect(accumulator),
              ElementsAre(Pair(10, ElementsAre(0, 5)), Pair(20, ElementsAre(5)),
                          Pair(30, ElementsAre(0)),
                          Pair(kTrigramMask, ElementsAre(5))));
}

// Deltas and first ids of every varint width, in one run and across runs.
TEST(PostingAccumulatorTest, VarintWidths) {
  const std::vector<ChunkId> ids = {0,
                                    1,
                                    127,
                                    128,
                                    16383,
                                    16384,
                                    2097151,
                                    2097152,
                                    268435455,
                                    268435456,
                                    std::numeric_limits<ChunkId>::max() - 1};
  for (const size_t partition : {size_t{1000}, size_t{3}}) {
    PostingAccumulator accumulator(partition);
    for (const ChunkId id : ids) {
      accumulator.AddChunk(id, std::vector<Trigram>{7, 9});
    }
    EXPECT_THAT(Collect(accumulator),
                ElementsAre(Pair(7, ElementsAreArray(ids)),
                            Pair(9, ElementsAreArray(ids))));
  }
}

TEST(PostingAccumulatorTest, StopsAtFirstError) {
  PostingAccumulator accumulator(1000);
  accumulator.AddChunk(0, std::vector<Trigram>{1, 2, 3});
  int calls = 0;
  EXPECT_THAT(
      accumulator.ForEachList([&](Trigram trigram, absl::Span<const ChunkId>) {
        ++calls;
        return trigram == 2 ? absl::DataLossError("boom") : absl::OkStatus();
      }),
      StatusIs(absl::StatusCode::kDataLoss));
  EXPECT_EQ(calls, 2);
}

// Tiny partitions force many runs, including trigrams that appear in some
// runs and not others, and radix digits that differ only in either half.
TEST(PostingAccumulatorTest, RandomMatchesReference) {
  std::mt19937_64 rng(ferret_testing::TestSeed(11));
  for (const size_t partition : {1u, 7u, 100u, 100000u}) {
    PostingAccumulator accumulator(partition);
    Lists expected;
    ChunkId chunk = 0;
    uint64_t postings = 0;
    for (int c = 0; c < 400; ++c) {
      chunk += 1 + static_cast<ChunkId>(rng() % 3);
      std::vector<Trigram> trigrams;
      const size_t count = rng() % 12;
      for (size_t i = 0; i < count; ++i) {
        // Mix a few shared low and high digits with fully random trigrams.
        const Trigram trigram =
            rng() % 2 == 0
                ? static_cast<Trigram>(rng() & kTrigramMask)
                : static_cast<Trigram>(((rng() % 4) << 12) | (rng() % 4));
        if (std::find(trigrams.begin(), trigrams.end(), trigram) ==
            trigrams.end()) {
          trigrams.push_back(trigram);
        }
      }
      for (const Trigram trigram : trigrams) {
        expected[trigram].push_back(chunk);
      }
      postings += trigrams.size();
      accumulator.AddChunk(chunk, trigrams);
    }
    EXPECT_EQ(accumulator.num_postings(), postings);
    EXPECT_EQ(Collect(accumulator), expected) << "partition " << partition;
  }
}

}  // namespace
}  // namespace ferret::segment_internal

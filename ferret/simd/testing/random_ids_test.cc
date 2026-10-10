// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/simd/testing/random_ids.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ferret {
namespace {

using ::ferret_testing::IdListPair;
using ::ferret_testing::RandomIdListPair;
using ::ferret_testing::RandomIncreasingIds;
using ::ferret_testing::RandomListPairsOverSameRange;
using ::ferret_testing::RandomSubset;
using ::ferret_testing::UniformInt;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::IsEmpty;
using ::testing::IsSubsetOf;
using ::testing::Le;
using ::testing::SizeIs;

constexpr uint32_t kMaxId = std::numeric_limits<uint32_t>::max();

bool IsStrictlyIncreasing(const std::vector<uint32_t>& ids) {
  return std::adjacent_find(ids.begin(), ids.end(),
                            std::greater_equal<uint32_t>()) == ids.end();
}

TEST(RandomIdsTest, UniformIntCoversClosedRange) {
  std::mt19937_64 rng(1);
  std::vector<int> histogram(4);
  for (int draw = 0; draw < 1000; ++draw) {
    const uint64_t value = UniformInt(rng, 3, 6);
    ASSERT_GE(value, 3u);
    ASSERT_LE(value, 6u);
    ++histogram[value - 3];
  }
  EXPECT_THAT(histogram, Each(Gt(150)));
  EXPECT_EQ(UniformInt(rng, 9, 9), 9u);
  // The full range takes the generator's output as is.
  std::mt19937_64 copy = rng;
  EXPECT_EQ(UniformInt(rng, 0, std::numeric_limits<uint64_t>::max()), copy());
}

TEST(RandomIdsTest, IncreasingIdsHonorFirstIdAndGapBound) {
  std::mt19937_64 rng(2);
  const std::vector<uint32_t> ids = RandomIncreasingIds(rng, 1000, 5, 3);
  ASSERT_THAT(ids, SizeIs(1000));
  EXPECT_EQ(ids.front(), 5u);
  std::vector<uint32_t> gaps;
  std::adjacent_difference(ids.begin(), ids.end(), std::back_inserter(gaps));
  gaps.erase(gaps.begin());
  EXPECT_THAT(gaps, Each(AllOf(Ge(1u), Le(3u))));
  EXPECT_THAT(gaps, Contains(1u));
  EXPECT_THAT(gaps, Contains(3u));
}

TEST(RandomIdsTest, IncreasingIdsWithUnitGapAreConsecutiveUpToMaxId) {
  std::mt19937_64 rng(3);
  EXPECT_THAT(RandomIncreasingIds(rng, 3, kMaxId - 2, 1),
              ElementsAre(kMaxId - 2, kMaxId - 1, kMaxId));
  EXPECT_THAT(RandomIncreasingIds(rng, 0, 0, 1), IsEmpty());
}

TEST(RandomIdsTest, SubsetIsOrderedWithRequestedSize) {
  std::mt19937_64 rng(4);
  const std::vector<uint32_t> ids = RandomIncreasingIds(rng, 100, 0, 5);
  for (const size_t size : {0, 1, 37, 100}) {
    const std::vector<uint32_t> subset = RandomSubset(rng, ids, size);
    EXPECT_THAT(subset, SizeIs(size));
    EXPECT_TRUE(IsStrictlyIncreasing(subset));
    EXPECT_THAT(subset, IsSubsetOf(ids));
  }
}

TEST(RandomIdsTest, ListPairsOverSameRangeHaveRequestedShape) {
  std::mt19937_64 rng(5);
  // Each pair holds 900 ids, so it takes 3 pairs to hold 2000.
  const std::vector<IdListPair> pairs =
      RandomListPairsOverSameRange(rng, 100, 8, 4, /*min_total_ids=*/2000);
  ASSERT_THAT(pairs, SizeIs(3));
  for (const IdListPair& pair : pairs) {
    ASSERT_THAT(pair.a, SizeIs(800));
    ASSERT_THAT(pair.b, SizeIs(100));
    EXPECT_TRUE(IsStrictlyIncreasing(pair.a));
    EXPECT_TRUE(IsStrictlyIncreasing(pair.b));
    // Both lists span about 800 * 4 ids; the bounds are several standard
    // deviations wide.
    EXPECT_NEAR(pair.a.back(), 3200, 400);
    EXPECT_NEAR(pair.b.back(), 3200, 800);
  }
  EXPECT_NE(pairs[0].a, pairs[1].a);
  EXPECT_NE(pairs[0].b, pairs[1].b);
}

TEST(RandomIdsTest, ListPairsOverSameRangeHoldAtLeastMinTotalIds) {
  std::mt19937_64 rng(8);
  // Each pair holds 10 + 10 ids.
  EXPECT_THAT(RandomListPairsOverSameRange(rng, 10, 1, 1, 0), SizeIs(1));
  EXPECT_THAT(RandomListPairsOverSameRange(rng, 10, 1, 1, 20), SizeIs(1));
  EXPECT_THAT(RandomListPairsOverSameRange(rng, 10, 1, 1, 21), SizeIs(2));
}

TEST(RandomIdsTest, ListPairsAreReproducible) {
  std::mt19937_64 rng1(6);
  std::mt19937_64 rng2(6);
  for (int iteration = 0; iteration < 20; ++iteration) {
    const IdListPair pair1 = RandomIdListPair(rng1);
    const IdListPair pair2 = RandomIdListPair(rng2);
    EXPECT_EQ(pair1.a, pair2.a);
    EXPECT_EQ(pair1.b, pair2.b);
  }
}

// The kernel tests rely on RandomIdListPair to reach these cases, so check
// that it does.
TEST(RandomIdsTest, ListPairsCoverKernelEdgeCases) {
  std::mt19937_64 rng(7);
  bool one_empty = false;
  bool a_much_longer = false;
  bool b_much_longer = false;
  bool long_list = false;
  bool contains = false;
  bool disjoint = false;
  bool consecutive = false;
  bool has_zero = false;
  bool has_max_id = false;
  for (int iteration = 0; iteration < 2000; ++iteration) {
    SCOPED_TRACE(absl::StrCat("iteration ", iteration));
    const IdListPair pair = RandomIdListPair(rng);
    ASSERT_TRUE(IsStrictlyIncreasing(pair.a));
    ASSERT_TRUE(IsStrictlyIncreasing(pair.b));
    ASSERT_THAT(pair.a, SizeIs(Le(10'000u)));
    ASSERT_THAT(pair.b, SizeIs(Le(10'000u)));
    const std::vector<uint32_t>& shorter =
        pair.a.size() < pair.b.size() ? pair.a : pair.b;
    const std::vector<uint32_t>& longer =
        pair.a.size() < pair.b.size() ? pair.b : pair.a;
    one_empty |= shorter.empty() && !longer.empty();
    a_much_longer |= !pair.b.empty() && pair.a.size() >= 512 * pair.b.size();
    b_much_longer |= !pair.a.empty() && pair.b.size() >= 512 * pair.a.size();
    long_list |= longer.size() >= 5'000;
    if (shorter.size() >= 4) {
      std::vector<uint32_t> common;
      std::set_intersection(shorter.begin(), shorter.end(), longer.begin(),
                            longer.end(), std::back_inserter(common));
      contains |= common.size() == shorter.size();
      disjoint |= common.empty();
    }
    for (const std::vector<uint32_t>* ids : {&pair.a, &pair.b}) {
      consecutive |= std::adjacent_find(ids->begin(), ids->end(),
                                        [](uint32_t x, uint32_t y) {
                                          return y == x + 1;
                                        }) != ids->end();
      has_zero |= !ids->empty() && ids->front() == 0;
      has_max_id |= !ids->empty() && ids->back() == kMaxId;
    }
  }
  EXPECT_TRUE(one_empty);
  EXPECT_TRUE(a_much_longer);
  EXPECT_TRUE(b_much_longer);
  EXPECT_TRUE(long_list);
  EXPECT_TRUE(contains);
  EXPECT_TRUE(disjoint);
  EXPECT_TRUE(consecutive);
  EXPECT_TRUE(has_zero);
  EXPECT_TRUE(has_max_id);
}

}  // namespace
}  // namespace ferret

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/base/bitset.h"

#include <cstdint>
#include <random>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/testing/test_seed.h"

namespace ferret {
namespace {

using ::testing::ElementsAre;

// A draw from [0, bound), written out because standard distributions differ
// across standard libraries. The modulo bias is irrelevant here.
uint32_t Draw(std::mt19937_64& rng, uint32_t bound) {
  return static_cast<uint32_t>(rng() % bound);
}

std::vector<uint32_t> Ids(const Bitset& bits) {
  std::vector<uint32_t> ids;
  bits.ForEachSetBit([&ids](uint32_t id) { ids.push_back(id); });
  return ids;
}

TEST(BitsetTest, DefaultIsEmpty) {
  const Bitset bits;
  EXPECT_EQ(bits.size(), 0u);
  EXPECT_EQ(bits.Count(), 0);
  EXPECT_THAT(Ids(bits), ElementsAre());
}

TEST(BitsetTest, SetClearAndTestAcrossWordBoundaries) {
  Bitset bits(130);
  for (const uint32_t id : {0u, 63u, 64u, 129u}) bits.Set(id);
  EXPECT_TRUE(bits.Test(63));
  EXPECT_FALSE(bits.Test(62));
  EXPECT_EQ(bits.Count(), 4);
  EXPECT_THAT(Ids(bits), ElementsAre(0, 63, 64, 129));
  bits.Clear(64);
  bits.Clear(65);
  EXPECT_FALSE(bits.Test(64));
  EXPECT_THAT(Ids(bits), ElementsAre(0, 63, 129));
}

TEST(BitsetTest, ShrinkingDropsIdsForGood) {
  Bitset bits(100);
  for (const uint32_t id : {10u, 64u, 70u, 99u}) bits.Set(id);
  bits.Resize(65);
  EXPECT_EQ(bits.size(), 65u);
  EXPECT_THAT(Ids(bits), ElementsAre(10, 64));
  bits.Resize(200);
  EXPECT_THAT(Ids(bits), ElementsAre(10, 64));
  EXPECT_EQ(bits.Count(), 2);
}

TEST(BitsetTest, EqualityComparesSizeAndIds) {
  Bitset a(10);
  Bitset b(11);
  EXPECT_NE(a, b);
  b.Resize(10);
  EXPECT_EQ(a, b);
  a.Set(3);
  EXPECT_NE(a, b);
  b.Set(3);
  EXPECT_EQ(a, b);
}

TEST(BitsetTest, CopiesAreIndependent) {
  Bitset original(64);
  original.Set(1);
  Bitset copy = original;
  copy.Set(2);
  EXPECT_THAT(Ids(original), ElementsAre(1));
  EXPECT_THAT(Ids(copy), ElementsAre(1, 2));
}

TEST(BitsetTest, IntersectWithShorterSetClearsIdsBeyondIt) {
  Bitset bits(200);
  bits.Set(5);
  bits.Set(6);
  bits.Set(150);
  Bitset other(100);
  other.Set(5);
  bits.IntersectWith(other);
  EXPECT_EQ(bits.size(), 200u);
  EXPECT_THAT(Ids(bits), ElementsAre(5));
}

TEST(BitsetTest, UnionWithLongerSetIgnoresIdsBeyondSize) {
  Bitset bits(70);
  bits.Set(1);
  Bitset other(200);
  for (const uint32_t id : {69u, 70u, 127u, 150u}) other.Set(id);
  bits.UnionWith(other);
  EXPECT_EQ(bits.size(), 70u);
  EXPECT_THAT(Ids(bits), ElementsAre(1, 69));
  EXPECT_EQ(bits.Count(), 2);
  Bitset expected(70);
  expected.Set(1);
  expected.Set(69);
  EXPECT_EQ(bits, expected);
}

// A random set of `size` ids with a random density, and the same set as a
// std::vector<bool>.
Bitset RandomBitset(std::mt19937_64& rng, uint32_t size,
                    std::vector<bool>& model) {
  Bitset bits(size);
  model.assign(size, false);
  const uint32_t density = Draw(rng, 101);
  for (uint32_t id = 0; id < size; ++id) {
    if (Draw(rng, 100) < density) {
      bits.Set(id);
      model[id] = true;
    }
  }
  return bits;
}

void ExpectMatchesModel(const Bitset& bits, const std::vector<bool>& model) {
  ASSERT_EQ(bits.size(), model.size());
  std::vector<uint32_t> expected_ids;
  Bitset expected(static_cast<uint32_t>(model.size()));
  for (uint32_t id = 0; id < model.size(); ++id) {
    ASSERT_EQ(bits.Test(id), model[id]) << "id " << id;
    if (model[id]) {
      expected_ids.push_back(id);
      expected.Set(id);
    }
  }
  EXPECT_EQ(bits.Count(), static_cast<int64_t>(expected_ids.size()));
  EXPECT_EQ(Ids(bits), expected_ids);
  // Holds only if every operation kept the bits beyond size() clear.
  EXPECT_EQ(bits, expected);
}

TEST(BitsetTest, MatchesVectorBoolModel) {
  const uint64_t seed = ferret_testing::TestSeed(20260923);
  SCOPED_TRACE(::testing::Message() << "seed " << seed);
  std::mt19937_64 rng(seed);
  constexpr uint32_t kMaxSize = 300;
  for (int round = 0; round < 100; ++round) {
    std::vector<bool> model;
    Bitset bits = RandomBitset(rng, Draw(rng, kMaxSize), model);
    for (int step = 0; step < 200; ++step) {
      const uint32_t size = bits.size();
      switch (Draw(rng, 6)) {
        case 0:
          if (size > 0) {
            const uint32_t id = Draw(rng, size);
            bits.Set(id);
            model[id] = true;
          }
          break;
        case 1:
          if (size > 0) {
            const uint32_t id = Draw(rng, size);
            bits.Clear(id);
            model[id] = false;
          }
          break;
        case 2: {
          const uint32_t new_size = Draw(rng, kMaxSize);
          bits.Resize(new_size);
          model.resize(new_size, false);
          break;
        }
        case 3: {
          std::vector<bool> other_model;
          const Bitset other =
              RandomBitset(rng, Draw(rng, kMaxSize), other_model);
          bits.IntersectWith(other);
          for (uint32_t id = 0; id < size; ++id) {
            model[id] = model[id] && id < other_model.size() && other_model[id];
          }
          break;
        }
        case 4: {
          std::vector<bool> other_model;
          const Bitset other =
              RandomBitset(rng, Draw(rng, kMaxSize), other_model);
          bits.UnionWith(other);
          for (uint32_t id = 0; id < size && id < other_model.size(); ++id) {
            model[id] = model[id] || other_model[id];
          }
          break;
        }
        default: {
          const Bitset copy = bits;
          EXPECT_EQ(copy, bits);
          bits = copy;
          break;
        }
      }
      ASSERT_NO_FATAL_FAILURE(ExpectMatchesModel(bits, model))
          << "round " << round << " step " << step;
    }
  }
}

}  // namespace
}  // namespace ferret

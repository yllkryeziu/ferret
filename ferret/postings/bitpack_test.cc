// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/postings/bitpack.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/postings/testing/random_posting_lists.h"
#include "ferret/testing/test_seed.h"

namespace ferret {
namespace {

using ::ferret_testing::UniformBelow;
using ::testing::ElementsAreArray;

using Block = std::array<uint32_t, kBitPackBlockSize>;

struct Unpacker {
  const char* name;
  decltype(&UnpackDeltaBlockScalar) unpack;
};

constexpr Unpacker kUnpackers[] = {
    {"scalar", &UnpackDeltaBlockScalar},
#if defined(__aarch64__)
    {"neon", &UnpackDeltaBlockNeon},
#endif  // defined(__aarch64__)
};

constexpr DeltaCoding kCodings[] = {DeltaCoding::kD1, DeltaCoding::kD4};

const char* CodingName(DeltaCoding coding) {
  return coding == DeltaCoding::kD1 ? "D1" : "D4";
}

// Returns 128 deltas whose smallest common bit width is exactly `bit_width`.
Block RandomDeltas(std::mt19937_64& rng, int bit_width) {
  Block deltas = {};
  if (bit_width == 0) return deltas;
  const uint64_t bound = uint64_t{1} << bit_width;
  for (uint32_t& delta : deltas) {
    delta = static_cast<uint32_t>(UniformBelow(rng, bound));
  }
  deltas[UniformBelow(rng, kBitPackBlockSize)] |= uint32_t{1}
                                                  << (bit_width - 1);
  return deltas;
}

// Inverts the delta coding documented in bitpack.h, modulo 2^32.
Block ValuesFromDeltas(DeltaCoding coding, uint32_t base, const Block& deltas) {
  Block values;
  for (size_t i = 0; i < values.size(); ++i) {
    uint32_t predecessor;
    if (coding == DeltaCoding::kD1) {
      predecessor = i == 0 ? base : values[i - 1];
    } else {
      predecessor =
          i >= 4 ? values[i - 4] : base - 3 + static_cast<uint32_t>(i);
    }
    const uint32_t min_step = coding == DeltaCoding::kD1 ? 1 : 4;
    values[i] = predecessor + deltas[i] + min_step;
  }
  return values;
}

// Packs into a heap buffer of exactly the packed size, so that ASan catches
// any write past it.
std::unique_ptr<uint8_t[]> Pack(DeltaCoding coding, uint32_t base,
                                const Block& values, int bit_width) {
  auto packed = std::make_unique<uint8_t[]>(
      static_cast<size_t>(PackedBlockBytes(bit_width)));
  PackDeltaBlock(coding, base, values, bit_width, packed.get());
  return packed;
}

// Unpacks into a heap buffer of exactly one block.
std::vector<uint32_t> Unpack(const Unpacker& unpacker, DeltaCoding coding,
                             uint32_t base, const uint8_t* packed,
                             int bit_width) {
  auto out = std::make_unique<uint32_t[]>(kBitPackBlockSize);
  unpacker.unpack(coding, base, packed, bit_width, out.get());
  return std::vector<uint32_t>(out.get(), out.get() + kBitPackBlockSize);
}

std::vector<uint8_t> Bytes(const uint8_t* data, int size) {
  return std::vector<uint8_t>(data, data + size);
}

// Values 0..127 after base -1, except that every value from index 5 on is one
// higher: a single D1 delta of 1 at index 5.
Block ShiftedAtFive() {
  Block values;
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<uint32_t>(i < 5 ? i : i + 1);
  }
  return values;
}

TEST(BitPackFormatTest, D1LayoutIsVerticalAcrossFourLanes) {
  constexpr uint32_t kBase = 0xFFFFFFFF;
  const Block values = ShiftedAtFive();
  ASSERT_EQ(DeltaBlockBitWidth(DeltaCoding::kD1, kBase, values), 1);
  const std::unique_ptr<uint8_t[]> packed =
      Pack(DeltaCoding::kD1, kBase, values, 1);
  // Delta 5 is value 1 of lane 1: bit 1 of word 4 * 0 + 1.
  EXPECT_THAT(Bytes(packed.get(), 16),
              ElementsAreArray<uint8_t>(
                  {0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(BitPackFormatTest, D4DeltasAreTakenFourPositionsBack) {
  constexpr uint32_t kBase = 0xFFFFFFFF;
  const Block values = ShiftedAtFive();
  ASSERT_EQ(DeltaBlockBitWidth(DeltaCoding::kD4, kBase, values), 1);
  const std::unique_ptr<uint8_t[]> packed =
      Pack(DeltaCoding::kD4, kBase, values, 1);
  // The shift shows in deltas 5, 6, 7 and 8: value 1 of lanes 1, 2 and 3 and
  // value 2 of lane 0.
  EXPECT_THAT(Bytes(packed.get(), 16),
              ElementsAreArray<uint8_t>(
                  {4, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0}));
}

TEST(BitPackFormatTest, ValueCrossingAWordContinuesInItsLanesNextWord) {
  // At width 3, value 40 (lane 0, row 10) occupies bits 30..32 of lane 0:
  // the top two bits of lane 0's word 0 (stored word 0) and the lowest bit of
  // its word 1 (stored word 4 * 1 + 0, bytes 16..19).
  Block deltas = {};
  deltas[40] = 7;
  constexpr uint32_t kBase = 1000;
  const Block values = ValuesFromDeltas(DeltaCoding::kD1, kBase, deltas);
  ASSERT_EQ(DeltaBlockBitWidth(DeltaCoding::kD1, kBase, values), 3);
  const std::unique_ptr<uint8_t[]> packed =
      Pack(DeltaCoding::kD1, kBase, values, 3);
  std::vector<uint8_t> expected(static_cast<size_t>(PackedBlockBytes(3)), 0);
  expected[3] = 0xC0;
  expected[16] = 0x01;
  EXPECT_EQ(Bytes(packed.get(), PackedBlockBytes(3)), expected);
  for (const Unpacker& unpacker : kUnpackers) {
    EXPECT_THAT(Unpack(unpacker, DeltaCoding::kD1, kBase, packed.get(), 3),
                ElementsAreArray(values))
        << unpacker.name;
  }
}

TEST(BitPackFormatTest, ConsecutiveValuesPackAtWidthZero) {
  for (const uint32_t base : {uint32_t{0xFFFFFFFF}, uint32_t{0},
                              uint32_t{0xFFFFFFFF - kBitPackBlockSize}}) {
    Block values;
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = base + 1 + static_cast<uint32_t>(i);
    }
    for (const DeltaCoding coding : kCodings) {
      SCOPED_TRACE(CodingName(coding));
      EXPECT_EQ(DeltaBlockBitWidth(coding, base, values), 0);
      const std::unique_ptr<uint8_t[]> packed = Pack(coding, base, values, 0);
      for (const Unpacker& unpacker : kUnpackers) {
        EXPECT_THAT(Unpack(unpacker, coding, base, packed.get(), 0),
                    ElementsAreArray(values))
            << unpacker.name;
      }
    }
  }
}

class BitPackTest : public ::testing::TestWithParam<int> {};

TEST_P(BitPackTest, RoundTripsAtExactWidth) {
  const int bit_width = GetParam();
  const uint64_t seed = ferret_testing::TestSeed(0x5eed0100 + bit_width);
  SCOPED_TRACE(testing::Message() << "seed " << seed);
  std::mt19937_64 rng(seed);
  for (int iteration = 0; iteration < 50; ++iteration) {
    for (const DeltaCoding coding : kCodings) {
      SCOPED_TRACE(CodingName(coding));
      const auto base = static_cast<uint32_t>(rng());
      const Block values =
          ValuesFromDeltas(coding, base, RandomDeltas(rng, bit_width));
      ASSERT_EQ(DeltaBlockBitWidth(coding, base, values), bit_width);
      const std::unique_ptr<uint8_t[]> packed =
          Pack(coding, base, values, bit_width);
      for (const Unpacker& unpacker : kUnpackers) {
        ASSERT_THAT(Unpack(unpacker, coding, base, packed.get(), bit_width),
                    ElementsAreArray(values))
            << unpacker.name;
      }
    }
  }
}

TEST_P(BitPackTest, RoundTripsAtWiderWidth) {
  const int bit_width = GetParam();
  const uint64_t seed = ferret_testing::TestSeed(0x5eed0200 + bit_width);
  SCOPED_TRACE(testing::Message() << "seed " << seed);
  std::mt19937_64 rng(seed);
  for (const DeltaCoding coding : kCodings) {
    SCOPED_TRACE(CodingName(coding));
    const auto base = static_cast<uint32_t>(rng());
    const Block values =
        ValuesFromDeltas(coding, base, RandomDeltas(rng, bit_width));
    for (int wider = bit_width; wider <= kMaxBitPackWidth; ++wider) {
      const std::unique_ptr<uint8_t[]> packed =
          Pack(coding, base, values, wider);
      for (const Unpacker& unpacker : kUnpackers) {
        ASSERT_THAT(Unpack(unpacker, coding, base, packed.get(), wider),
                    ElementsAreArray(values))
            << unpacker.name << " at width " << wider;
      }
    }
  }
}

// Every payload decodes to something; the NEON kernels must agree with the
// scalar reference on arbitrary bytes, not just on what PackDeltaBlock
// writes.
TEST_P(BitPackTest, UnpackersAgreeOnArbitraryBytes) {
  const int bit_width = GetParam();
  const uint64_t seed = ferret_testing::TestSeed(0x5eed0300 + bit_width);
  SCOPED_TRACE(testing::Message() << "seed " << seed);
  std::mt19937_64 rng(seed);
  const auto size = static_cast<size_t>(PackedBlockBytes(bit_width));
  for (int iteration = 0; iteration < 50; ++iteration) {
    auto packed = std::make_unique<uint8_t[]>(size);
    for (size_t i = 0; i < size; ++i) packed[i] = static_cast<uint8_t>(rng());
    const auto base = static_cast<uint32_t>(rng());
    for (const DeltaCoding coding : kCodings) {
      SCOPED_TRACE(CodingName(coding));
      const std::vector<uint32_t> expected =
          Unpack(kUnpackers[0], coding, base, packed.get(), bit_width);
      for (const Unpacker& unpacker : kUnpackers) {
        ASSERT_EQ(Unpack(unpacker, coding, base, packed.get(), bit_width),
                  expected)
            << unpacker.name;
      }
      // And packing what came out reproduces the bytes exactly.
      Block values;
      std::copy(expected.begin(), expected.end(), values.begin());
      const std::unique_ptr<uint8_t[]> repacked =
          Pack(coding, base, values, bit_width);
      ASSERT_EQ(Bytes(repacked.get(), PackedBlockBytes(bit_width)),
                Bytes(packed.get(), PackedBlockBytes(bit_width)));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(AllBitWidths, BitPackTest,
                         ::testing::Range(0, kMaxBitPackWidth + 1));

}  // namespace
}  // namespace ferret

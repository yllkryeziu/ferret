// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/postings/varint.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/postings/testing/random_posting_lists.h"
#include "ferret/testing/test_seed.h"

namespace ferret {
namespace {

using ::testing::Optional;

std::string Varint32(uint32_t value) {
  std::string out;
  AppendVarint32(value, out);
  return out;
}

std::string Varint64(uint64_t value) {
  std::string out;
  AppendVarint64(value, out);
  return out;
}

TEST(VarintTest, EncodesLeastSignificantGroupFirst) {
  EXPECT_EQ(Varint32(0), std::string(1, '\0'));
  EXPECT_EQ(Varint32(1), "\x01");
  EXPECT_EQ(Varint32(127), "\x7F");
  EXPECT_EQ(Varint32(128), "\x80\x01");
  EXPECT_EQ(Varint32(300), "\xAC\x02");
  EXPECT_EQ(Varint32(std::numeric_limits<uint32_t>::max()),
            "\xFF\xFF\xFF\xFF\x0F");
  EXPECT_EQ(Varint64(std::numeric_limits<uint64_t>::max()),
            "\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\x01");
}

TEST(VarintTest, LengthGrowsAtEverySevenBitBoundary) {
  for (int bits = 1; bits <= 32; ++bits) {
    const uint32_t largest = static_cast<uint32_t>((uint64_t{1} << bits) - 1);
    EXPECT_EQ(Varint32(largest).size(), static_cast<size_t>((bits + 6) / 7))
        << "bits " << bits;
  }
  for (int bits = 1; bits <= 64; ++bits) {
    const uint64_t largest = bits == 64 ? std::numeric_limits<uint64_t>::max()
                                        : (uint64_t{1} << bits) - 1;
    EXPECT_EQ(Varint64(largest).size(), static_cast<size_t>((bits + 6) / 7))
        << "bits " << bits;
  }
}

TEST(VarintTest, BoundaryValuesRoundTrip) {
  for (int bits = 0; bits <= 64; ++bits) {
    const uint64_t power = bits == 64 ? 0 : uint64_t{1} << bits;
    for (const uint64_t value : {power - 1, power, power + 1}) {
      std::string encoded = Varint64(value);
      absl::string_view input = encoded;
      EXPECT_THAT(ConsumeVarint64(input), Optional(value));
      EXPECT_TRUE(input.empty());
      if (value <= std::numeric_limits<uint32_t>::max()) {
        const auto value32 = static_cast<uint32_t>(value);
        encoded = Varint32(value32);
        input = encoded;
        EXPECT_THAT(ConsumeVarint32(input), Optional(value32));
        EXPECT_TRUE(input.empty());
      }
    }
  }
}

TEST(VarintTest, ConsumesOneVarintFromAStream) {
  std::string stream;
  AppendVarint32(5, stream);
  AppendVarint32(1000000, stream);
  stream += "rest";
  absl::string_view input = stream;
  EXPECT_THAT(ConsumeVarint32(input), Optional(5u));
  EXPECT_THAT(ConsumeVarint32(input), Optional(1000000u));
  EXPECT_EQ(input, "rest");
}

TEST(VarintTest, TruncatedInputIsRejectedAndNotConsumed) {
  const std::string encoded32 = Varint32(std::numeric_limits<uint32_t>::max());
  const std::string encoded64 = Varint64(std::numeric_limits<uint64_t>::max());
  for (size_t length = 0; length < encoded32.size(); ++length) {
    absl::string_view input(encoded32.data(), length);
    EXPECT_EQ(ConsumeVarint32(input), std::nullopt) << length;
    EXPECT_EQ(input.size(), length);
  }
  for (size_t length = 0; length < encoded64.size(); ++length) {
    absl::string_view input(encoded64.data(), length);
    EXPECT_EQ(ConsumeVarint64(input), std::nullopt) << length;
    EXPECT_EQ(input.size(), length);
  }
}

TEST(VarintTest, ValuesWiderThanTheTypeAreRejected) {
  for (const absl::string_view bad : {
           absl::string_view("\xFF\xFF\xFF\xFF\x10"),         // Bit 32.
           absl::string_view("\x80\x80\x80\x80\x80\x00", 6),  // 6 bytes.
       }) {
    absl::string_view input = bad;
    EXPECT_EQ(ConsumeVarint32(input), std::nullopt) << absl::CHexEscape(bad);
    EXPECT_EQ(input, bad);
  }
  for (const absl::string_view bad : {
           absl::string_view("\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\x02"),
           absl::string_view("\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x00",
                             11),
       }) {
    absl::string_view input = bad;
    EXPECT_EQ(ConsumeVarint64(input), std::nullopt) << absl::CHexEscape(bad);
    EXPECT_EQ(input, bad);
  }
}

TEST(VarintTest, RedundantZeroGroupsAreAccepted) {
  absl::string_view input("\x81\x80\x00", 3);
  EXPECT_THAT(ConsumeVarint32(input), Optional(1u));
  EXPECT_TRUE(input.empty());
}

TEST(VarintTest, RandomStreamsRoundTrip) {
  const uint64_t seed = ferret_testing::TestSeed(0x5eed0001);
  SCOPED_TRACE(testing::Message() << "seed " << seed);
  std::mt19937_64 rng(seed);
  for (int iteration = 0; iteration < 200; ++iteration) {
    std::vector<uint64_t> values(ferret_testing::UniformBelow(rng, 64));
    std::string stream;
    for (uint64_t& value : values) {
      // Uniform bit lengths, so that every encoded length is common.
      const int bits = static_cast<int>(ferret_testing::UniformBelow(rng, 65));
      value = bits == 0 ? 0 : rng() >> (64 - bits);
      AppendVarint64(value, stream);
    }
    absl::string_view input = stream;
    for (const uint64_t value : values) {
      ASSERT_THAT(ConsumeVarint64(input), Optional(value));
    }
    EXPECT_TRUE(input.empty());
  }
}

}  // namespace
}  // namespace ferret

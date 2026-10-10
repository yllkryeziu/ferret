// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/simd/testing/exact_id_array.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ferret {
namespace {

using ::ferret_testing::AllSubsets;
using ::ferret_testing::ExactIdArray;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(ExactIdArrayTest, SizedArrayIsZeroed) {
  ExactIdArray array(3);
  EXPECT_EQ(array.size(), 3u);
  EXPECT_THAT(array.span(), ElementsAre(0, 0, 0));
  array.data()[1] = 7;
  EXPECT_THAT(array.span(), ElementsAre(0, 7, 0));
}

TEST(ExactIdArrayTest, CopiesIds) {
  const std::vector<uint32_t> ids = {1, 5, 9};
  EXPECT_THAT(ExactIdArray(ids).span(), ElementsAre(1, 5, 9));
  EXPECT_THAT(ExactIdArray(absl::Span<const uint32_t>()).span(), IsEmpty());
}

TEST(ExactIdArrayTest, MoveLeavesSourceEmpty) {
  ExactIdArray source(std::vector<uint32_t>{1, 5, 9});
  ExactIdArray constructed(std::move(source));
  EXPECT_THAT(constructed.span(), ElementsAre(1, 5, 9));
  EXPECT_EQ(source.size(), 0u);  // NOLINT(bugprone-use-after-move)
  EXPECT_THAT(source.span(), IsEmpty());

  ExactIdArray assigned(2);
  assigned = std::move(constructed);
  EXPECT_THAT(assigned.span(), ElementsAre(1, 5, 9));
  EXPECT_EQ(constructed.size(), 0u);  // NOLINT(bugprone-use-after-move)
  EXPECT_THAT(constructed.span(), IsEmpty());
}

TEST(ExactIdArrayTest, AllSubsetsAreIndexedByBitmask) {
  const std::vector<ExactIdArray> subsets = AllSubsets(3);
  ASSERT_EQ(subsets.size(), 8u);
  EXPECT_THAT(subsets[0b000].span(), IsEmpty());
  EXPECT_THAT(subsets[0b001].span(), ElementsAre(0));
  EXPECT_THAT(subsets[0b110].span(), ElementsAre(1, 2));
  EXPECT_THAT(subsets[0b111].span(), ElementsAre(0, 1, 2));
}

}  // namespace
}  // namespace ferret

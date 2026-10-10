// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/simd/dispatch.h"

#include <cstddef>
#include <limits>

#include "gtest/gtest.h"

namespace ferret {
namespace {

TEST(DispatchTest, GallopsFromTheCrossoverRatio) {
  EXPECT_FALSE(IntersectShouldGallop(100, 100));
  EXPECT_FALSE(IntersectShouldGallop(100, 100 * kIntersectGallopRatio - 1));
  EXPECT_TRUE(IntersectShouldGallop(100, 100 * kIntersectGallopRatio));
  EXPECT_FALSE(IntersectShouldGallop(100 * kIntersectGallopRatio - 1, 100));
  EXPECT_TRUE(IntersectShouldGallop(100 * kIntersectGallopRatio, 100));
}

TEST(DispatchTest, GallopsWhenAnInputIsEmpty) {
  EXPECT_TRUE(IntersectShouldGallop(0, 0));
  EXPECT_TRUE(IntersectShouldGallop(0, 1));
  EXPECT_TRUE(IntersectShouldGallop(1, 0));
}

TEST(DispatchTest, HugeSizesDoNotOverflow) {
  constexpr size_t kHuge = std::numeric_limits<size_t>::max() / 2;
  EXPECT_FALSE(IntersectShouldGallop(kHuge, kHuge));
  EXPECT_TRUE(IntersectShouldGallop(1, kHuge));
}

}  // namespace
}  // namespace ferret

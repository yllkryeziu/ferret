// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/simd/intersect.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/simd/dispatch.h"
#include "ferret/simd/testing/exact_id_array.h"
#include "ferret/simd/testing/random_ids.h"
#include "ferret/testing/test_seed.h"

namespace ferret {
namespace {

using ::ferret_testing::ExactIdArray;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::PrintToString;

constexpr uint32_t kMaxId = std::numeric_limits<uint32_t>::max();

// Fills output buffers before each call, so that a kernel that skips a write
// cannot pass on a value left by an earlier call or by zero-initialization.
constexpr uint32_t kUnwritten = 0xdeadbeef;

struct IntersectKernel {
  const char* name;
  size_t (*fn)(absl::Span<const uint32_t>, absl::Span<const uint32_t>,
               uint32_t* absl_nonnull);
};

void PrintTo(const IntersectKernel& kernel, std::ostream* os) {
  *os << kernel.name;
}

constexpr IntersectKernel kIntersectKernels[] = {
    {"Dispatch", &Intersect},
    {"Scalar", &IntersectScalar},
    {"Galloping", &IntersectGalloping},
#if defined(__aarch64__)
    {"Neon", &IntersectNeon},
#endif
};

// std::set_intersection is independent of every kernel here, including
// IntersectScalar, which the rest of the project uses as its oracle.
std::vector<uint32_t> ReferenceIntersection(absl::Span<const uint32_t> a,
                                            absl::Span<const uint32_t> b) {
  std::vector<uint32_t> result;
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(result));
  return result;
}

std::vector<uint32_t> Range(uint32_t first, uint32_t last, uint32_t step) {
  std::vector<uint32_t> ids;
  for (uint32_t id = first; id < last; id += step) ids.push_back(id);
  return ids;
}

class IntersectKernelTest : public ::testing::TestWithParam<IntersectKernel> {
 protected:
  // Runs the kernel under test with its inputs and output in exactly sized
  // heap arrays, so that AddressSanitizer reports any access the contract
  // does not allow, and returns the result.
  std::vector<uint32_t> Run(absl::Span<const uint32_t> a,
                            absl::Span<const uint32_t> b) const {
    const ExactIdArray exact_a(a);
    const ExactIdArray exact_b(b);
    ExactIdArray out(IntersectCapacity(a.size(), b.size()));
    std::fill_n(out.data(), out.size(), kUnwritten);
    const size_t count =
        GetParam().fn(exact_a.span(), exact_b.span(), out.data());
    EXPECT_LE(count, std::min(a.size(), b.size()));
    const absl::Span<const uint32_t> result =
        out.span().first(std::min(count, out.size()));
    return std::vector<uint32_t>(result.begin(), result.end());
  }
};

TEST_P(IntersectKernelTest, EmptyInputs) {
  const std::vector<uint32_t> ids = Range(1, 9, 1);
  EXPECT_THAT(Run({}, {}), IsEmpty());
  EXPECT_THAT(Run({}, ids), IsEmpty());
  EXPECT_THAT(Run(ids, {}), IsEmpty());
}

TEST_P(IntersectKernelTest, IdenticalInputs) {
  const std::vector<uint32_t> ids = Range(0, 309, 3);
  EXPECT_EQ(Run(ids, ids), ids);
}

TEST_P(IntersectKernelTest, InterleavedInputsAreDisjoint) {
  const std::vector<uint32_t> even = Range(0, 202, 2);
  const std::vector<uint32_t> odd = Range(1, 202, 2);
  EXPECT_THAT(Run(even, odd), IsEmpty());
  EXPECT_THAT(Run(odd, even), IsEmpty());
}

TEST_P(IntersectKernelTest, ContainedInputIsReturned) {
  const std::vector<uint32_t> all = Range(0, 1000, 1);
  const std::vector<uint32_t> some = Range(5, 1000, 37);
  EXPECT_EQ(Run(all, some), some);
  EXPECT_EQ(Run(some, all), some);
}

TEST_P(IntersectKernelTest, ExtremeIds) {
  const std::vector<uint32_t> a = {0, 1, 2, 3, kMaxId - 2, kMaxId - 1, kMaxId};
  const std::vector<uint32_t> b = {0, 2, kMaxId - 1, kMaxId};
  EXPECT_THAT(Run(a, b), ElementsAre(0, 2, kMaxId - 1, kMaxId));
  EXPECT_THAT(Run(b, a), ElementsAre(0, 2, kMaxId - 1, kMaxId));
}

// Every pair of subsets of {0, ..., kUniverse - 1}. This covers every
// interleaving, with and without shared ids, of two lists whose lengths sum
// to at most kUniverse, and multi-block pairs for the NEON kernel.
TEST_P(IntersectKernelTest, MatchesBitmaskOnAllSubsetPairs) {
  constexpr int kUniverse = 11;
  const std::vector<ExactIdArray> subsets =
      ferret_testing::AllSubsets(kUniverse);
  // outputs[n] has exactly the capacity required when the shorter input has n
  // elements; reusing them keeps the test fast.
  std::vector<ExactIdArray> outputs;
  for (size_t n = 0; n <= kUniverse; ++n) {
    outputs.emplace_back(IntersectCapacity(n, n));
  }
  for (size_t mask_a = 0; mask_a < subsets.size(); ++mask_a) {
    for (size_t mask_b = 0; mask_b < subsets.size(); ++mask_b) {
      const absl::Span<const uint32_t> a = subsets[mask_a].span();
      const absl::Span<const uint32_t> b = subsets[mask_b].span();
      const absl::Span<const uint32_t> expected =
          subsets[mask_a & mask_b].span();
      ExactIdArray& out = outputs[std::min(a.size(), b.size())];
      std::fill_n(out.data(), out.size(), kUnwritten);
      const size_t count = GetParam().fn(a, b, out.data());
      ASSERT_EQ(count, expected.size())
          << "a=" << PrintToString(a) << " b=" << PrintToString(b);
      ASSERT_EQ(out.span().first(count), expected)
          << "a=" << PrintToString(a) << " b=" << PrintToString(b);
    }
  }
}

TEST_P(IntersectKernelTest, MatchesReferenceAroundGallopCrossover) {
  const uint64_t seed = ferret_testing::TestSeed(0x6a11095);
  SCOPED_TRACE(absl::StrCat("seed=", seed));
  std::mt19937_64 rng(seed);
  for (const size_t shorter_size : {1, 7, 100}) {
    for (const size_t ratio : {kIntersectGallopRatio - 1, kIntersectGallopRatio,
                               kIntersectGallopRatio + 1}) {
      const std::vector<uint32_t> pool = ferret_testing::RandomIncreasingIds(
          rng, 2 * shorter_size * ratio, 0, 3);
      const std::vector<uint32_t> longer =
          ferret_testing::RandomSubset(rng, pool, shorter_size * ratio);
      const std::vector<uint32_t> shorter =
          ferret_testing::RandomSubset(rng, pool, shorter_size);
      const std::vector<uint32_t> expected =
          ReferenceIntersection(shorter, longer);
      EXPECT_EQ(Run(shorter, longer), expected) << "ratio " << ratio;
      EXPECT_EQ(Run(longer, shorter), expected) << "ratio " << ratio;
    }
  }
}

TEST_P(IntersectKernelTest, MatchesReferenceOnRandomLists) {
  const uint64_t seed = ferret_testing::TestSeed(0x1d5eed);
  SCOPED_TRACE(absl::StrCat("seed=", seed));
  std::mt19937_64 rng(seed);
  for (int iteration = 0; iteration < 1000; ++iteration) {
    const ferret_testing::IdListPair pair =
        ferret_testing::RandomIdListPair(rng);
    ASSERT_EQ(Run(pair.a, pair.b), ReferenceIntersection(pair.a, pair.b))
        << "iteration " << iteration << ": |a|=" << pair.a.size()
        << " |b|=" << pair.b.size();
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllKernels, IntersectKernelTest, ::testing::ValuesIn(kIntersectKernels),
    [](const ::testing::TestParamInfo<IntersectKernel>& info) {
      return std::string(info.param.name);
    });

}  // namespace
}  // namespace ferret

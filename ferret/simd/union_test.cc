// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/simd/union.h"

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

struct UnionKernel {
  const char* name;
  size_t (*fn)(absl::Span<const uint32_t>, absl::Span<const uint32_t>,
               uint32_t* absl_nonnull);
};

void PrintTo(const UnionKernel& kernel, std::ostream* os) {
  *os << kernel.name;
}

constexpr UnionKernel kUnionKernels[] = {
    {"Dispatch", &Union},
    {"Scalar", &UnionScalar},
#if defined(__aarch64__)
    {"Neon", &UnionNeon},
#endif
};

// std::set_union is independent of every kernel here, including UnionScalar,
// which the rest of the project uses as its oracle.
std::vector<uint32_t> ReferenceUnion(absl::Span<const uint32_t> a,
                                     absl::Span<const uint32_t> b) {
  std::vector<uint32_t> result;
  std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                 std::back_inserter(result));
  return result;
}

std::vector<uint32_t> Range(uint32_t first, uint32_t last, uint32_t step) {
  std::vector<uint32_t> ids;
  for (uint32_t id = first; id < last; id += step) ids.push_back(id);
  return ids;
}

class UnionKernelTest : public ::testing::TestWithParam<UnionKernel> {
 protected:
  // Runs the kernel under test with its inputs and output in exactly sized
  // heap arrays, so that AddressSanitizer reports any access the contract
  // does not allow, and returns the result.
  std::vector<uint32_t> Run(absl::Span<const uint32_t> a,
                            absl::Span<const uint32_t> b) const {
    const ExactIdArray exact_a(a);
    const ExactIdArray exact_b(b);
    ExactIdArray out(UnionCapacity(a.size(), b.size()));
    std::fill_n(out.data(), out.size(), kUnwritten);
    const size_t count =
        GetParam().fn(exact_a.span(), exact_b.span(), out.data());
    EXPECT_LE(count, a.size() + b.size());
    const absl::Span<const uint32_t> result =
        out.span().first(std::min(count, out.size()));
    return std::vector<uint32_t>(result.begin(), result.end());
  }
};

TEST_P(UnionKernelTest, EmptyInputs) {
  const std::vector<uint32_t> ids = Range(1, 9, 1);
  EXPECT_THAT(Run({}, {}), IsEmpty());
  EXPECT_EQ(Run({}, ids), ids);
  EXPECT_EQ(Run(ids, {}), ids);
}

TEST_P(UnionKernelTest, IdenticalInputsAreDeduplicated) {
  const std::vector<uint32_t> ids = Range(0, 309, 3);
  EXPECT_EQ(Run(ids, ids), ids);
}

TEST_P(UnionKernelTest, InterleavedInputsMerge) {
  const std::vector<uint32_t> even = Range(0, 202, 2);
  const std::vector<uint32_t> odd = Range(1, 202, 2);
  const std::vector<uint32_t> all = Range(0, 202, 1);
  EXPECT_EQ(Run(even, odd), all);
  EXPECT_EQ(Run(odd, even), all);
}

TEST_P(UnionKernelTest, SeparatedInputsConcatenate) {
  const std::vector<uint32_t> low = Range(0, 100, 1);
  const std::vector<uint32_t> high = Range(1000, 1100, 1);
  std::vector<uint32_t> both = low;
  both.insert(both.end(), high.begin(), high.end());
  EXPECT_EQ(Run(low, high), both);
  EXPECT_EQ(Run(high, low), both);
}

TEST_P(UnionKernelTest, ExtremeIds) {
  const std::vector<uint32_t> a = {0, 1, 2, 3, kMaxId - 2, kMaxId - 1, kMaxId};
  const std::vector<uint32_t> b = {0, 2, 5, kMaxId - 3, kMaxId - 1, kMaxId};
  EXPECT_THAT(Run(a, b), ElementsAre(0, 1, 2, 3, 5, kMaxId - 3, kMaxId - 2,
                                     kMaxId - 1, kMaxId));
  EXPECT_THAT(Run(b, a), ElementsAre(0, 1, 2, 3, 5, kMaxId - 3, kMaxId - 2,
                                     kMaxId - 1, kMaxId));
}

// Every pair of subsets of {0, ..., kUniverse - 1}. This covers every
// interleaving, with and without shared ids, of two lists whose lengths sum
// to at most kUniverse, and multi-block pairs for the NEON kernel.
TEST_P(UnionKernelTest, MatchesBitmaskOnAllSubsetPairs) {
  constexpr int kUniverse = 11;
  const std::vector<ExactIdArray> subsets =
      ferret_testing::AllSubsets(kUniverse);
  // outputs[n] has exactly the capacity required when the inputs have n
  // elements in total; reusing them keeps the test fast.
  std::vector<ExactIdArray> outputs;
  for (size_t n = 0; n <= 2 * size_t{kUniverse}; ++n) {
    outputs.emplace_back(UnionCapacity(n, 0));
  }
  for (size_t mask_a = 0; mask_a < subsets.size(); ++mask_a) {
    for (size_t mask_b = 0; mask_b < subsets.size(); ++mask_b) {
      const absl::Span<const uint32_t> a = subsets[mask_a].span();
      const absl::Span<const uint32_t> b = subsets[mask_b].span();
      const absl::Span<const uint32_t> expected =
          subsets[mask_a | mask_b].span();
      ExactIdArray& out = outputs[a.size() + b.size()];
      std::fill_n(out.data(), out.size(), kUnwritten);
      const size_t count = GetParam().fn(a, b, out.data());
      ASSERT_EQ(count, expected.size())
          << "a=" << PrintToString(a) << " b=" << PrintToString(b);
      ASSERT_EQ(out.span().first(count), expected)
          << "a=" << PrintToString(a) << " b=" << PrintToString(b);
    }
  }
}

TEST_P(UnionKernelTest, MatchesReferenceOnRandomLists) {
  const uint64_t seed = ferret_testing::TestSeed(0x1d5eed);
  SCOPED_TRACE(absl::StrCat("seed=", seed));
  std::mt19937_64 rng(seed);
  for (int iteration = 0; iteration < 1000; ++iteration) {
    const ferret_testing::IdListPair pair =
        ferret_testing::RandomIdListPair(rng);
    ASSERT_EQ(Run(pair.a, pair.b), ReferenceUnion(pair.a, pair.b))
        << "iteration " << iteration << ": |a|=" << pair.a.size()
        << " |b|=" << pair.b.size();
  }
}

INSTANTIATE_TEST_SUITE_P(AllKernels, UnionKernelTest,
                         ::testing::ValuesIn(kUnionKernels),
                         [](const ::testing::TestParamInfo<UnionKernel>& info) {
                           return std::string(info.param.name);
                         });

}  // namespace
}  // namespace ferret

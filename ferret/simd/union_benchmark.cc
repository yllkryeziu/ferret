// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// UnionScalar against UnionNeon, which decides what Union() dispatches to.
//
// The arguments are those of intersect_benchmark.cc: the shorter list's
// length, the length ratio, and the mean gap between consecutive ids of the
// longer list, with the shorter list spanning the same id range. Run on an
// idle machine:
//
//   bazel run --config=opt //ferret/simd:union_benchmark

#include "ferret/simd/union.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/types/span.h"
#include "benchmark/benchmark.h"

#include "ferret/simd/testing/random_ids.h"

namespace ferret {
namespace {

using UnionFn = size_t (*)(absl::Span<const uint32_t>,
                           absl::Span<const uint32_t>, uint32_t* absl_nonnull);

// As in intersect_benchmark.cc, each iteration takes the next of enough pairs
// to hold this many ids, so that the kernels run on cache-resident lists
// whose branch outcomes the predictor has not learned.
constexpr size_t kMinIdsPerRotation = size_t{1} << 20;

void BM_Union(benchmark::State& state, UnionFn kernel) {
  std::mt19937_64 rng(42);
  // In each pair, `a` is the longer list and `b` the shorter.
  const std::vector<ferret_testing::IdListPair> pairs =
      ferret_testing::RandomListPairsOverSameRange(
          rng, static_cast<size_t>(state.range(0)),
          static_cast<size_t>(state.range(1)),
          static_cast<size_t>(state.range(2)), kMinIdsPerRotation);
  const size_t shorter_size = pairs[0].b.size();
  const size_t longer_size = pairs[0].a.size();
  std::vector<uint32_t> out(UnionCapacity(shorter_size, longer_size));
  size_t next = 0;
  for (auto _ : state) {
    const ferret_testing::IdListPair& pair = pairs[next];
    next = next + 1 == pairs.size() ? 0 : next + 1;
    benchmark::DoNotOptimize(kernel(pair.b, pair.a, out.data()));
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(shorter_size + longer_size));
}

void Sweep(benchmark::Benchmark* benchmark) {
  benchmark->ArgNames({"shorter", "ratio", "gap"});
  for (const int64_t shorter : {128, 4096, 65536}) {
    for (const int64_t ratio : {1, 4, 16, 64}) {
      for (const int64_t gap : {1, 4, 32}) {
        benchmark->Args({shorter, ratio, gap});
      }
    }
  }
}

BENCHMARK_CAPTURE(BM_Union, Scalar, &UnionScalar)->Apply(Sweep);
#if defined(__aarch64__)
BENCHMARK_CAPTURE(BM_Union, Neon, &UnionNeon)->Apply(Sweep);
#endif

}  // namespace
}  // namespace ferret

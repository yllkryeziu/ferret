// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The ratio sweep that sets kIntersectGallopRatio in dispatch.h.
//
// Each benchmark intersects a shorter list with a longer one. The arguments
// are the shorter list's length, the length ratio (1 to 1024), and the mean
// gap between consecutive ids of the longer list: 1 means consecutive ids,
// 32 a density of 1/32. The shorter list spans the same id range, so each of
// its ids is in the longer list with probability about 1/gap. The crossover
// is the smallest ratio from which Galloping beats both merges at every
// length and gap. Run on an idle machine:
//
//   bazel run --config=opt //ferret/simd:intersect_benchmark

#include "ferret/simd/intersect.h"

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

using IntersectFn = size_t (*)(absl::Span<const uint32_t>,
                               absl::Span<const uint32_t>,
                               uint32_t* absl_nonnull);

// Bounds the longer list at 16 MiB.
constexpr int64_t kMaxLongerSize = int64_t{1} << 22;

// Galloping branches on the data, and the branch predictor learns the
// outcomes for inputs it sees over and over: on the M3, galloping over one
// repeated pair of 4096-id lists ran more than twice as fast as over unseen
// lists, which would bias the crossover towards galloping. Each iteration
// therefore takes the next of enough pairs to hold this many ids, which still
// fit in the L2 cache, as freshly decoded lists would.
constexpr size_t kMinIdsPerRotation = size_t{1} << 20;

void BM_Intersect(benchmark::State& state, IntersectFn kernel) {
  std::mt19937_64 rng(42);
  // In each pair, `a` is the longer list and `b` the shorter.
  const std::vector<ferret_testing::IdListPair> pairs =
      ferret_testing::RandomListPairsOverSameRange(
          rng, static_cast<size_t>(state.range(0)),
          static_cast<size_t>(state.range(1)),
          static_cast<size_t>(state.range(2)), kMinIdsPerRotation);
  const size_t shorter_size = pairs[0].b.size();
  const size_t longer_size = pairs[0].a.size();
  std::vector<uint32_t> out(IntersectCapacity(shorter_size, longer_size));
  size_t next = 0;
  int64_t matches = 0;
  for (auto _ : state) {
    const ferret_testing::IdListPair& pair = pairs[next];
    next = next + 1 == pairs.size() ? 0 : next + 1;
    matches += static_cast<int64_t>(kernel(pair.b, pair.a, out.data()));
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(shorter_size + longer_size));
  state.counters["matches"] = benchmark::Counter(
      static_cast<double>(matches), benchmark::Counter::kAvgIterations);
}

void RatioSweep(benchmark::Benchmark* benchmark) {
  benchmark->ArgNames({"shorter", "ratio", "gap"});
  for (const int64_t shorter : {128, 4096, 65536}) {
    for (int64_t ratio = 1; ratio <= 1024 && shorter * ratio <= kMaxLongerSize;
         ratio *= 2) {
      for (const int64_t gap : {1, 4, 32}) {
        benchmark->Args({shorter, ratio, gap});
      }
    }
  }
}

BENCHMARK_CAPTURE(BM_Intersect, Scalar, &IntersectScalar)->Apply(RatioSweep);
BENCHMARK_CAPTURE(BM_Intersect, Galloping, &IntersectGalloping)
    ->Apply(RatioSweep);
#if defined(__aarch64__)
BENCHMARK_CAPTURE(BM_Intersect, Neon, &IntersectNeon)->Apply(RatioSweep);
#endif
BENCHMARK_CAPTURE(BM_Intersect, Dispatch, &Intersect)->Apply(RatioSweep);

}  // namespace
}  // namespace ferret

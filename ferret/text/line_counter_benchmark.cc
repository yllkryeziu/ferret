// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Throughput of CountNewlines against the scalar reference, over a chunk-sized
// input (8 KiB) and a file-sized one (1 MiB).

#include "ferret/text/line_counter.h"

#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"

#include "ferret/text/internal/line_counter_kernels.h"
#include "ferret/text/testing/synthetic_source.h"

namespace ferret {
namespace {

template <int64_t (*kCount)(absl::string_view)>
void BM_CountNewlines(benchmark::State& state) {
  const std::string text =
      ferret_testing::SyntheticSource(state.range(0), /*seed=*/1);
  for (auto _ : state) {
    benchmark::DoNotOptimize(kCount(text));
  }
  state.SetBytesProcessed(state.iterations() * state.range(0));
}

BENCHMARK(BM_CountNewlines<CountNewlines>)->Arg(8 << 10)->Arg(1 << 20);
BENCHMARK(BM_CountNewlines<text_internal::CountNewlinesScalar>)
    ->Arg(8 << 10)
    ->Arg(1 << 20);

}  // namespace
}  // namespace ferret

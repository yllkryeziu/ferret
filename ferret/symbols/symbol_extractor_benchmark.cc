// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Measures symbol extraction throughput on repeated realistic source files.

#include "ferret/symbols/symbol_extractor.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"

#include "ferret/symbols/testing/sample_sources.h"
#include "ferret/text/language.h"

namespace ferret {
namespace {

void ExtractThroughput(benchmark::State& state, Language language) {
  const absl::string_view sample = ferret_testing::SampleSource(language);
  std::string input;
  while (input.size() < static_cast<size_t>(state.range(0))) {
    input.append(sample);
  }
  SymbolExtractor extractor;
  // Compile the language's query outside the timed loop.
  CHECK(extractor.Extract(language, input).ok());
  for (auto unused : state) {
    (void)unused;
    const auto result = extractor.Extract(language, input);
    CHECK(result.ok()) << result.status();
    benchmark::DoNotOptimize(result->size());
  }
  state.SetBytesProcessed(state.iterations() *
                          static_cast<int64_t>(input.size()));
}

void ExtractC(benchmark::State& state) {
  ExtractThroughput(state, Language::kC);
}
void ExtractCpp(benchmark::State& state) {
  ExtractThroughput(state, Language::kCpp);
}
void ExtractPython(benchmark::State& state) {
  ExtractThroughput(state, Language::kPython);
}

BENCHMARK(ExtractC)->Arg(1 << 10)->Arg(16 << 10)->Arg(256 << 10);
BENCHMARK(ExtractCpp)->Arg(1 << 10)->Arg(16 << 10)->Arg(256 << 10);
BENCHMARK(ExtractPython)->Arg(1 << 10)->Arg(16 << 10)->Arg(256 << 10);

}  // namespace
}  // namespace ferret

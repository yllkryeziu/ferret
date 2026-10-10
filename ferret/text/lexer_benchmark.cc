// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/lexer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"

#include "ferret/text/language.h"

namespace ferret {
namespace {

struct ChunkMatches {
  std::string content;
  std::vector<size_t> offsets;
};

ChunkMatches MakeChunkMatches() {
  ChunkMatches chunk;
  constexpr absl::string_view kLine =
      "int variable = function(variable); // comment\n";
  while (chunk.content.size() + kLine.size() <= 8192) {
    chunk.offsets.push_back(chunk.content.size() + 4);
    chunk.content += kLine;
  }
  return chunk;
}

void BM_ClassifyEachLineFromStart(benchmark::State& state) {
  const ChunkMatches chunk = MakeChunkMatches();
  for (auto _ : state) {
    for (const size_t begin : chunk.offsets) {
      benchmark::DoNotOptimize(ClassifyMatch(
          Language::kC, chunk.content, LexState::kCode, begin, begin + 8));
    }
  }
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations() * chunk.content.size()));
}
BENCHMARK(BM_ClassifyEachLineFromStart);

void BM_ClassifyEachLineIncrementally(benchmark::State& state) {
  const ChunkMatches chunk = MakeChunkMatches();
  for (auto _ : state) {
    MatchClassifier classifier(Language::kC, chunk.content, LexState::kCode);
    for (const size_t begin : chunk.offsets) {
      benchmark::DoNotOptimize(classifier.Classify(begin, begin + 8));
    }
  }
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations() * chunk.content.size()));
}
BENCHMARK(BM_ClassifyEachLineIncrementally);

}  // namespace
}  // namespace ferret

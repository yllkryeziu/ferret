// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Compares three ways to compute a chunk's set of distinct indexable trigrams:
// TrigramExtractor's bitmap with a touched list, sort and unique, and
// absl::flat_hash_set. All three fold first and skip windows that contain '\n',
// so they produce the same set.

#include "ferret/text/trigram_extractor.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "benchmark/benchmark.h"

#include "ferret/text/fold.h"
#include "ferret/text/testing/synthetic_source.h"
#include "ferret/text/trigram.h"

namespace ferret {
namespace {

void BM_BitmapExtractor(benchmark::State& state) {
  const std::string text =
      ferret_testing::SyntheticSource(state.range(0), /*seed=*/1);
  TrigramExtractor extractor;
  std::vector<Trigram> trigrams;
  for (auto _ : state) {
    trigrams.clear();
    extractor.Extract(text, trigrams);
    benchmark::DoNotOptimize(trigrams.data());
  }
  state.SetBytesProcessed(state.iterations() * state.range(0));
}

void BM_SortUnique(benchmark::State& state) {
  const std::string text =
      ferret_testing::SyntheticSource(state.range(0), /*seed=*/1);
  std::string folded;
  std::vector<Trigram> trigrams;
  for (auto _ : state) {
    folded.clear();
    AppendFolded(text, folded);
    trigrams.clear();
    for (size_t i = 0; i + 3 <= folded.size(); ++i) {
      const Trigram trigram = TrigramAt(folded, i);
      if (!TrigramHasNewline(trigram)) trigrams.push_back(trigram);
    }
    std::sort(trigrams.begin(), trigrams.end());
    trigrams.erase(std::unique(trigrams.begin(), trigrams.end()),
                   trigrams.end());
    benchmark::DoNotOptimize(trigrams.data());
  }
  state.SetBytesProcessed(state.iterations() * state.range(0));
}

void BM_FlatHashSet(benchmark::State& state) {
  const std::string text =
      ferret_testing::SyntheticSource(state.range(0), /*seed=*/1);
  std::string folded;
  absl::flat_hash_set<Trigram> trigrams;
  for (auto _ : state) {
    folded.clear();
    AppendFolded(text, folded);
    trigrams.clear();
    for (size_t i = 0; i + 3 <= folded.size(); ++i) {
      const Trigram trigram = TrigramAt(folded, i);
      if (!TrigramHasNewline(trigram)) trigrams.insert(trigram);
    }
    benchmark::DoNotOptimize(trigrams.size());
  }
  state.SetBytesProcessed(state.iterations() * state.range(0));
}

// A small chunk, the default chunk size (8 KiB), and a large file.
BENCHMARK(BM_BitmapExtractor)->Arg(1 << 10)->Arg(8 << 10)->Arg(256 << 10);
BENCHMARK(BM_SortUnique)->Arg(1 << 10)->Arg(8 << 10)->Arg(256 << 10);
BENCHMARK(BM_FlatHashSet)->Arg(1 << 10)->Arg(8 << 10)->Arg(256 << 10);

}  // namespace
}  // namespace ferret

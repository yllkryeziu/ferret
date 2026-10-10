// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Measures ingestion of a 10 KiB source file, including trigram extraction.

#include "ferret/index/memtable.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "benchmark/benchmark.h"

#include "ferret/git/blob_hash.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/segment_writer.h"

namespace ferret {
namespace {

void BM_MemtableAdd10KiB(benchmark::State& state) {
  std::string content;
  while (content.size() < size_t{10} * 1024) {
    content.append("int example_value = get_result(42); // source line\n");
  }
  const SegmentDocument doc = {
      .hint_path = "src/example.cc",
      .content = content,
      .rank_key = ComputeRankKey("src/example.cc", content.size(), false,
                                 HashBlob(content))};
  auto mem = std::make_unique<Memtable>();
  for (auto _ : state) {
    if (mem->published_docs() == 700) {
      state.PauseTiming();
      mem = std::make_unique<Memtable>();
      state.ResumeTiming();
    }
    benchmark::DoNotOptimize(mem->Add(doc));
  }
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations() * content.size()));
}
BENCHMARK(BM_MemtableAdd10KiB);

}  // namespace
}  // namespace ferret

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Compares merging four mapped segments against re-extracting their contents.

#include "ferret/index/compactor.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "benchmark/benchmark.h"

#include "ferret/base/env.h"
#include "ferret/git/blob_hash.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/segment.h"
#include "ferret/index/segment_writer.h"
#include "ferret/postings/ids.h"
#include "ferret/testing/temp_dir.h"

namespace ferret {
namespace {

struct BenchmarkCorpus {
  ferret_testing::TempDir dir;
  std::vector<std::string> paths;
  std::vector<std::string> content;
  std::vector<std::unique_ptr<Segment>> opened;
  std::vector<const Segment*> inputs;
  std::vector<size_t> order;

  BenchmarkCorpus() {
    std::string source;
    for (int line = 0; line < 240; ++line) {
      source +=
          "int function_name(int input) { return input * 13 + 7; } // code\n";
    }
    for (int i = 0; i < 1600; ++i) {
      paths.push_back("src/module/file" + std::to_string(i) + ".cc");
      content.push_back(source + "// unique " + std::to_string(i) + "\n");
    }
    std::vector<std::vector<size_t>> partitions(4);
    for (size_t i = 0; i < paths.size(); ++i) {
      order.push_back(i);
      partitions[i % 4].push_back(i);
    }
    const auto less = [&](size_t a, size_t b) {
      return Document(a).rank_key < Document(b).rank_key;
    };
    std::sort(order.begin(), order.end(), less);
    for (int i = 0; i < 4; ++i) {
      std::sort(partitions[i].begin(), partitions[i].end(), less);
      const std::string path = dir.Join("input" + std::to_string(i));
      auto writer = SegmentWriter::Create(*Env::Default(), path);
      CHECK(writer.ok());
      for (size_t id : partitions[i]) CHECK((*writer)->Add(Document(id)).ok());
      CHECK((*writer)->Finish().ok());
      auto segment = Segment::Open(*Env::Default(), path);
      CHECK(segment.ok());
      inputs.push_back(segment->get());
      opened.push_back(std::move(*segment));
    }
  }

  SegmentDocument Document(size_t i) const {
    return {.hint_path = paths[i],
            .content = content[i],
            .rank_key = ComputeRankKey(paths[i], content[i].size(), false,
                                       HashBlob(content[i]))};
  }
};

void BM_Compactor(benchmark::State& state) {
  BenchmarkCorpus corpus;
  for (auto _ : state) {
    auto result = MergeSegments(
        *Env::Default(), corpus.inputs, [](size_t, DocId) { return true; },
        corpus.dir.Join("merged.fseg"));
    CHECK(result.ok());
    benchmark::DoNotOptimize(result->file_bytes);
  }
  state.SetBytesProcessed(state.iterations() * 24 * 1024 * 1024);
}

void BM_Rebuild(benchmark::State& state) {
  BenchmarkCorpus corpus;
  for (auto _ : state) {
    auto writer =
        SegmentWriter::Create(*Env::Default(), corpus.dir.Join("rebuilt.fseg"));
    CHECK(writer.ok());
    for (size_t id : corpus.order) {
      CHECK((*writer)->Add(corpus.Document(id)).ok());
    }
    auto result = (*writer)->Finish();
    CHECK(result.ok());
    benchmark::DoNotOptimize(result->file_bytes);
  }
  state.SetBytesProcessed(state.iterations() * 24 * 1024 * 1024);
}

BENCHMARK(BM_Compactor)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Rebuild)->Unit(benchmark::kMillisecond);

}  // namespace
}  // namespace ferret

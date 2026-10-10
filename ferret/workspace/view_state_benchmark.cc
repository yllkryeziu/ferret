// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Run with bazel run -c opt //ferret/workspace:view_state_benchmark.

#include "ferret/workspace/view_state.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "ferret/index/blob_registry.h"
#include "ferret/workspace/path_table.h"

namespace ferret {
namespace {

void Measure() {
  constexpr size_t kFiles = 500000;
  constexpr size_t kBlobs = 15000;
  ViewWriter writer;
  std::vector<ViewEdit> initial;
  initial.reserve(kFiles);
  for (size_t i = 0; i < kFiles; ++i) {
    initial.push_back(
        {static_cast<PathId>(i), static_cast<BlobId>(i % kBlobs)});
  }
  auto snapshot = writer.ApplyBatch(initial);
  for (size_t batch_size : {size_t{1}, size_t{16}, size_t{256}}) {
    std::vector<double> samples;
    for (size_t trial = 0; trial < 400; ++trial) {
      std::vector<ViewEdit> edits;
      edits.reserve(batch_size);
      for (size_t j = 0; j < batch_size; ++j) {
        const size_t path = (trial * 257 + j) % kFiles;
        edits.push_back({static_cast<PathId>(path),
                         static_cast<BlobId>((trial + j) % kBlobs)});
      }
      const absl::Time start = absl::Now();
      snapshot = writer.ApplyBatch(edits);
      samples.push_back(absl::ToDoubleMicroseconds(absl::Now() - start));
    }
    std::sort(samples.begin(), samples.end());
    CHECK(snapshot->Contains(BlobId{0}));
    std::printf("batch=%zu p50_us=%.1f p99_us=%.1f snapshot_bytes=%zu\n",
                batch_size, samples[samples.size() / 2],
                samples[samples.size() * 99 / 100], snapshot->memory_bytes());
  }
}

}  // namespace
}  // namespace ferret

int main() {
  absl::InitializeLog();
  ferret::Measure();
  return 0;
}

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// HashBlob throughput. Registration hashes every dirty or unknown file and
// ingest hashes every changed one, so SHA-1 speed bounds both.

#include "ferret/git/blob_hash.h"

#include <cstddef>
#include <string>

#include "benchmark/benchmark.h"

#include "ferret/git/oid.h"

namespace ferret {
namespace {

void BM_HashBlob(benchmark::State& state) {
  const std::string content(static_cast<size_t>(state.range(0)), 'x');
  for (auto _ : state) {
    Oid oid = HashBlob(content);
    benchmark::DoNotOptimize(oid);
  }
  state.SetBytesProcessed(state.iterations() * state.range(0));
}
// From a typical source file to the 1 MiB eligibility limit.
BENCHMARK(BM_HashBlob)->Arg(4 << 10)->Arg(64 << 10)->Arg(1 << 20);

}  // namespace
}  // namespace ferret

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Decode throughput of the block unpackers at every bit width: kD1 against
// kD4, scalar against NEON. Items are decoded integers. Blocks are chained,
// each starting from the last value of the one before, as they are in a
// posting list, and the working set stays within L1 so that the kernels, not
// memory, are measured.

#include "ferret/postings/bitpack.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "benchmark/benchmark.h"

namespace ferret {
namespace {

constexpr size_t kBlocks = 128;

using UnpackFn = decltype(&UnpackDeltaBlockScalar);

void BenchmarkUnpack(benchmark::State& state, DeltaCoding coding,
                     UnpackFn unpack) {
  const auto bit_width = static_cast<int>(state.range(0));
  const auto block_bytes = static_cast<size_t>(PackedBlockBytes(bit_width));
  // Unpacking costs the same whatever the bits are.
  std::vector<uint8_t> packed(kBlocks * block_bytes);
  std::mt19937_64 rng(1);
  for (uint8_t& byte : packed) byte = static_cast<uint8_t>(rng());
  std::vector<uint32_t> out(kBlocks * kBitPackBlockSize);
  for (auto _ : state) {
    uint32_t base = 0;
    for (size_t block = 0; block < kBlocks; ++block) {
      uint32_t* block_out = out.data() + block * kBitPackBlockSize;
      unpack(coding, base, packed.data() + block * block_bytes, bit_width,
             block_out);
      base = block_out[kBitPackBlockSize - 1];
    }
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(kBlocks * kBitPackBlockSize));
}

void BM_UnpackScalarD1(benchmark::State& state) {
  BenchmarkUnpack(state, DeltaCoding::kD1, &UnpackDeltaBlockScalar);
}
void BM_UnpackScalarD4(benchmark::State& state) {
  BenchmarkUnpack(state, DeltaCoding::kD4, &UnpackDeltaBlockScalar);
}
BENCHMARK(BM_UnpackScalarD1)->DenseRange(0, kMaxBitPackWidth);
BENCHMARK(BM_UnpackScalarD4)->DenseRange(0, kMaxBitPackWidth);

#if defined(__aarch64__)
void BM_UnpackNeonD1(benchmark::State& state) {
  BenchmarkUnpack(state, DeltaCoding::kD1, &UnpackDeltaBlockNeon);
}
void BM_UnpackNeonD4(benchmark::State& state) {
  BenchmarkUnpack(state, DeltaCoding::kD4, &UnpackDeltaBlockNeon);
}
BENCHMARK(BM_UnpackNeonD1)->DenseRange(0, kMaxBitPackWidth);
BENCHMARK(BM_UnpackNeonD4)->DenseRange(0, kMaxBitPackWidth);
#endif  // defined(__aarch64__)

}  // namespace
}  // namespace ferret

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// CompressStore: the lane-compaction step shared by the NEON kernels.
//
// NEON has no compress-store instruction (AVX-512's VPCOMPRESSD). The
// standard substitute turns a lane mask into a 4-bit index and looks up a
// byte shuffle for TBL that moves the selected 32-bit lanes to the front
// (Lemire, Boytsov and Kurz, SPE 2016). The table is 256 bytes and is built
// at compile time.
//
// This header uses A64 intrinsics, so only the *_neon.cc files include it,
// and the BUILD file compiles those only on AArch64.

#ifndef FERRET_SIMD_INTERNAL_COMPRESS_STORE_H_
#define FERRET_SIMD_INTERNAL_COMPRESS_STORE_H_

#include <arm_neon.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

#include "absl/base/nullability.h"

namespace ferret::simd_internal {

using CompressShuffles = std::array<std::array<uint8_t, 16>, 16>;

// Row m is the TBL byte shuffle that packs the 32-bit lanes selected by the
// bits of m to the front, in lane order. The unused tail indexes byte 0xFF,
// which TBL turns into zero.
constexpr CompressShuffles MakeCompressShuffles() {
  CompressShuffles shuffles = {};
  for (size_t mask = 0; mask < 16; ++mask) {
    std::array<uint8_t, 16>& shuffle = shuffles[mask];
    shuffle.fill(0xFF);
    size_t kept = 0;
    for (size_t lane = 0; lane < 4; ++lane) {
      if ((mask & (size_t{1} << lane)) == 0) continue;
      for (size_t byte = 0; byte < 4; ++byte) {
        shuffle[4 * kept + byte] = static_cast<uint8_t>(4 * lane + byte);
      }
      ++kept;
    }
  }
  return shuffles;
}

inline constexpr CompressShuffles kCompressShuffles = MakeCompressShuffles();

// Stores the lanes of `values` whose `keep` lane is all ones to out[0, n), in
// lane order, and returns n. Each lane of `keep` must be all ones or all
// zeros, as NEON comparisons produce. Always writes out[0, 4).
inline size_t CompressStore(uint32x4_t values, uint32x4_t keep,
                            uint32_t* absl_nonnull out) {
  constexpr uint32x4_t kLaneBits = {1, 2, 4, 8};
  const uint32_t mask = vaddvq_u32(vandq_u32(keep, kLaneBits));
  const uint8x16_t shuffle = vld1q_u8(kCompressShuffles[mask].data());
  vst1q_u32(out, vreinterpretq_u32_u8(
                     vqtbl1q_u8(vreinterpretq_u8_u32(values), shuffle)));
  return static_cast<size_t>(std::popcount(mask));
}

}  // namespace ferret::simd_internal

#endif  // FERRET_SIMD_INTERNAL_COMPRESS_STORE_H_

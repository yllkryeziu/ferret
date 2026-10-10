// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Bit packing of 128-integer blocks in the 4-lane vertical layout of
// SIMD-BP128, with the delta coding fused into unpacking.
//
// A block is 128 values x[0..128) that follow a `base` value. It is stored as
// deltas d[i] of `bit_width` bits each (0 <= bit_width <= 32). Value i lives
// in lane i % 4 as that lane's (i / 4)-th value; each lane's 32 values are
// concatenated least significant bit first into bit_width 32-bit words, and
// word j of lane l is stored as little-endian uint32 number 4 * j + l. A
// packed block is thus exactly 16 * bit_width bytes, a whole number of
// 16-byte vectors, and 4 consecutive values come out of one vector with
// shifts and masks (Lemire and Boytsov, "Decoding billions of integers per
// second through vectorization", Software: Practice and Experience 45(1),
// 2015).
//
// Deltas are computed modulo 2^32 in one of two codings:
//   kD1: d[i] = x[i] - x[i-1] - 1, with x[-1] = base. Decoding needs a prefix
//        sum across lanes, done in registers.
//   kD4: d[i] = x[i] - x[i-4] - 4, with x[j-4] = base - 3 + j for j < 4.
//        Decoding is one vector add per 4 values; deltas are about 4x larger.
// Subtracting 1 (4) makes a run of consecutive integers pack at width 0 under
// either coding. Posting lists use kD1, which measured 9% smaller on
// Chromium; kD4 exists for the published size and speed comparison.
//
// Unpacking is the query-time hot path. UnpackDeltaBlockNeon is specialized
// for every bit width and delta coding; UnpackDeltaBlockScalar is the
// reference and the implementation on targets without NEON. Packing runs only
// when segments are written and is scalar.

#ifndef FERRET_POSTINGS_BITPACK_H_
#define FERRET_POSTINGS_BITPACK_H_

#include <bit>
#include <cstdint>

#include "absl/base/nullability.h"
#include "absl/types/span.h"

#include "ferret/base/ablation.h"

namespace ferret {

static_assert(std::endian::native == std::endian::little,
              "packed words are stored little-endian");

inline constexpr int kBitPackBlockSize = 128;
inline constexpr int kMaxBitPackWidth = 32;

enum class DeltaCoding : uint8_t {
  kD1,  // Difference from the previous value, minus 1.
  kD4,  // Difference from the value 4 positions back, minus 4.
};

// Returns the size of a block packed at `bit_width`: 16 * bit_width bytes.
constexpr int PackedBlockBytes(int bit_width) { return 16 * bit_width; }

// Returns the smallest bit width that holds every delta of `values` after
// `base`. Requires values.size() == kBitPackBlockSize.
[[nodiscard]] int DeltaBlockBitWidth(DeltaCoding coding, uint32_t base,
                                     absl::Span<const uint32_t> values);

// Writes the deltas of `values` after `base`, packed at `bit_width`, to
// out[0, PackedBlockBytes(bit_width)). Requires
// values.size() == kBitPackBlockSize and
// DeltaBlockBitWidth(coding, base, values) <= bit_width <= kMaxBitPackWidth.
void PackDeltaBlock(DeltaCoding coding, uint32_t base,
                    absl::Span<const uint32_t> values, int bit_width,
                    uint8_t* absl_nonnull out);

// Reads a block packed at `bit_width` from in[0, PackedBlockBytes(bit_width))
// and writes its kBitPackBlockSize values, reconstructed from `base`, to
// out[0, kBitPackBlockSize). Neither buffer needs any alignment, and nothing
// outside those ranges is accessed. Every input decodes to some output: the
// arithmetic is modulo 2^32. Requires 0 <= bit_width <= kMaxBitPackWidth.
void UnpackDeltaBlockScalar(DeltaCoding coding, uint32_t base,
                            const uint8_t* absl_nonnull in, int bit_width,
                            uint32_t* absl_nonnull out);
#if defined(__aarch64__)
// The same contract and results as UnpackDeltaBlockScalar. AArch64 only,
// where Advanced SIMD is always present.
void UnpackDeltaBlockNeon(DeltaCoding coding, uint32_t base,
                          const uint8_t* absl_nonnull in, int bit_width,
                          uint32_t* absl_nonnull out);
#endif  // defined(__aarch64__)

// The fastest UnpackDeltaBlock* available on this target.
inline void UnpackDeltaBlock(DeltaCoding coding, uint32_t base,
                             const uint8_t* absl_nonnull in, int bit_width,
                             uint32_t* absl_nonnull out) {
#if defined(__aarch64__)
  // The ablation switch (ferret/base/ablation.h) selects the scalar decoder.
  if (AblationOn(Ablation::scalar_decode)) [[unlikely]] {
    ++ThisThreadAblationCounts().scalar_blocks;
    UnpackDeltaBlockScalar(coding, base, in, bit_width, out);
    return;
  }
  UnpackDeltaBlockNeon(coding, base, in, bit_width, out);
#else
  UnpackDeltaBlockScalar(coding, base, in, bit_width, out);
#endif  // defined(__aarch64__)
}

}  // namespace ferret

#endif  // FERRET_POSTINGS_BITPACK_H_

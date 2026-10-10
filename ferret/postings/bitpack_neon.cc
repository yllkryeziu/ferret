// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// NEON unpacking with fused delta decoding, one fully unrolled kernel per
// (bit width, delta coding) pair.
//
// A block is 32 rows of 4 values, one value per lane, so each row is one
// uint32x4_t. With the width W a template parameter, the word index, shift and
// word-crossing of row K are compile-time constants: extracting a row is a
// shift, an optional OR with the next input vector, and a mask, and the input
// is read exactly once, 16 bytes at a time. The delta decoding then runs on
// the row in registers:
//   kD1: an in-register inclusive prefix sum (two shifted adds) plus a carry
//        vector holding the previous row's last value. Only the carry update
//        is on the loop-carried dependency chain.
//   kD4: one add of the previous row.
// The 66 kernels sit in two tables indexed by width.

#include "ferret/postings/bitpack.h"

#include <arm_neon.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/check.h"

namespace ferret {
namespace {

constexpr size_t kRows = kBitPackBlockSize / 4;

uint32x4_t LoadVector(const uint8_t* p) {
  return vreinterpretq_u32_u8(vld1q_u8(p));
}

// Returns row K of a block packed at width W. `word` holds the input vector
// containing the first bit of row K on entry (except at word boundaries,
// where this loads it) and the vector containing its last bit on exit.
template <int W, size_t K>
uint32x4_t ExtractRow(const uint8_t* in, uint32x4_t& word) {
  static_assert(0 < W && W <= kMaxBitPackWidth);
  constexpr size_t kFirstBit = K * W;
  constexpr size_t kWord = kFirstBit / 32;
  constexpr int kShift = kFirstBit % 32;
  if constexpr (kShift == 0) word = LoadVector(in + 16 * kWord);
  uint32x4_t row = word;
  if constexpr (kShift != 0) row = vshrq_n_u32(word, kShift);
  if constexpr (kShift + W > 32) {
    word = LoadVector(in + 16 * (kWord + 1));
    row = vorrq_u32(row, vshlq_n_u32(word, 32 - kShift));
  }
  if constexpr (W < 32) row = vandq_u32(row, vdupq_n_u32((1u << W) - 1));
  return row;
}

// Turns rows of deltas into rows of values and stores them.
template <DeltaCoding D>
class RowDecoder;

template <>
class RowDecoder<DeltaCoding::kD1> {
 public:
  explicit RowDecoder(uint32_t base) : carry_(vdupq_n_u32(base)) {}

  void Store(uint32x4_t deltas, uint32_t* out) {
    const uint32x4_t zero = vdupq_n_u32(0);
    uint32x4_t sum = vaddq_u32(deltas, vdupq_n_u32(1));
    sum = vaddq_u32(sum, vextq_u32(zero, sum, 3));
    sum = vaddq_u32(sum, vextq_u32(zero, sum, 2));
    vst1q_u32(out, vaddq_u32(sum, carry_));
    carry_ = vaddq_u32(carry_, vdupq_laneq_u32(sum, 3));
  }

 private:
  uint32x4_t carry_;  // The last value stored, in every lane.
};

template <>
class RowDecoder<DeltaCoding::kD4> {
 public:
  explicit RowDecoder(uint32_t base) {
    // The virtual row before the block: base - 3, base - 2, base - 1, base.
    static constexpr uint32_t kOffsets[4] = {0xFFFFFFFD, 0xFFFFFFFE, 0xFFFFFFFF,
                                             0};
    previous_ = vaddq_u32(vdupq_n_u32(base), vld1q_u32(kOffsets));
  }

  void Store(uint32x4_t deltas, uint32_t* out) {
    previous_ = vaddq_u32(previous_, vaddq_u32(deltas, vdupq_n_u32(4)));
    vst1q_u32(out, previous_);
  }

 private:
  uint32x4_t previous_;  // The last row stored.
};

template <int W, DeltaCoding D, size_t... K>
void UnpackRows(uint32_t base, const uint8_t* in, uint32_t* out,
                std::index_sequence<K...> /*rows*/) {
  RowDecoder<D> decoder(base);
  uint32x4_t word = vdupq_n_u32(0);
  (decoder.Store(ExtractRow<W, K>(in, word), out + 4 * K), ...);
}

template <int W, DeltaCoding D>
void UnpackBlock(uint32_t base, [[maybe_unused]] const uint8_t* in,
                 uint32_t* out) {
  if constexpr (W == 0) {
    // Every delta is 0 and the payload is empty. Under either coding the
    // block is base + 1, ..., base + 128.
    static constexpr uint32_t kOneToFour[4] = {1, 2, 3, 4};
    uint32x4_t row = vaddq_u32(vdupq_n_u32(base), vld1q_u32(kOneToFour));
    for (size_t k = 0; k < kRows; ++k) {
      vst1q_u32(out + 4 * k, row);
      row = vaddq_u32(row, vdupq_n_u32(4));
    }
  } else {
    UnpackRows<W, D>(base, in, out, std::make_index_sequence<kRows>());
  }
}

using UnpackBlockFn = void (*)(uint32_t base, const uint8_t* in, uint32_t* out);
using UnpackTable = std::array<UnpackBlockFn, kMaxBitPackWidth + 1>;

template <DeltaCoding D, int... W>
constexpr UnpackTable MakeUnpackTable(std::integer_sequence<int, W...>) {
  return {&UnpackBlock<W, D>...};
}

constexpr UnpackTable kUnpackD1 = MakeUnpackTable<DeltaCoding::kD1>(
    std::make_integer_sequence<int, kMaxBitPackWidth + 1>());
constexpr UnpackTable kUnpackD4 = MakeUnpackTable<DeltaCoding::kD4>(
    std::make_integer_sequence<int, kMaxBitPackWidth + 1>());

}  // namespace

void UnpackDeltaBlockNeon(DeltaCoding coding, uint32_t base,
                          const uint8_t* absl_nonnull in, int bit_width,
                          uint32_t* absl_nonnull out) {
  DCHECK(0 <= bit_width && bit_width <= kMaxBitPackWidth) << bit_width;
  const UnpackTable& table = coding == DeltaCoding::kD1 ? kUnpackD1 : kUnpackD4;
  table[static_cast<size_t>(bit_width)](base, in, out);
}

}  // namespace ferret

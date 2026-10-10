// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Scalar bit packing: the only packer, and the reference unpacker that the
// NEON kernels are tested against. Written for clarity over speed.

#include "ferret/postings/bitpack.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/types/span.h"

namespace ferret {
namespace {

constexpr int kLanes = 4;
constexpr int kValuesPerLane = kBitPackBlockSize / kLanes;

// Returns x[i - 4] as kD4 defines it, including the virtual values before the
// block.
uint32_t D4Predecessor(uint32_t base, const uint32_t* values, int i) {
  return i >= kLanes ? values[i - kLanes] : base - 3 + static_cast<uint32_t>(i);
}

void ComputeDeltas(DeltaCoding coding, uint32_t base,
                   absl::Span<const uint32_t> values, uint32_t* deltas) {
  DCHECK_EQ(values.size(), size_t{kBitPackBlockSize});
  const uint32_t* x = values.data();
  switch (coding) {
    case DeltaCoding::kD1:
      for (int i = 0; i < kBitPackBlockSize; ++i) {
        deltas[i] = x[i] - (i == 0 ? base : x[i - 1]) - 1;
      }
      return;
    case DeltaCoding::kD4:
      for (int i = 0; i < kBitPackBlockSize; ++i) {
        deltas[i] = x[i] - D4Predecessor(base, x, i) - 4;
      }
      return;
  }
}

uint32_t LoadWord(const uint8_t* in, int index) {
  uint32_t word;
  std::memcpy(&word, in + sizeof(word) * static_cast<size_t>(index),
              sizeof(word));
  return word;
}

void StoreWord(uint8_t* out, int index, uint32_t word) {
  std::memcpy(out + sizeof(word) * static_cast<size_t>(index), &word,
              sizeof(word));
}

}  // namespace

int DeltaBlockBitWidth(DeltaCoding coding, uint32_t base,
                       absl::Span<const uint32_t> values) {
  uint32_t deltas[kBitPackBlockSize];
  ComputeDeltas(coding, base, values, deltas);
  uint32_t any_bits = 0;
  for (const uint32_t delta : deltas) any_bits |= delta;
  return std::bit_width(any_bits);
}

void PackDeltaBlock(DeltaCoding coding, uint32_t base,
                    absl::Span<const uint32_t> values, int bit_width,
                    uint8_t* absl_nonnull out) {
  DCHECK_LE(DeltaBlockBitWidth(coding, base, values), bit_width);
  DCHECK_LE(bit_width, kMaxBitPackWidth);
  uint32_t deltas[kBitPackBlockSize];
  ComputeDeltas(coding, base, values, deltas);
  for (int lane = 0; lane < kLanes; ++lane) {
    // Bits not yet stored, least significant first.
    uint64_t pending = 0;
    int pending_bits = 0;
    int word = 0;
    for (int row = 0; row < kValuesPerLane; ++row) {
      pending |= uint64_t{deltas[kLanes * row + lane]} << pending_bits;
      pending_bits += bit_width;
      if (pending_bits >= 32) {
        StoreWord(out, kLanes * word + lane, static_cast<uint32_t>(pending));
        ++word;
        pending >>= 32;
        pending_bits -= 32;
      }
    }
    DCHECK_EQ(pending_bits, 0);
  }
}

void UnpackDeltaBlockScalar(DeltaCoding coding, uint32_t base,
                            const uint8_t* absl_nonnull in, int bit_width,
                            uint32_t* absl_nonnull out) {
  DCHECK(0 <= bit_width && bit_width <= kMaxBitPackWidth) << bit_width;
  const uint64_t mask = (uint64_t{1} << bit_width) - 1;
  uint32_t deltas[kBitPackBlockSize];
  for (int lane = 0; lane < kLanes; ++lane) {
    uint64_t pending = 0;
    int pending_bits = 0;
    int word = 0;
    for (int row = 0; row < kValuesPerLane; ++row) {
      if (pending_bits < bit_width) {
        pending |= uint64_t{LoadWord(in, kLanes * word + lane)} << pending_bits;
        ++word;
        pending_bits += 32;
      }
      deltas[kLanes * row + lane] = static_cast<uint32_t>(pending & mask);
      pending >>= bit_width;
      pending_bits -= bit_width;
    }
  }
  switch (coding) {
    case DeltaCoding::kD1: {
      uint32_t previous = base;
      for (int i = 0; i < kBitPackBlockSize; ++i) {
        previous += deltas[i] + 1;
        out[i] = previous;
      }
      return;
    }
    case DeltaCoding::kD4:
      for (int i = 0; i < kBitPackBlockSize; ++i) {
        out[i] = D4Predecessor(base, out, i) + deltas[i] + 4;
      }
      return;
  }
}

}  // namespace ferret

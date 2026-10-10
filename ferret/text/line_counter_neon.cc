// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// vceqq_u8 sets a lane to 0xFF, which is -1 as a byte, where the input is
// '\n', so subtracting the comparison from an accumulator adds one per match.
// Four accumulators keep the loop free of a serial dependency. A byte lane
// overflows after 255 matches, so the accumulators are widened and drained
// every 255 iterations.

#include <arm_neon.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "absl/strings/string_view.h"

#include "ferret/text/internal/line_counter_kernels.h"

namespace ferret::text_internal {

int64_t CountNewlinesNeon(absl::string_view text) {
  constexpr size_t kBlockBytes = 64;
  constexpr size_t kMaxBlocksPerDrain = 255;
  const uint8_t* p = reinterpret_cast<const uint8_t*>(text.data());
  size_t remaining = text.size();
  const uint8x16_t newline = vdupq_n_u8('\n');
  int64_t count = 0;
  while (remaining >= kBlockBytes) {
    const size_t blocks = std::min(remaining / kBlockBytes, kMaxBlocksPerDrain);
    uint8x16_t acc0 = vdupq_n_u8(0);
    uint8x16_t acc1 = vdupq_n_u8(0);
    uint8x16_t acc2 = vdupq_n_u8(0);
    uint8x16_t acc3 = vdupq_n_u8(0);
    for (size_t i = 0; i < blocks; ++i, p += kBlockBytes) {
      acc0 = vsubq_u8(acc0, vceqq_u8(vld1q_u8(p), newline));
      acc1 = vsubq_u8(acc1, vceqq_u8(vld1q_u8(p + 16), newline));
      acc2 = vsubq_u8(acc2, vceqq_u8(vld1q_u8(p + 32), newline));
      acc3 = vsubq_u8(acc3, vceqq_u8(vld1q_u8(p + 48), newline));
    }
    remaining -= blocks * kBlockBytes;
    // Each 16-bit lane of a pairwise sum holds at most 2 * 255; four of them
    // still fit.
    const uint16x8_t sum =
        vaddq_u16(vaddq_u16(vpaddlq_u8(acc0), vpaddlq_u8(acc1)),
                  vaddq_u16(vpaddlq_u8(acc2), vpaddlq_u8(acc3)));
    count += vaddlvq_u16(sum);
  }
  for (; remaining >= 16; remaining -= 16, p += 16) {
    const uint8x16_t matches = vceqq_u8(vld1q_u8(p), newline);
    count += vaddvq_u8(vshrq_n_u8(matches, 7));
  }
  for (size_t i = 0; i < remaining; ++i) {
    if (p[i] == '\n') ++count;
  }
  return count;
}

}  // namespace ferret::text_internal

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The newline-counting kernels behind CountNewlines (line_counter.h).
//
// Package-private so that tests and benchmarks can compare the NEON kernel
// with the scalar reference. Both have CountNewlines's contract.

#ifndef FERRET_TEXT_INTERNAL_LINE_COUNTER_KERNELS_H_
#define FERRET_TEXT_INTERNAL_LINE_COUNTER_KERNELS_H_

#include <cstdint>

#include "absl/strings/string_view.h"

namespace ferret::text_internal {

// The portable reference: one byte at a time.
[[nodiscard]] int64_t CountNewlinesScalar(absl::string_view text);

#if defined(__aarch64__)
// Compares 64 bytes per iteration and accumulates matches in byte lanes.
[[nodiscard]] int64_t CountNewlinesNeon(absl::string_view text);
#endif

}  // namespace ferret::text_internal

#endif  // FERRET_TEXT_INTERNAL_LINE_COUNTER_KERNELS_H_

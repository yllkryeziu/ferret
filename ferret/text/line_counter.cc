// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/line_counter.h"

#include <cstddef>
#include <cstdint>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"

#include "ferret/text/internal/line_counter_kernels.h"

namespace ferret {

int64_t CountNewlines(absl::string_view text) {
#if defined(__aarch64__)
  return text_internal::CountNewlinesNeon(text);
#else
  return text_internal::CountNewlinesScalar(text);
#endif
}

size_t LineBegin(absl::string_view text, size_t pos) {
  DCHECK_LE(pos, text.size());
  const size_t newline = text.substr(0, pos).rfind('\n');
  return newline == absl::string_view::npos ? 0 : newline + 1;
}

size_t LineEnd(absl::string_view text, size_t pos) {
  DCHECK_LE(pos, text.size());
  const size_t newline = text.find('\n', pos);
  return newline == absl::string_view::npos ? text.size() : newline;
}

}  // namespace ferret

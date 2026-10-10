// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/fold.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"

namespace ferret {

void AppendFolded(absl::string_view in, std::string& out) {
  // The output is never longer than the input, so write through a pointer
  // into a buffer of the input's size instead of growing byte by byte.
  const size_t n = in.size();
  const size_t base = out.size();
  out.resize(base + n);
  char* dst = out.data() + base;
  size_t i = 0;
  while (i < n) {
    const uint8_t b = static_cast<uint8_t>(in[i]);
    if (b == 0xE2 && i + 2 < n && static_cast<uint8_t>(in[i + 1]) == 0x84 &&
        static_cast<uint8_t>(in[i + 2]) == 0xAA) {
      *dst++ = 'k';  // U+212A KELVIN SIGN.
      i += 3;
    } else if (b == 0xC5 && i + 1 < n &&
               static_cast<uint8_t>(in[i + 1]) == 0xBF) {
      *dst++ = 's';  // U+017F LATIN SMALL LETTER LONG S.
      i += 2;
    } else {
      *dst++ = static_cast<char>(FoldAsciiByte(b));
      ++i;
    }
  }
  out.resize(static_cast<size_t>(dst - out.data()));
}

std::string Fold(absl::string_view in) {
  std::string out;
  AppendFolded(in, out);
  return out;
}

}  // namespace ferret

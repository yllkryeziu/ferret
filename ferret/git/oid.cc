// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/oid.h"

#include <cstdint>
#include <cstring>
#include <string>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace ferret {
namespace {

int HexValue(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

}  // namespace

absl::StatusOr<Oid> Oid::FromHex(absl::string_view hex) {
  if (hex.size() != 2 * kSize) {
    return absl::InvalidArgumentError(
        absl::StrCat("oid: expected 40 hex digits, got ", hex.size(),
                     " characters: \"", absl::CHexEscape(hex), "\""));
  }
  Bytes bytes;
  for (size_t i = 0; i < kSize; ++i) {
    const int hi = HexValue(hex[2 * i]);
    const int lo = HexValue(hex[2 * i + 1]);
    if (hi < 0 || lo < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "oid: invalid hex digit in \"", absl::CHexEscape(hex), "\""));
    }
    bytes[i] = static_cast<uint8_t>((hi << 4) | lo);
  }
  return Oid(bytes);
}

Oid Oid::FromRaw(absl::string_view raw) {
  DCHECK_EQ(raw.size(), kSize);
  Bytes bytes;
  std::memcpy(bytes.data(), raw.data(), kSize);
  return Oid(bytes);
}

std::string Oid::ToHex() const { return absl::BytesToHexString(raw()); }

}  // namespace ferret

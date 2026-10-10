// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/postings/varint.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"

namespace ferret {
namespace {

template <typename T>
void AppendVarint(T value, std::string& out) {
  while (value >= 0x80) {
    out.push_back(static_cast<char>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<char>(value));
}

template <typename T>
std::optional<T> ConsumeVarint(absl::string_view& input) {
  constexpr int kBits = std::numeric_limits<T>::digits;
  constexpr size_t kMaxBytes = (kBits + 6) / 7;
  const size_t available = input.size() < kMaxBytes ? input.size() : kMaxBytes;
  T value = 0;
  for (size_t i = 0; i < available; ++i) {
    const auto byte = static_cast<uint8_t>(input[i]);
    const int shift = static_cast<int>(7 * i);
    // In the last possible byte, every bit at or above the type's width,
    // including the continuation bit, must be clear.
    if (i == kMaxBytes - 1 && (byte >> (kBits - shift)) != 0) {
      return std::nullopt;
    }
    value |= static_cast<T>(byte & 0x7F) << shift;
    if ((byte & 0x80) == 0) {
      input.remove_prefix(i + 1);
      return value;
    }
  }
  return std::nullopt;
}

}  // namespace

void AppendVarint32(uint32_t value, std::string& out) {
  AppendVarint(value, out);
}

void AppendVarint64(uint64_t value, std::string& out) {
  AppendVarint(value, out);
}

std::optional<uint32_t> ConsumeVarint32(absl::string_view& input) {
  return ConsumeVarint<uint32_t>(input);
}

std::optional<uint64_t> ConsumeVarint64(absl::string_view& input) {
  return ConsumeVarint<uint64_t>(input);
}

}  // namespace ferret

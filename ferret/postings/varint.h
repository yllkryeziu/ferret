// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// LEB128 variable-length integers: 7 value bits per byte, least significant
// group first, high bit set on every byte but the last.
//
// Posting lists store short lists and block tails as varint gaps. Decoding
// reads bytes from disk, so it is bounds-checked and rejects encodings that
// do not fit the target type.

#ifndef FERRET_POSTINGS_VARINT_H_
#define FERRET_POSTINGS_VARINT_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"

namespace ferret {

inline constexpr int kMaxVarint32Bytes = 5;
inline constexpr int kMaxVarint64Bytes = 10;

// Appends the shortest encoding of `value` to `out`.
void AppendVarint32(uint32_t value, std::string& out);
void AppendVarint64(uint64_t value, std::string& out);

// Decodes the varint at the front of `input` and removes it from `input`.
// Returns nullopt, leaving `input` unchanged, if `input` ends inside the
// varint or the varint does not fit the result type: longer than
// kMaxVarint32Bytes (kMaxVarint64Bytes), or with value bits beyond bit 31
// (63). Redundant trailing zero groups, as in 0x80 0x00, are accepted, as in
// protobuf.
std::optional<uint32_t> ConsumeVarint32(absl::string_view& input);
std::optional<uint64_t> ConsumeVarint64(absl::string_view& input);

}  // namespace ferret

#endif  // FERRET_POSTINGS_VARINT_H_

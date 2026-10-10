// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Oid: the 20-byte content identifier that keys the index.
//
// An Oid is SHA-1("blob " + decimal(len) + '\0' + bytes) over the exact
// worktree bytes Ferret indexed. It equals git's blob object id whenever
// checkout conversion (CRLF, filters, ident) is the identity, which is what
// lets the index trust `git ls-files -s` as a skip hint.
// Content addressing is what makes the index append-only and shareable across
// worktrees.

#ifndef FERRET_GIT_OID_H_
#define FERRET_GIT_OID_H_

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace ferret {

// A regular, trivially copyable value type. Thread-compatible.
class Oid final {
 public:
  static constexpr size_t kSize = 20;
  using Bytes = std::array<uint8_t, kSize>;

  // All zeros (git's null object id).
  constexpr Oid() = default;
  explicit constexpr Oid(const Bytes& bytes) : bytes_(bytes) {}

  Oid(const Oid&) = default;
  Oid& operator=(const Oid&) = default;

  // Parses exactly 40 hex digits (either case). InvalidArgument otherwise.
  static absl::StatusOr<Oid> FromHex(absl::string_view hex);

  // Copies kSize raw bytes, e.g. from an on-disk record.
  // Requires raw.size() == kSize (DCHECKed).
  static Oid FromRaw(absl::string_view raw);

  const Bytes& bytes() const ABSL_ATTRIBUTE_LIFETIME_BOUND { return bytes_; }

  // The bytes as a string_view, valid while *this is alive.
  absl::string_view raw() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return absl::string_view(reinterpret_cast<const char*>(bytes_.data()),
                             kSize);
  }

  bool is_null() const { return *this == Oid(); }

  // Lowercase, 40 characters.
  std::string ToHex() const;

  friend bool operator==(const Oid&, const Oid&) = default;
  friend std::strong_ordering operator<=>(const Oid&, const Oid&) = default;

  template <typename H>
  friend H AbslHashValue(H h, const Oid& oid) {
    return H::combine_contiguous(std::move(h), oid.bytes_.data(), kSize);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Oid& oid) {
    sink.Append(oid.ToHex());
  }

 private:
  Bytes bytes_ = {};
};

static_assert(sizeof(Oid) == Oid::kSize);
static_assert(std::is_trivially_copyable_v<Oid>);

}  // namespace ferret

#endif  // FERRET_GIT_OID_H_

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The one layer through which Ferret parses, reads, and writes JSON.
//
// nlohmann::json throws on malformed input, on a typed read of the wrong
// type, and when it serializes invalid UTF-8. Ferret builds without
// exceptions, so each of those is std::abort() in a daemon that reads input
// from agents. This file provides the three safe operations:
//
//   * ParseJson rejects malformed, oversized, and over-nested input with a
//     Status instead of throwing.
//   * JsonView reads typed values and returns InvalidArgument naming the
//     offending field by its JSON Pointer (RFC 6901), such as
//     "/params/arguments/query: expected string, got integer".
//   * SerializeJson replaces invalid UTF-8 with U+FFFD and always produces a
//     single line, which is what newline-delimited framing requires.
//
// Every target that depends on this one is compiled with
// JSON_USE_IMPLICIT_CONVERSIONS=0, so a json value never converts to a C++
// type behind the reader's back.

#ifndef FERRET_RPC_JSON_CODEC_H_
#define FERRET_RPC_JSON_CODEC_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "nlohmann/json.hpp"

namespace ferret {

struct JsonParseOptions {
  // Longest accepted document. Longer input is rejected without being parsed.
  int64_t max_bytes = int64_t{16} << 20;
  // Deepest accepted nesting of arrays and objects; "[]" has depth 1. The
  // parser itself is iterative, but copying, comparing, and serializing a
  // json value recurse once per level, and a thread stack (512 KiB on macOS)
  // must survive whatever an untrusted peer sends.
  int max_depth = 64;
};

// Parses `text` as exactly one JSON value (RFC 8259; no comments, no trailing
// content, strings must be valid UTF-8). Returns InvalidArgument if the text
// is longer than options.max_bytes, nests deeper than options.max_depth, or
// is not JSON; a syntax error names the byte offset where parsing failed.
// O(text.size()). Thread-safe.
absl::StatusOr<nlohmann::json> ParseJson(absl::string_view text,
                                         const JsonParseOptions& options = {});

// Serializes `value` compactly. Invalid UTF-8 in strings and keys becomes
// U+FFFD, and control characters are escaped, so the result is valid UTF-8
// and never contains a raw '\n' or '\r'. Thread-safe.
std::string SerializeJson(const nlohmann::json& value);

// A read-only position in a parsed JSON document: a value plus the JSON
// Pointer at which it sits, so that every checked read can name the field it
// rejects. Paths of members are built from the keys the caller asks for,
// never from keys in the input.
//
// Reads never abort: a type mismatch, an absent required member, or an
// integer out of range returns InvalidArgument. Optional reads treat an
// explicit null like an absent member, as the protobuf JSON mapping does,
// because JSON clients commonly send null for "not set".
//
// A JsonView borrows the document, which must outlive it. Thread-compatible.
class JsonView final {
 public:
  // Views `value` as the root of its document.
  explicit JsonView(const nlohmann::json& value ABSL_ATTRIBUTE_LIFETIME_BOUND)
      : value_(&value) {}
  // Views `value` as if it sat at `path` (a JSON Pointer) in an enclosing
  // document, so that errors about a message's params can say "/params/...".
  JsonView(const nlohmann::json& value ABSL_ATTRIBUTE_LIFETIME_BOUND,
           std::string path)
      : value_(&value), path_(std::move(path)) {}

  JsonView(const JsonView&) = default;
  JsonView& operator=(const JsonView&) = default;
  JsonView(JsonView&&) = default;
  JsonView& operator=(JsonView&&) = default;

  // The viewed value. Valid while the document is alive.
  const nlohmann::json& value() const { return *value_; }
  // The JSON Pointer of the viewed value; empty for the document root.
  const std::string& path() const { return path_; }

  // Typed reads of the viewed value itself. Integers must be JSON integers
  // (1.0 is rejected) that fit the requested type.
  absl::StatusOr<std::string> AsString() const;
  absl::StatusOr<int64_t> AsInt64() const;
  absl::StatusOr<uint64_t> AsUint64() const;
  absl::StatusOr<bool> AsBool() const;
  // Return *this if the viewed value is an object (or an array).
  absl::StatusOr<JsonView> AsObject() const;
  absl::StatusOr<JsonView> AsArray() const;

  // Returns the InvalidArgument status that the typed reads return when the
  // viewed value is not what the caller `expected`, for callers that accept
  // a union of types: TypeError("integer or string").
  absl::Status TypeError(absl::string_view expected) const;

  // Returns member `key` of any type, including null, or nullopt if it is
  // absent or the viewed value is not an object.
  std::optional<JsonView> Find(absl::string_view key) const;

  // Required members. Each fails if the viewed value is not an object, if
  // the member is absent or null, or if it has the wrong type.
  absl::StatusOr<std::string> GetString(absl::string_view key) const;
  absl::StatusOr<int64_t> GetInt64(absl::string_view key) const;
  absl::StatusOr<uint64_t> GetUint64(absl::string_view key) const;
  absl::StatusOr<bool> GetBool(absl::string_view key) const;
  absl::StatusOr<JsonView> GetObject(absl::string_view key) const;
  absl::StatusOr<JsonView> GetArray(absl::string_view key) const;

  // Optional members: nullopt if the member is absent or null. Each fails if
  // the viewed value is not an object or the member has the wrong type.
  absl::StatusOr<std::optional<std::string>> GetOptionalString(
      absl::string_view key) const;
  absl::StatusOr<std::optional<int64_t>> GetOptionalInt64(
      absl::string_view key) const;
  absl::StatusOr<std::optional<uint64_t>> GetOptionalUint64(
      absl::string_view key) const;
  absl::StatusOr<std::optional<bool>> GetOptionalBool(
      absl::string_view key) const;
  absl::StatusOr<std::optional<JsonView>> GetOptionalObject(
      absl::string_view key) const;
  absl::StatusOr<std::optional<JsonView>> GetOptionalArray(
      absl::string_view key) const;

  // The number of elements if the viewed value is an array, and 0 otherwise.
  size_t size() const { return value_->is_array() ? value_->size() : 0; }
  // Element `index` of an array. Requires value().is_array() and
  // index < size().
  JsonView Element(size_t index) const;

 private:
  // The member `key` for a required read: absent and null are errors.
  absl::StatusOr<JsonView> RequiredMember(absl::string_view key) const;
  // The member `key` for an optional read: absent and null are nullopt.
  absl::StatusOr<std::optional<JsonView>> OptionalMember(
      absl::string_view key) const;

  template <typename T>
  absl::StatusOr<T> GetRequired(absl::string_view key,
                                absl::StatusOr<T> (JsonView::*read)()
                                    const) const;
  template <typename T>
  absl::StatusOr<std::optional<T>> GetOptional(
      absl::string_view key, absl::StatusOr<T> (JsonView::*read)() const) const;

  std::string ChildPath(absl::string_view key) const;

  const nlohmann::json* value_;
  std::string path_;
};

}  // namespace ferret

#endif  // FERRET_RPC_JSON_CODEC_H_

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/rpc/json_codec.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"

namespace ferret {
namespace {

using Json = nlohmann::json;

// How much of the offending token a syntax error quotes. A token can be as
// long as the whole document, and the message goes back to the peer.
constexpr size_t kMaxQuotedTokenBytes = 32;

// Checks syntax and nesting depth without building a DOM. nlohmann's DOM
// parser has no depth limit and, without exceptions, cannot say where
// parsing failed; this SAX pass supplies both, at the cost of lexing the
// input twice.
class JsonValidator final : public nlohmann::json_sax<Json> {
 public:
  explicit JsonValidator(int max_depth) : max_depth_(max_depth) {}

  bool null() override { return true; }
  bool boolean(bool) override { return true; }
  bool number_integer(number_integer_t) override { return true; }
  bool number_unsigned(number_unsigned_t) override { return true; }
  bool number_float(number_float_t, const string_t&) override { return true; }
  bool string(string_t&) override { return true; }
  bool binary(binary_t&) override { return true; }
  bool key(string_t&) override { return true; }
  bool start_object(size_t) override { return Enter(); }
  bool end_object() override { return Leave(); }
  bool start_array(size_t) override { return Enter(); }
  bool end_array() override { return Leave(); }

  bool parse_error(size_t position, const std::string& last_token,
                   const Json::exception&) override {
    status_ = absl::InvalidArgumentError(absl::StrCat(
        "invalid json at byte ", position, " near \"",
        absl::CHexEscape(
            absl::string_view(last_token).substr(0, kMaxQuotedTokenBytes)),
        "\""));
    return false;
  }

  const absl::Status& status() const { return status_; }

 private:
  bool Enter() {
    if (++depth_ <= max_depth_) return true;
    status_ = absl::InvalidArgumentError(
        absl::StrCat("json nesting exceeds the limit of ", max_depth_));
    return false;
  }
  bool Leave() {
    --depth_;
    return true;
  }

  const int max_depth_;
  int depth_ = 0;
  absl::Status status_;
};

absl::string_view JsonTypeName(const Json& value) {
  switch (value.type()) {
    case Json::value_t::null:
      return "null";
    case Json::value_t::boolean:
      return "boolean";
    case Json::value_t::number_integer:
    case Json::value_t::number_unsigned:
      return "integer";
    case Json::value_t::number_float:
      return "number";
    case Json::value_t::string:
      return "string";
    case Json::value_t::array:
      return "array";
    case Json::value_t::object:
      return "object";
    case Json::value_t::binary:
      return "binary";
    case Json::value_t::discarded:
      return "discarded";
  }
  return "unknown";
}

// The pointer as it appears in messages: the root's pointer is the empty
// string, which would read as nothing at all.
absl::string_view DisplayPath(absl::string_view path) {
  return path.empty() ? "(root)" : path;
}

}  // namespace

absl::StatusOr<Json> ParseJson(absl::string_view text,
                               const JsonParseOptions& options) {
  if (static_cast<int64_t>(text.size()) > options.max_bytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("json document of ", text.size(),
                     " bytes exceeds the limit of ", options.max_bytes));
  }
  JsonValidator validator(options.max_depth);
  if (!Json::sax_parse(text.begin(), text.end(), &validator)) {
    return validator.status();
  }
  Json value = Json::parse(text.begin(), text.end(), /*cb=*/nullptr,
                           /*allow_exceptions=*/false);
  // The validator accepted the same bytes with the same lexer, so this
  // cannot fail; the check keeps a discarded value, which serializes as the
  // non-JSON "<discarded>", from ever escaping.
  if (value.is_discarded()) {
    return absl::InternalError("json parser rejected validated input");
  }
  return value;
}

std::string SerializeJson(const Json& value) {
  return value.dump(/*indent=*/-1, /*indent_char=*/' ', /*ensure_ascii=*/false,
                    Json::error_handler_t::replace);
}

absl::StatusOr<std::string> JsonView::AsString() const {
  if (!value_->is_string()) return TypeError("string");
  return value_->get_ref<const Json::string_t&>();
}

absl::StatusOr<int64_t> JsonView::AsInt64() const {
  if (value_->is_number_unsigned()) {
    const uint64_t value = value_->get<uint64_t>();
    if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return absl::InvalidArgumentError(
          absl::StrCat(DisplayPath(path_), ": integer ", value,
                       " is out of range for int64"));
    }
    return static_cast<int64_t>(value);
  }
  if (value_->is_number_integer()) return value_->get<int64_t>();
  return TypeError("integer");
}

absl::StatusOr<uint64_t> JsonView::AsUint64() const {
  if (value_->is_number_unsigned()) return value_->get<uint64_t>();
  if (value_->is_number_integer()) {
    const int64_t value = value_->get<int64_t>();
    if (value < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat(DisplayPath(path_), ": integer ", value,
                       " is out of range for uint64"));
    }
    return static_cast<uint64_t>(value);
  }
  return TypeError("integer");
}

absl::StatusOr<bool> JsonView::AsBool() const {
  if (!value_->is_boolean()) return TypeError("boolean");
  return value_->get<bool>();
}

absl::StatusOr<JsonView> JsonView::AsObject() const {
  if (!value_->is_object()) return TypeError("object");
  return *this;
}

absl::StatusOr<JsonView> JsonView::AsArray() const {
  if (!value_->is_array()) return TypeError("array");
  return *this;
}

absl::Status JsonView::TypeError(absl::string_view expected) const {
  return absl::InvalidArgumentError(
      absl::StrCat(DisplayPath(path_), ": expected ", expected, ", got ",
                   JsonTypeName(*value_)));
}

std::optional<JsonView> JsonView::Find(absl::string_view key) const {
  const auto it = value_->find(key);
  if (it == value_->end()) return std::nullopt;
  return JsonView(*it, ChildPath(key));
}

absl::StatusOr<JsonView> JsonView::RequiredMember(absl::string_view key) const {
  ABSL_ASSIGN_OR_RETURN(std::optional<JsonView> member, OptionalMember(key));
  if (!member.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat(DisplayPath(ChildPath(key)), ": missing required member"));
  }
  return *std::move(member);
}

absl::StatusOr<std::optional<JsonView>> JsonView::OptionalMember(
    absl::string_view key) const {
  if (!value_->is_object()) return TypeError("object");
  std::optional<JsonView> member = Find(key);
  if (member.has_value() && member->value().is_null()) return std::nullopt;
  return member;
}

template <typename T>
absl::StatusOr<T> JsonView::GetRequired(absl::string_view key,
                                        absl::StatusOr<T> (JsonView::*read)()
                                            const) const {
  ABSL_ASSIGN_OR_RETURN(const JsonView member, RequiredMember(key));
  return (member.*read)();
}

template <typename T>
absl::StatusOr<std::optional<T>> JsonView::GetOptional(
    absl::string_view key, absl::StatusOr<T> (JsonView::*read)() const) const {
  ABSL_ASSIGN_OR_RETURN(const std::optional<JsonView> member,
                        OptionalMember(key));
  if (!member.has_value()) return std::nullopt;
  ABSL_ASSIGN_OR_RETURN(T value, ((*member).*read)());
  return value;
}

absl::StatusOr<std::string> JsonView::GetString(absl::string_view key) const {
  return GetRequired(key, &JsonView::AsString);
}
absl::StatusOr<int64_t> JsonView::GetInt64(absl::string_view key) const {
  return GetRequired(key, &JsonView::AsInt64);
}
absl::StatusOr<uint64_t> JsonView::GetUint64(absl::string_view key) const {
  return GetRequired(key, &JsonView::AsUint64);
}
absl::StatusOr<bool> JsonView::GetBool(absl::string_view key) const {
  return GetRequired(key, &JsonView::AsBool);
}
absl::StatusOr<JsonView> JsonView::GetObject(absl::string_view key) const {
  return GetRequired(key, &JsonView::AsObject);
}
absl::StatusOr<JsonView> JsonView::GetArray(absl::string_view key) const {
  return GetRequired(key, &JsonView::AsArray);
}

absl::StatusOr<std::optional<std::string>> JsonView::GetOptionalString(
    absl::string_view key) const {
  return GetOptional(key, &JsonView::AsString);
}
absl::StatusOr<std::optional<int64_t>> JsonView::GetOptionalInt64(
    absl::string_view key) const {
  return GetOptional(key, &JsonView::AsInt64);
}
absl::StatusOr<std::optional<uint64_t>> JsonView::GetOptionalUint64(
    absl::string_view key) const {
  return GetOptional(key, &JsonView::AsUint64);
}
absl::StatusOr<std::optional<bool>> JsonView::GetOptionalBool(
    absl::string_view key) const {
  return GetOptional(key, &JsonView::AsBool);
}
absl::StatusOr<std::optional<JsonView>> JsonView::GetOptionalObject(
    absl::string_view key) const {
  return GetOptional(key, &JsonView::AsObject);
}
absl::StatusOr<std::optional<JsonView>> JsonView::GetOptionalArray(
    absl::string_view key) const {
  return GetOptional(key, &JsonView::AsArray);
}

JsonView JsonView::Element(size_t index) const {
  DCHECK(value_->is_array());
  const Json::array_t& elements = value_->get_ref<const Json::array_t&>();
  DCHECK_LT(index, elements.size());
  return JsonView(elements[index], absl::StrCat(path_, "/", index));
}

std::string JsonView::ChildPath(absl::string_view key) const {
  // RFC 6901 §3: '~' is written "~0" and '/' is written "~1".
  return absl::StrCat(path_, "/",
                      absl::StrReplaceAll(key, {{"~", "~0"}, {"/", "~1"}}));
}

}  // namespace ferret

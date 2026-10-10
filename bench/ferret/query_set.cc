// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "bench/ferret/query_set.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "nlohmann/json.hpp"

#include "ferret/rpc/json_codec.h"

namespace ferret_bench {
namespace {

absl::StatusOr<QuerySpec> ParseQuery(absl::string_view line) {
  ABSL_ASSIGN_OR_RETURN(const nlohmann::json json, ferret::ParseJson(line));
  const ferret::JsonView view(json);
  QuerySpec spec;
  ABSL_ASSIGN_OR_RETURN(spec.id, view.GetString("id"));
  ABSL_ASSIGN_OR_RETURN(spec.query_class, view.GetString("class"));
  ABSL_ASSIGN_OR_RETURN(const std::string kind, view.GetString("kind"));
  ABSL_ASSIGN_OR_RETURN(spec.pattern.pattern, view.GetString("pattern"));
  ABSL_ASSIGN_OR_RETURN(spec.pattern.case_sensitive,
                        view.GetBool("case_sensitive"));
  if (spec.id.empty()) return absl::InvalidArgumentError("empty id");
  if (spec.query_class.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("query ", spec.id, ": empty class"));
  }
  if (kind != "literal" && kind != "regex") {
    return absl::InvalidArgumentError(
        absl::StrCat("query ", spec.id, ": kind \"", kind,
                     "\" is neither \"literal\" nor \"regex\""));
  }
  spec.pattern.literal = kind == "literal";
  if (spec.pattern.pattern.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("query ", spec.id, ": empty pattern"));
  }
  return spec;
}

}  // namespace

absl::StatusOr<std::vector<QuerySpec>> ParseQuerySet(absl::string_view text) {
  if (text.empty()) return absl::InvalidArgumentError("no queries");
  std::vector<QuerySpec> specs;
  absl::flat_hash_map<std::string, size_t> line_of_id;
  size_t line_number = 0;
  for (const absl::string_view line :
       absl::StrSplit(absl::StripSuffix(text, "\n"), '\n')) {
    ++line_number;
    absl::StatusOr<QuerySpec> spec = ParseQuery(line);
    if (!spec.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("line ", line_number, ": ", spec.status().message()));
    }
    const auto [it, inserted] = line_of_id.emplace(spec->id, line_number);
    if (!inserted) {
      return absl::InvalidArgumentError(
          absl::StrCat("line ", line_number, ": id \"", spec->id,
                       "\" duplicates line ", it->second));
    }
    specs.push_back(*std::move(spec));
  }
  return specs;
}

}  // namespace ferret_bench

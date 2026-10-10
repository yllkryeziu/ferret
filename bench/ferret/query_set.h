// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Reads a benchmark query set (bench/queries/*.jsonl): one JSON object per
// line with at least
//
//   {"id": "linux-0001", "class": "identifier", "kind": "literal",
//    "pattern": "sched_entity", "case_sensitive": true}
//
// Other fields, such as the generator's labels, are ignored. The validation
// matches bench/zoekt/queries.go, so both engines accept exactly the same
// query sets.

#ifndef BENCH_FERRET_QUERY_SET_H_
#define BENCH_FERRET_QUERY_SET_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "ferret/regex/compile.h"

namespace ferret_bench {

struct QuerySpec {
  std::string id;
  // The query class, such as "identifier" or "zero".
  std::string query_class;
  ferret::PatternSpec pattern;
};

// Parses a query set. InvalidArgument, naming the line, for malformed JSON,
// a missing or empty id, class, or pattern, a kind other than "literal" or
// "regex", a missing case_sensitive, a duplicate id, or an empty set.
absl::StatusOr<std::vector<QuerySpec>> ParseQuerySet(absl::string_view text);

}  // namespace ferret_bench

#endif  // BENCH_FERRET_QUERY_SET_H_

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The (path, line) result sets that the benchmark parity check compares, and
// their canonical digest. bench/zoekt/results.go defines the same encoding
// for Zoekt; the two share golden vectors, so a digest match means the
// engines found exactly the same lines.

#ifndef BENCH_FERRET_RESULT_SET_H_
#define BENCH_FERRET_RESULT_SET_H_

#include <compare>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/types/span.h"

#include "ferret/search/search_types.h"

namespace ferret_bench {

// One matching line: a repo-relative path and a 1-based line number.
struct LineRef {
  std::string path;
  int64_t line = 0;

  // Orders by path bytes, then by line number.
  friend auto operator<=>(const LineRef&, const LineRef&) = default;
};

// Returns the distinct matching lines of `response`, sorted. Context lines
// are not matches and are left out. O(n log n) in the kept lines.
std::vector<LineRef> MatchedLines(const ferret::SearchResponse& response);

// Returns the lowercase hex SHA-256 of, for each entry of the sorted,
// distinct `lines` in order, the path bytes, a NUL byte, the decimal line
// number, and a newline. O(total bytes).
std::string LineSetDigest(absl::Span<const LineRef> lines);

}  // namespace ferret_bench

#endif  // BENCH_FERRET_RESULT_SET_H_

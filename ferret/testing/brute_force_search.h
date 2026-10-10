// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The definition of a text search's results, for differential tests: every
// line of every file is matched on its own, with no index, no chunks and no
// budget. Deliberately simple and slow.

#ifndef FERRET_TESTING_BRUTE_FORCE_SEARCH_H_
#define FERRET_TESTING_BRUTE_FORCE_SEARCH_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "ferret/index/rank_key.h"
#include "ferret/search/search_types.h"

namespace ferret_testing {

struct CorpusFile {
  std::string path;
  std::string content;
};

// Returns a FileResult for each file of `corpus` that passes the path glob
// and has a matching line, in corpus order. Each has `path`,
// `num_matching_lines`, and `lines`: the first `max_lines_per_file` matching
// lines, and each line within `context_lines` of one of them as context.
// Ignores `max_lines`, `max_files` and `max_verify_bytes`, and leaves the
// other fields,
// including each line's `match_class`, at their defaults. InvalidArgument
// for a pattern or glob that does not compile.
absl::StatusOr<std::vector<ferret::FileResult>> BruteForceSearch(
    absl::Span<const CorpusFile> corpus, const ferret::SearchRequest& request);

// One path in a worktree view. Equal content must have the same ingest key.
struct ViewCorpusFile {
  std::string path;
  std::string content;
  ferret::RankKey rank_key;
};

// Exhaustive view oracle: verifies each distinct content once in ingest rank
// order, expands matching paths, and applies result and line caps. It does not
// model chunk byte budgets or early termination.
absl::StatusOr<ferret::SearchResponse> BruteForceViewSearch(
    absl::Span<const ViewCorpusFile> corpus,
    const ferret::SearchRequest& request);

}  // namespace ferret_testing

#endif  // FERRET_TESTING_BRUTE_FORCE_SEARCH_H_

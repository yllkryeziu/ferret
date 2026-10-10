// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "bench/ferret/rpc_bench.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "bench/ferret/query_set.h"
#include "bench/ferret/result_set.h"
#include "bench/ferret/search_bench.h"

#include "ferret/engine/engine.h"
#include "ferret/search/search_types.h"
#include "ferret/workspace/workspace.h"

namespace ferret_bench {

absl::StatusOr<std::vector<SearchRecord>> RunBackendBench(
    const ferret::SearchBackend& backend, absl::string_view root,
    absl::string_view engine_name, absl::Span<const QuerySpec> queries,
    const SearchBenchConfig& config) {
  if (queries.empty() || config.warmup_passes < 0 || config.timed_runs < 1) {
    return absl::InvalidArgumentError("invalid backend benchmark limits");
  }
  std::vector<SearchRecord> records;
  std::vector<ferret::SearchRequest> requests;
  for (const QuerySpec& query : queries) {
    requests.push_back(MakeSearchRequest(query, config));
    SearchRecord record;
    record.id = query.id;
    record.query_class = query.query_class;
    record.engine = std::string(engine_name);
    record.mode = config.mode;
    if (config.mode == SearchMode::kTopK) record.k = config.k;
    record.context = config.context_lines;
    record.search_threads = config.search_threads;
    records.push_back(std::move(record));
  }
  const ferret::SearchConsistency consistency{
      .mode = ferret::SearchConsistencyMode::kNone};
  for (int pass = 0; pass < config.warmup_passes + config.timed_runs; ++pass) {
    const bool timed = pass >= config.warmup_passes;
    for (const size_t i :
         ShuffledQueryOrder(queries.size(), config.seed, pass)) {
      const auto start = std::chrono::steady_clock::now();
      absl::StatusOr<ferret::WorkspaceSearchResult> result =
          backend.Search(root, requests[i], consistency);
      const int64_t elapsed_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - start)
              .count();
      if (!result.ok()) {
        return absl::Status(result.status().code(),
                            absl::StrCat("query ", queries[i].id, ": ",
                                         result.status().message()));
      }
      if (!timed) continue;
      SearchRecord& record = records[i];
      record.latencies_ns.push_back(elapsed_ns);
      const std::vector<LineRef> lines = MatchedLines(result->response);
      const std::string digest = LineSetDigest(lines);
      if (record.latencies_ns.size() > 1) {
        record.digest_stable &= record.result_digest == digest;
        continue;
      }
      record.truncated = !result->response.complete;
      record.terminated_early = result->response.terminated_early;
      record.file_count = static_cast<int64_t>(result->response.files.size());
      record.result_count = static_cast<int64_t>(lines.size());
      record.result_digest = digest;
      record.engine_stats.num_matching_files =
          result->response.num_matching_files;
      record.engine_stats.num_matching_lines =
          result->response.num_matching_lines;
    }
  }
  return records;
}

}  // namespace ferret_bench

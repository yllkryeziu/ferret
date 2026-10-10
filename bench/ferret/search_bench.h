// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#ifndef BENCH_FERRET_SEARCH_BENCH_H_
#define BENCH_FERRET_SEARCH_BENCH_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "bench/ferret/query_set.h"
#include "bench/ferret/result_set.h"
#include "nlohmann/json.hpp"

#include "ferret/index/read_state.h"
#include "ferret/index/segment.h"
#include "ferret/search/searcher.h"

namespace ferret_bench {

enum class SearchMode { kExhaustive, kTopK };

// One search's wall time by stage, in nanoseconds. All but `worker_scan` are
// the calling thread's wall time and partition `total` together with
// `other`.
struct StageBreakdown {
  // Compiling the pattern and planning the trigram query, without the
  // dictionary lookups below.
  int64_t plan = 0;
  // Looking up every trigram of the query in the segment dictionary.
  int64_t dictionary = 0;
  // Decoding posting lists.
  int64_t decode = 0;
  // Intersecting candidate sets with lists, merging lists, and probing skip
  // tables.
  int64_t intersect = 0;
  // The rest of candidate evaluation: windows, buffers, rank lookups.
  int64_t window = 0;
  // The priority scan of the rank-stop bound.
  int64_t priority = 0;
  // Verifying candidate chunks line by line, ranking documents as they
  // complete, and waiting for the parallel pipeline.
  int64_t verify = 0;
  // The sampled count estimate.
  int64_t sample = 0;
  // Ranking and trimming the files.
  int64_t rank = 0;
  // Encoding the response (the shaping step of the serving path).
  int64_t encode = 0;
  // `total` minus everything above: setup, teardown, and the clock reads.
  int64_t other = 0;
  int64_t total = 0;
  // Pipeline threads' time scanning grains, summed over threads.
  int64_t worker_scan = 0;
  // Inside `sample`: its dictionary lookups, window evaluation (decode and
  // intersect are inside it), verifying the selected documents. The rest of
  // `sample` is the per-candidate tests. Profiling only.
  int64_t sample_dictionary = 0;
  int64_t sample_evaluate = 0;
  int64_t sample_decode = 0;
  int64_t sample_intersect = 0;
  int64_t sample_verify = 0;
};

// Splits a search that took `total_ns`, with `encode_ns` of it encoding the
// response, into stages. Requires stats from a search run with
// SearcherOptions::stage_timing.
StageBreakdown ComputeStages(const ferret::SearchStats& stats, int64_t total_ns,
                             int64_t encode_ns);

// The ordered files of a response with each file's kept lines: what a client
// sees, and what the oracle comparison checks.
std::string ResponseSignature(const ferret::SearchResponse& response);

// A response against the brute-force top K (SearchBenchConfig::oracle).
struct OracleResult {
  // Whether no byte budget cut the response, so it must equal the oracle's.
  bool complete = false;
  // Whether the files and kept lines equal the oracle's.
  bool equal = false;
  int64_t oracle_files = 0;
  // Files of the response that are also in the oracle's top K.
  int64_t common_files = 0;
  int64_t oracle_matching_files = 0;
  int64_t oracle_matching_lines = 0;
  // The response's (possibly estimated) totals.
  int64_t matching_files = 0;
  int64_t matching_lines = 0;
  // The oracle's wall time, which is not part of any latency.
  int64_t oracle_ns = 0;

  // A complete response that differs from the oracle: a bug.
  bool mismatch() const { return complete && !equal; }
};

// Compares `got` with `oracle`, the search that had no early stop, no byte
// budget and no counting-only shortcut.
OracleResult CompareWithOracle(const ferret::SearchResponse& got,
                               const ferret::SearchResponse& oracle);

struct SearchBenchConfig {
  SearchMode mode = SearchMode::kTopK;
  int k = 20;
  int context_lines = 2;
  int64_t max_matches = 1'000'000;
  int clients = 1;
  int search_threads = 1;
  // Grains per worker of run-ahead once the top-K heap is full.
  int full_run_ahead = 4;
  // The segment verification budget of a top-K query.
  int64_t max_verify_bytes = int64_t{128} << 20;
  int warmup_passes = 3;
  int timed_runs = 10;
  uint64_t seed = 1;
  bool full_results = false;
  // Print, for each query's first timed run on one thread and one client, a
  // JSON line on stderr that says what blocked its rank stop.
  bool diagnose_stop = false;
  // Print, for each query's first timed run on one thread and one client,
  // every verified document and the priority and code-token documents as a
  // JSON line on stderr. Meant for the exhaustive mode.
  bool trace_docs = false;
  // SearchRequest::count_sample in top-K mode: estimated totals, as G1 is
  // defined. It turns counting-only verification off.
  bool count_sample = true;
  // Whether the timed region also encodes the response with the daemon's
  // codec, so a latency is "request decoded -> response encoded".
  bool encode_response = true;
  // Collect the stage breakdown of every timed run (SearcherOptions::
  // stage_timing).
  bool stage_timing = false;
  // After the first timed run of a query, search again with every lever that
  // can change a result turned off and compare (untimed).
  bool oracle = false;
  // Ablation switches; each turns one lever off. Results must not change,
  // except that a search without early termination can run out of its byte
  // budget where the default does not.
  bool early_termination = true;
  bool name_aware_bound = true;
  bool count_only = true;
  // SearcherOptions::count_sample_level.
  int count_sample_level = 3;
  bool parallel_verify = true;
  bool scalar_decode = false;
  bool scalar_intersect = false;
  bool skip_probe = true;
  // SearcherOptions::stop_at_line_cap.
  bool line_cap_stop = true;
  // SearcherOptions::priority_restriction and priority_parallel.
  ferret::PriorityRestriction priority_restriction =
      ferret::PriorityRestriction::kAuto;
  bool priority_parallel = true;
  bool priority_alternatives = true;
  // SearcherOptions::max_bigram_posting_share.
  double bigram_posting_share = 8;
  // Non-owning snapshot for the duration of RunSearchBench; null uses hints.
  const ferret::ReadState* state = nullptr;
  ferret::SearchView* view = nullptr;
  // The single-client edit workload invalidates one view chunk per query.
  ferret::ViewWriter* view_writer = nullptr;
  ferret::PathId edit_path{};
  ferret::BlobId edit_blob{};
};

struct SearchEngineStats {
  int64_t candidate_chunks = 0;
  int64_t verified_chunks = 0;
  int64_t verified_bytes = 0;
  // Of the verified bytes, those of documents that were counted only.
  int64_t count_only_bytes = 0;
  int64_t count_only_documents = 0;
  // The count sample: candidate chunks its streams yielded, documents of its
  // frame, bytes of the documents it verified.
  int64_t count_sample_chunks = 0;
  int64_t count_sample_documents = 0;
  int64_t count_sample_bytes = 0;
  int64_t priority_documents = 0;
  int64_t priority_basenames_tested = 0;
  int64_t priority_basenames_scanned = 0;
  int64_t priority_restricted_sources = 0;
  int64_t priority_scan_ns = 0;
  int64_t compile_ns = 0;
  int64_t evaluate_ns = 0;
  int64_t verify_ns = 0;
  int64_t num_matching_files = 0;
  int64_t num_matching_lines = 0;
};

struct SearchRecord {
  std::string id;
  std::string query_class;
  std::string engine = "ferret";
  SearchMode mode = SearchMode::kTopK;
  std::optional<int> k;
  int context = 0;
  int threads = 1;
  int search_threads = 1;
  int clients = 1;
  std::vector<int64_t> latencies_ns;
  // The priority scan time of every run; the record reports their median.
  std::vector<int64_t> priority_scan_runs_ns;
  // The stages of every timed run, with stage timing.
  std::vector<StageBreakdown> stage_runs;
  bool truncated = false;
  bool terminated_early = false;
  int64_t file_count = 0;
  int64_t result_count = 0;
  std::string result_digest;
  bool digest_stable = true;
  SearchEngineStats engine_stats;
  // Bytes of the encoded response, when it is encoded.
  int64_t response_bytes = 0;
  // The stages of the timed run with the median latency, with stage timing.
  std::optional<StageBreakdown> stages;
  std::optional<OracleResult> oracle;
  int64_t cold_view_names_ns = 0;
  int64_t view_rebuild_ns = 0;
  std::optional<std::vector<LineRef>> results;
};

// Runs every query once per client in each pass. The returned records are in
// query-set order; only Search is included in the measured latency.
absl::StatusOr<std::vector<SearchRecord>> RunSearchBench(
    const ferret::Segment& segment, absl::Span<const QuerySpec> queries,
    const SearchBenchConfig& config);

// The request a query makes in `config`'s mode (K, context, estimated totals).
ferret::SearchRequest MakeSearchRequest(const QuerySpec& query,
                                        const SearchBenchConfig& config);

// The seeded visiting order of one pass over `count` queries, for one client.
std::vector<size_t> ShuffledQueryOrder(size_t count, uint64_t seed, int pass);

// Converts a record to the JSONL search record schema.
nlohmann::json SearchRecordToJson(const SearchRecord& record);

}  // namespace ferret_bench
#endif  // BENCH_FERRET_SEARCH_BENCH_H_

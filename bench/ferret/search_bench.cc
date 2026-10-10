// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "bench/ferret/search_bench.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "bench/ferret/query_set.h"
#include "bench/ferret/result_set.h"
#include "nlohmann/json.hpp"

#include "ferret/base/ablation.h"
#include "ferret/base/thread_pool.h"
#include "ferret/index/blob_registry.h"
#include "ferret/index/segment.h"
#include "ferret/postings/ids.h"
#include "ferret/regex/compile.h"
#include "ferret/rpc/json_codec.h"
#include "ferret/rpc/search_codec.h"
#include "ferret/search/ranking.h"
#include "ferret/search/search_types.h"
#include "ferret/search/searcher.h"
#include "ferret/text/literal_finder.h"
#include "ferret/workspace/path_table.h"
#include "ferret/workspace/view_state.h"
#include "ferret/workspace/workspace.h"

namespace ferret_bench {
namespace {

// A worker retains only scalar results and, when requested, matching line
// references. SearchResponse and its line text die before the next query.
struct Measurement {
  int64_t latency_ns = 0;
  std::string digest;
  int64_t result_count = 0;
  int64_t file_count = 0;
  bool truncated = false;
  bool terminated_early = false;
  SearchEngineStats engine_stats;
  int64_t response_bytes = 0;
  std::optional<StageBreakdown> stages;
  std::optional<OracleResult> oracle;
  int64_t view_rebuild_ns = 0;
  std::optional<std::vector<LineRef>> results;
};

ferret::SearchRequest MakeRequestImpl(const QuerySpec& query,
                                      const SearchBenchConfig& config) {
  ferret::SearchRequest request;
  request.pattern = query.pattern.pattern;
  request.regex = !query.pattern.literal;
  request.case_sensitive = query.pattern.case_sensitive;
  request.context_lines = config.context_lines;
  request.max_lines_per_file = std::numeric_limits<int>::max();
  if (config.mode == SearchMode::kTopK) {
    request.max_files = config.k;
    request.max_verify_bytes = config.max_verify_bytes;
    request.count_sample = config.count_sample;
  } else {
    request.max_files = std::numeric_limits<int64_t>::max();
    request.exact_counts = true;
    request.max_lines = config.max_matches;
    request.max_verify_bytes = std::numeric_limits<int64_t>::max();
  }
  return request;
}

// Seed all 64 bits of each input rather than truncating the pass or client.
std::vector<size_t> ShuffledOrder(size_t count, uint64_t seed, int pass,
                                  int client) {
  std::seed_seq seeds{
      static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32),
      static_cast<uint32_t>(pass), static_cast<uint32_t>(client)};
  std::mt19937_64 rng(seeds);
  std::vector<size_t> order(count);
  std::iota(order.begin(), order.end(), 0);
  for (size_t i = count; i > 1; --i) {
    const size_t j = static_cast<size_t>(rng() % i);
    std::swap(order[i - 1], order[j]);
  }
  return order;
}

Measurement SummarizeResponse(int64_t latency_ns,
                              const ferret::SearchResponse& response,
                              const SearchBenchConfig& config) {
  std::vector<LineRef> lines = MatchedLines(response);
  const ferret::SearchStats& stats = response.stats;
  Measurement measurement{
      .latency_ns = latency_ns,
      .digest = LineSetDigest(lines),
      .result_count = static_cast<int64_t>(lines.size()),
      .file_count = static_cast<int64_t>(response.files.size()),
      .truncated = !response.complete ||
                   (config.mode == SearchMode::kExhaustive &&
                    response.num_matching_lines > config.max_matches),
      .terminated_early = response.terminated_early,
      .engine_stats = {
          .candidate_chunks = stats.candidate_chunks,
          .verified_chunks = stats.verified_chunks,
          .verified_bytes = stats.verified_bytes,
          .count_only_bytes = stats.count_only_bytes,
          .count_only_documents = stats.count_only_documents,
          .count_sample_chunks = stats.count_sample_chunks,
          .count_sample_documents = stats.count_sample_documents,
          .count_sample_bytes = stats.count_sample_bytes,
          .priority_documents = stats.priority_documents,
          .priority_basenames_tested = stats.priority_basenames_tested,
          .priority_basenames_scanned = stats.priority_basenames_scanned,
          .priority_restricted_sources = stats.priority_restricted_sources,
          .priority_scan_ns =
              absl::ToInt64Nanoseconds(stats.priority_scan_time),
          .compile_ns = absl::ToInt64Nanoseconds(stats.compile_time),
          .evaluate_ns = absl::ToInt64Nanoseconds(stats.evaluate_time),
          .verify_ns = absl::ToInt64Nanoseconds(stats.verify_time),
          .num_matching_files = response.num_matching_files,
          .num_matching_lines = response.num_matching_lines}};
  if (config.full_results) {
    measurement.results = std::move(lines);
  }
  return measurement;
}

void RecordResult(SearchRecord& record, Measurement measurement) {
  record.latencies_ns.push_back(measurement.latency_ns);
  record.priority_scan_runs_ns.push_back(
      measurement.engine_stats.priority_scan_ns);
  if (measurement.stages.has_value()) {
    record.stage_runs.push_back(*measurement.stages);
  }
  if (record.latencies_ns.size() > 1) {
    record.digest_stable &= record.result_digest == measurement.digest;
    return;
  }
  record.truncated = measurement.truncated;
  record.terminated_early = measurement.terminated_early;
  record.file_count = measurement.file_count;
  record.result_count = measurement.result_count;
  record.result_digest = std::move(measurement.digest);
  record.engine_stats = measurement.engine_stats;
  record.response_bytes = measurement.response_bytes;
  record.oracle = measurement.oracle;
  record.view_rebuild_ns = measurement.view_rebuild_ns;
  record.results = std::move(measurement.results);
}

// Summarizes the rank-stop tests of one search as a JSON line on stderr.
void PrintStopDiagnosis(const QuerySpec& query,
                        const std::vector<ferret::StopProbe>& probes,
                        const ferret::SearchResponse& response) {
  nlohmann::json json = {
      {"diagnosis", query.id},
      {"pattern", query.pattern.pattern},
      {"case_sensitive", query.pattern.case_sensitive},
      {"tests", probes.size()},
      {"terminated_early", response.terminated_early},
      {"complete", response.complete},
      {"priority_documents", response.stats.priority_documents}};
  int64_t unlisted = 0;
  int64_t priority = 0;
  int64_t priority_in_tokens = 0;
  int64_t code_token = 0;
  int64_t untokenized = 0;
  int64_t exhausted = 0;
  for (const ferret::StopProbe& probe : probes) {
    exhausted += probe.lines_exhausted;
    unlisted += probe.unlisted_blocks;
    untokenized += probe.untokenized_blocks;
    priority += probe.priority_blocks;
    priority_in_tokens += probe.priority_in_tokens_blocks;
    code_token += probe.code_token_blocks;
  }
  json["tests_with_line_cap_spent"] = exhausted;
  json["tests_blocked_by_unlisted_ceiling"] = unlisted;
  json["tests_blocked_by_untokenized"] = untokenized;
  json["tests_blocked_by_priority"] = priority;
  json["tests_blocked_by_priority_in_tokens"] = priority_in_tokens;
  json["tests_blocked_by_code_token"] = code_token;
  auto describe = [](const ferret::StopProbe& probe) {
    return nlohmann::json{
        {"next_tier", static_cast<int>(probe.next_tier)},
        {"next_prior", probe.next_prior},
        {"worst_tier", static_cast<int>(probe.worst_tier)},
        {"worst_score", probe.worst_score},
        {"stop", probe.stop},
        {"heap_full", probe.heap_full},
        {"kept_lines", probe.kept_lines},
        {"lines_exhausted", probe.lines_exhausted},
        {"name_aware", probe.name_aware},
        {"unlisted_blocks", probe.unlisted_blocks},
        {"untokenized_blocks", probe.untokenized_blocks},
        {"priority_blocks", probe.priority_blocks},
        {"priority_in_tokens_blocks", probe.priority_in_tokens_blocks},
        {"code_token_blocks", probe.code_token_blocks},
        {"priority_remaining", probe.priority_remaining},
        {"priority_ahead", probe.priority_ahead}};
  };
  if (!probes.empty()) {
    json["first_test"] = describe(probes.front());
    json["last_test"] = describe(probes.back());
  }
  std::cerr << ferret::SerializeJson(json) << "\n";
}

// Prints the documents a search verified, with the segment's priority
// (basename match) and code-token documents, as one JSON line on stderr.
void PrintDocTrace(const ferret::Segment& segment, const QuerySpec& query,
                   const ferret::SearchRequest& request,
                   const std::vector<ferret::DocTrace>& traces,
                   const ferret::SearchResponse& response) {
  auto row = [&](uint32_t id) {
    const ferret::SegmentDoc doc = segment.doc(id);
    return nlohmann::json::array({id, static_cast<int>(doc.tier),
                                  static_cast<int>(doc.prior), doc.num_lines});
  };
  nlohmann::json docs = nlohmann::json::array();
  for (const ferret::DocTrace& trace : traces) {
    nlohmann::json entry = row(trace.doc);
    entry.push_back(trace.chunks);
    entry.push_back(trace.bytes);
    entry.push_back(trace.matching_lines);
    entry.push_back(trace.first_identifier_line);
    docs.push_back(std::move(entry));
  }
  nlohmann::json priority = nlohmann::json::array();
  absl::StatusOr<std::unique_ptr<ferret::CompiledPattern>> pattern =
      ferret::CompilePattern({.pattern = request.pattern,
                              .literal = !request.regex,
                              .case_sensitive = request.case_sensitive});
  if (pattern.ok()) {
    for (uint32_t id = 0; id < segment.num_docs(); ++id) {
      const absl::string_view path = segment.doc(id).hint_path;
      const size_t slash = path.rfind('/');
      const absl::string_view base =
          slash == absl::string_view::npos ? path : path.substr(slash + 1);
      if ((*pattern)->MatchesWithin(base)) priority.push_back(row(id));
    }
  }
  nlohmann::json tokens = nlohmann::json::array();
  std::vector<ferret::DocId> token_docs;
  bool has_tokens = false;
  if (const auto key = ferret::CodeTokenBoundKey(request); key.has_value()) {
    has_tokens = segment.FindCodeTokenDocs(*key, token_docs);
    for (const ferret::DocId id : token_docs) tokens.push_back(row(id));
  }
  std::cerr << ferret::SerializeJson(nlohmann::json{
                   {"trace", query.id},
                   {"pattern", query.pattern.pattern},
                   {"case_sensitive", query.pattern.case_sensitive},
                   {"has_tokens", has_tokens},
                   {"matching_files", response.num_matching_files},
                   {"docs", std::move(docs)},
                   {"priority", std::move(priority)},
                   {"tokens", std::move(tokens)}})
            << "\n";
}

int64_t Ns(absl::Duration duration) {
  return absl::ToInt64Nanoseconds(duration);
}

// The oracle's request and options: every lever that can change what a
// response holds is off. The byte budget is unlimited, there is no rank stop
// or line-cap stop, no counting-only shortcut, no estimate and no threads, so
// the engine verifies every candidate chunk of every matching file in rank
// order and ranks the files by the same ScoreFile.
ferret::SearchRequest OracleRequest(ferret::SearchRequest request) {
  request.max_verify_bytes = std::numeric_limits<int64_t>::max();
  request.count_sample = false;
  return request;
}

ferret::SearcherOptions OracleOptions() {
  ferret::SearcherOptions options;
  options.early_termination = false;
  options.stop_at_line_cap = false;
  options.count_only_excluded = false;
  options.executor = nullptr;
  return options;
}

absl::Status ValidateConfig(const SearchBenchConfig& config) {
  if ((config.mode == SearchMode::kTopK && config.k < 1) ||
      config.context_lines < 0 ||
      config.context_lines > ferret::kMaxContextLines ||
      config.max_matches < 1 || config.clients < 1 ||
      config.search_threads < 1 || config.full_run_ahead < 1 ||
      config.warmup_passes < 0 || config.timed_runs < 1 ||
      (config.view_writer && (config.view == nullptr || config.clients != 1)) ||
      config.warmup_passes >
          std::numeric_limits<int>::max() - config.timed_runs) {
    return absl::InvalidArgumentError("invalid search benchmark limits");
  }
  return absl::OkStatus();
}

// Applies the process-wide kernel switches of a config for one run, and puts
// them back afterwards.
class AblationScope final {
 public:
  explicit AblationScope(const SearchBenchConfig& config)
      : decode_(ferret::Ablation::scalar_decode.exchange(config.scalar_decode)),
        intersect_(ferret::Ablation::scalar_intersect.exchange(
            config.scalar_intersect)),
        skip_(ferret::Ablation::no_skip_probe.exchange(!config.skip_probe)) {}
  AblationScope(const AblationScope&) = delete;
  AblationScope& operator=(const AblationScope&) = delete;
  ~AblationScope() {
    ferret::Ablation::scalar_decode = decode_;
    ferret::Ablation::scalar_intersect = intersect_;
    ferret::Ablation::no_skip_probe = skip_;
  }

 private:
  bool decode_;
  bool intersect_;
  bool skip_;
};

}  // namespace

ferret::SearchRequest MakeSearchRequest(const QuerySpec& query,
                                        const SearchBenchConfig& config) {
  return MakeRequestImpl(query, config);
}

std::vector<size_t> ShuffledQueryOrder(size_t count, uint64_t seed, int pass) {
  return ShuffledOrder(count, seed, pass, /*client=*/0);
}

StageBreakdown ComputeStages(const ferret::SearchStats& stats, int64_t total_ns,
                             int64_t encode_ns) {
  StageBreakdown stages;
  stages.dictionary = Ns(stats.dictionary_time);
  stages.plan = Ns(stats.compile_time) - stages.dictionary;
  stages.decode = Ns(stats.decode_time);
  stages.intersect = Ns(stats.intersect_time);
  stages.priority = Ns(stats.priority_scan_time);
  // evaluate_time holds the priority scan once the rank-stop test needed it.
  stages.window = Ns(stats.evaluate_time) - stages.priority - stages.decode -
                  stages.intersect;
  stages.verify = Ns(stats.verify_time);
  stages.sample = Ns(stats.count_sample_time);
  stages.rank = Ns(stats.rank_time);
  stages.encode = encode_ns;
  stages.total = total_ns;
  stages.other = total_ns - stages.plan - stages.dictionary - stages.decode -
                 stages.intersect - stages.window - stages.priority -
                 stages.verify - stages.sample - stages.rank - stages.encode;
  stages.worker_scan = stats.worker_scan_ns;
  stages.sample_dictionary = Ns(stats.count_sample_dictionary_time);
  stages.sample_evaluate = Ns(stats.count_sample_evaluate_time);
  stages.sample_decode = Ns(stats.count_sample_decode_time);
  stages.sample_intersect = Ns(stats.count_sample_intersect_time);
  stages.sample_verify = Ns(stats.count_sample_verify_time);
  return stages;
}

std::string ResponseSignature(const ferret::SearchResponse& response) {
  std::string signature;
  for (const ferret::FileResult& file : response.files) {
    absl::StrAppend(&signature, file.path, "\n");
    for (const ferret::ResultLine& line : file.lines) {
      absl::StrAppend(&signature, line.is_match ? "m" : "c", line.number, "\n");
    }
  }
  return signature;
}

OracleResult CompareWithOracle(const ferret::SearchResponse& got,
                               const ferret::SearchResponse& oracle) {
  OracleResult result;
  result.complete = got.complete;
  result.equal = ResponseSignature(got) == ResponseSignature(oracle);
  result.oracle_files = static_cast<int64_t>(oracle.files.size());
  absl::flat_hash_set<std::string> paths;
  for (const ferret::FileResult& file : oracle.files) paths.insert(file.path);
  for (const ferret::FileResult& file : got.files) {
    result.common_files += paths.contains(file.path);
  }
  result.oracle_matching_files = oracle.num_matching_files;
  result.oracle_matching_lines = oracle.num_matching_lines;
  result.matching_files = got.num_matching_files;
  result.matching_lines = got.num_matching_lines;
  return result;
}

absl::StatusOr<std::vector<SearchRecord>> RunSearchBench(
    const ferret::Segment& segment, absl::Span<const QuerySpec> queries,
    const SearchBenchConfig& config) {
  ABSL_RETURN_IF_ERROR(ValidateConfig(config));
  if (queries.empty()) {
    return absl::InvalidArgumentError("no queries");
  }
  std::vector<ferret::SearchRequest> requests;
  std::vector<SearchRecord> records;
  for (const QuerySpec& query : queries) {
    requests.push_back(MakeSearchRequest(query, config));
    SearchRecord record;
    record.id = query.id;
    record.query_class = query.query_class;
    record.mode = config.mode;
    if (config.mode == SearchMode::kTopK) {
      record.k = config.k;
    }
    record.context = config.context_lines;
    record.clients = config.clients;
    record.search_threads = config.search_threads;
    records.push_back(std::move(record));
  }

  if (config.view) {
    const auto start = std::chrono::steady_clock::now();
    config.view->state->ForEachBasenameContaining(
        "", ferret::LiteralFinder::Mode::kAsciiCaseInsensitive,
        config.view->paths, [](ferret::BlobId, ferret::PathId) {});
    const int64_t cold_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count();
    for (SearchRecord& record : records) record.cold_view_names_ns = cold_ns;
  }
  ferret::SearchExecutor executor(config.search_threads);
  ferret::SearcherOptions options;
  options.executor = &executor;
  options.stop_at_line_cap = config.line_cap_stop;
  options.priority_restriction = config.priority_restriction;
  options.priority_parallel = config.priority_parallel;
  options.priority_alternatives = config.priority_alternatives;
  options.max_bigram_posting_share = config.bigram_posting_share;
  options.full_ring_slots_per_worker = config.full_run_ahead;
  options.early_termination = config.early_termination;
  options.name_aware_bound = config.name_aware_bound;
  options.count_only_excluded = config.count_only;
  options.count_sample_level = config.count_sample_level;
  options.stage_timing = config.stage_timing;
  if (!config.parallel_verify) {
    options.parallel_threshold_bytes = std::numeric_limits<int64_t>::max();
  }
  const AblationScope ablation(config);
  ferret::ThreadPool pool(std::max(config.clients - 1, 1),
                          ferret::ThreadQos::kUserInteractive);
  for (int pass = 0; pass < config.warmup_passes + config.timed_runs; ++pass) {
    const bool timed = pass >= config.warmup_passes;
    if (config.view_writer) {
      ferret::SearchView& view = *config.view;
      view.state = config.view_writer->ApplyBatch(
          {ferret::ViewEdit{config.edit_path, config.edit_blob}});
    }
    // Each client writes only its own slot; merging in client order makes the
    // first timed result independent of the workers' scheduling order.
    std::vector<std::vector<Measurement>> measurements(
        config.clients, std::vector<Measurement>(queries.size()));
    std::vector<absl::Status> errors(config.clients);
    pool.ParallelFor(
        config.clients, /*grain=*/1, config.clients,
        [&](int /*worker*/, int64_t begin, int64_t end) {
          for (int64_t client = begin; client < end; ++client) {
            const std::vector<size_t> order = ShuffledOrder(
                queries.size(), config.seed, pass, static_cast<int>(client));
            size_t edit_index = 0;
            for (const size_t i : order) {
              // Toggle a path on an unindexed blob in the first chunk. The
              // snapshot changes but no query result can change.
              if (config.view_writer) {
                const ferret::ViewEdit edit{
                    config.edit_path,
                    (++edit_index % 2 == 0)
                        ? std::optional<ferret::BlobId>(config.edit_blob)
                        : std::nullopt};
                // Reset at every pass to the known, initially present path.
                ferret::SearchView& view = *config.view;
                view.state = config.view_writer->ApplyBatch({edit});
              }
              const auto start = std::chrono::steady_clock::now();
              int64_t rebuild_ns = 0;
              if (config.view_writer) {
                const auto rebuild_start = std::chrono::steady_clock::now();
                config.view->state->ForEachBasenameContaining(
                    "", ferret::LiteralFinder::Mode::kAsciiCaseInsensitive,
                    config.view->paths, [](ferret::BlobId, ferret::PathId) {});
                rebuild_ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - rebuild_start)
                        .count();
              }
              std::vector<ferret::StopProbe> probes;
              const auto record = [&probes](const ferret::StopProbe& p) {
                probes.push_back(p);
              };
              const absl::FunctionRef<void(const ferret::StopProbe&)>
                  record_probe(record);
              std::vector<ferret::DocTrace> traces;
              const auto record_trace = [&traces](const ferret::DocTrace& t) {
                traces.push_back(t);
              };
              const absl::FunctionRef<void(const ferret::DocTrace&)>
                  trace_probe(record_trace);
              ferret::SearcherOptions query_options = options;
              if (config.trace_docs && timed) {
                query_options.doc_probe = &trace_probe;
              }
              if (config.diagnose_stop && timed) {
                query_options.stop_probe = &record_probe;
              }
              absl::StatusOr<ferret::SearchResponse> response =
                  config.view ? ferret::Search(*config.state, *config.view,
                                               requests[i], query_options)
                              : ferret::SearchSegment(segment, requests[i],
                                                      query_options);
              // The serving path's last step: the response as the daemon
              // writes it. The response moves into the wire object and back.
              int64_t encode_ns = 0;
              int64_t response_bytes = 0;
              ferret::WorkspaceSearchResult wire;
              const bool encode = config.encode_response && response.ok();
              if (encode) {
                const auto encode_start = std::chrono::steady_clock::now();
                wire.response = std::move(*response);
                const std::string encoded = ferret::SerializeJson(
                    ferret::EncodeWorkspaceSearchResult(wire));
                response_bytes = static_cast<int64_t>(encoded.size());
                encode_ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - encode_start)
                        .count();
              }
              const auto elapsed = std::chrono::steady_clock::now() - start;
              if (encode) *response = std::move(wire.response);
              if (!response.ok()) {
                errors[client] =
                    absl::Status(response.status().code(),
                                 absl::StrCat("query ", queries[i].id, ": ",
                                              response.status().message()));
                break;
              }
              if (config.diagnose_stop && timed &&
                  pass == config.warmup_passes) {
                PrintStopDiagnosis(queries[i], probes, *response);
              }
              if (config.trace_docs && timed && pass == config.warmup_passes) {
                PrintDocTrace(segment, queries[i], requests[i], traces,
                              *response);
              }
              if (timed) {
                measurements[client][i] = SummarizeResponse(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        elapsed)
                        .count(),
                    *response, config);
                measurements[client][i].view_rebuild_ns = rebuild_ns;
                measurements[client][i].response_bytes = response_bytes;
                if (config.stage_timing) {
                  measurements[client][i].stages = ComputeStages(
                      response->stats,
                      std::chrono::duration_cast<std::chrono::nanoseconds>(
                          elapsed)
                          .count(),
                      encode_ns);
                }
                // Untimed: the same query with the early stops, the byte
                // budget and the counting-only shortcut off.
                if (config.oracle && client == 0 &&
                    pass == config.warmup_passes) {
                  const auto oracle_start = std::chrono::steady_clock::now();
                  const ferret::SearchRequest oracle_request =
                      OracleRequest(requests[i]);
                  const ferret::SearcherOptions oracle_options =
                      OracleOptions();
                  absl::StatusOr<ferret::SearchResponse> oracle =
                      config.view
                          ? ferret::Search(*config.state, *config.view,
                                           oracle_request, oracle_options)
                          : ferret::SearchSegment(segment, oracle_request,
                                                  oracle_options);
                  if (!oracle.ok()) {
                    errors[client] = absl::Status(
                        oracle.status().code(),
                        absl::StrCat("oracle for ", queries[i].id, ": ",
                                     oracle.status().message()));
                    break;
                  }
                  OracleResult result = CompareWithOracle(*response, *oracle);
                  result.oracle_ns =
                      std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - oracle_start)
                          .count();
                  measurements[client][i].oracle = result;
                }
              }
            }
          }
        });
    for (const absl::Status& error : errors) {
      ABSL_RETURN_IF_ERROR(error);
    }
    if (!timed) {
      continue;
    }
    for (int client = 0; client < config.clients; ++client) {
      for (size_t i = 0; i < queries.size(); ++i) {
        RecordResult(records[i], std::move(measurements[client][i]));
      }
    }
  }
  return records;
}

namespace {
// The median of `runs`, or `fallback` when there are none.
int64_t MedianNs(std::vector<int64_t> runs, int64_t fallback) {
  if (runs.empty()) return fallback;
  std::sort(runs.begin(), runs.end());
  return runs[runs.size() / 2];
}
}  // namespace

nlohmann::json SearchRecordToJson(const SearchRecord& record) {
  const SearchEngineStats& stats = record.engine_stats;
  nlohmann::json json = {
      {"id", record.id},
      {"class", record.query_class},
      {"engine", record.engine},
      {"mode", record.mode == SearchMode::kTopK ? "topk" : "exhaustive"},
      {"context", record.context},
      {"threads", record.threads},
      {"search_threads", record.search_threads},
      {"clients", record.clients},
      {"latencies_ns", record.latencies_ns},
      {"truncated", record.truncated},
      {"terminated_early", record.terminated_early},
      {"file_count", record.file_count},
      {"result_count", record.result_count},
      {"result_digest", record.result_digest},
      {"digest_stable", record.digest_stable},
      {"cold_view_names_ns", record.cold_view_names_ns},
      {"view_rebuild_ns", record.view_rebuild_ns},
      {"engine_stats",
       {{"candidate_chunks", stats.candidate_chunks},
        {"verified_chunks", stats.verified_chunks},
        {"verified_bytes", stats.verified_bytes},
        {"priority_documents", stats.priority_documents},
        {"priority_basenames_tested", stats.priority_basenames_tested},
        {"priority_basenames_scanned", stats.priority_basenames_scanned},
        {"priority_restricted_sources", stats.priority_restricted_sources},
        {"priority_scan_ns",
         MedianNs(record.priority_scan_runs_ns, stats.priority_scan_ns)},
        {"compile_ns", stats.compile_ns},
        {"evaluate_ns", stats.evaluate_ns},
        {"verify_ns", stats.verify_ns},
        {"num_matching_files", stats.num_matching_files},
        {"num_matching_lines", stats.num_matching_lines}}}};
  if (stats.count_only_documents > 0) {
    // Optional: only a search that counted documents without keeping them
    // reports these.
    json["engine_stats"]["count_only_bytes"] = stats.count_only_bytes;
    json["engine_stats"]["count_only_documents"] = stats.count_only_documents;
  }
  if (stats.count_sample_chunks > 0) {
    // Optional: only a search that drew a count sample reports these.
    json["engine_stats"]["count_sample_chunks"] = stats.count_sample_chunks;
    json["engine_stats"]["count_sample_documents"] =
        stats.count_sample_documents;
    json["engine_stats"]["count_sample_bytes"] = stats.count_sample_bytes;
  }
  if (record.response_bytes > 0) json["response_bytes"] = record.response_bytes;
  if (!record.stage_runs.empty()) {
    // The run with the median total, so the stages add up to one latency.
    std::vector<StageBreakdown> runs = record.stage_runs;
    std::sort(runs.begin(), runs.end(),
              [](const StageBreakdown& a, const StageBreakdown& b) {
                return a.total < b.total;
              });
    const StageBreakdown& s = runs[runs.size() / 2];
    json["stages_ns"] = {{"plan", s.plan},
                         {"dictionary", s.dictionary},
                         {"decode", s.decode},
                         {"intersect", s.intersect},
                         {"window", s.window},
                         {"priority", s.priority},
                         {"verify", s.verify},
                         {"sample", s.sample},
                         {"rank", s.rank},
                         {"encode", s.encode},
                         {"other", s.other},
                         {"total", s.total},
                         {"worker_scan", s.worker_scan},
                         {"sample_dictionary", s.sample_dictionary},
                         {"sample_evaluate", s.sample_evaluate},
                         {"sample_decode", s.sample_decode},
                         {"sample_intersect", s.sample_intersect},
                         {"sample_verify", s.sample_verify}};
  }
  if (record.oracle.has_value()) {
    const OracleResult& o = *record.oracle;
    json["oracle"] = {{"complete", o.complete},
                      {"equal", o.equal},
                      {"mismatch", o.mismatch()},
                      {"files", o.oracle_files},
                      {"common_files", o.common_files},
                      {"matching_files", o.oracle_matching_files},
                      {"matching_lines", o.oracle_matching_lines},
                      {"estimated_files", o.matching_files},
                      {"estimated_lines", o.matching_lines},
                      {"oracle_ns", o.oracle_ns}};
  }
  if (record.k.has_value()) {
    json["k"] = *record.k;
  }
  if (record.results.has_value()) {
    nlohmann::json results = nlohmann::json::array();
    for (const LineRef& line : *record.results) {
      results.push_back({line.path, line.line});
    }
    json["results"] = std::move(results);
  }
  return json;
}

}  // namespace ferret_bench

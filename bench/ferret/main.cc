// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include <sys/resource.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/commandlineflag.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/reflection.h"
#include "absl/flags/usage.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "bench/ferret/file_list.h"
#include "bench/ferret/query_set.h"
#include "bench/ferret/rpc_bench.h"
#include "bench/ferret/search_bench.h"
#include "nlohmann/json.hpp"

#include "ferret/base/env.h"
#include "ferret/base/thread_pool.h"
#include "ferret/engine/engine.h"
#include "ferret/git/blob_hash.h"
#include "ferret/index/blob_registry.h"
#include "ferret/index/bulk_build.h"
#include "ferret/index/read_state.h"
#include "ferret/index/segment.h"
#include "ferret/index/segment_writer.h"
#include "ferret/index/version_set.h"
#include "ferret/postings/ids.h"
#include "ferret/rpc/json_codec.h"
#include "ferret/rpc/rpc_client.h"
#include "ferret/search/searcher.h"
#include "ferret/text/eligibility.h"
#include "ferret/workspace/path_table.h"
#include "ferret/workspace/view_state.h"

ABSL_FLAG(std::string, root, "", "Corpus root.");
ABSL_FLAG(std::string, files, "", "File list for build.");
ABSL_FLAG(std::string, out, "", "Output file for files or build.");
ABSL_FLAG(std::string, from_list, "",
          "For files: filter this already-eligible file list by --path_glob "
          "instead of listing --root. Reads no file content.");
ABSL_FLAG(std::string, name, "",
          "Repository name (defaults to root basename).");
ABSL_FLAG(int, threads, 8, "Build scan threads.");
ABSL_FLAG(bool, symbols, true, "Build: write the symbols section.");
ABSL_FLAG(int, symbol_threads, 0, "Build: symbol threads (0 = --threads).");
ABSL_FLAG(int, scan_threads, 0, "Build: scan threads (0 = default).");
ABSL_FLAG(int64_t, partition_postings, 0,
          "Build: postings per sorted run (0 = default).");
ABSL_FLAG(std::string, symbol_qos, "interactive",
          "Build: symbol QoS, interactive or utility.");
ABSL_FLAG(std::string, path_glob, "",
          "For files: keep only the paths this search-tool path_glob admits.");
ABSL_FLAG(std::string, policy, "zoekt", "Eligibility policy: zoekt or agent.");
ABSL_FLAG(std::string, index, "", "Segment for search.");
ABSL_FLAG(bool, view_edit, false,
          "Edit one shared view chunk before every query (single client). ");
ABSL_FLAG(bool, view, false,
          "Search through a cached synthetic view snapshot.");
ABSL_FLAG(std::string, queries, "", "JSONL query set.");
ABSL_FLAG(std::string, mode, "topk", "Search mode: topk or exhaustive.");
ABSL_FLAG(int, k, 20, "Maximum files in topk mode.");
ABSL_FLAG(int, context, 2, "Context lines.");
ABSL_FLAG(int64_t, max_matches, 1'000'000, "Exhaustive match cap.");
ABSL_FLAG(int, clients, 1, "Concurrent clients.");
ABSL_FLAG(int, search_threads, 1, "Query verification threads.");
ABSL_FLAG(int, full_run_ahead, 4,
          "Run-ahead grains per worker once the top-K heap is full.");
ABSL_FLAG(int64_t, max_verify_bytes, int64_t{128} << 20,
          "Segment verification budget per query in topk mode; exhaustive "
          "mode is always unlimited.");
ABSL_FLAG(int, warmup, 3, "Untimed passes.");
ABSL_FLAG(int, runs, 10, "Timed passes.");
ABSL_FLAG(uint64_t, seed, 1, "Query order seed.");
ABSL_FLAG(double, bigram_posting_share, 8,
          "SearcherOptions::max_bigram_posting_share.");
ABSL_FLAG(bool, line_cap_stop, true,
          "Stop a top-K search once the global line cap is spent.");
ABSL_FLAG(std::string, priority_restriction, "auto",
          "SearcherOptions::priority_restriction: auto, never or always.");
ABSL_FLAG(bool, priority_alternatives, true,
          "SearcherOptions::priority_alternatives.");
ABSL_FLAG(bool, priority_parallel, true, "SearcherOptions::priority_parallel.");
ABSL_FLAG(bool, diagnose_stop, false,
          "Print what blocked each query's rank stop on stderr (one thread, "
          "one client).");
ABSL_FLAG(bool, trace_docs, false,
          "Print every verified document of each query on stderr (exhaustive "
          "mode, one thread, one client).");
ABSL_FLAG(bool, full_results, false, "Include matching path and line pairs.");
// The configuration the search-time benchmark measures (bench/README.md).
ABSL_FLAG(bool, count_sample, true,
          "Top-K: estimated totals (SearchRequest::count_sample), as G1.");
ABSL_FLAG(bool, encode_response, true,
          "Include encoding the response in the timed region, as G1.");
ABSL_FLAG(bool, stages, false, "Record the stage breakdown of every query.");
ABSL_FLAG(bool, oracle, false,
          "After each query's first timed run, compare it with the brute-force "
          "top K (untimed); a complete response that differs fails the run.");
// Ablation switches: each turns one lever off.
ABSL_FLAG(bool, early_termination, true, "Ablation: the rank-stop bound.");
ABSL_FLAG(bool, name_aware_bound, true, "Ablation: the name-aware bound.");
ABSL_FLAG(bool, count_only, true, "Ablation: counting-only verification.");
ABSL_FLAG(
    int, count_sample_level, 3,
    "Ablation: how the count sample is drawn (SearcherOptions::"
    "count_sample_level): 0 reference, 1 bounded heap, 2 resumed frame, 3 "
    "Oid set kept small and frame split over workers.");
ABSL_FLAG(bool, parallel_verify, true, "Ablation: the verification pipeline.");
ABSL_FLAG(bool, scalar_decode, false, "Ablation: scalar posting decode.");
ABSL_FLAG(bool, scalar_intersect, false, "Ablation: scalar id-list merge.");
ABSL_FLAG(bool, skip_probe, true, "Ablation: skip-table probes.");
// The rpc subcommand: the serving path against the in-process search.
ABSL_FLAG(std::string, rpc_backend, "daemon",
          "rpc: daemon (RpcClient and a spawned `ferret serve`) or engine "
          "(Engine in this process).");
ABSL_FLAG(std::string, rpc_binary, "", "rpc: the ferret executable to spawn.");
ABSL_FLAG(std::string, rpc_index_dir, "", "rpc: directory for the index.");
ABSL_FLAG(int64_t, chunk_bytes, 0,
          "Build: chunk size target in bytes (0 = default 8 KiB).");

namespace ferret_bench {
namespace {

// One report is emitted before a build with skipped files fails parity.
void PrintJson(const nlohmann::json& value) {
  std::cout << ferret::SerializeJson(value) << '\n';
}

absl::StatusOr<ferret::EligibilityOptions> PolicyOptions(
    absl::string_view policy) {
  ferret::EligibilityOptions options;
  if (policy == "zoekt") {
    options.policy = ferret::EligibilityPolicy::kZoekt;
  } else if (policy != "agent") {
    return absl::InvalidArgumentError("--policy must be zoekt or agent");
  }
  return options;
}

// Only populated reasons appear, matching the Zoekt build report's skipped
// map (with counts instead of paths for Ferret).
template <typename Counts>
nlohmann::json SkippedJson(const Counts& counts) {
  nlohmann::json skipped = nlohmann::json::object();
  for (int i = 1; i < ferret::kNumSkipReasons; ++i) {
    if (counts[i] != 0) {
      skipped[std::string(ferret::SkipReasonName(
          static_cast<ferret::SkipReason>(i)))] = counts[i];
    }
  }
  return skipped;
}

// `files --from_list`: filters an eligible listing, which a previous `files`
// wrote, by the path glob. Equal to `files --root --path_glob` on the same
// tree, without reading any file.
absl::Status RunFilesFromList(ferret::Env& env) {
  const std::string out = absl::GetFlag(FLAGS_out);
  const std::string from_list = absl::GetFlag(FLAGS_from_list);
  const std::string path_glob = absl::GetFlag(FLAGS_path_glob);
  ABSL_ASSIGN_OR_RETURN(const std::string text, env.ReadFile(from_list));
  ABSL_ASSIGN_OR_RETURN(const std::vector<std::string> eligible,
                        ParseFileList(text));
  ABSL_ASSIGN_OR_RETURN(const std::vector<std::string> admitted,
                        FilterByPathGlob(eligible, path_glob));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<ferret::WritableFile> output,
                        env.NewWritableFile(out));
  ABSL_RETURN_IF_ERROR(output->Append(FormatFileList(admitted)));
  ABSL_RETURN_IF_ERROR(output->Close());
  PrintJson({{"engine", "ferret"},
             {"from_list", from_list},
             {"files_eligible", eligible.size()},
             {"path_glob", path_glob},
             {"files_written", admitted.size()}});
  return absl::OkStatus();
}

absl::Status RunFiles(ferret::Env& env) {
  const std::string root = absl::GetFlag(FLAGS_root);
  const std::string out = absl::GetFlag(FLAGS_out);
  const std::string policy = absl::GetFlag(FLAGS_policy);
  ABSL_ASSIGN_OR_RETURN(const ferret::EligibilityOptions options,
                        PolicyOptions(policy));
  ABSL_ASSIGN_OR_RETURN(EligibleFiles files,
                        ListEligibleFiles(env, root, options));
  const int64_t files_eligible = static_cast<int64_t>(files.paths.size());
  const std::string path_glob = absl::GetFlag(FLAGS_path_glob);
  if (!path_glob.empty()) {
    ABSL_ASSIGN_OR_RETURN(files.paths,
                          FilterByPathGlob(files.paths, path_glob));
  }
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<ferret::WritableFile> output,
                        env.NewWritableFile(out));
  ABSL_RETURN_IF_ERROR(output->Append(FormatFileList(files.paths)));
  ABSL_RETURN_IF_ERROR(output->Close());
  PrintJson({{"engine", "ferret"},
             {"policy", policy},
             {"files_listed", files.files_listed},
             {"files_eligible", files_eligible},
             {"path_glob", path_glob},
             {"files_written", files.paths.size()},
             {"skipped", SkippedJson(files.files_skipped)}});
  return absl::OkStatus();
}

std::string RepoName(absl::string_view name, absl::string_view root) {
  if (!name.empty()) return std::string(name);
  while (root.size() > 1 && root.back() == '/') root.remove_suffix(1);
  return std::string(root.substr(
      root.rfind('/') == absl::string_view::npos ? 0 : root.rfind('/') + 1));
}

// getrusage records the process's high-water RSS, not a per-build delta.
int64_t PeakRSSBytes(const rusage& usage) {
#ifdef __APPLE__
  return usage.ru_maxrss;
#else
  return static_cast<int64_t>(usage.ru_maxrss) * 1024;
#endif
}

double CPUSeconds(const timeval& time) {
  return static_cast<double>(time.tv_sec) +
         static_cast<double>(time.tv_usec) / 1'000'000;
}

absl::Status RunBuild(ferret::Env& env) {
  const std::string root = absl::GetFlag(FLAGS_root);
  const std::string out = absl::GetFlag(FLAGS_out);
  const std::string policy = absl::GetFlag(FLAGS_policy);
  const int threads = absl::GetFlag(FLAGS_threads);
  if (threads < 1) {
    return absl::InvalidArgumentError("--threads must be positive");
  }
  ABSL_ASSIGN_OR_RETURN(const ferret::EligibilityOptions eligibility,
                        PolicyOptions(policy));
  ABSL_ASSIGN_OR_RETURN(const std::string text,
                        env.ReadFile(absl::GetFlag(FLAGS_files)));
  ABSL_ASSIGN_OR_RETURN(const std::vector<std::string> paths,
                        ParseFileList(text));
  rusage before = {};
  if (getrusage(RUSAGE_SELF, &before) != 0) {
    return absl::InternalError("getrusage before build failed");
  }
  const absl::Time start = absl::Now();
  ferret::BulkBuildOptions options = {.eligibility = eligibility,
                                      .num_threads = threads};
  options.writer.write_symbols = absl::GetFlag(FLAGS_symbols);
  if (absl::GetFlag(FLAGS_chunk_bytes) > 0) {
    options.writer.chunk_bytes = absl::GetFlag(FLAGS_chunk_bytes);
  }
  const std::string symbol_qos = absl::GetFlag(FLAGS_symbol_qos);
  if (symbol_qos != "interactive" && symbol_qos != "utility") {
    return absl::InvalidArgumentError(
        "--symbol_qos must be interactive or utility");
  }
  if (absl::GetFlag(FLAGS_partition_postings) > (int64_t{1} << 28)) {
    return absl::InvalidArgumentError("--partition_postings is at most 2^28");
  }
  if (absl::GetFlag(FLAGS_partition_postings) > 0) {
    options.writer.max_partition_postings =
        static_cast<size_t>(absl::GetFlag(FLAGS_partition_postings));
  }
  options.symbol_threads = absl::GetFlag(FLAGS_symbol_threads);
  options.scan_threads = absl::GetFlag(FLAGS_scan_threads);
  options.symbol_qos = symbol_qos == "interactive"
                           ? ferret::ThreadQos::kUserInteractive
                           : ferret::ThreadQos::kUtility;
  ABSL_ASSIGN_OR_RETURN(const ferret::BulkBuildStats stats,
                        ferret::BuildSegment(env, root, paths, out, options));
  const absl::Duration wall = absl::Now() - start;
  rusage after = {};
  if (getrusage(RUSAGE_SELF, &after) != 0) {
    return absl::InternalError("getrusage after build failed");
  }
  const double user = CPUSeconds(after.ru_utime) - CPUSeconds(before.ru_utime);
  const double sys = CPUSeconds(after.ru_stime) - CPUSeconds(before.ru_stime);
  const int64_t postings = static_cast<int64_t>(stats.segment.postings_bytes);
  const int64_t content = static_cast<int64_t>(stats.segment.content_bytes);
  const int64_t index_bytes = static_cast<int64_t>(stats.segment.file_bytes);
  nlohmann::json skipped = SkippedJson(stats.files_skipped);
  if (stats.files_unreadable != 0) {
    skipped["unreadable"] = stats.files_unreadable;
  }
  const auto sec = [](absl::Duration d) { return absl::ToDoubleSeconds(d); };
  const auto ns = [](int64_t n) { return static_cast<double>(n) / 1e9; };
  const ferret::SegmentWriterTimings& w = stats.writer;
  const nlohmann::json phases = {
      {"list_wall", sec(stats.list_time)},
      {"scan_wall", sec(stats.scan_time)},
      {"sort_wall", sec(stats.sort_time)},
      {"write_wall", sec(stats.write_time)},
      {"symbols_wall", sec(stats.symbols_time)},
      {"symbols_wait", sec(stats.symbols_wait_time)},
      {"finish_wall", sec(stats.finish_time)},
      {"partitions", stats.partitions},
      {"peak_rss_after_scan", stats.peak_rss_after_scan},
      {"peak_rss_after_assemble", stats.peak_rss_after_assemble},
      {"peak_rss_after_finish", stats.peak_rss_after_finish},
      {"assemble_wall", sec(stats.assemble_time)},
      {"pack_thread", sec(stats.pack_thread_time)},
      {"symbols_read_thread", sec(stats.symbols_read_thread_time)},
      {"scan_read_thread", sec(stats.scan_read_thread_time)},
      {"scan_prepare_thread", sec(stats.scan_prepare_thread_time)},
      {"symbols_thread", sec(stats.symbols_thread_time)},
      {"reread_thread", sec(stats.reread_thread_time)},
      {"verify_hash_thread", sec(stats.verify_hash_thread_time)},
      {"chunking_thread", ns(w.chunking_ns)},
      {"trigrams_thread", ns(w.trigrams_ns)},
      {"postings_add_thread", ns(w.postings_add_ns)},
      {"code_tokens_thread", ns(w.code_tokens_ns)},
      {"content_thread", ns(w.content_ns)},
      {"writer_symbols_thread", ns(w.symbols_ns)},
      {"finish_tail_sections", ns(w.tail_sections_ns)},
      {"finish_symbols_write", ns(w.symbols_write_ns)},
      {"finish_code_tokens_write", ns(w.code_tokens_write_ns)},
      {"finish_lists_merge", ns(w.lists_merge_ns)},
      {"finish_lists_encode", ns(w.lists_encode_ns)},
      {"finish_dictionary", ns(w.dictionary_ns)},
      {"finish_sync", ns(w.sync_ns)}};
  PrintJson({{"engine", "ferret"},
             {"phases", phases},
             {"engine_version",
              absl::StrCat("fseg-v", ferret::Segment::FormatVersion())},
             {"name", RepoName(absl::GetFlag(FLAGS_name), root)},
             {"files_listed", stats.files_listed},
             {"files_indexed", stats.files_indexed},
             {"bytes_indexed", stats.bytes_indexed},
             {"skipped", skipped},
             {"shards", 1},
             {"index_bytes", index_bytes},
             {"wall_seconds", absl::ToDoubleSeconds(wall)},
             {"user_seconds", user},
             {"sys_seconds", sys},
             {"cpu_seconds", user + sys},
             {"peak_rss_bytes", PeakRSSBytes(after)},
             {"sections",
              {{"postings_bytes", postings},
               {"content_bytes", content},
               {"other_bytes", index_bytes - postings - content}}},
             {"options",
              {{"policy", policy},
               {"size_max", eligibility.max_file_bytes},
               {"trigram_max", eligibility.max_distinct_trigrams},
               {"chunk_bytes", options.writer.chunk_bytes},
               {"threads", threads}}}});
  if (stats.files_indexed != stats.files_listed) {
    return absl::FailedPreconditionError(
        "some listed files were skipped or unreadable");
  }
  return absl::OkStatus();
}

absl::StatusOr<SearchBenchConfig> SearchConfigFromFlags() {
  SearchBenchConfig config;
  const std::string mode = absl::GetFlag(FLAGS_mode);
  if (mode == "exhaustive") {
    config.mode = SearchMode::kExhaustive;
  } else if (mode != "topk") {
    return absl::InvalidArgumentError("--mode must be exhaustive or topk");
  }
  config.k = absl::GetFlag(FLAGS_k);
  config.context_lines = absl::GetFlag(FLAGS_context);
  config.max_matches = absl::GetFlag(FLAGS_max_matches);
  config.clients = absl::GetFlag(FLAGS_clients);
  config.search_threads = absl::GetFlag(FLAGS_search_threads);
  config.full_run_ahead = absl::GetFlag(FLAGS_full_run_ahead);
  config.max_verify_bytes = absl::GetFlag(FLAGS_max_verify_bytes);
  config.warmup_passes = absl::GetFlag(FLAGS_warmup);
  config.timed_runs = absl::GetFlag(FLAGS_runs);
  config.seed = absl::GetFlag(FLAGS_seed);
  config.full_results = absl::GetFlag(FLAGS_full_results);
  config.diagnose_stop = absl::GetFlag(FLAGS_diagnose_stop);
  config.trace_docs = absl::GetFlag(FLAGS_trace_docs);
  config.line_cap_stop = absl::GetFlag(FLAGS_line_cap_stop);
  config.priority_parallel = absl::GetFlag(FLAGS_priority_parallel);
  config.priority_alternatives = absl::GetFlag(FLAGS_priority_alternatives);
  const std::string restriction = absl::GetFlag(FLAGS_priority_restriction);
  if (restriction == "auto") {
    config.priority_restriction = ferret::PriorityRestriction::kAuto;
  } else if (restriction == "never") {
    config.priority_restriction = ferret::PriorityRestriction::kNever;
  } else if (restriction == "always") {
    config.priority_restriction = ferret::PriorityRestriction::kAlways;
  } else {
    return absl::InvalidArgumentError(
        "--priority_restriction must be auto, never or always");
  }
  config.bigram_posting_share = absl::GetFlag(FLAGS_bigram_posting_share);
  config.count_sample = absl::GetFlag(FLAGS_count_sample);
  config.encode_response = absl::GetFlag(FLAGS_encode_response);
  config.stage_timing = absl::GetFlag(FLAGS_stages);
  config.oracle = absl::GetFlag(FLAGS_oracle);
  config.early_termination = absl::GetFlag(FLAGS_early_termination);
  config.name_aware_bound = absl::GetFlag(FLAGS_name_aware_bound);
  config.count_only = absl::GetFlag(FLAGS_count_only);
  config.count_sample_level = absl::GetFlag(FLAGS_count_sample_level);
  config.parallel_verify = absl::GetFlag(FLAGS_parallel_verify);
  config.scalar_decode = absl::GetFlag(FLAGS_scalar_decode);
  config.scalar_intersect = absl::GetFlag(FLAGS_scalar_intersect);
  config.skip_probe = absl::GetFlag(FLAGS_skip_probe);
  if ((config.mode == SearchMode::kTopK && config.k < 1) ||
      config.context_lines < 0 ||
      config.context_lines > ferret::kMaxContextLines ||
      config.max_matches < 1 || config.clients < 1 ||
      config.search_threads < 1 || config.full_run_ahead < 1 ||
      ((config.diagnose_stop || config.trace_docs) &&
       (config.search_threads != 1 || config.clients != 1)) ||
      config.max_verify_bytes < 1 || config.warmup_passes < 0 ||
      config.timed_runs < 1 ||
      config.warmup_passes >
          std::numeric_limits<int>::max() - config.timed_runs) {
    return absl::InvalidArgumentError("invalid search options");
  }
  return config;
}

absl::Status RunSearch(ferret::Env& env) {
  ABSL_ASSIGN_OR_RETURN(SearchBenchConfig config, SearchConfigFromFlags());
  ABSL_ASSIGN_OR_RETURN(const std::string text,
                        env.ReadFile(absl::GetFlag(FLAGS_queries)));
  ABSL_ASSIGN_OR_RETURN(const std::vector<QuerySpec> queries,
                        ParseQuerySet(text));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<ferret::Segment> segment,
                        ferret::Segment::Open(env, absl::GetFlag(FLAGS_index)));
  const std::shared_ptr<const ferret::Segment> indexed_segment =
      std::move(segment);
  // Give each document its hint path in the synthetic view. Both the view
  // and the process-local blob mapping are shared by every timed query.
  ferret::PathTable paths;
  ferret::ViewWriter writer;
  ferret::ReadState state;
  ferret::SearchView view{nullptr, paths};
  if (absl::GetFlag(FLAGS_view) || absl::GetFlag(FLAGS_view_edit)) {
    if (absl::GetFlag(FLAGS_view_edit) && config.clients != 1) {
      return absl::InvalidArgumentError("--view_edit requires --clients=1");
    }
    state.blobs = std::make_shared<ferret::BlobRegistry>();
    // Dummy blobs do not occur in the index, but live in the first chunk so
    // toggling their path invalidates names shared with indexed documents.
    const ferret::BlobId edit_blob = state.blobs->Intern(
        ferret::HashBlob("ferret benchmark view edit dummy"));
    auto version = std::make_shared<ferret::Version>();
    auto blob_of_doc = std::make_shared<std::vector<ferret::BlobId>>();
    blob_of_doc->reserve(indexed_segment->num_docs());
    std::vector<ferret::ViewEdit> edits;
    edits.reserve(indexed_segment->num_docs());
    for (ferret::DocId id = 0; id < indexed_segment->num_docs(); ++id) {
      const ferret::SegmentDoc doc = indexed_segment->doc(id);
      const ferret::BlobId blob = state.blobs->Intern(doc.oid);
      blob_of_doc->push_back(blob);
      ABSL_ASSIGN_OR_RETURN(const ferret::PathId path,
                            paths.Intern(doc.hint_path));
      edits.push_back({path, blob});
    }
    if (absl::GetFlag(FLAGS_view_edit)) {
      ABSL_ASSIGN_OR_RETURN(const ferret::PathId edit_path,
                            paths.Intern("ferret_benchmark_dummy_view_path"));
      edits.push_back({edit_path, edit_blob});
      config.edit_path = edit_path;
      config.edit_blob = edit_blob;
      config.view_writer = &writer;
    }
    view.state = writer.ApplyBatch(edits);
    auto docs_of_blob =
        std::make_shared<const ferret::DocsOfBlob>(*blob_of_doc);
    version->segments.push_back({.segment = indexed_segment,
                                 .blob_of_doc = std::move(blob_of_doc),
                                 .docs_of_blob = std::move(docs_of_blob)});
    state.version = std::move(version);
    config.state = &state;
    config.view = &view;
  }
  ABSL_ASSIGN_OR_RETURN(const std::vector<SearchRecord> records,
                        RunSearchBench(*indexed_segment, queries, config));
  std::string mismatches;
  for (const SearchRecord& record : records) {
    PrintJson(SearchRecordToJson(record));
    if (record.oracle.has_value() && record.oracle->mismatch()) {
      absl::StrAppend(&mismatches, " ", record.id);
    }
  }
  if (!mismatches.empty()) {
    return absl::DataLossError(absl::StrCat(
        "complete responses differ from the oracle's:", mismatches));
  }
  return absl::OkStatus();
}

absl::Status RunRpc(ferret::Env& env) {
  ABSL_ASSIGN_OR_RETURN(SearchBenchConfig config, SearchConfigFromFlags());
  ABSL_ASSIGN_OR_RETURN(const std::string text,
                        env.ReadFile(absl::GetFlag(FLAGS_queries)));
  ABSL_ASSIGN_OR_RETURN(const std::vector<QuerySpec> queries,
                        ParseQuerySet(text));
  const std::string kind = absl::GetFlag(FLAGS_rpc_backend);
  const std::string index_dir = absl::GetFlag(FLAGS_rpc_index_dir);
  std::vector<SearchRecord> records;
  if (kind == "engine") {
    ferret::Engine engine(env, index_dir);
    ABSL_RETURN_IF_ERROR(engine.Register(absl::GetFlag(FLAGS_root)));
    const ferret::EngineBackend backend(engine);
    ABSL_ASSIGN_OR_RETURN(
        records, RunBackendBench(backend, backend.Roots().front(), "engine",
                                 queries, config));
  } else if (kind == "daemon") {
    ferret::RpcClient client(absl::GetFlag(FLAGS_rpc_binary), index_dir);
    ABSL_RETURN_IF_ERROR(client.Register(absl::GetFlag(FLAGS_root)));
    absl::StatusOr<std::vector<SearchRecord>> run = RunBackendBench(
        client, client.Roots().front(), "daemon", queries, config);
    // Leave no daemon behind, whether or not the run worked.
    ABSL_RETURN_IF_ERROR(client.Drain());
    ABSL_ASSIGN_OR_RETURN(records, std::move(run));
  } else {
    return absl::InvalidArgumentError("--rpc_backend must be daemon or engine");
  }
  for (const SearchRecord& record : records) {
    PrintJson(SearchRecordToJson(record));
  }
  return absl::OkStatus();
}

// Abseil exits with code 1 on malformed flag values. Check the syntax before
// parsing so invocation errors return the same status as missing flags.
absl::Status CheckFlagSyntax(int argc, char** argv) {
  for (int i = 2; i < argc; ++i) {
    absl::string_view arg(argv[i]);
    if (arg == "--") break;
    if (!arg.starts_with('-')) continue;
    arg.remove_prefix(arg.starts_with("--") ? 2 : 1);
    const size_t equals = arg.find('=');
    const absl::string_view name = arg.substr(0, equals);
    const absl::CommandLineFlag* flag = absl::FindCommandLineFlag(name);
    bool negated = false;
    if (flag == nullptr && name.starts_with("no")) {
      flag = absl::FindCommandLineFlag(name.substr(2));
      negated = flag != nullptr && flag->IsOfType<bool>();
    }
    if (flag == nullptr) {
      return absl::InvalidArgumentError(absl::StrCat("unknown flag --", name));
    }
    if (negated && equals != absl::string_view::npos) {
      return absl::InvalidArgumentError("negated boolean flag has a value");
    }
    if (negated ||
        (flag->IsOfType<bool>() && equals == absl::string_view::npos)) {
      continue;
    }
    absl::string_view value;
    if (equals != absl::string_view::npos) {
      value = arg.substr(equals + 1);
    } else if (++i < argc) {
      value = argv[i];
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("missing value for --", name));
    }
    int integer = 0;
    int64_t long_integer = 0;
    uint64_t unsigned_integer = 0;
    bool boolean = false;
    if ((flag->IsOfType<int>() && !absl::SimpleAtoi(value, &integer)) ||
        (flag->IsOfType<int64_t>() &&
         !absl::SimpleAtoi(value, &long_integer)) ||
        (flag->IsOfType<uint64_t>() &&
         !absl::SimpleAtoi(value, &unsigned_integer)) ||
        (flag->IsOfType<bool>() && !absl::SimpleAtob(value, &boolean))) {
      return absl::InvalidArgumentError(
          absl::StrCat("invalid value for --", name));
    }
  }
  return absl::OkStatus();
}

int Main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: ferret_bench <files|build|search|rpc> [flags]\n";
    return 2;
  }
  const std::string command = argv[1];
  if (command != "files" && command != "build" && command != "search" &&
      command != "rpc") {
    std::cerr << "ferret_bench: unknown subcommand: " << command << '\n';
    return 2;
  }
  if (const absl::Status syntax = CheckFlagSyntax(argc, argv); !syntax.ok()) {
    std::cerr << "ferret_bench: " << syntax << '\n';
    return 2;
  }
  // Abseil accepts flags after the subcommand; remove it before parsing.
  for (int i = 1; i < argc - 1; ++i) argv[i] = argv[i + 1];
  std::vector<char*> args;
  std::vector<absl::UnrecognizedFlag> unknown;
  absl::ParseAbseilFlagsOnly(argc - 1, argv, args, unknown);
  if (!unknown.empty()) {
    absl::ReportUnrecognizedFlags(unknown);
    return 2;
  }
  if (args.size() != 1) {
    std::cerr << "ferret_bench: unexpected positional argument\n";
    return 2;
  }
  const auto required = [&](absl::string_view name, const std::string& value) {
    if (!value.empty()) return true;
    std::cerr << "ferret_bench " << command << ": --" << name
              << " is required\n";
    return false;
  };
  const bool from_list =
      command == "files" && !absl::GetFlag(FLAGS_from_list).empty();
  if (from_list) {
    if (!required("path_glob", absl::GetFlag(FLAGS_path_glob)) ||
        !required("out", absl::GetFlag(FLAGS_out))) {
      return 2;
    }
    if (!absl::GetFlag(FLAGS_root).empty()) {
      std::cerr << "ferret_bench files: --from_list and --root are exclusive\n";
      return 2;
    }
  } else if (command == "files" || command == "build") {
    if (!required("root", absl::GetFlag(FLAGS_root)) ||
        !required("out", absl::GetFlag(FLAGS_out))) {
      return 2;
    }
  }
  if (command == "build" && !required("files", absl::GetFlag(FLAGS_files))) {
    return 2;
  }
  if (command == "search" &&
      (!required("index", absl::GetFlag(FLAGS_index)) ||
       !required("queries", absl::GetFlag(FLAGS_queries)))) {
    return 2;
  }
  if (command == "rpc" &&
      (!required("root", absl::GetFlag(FLAGS_root)) ||
       !required("queries", absl::GetFlag(FLAGS_queries)) ||
       (absl::GetFlag(FLAGS_rpc_backend) == "daemon" &&
        !required("rpc_binary", absl::GetFlag(FLAGS_rpc_binary))))) {
    return 2;
  }
  if (command == "files" || command == "build") {
    if (!PolicyOptions(absl::GetFlag(FLAGS_policy)).ok() ||
        (command == "build" && absl::GetFlag(FLAGS_threads) < 1)) {
      std::cerr << "ferret_bench: invalid build policy or thread count\n";
      return 2;
    }
  }
  if (command == "search" || command == "rpc") {
    const absl::StatusOr<SearchBenchConfig> config = SearchConfigFromFlags();
    if (!config.ok()) {
      std::cerr << "ferret_bench: " << config.status() << '\n';
      return 2;
    }
  }
  ferret::Env& env = *ferret::Env::Default();
  absl::Status status;
  if (command == "files") {
    status = from_list ? RunFilesFromList(env) : RunFiles(env);
  }
  if (command == "build") status = RunBuild(env);
  if (command == "search") status = RunSearch(env);
  if (command == "rpc") status = RunRpc(env);
  if (!status.ok()) {
    std::cerr << "ferret_bench " << command << ": " << status << '\n';
    return 1;
  }
  return 0;
}

}  // namespace
}  // namespace ferret_bench

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage("ferret_bench <files|build|search|rpc> [flags]");
  absl::DisableFlagfileAndEnvParsing();
  return ferret_bench::Main(argc, argv);
}

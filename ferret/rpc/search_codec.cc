// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Checked wire conversions for the daemon protocol. Hex encodes every byte
// string, preserving worktree paths and source text without UTF-8 repair.
#include "ferret/rpc/search_codec.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "nlohmann/json.hpp"

#include "ferret/engine/engine.h"
#include "ferret/git/oid.h"
#include "ferret/index/rank_key.h"
#include "ferret/rpc/json_codec.h"
#include "ferret/search/search_types.h"
#include "ferret/text/lexer.h"
#include "ferret/workspace/workspace.h"

namespace ferret {
namespace {

std::string Hex(const std::string& bytes) {
  return absl::BytesToHexString(bytes);
}

absl::StatusOr<std::string> Unhex(const JsonView& value,
                                  absl::string_view field) {
  ABSL_ASSIGN_OR_RETURN(std::string hex, value.GetString(field));
  if (hex.size() % 2 != 0) return absl::InvalidArgumentError("odd hex length");
  std::string bytes;
  if (!absl::HexStringToBytes(hex, &bytes)) {
    return absl::InvalidArgumentError("invalid hex string");
  }
  return bytes;
}

nlohmann::json EncodeLine(const ResultLine& line) {
  return {{"number", line.number},
          {"text_hex", Hex(line.text)},
          {"is_match", line.is_match},
          {"match_begin", line.match_begin},
          {"match_end", line.match_end},
          {"match_class", static_cast<int>(line.match_class)}};
}

absl::StatusOr<ResultLine> DecodeLine(const JsonView& value) {
  ResultLine line;
  ABSL_ASSIGN_OR_RETURN(line.number, value.GetInt64("number"));
  ABSL_ASSIGN_OR_RETURN(line.text, Unhex(value, "text_hex"));
  ABSL_ASSIGN_OR_RETURN(line.is_match, value.GetBool("is_match"));
  ABSL_ASSIGN_OR_RETURN(uint64_t begin, value.GetUint64("match_begin"));
  ABSL_ASSIGN_OR_RETURN(uint64_t end, value.GetUint64("match_end"));
  if (begin > line.text.size() || end > line.text.size() || begin > end) {
    return absl::InvalidArgumentError("invalid match range");
  }
  line.match_begin = begin;
  line.match_end = end;
  ABSL_ASSIGN_OR_RETURN(int64_t kind, value.GetInt64("match_class"));
  if (kind < 0 || kind > static_cast<int>(MatchClass::kString)) {
    return absl::InvalidArgumentError("invalid match class");
  }
  line.match_class = static_cast<MatchClass>(kind);
  return line;
}

nlohmann::json EncodeFile(const FileResult& file) {
  nlohmann::json lines = nlohmann::json::array();
  for (const ResultLine& line : file.lines) lines.push_back(EncodeLine(line));
  return {{"path_hex", Hex(file.path)},
          {"oid", file.oid.ToHex()},
          {"tier", static_cast<int>(file.tier)},
          {"prior", file.prior},
          {"name_matches", file.name_matches},
          {"score", file.score},
          {"num_matching_lines", file.num_matching_lines},
          {"lines", std::move(lines)}};
}

absl::StatusOr<FileResult> DecodeFile(const JsonView& value) {
  FileResult file;
  ABSL_ASSIGN_OR_RETURN(file.path, Unhex(value, "path_hex"));
  ABSL_ASSIGN_OR_RETURN(std::string oid, value.GetString("oid"));
  ABSL_ASSIGN_OR_RETURN(file.oid, Oid::FromHex(oid));
  ABSL_ASSIGN_OR_RETURN(int64_t tier, value.GetInt64("tier"));
  if (tier < 0 || tier > static_cast<int>(RankTier::kLongLines)) {
    return absl::InvalidArgumentError("invalid rank tier");
  }
  file.tier = static_cast<RankTier>(tier);
  ABSL_ASSIGN_OR_RETURN(uint64_t prior, value.GetUint64("prior"));
  if (prior > UINT16_MAX) return absl::InvalidArgumentError("invalid prior");
  file.prior = prior;
  ABSL_ASSIGN_OR_RETURN(file.name_matches, value.GetBool("name_matches"));
  ABSL_ASSIGN_OR_RETURN(file.score, value.GetInt64("score"));
  ABSL_ASSIGN_OR_RETURN(file.num_matching_lines,
                        value.GetInt64("num_matching_lines"));
  ABSL_ASSIGN_OR_RETURN(JsonView lines, value.GetArray("lines"));
  for (size_t i = 0; i < lines.size(); ++i) {
    ABSL_ASSIGN_OR_RETURN(ResultLine line, DecodeLine(lines.Element(i)));
    file.lines.push_back(std::move(line));
  }
  return file;
}

nlohmann::json EncodeStats(const SearchStats& stats) {
  return {{"candidate_chunks", stats.candidate_chunks},
          {"view_filtered_chunks", stats.view_filtered_chunks},
          {"verified_chunks", stats.verified_chunks},
          {"verified_bytes", stats.verified_bytes},
          {"priority_documents", stats.priority_documents},
          {"parallel_grains", stats.parallel_grains},
          {"inconsistent_files", stats.inconsistent_files},
          {"compile_ns", absl::ToInt64Nanoseconds(stats.compile_time)},
          {"evaluate_ns", absl::ToInt64Nanoseconds(stats.evaluate_time)},
          {"verify_ns", absl::ToInt64Nanoseconds(stats.verify_time)}};
}

absl::StatusOr<SearchStats> DecodeStats(const JsonView& value) {
  SearchStats stats;
  ABSL_ASSIGN_OR_RETURN(stats.candidate_chunks,
                        value.GetInt64("candidate_chunks"));
  ABSL_ASSIGN_OR_RETURN(stats.view_filtered_chunks,
                        value.GetInt64("view_filtered_chunks"));
  ABSL_ASSIGN_OR_RETURN(stats.verified_chunks,
                        value.GetInt64("verified_chunks"));
  ABSL_ASSIGN_OR_RETURN(stats.verified_bytes, value.GetInt64("verified_bytes"));
  ABSL_ASSIGN_OR_RETURN(stats.priority_documents,
                        value.GetInt64("priority_documents"));
  ABSL_ASSIGN_OR_RETURN(stats.parallel_grains,
                        value.GetInt64("parallel_grains"));
  ABSL_ASSIGN_OR_RETURN(stats.inconsistent_files,
                        value.GetInt64("inconsistent_files"));
  ABSL_ASSIGN_OR_RETURN(int64_t compile, value.GetInt64("compile_ns"));
  ABSL_ASSIGN_OR_RETURN(int64_t evaluate, value.GetInt64("evaluate_ns"));
  ABSL_ASSIGN_OR_RETURN(int64_t verify, value.GetInt64("verify_ns"));
  stats.compile_time = absl::Nanoseconds(compile);
  stats.evaluate_time = absl::Nanoseconds(evaluate);
  stats.verify_time = absl::Nanoseconds(verify);
  return stats;
}

}  // namespace

nlohmann::json EncodeSearchRequest(const SearchRequest& value) {
  return {{"pattern_hex", Hex(value.pattern)},
          {"regex", value.regex},
          {"case_sensitive", value.case_sensitive},
          {"path_glob_hex", Hex(value.path_glob)},
          {"context_lines", value.context_lines},
          {"max_lines_per_file", value.max_lines_per_file},
          {"max_lines", value.max_lines},
          {"max_files", value.max_files},
          {"max_verify_bytes", value.max_verify_bytes},
          {"exact_counts", value.exact_counts},
          {"count_sample", value.count_sample}};
}

absl::StatusOr<SearchRequest> DecodeSearchRequest(const JsonView& value) {
  SearchRequest request;
  ABSL_ASSIGN_OR_RETURN(request.pattern, Unhex(value, "pattern_hex"));
  ABSL_ASSIGN_OR_RETURN(request.regex, value.GetBool("regex"));
  ABSL_ASSIGN_OR_RETURN(request.case_sensitive,
                        value.GetBool("case_sensitive"));
  ABSL_ASSIGN_OR_RETURN(request.path_glob, Unhex(value, "path_glob_hex"));
  ABSL_ASSIGN_OR_RETURN(int64_t context, value.GetInt64("context_lines"));
  ABSL_ASSIGN_OR_RETURN(int64_t per_file, value.GetInt64("max_lines_per_file"));
  if (context < 0 || context > INT32_MAX || per_file < 0 ||
      per_file > INT32_MAX) {
    return absl::InvalidArgumentError("invalid line limit");
  }
  request.context_lines = static_cast<int>(context);
  request.max_lines_per_file = static_cast<int>(per_file);
  ABSL_ASSIGN_OR_RETURN(request.max_lines, value.GetInt64("max_lines"));
  ABSL_ASSIGN_OR_RETURN(request.max_files, value.GetInt64("max_files"));
  ABSL_ASSIGN_OR_RETURN(request.max_verify_bytes,
                        value.GetInt64("max_verify_bytes"));
  ABSL_ASSIGN_OR_RETURN(request.exact_counts, value.GetBool("exact_counts"));
  ABSL_ASSIGN_OR_RETURN(request.count_sample, value.GetBool("count_sample"));
  return request;
}

nlohmann::json EncodeSearchConsistency(const SearchConsistency& value) {
  return {{"mode", static_cast<int>(value.mode)},
          {"min_generation", value.min_generation},
          {"timeout_ns", absl::ToInt64Nanoseconds(value.timeout)}};
}

absl::StatusOr<SearchConsistency> DecodeSearchConsistency(
    const JsonView& value) {
  SearchConsistency consistency;
  ABSL_ASSIGN_OR_RETURN(int64_t mode, value.GetInt64("mode"));
  if (mode < 0 || mode > static_cast<int>(SearchConsistencyMode::kSync)) {
    return absl::InvalidArgumentError("invalid consistency mode");
  }
  consistency.mode = static_cast<SearchConsistencyMode>(mode);
  ABSL_ASSIGN_OR_RETURN(consistency.min_generation,
                        value.GetUint64("min_generation"));
  ABSL_ASSIGN_OR_RETURN(int64_t timeout, value.GetInt64("timeout_ns"));
  if (timeout < 0) return absl::InvalidArgumentError("negative timeout");
  consistency.timeout = absl::Nanoseconds(timeout);
  return consistency;
}

nlohmann::json EncodeWorkspaceSearchResult(const WorkspaceSearchResult& value) {
  nlohmann::json files = nlohmann::json::array();
  for (const FileResult& file : value.response.files) {
    files.push_back(EncodeFile(file));
  }
  const SearchResponse& response = value.response;
  return {{"files", std::move(files)},
          {"num_matching_files", response.num_matching_files},
          {"num_matching_lines", response.num_matching_lines},
          {"complete", response.complete},
          {"terminated_early", response.terminated_early},
          {"counts_estimated", response.counts_estimated},
          {"count_sample_size", response.count_sample_size},
          {"match_classes", response.match_classes},
          {"token_precision", response.token_precision},
          {"verified_fraction", response.verified_fraction},
          {"stats", EncodeStats(response.stats)},
          {"stale", value.stale},
          {"generation", value.generation}};
}

absl::StatusOr<WorkspaceSearchResult> DecodeWorkspaceSearchResult(
    const JsonView& value) {
  WorkspaceSearchResult result;
  ABSL_ASSIGN_OR_RETURN(JsonView files, value.GetArray("files"));
  for (size_t i = 0; i < files.size(); ++i) {
    ABSL_ASSIGN_OR_RETURN(FileResult file, DecodeFile(files.Element(i)));
    result.response.files.push_back(std::move(file));
  }
  ABSL_ASSIGN_OR_RETURN(result.response.num_matching_files,
                        value.GetInt64("num_matching_files"));
  ABSL_ASSIGN_OR_RETURN(result.response.num_matching_lines,
                        value.GetInt64("num_matching_lines"));
  ABSL_ASSIGN_OR_RETURN(result.response.complete, value.GetBool("complete"));
  ABSL_ASSIGN_OR_RETURN(result.response.terminated_early,
                        value.GetBool("terminated_early"));
  ABSL_ASSIGN_OR_RETURN(result.response.counts_estimated,
                        value.GetBool("counts_estimated"));
  ABSL_ASSIGN_OR_RETURN(result.response.count_sample_size,
                        value.GetInt64("count_sample_size"));
  const std::optional<JsonView> classes = value.Find("match_classes");
  if (!classes || !classes->value().is_array() ||
      classes->value().size() != 4) {
    return absl::InvalidArgumentError("invalid match classes");
  }
  for (size_t i = 0; i < 4; ++i) {
    if (!classes->value()[i].is_number()) {
      return absl::InvalidArgumentError("invalid match class count");
    }
    result.response.match_classes[i] = classes->value()[i].get<double>();
  }
  const std::optional<JsonView> precision = value.Find("token_precision");
  if (!precision || !precision->value().is_number()) {
    return absl::InvalidArgumentError("invalid token precision");
  }
  result.response.token_precision = precision->value().get<double>();
  ABSL_ASSIGN_OR_RETURN(JsonView fraction, value.GetObject("stats"));
  ABSL_ASSIGN_OR_RETURN(result.response.stats, DecodeStats(fraction));
  const std::optional<JsonView> ratio = value.Find("verified_fraction");
  if (!ratio || !ratio->value().is_number()) {
    return absl::InvalidArgumentError("invalid verified_fraction");
  }
  result.response.verified_fraction = ratio->value().get<double>();
  ABSL_ASSIGN_OR_RETURN(result.stale, value.GetBool("stale"));
  ABSL_ASSIGN_OR_RETURN(result.generation, value.GetUint64("generation"));
  return result;
}

nlohmann::json EncodeWorktreeIndexStatus(const WorktreeIndexStatus& value) {
  return {{"root_hex", Hex(value.root)},
          {"file_count", value.file_count},
          {"blob_count", value.blob_count},
          {"index_bytes", value.index_bytes},
          {"segment_count", value.segment_count},
          {"pending_symbols", value.pending_symbols},
          {"failed_symbols", value.failed_symbols},
          {"applied_generation", value.applied_generation},
          {"assigned_generation", value.assigned_generation},
          {"queued_paths", value.queued_paths},
          {"unstable_paths", value.unstable_paths},
          {"skipped_paths", value.skipped_paths},
          {"watcher_running", value.watcher_running},
          {"watcher_failed", value.watcher_failed}};
}

absl::StatusOr<WorktreeIndexStatus> DecodeWorktreeIndexStatus(
    const JsonView& value) {
  WorktreeIndexStatus status;
  ABSL_ASSIGN_OR_RETURN(status.root, Unhex(value, "root_hex"));
  ABSL_ASSIGN_OR_RETURN(status.file_count, value.GetInt64("file_count"));
  ABSL_ASSIGN_OR_RETURN(status.blob_count, value.GetInt64("blob_count"));
  ABSL_ASSIGN_OR_RETURN(status.index_bytes, value.GetInt64("index_bytes"));
  ABSL_ASSIGN_OR_RETURN(status.segment_count, value.GetInt64("segment_count"));
  ABSL_ASSIGN_OR_RETURN(status.pending_symbols,
                        value.GetUint64("pending_symbols"));
  ABSL_ASSIGN_OR_RETURN(status.failed_symbols,
                        value.GetUint64("failed_symbols"));
  ABSL_ASSIGN_OR_RETURN(status.applied_generation,
                        value.GetUint64("applied_generation"));
  ABSL_ASSIGN_OR_RETURN(status.assigned_generation,
                        value.GetUint64("assigned_generation"));
  ABSL_ASSIGN_OR_RETURN(status.queued_paths, value.GetInt64("queued_paths"));
  ABSL_ASSIGN_OR_RETURN(status.unstable_paths,
                        value.GetInt64("unstable_paths"));
  ABSL_ASSIGN_OR_RETURN(status.skipped_paths, value.GetInt64("skipped_paths"));
  ABSL_ASSIGN_OR_RETURN(status.watcher_running,
                        value.GetBool("watcher_running"));
  ABSL_ASSIGN_OR_RETURN(status.watcher_failed, value.GetBool("watcher_failed"));
  return status;
}

}  // namespace ferret

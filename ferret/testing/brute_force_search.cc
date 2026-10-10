// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/testing/brute_force_search.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "re2/re2.h"

#include "ferret/git/path_glob.h"
#include "ferret/regex/compile.h"
#include "ferret/search/ranking.h"
#include "ferret/search/search_types.h"

namespace ferret_testing {
namespace {

using ::ferret::FileResult;
using ::ferret::ResultLine;

// Splits `content` into lines; a final '\n' ends the last line.
std::vector<absl::string_view> SplitLines(absl::string_view content) {
  std::vector<absl::string_view> lines;
  size_t begin = 0;
  while (begin < content.size()) {
    size_t end = content.find('\n', begin);
    if (end == absl::string_view::npos) end = content.size();
    lines.push_back(content.substr(begin, end - begin));
    begin = end + 1;
  }
  return lines;
}

}  // namespace

absl::StatusOr<std::vector<FileResult>> BruteForceSearch(
    absl::Span<const CorpusFile> corpus, const ferret::SearchRequest& request) {
  ABSL_ASSIGN_OR_RETURN(
      const std::unique_ptr<ferret::CompiledPattern> pattern,
      ferret::CompilePattern({.pattern = request.pattern,
                              .literal = !request.regex,
                              .case_sensitive = request.case_sensitive}));
  std::optional<ferret::PathGlob> glob;
  if (!request.path_glob.empty()) {
    ABSL_ASSIGN_OR_RETURN(glob, ferret::PathGlob::Parse(request.path_glob));
  }
  const RE2& matcher = pattern->matcher();

  std::vector<FileResult> results;
  for (const CorpusFile& file : corpus) {
    if (glob.has_value() && !glob->Matches(file.path)) continue;
    const std::vector<absl::string_view> lines = SplitLines(file.content);
    const auto num_lines = static_cast<int64_t>(lines.size());
    // The first max_lines_per_file matching lines, and where they match.
    std::vector<int64_t> kept;
    std::vector<std::optional<std::pair<size_t, size_t>>> match(lines.size());
    int64_t num_matching_lines = 0;
    for (int64_t i = 0; i < num_lines; ++i) {
      const absl::string_view line = lines[static_cast<size_t>(i)];
      absl::string_view found;
      if (!matcher.Match(line, 0, line.size(), RE2::UNANCHORED, &found, 1)) {
        continue;
      }
      ++num_matching_lines;
      if (std::ssize(kept) < request.max_lines_per_file) {
        kept.push_back(i);
        const auto begin = static_cast<size_t>(found.data() - line.data());
        match[static_cast<size_t>(i)] = {begin, begin + found.size()};
      }
    }
    if (num_matching_lines == 0) continue;

    FileResult result;
    result.path = file.path;
    result.num_matching_lines = num_matching_lines;
    for (int64_t i = 0; i < num_lines; ++i) {
      bool near_kept = false;
      for (const int64_t k : kept) {
        near_kept |=
            i - request.context_lines <= k && k <= i + request.context_lines;
      }
      if (!near_kept) continue;
      const auto index = static_cast<size_t>(i);
      ResultLine line{.number = i + 1, .text = std::string(lines[index])};
      if (match[index].has_value()) {
        line.is_match = true;
        line.match_begin = match[index]->first;
        line.match_end = match[index]->second;
      }
      result.lines.push_back(std::move(line));
    }
    results.push_back(std::move(result));
  }
  return results;
}

absl::StatusOr<ferret::SearchResponse> BruteForceViewSearch(
    absl::Span<const ViewCorpusFile> corpus,
    const ferret::SearchRequest& request) {
  ABSL_ASSIGN_OR_RETURN(
      const std::unique_ptr<ferret::CompiledPattern> pattern,
      ferret::CompilePattern({.pattern = request.pattern,
                              .literal = !request.regex,
                              .case_sensitive = request.case_sensitive}));
  std::vector<ViewCorpusFile> ordered(corpus.begin(), corpus.end());
  std::sort(ordered.begin(), ordered.end(),
            [](const ViewCorpusFile& a, const ViewCorpusFile& b) {
              return a.rank_key < b.rank_key ||
                     (a.rank_key == b.rank_key && a.path < b.path);
            });
  ferret::SearchResponse response;
  int64_t kept_lines = 0;
  for (size_t i = 0; i < ordered.size();) {
    size_t end = i + 1;
    while (end < ordered.size() && ordered[end].content == ordered[i].content) {
      ++end;
    }
    std::vector<CorpusFile> paths;
    for (size_t j = i; j < end; ++j) {
      paths.push_back({ordered[j].path, ordered[j].content});
    }
    ferret::SearchRequest per_blob = request;
    const int64_t remaining =
        std::max<int64_t>(0, request.max_lines - kept_lines);
    per_blob.max_lines_per_file = static_cast<int>(std::min<int64_t>(
        request.max_lines_per_file, std::max<int64_t>(remaining, 1)));
    ABSL_ASSIGN_OR_RETURN(std::vector<FileResult> matches,
                          BruteForceSearch(paths, per_blob));
    if (!matches.empty()) {
      const int64_t kept = std::min<int64_t>(
          remaining, std::min<int64_t>(request.max_lines_per_file,
                                       matches.front().num_matching_lines));
      kept_lines += kept;
      for (FileResult& file : matches) {
        ++response.num_matching_files;
        response.num_matching_lines += file.num_matching_lines;
        if (kept == 0) continue;
        file.oid = ordered[i].rank_key.oid;
        file.tier = ordered[i].rank_key.tier;
        file.prior = ordered[i].rank_key.prior;
        const size_t slash = file.path.rfind('/');
        file.name_matches = RE2::PartialMatch(
            file.path.substr(slash == std::string::npos ? 0 : slash + 1),
            pattern->matcher());
        response.files.push_back(std::move(file));
      }
    }
    i = end;
  }
  ferret::RankFiles(response.files);
  if (std::ssize(response.files) > request.max_files) {
    response.files.resize(static_cast<size_t>(request.max_files));
  }
  return response;
}

}  // namespace ferret_testing

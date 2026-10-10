// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "bench/ferret/file_list.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"

#include "ferret/base/env.h"
#include "ferret/git/path_glob.h"
#include "ferret/git/worktree_files.h"
#include "ferret/text/eligibility.h"
#include "ferret/text/trigram_extractor.h"

namespace ferret_bench {
namespace {

bool IsCleanRelativePath(absl::string_view path) {
  for (const absl::string_view component : absl::StrSplit(path, '/')) {
    if (component.empty() || component == "." || component == "..") {
      return false;
    }
  }
  return true;
}

}  // namespace

absl::StatusOr<std::vector<std::string>> ParseFileList(absl::string_view text) {
  if (text.empty()) return absl::InvalidArgumentError("empty file list");
  std::vector<std::string> paths;
  absl::flat_hash_map<absl::string_view, size_t> line_of_path;
  size_t line_number = 0;
  for (const absl::string_view path :
       absl::StrSplit(absl::StripSuffix(text, "\n"), '\n')) {
    ++line_number;
    const auto error = [&](absl::string_view what) {
      return absl::InvalidArgumentError(
          absl::StrCat("line ", line_number, ": ", what));
    };
    if (path.empty()) return error("empty path");
    if (absl::StrContains(path, '\r') || absl::StrContains(path, '\0')) {
      return error(absl::StrCat("path \"", path, "\" contains CR or NUL"));
    }
    if (!IsCleanRelativePath(path)) {
      return error(absl::StrCat("\"", path, "\" is not a clean relative path"));
    }
    const auto [it, inserted] = line_of_path.emplace(path, line_number);
    if (!inserted) {
      return error(absl::StrCat("\"", path, "\" duplicates line ", it->second));
    }
    paths.emplace_back(path);
  }
  return paths;
}

std::string FormatFileList(const std::vector<std::string>& paths) {
  std::string text;
  for (const std::string& path : paths) absl::StrAppend(&text, path, "\n");
  return text;
}

absl::StatusOr<std::vector<std::string>> FilterByPathGlob(
    const std::vector<std::string>& paths, absl::string_view glob) {
  ABSL_ASSIGN_OR_RETURN(const ferret::PathGlob matcher,
                        ferret::PathGlob::Parse(glob));
  std::vector<std::string> admitted;
  for (const std::string& path : paths) {
    if (matcher.Matches(path)) admitted.push_back(path);
  }
  return admitted;
}

absl::StatusOr<EligibleFiles> ListEligibleFiles(
    ferret::Env& env, absl::string_view root,
    const ferret::EligibilityOptions& options) {
  ABSL_ASSIGN_OR_RETURN(std::vector<std::string> listed,
                        ferret::ListWorktreeFiles(root));
  std::sort(listed.begin(), listed.end());
  EligibleFiles result;
  result.files_listed = static_cast<int64_t>(listed.size());
  ferret::TrigramExtractor extractor;
  for (std::string& path : listed) {
    const std::string full_path = absl::StrCat(root, "/", path);
    ABSL_ASSIGN_OR_RETURN(const int64_t size, env.GetFileSize(full_path));
    ferret::SkipReason reason = ferret::SkipReason::kTooLarge;
    if (size <= options.max_file_bytes) {
      ABSL_ASSIGN_OR_RETURN(const std::string content, env.ReadFile(full_path));
      if (static_cast<int64_t>(content.size()) <= options.max_file_bytes) {
        reason =
            ferret::CheckEligibility(content, options, extractor).skip_reason;
      }
    }
    if (reason == ferret::SkipReason::kNone) {
      result.paths.push_back(std::move(path));
    } else {
      ++result.files_skipped[static_cast<size_t>(reason)];
    }
  }
  return result;
}

}  // namespace ferret_bench

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Git check-ignore tests directories as directories, and an excluded ancestor
// cannot be re-included by a negation below it. Unescaped trailing spaces are
// stripped, but escaped ones match literally. A leading **/ includes zero
// directories, a middle /**/ includes zero or more, and a trailing /**
// matches descendants. With core.ignoreCase, bracket classes fold case too.

#include "ferret/git/ignore.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "ferret/git/internal/path_pattern.h"

namespace ferret {
namespace {

using Rules = std::vector<git_internal::PathPattern>;

// Git's add_patterns_from_buffer() trims CRLF and unescaped trailing spaces.
Rules ParseRules(absl::string_view contents, absl::string_view base) {
  Rules rules;
  while (!contents.empty()) {
    const size_t end = contents.find('\n');
    absl::string_view line = contents.substr(0, end);
    if (end == contents.npos) {
      contents = {};
    } else {
      contents.remove_prefix(end + 1);
    }
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    while (!line.empty() && line.back() == ' ') {
      size_t slashes = 0;
      for (size_t i = line.size() - 1; i > 0 && line[i - 1] == '\\'; --i) {
        ++slashes;
      }
      if (slashes % 2 != 0) break;
      line.remove_suffix(1);
    }
    if (line.empty() || line.front() == '#') continue;
    rules.emplace_back(line, base);
  }
  return rules;
}

void ApplyRules(const Rules& rules, absl::string_view path, bool is_dir,
                bool ignore_case, bool* excluded) {
  for (const git_internal::PathPattern& rule : rules) {
    if (rule.Matches(path, is_dir, ignore_case)) {
      *excluded = !rule.negative();
    }
  }
}

std::string Join(absl::string_view root, absl::string_view relative) {
  if (relative.empty()) return std::string(root);
  return std::string(root) + "/" + std::string(relative);
}

}  // namespace

absl::StatusOr<std::optional<std::string>> ReadIgnoreFile(
    absl::string_view path) {
  std::error_code ec;
  const std::filesystem::path filename(path);
  const bool per_directory = filename.filename() == ".gitignore" ||
                             filename.filename() == ".ferretignore";
  // Git does not follow symlinks to per-directory ignore files, unlike the
  // explicitly supplied info and global excludes files.
  const std::filesystem::file_status status =
      per_directory ? std::filesystem::symlink_status(filename, ec)
                    : std::filesystem::status(filename, ec);
  if (ec == std::errc::no_such_file_or_directory) return std::nullopt;
  if (ec) return absl::UnknownError(ec.message());
  if (!std::filesystem::is_regular_file(status)) return std::nullopt;
  std::ifstream input(std::string(path), std::ios::binary);
  if (!input) return absl::UnknownError("could not open ignore file");
  std::string text(std::istreambuf_iterator<char>{input}, {});
  if (input.bad()) return absl::UnknownError("could not read ignore file");
  return text;
}

IgnoreMatcher::IgnoreMatcher(IgnoreOptions options, IgnoreFileReader reader)
    : options_(std::move(options)), reader_(std::move(reader)) {
  for (const std::string& pattern : options_.default_excludes) {
    defaults_.emplace_back(pattern, "");
  }
}

absl::StatusOr<const IgnoreMatcher::DirectoryRules*> IgnoreMatcher::RulesFor(
    absl::string_view dir) {
  if (auto it = directories_.find(std::string(dir)); it != directories_.end()) {
    return &it->second;
  }
  DirectoryRules loaded;
  const std::string prefix = Join(options_.root, dir);
  ABSL_ASSIGN_OR_RETURN(auto git, reader_(prefix + "/.gitignore"));
  if (git) loaded.gitignore = ParseRules(*git, dir);
  if (options_.ferretignore) {
    ABSL_ASSIGN_OR_RETURN(auto ferret, reader_(prefix + "/.ferretignore"));
    if (ferret) loaded.ferretignore = ParseRules(*ferret, dir);
  }
  auto it = directories_.emplace(std::string(dir), std::move(loaded)).first;
  return &it->second;
}

absl::StatusOr<bool> IgnoreMatcher::IsExcludedHere(absl::string_view path,
                                                   bool is_dir,
                                                   absl::string_view parent) {
  bool excluded = false;
  ApplyRules(defaults_, path, is_dir, options_.ignore_case, &excluded);
  if (!global_) {
    Rules loaded;
    if (!options_.global_excludes_path.empty()) {
      ABSL_ASSIGN_OR_RETURN(auto text, reader_(options_.global_excludes_path));
      if (text) loaded = ParseRules(*text, "");
    }
    global_ = std::move(loaded);
  }
  ApplyRules(*global_, path, is_dir, options_.ignore_case, &excluded);
  if (!info_) {
    Rules loaded;
    if (!options_.info_exclude_path.empty()) {
      ABSL_ASSIGN_OR_RETURN(auto text, reader_(options_.info_exclude_path));
      if (text) loaded = ParseRules(*text, "");
    }
    info_ = std::move(loaded);
  }
  ApplyRules(*info_, path, is_dir, options_.ignore_case, &excluded);

  // Each directory's rules are consulted in order, with the closest source
  // taking precedence. A basename-only rule applies only below its own base.
  absl::string_view dir;
  while (true) {
    ABSL_ASSIGN_OR_RETURN(const DirectoryRules* rules, RulesFor(dir));
    ApplyRules(rules->gitignore, path, is_dir, options_.ignore_case, &excluded);
    ApplyRules(rules->ferretignore, path, is_dir, options_.ignore_case,
               &excluded);
    if (dir == parent) break;
    const size_t slash = parent.find('/', dir.empty() ? 0 : dir.size() + 1);
    dir = parent.substr(0, slash);
  }
  return excluded;
}

absl::StatusOr<bool> IgnoreMatcher::IsIgnored(absl::string_view rel_path,
                                              bool is_dir) {
  // Test each ancestor before the leaf. Git never visits an excluded
  // directory, even if a negative pattern would match a descendant.
  for (size_t slash = rel_path.find('/'); slash != rel_path.npos;
       slash = rel_path.find('/', slash + 1)) {
    const absl::string_view ancestor = rel_path.substr(0, slash);
    const size_t parent_end = ancestor.rfind('/');
    const absl::string_view parent = parent_end == ancestor.npos
                                         ? absl::string_view()
                                         : ancestor.substr(0, parent_end);
    ABSL_ASSIGN_OR_RETURN(bool excluded,
                          IsExcludedHere(ancestor, /*is_dir=*/true, parent));
    if (excluded) return true;
  }
  const size_t parent_end = rel_path.rfind('/');
  return IsExcludedHere(rel_path, is_dir,
                        parent_end == rel_path.npos
                            ? absl::string_view()
                            : rel_path.substr(0, parent_end));
}

void IgnoreMatcher::Invalidate(absl::string_view rel_dir) {
  directories_.erase(std::string(rel_dir));
  if (rel_dir.empty()) {
    global_.reset();
    info_.reset();
  }
}

}  // namespace ferret

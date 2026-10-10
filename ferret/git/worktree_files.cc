// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/worktree_files.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

#include "ferret/base/subprocess.h"
#include "ferret/git/ignore.h"
#include "ferret/git/ls_files.h"

namespace ferret {
namespace {

constexpr uint32_t kModeRegular = 0100644;
constexpr uint32_t kModeExecutable = 0100755;

// Runs `git -C root <args>` and returns its stdout. GIT_OPTIONAL_LOCKS=0
// keeps git from taking index.lock, which would race with the user's own
// git commands; core.quotepath=off keeps non-ASCII paths verbatim.
absl::StatusOr<std::string> RunGit(absl::string_view root,
                                   std::vector<std::string> args) {
  std::vector<std::string> argv = {"git", "-C", std::string(root), "-c",
                                   "core.quotepath=off"};
  argv.insert(argv.end(), std::make_move_iterator(args.begin()),
              std::make_move_iterator(args.end()));
  ABSL_ASSIGN_OR_RETURN(
      SubprocessResult result,
      RunSubprocess(argv,
                    {.env_overrides = {"GIT_OPTIONAL_LOCKS=0", "LC_ALL=C"}}));
  if (result.exit_code != 0) {
    return absl::FailedPreconditionError(
        absl::StrCat("git ", argv[5], " in ", root, " exited with ",
                     result.exit_code, ": ", result.stderr_data));
  }
  return std::move(result.stdout_data);
}

}  // namespace

absl::StatusOr<std::vector<std::string>> ListWorktreeFiles(
    absl::string_view root) {
  ABSL_ASSIGN_OR_RETURN(const std::string tracked_output,
                        RunGit(root, {"ls-files", "-s", "-v", "-z"}));
  ABSL_ASSIGN_OR_RETURN(const std::vector<LsFilesEntry> tracked,
                        ParseLsFiles(tracked_output));
  ABSL_ASSIGN_OR_RETURN(
      const std::string untracked_output,
      RunGit(root, {"ls-files", "-z", "--others", "--exclude-standard"}));

  std::vector<std::string> paths;
  for (const LsFilesEntry& entry : tracked) {
    if (entry.skip_worktree) continue;
    if (entry.mode != kModeRegular && entry.mode != kModeExecutable) continue;
    paths.push_back(entry.path);
  }
  for (const absl::string_view path :
       absl::StrSplit(untracked_output, '\0', absl::SkipEmpty())) {
    paths.emplace_back(path);
  }
  std::sort(paths.begin(), paths.end());
  paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
  return paths;
}

absl::StatusOr<std::vector<std::string>> ListDirectoryFiles(
    absl::string_view root, const DirectoryFilesOptions& options) {
  IgnoreOptions ignore_options;
  ignore_options.root = std::string(root);
  ignore_options.ferretignore = true;
  ignore_options.ignore_case = options.ignore_case;
  ignore_options.info_exclude_path = options.info_exclude_path;
  ignore_options.default_excludes = {".git/", "node_modules/", "out/",
                                     "bazel-*/"};
  if (!options.index_directory.empty()) {
    ignore_options.default_excludes.push_back("/" + options.index_directory +
                                              "/");
  }
  ignore_options.default_excludes.insert(ignore_options.default_excludes.end(),
                                         options.default_excludes.begin(),
                                         options.default_excludes.end());
  IgnoreMatcher matcher(std::move(ignore_options));
  std::vector<std::string> paths;
  const std::filesystem::path base(root);
  std::error_code ec;
  std::filesystem::recursive_directory_iterator it(base, ec), end;
  if (ec) return absl::UnknownError(ec.message());
  while (it != end) {
    const std::filesystem::directory_entry& entry = *it;
    const std::string relative =
        entry.path().lexically_relative(base).generic_string();
    const std::filesystem::file_status status = entry.symlink_status(ec);
    if (ec) return absl::UnknownError(ec.message());
    if (std::filesystem::is_directory(status)) {
      ABSL_ASSIGN_OR_RETURN(bool ignored, matcher.IsIgnored(relative, true));
      if (ignored) it.disable_recursion_pending();
    } else {
      it.disable_recursion_pending();
      if (std::filesystem::is_regular_file(status)) {
        ABSL_ASSIGN_OR_RETURN(bool ignored, matcher.IsIgnored(relative, false));
        if (!ignored) paths.push_back(relative);
      }
    }
    it.increment(ec);
    if (ec) return absl::UnknownError(ec.message());
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

}  // namespace ferret

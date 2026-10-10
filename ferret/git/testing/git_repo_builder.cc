// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/testing/git_repo_builder.h"

#include <_stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/_types/_mode_t.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "ferret/base/subprocess.h"
#include "ferret/testing/temp_dir.h"

namespace ferret_testing {
namespace {

// Runs git in `cwd` with the isolated environment rooted at `home`.
absl::StatusOr<ferret::SubprocessResult> RunGitIn(
    absl::string_view home, absl::string_view cwd,
    std::vector<std::string> args, absl::string_view stdin_data) {
  std::vector<std::string> argv = {"git", "-c", "init.defaultBranch=main", "-c",
                                   "core.hooksPath=/dev/null"};
  argv.insert(argv.end(), std::make_move_iterator(args.begin()),
              std::make_move_iterator(args.end()));
  return ferret::RunSubprocess(argv,
                               {.cwd = std::string(cwd),
                                .env_overrides = IsolatedGitEnvironment(home),
                                .stdin_data = std::string(stdin_data)});
}

absl::StatusOr<std::string> CheckedOutput(
    absl::StatusOr<ferret::SubprocessResult> result,
    absl::Span<const std::string> args) {
  ABSL_RETURN_IF_ERROR(result.status());
  if (result->exit_code != 0) {
    return absl::InternalError(absl::StrCat("git ", absl::StrJoin(args, " "),
                                            " exited with ", result->exit_code,
                                            ": ", result->stderr_data));
  }
  return std::move(result->stdout_data);
}

absl::Status ErrnoStatus(absl::string_view op, absl::string_view path) {
  return absl::ErrnoToStatus(errno, absl::StrCat(op, "(", path, ")"));
}

}  // namespace

std::vector<std::string> IsolatedGitEnvironment(absl::string_view home) {
  return {
      "GIT_CONFIG_NOSYSTEM=1",
      "GIT_CONFIG_GLOBAL=/dev/null",
      "GIT_ATTR_NOSYSTEM=1",
      absl::StrCat("HOME=", home),
      absl::StrCat("XDG_CONFIG_HOME=", home, "/.config"),
      // Keeps repository discovery from escaping the test's directory.
      absl::StrCat("GIT_CEILING_DIRECTORIES=", home),
      "GIT_AUTHOR_NAME=Ferret Test",
      "GIT_AUTHOR_EMAIL=test@ferret.invalid",
      "GIT_AUTHOR_DATE=2026-01-01T00:00:00Z",
      "GIT_COMMITTER_NAME=Ferret Test",
      "GIT_COMMITTER_EMAIL=test@ferret.invalid",
      "GIT_COMMITTER_DATE=2026-01-01T00:00:00Z",
      "LC_ALL=C",
  };
}

void IsolateGitEnvironment(absl::string_view home) {
  for (const std::string& entry : IsolatedGitEnvironment(home)) {
    const size_t equals = entry.find('=');
    setenv(entry.substr(0, equals).c_str(), entry.c_str() + equals + 1,
           /*overwrite=*/1);
  }
}

GitRepoBuilder::GitRepoBuilder(const TempDir* absl_nonnull temp_dir,
                               std::string name)
    : temp_dir_(temp_dir),
      name_(std::move(name)),
      root_(temp_dir_->Join(name_)) {}

absl::StatusOr<GitRepoBuilder> GitRepoBuilder::Init(
    const TempDir& temp_dir, absl::string_view name,
    const GitRepoOptions& options) {
  ABSL_RETURN_IF_ERROR(temp_dir.MakeDirs(name));
  GitRepoBuilder repo(&temp_dir, std::string(name));
  ABSL_RETURN_IF_ERROR(
      repo.Git({"init", "-q",
                absl::StrCat("--object-format=", options.object_format)})
          .status());
  return repo;
}

absl::StatusOr<GitRepoBuilder> GitRepoBuilder::ShallowClone(
    const GitRepoBuilder& source, absl::string_view name, int depth) {
  const TempDir& temp_dir = *source.temp_dir_;
  const std::vector<std::string> args = {
      "clone", "-q", absl::StrCat("--depth=", depth),
      absl::StrCat("file://", source.root()), temp_dir.Join(name)};
  ABSL_RETURN_IF_ERROR(
      CheckedOutput(RunGitIn(temp_dir.path(), temp_dir.path(), args, ""), args)
          .status());
  return GitRepoBuilder(&temp_dir, std::string(name));
}

std::string GitRepoBuilder::Path(absl::string_view relative_path) const {
  return absl::StrCat(root_, "/", relative_path);
}

absl::Status GitRepoBuilder::WriteFile(absl::string_view relative_path,
                                       absl::string_view content) const {
  return temp_dir_->WriteFile(absl::StrCat(name_, "/", relative_path), content);
}

absl::Status GitRepoBuilder::DeleteFile(absl::string_view relative_path) const {
  const std::string path = Path(relative_path);
  if (unlink(path.c_str()) != 0) return ErrnoStatus("unlink", path);
  return absl::OkStatus();
}

absl::Status GitRepoBuilder::Rename(absl::string_view from,
                                    absl::string_view to) const {
  const std::string from_path = Path(from);
  if (rename(from_path.c_str(), Path(to).c_str()) != 0) {
    return ErrnoStatus("rename", from_path);
  }
  return absl::OkStatus();
}

absl::Status GitRepoBuilder::Chmod(absl::string_view relative_path,
                                   mode_t mode) const {
  const std::string path = Path(relative_path);
  if (chmod(path.c_str(), mode) != 0) return ErrnoStatus("chmod", path);
  return absl::OkStatus();
}

absl::Status GitRepoBuilder::Symlink(absl::string_view target,
                                     absl::string_view relative_path) const {
  const size_t slash = relative_path.rfind('/');
  if (slash != absl::string_view::npos) {
    ABSL_RETURN_IF_ERROR(temp_dir_->MakeDirs(
        absl::StrCat(name_, "/", relative_path.substr(0, slash))));
  }
  const std::string path = Path(relative_path);
  if (symlink(std::string(target).c_str(), path.c_str()) != 0) {
    return ErrnoStatus("symlink", path);
  }
  return absl::OkStatus();
}

absl::StatusOr<ferret::SubprocessResult> GitRepoBuilder::RunGit(
    std::vector<std::string> args, absl::string_view stdin_data) const {
  return RunGitIn(temp_dir_->path(), root_, std::move(args), stdin_data);
}

absl::StatusOr<std::string> GitRepoBuilder::Git(
    const std::vector<std::string>& args, absl::string_view stdin_data) const {
  absl::StatusOr<ferret::SubprocessResult> result = RunGit(args, stdin_data);
  return CheckedOutput(std::move(result), args);
}

absl::Status GitRepoBuilder::Add(
    std::initializer_list<absl::string_view> paths) const {
  std::vector<std::string> args = {"add", "-A", "--"};
  args.insert(args.end(), paths.begin(), paths.end());
  return Git(args).status();
}

absl::Status GitRepoBuilder::Commit(absl::string_view message) const {
  return Git({"commit", "-q", "--allow-empty", "-m", std::string(message)})
      .status();
}

absl::Status GitRepoBuilder::SetConfig(absl::string_view key,
                                       absl::string_view value) const {
  return Git({"config", std::string(key), std::string(value)}).status();
}

absl::Status GitRepoBuilder::AddGitlink(absl::string_view path,
                                        absl::string_view commit_hex) const {
  return Git({"update-index", "--add", "--cacheinfo",
              absl::StrCat("160000,", commit_hex, ",", path)})
      .status();
}

absl::Status GitRepoBuilder::SetSkipWorktree(absl::string_view path) const {
  return Git({"update-index", "--skip-worktree", "--", std::string(path)})
      .status();
}

absl::Status GitRepoBuilder::SetAssumeUnchanged(absl::string_view path) const {
  return Git({"update-index", "--assume-unchanged", "--", std::string(path)})
      .status();
}

absl::StatusOr<GitRepoBuilder> GitRepoBuilder::AddWorktree(
    absl::string_view name) const {
  ABSL_RETURN_IF_ERROR(
      Git({"worktree", "add", "-q", "--detach", temp_dir_->Join(name)})
          .status());
  return GitRepoBuilder(temp_dir_, std::string(name));
}

}  // namespace ferret_testing

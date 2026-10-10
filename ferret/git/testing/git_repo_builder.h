// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Throwaway git repositories for tests, built with the real git binary.
//
// Every git command the builder runs gets an isolated environment (see
// IsolatedGitEnvironment()): no system or user configuration, no global
// excludes or attributes, no hooks, a private HOME, and fixed identities and
// dates. Tests therefore behave the same on every machine, and commits made
// from the same inputs have the same ids.
//
// Code under test that spawns git itself inherits the test's environment,
// not the builder's; tests call IsolateGitEnvironment() in SetUp() so that
// both see the same configuration.

#ifndef FERRET_GIT_TESTING_GIT_REPO_BUILDER_H_
#define FERRET_GIT_TESTING_GIT_REPO_BUILDER_H_

#include <sys/types.h>

#include <initializer_list>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "ferret/base/subprocess.h"
#include "ferret/testing/temp_dir.h"

namespace ferret_testing {

// Returns the "KEY=VALUE" entries that isolate git from the machine it runs
// on, with HOME and XDG_CONFIG_HOME pointing into `home`.
std::vector<std::string> IsolatedGitEnvironment(absl::string_view home);

// Applies IsolatedGitEnvironment(home) to this process with setenv(), so that
// git processes spawned by the code under test inherit it. Call it before
// starting any thread.
void IsolateGitEnvironment(absl::string_view home);

struct GitRepoOptions {
  // Passed to `git init --object-format`: "sha1" or "sha256".
  std::string object_format = "sha1";
};

// A handle to one worktree inside a TempDir, which must outlive it. Every
// method that touches the filesystem or runs git returns its failure as a
// Status. Not thread-safe.
class GitRepoBuilder final {
 public:
  // Runs `git init` in the new directory `temp_dir.Join(name)`.
  static absl::StatusOr<GitRepoBuilder> Init(
      const TempDir& temp_dir, absl::string_view name,
      const GitRepoOptions& options = {});

  // Clones `source` with `git clone --depth=<depth> file://...` into
  // `temp_dir.Join(name)`; the result is a shallow repository.
  static absl::StatusOr<GitRepoBuilder> ShallowClone(
      const GitRepoBuilder& source, absl::string_view name, int depth);

  GitRepoBuilder(const GitRepoBuilder&) = default;
  GitRepoBuilder& operator=(const GitRepoBuilder&) = default;

  // Canonical absolute path of the worktree.
  const std::string& root() const { return root_; }

  // Returns root() + "/" + relative_path.
  std::string Path(absl::string_view relative_path) const;

  // Worktree edits. Parent directories are created as needed.
  absl::Status WriteFile(absl::string_view relative_path,
                         absl::string_view content) const;
  absl::Status DeleteFile(absl::string_view relative_path) const;
  absl::Status Rename(absl::string_view from, absl::string_view to) const;
  absl::Status Chmod(absl::string_view relative_path, mode_t mode) const;
  absl::Status Symlink(absl::string_view target,
                       absl::string_view relative_path) const;

  // Runs `git <args>` in root() and returns the result whatever the exit
  // code.
  absl::StatusOr<ferret::SubprocessResult> RunGit(
      std::vector<std::string> args, absl::string_view stdin_data = "") const;

  // Runs `git <args>` in root() and returns its stdout; a non-zero exit is an
  // error carrying git's stderr.
  absl::StatusOr<std::string> Git(const std::vector<std::string>& args,
                                  absl::string_view stdin_data = "") const;

  // `git add -A -- <paths>`; no paths means the whole worktree.
  absl::Status Add(std::initializer_list<absl::string_view> paths = {}) const;
  // `git commit -q --allow-empty -m <message>`.
  absl::Status Commit(absl::string_view message) const;
  // `git config <key> <value>`, in the repository's own config file.
  absl::Status SetConfig(absl::string_view key, absl::string_view value) const;
  // Adds a submodule entry (mode 160000) for `commit_hex` at `path` to the
  // index without cloning anything, as an uninitialized submodule looks.
  absl::Status AddGitlink(absl::string_view path,
                          absl::string_view commit_hex) const;
  // `git update-index --skip-worktree <path>`.
  absl::Status SetSkipWorktree(absl::string_view path) const;
  // `git update-index --assume-unchanged <path>`.
  absl::Status SetAssumeUnchanged(absl::string_view path) const;

  // `git worktree add --detach <temp_dir.Join(name)>`: a second worktree that
  // shares this repository's object store and refs.
  absl::StatusOr<GitRepoBuilder> AddWorktree(absl::string_view name) const;

 private:
  GitRepoBuilder(const TempDir* absl_nonnull temp_dir, std::string name);

  const TempDir* absl_nonnull temp_dir_;
  std::string name_;  // Relative to temp_dir_.
  std::string root_;
};

}  // namespace ferret_testing

#endif  // FERRET_GIT_TESTING_GIT_REPO_BUILDER_H_

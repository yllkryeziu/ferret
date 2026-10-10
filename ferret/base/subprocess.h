// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Runs a child process with an argv (never a shell) and captures its output.
//
// Ferret shells out to git for worktree scans and to ripgrep for the baseline
// backend. Both take agent-influenced strings as arguments, so the command
// line is always an argv passed to posix_spawnp: there is no shell to inject
// into. All inherited descriptors except the three standard streams are
// closed in the child.

#ifndef FERRET_BASE_SUBPROCESS_H_
#define FERRET_BASE_SUBPROCESS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

namespace ferret {

struct SubprocessOptions {
  // Working directory for the child; empty means inherit.
  std::string cwd;
  // "KEY=VALUE" entries added to the inherited environment, replacing any
  // inherited entry with the same KEY.
  std::vector<std::string> env_overrides;
  // Bytes written to the child's stdin, which is then closed.
  std::string stdin_data;
  // Combined stdout + stderr cap. Exceeding it kills the child and returns
  // ResourceExhausted.
  int64_t max_output_bytes = int64_t{512} << 20;
  // Kills the child and returns DeadlineExceeded when exceeded.
  absl::Duration timeout = absl::InfiniteDuration();
};

struct SubprocessResult {
  // The exit status, or 128 + signal number if the child was killed.
  int exit_code = 0;
  std::string stdout_data;
  std::string stderr_data;
};

// Runs `argv` (argv[0] is looked up on PATH) and waits for it to exit.
// Returns an error only if the child could not be run to completion (spawn
// failure, timeout, output cap); a non-zero exit code is not an error.
// Thread-safe.
absl::StatusOr<SubprocessResult> RunSubprocess(
    absl::Span<const std::string> argv, const SubprocessOptions& options = {});

// Calls `on_stdout` for each pipe chunk, retaining only stderr. A callback
// error kills and reaps the child. The output cap applies to stderr only.
absl::StatusOr<SubprocessResult> StreamSubprocess(
    absl::Span<const std::string> argv, const SubprocessOptions& options,
    absl::FunctionRef<absl::Status(absl::string_view)> on_stdout);

}  // namespace ferret

#endif  // FERRET_BASE_SUBPROCESS_H_

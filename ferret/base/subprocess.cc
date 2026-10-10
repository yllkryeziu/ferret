// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/base/subprocess.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/_types/_pid_t.h>
#include <sys/_types/_ssize_t.h>
#include <sys/fcntl.h>
#include <sys/poll.h>
#include <sys/signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

extern char** environ;

namespace ferret {
namespace {

// A pipe whose ends are closed on destruction unless released.
class Pipe {
 public:
  Pipe() = default;
  Pipe(const Pipe&) = delete;
  Pipe& operator=(const Pipe&) = delete;
  ~Pipe() {
    CloseRead();
    CloseWrite();
  }

  absl::Status Open() {
    int fds[2];
    if (pipe(fds) != 0) return absl::ErrnoToStatus(errno, "pipe()");
    read_fd_ = fds[0];
    write_fd_ = fds[1];
    // The parent's ends must not leak into other children spawned
    // concurrently from other threads.
    fcntl(read_fd_, F_SETFD, FD_CLOEXEC);
    fcntl(write_fd_, F_SETFD, FD_CLOEXEC);
    return absl::OkStatus();
  }

  int read_fd() const { return read_fd_; }
  int write_fd() const { return write_fd_; }

  void CloseRead() {
    if (read_fd_ >= 0) close(read_fd_);
    read_fd_ = -1;
  }
  void CloseWrite() {
    if (write_fd_ >= 0) close(write_fd_);
    write_fd_ = -1;
  }

 private:
  int read_fd_ = -1;
  int write_fd_ = -1;
};

// Returns the inherited environment with `overrides` applied. The strings
// must outlive the returned pointers.
std::vector<std::string> MergeEnvironment(
    absl::Span<const std::string> overrides) {
  std::vector<std::string> merged;
  for (char** entry = environ; *entry != nullptr; ++entry) {
    const absl::string_view kv(*entry);
    const absl::string_view key = kv.substr(0, kv.find('='));
    const bool overridden = std::any_of(
        overrides.begin(), overrides.end(),
        [key](absl::string_view o) { return o.substr(0, o.find('=')) == key; });
    if (!overridden) merged.emplace_back(kv);
  }
  merged.insert(merged.end(), overrides.begin(), overrides.end());
  return merged;
}

void KillAndReap(pid_t pid) {
  kill(pid, SIGKILL);
  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
}

}  // namespace

absl::StatusOr<SubprocessResult> StreamSubprocess(
    absl::Span<const std::string> argv, const SubprocessOptions& options,
    absl::FunctionRef<absl::Status(absl::string_view)> on_stdout) {
  if (argv.empty()) return absl::InvalidArgumentError("subprocess: empty argv");

  Pipe out_pipe, err_pipe, in_pipe;
  ABSL_RETURN_IF_ERROR(out_pipe.Open());
  ABSL_RETURN_IF_ERROR(err_pipe.Open());
  ABSL_RETURN_IF_ERROR(in_pipe.Open());

  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  absl::Cleanup destroy_actions = [&] {
    posix_spawn_file_actions_destroy(&actions);
  };
  posix_spawn_file_actions_adddup2(&actions, in_pipe.read_fd(), STDIN_FILENO);
  posix_spawn_file_actions_adddup2(&actions, out_pipe.write_fd(),
                                   STDOUT_FILENO);
  posix_spawn_file_actions_adddup2(&actions, err_pipe.write_fd(),
                                   STDERR_FILENO);
  if (!options.cwd.empty()) {
    posix_spawn_file_actions_addchdir_np(&actions, options.cwd.c_str());
  }

  posix_spawnattr_t attr;
  posix_spawnattr_init(&attr);
  absl::Cleanup destroy_attr = [&] { posix_spawnattr_destroy(&attr); };
#if defined(POSIX_SPAWN_CLOEXEC_DEFAULT)
  // macOS: close every descriptor not named in `actions`.
  posix_spawnattr_setflags(&attr, POSIX_SPAWN_CLOEXEC_DEFAULT);
#endif

  std::vector<char*> c_argv;
  c_argv.reserve(argv.size() + 1);
  for (const std::string& arg : argv) {
    c_argv.push_back(const_cast<char*>(arg.c_str()));
  }
  c_argv.push_back(nullptr);

  std::vector<std::string> env_storage;
  std::vector<char*> c_env;
  char** envp = environ;
  if (!options.env_overrides.empty()) {
    env_storage = MergeEnvironment(options.env_overrides);
    c_env.reserve(env_storage.size() + 1);
    for (std::string& entry : env_storage) c_env.push_back(entry.data());
    c_env.push_back(nullptr);
    envp = c_env.data();
  }

  pid_t pid = 0;
  const int spawn_error =
      posix_spawnp(&pid, c_argv[0], &actions, &attr, c_argv.data(), envp);
  if (spawn_error != 0) {
    return absl::ErrnoToStatus(spawn_error,
                               absl::StrCat("posix_spawnp(", argv[0], ")"));
  }
  in_pipe.CloseRead();
  out_pipe.CloseWrite();
  err_pipe.CloseWrite();
  // A blocking write of more than the pipe's free space would wait for the
  // child to read while the child waits for us to drain its full stdout: poll
  // only promises that *some* bytes fit. A child that exits without reading
  // stdin must yield EPIPE, not a SIGPIPE that kills this process.
  if (in_pipe.write_fd() >= 0) {
    const int flags = fcntl(in_pipe.write_fd(), F_GETFL);
    if (flags < 0 ||
        fcntl(in_pipe.write_fd(), F_SETFL, flags | O_NONBLOCK) < 0) {
      const int saved = errno;
      KillAndReap(pid);
      return absl::ErrnoToStatus(saved, "fcntl(O_NONBLOCK)");
    }
#if defined(F_SETNOSIGPIPE)
    if (fcntl(in_pipe.write_fd(), F_SETNOSIGPIPE, 1) < 0) {
      const int saved = errno;
      KillAndReap(pid);
      return absl::ErrnoToStatus(saved, "fcntl(F_SETNOSIGPIPE)");
    }
#endif
  }

  // Feed stdin, drain stdout/stderr, and enforce the deadline and cap with
  // one poll loop so that a chatty child can never deadlock on a full pipe.
  SubprocessResult result;
  const absl::Time deadline = absl::Now() + options.timeout;
  size_t stdin_written = 0;
  if (options.stdin_data.empty()) in_pipe.CloseWrite();
  std::array<char, size_t{64} * 1024> buffer;
  while (out_pipe.read_fd() >= 0 || err_pipe.read_fd() >= 0) {
    std::array<pollfd, 3> fds;
    int nfds = 0;
    if (out_pipe.read_fd() >= 0) fds[nfds++] = {out_pipe.read_fd(), POLLIN, 0};
    if (err_pipe.read_fd() >= 0) fds[nfds++] = {err_pipe.read_fd(), POLLIN, 0};
    if (in_pipe.write_fd() >= 0) fds[nfds++] = {in_pipe.write_fd(), POLLOUT, 0};

    int timeout_ms = -1;
    if (options.timeout != absl::InfiniteDuration()) {
      const absl::Duration left = deadline - absl::Now();
      if (left <= absl::ZeroDuration()) {
        KillAndReap(pid);
        return absl::DeadlineExceededError(
            absl::StrCat("subprocess ", argv[0], " timed out after ",
                         absl::FormatDuration(options.timeout)));
      }
      timeout_ms = static_cast<int>(absl::ToInt64Milliseconds(left)) + 1;
    }
    const int ready = poll(fds.data(), static_cast<nfds_t>(nfds), timeout_ms);
    if (ready < 0) {
      if (errno == EINTR) continue;
      const int saved = errno;
      KillAndReap(pid);
      return absl::ErrnoToStatus(saved, "poll()");
    }
    for (int i = 0; i < nfds; ++i) {
      if (fds[i].revents == 0) continue;
      if (fds[i].fd == in_pipe.write_fd()) {
        const ssize_t n =
            write(in_pipe.write_fd(), options.stdin_data.data() + stdin_written,
                  options.stdin_data.size() - stdin_written);
        if (n > 0) stdin_written += static_cast<size_t>(n);
        if (n < 0 && errno != EINTR && errno != EAGAIN) in_pipe.CloseWrite();
        if (stdin_written == options.stdin_data.size()) in_pipe.CloseWrite();
        continue;
      }
      const bool is_out = fds[i].fd == out_pipe.read_fd();
      const ssize_t n = read(fds[i].fd, buffer.data(), buffer.size());
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) {
        is_out ? out_pipe.CloseRead() : err_pipe.CloseRead();
        continue;
      }
      if (is_out) {
        absl::Status parsed =
            on_stdout(absl::string_view(buffer.data(), static_cast<size_t>(n)));
        if (!parsed.ok()) {
          KillAndReap(pid);
          return parsed;
        }
      } else {
        result.stderr_data.append(buffer.data(), static_cast<size_t>(n));
      }
      if (static_cast<int64_t>(result.stdout_data.size()) +
              static_cast<int64_t>(result.stderr_data.size()) >
          options.max_output_bytes) {
        KillAndReap(pid);
        return absl::ResourceExhaustedError(
            absl::StrCat("subprocess ", argv[0], " exceeded output cap of ",
                         options.max_output_bytes, " bytes"));
      }
    }
  }
  in_pipe.CloseWrite();

  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) return absl::ErrnoToStatus(errno, "waitpid()");
  }
  if (WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    result.exit_code = 128 + WTERMSIG(status);
  }
  return result;
}

absl::StatusOr<SubprocessResult> RunSubprocess(
    absl::Span<const std::string> argv, const SubprocessOptions& options) {
  std::string output;
  absl::StatusOr<SubprocessResult> result = StreamSubprocess(
      argv, options, [&](absl::string_view chunk) -> absl::Status {
        output.append(chunk);
        if (static_cast<int64_t>(output.size()) > options.max_output_bytes) {
          return absl::ResourceExhaustedError("subprocess exceeded output cap");
        }
        return absl::OkStatus();
      });
  if (result.ok()) result->stdout_data = std::move(output);
  return result;
}

}  // namespace ferret

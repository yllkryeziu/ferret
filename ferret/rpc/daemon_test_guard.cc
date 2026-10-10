// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Test daemon cleanup: obtain the peer PID before a drain can close the socket.
#include "ferret/rpc/daemon_test_guard.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "ferret/rpc/daemon.h"
#include "ferret/rpc/rpc_client.h"

// Apple SDK declarations for signals and F_OK confuse include-cleaner.
// NOLINTBEGIN(misc-include-cleaner)
namespace ferret {
namespace {

pid_t PeerPid(const std::string& path) {
  const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  sockaddr_un address = {};
  address.sun_family = AF_UNIX;
  if (path.size() >= sizeof(address.sun_path)) {
    close(fd);
    return -1;
  }
  path.copy(address.sun_path, path.size());
  pid_t pid = -1;
  if (connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) ==
      0) {
#if defined(__APPLE__)
    socklen_t length = sizeof(pid);
    if (getsockopt(fd, SOL_LOCAL, LOCAL_PEERPID, &pid, &length) != 0) pid = -1;
#else
    struct ucred credentials;
    socklen_t length = sizeof(credentials);
    if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &credentials, &length) == 0) {
      pid = credentials.pid;
    }
#endif
  }
  close(fd);
  return pid;
}

bool Exited(pid_t pid) {
  if (pid <= 0) return true;
  int status = 0;
  if (waitpid(pid, &status, WNOHANG) == pid) return true;
  return kill(pid, 0) != 0 && errno == ESRCH;
}

}  // namespace

void DaemonTestGuard::Watch(absl::string_view root) {
  roots_.emplace_back(root);
}

void DaemonTestGuard::WatchPid(pid_t pid) {
  if (pid > 0) pids_.push_back(pid);
}

void DaemonTestGuard::Stop(absl::string_view executable,
                           absl::string_view index_dir) {
  for (const std::string& root : roots_) {
    auto socket = DaemonSocketPath(root);
    if (!socket.ok() || access(socket->c_str(), F_OK) != 0) continue;
    WatchPid(PeerPid(*socket));
    RpcClient client{std::string(executable), std::string(index_dir)};
    const auto registered = client.Register(root);
    if (registered.ok()) {
      WatchPid(PeerPid(*socket));
      EXPECT_TRUE(client.Drain().ok());
    } else {
      ADD_FAILURE() << "unable to drain daemon: " << registered;
    }
    for (int i = 0; i < 1500 && access(socket->c_str(), F_OK) == 0; ++i) {
      usleep(10000);
    }
    EXPECT_NE(access(socket->c_str(), F_OK), 0) << *socket;
  }
  for (pid_t pid : pids_) {
    for (int i = 0; i < 1500 && !Exited(pid); ++i) usleep(10000);
    const bool outlived_test = !Exited(pid);
    if (outlived_test) {
      kill(pid, SIGKILL);
      waitpid(pid, nullptr, 0);
    }
    EXPECT_FALSE(outlived_test) << "daemon pid " << pid << " outlived test";
  }
  for (const std::string& root : roots_) {
    auto socket = DaemonSocketPath(root);
    if (socket.ok() && access(socket->c_str(), F_OK) == 0) {
      ADD_FAILURE() << "daemon left socket " << *socket;
      unlink(socket->c_str());
    }
  }
}

}  // namespace ferret
// NOLINTEND(misc-include-cleaner)

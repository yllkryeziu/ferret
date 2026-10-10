// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/rpc/testing/test_pipe.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/_types/_ssize_t.h>
#include <sys/fcntl.h>
#include <unistd.h>

#include <cstddef>
#include <string>

#include "absl/log/check.h"

namespace ferret_testing {
namespace {

void SetNonBlocking(int fd) {
  const int flags = fcntl(fd, F_GETFL);
  PCHECK(flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0)
      << "fcntl(" << fd << ")";
}

void Close(int& fd) {
  if (fd >= 0) PCHECK(close(fd) == 0) << "close(" << fd << ")";
  fd = -1;
}

}  // namespace

TestPipe::TestPipe() {
  int fds[2];
  PCHECK(pipe(fds) == 0) << "pipe()";
  read_fd_ = fds[0];
  write_fd_ = fds[1];
}

TestPipe::~TestPipe() {
  Close(read_fd_);
  Close(write_fd_);
}

void TestPipe::CloseRead() { Close(read_fd_); }

void TestPipe::CloseWrite() { Close(write_fd_); }

void TestPipe::SetReadNonBlocking() const { SetNonBlocking(read_fd_); }

void TestPipe::SetWriteNonBlocking() const { SetNonBlocking(write_fd_); }

std::string TestPipe::ReadAll() const {
  std::string bytes;
  char buffer[4096];
  while (true) {
    const ssize_t n = read(read_fd_, buffer, sizeof(buffer));
    if (n == 0) return bytes;
    if (n < 0) {
      PCHECK(errno == EINTR) << "read(" << read_fd_ << ")";
      continue;
    }
    bytes.append(buffer, static_cast<size_t>(n));
  }
}

}  // namespace ferret_testing

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// TestPipe: a pipe(2) whose ends close themselves, for tests that drive a
// transport through real file descriptors.

#ifndef FERRET_RPC_TESTING_TEST_PIPE_H_
#define FERRET_RPC_TESTING_TEST_PIPE_H_

#include <string>

namespace ferret_testing {

// Owns both ends of a pipe. Closing an end signals end-of-file (write end)
// or makes writes fail with EPIPE (read end). Thread-compatible; the
// descriptors themselves may be used from any thread.
class TestPipe final {
 public:
  // CHECK-fails if pipe(2) fails, which in a test is a broken sandbox.
  TestPipe();
  TestPipe(const TestPipe&) = delete;
  TestPipe& operator=(const TestPipe&) = delete;
  ~TestPipe();

  // -1 once closed.
  int read_fd() const { return read_fd_; }
  int write_fd() const { return write_fd_; }

  void CloseRead();
  void CloseWrite();

  // Sets O_NONBLOCK on one end.
  void SetReadNonBlocking() const;
  void SetWriteNonBlocking() const;

  // Reads until end-of-file. CHECK-fails on a read error.
  std::string ReadAll() const;

 private:
  int read_fd_ = -1;
  int write_fd_ = -1;
};

}  // namespace ferret_testing

#endif  // FERRET_RPC_TESTING_TEST_PIPE_H_

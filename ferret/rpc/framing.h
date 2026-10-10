// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Newline-delimited framing over a file descriptor: one message per line.
//
// The MCP stdio transport and Ferret's daemon RPC share this wire format
// ("Messages are delimited by newlines, and MUST NOT contain embedded
// newlines",
// https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio).
// It works unchanged over pipes and Unix domain sockets. The reader bounds
// each frame, so a peer cannot make it buffer without limit, and skips an
// oversized frame without losing the stream. The writer emits each frame as
// one uninterrupted line even when several threads answer at once.
//
// Writing to a pipe or socket whose reader has gone raises SIGPIPE, which
// kills the process by default. Binaries that serve RPC set SIGPIPE to
// SIG_IGN in main, so that the write fails with EPIPE instead.

#ifndef FERRET_RPC_FRAMING_H_
#define FERRET_RPC_FRAMING_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace ferret {

struct RpcFrameReaderOptions {
  // Longest accepted frame, not counting its '\n'.
  int64_t max_frame_bytes = int64_t{16} << 20;
};

// Reads frames from a descriptor that it borrows and never closes. Blocking
// and non-blocking descriptors both work. Memory use is bounded by
// max_frame_bytes plus one 64 KiB read. Not thread-safe.
class RpcFrameReader final {
 public:
  explicit RpcFrameReader(int fd, const RpcFrameReaderOptions& options = {});
  RpcFrameReader(const RpcFrameReader&) = delete;
  RpcFrameReader& operator=(const RpcFrameReader&) = delete;

  // Returns the next frame without its '\n'. Errors:
  //   * OutOfRange: the stream ended cleanly, at a frame boundary.
  //   * DataLoss: the stream ended inside a frame (a final line with no
  //     '\n'). The partial frame is dropped; the next call returns
  //     OutOfRange.
  //   * ResourceExhausted: the frame is longer than max_frame_bytes. The
  //     rest of it is skipped, and the next call returns the frame after it.
  //   * Any other code: read(2) failed.
  // Interrupted reads (EINTR) are retried.
  absl::StatusOr<std::string> Read();

 private:
  // Appends one read(2) worth of bytes to buffer_, or sets eof_.
  absl::Status Fill();

  const int fd_;
  const int64_t max_frame_bytes_;
  // Bytes read but not yet consumed are buffer_[begin_, buffer_.size()), and
  // buffer_[begin_, scanned_) is known to contain no '\n'.
  std::string buffer_;
  size_t begin_ = 0;
  size_t scanned_ = 0;
  // True while discarding the tail of an oversized frame.
  bool skipping_ = false;
  bool eof_ = false;
};

// Writes frames to a descriptor that it borrows and never closes. Blocking
// and non-blocking descriptors both work. Thread-safe: concurrent Write calls
// never interleave their bytes.
class RpcFrameWriter final {
 public:
  explicit RpcFrameWriter(int fd) : fd_(fd) {}
  RpcFrameWriter(const RpcFrameWriter&) = delete;
  RpcFrameWriter& operator=(const RpcFrameWriter&) = delete;

  // Writes `frame` and a '\n' in full, retrying short and interrupted
  // writes. Returns InvalidArgument, having written nothing, if `frame`
  // contains '\n'. After a write error the stream may end in a partial
  // frame, so that call and every later one fail.
  absl::Status Write(absl::string_view frame) ABSL_LOCKS_EXCLUDED(mu_);

  // OK, or the write error that broke the stream.
  absl::Status status() const ABSL_LOCKS_EXCLUDED(mu_);

 private:
  const int fd_;
  // Held across the write(2) calls of a frame: preventing interleaved frames
  // is its only job, and nothing else is ever locked while it is held.
  mutable absl::Mutex mu_;
  // The error that broke the stream, if any.
  absl::Status failure_ ABSL_GUARDED_BY(mu_);
};

}  // namespace ferret

#endif  // FERRET_RPC_FRAMING_H_

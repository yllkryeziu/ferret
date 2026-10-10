// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/rpc/framing.h"

#include <errno.h>
#include <poll.h>
#include <sys/_types/_iovec_t.h>
#include <sys/_types/_ssize_t.h>
#include <sys/poll.h>
#include <sys/uio.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace ferret {
namespace {

constexpr size_t kReadChunkBytes = size_t{64} << 10;

// Blocks until `fd` is ready for `events`. Used only after a non-blocking
// descriptor reported EAGAIN.
absl::Status AwaitReady(int fd, int16_t events) {
  pollfd entry = {fd, static_cast<decltype(pollfd{}.events)>(events), 0};
  while (poll(&entry, 1, /*timeout=*/-1) < 0) {
    if (errno != EINTR) return absl::ErrnoToStatus(errno, "poll()");
  }
  return absl::OkStatus();
}

}  // namespace

RpcFrameReader::RpcFrameReader(int fd, const RpcFrameReaderOptions& options)
    : fd_(fd), max_frame_bytes_(options.max_frame_bytes) {}

absl::StatusOr<std::string> RpcFrameReader::Read() {
  const auto too_long = [this] {
    return absl::ResourceExhaustedError(absl::StrCat(
        "frame exceeds the limit of ", max_frame_bytes_, " bytes"));
  };
  while (true) {
    const size_t size = buffer_.size();
    const void* newline =
        std::memchr(buffer_.data() + scanned_, '\n', size - scanned_);
    if (newline != nullptr) {
      const size_t end = static_cast<size_t>(static_cast<const char*>(newline) -
                                             buffer_.data());
      const size_t frame_begin = begin_;
      begin_ = scanned_ = end + 1;
      if (skipping_) {
        skipping_ = false;
        continue;
      }
      // A whole oversized frame can arrive in a single read.
      if (static_cast<int64_t>(end - frame_begin) > max_frame_bytes_) {
        return too_long();
      }
      return buffer_.substr(frame_begin, end - frame_begin);
    }
    scanned_ = size;
    if (skipping_) {
      begin_ = size;
    } else if (static_cast<int64_t>(size - begin_) > max_frame_bytes_) {
      begin_ = size;
      skipping_ = true;
      return too_long();
    }
    if (eof_) {
      if (begin_ == size) return absl::OutOfRangeError("end of stream");
      const size_t partial_bytes = size - begin_;
      begin_ = size;
      return absl::DataLossError(absl::StrCat(
          "stream ended inside a frame after ", partial_bytes, " bytes"));
    }
    ABSL_RETURN_IF_ERROR(Fill());
  }
}

absl::Status RpcFrameReader::Fill() {
  // Drop consumed bytes first, so that the buffer never holds more than one
  // frame plus one read.
  buffer_.erase(0, begin_);
  scanned_ -= begin_;
  begin_ = 0;
  const size_t old_size = buffer_.size();
  buffer_.resize(old_size + kReadChunkBytes);
  while (true) {
    const ssize_t n = read(fd_, buffer_.data() + old_size, kReadChunkBytes);
    if (n >= 0) {
      buffer_.resize(old_size + static_cast<size_t>(n));
      eof_ = n == 0;
      return absl::OkStatus();
    }
    if (errno == EINTR) continue;
    absl::Status status = errno == EAGAIN || errno == EWOULDBLOCK
                              ? AwaitReady(fd_, POLLIN)
                              : absl::ErrnoToStatus(errno, "read()");
    if (!status.ok()) {
      buffer_.resize(old_size);
      return status;
    }
  }
}

absl::Status RpcFrameWriter::Write(absl::string_view frame) {
  if (const size_t newline = frame.find('\n');
      newline != absl::string_view::npos) {
    return absl::InvalidArgumentError(
        absl::StrCat("frame contains a newline at byte ", newline));
  }
  absl::MutexLock lock(mu_);
  if (!failure_.ok()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "stream broken by an earlier write: ", failure_.message()));
  }
  char newline = '\n';
  // writev(2) never writes through iov_base; the cast only satisfies iovec.
  iovec parts[2] = {{const_cast<char*>(frame.data()), frame.size()},
                    {&newline, 1}};
  iovec* next = parts;
  int count = 2;
  while (count > 0) {
    const ssize_t n = writev(fd_, next, count);
    if (n < 0) {
      if (errno == EINTR) continue;
      absl::Status status = errno == EAGAIN || errno == EWOULDBLOCK
                                ? AwaitReady(fd_, POLLOUT)
                                : absl::ErrnoToStatus(errno, "writev()");
      if (status.ok()) continue;
      failure_ = status;
      return status;
    }
    size_t written = static_cast<size_t>(n);
    while (count > 0 && written >= next->iov_len) {
      written -= next->iov_len;
      ++next;
      --count;
    }
    if (count > 0) {
      next->iov_base = static_cast<char*>(next->iov_base) + written;
      next->iov_len -= written;
    }
  }
  return absl::OkStatus();
}

absl::Status RpcFrameWriter::status() const {
  absl::MutexLock lock(mu_);
  return failure_;
}

}  // namespace ferret

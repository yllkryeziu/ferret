// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/internal/pack_file.h"

#include <fcntl.h>
#include <sys/_types/_ssize_t.h>
#include <sys/fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

#include "ferret/base/env.h"

namespace ferret::segment_internal {

absl::StatusOr<std::unique_ptr<PackFile>> PackFile::Create(
    Env& env, absl::string_view path) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<WritableFile> file,
                        env.NewWritableFile(path));
  std::unique_ptr<PackFile> pack(
      new PackFile(env, std::string(path), std::move(file)));
  // Opened now so that contents already appended can be read while others
  // are still being written (pwrite is not buffered).
  pack->read_fd_ = open(pack->path_.c_str(), O_RDONLY | O_CLOEXEC);
  if (pack->read_fd_ < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("open(", pack->path_, ")"));
  }
  return pack;
}

PackFile::~PackFile() {
  if (read_fd_ >= 0) close(read_fd_);
  {
    absl::MutexLock lock(mu_);
    if (file_ != nullptr) file_->Close().IgnoreError();
  }
  env_.DeleteFile(path_).IgnoreError();
}

absl::StatusOr<uint64_t> PackFile::Append(absl::string_view data) {
  absl::MutexLock lock(mu_);
  if (file_ == nullptr) {
    return absl::FailedPreconditionError("pack file is closed for writing");
  }
  const auto offset = static_cast<uint64_t>(file_->size());
  ABSL_RETURN_IF_ERROR(file_->Append(data));
  return offset;
}

absl::Status PackFile::FinishWriting() {
  {
    absl::MutexLock lock(mu_);
    if (file_ != nullptr) {
      ABSL_RETURN_IF_ERROR(file_->Close());
      file_.reset();
    }
  }
  if (read_fd_ < 0) {
    read_fd_ = open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (read_fd_ < 0) {
      return absl::ErrnoToStatus(errno, absl::StrCat("open(", path_, ")"));
    }
  }
  return absl::OkStatus();
}

absl::Status PackFile::ReadAt(uint64_t offset, size_t length,
                              std::string& out) const {
  out.resize(length);
  size_t done = 0;
  while (done < length) {
    const ssize_t n = pread(read_fd_, out.data() + done, length - done,
                            static_cast<off_t>(offset + done));
    if (n < 0) {
      if (errno == EINTR) continue;
      return absl::ErrnoToStatus(errno, absl::StrCat("pread(", path_, ")"));
    }
    if (n == 0) {
      return absl::DataLossError("pack file is shorter than expected");
    }
    done += static_cast<size_t>(n);
  }
  return absl::OkStatus();
}

uint64_t PackFile::size() const {
  absl::MutexLock lock(mu_);
  return file_ != nullptr ? static_cast<uint64_t>(file_->size()) : 0;
}

}  // namespace ferret::segment_internal

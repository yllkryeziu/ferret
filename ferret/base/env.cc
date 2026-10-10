// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/base/env.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/_types/_ssize_t.h>
#include <sys/dirent.h>
#include <sys/fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "ferret/base/mapped_file.h"

namespace ferret {
namespace {

// macOS rejects a single read or write of more than INT_MAX bytes with
// EINVAL, so large transfers are split.
constexpr size_t kMaxTransferBytes = size_t{1} << 30;

absl::Status PosixError(int error_number, absl::string_view call,
                        absl::string_view path) {
  return absl::ErrnoToStatus(error_number, absl::StrCat(call, "(", path, ")"));
}

enum class SyncKind : uint8_t { kBarrier, kFull };

absl::Status SyncDescriptor(int fd, SyncKind kind, absl::string_view path) {
#if defined(F_BARRIERFSYNC) && defined(F_FULLFSYNC)
  // Both fail on file systems that do not support them (for example some
  // network mounts); fsync is then the best available.
  if (fcntl(fd, kind == SyncKind::kFull ? F_FULLFSYNC : F_BARRIERFSYNC) == 0) {
    return absl::OkStatus();
  }
#else
  static_cast<void>(kind);
#endif
  if (fsync(fd) != 0) return PosixError(errno, "fsync", path);
  return absl::OkStatus();
}

class PosixWritableFile final : public WritableFile {
 public:
  PosixWritableFile(std::string path, int fd)
      : path_(std::move(path)), fd_(fd) {}
  PosixWritableFile(const PosixWritableFile&) = delete;
  PosixWritableFile& operator=(const PosixWritableFile&) = delete;
  ~PosixWritableFile() override {
    if (fd_ >= 0) close(fd_);
  }

  absl::Status Append(absl::string_view data) override {
    ABSL_RETURN_IF_ERROR(Write(size_, data));
    size_ += static_cast<int64_t>(data.size());
    return absl::OkStatus();
  }

  absl::Status WriteAt(int64_t offset, absl::string_view data) override {
    if (offset < 0 || static_cast<int64_t>(data.size()) > size_ - offset) {
      return absl::InvalidArgumentError(absl::StrCat(
          "pwrite(", path_, "): ", data.size(), " bytes at offset ", offset,
          " extend past file size ", size_));
    }
    return Write(offset, data);
  }

  int64_t size() const override { return size_; }

  absl::Status Sync() override {
    ABSL_RETURN_IF_ERROR(CheckOpen("sync"));
    return SyncDescriptor(fd_, SyncKind::kBarrier, path_);
  }

  absl::Status FullSync() override {
    ABSL_RETURN_IF_ERROR(CheckOpen("sync"));
    return SyncDescriptor(fd_, SyncKind::kFull, path_);
  }

  absl::Status Close() override {
    ABSL_RETURN_IF_ERROR(CheckOpen("close"));
    const int result = close(fd_);
    // The descriptor is released even when close reports an error, so it
    // must not be closed again.
    fd_ = -1;
    if (result != 0) return PosixError(errno, "close", path_);
    return absl::OkStatus();
  }

 private:
  absl::Status CheckOpen(absl::string_view call) const {
    if (fd_ >= 0) return absl::OkStatus();
    return absl::FailedPreconditionError(
        absl::StrCat(call, "(", path_, "): file already closed"));
  }

  absl::Status Write(int64_t offset, absl::string_view data) {
    ABSL_RETURN_IF_ERROR(CheckOpen("pwrite"));
    while (!data.empty()) {
      const ssize_t n = pwrite(
          fd_, data.data(), std::min(data.size(), kMaxTransferBytes), offset);
      if (n < 0) {
        if (errno == EINTR) continue;
        return PosixError(errno, "pwrite", path_);
      }
      data.remove_prefix(static_cast<size_t>(n));
      offset += n;
    }
    return absl::OkStatus();
  }

  const std::string path_;
  int fd_;
  int64_t size_ = 0;
};

class PosixFileLock final : public FileLock {
 public:
  explicit PosixFileLock(int fd) : fd_(fd) {}
  PosixFileLock(const PosixFileLock&) = delete;
  PosixFileLock& operator=(const PosixFileLock&) = delete;
  // Closing the only descriptor of the open file description releases the
  // flock.
  ~PosixFileLock() override { close(fd_); }

 private:
  const int fd_;
};

class PosixEnv final : public Env {
 public:
  PosixEnv() = default;
  PosixEnv(const PosixEnv&) = delete;
  PosixEnv& operator=(const PosixEnv&) = delete;

  absl::StatusOr<std::unique_ptr<WritableFile>> NewWritableFile(
      absl::string_view path) override {
    std::string path_string(path);
    const int fd = open(path_string.c_str(),
                        O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return PosixError(errno, "open", path);
    return std::make_unique<PosixWritableFile>(std::move(path_string), fd);
  }

  absl::StatusOr<std::string> ReadFile(absl::string_view path) override {
    ABSL_ASSIGN_OR_RETURN(
        std::optional<std::string> content,
        ReadFileIfAtMost(path, std::numeric_limits<int64_t>::max()));
    if (!content.has_value()) {
      return absl::InternalError(absl::StrCat("read(", path, "): too large"));
    }
    return *std::move(content);
  }

  absl::StatusOr<std::optional<std::string>> ReadFileIfAtMost(
      absl::string_view path, int64_t max_bytes) override {
    const std::string path_string(path);
    const int fd = open(path_string.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return PosixError(errno, "open", path);
    absl::Cleanup close_fd = [fd] { close(fd); };
    struct stat st;
    if (fstat(fd, &st) != 0) return PosixError(errno, "fstat", path);
    if (static_cast<int64_t>(st.st_size) > max_bytes) return std::nullopt;
    // One byte of slack lets the last read see end-of-file without growing
    // the buffer; a file that grows meanwhile is still read to its end.
    std::string content(static_cast<size_t>(st.st_size) + 1, '\0');
    size_t length = 0;
    while (true) {
      if (length == content.size()) content.resize(2 * content.size());
      const ssize_t n =
          read(fd, content.data() + length,
               std::min(content.size() - length, kMaxTransferBytes));
      if (n < 0) {
        if (errno == EINTR) continue;
        return PosixError(errno, "read", path);
      }
      if (n == 0) break;
      length += static_cast<size_t>(n);
    }
    content.resize(length);
    if (static_cast<int64_t>(length) > max_bytes) return std::nullopt;
    return std::optional<std::string>(std::move(content));
  }

  absl::StatusOr<std::unique_ptr<MappedFile>> NewMappedFile(
      absl::string_view path) override {
    return MappedFile::Open(path);
  }

  absl::StatusOr<int64_t> GetFileSize(absl::string_view path) override {
    const std::string path_string(path);
    struct stat st;
    if (stat(path_string.c_str(), &st) != 0) {
      return PosixError(errno, "stat", path);
    }
    return static_cast<int64_t>(st.st_size);
  }

  absl::StatusOr<bool> FileExists(absl::string_view path) override {
    const std::string path_string(path);
    struct stat st;
    if (stat(path_string.c_str(), &st) == 0) return true;
    if (errno == ENOENT || errno == ENOTDIR) return false;
    return PosixError(errno, "stat", path);
  }

  absl::StatusOr<std::vector<std::string>> GetChildren(
      absl::string_view path) override {
    // fdopendir rather than opendir, so that the descriptor has O_CLOEXEC
    // from the start.
    const std::string path_string(path);
    const int fd =
        open(path_string.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return PosixError(errno, "open", path);
    DIR* const dir = fdopendir(fd);
    if (dir == nullptr) {
      const int error = errno;
      close(fd);
      return PosixError(error, "fdopendir", path);
    }
    absl::Cleanup close_dir = [dir] { closedir(dir); };
    std::vector<std::string> children;
    while (true) {
      errno = 0;
      const dirent* const entry = readdir(dir);
      if (entry == nullptr) {
        if (errno != 0) return PosixError(errno, "readdir", path);
        break;
      }
      const absl::string_view name = entry->d_name;
      if (name != "." && name != "..") children.emplace_back(name);
    }
    std::sort(children.begin(), children.end());
    return children;
  }

  absl::Status CreateDir(absl::string_view path) override {
    const std::string path_string(path);
    // Creates each prefix that ends before a '/', then the whole path.
    for (size_t end = path_string.find('/', 1);;
         end = path_string.find('/', end + 1)) {
      const std::string dir = path_string.substr(0, end);
      if (mkdir(dir.c_str(), 0755) != 0) {
        // An existing directory is fine whatever mkdir reports for it
        // (EEXIST, or EISDIR for "/" on macOS); an existing file is not.
        const int error = errno;
        struct stat st;
        if (stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
          return PosixError(error, "mkdir", dir);
        }
      }
      if (end == std::string::npos) return absl::OkStatus();
    }
  }

  absl::Status DeleteFile(absl::string_view path) override {
    const std::string path_string(path);
    if (unlink(path_string.c_str()) != 0) {
      return PosixError(errno, "unlink", path);
    }
    return absl::OkStatus();
  }

  absl::Status RenameFile(absl::string_view from,
                          absl::string_view to) override {
    const std::string from_string(from);
    const std::string to_string(to);
    if (rename(from_string.c_str(), to_string.c_str()) != 0) {
      return absl::ErrnoToStatus(errno,
                                 absl::StrCat("rename(", from, ", ", to, ")"));
    }
    return absl::OkStatus();
  }

  absl::Status SyncDir(absl::string_view path) override {
    const std::string path_string(path);
    const int fd =
        open(path_string.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return PosixError(errno, "open", path);
    absl::Cleanup close_fd = [fd] { close(fd); };
    return SyncDescriptor(fd, SyncKind::kBarrier, path);
  }

  absl::StatusOr<std::unique_ptr<FileLock>> LockFile(
      absl::string_view path) override {
    const std::string path_string(path);
    const int fd =
        open(path_string.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) return PosixError(errno, "open", path);
    while (flock(fd, LOCK_EX | LOCK_NB) != 0) {
      if (errno == EINTR) continue;
      const int error = errno;
      close(fd);
      if (error == EWOULDBLOCK) {
        return absl::FailedPreconditionError(absl::StrCat(
            "flock(", path, "): held by another process or open file"));
      }
      return PosixError(error, "flock", path);
    }
    return std::make_unique<PosixFileLock>(fd);
  }
};

}  // namespace

Env* absl_nonnull Env::Default() {
  static absl::NoDestructor<PosixEnv> env;
  return env.get();
}

}  // namespace ferret

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Env: the interface for all index file I/O, and its POSIX implementation.
//
// The index is a cache, so its commit protocol needs ordering rather than
// durability: write a file, Sync() it, rename it into place, then SyncDir()
// before anything depends on the rename, and unlink obsolete files only after
// that. Every step goes through Env so that crash tests can run the same code
// over FaultInjectionEnv (ferret/base/testing/fault_injection_env.h), which
// discards whatever was not synced.
//
// Files are written with pwrite, never through a writable mapping: a store
// into a mapped page that the file system cannot back (disk full) raises
// SIGBUS instead of returning ENOSPC.
//
// Errors carry the failing call and the path, for example
// "rename(/idx/7.fseg.tmp, /idx/7.fseg): No such file or directory", and
// map errno through absl::ErrnoToStatus, so ENOSPC and EDQUOT become
// ResourceExhausted. Every descriptor is opened with O_CLOEXEC so that it
// cannot leak into git or ripgrep children; a leaked LOCK descriptor would
// keep the index locked after the daemon died.

#ifndef FERRET_BASE_ENV_H_
#define FERRET_BASE_ENV_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "ferret/base/mapped_file.h"

namespace ferret {

// A file being written, returned by Env::NewWritableFile. Every write is a
// pwrite at an explicit offset. Destroying a file that was not closed closes
// it and drops any error. Not thread-safe.
class WritableFile {
 public:
  WritableFile() = default;
  WritableFile(const WritableFile&) = delete;
  WritableFile& operator=(const WritableFile&) = delete;
  virtual ~WritableFile() = default;

  // Writes `data` at offset size(), extending the file.
  virtual absl::Status Append(absl::string_view data) = 0;

  // Overwrites bytes already written, for example a header whose checksums
  // are known only at the end. InvalidArgument unless
  // 0 <= offset && offset + data.size() <= size().
  virtual absl::Status WriteAt(int64_t offset, absl::string_view data) = 0;

  // The file's length: the number of bytes appended so far.
  virtual int64_t size() const = 0;

  // Hands every write so far to the drive, ordered before any later write:
  // fcntl(F_BARRIERFSYNC) on macOS, where plain fsync neither flushes nor
  // orders the drive cache; fsync where F_BARRIERFSYNC is unavailable or
  // unsupported. This is the sync the commit protocol needs before a rename.
  virtual absl::Status Sync() = 0;

  // Flushes every write so far to stable storage, including the drive's
  // cache: fcntl(F_FULLFSYNC) on macOS, fsync elsewhere. The drive flushes
  // its whole cache, so this also makes durable whatever earlier Sync() and
  // Env::SyncDir() calls handed to it. It costs milliseconds, so it is used
  // only on clean shutdown.
  virtual absl::Status FullSync() = 0;

  // Closes the file. Every later call except size() fails with
  // FailedPrecondition.
  virtual absl::Status Close() = 0;
};

// An exclusive lock on a file, released when this object is destroyed (or
// when the process dies). Each Env returns its own implementation.
// Thread-safe.
class FileLock {
 public:
  FileLock() = default;
  FileLock(const FileLock&) = delete;
  FileLock& operator=(const FileLock&) = delete;
  virtual ~FileLock() = default;
};

// The file system operations the index uses. Paths are passed to the
// operating system unchanged. Implementations must allow concurrent calls.
// Thread-safe.
class Env {
 public:
  Env() = default;
  Env(const Env&) = delete;
  Env& operator=(const Env&) = delete;
  virtual ~Env() = default;

  // The POSIX implementation. Never destroyed.
  static Env* absl_nonnull Default();

  // Creates `path` for writing, truncating any existing file. The new
  // directory entry is durable only after SyncDir() of its parent.
  virtual absl::StatusOr<std::unique_ptr<WritableFile>> NewWritableFile(
      absl::string_view path) = 0;

  // Returns the whole content of `path`. Meant for small files such as the
  // MANIFEST; map large ones.
  virtual absl::StatusOr<std::string> ReadFile(absl::string_view path) = 0;

  // Like ReadFile(), but returns std::nullopt without reading when the file
  // is larger than `max_bytes`, or turns out to be larger while it is read.
  // The POSIX implementation opens the file once, where GetFileSize() and
  // ReadFile() look the path up twice, which matters for the bulk build's
  // hundreds of thousands of files.
  virtual absl::StatusOr<std::optional<std::string>> ReadFileIfAtMost(
      absl::string_view path, int64_t max_bytes) {
    absl::StatusOr<int64_t> size = GetFileSize(path);
    if (!size.ok()) return size.status();
    if (*size > max_bytes) return std::nullopt;
    absl::StatusOr<std::string> content = ReadFile(path);
    if (!content.ok()) return content.status();
    if (static_cast<int64_t>(content->size()) > max_bytes) return std::nullopt;
    return std::optional<std::string>(std::move(*content));
  }

  // Maps `path` read-only; see mapped_file.h.
  virtual absl::StatusOr<std::unique_ptr<MappedFile>> NewMappedFile(
      absl::string_view path) = 0;

  // Returns the size of `path` in bytes.
  virtual absl::StatusOr<int64_t> GetFileSize(absl::string_view path) = 0;

  // Returns whether `path` names an existing file or directory. A missing
  // entry, or a parent that is not a directory, yields false; any other
  // failure (for example EACCES) is an error.
  virtual absl::StatusOr<bool> FileExists(absl::string_view path) = 0;

  // Returns the names (not paths) of the entries of directory `path`,
  // excluding "." and "..", sorted bytewise.
  virtual absl::StatusOr<std::vector<std::string>> GetChildren(
      absl::string_view path) = 0;

  // Creates directory `path` and any missing parents. Succeeds if `path`
  // already is a directory.
  virtual absl::Status CreateDir(absl::string_view path) = 0;

  // Removes the directory entry `path`, which must not be a directory. Open
  // files and mappings of it stay valid.
  virtual absl::Status DeleteFile(absl::string_view path) = 0;

  // Renames `from` to `to`, atomically replacing any existing file at `to`:
  // after a crash, `to` names either the old or the new file. The rename is
  // durable only after SyncDir() of both parent directories.
  virtual absl::Status RenameFile(absl::string_view from,
                                  absl::string_view to) = 0;

  // Makes the creates, renames and deletes of entries in directory `path` so
  // far durable, with the same barrier semantics as WritableFile::Sync().
  virtual absl::Status SyncDir(absl::string_view path) = 0;

  // Takes an exclusive lock on `path` without blocking, creating the file if
  // needed: flock(LOCK_EX | LOCK_NB). FailedPrecondition if the lock is held,
  // including through another LockFile() in this process. flock locks belong
  // to the open file description, so unlike fcntl record locks they exclude
  // a second opener in the same process, and closing some unrelated
  // descriptor of the file does not silently drop them.
  virtual absl::StatusOr<std::unique_ptr<FileLock>> LockFile(
      absl::string_view path) = 0;
};

}  // namespace ferret

#endif  // FERRET_BASE_ENV_H_

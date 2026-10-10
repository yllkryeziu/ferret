// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// A scratch file that holds copies of the contents the bulk build has read.
//
// Opening 460,000 files is slow on macOS (about 15 s for Chromium, even with
// everything in the page cache, because the vnode cache is smaller than the
// file count), so the build reads each file once. Threads append the
// contents here in the order they finish; after the files are ranked, the
// build reads them back in rank order with pread, which neither opens files
// nor maps them (a mapping would count toward the process's resident size).
// The file is deleted when the PackFile is destroyed.

#ifndef FERRET_INDEX_INTERNAL_PACK_FILE_H_
#define FERRET_INDEX_INTERNAL_PACK_FILE_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

#include "ferret/base/env.h"

namespace ferret::segment_internal {

class PackFile final {
 public:
  // Creates the file at `path`, replacing any file there. `env` must outlive
  // the PackFile.
  static absl::StatusOr<std::unique_ptr<PackFile>> Create(
      Env& env, absl::string_view path);

  PackFile(const PackFile&) = delete;
  PackFile& operator=(const PackFile&) = delete;
  // Closes and deletes the file.
  ~PackFile();

  // Appends `data` and returns its offset. Thread-safe. Not after
  // FinishWriting().
  absl::StatusOr<uint64_t> Append(absl::string_view data)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Closes the writer. Reads stay possible.
  absl::Status FinishWriting() ABSL_LOCKS_EXCLUDED(mu_);

  // Replaces `out` with the `length` bytes at `offset`. Thread-safe. The
  // bytes must have been appended already (Append() returned), also while
  // other threads append.
  absl::Status ReadAt(uint64_t offset, size_t length, std::string& out) const;

  // The number of bytes appended.
  uint64_t size() const ABSL_LOCKS_EXCLUDED(mu_);

 private:
  PackFile(Env& env, std::string path, std::unique_ptr<WritableFile> file)
      : env_(env), path_(std::move(path)), file_(std::move(file)) {}

  Env& env_;
  const std::string path_;
  mutable absl::Mutex mu_;
  std::unique_ptr<WritableFile> file_ ABSL_GUARDED_BY(mu_);
  int read_fd_ = -1;
};

}  // namespace ferret::segment_internal

#endif  // FERRET_INDEX_INTERNAL_PACK_FILE_H_

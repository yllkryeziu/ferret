// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// A self-deleting temporary directory for tests.
//
// Created under $TEST_TMPDIR (set by `bazel test`), falling back to $TMPDIR.
// The path is canonicalized with realpath(), because on macOS /tmp and
// /var/folders are symlinks into /private and FSEvents reports canonical
// paths.

#ifndef FERRET_TESTING_TEMP_DIR_H_
#define FERRET_TESTING_TEMP_DIR_H_

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace ferret_testing {

// Not thread-safe.
class TempDir final {
 public:
  // Creates a fresh directory; CHECK-fails if that is impossible, which in a
  // test environment is a broken sandbox rather than a recoverable error.
  TempDir();
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  // Recursively deletes the directory.
  ~TempDir();

  // Canonical absolute path, without a trailing slash.
  const std::string& path() const { return path_; }

  // Returns path() + "/" + relative_path.
  std::string Join(absl::string_view relative_path) const;

  // Writes `content` to `relative_path`, creating parent directories.
  absl::Status WriteFile(absl::string_view relative_path,
                         absl::string_view content) const;

  // Creates `relative_path` and any missing parents.
  absl::Status MakeDirs(absl::string_view relative_path) const;

 private:
  std::string path_;
};

}  // namespace ferret_testing

#endif  // FERRET_TESTING_TEMP_DIR_H_

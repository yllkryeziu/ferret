// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Process-local identifiers for repo-relative paths.

#ifndef FERRET_WORKSPACE_PATH_TABLE_H_
#define FERRET_WORKSPACE_PATH_TABLE_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace ferret {

// An interned, process-local repo-relative path. Thread-compatible.
enum class PathId : uint32_t {};

// Stores normalized, slash-separated relative paths. All methods are
// thread-safe; callers own any returned string copies.
class PathTable final {
 public:
  PathTable() = default;
  PathTable(const PathTable&) = delete;
  PathTable& operator=(const PathTable&) = delete;
  PathTable(PathTable&&) = delete;
  PathTable& operator=(PathTable&&) = delete;

  // Rejects empty, absolute, dot, parent, empty-component, backslash and NUL
  // paths with InvalidArgument; otherwise returns the first-seen id.
  absl::StatusOr<PathId> Intern(absl::string_view path)
      ABSL_LOCKS_EXCLUDED(mu_);
  // Validates without inserting a path. Uses the same rules as Intern.
  static absl::Status Validate(absl::string_view path);
  // Returns nullopt if the path has not been interned.
  std::optional<PathId> Find(absl::string_view path) const
      ABSL_LOCKS_EXCLUDED(mu_);
  // Requires an id returned by Intern(). Returns a copy.
  std::string path(PathId id) const ABSL_LOCKS_EXCLUDED(mu_);
  // Returns the number of paths in the table.
  size_t size() const ABSL_LOCKS_EXCLUDED(mu_);

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, PathId> ids_ ABSL_GUARDED_BY(mu_);
  std::vector<std::string> paths_ ABSL_GUARDED_BY(mu_);
};

}  // namespace ferret

#endif  // FERRET_WORKSPACE_PATH_TABLE_H_

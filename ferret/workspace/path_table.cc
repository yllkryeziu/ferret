// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/workspace/path_table.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace ferret {
namespace {

bool ValidPath(absl::string_view path) {
  if (path.empty() || path.front() == '/' || path.find('\\') != path.npos ||
      path.find('\0') != path.npos) {
    return false;
  }
  while (!path.empty()) {
    const size_t end = path.find('/');
    const absl::string_view component = path.substr(0, end);
    if (component.empty() || component == "." || component == "..") {
      return false;
    }
    if (end == path.npos) return true;
    path.remove_prefix(end + 1);
  }
  return false;
}

}  // namespace

absl::Status PathTable::Validate(absl::string_view path) {
  if (!ValidPath(path)) {
    return absl::InvalidArgumentError("invalid repo-relative path");
  }
  return absl::OkStatus();
}

absl::StatusOr<PathId> PathTable::Intern(absl::string_view path) {
  absl::Status valid = Validate(path);
  if (!valid.ok()) return valid;
  absl::MutexLock lock(mu_);
  auto it = ids_.find(path);
  if (it != ids_.end()) return it->second;
  CHECK_LT(paths_.size(), uint64_t{1} << 32);
  const PathId id = static_cast<PathId>(paths_.size());
  paths_.emplace_back(path);
  ids_.emplace(paths_.back(), id);
  return id;
}

std::optional<PathId> PathTable::Find(absl::string_view path) const {
  absl::MutexLock lock(mu_);
  auto it = ids_.find(path);
  if (it == ids_.end()) return std::nullopt;
  return it->second;
}

std::string PathTable::path(PathId id) const {
  absl::MutexLock lock(mu_);
  DCHECK_LT(static_cast<size_t>(id), paths_.size());
  return paths_[static_cast<size_t>(id)];
}

size_t PathTable::size() const {
  absl::MutexLock lock(mu_);
  return paths_.size();
}

}  // namespace ferret

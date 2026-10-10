// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/internal/git_fields.h"

#include <cstdint>
#include <optional>

#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

namespace ferret::git_internal {

bool IsNormalizedRepoPath(absl::string_view path) {
  if (path.empty()) return false;
  for (const absl::string_view component : absl::StrSplit(path, '/')) {
    if (component.empty() || component == "." || component == ".." ||
        absl::EqualsIgnoreCase(component, ".git") ||
        absl::StrContains(component, '\0')) {
      return false;
    }
  }
  return true;
}

std::optional<uint32_t> ParseGitMode(absl::string_view field) {
  if (field.size() != 6) return std::nullopt;
  uint32_t mode = 0;
  for (const char c : field) {
    if (c < '0' || c > '7') return std::nullopt;
    mode = (mode << 3) | static_cast<uint32_t>(c - '0');
  }
  return mode;
}

}  // namespace ferret::git_internal

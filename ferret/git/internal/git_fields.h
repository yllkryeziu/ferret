// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Validation of the fields git prints in its machine-readable output.
//
// Paths read from `git ls-files` and `git status` are later joined to the
// worktree root and opened, so git output is treated as untrusted: a crafted
// index must not be able to name a file outside the worktree or inside
// $GIT_DIR. The path rule is the one git's own verify_path() applies before
// it writes an index entry.

#ifndef FERRET_GIT_INTERNAL_GIT_FIELDS_H_
#define FERRET_GIT_INTERNAL_GIT_FIELDS_H_

#include <cstdint>
#include <optional>

#include "absl/strings/string_view.h"

namespace ferret::git_internal {

// Returns true if `path` is a normalized repo-relative path: non-empty,
// '/'-separated, without a leading or trailing '/', and with no component
// that is empty, ".", "..", ".git" in any ASCII case, or contains a NUL byte.
bool IsNormalizedRepoPath(absl::string_view path);

// Parses a file mode as git prints it in porcelain output: exactly six octal
// digits, e.g. "100644". Returns nullopt for anything else.
std::optional<uint32_t> ParseGitMode(absl::string_view field);

}  // namespace ferret::git_internal

#endif  // FERRET_GIT_INTERNAL_GIT_FIELDS_H_

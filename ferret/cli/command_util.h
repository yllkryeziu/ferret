// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Helpers shared by the ferret subcommands.

#ifndef FERRET_CLI_COMMAND_UTIL_H_
#define FERRET_CLI_COMMAND_UTIL_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "ferret/index/segment.h"

namespace ferret {

// Returns where the index of the worktree at `root` lives by default:
// <cache>/ferret/<name>-<id>.fseg, where <cache> is $XDG_CACHE_HOME, or
// $HOME/.cache if that is unset, <name> is the worktree directory's name, and
// <id> is 12 hex digits of the SHA-1 of its canonical path, so each worktree
// has its own file. NotFound if `root` does not exist; FailedPrecondition if
// neither variable is set.
absl::StatusOr<std::string> DefaultIndexPath(absl::string_view root);

// Returns `index_path`, or DefaultIndexPath(root) if it is empty.
absl::StatusOr<std::string> ResolveIndexPath(absl::string_view root,
                                             absl::string_view index_path);

// Opens the segment at `path`: NotFound, suggesting `ferret index`, if there
// is none.
absl::StatusOr<std::unique_ptr<Segment>> OpenIndex(absl::string_view path);

// Formats a byte count for people: "812 B", "12.3 KiB", "4.0 GiB".
std::string HumanBytes(int64_t bytes);

}  // namespace ferret

#endif  // FERRET_CLI_COMMAND_UTIL_H_

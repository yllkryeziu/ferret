// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/watch/internal/canonical_root.h"

#include <_stdlib.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/syslimits.h>

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace ferret::watch_internal {

absl::StatusOr<std::string> CanonicalRoot(const std::string& root) {
  char resolved[PATH_MAX];
  if (realpath(root.c_str(), resolved) == nullptr) {
    return absl::ErrnoToStatus(errno, absl::StrCat("realpath(", root, ")"));
  }
  struct stat st;
  if (stat(resolved, &st) != 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("stat(", resolved, ")"));
  }
  if (!S_ISDIR(st.st_mode)) {
    return absl::InvalidArgumentError(
        absl::StrCat("watch root is not a directory: ", root));
  }
  return std::string(resolved);
}

}  // namespace ferret::watch_internal

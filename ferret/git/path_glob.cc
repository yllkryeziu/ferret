// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/path_glob.h"

#include <cstddef>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"

#include "ferret/git/internal/path_pattern.h"

namespace ferret {

absl::StatusOr<PathGlob> PathGlob::Parse(absl::string_view glob) {
  if (glob.empty() || glob == "!") {
    return absl::InvalidArgumentError("empty path glob");
  }
  if (absl::StrContains(glob, '\0')) {
    return absl::InvalidArgumentError("path glob contains a NUL byte");
  }
  return PathGlob(git_internal::PathPattern(glob, /*base=*/""));
}

bool PathGlob::Matches(absl::string_view path) const {
  bool matched = false;
  for (size_t slash = path.find('/'); !matched && slash != path.npos;
       slash = path.find('/', slash + 1)) {
    matched = pattern_.Matches(path.substr(0, slash), /*is_dir=*/true,
                               /*casefold=*/false);
  }
  if (!matched) {
    matched = pattern_.Matches(path, /*is_dir=*/false, /*casefold=*/false);
  }
  return matched != pattern_.negative();
}

}  // namespace ferret

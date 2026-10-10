// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/cli/command_util.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

#include "ferret/base/env.h"
#include "ferret/git/blob_hash.h"
#include "ferret/index/segment.h"

namespace ferret {

absl::StatusOr<std::string> DefaultIndexPath(absl::string_view root) {
  std::error_code error;
  const std::filesystem::path canonical =
      std::filesystem::canonical(std::filesystem::path(root), error);
  if (error) {
    return absl::NotFoundError(
        absl::StrCat("cannot resolve worktree ", root, ": ", error.message()));
  }
  std::string cache;
  if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
    cache = xdg;
  } else if (const char* home = std::getenv("HOME"); home && *home) {
    cache = absl::StrCat(home, "/.cache");
  } else {
    return absl::FailedPreconditionError(
        "neither XDG_CACHE_HOME nor HOME is set; pass --index");
  }
  std::string name = canonical.filename().string();
  if (name.empty()) name = "root";
  const std::string id = HashBlob(canonical.string()).ToHex().substr(0, 12);
  return absl::StrCat(cache, "/ferret/", name, "-", id, ".fseg");
}

absl::StatusOr<std::string> ResolveIndexPath(absl::string_view root,
                                             absl::string_view index_path) {
  if (!index_path.empty()) return std::string(index_path);
  return DefaultIndexPath(root);
}

absl::StatusOr<std::unique_ptr<Segment>> OpenIndex(absl::string_view path) {
  ABSL_ASSIGN_OR_RETURN(const bool exists, Env::Default()->FileExists(path));
  if (!exists) {
    return absl::NotFoundError(absl::StrCat(
        "no index at ", path, "; run `ferret index` in the worktree first"));
  }
  return Segment::Open(*Env::Default(), path);
}

std::string HumanBytes(int64_t bytes) {
  static constexpr const char* kUnits[] = {"KiB", "MiB", "GiB", "TiB"};
  if (bytes < 1024) return absl::StrCat(bytes, " B");
  double value = static_cast<double>(bytes) / 1024;
  int unit = 0;
  while (value >= 1024 && unit + 1 < static_cast<int>(std::size(kUnits))) {
    value /= 1024;
    ++unit;
  }
  return absl::StrFormat("%.1f %s", value, kUnits[unit]);
}

}  // namespace ferret

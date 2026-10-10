// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/watch/internal/cookie_directory.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace ferret::watch_internal {

absl::StatusOr<bool> MakeCookieDirectory(const std::string& path) {
  bool created = true;
  if (mkdir(path.c_str(), 0700) != 0) {
    if (errno != EEXIST) {
      return absl::ErrnoToStatus(errno, absl::StrCat("mkdir(", path, ")"));
    }
    created = false;
  }
  const std::string ignore = absl::StrCat(path, "/.gitignore");
  const int fd =
      open(ignore.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (fd < 0) {
    if (errno == EEXIST) return created;
    return absl::ErrnoToStatus(errno, absl::StrCat("open(", ignore, ")"));
  }
  static constexpr char kIgnoreAll[] = "*\n";
  const ssize_t written = write(fd, kIgnoreAll, sizeof(kIgnoreAll) - 1);
  const int error = errno;
  close(fd);
  if (written != sizeof(kIgnoreAll) - 1) {
    return absl::ErrnoToStatus(error, absl::StrCat("write(", ignore, ")"));
  }
  return created;
}

void RemoveCookieDirectory(const std::string& path) {
  unlink(absl::StrCat(path, "/.gitignore").c_str());
  rmdir(path.c_str());
}

}  // namespace ferret::watch_internal

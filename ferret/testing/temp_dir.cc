// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/testing/temp_dir.h"

#include <_stdio.h>
#include <_stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <ftw.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/_types/_ssize_t.h>
#include <sys/fcntl.h>
#include <sys/stat.h>
#include <sys/syslimits.h>
#include <unistd.h>

#include <cstddef>
#include <string>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace ferret_testing {
namespace {

int RemoveEntry(const char* path, const struct stat* /*sb*/, int /*type*/,
                struct FTW* /*ftw*/) {
  return remove(path);
}

std::string BaseTempDir() {
  for (const char* var : {"TEST_TMPDIR", "TMPDIR"}) {
    const char* value = getenv(var);
    if (value != nullptr && value[0] != '\0') return value;
  }
  return "/tmp";
}

}  // namespace

TempDir::TempDir() {
  std::string templ = absl::StrCat(BaseTempDir(), "/ferret-XXXXXX");
  CHECK(mkdtemp(templ.data()) != nullptr)
      << "mkdtemp(" << templ << "): " << strerror(errno);
  char resolved[PATH_MAX];
  CHECK(realpath(templ.c_str(), resolved) != nullptr)
      << "realpath(" << templ << "): " << strerror(errno);
  path_ = resolved;
}

TempDir::~TempDir() {
  nftw(path_.c_str(), RemoveEntry, 16, FTW_DEPTH | FTW_PHYS);
}

std::string TempDir::Join(absl::string_view relative_path) const {
  return absl::StrCat(path_, "/", relative_path);
}

absl::Status TempDir::MakeDirs(absl::string_view relative_path) const {
  std::string path = path_;
  size_t start = 0;
  while (start < relative_path.size()) {
    size_t slash = relative_path.find('/', start);
    if (slash == absl::string_view::npos) slash = relative_path.size();
    absl::StrAppend(&path, "/", relative_path.substr(start, slash - start));
    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
      return absl::ErrnoToStatus(errno, absl::StrCat("mkdir(", path, ")"));
    }
    start = slash + 1;
  }
  return absl::OkStatus();
}

absl::Status TempDir::WriteFile(absl::string_view relative_path,
                                absl::string_view content) const {
  const size_t slash = relative_path.rfind('/');
  if (slash != absl::string_view::npos) {
    ABSL_RETURN_IF_ERROR(MakeDirs(relative_path.substr(0, slash)));
  }
  const std::string path = Join(relative_path);
  const int fd =
      open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("open(", path, ")"));
  }
  size_t written = 0;
  while (written < content.size()) {
    const ssize_t n =
        write(fd, content.data() + written, content.size() - written);
    if (n < 0) {
      if (errno == EINTR) continue;
      const int saved = errno;
      close(fd);
      return absl::ErrnoToStatus(saved, absl::StrCat("write(", path, ")"));
    }
    written += static_cast<size_t>(n);
  }
  if (close(fd) != 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("close(", path, ")"));
  }
  return absl::OkStatus();
}

}  // namespace ferret_testing

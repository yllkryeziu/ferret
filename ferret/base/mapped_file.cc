// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/base/mapped_file.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/config.h"  // IWYU pragma: keep
#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "ferret/base/system_info.h"

namespace ferret {
namespace {

int ToMadvise(MappedFileAdvice advice) {
  switch (advice) {
    case MappedFileAdvice::kWillNeed:
      return MADV_WILLNEED;
    case MappedFileAdvice::kRandom:
      return MADV_RANDOM;
    case MappedFileAdvice::kSequential:
      return MADV_SEQUENTIAL;
    case MappedFileAdvice::kDontNeed:
      return MADV_DONTNEED;
  }
  LOG(FATAL) << "invalid MappedFileAdvice " << static_cast<int>(advice);
}

}  // namespace

absl::StatusOr<std::unique_ptr<MappedFile>> MappedFile::Open(
    absl::string_view path) {
  std::string path_string(path);
  const int fd = open(path_string.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("open(", path, ")"));
  }
  // The mapping holds its own reference to the file.
  absl::Cleanup close_fd = [fd] { close(fd); };
  struct stat st;
  if (fstat(fd, &st) != 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("fstat(", path, ")"));
  }
  if (!S_ISREG(st.st_mode)) {
    return absl::InvalidArgumentError(
        absl::StrCat("mmap(", path, "): not a regular file"));
  }
  const size_t size = static_cast<size_t>(st.st_size);
  if (size == 0) {
    return absl::WrapUnique(new MappedFile(std::move(path_string), nullptr, 0));
  }
  void* const addr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (addr == MAP_FAILED) {
    return absl::ErrnoToStatus(errno, absl::StrCat("mmap(", path, ")"));
  }
  return absl::WrapUnique(new MappedFile(std::move(path_string),
                                         static_cast<const char*>(addr), size));
}

MappedFile::~MappedFile() {
  if (data_ == nullptr) return;
  void* const addr = const_cast<char*>(data_);
#if !defined(NDEBUG) || defined(ABSL_HAVE_ADDRESS_SANITIZER)
  // MAP_FIXED atomically replaces the file pages with inaccessible anonymous
  // ones at the same address (the stale-view trap described in the header).
  void* const trap =
      mmap(addr, size_, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
  DCHECK(trap == addr) << "mmap(PROT_NONE) over " << path_ << ": "
                       << strerror(errno);
#else
  const int result = munmap(addr, size_);
  DCHECK_EQ(result, 0) << "munmap(" << path_ << "): " << strerror(errno);
#endif
}

absl::Status MappedFile::Advise(int64_t offset, int64_t length,
                                MappedFileAdvice advice) const {
  const int64_t size = static_cast<int64_t>(size_);
  if (offset < 0 || length < 0 || offset > size || length > size - offset) {
    return absl::InvalidArgumentError(
        absl::StrCat("madvise(", path_, "): range at offset ", offset,
                     " of length ", length, " exceeds file size ", size));
  }
  if (length == 0) return absl::OkStatus();
  // mmap placed data_ on a page boundary and mapped the last page whole, so
  // the widened range stays inside the mapping.
  const int64_t page = PageSize();
  const int64_t begin = offset / page * page;
  const int64_t end = (offset + length + page - 1) / page * page;
  if (madvise(const_cast<char*>(data_) + begin,
              static_cast<size_t>(end - begin), ToMadvise(advice)) != 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("madvise(", path_, ", offset ", offset, ", length ",
                            length, ")"));
  }
  return absl::OkStatus();
}

}  // namespace ferret

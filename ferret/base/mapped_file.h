// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// MappedFile: a whole file mapped read-only, and madvise over its sections.
//
// Segments are read in place from a PROT_READ, MAP_PRIVATE mapping. The
// descriptor is closed right after mmap, so an open segment costs no
// descriptor, and unlinking the file (the reaper's job) leaves the mapping
// intact. Segment files are written with pwrite and never change after they
// are renamed into place, so a published mapping cannot shrink underneath its
// readers; truncating a mapped file would turn accesses beyond the new end
// into SIGBUS.
//
// Index code maps files through Env::NewMappedFile (ferret/base/env.h), so
// crash tests can interpose; PosixEnv implements that with Open().
//
// Stale-view trap: in debug and ASan builds the destructor replaces the pages
// with an inaccessible mapping instead of unmapping them, so a view that
// outlives its MappedFile faults at once rather than reading whatever is
// mapped at that address next. That leaks address space, never memory.

#ifndef FERRET_BASE_MAPPED_FILE_H_
#define FERRET_BASE_MAPPED_FILE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace ferret {

// Hints for MappedFile::Advise, each mapping to the madvise flag of the same
// name.
enum class MappedFileAdvice : uint8_t {
  // Start reading the pages in now: issued on a new segment's dictionary,
  // chunk, doc and postings sections before queries can reach it.
  kWillNeed,
  // Disable read-ahead: content is touched at scattered candidate chunks.
  kRandom,
  // Read ahead aggressively: compaction copies content front to back.
  kSequential,
  // The pages will not be needed soon and may be reclaimed first.
  kDontNeed,
};

// Thread-safe.
class MappedFile final {
 public:
  // Maps the whole regular file at `path`. A zero-length file is valid and
  // has empty data(). Errors carry the failing call and the path, e.g.
  // NotFound "open(/idx/000001.fseg): No such file or directory".
  static absl::StatusOr<std::unique_ptr<MappedFile>> Open(
      absl::string_view path);

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  ~MappedFile();

  // The file's bytes as of Open(). Valid while *this is alive.
  //
  // string_view rather than Span<const uint8_t>: most readers hand these
  // bytes to text code (RE2, the literal finder, line counting), all of
  // which takes string_view, while fixed-layout records are read with
  // std::memcpy either way.
  absl::string_view data() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return absl::string_view(data_, size_);
  }

  // Applies `advice` to bytes [offset, offset + length), widened to whole
  // pages as madvise requires. InvalidArgument unless
  // 0 <= offset <= offset + length <= data().size(); an empty range is a
  // no-op. The advice is a hint: success does not mean the pages are
  // resident.
  absl::Status Advise(int64_t offset, int64_t length,
                      MappedFileAdvice advice) const;

 private:
  MappedFile(std::string path, const char* data, size_t size)
      : path_(std::move(path)), data_(data), size_(size) {}

  const std::string path_;
  // Null for an empty file, which mmap cannot map.
  const char* const data_;
  const size_t size_;
};

}  // namespace ferret

#endif  // FERRET_BASE_MAPPED_FILE_H_

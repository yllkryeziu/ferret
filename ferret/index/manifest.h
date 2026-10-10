// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The checksummed MANIFEST records committed segment metadata and view
// registrations.

#ifndef FERRET_INDEX_MANIFEST_H_
#define FERRET_INDEX_MANIFEST_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace ferret {

// The committed identity and expected layout of one segment.
struct ManifestSegment {
  // The number in NNNNNN.fseg.
  uint64_t file_number;
  // The expected byte length of the file.
  uint64_t file_size;
  // The expected document count in the segment header.
  uint32_t num_docs;
  bool operator==(const ManifestSegment&) const = default;
};

// A view registration retained across recovery.
struct ManifestView {
  // The canonical worktree root.
  std::string canonical_root;
  // The Git common directory shared by related worktrees.
  std::string git_common_dir;
  bool operator==(const ManifestView&) const = default;
};

// The complete committed directory state.
struct Manifest {
  // The monotonically increasing commit sequence.
  uint64_t manifest_seq = 0;
  // The first file number not persisted by an install.
  uint64_t next_file_number = 1;
  // The segments reachable from this commit.
  std::vector<ManifestSegment> segments;
  // The registered views, even if a segment was dropped during recovery.
  std::vector<ManifestView> views;
  bool operator==(const Manifest&) const = default;
};

// Format v1: magic, u32 version, u64 sequence, u64 next number, u32 segment
// count, (u64 number, u64 size, u32 docs)[], u32 view count, (u32 length,
// bytes root, u32 length, bytes git dir)[], u32 CRC32C of preceding bytes.
// All integers are little-endian. Returns InvalidArgument for unencodable data.
absl::StatusOr<std::string> EncodeManifest(const Manifest& manifest);

// DataLoss for corrupt input; FailedPrecondition for an unknown version.
absl::StatusOr<Manifest> DecodeManifest(absl::string_view bytes);

}  // namespace ferret

#endif  // FERRET_INDEX_MANIFEST_H_

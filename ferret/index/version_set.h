// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// VersionSet owns the MANIFEST and exclusive directory lock, recovering and
// publishing immutable segment snapshots.

#ifndef FERRET_INDEX_VERSION_SET_H_
#define FERRET_INDEX_VERSION_SET_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

#include "ferret/base/env.h"
#include "ferret/index/blob_registry.h"
#include "ferret/index/manifest.h"
#include "ferret/index/reaper.h"
#include "ferret/index/segment.h"

namespace ferret {

// The inverse of a segment's BlobId table: the documents of each blob, in
// increasing DocId order (usually one). Built once with the table, so that a
// view search can find the documents of a few named blobs without visiting
// every document. Immutable; thread-safe.
class DocsOfBlob final {
 public:
  // O(documents + the span of the BlobIds) when the ids are dense, otherwise
  // O(documents log documents).
  explicit DocsOfBlob(const std::vector<BlobId>& blob_of_doc);

  // The DocIds whose blob is `blob`, increasing; empty for any other blob.
  // O(1) when dense, O(log distinct blobs) otherwise.
  absl::Span<const DocId> Docs(BlobId blob) const;
  // Whether lookups are O(1), for tests.
  bool dense() const { return dense_; }
  // Heap bytes held.
  size_t memory_bytes() const;

 private:
  bool dense_ = true;
  // The smallest BlobId, for the dense table.
  uint32_t min_blob_ = 0;
  // Every DocId, grouped by blob in increasing blob order.
  std::vector<DocId> docs_;
  // Dense: group g of blob min_blob_ + g is docs_[starts_[g], starts_[g + 1]).
  // Sparse: group g belongs to keys_[g], and starts_ has the same shape.
  std::vector<uint32_t> starts_;
  std::vector<uint32_t> keys_;
};

// A mapped segment together with its persisted identity.
struct VersionSegment {
  // The metadata checked against the opened file.
  ManifestSegment metadata;
  // The mapping held alive by readers of this version.
  std::shared_ptr<const Segment> segment;
  // Dense process-local ids indexed by segment DocId. Built once when the
  // segment is opened or installed and shared by every Version that keeps
  // the segment, so installs copy a pointer, not 4 B per document.
  std::shared_ptr<const std::vector<BlobId>> blob_of_doc;
  // The inverse of blob_of_doc, built with it. Null when the caller did not
  // build one (a search then visits every document instead).
  std::shared_ptr<const DocsOfBlob> docs_of_blob;
};

// A published snapshot. Its fields are never modified after publication.
// Thread-safe.
struct Version {
  // The sequence of the MANIFEST that published this snapshot.
  uint64_t manifest_seq = 0;
  // Live segments in MANIFEST order: replacements occupy the first retired
  // input slot, so equal-key duplicates keep their original source priority.
  std::vector<VersionSegment> segments;
  // The registered views, including those needing a resync.
  std::vector<ManifestView> views;
};

// A MANIFEST entry that recovery could not open.
struct DroppedSegment {
  // The number of the unusable segment.
  uint64_t file_number;
  // The error encountered while opening it.
  std::string reason;
};

// Changes made while reconciling the directory on open.
struct RecoveryReport {
  // Listed segments removed from the corrected MANIFEST.
  std::vector<DroppedSegment> dropped_segments;
  // Names of orphan or temporary files removed after correction.
  std::vector<std::string> deleted_files;
  // True when a missing or corrupt MANIFEST reset a numbered index.
  bool manifest_reset = false;
};

// An atomic change to segments and view registrations.
struct VersionEdit {
  // Paths returned by NewSegmentFile(), written and synced by SegmentWriter.
  std::vector<std::string> add;
  // Unique live file numbers to retire after the MANIFEST commit.
  std::vector<uint64_t> remove;
  // An absent value preserves the existing views; an empty value clears them.
  std::optional<std::vector<ManifestView>> views;
};

// Owns the exclusive LOCK. The caller serializes Install and NewSegmentFile;
// current() is safe to call concurrently and only locks to copy its pointer.
// The Env and Reaper must outlive this object and every snapshot it returns.
// Thread-compatible.
class VersionSet final {
 public:
  // Creates `dir` if needed and holds its exclusive LOCK until destruction.
  // An optional registry shares ids with the caller; otherwise this set
  // creates its own registry for standalone use.
  // Reads or initializes MANIFEST, validates listed segment sizes, headers,
  // footers and document counts, corrects missing/corrupt entries, then
  // removes orphan and temporary files. Does not scan section CRCs. Returns
  // I/O errors, FailedPrecondition for a held lock or newer MANIFEST format,
  // and errors from writing the corrected MANIFEST or cleaning orphans. A
  // corrupt or absent MANIFEST resets a numbered index instead of failing.
  // The report lists dropped segments and deleted files, and flags a reset;
  // callers must resync affected views for dropped segments or a reset.
  static absl::StatusOr<std::pair<std::unique_ptr<VersionSet>, RecoveryReport>>
  Open(Env& env, absl::string_view dir, Reaper& reaper,
       std::shared_ptr<BlobRegistry> blobs = nullptr);

  VersionSet(const VersionSet&) = delete;
  VersionSet& operator=(const VersionSet&) = delete;
  ~VersionSet() = default;

  // Reserves a monotone file number in memory and returns its .fseg.tmp
  // path. The caller writes and syncs it before Install(). Reservations not
  // committed by an install may be reused after restart. Returns
  // ResourceExhausted at the number limit, or FailedPrecondition if reopening
  // is required. Serialize with Install().
  absl::StatusOr<std::string> NewSegmentFile();
  // Releases an abandoned reservation. The caller removes its temporary
  // file separately. Serialize with installs and NewSegmentFile().
  void DiscardSegmentFile(absl::string_view path);
  // Requires unique reserved, written and synced temp segments in add, and
  // unique live numbers in remove. Absent views preserve current registrations.
  // Renames and opens additions, syncs the directory when adding segments,
  // writes and syncs MANIFEST.tmp, renames it over MANIFEST, syncs the
  // directory, publishes the version, then marks removals obsolete. Returns
  // InvalidArgument for invalid edits, ResourceExhausted for sequence overflow,
  // or an I/O or segment validation error. Before the MANIFEST rename, errors
  // leave the old version usable and clean up additions best-effort; after
  // the rename, errors require reopening before any allocation or install.
  // Serialize with NewSegmentFile() and other installs.
  absl::Status Install(const VersionEdit& edit);
  // True if an install failed after renaming the MANIFEST. No further
  // allocations or installs are allowed until reopening. Serialize with
  // Install() and NewSegmentFile().
  bool reopen_required() const;
  // Copies the immutable published snapshot under a short mutex. Safe to
  // call concurrently with installs; the result retains its segment mappings.
  std::shared_ptr<const Version> current() const
      ABSL_LOCKS_EXCLUDED(current_mu_);

 private:
  VersionSet(Env& env, std::string dir, Reaper& reaper,
             std::unique_ptr<FileLock> lock,
             std::shared_ptr<BlobRegistry> blobs);

  std::string SegmentPath(uint64_t number, bool temporary) const;
  // `renamed` is set when the MANIFEST rename succeeds, even if SyncDir fails.
  absl::Status WriteManifest(const Manifest& manifest, bool* renamed = nullptr);
  absl::StatusOr<std::pair<ManifestSegment, std::shared_ptr<ReapedSegment>>>
  InstallSegment(uint64_t number, absl::string_view temp);
  absl::StatusOr<std::shared_ptr<ReapedSegment>> OpenSegment(
      const ManifestSegment& metadata);
  absl::Status Recover(RecoveryReport& report);

  VersionSegment MapSegment(const ManifestSegment& metadata,
                            std::shared_ptr<const Segment> segment);

  Env& env_;
  std::shared_ptr<BlobRegistry> blobs_;
  std::string dir_;
  Reaper& reaper_;
  std::unique_ptr<FileLock> lock_;
  uint64_t next_file_number_ = 1;
  bool reopen_required_ = false;
  absl::flat_hash_set<uint64_t> reserved_;
  std::vector<std::pair<uint64_t, std::shared_ptr<ReapedSegment>>> live_;
  mutable absl::Mutex current_mu_;
  std::shared_ptr<const Version> current_ ABSL_GUARDED_BY(current_mu_);
};

}  // namespace ferret

#endif  // FERRET_INDEX_VERSION_SET_H_

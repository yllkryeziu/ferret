// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/version_set.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

#include "ferret/base/env.h"
#include "ferret/index/blob_registry.h"
#include "ferret/index/manifest.h"
#include "ferret/index/reaper.h"
#include "ferret/index/segment.h"
#include "ferret/postings/ids.h"

namespace ferret {
namespace {

// Only canonical numbered segment names belong to this index.
bool ParseNumber(absl::string_view name, uint64_t& number) {
  if (!name.ends_with(".fseg")) return false;
  absl::string_view digits = name.substr(0, name.size() - 5);
  if (digits.size() < 6 || !absl::SimpleAtoi(digits, &number) || number == 0) {
    return false;
  }
  return absl::StrFormat("%06u.fseg", number) == name;
}

}  // namespace

DocsOfBlob::DocsOfBlob(const std::vector<BlobId>& blob_of_doc) {
  const size_t n = blob_of_doc.size();
  docs_.resize(n);
  if (n == 0) {
    starts_.push_back(0);
    return;
  }
  uint32_t lo = static_cast<uint32_t>(blob_of_doc[0]);
  uint32_t hi = lo;
  for (const BlobId blob : blob_of_doc) {
    lo = std::min(lo, static_cast<uint32_t>(blob));
    hi = std::max(hi, static_cast<uint32_t>(blob));
  }
  const uint64_t span = uint64_t{hi} - lo + 1;
  dense_ = span <= 4 * uint64_t{n} + 4096;
  if (dense_) {
    min_blob_ = lo;
    starts_.assign(span + 1, 0);
    for (const BlobId blob : blob_of_doc) {
      ++starts_[static_cast<uint32_t>(blob) - lo + 1];
    }
    for (size_t g = 0; g < span; ++g) starts_[g + 1] += starts_[g];
    std::vector<uint32_t> next(starts_.begin(), starts_.end() - 1);
    for (DocId doc = 0; doc < n; ++doc) {
      docs_[next[static_cast<uint32_t>(blob_of_doc[doc]) - lo]++] = doc;
    }
    return;
  }
  for (DocId doc = 0; doc < n; ++doc) docs_[doc] = doc;
  std::sort(docs_.begin(), docs_.end(), [&](DocId a, DocId b) {
    return std::pair(blob_of_doc[a], a) < std::pair(blob_of_doc[b], b);
  });
  for (size_t i = 0; i < n; ++i) {
    const uint32_t blob = static_cast<uint32_t>(blob_of_doc[docs_[i]]);
    if (keys_.empty() || keys_.back() != blob) {
      keys_.push_back(blob);
      starts_.push_back(static_cast<uint32_t>(i));
    }
  }
  starts_.push_back(static_cast<uint32_t>(n));
}

absl::Span<const DocId> DocsOfBlob::Docs(BlobId blob) const {
  const uint32_t id = static_cast<uint32_t>(blob);
  size_t group;
  if (dense_) {
    if (id < min_blob_ || id - min_blob_ + 1 >= starts_.size()) return {};
    group = id - min_blob_;
  } else {
    const auto it = std::lower_bound(keys_.begin(), keys_.end(), id);
    if (it == keys_.end() || *it != id) return {};
    group = static_cast<size_t>(it - keys_.begin());
  }
  return absl::MakeConstSpan(docs_).subspan(
      starts_[group], starts_[group + 1] - starts_[group]);
}

size_t DocsOfBlob::memory_bytes() const {
  return sizeof(*this) + docs_.capacity() * sizeof(DocId) +
         starts_.capacity() * sizeof(uint32_t) +
         keys_.capacity() * sizeof(uint32_t);
}

VersionSet::VersionSet(Env& env, std::string dir, Reaper& reaper,
                       std::unique_ptr<FileLock> lock,
                       std::shared_ptr<BlobRegistry> blobs)
    : env_(env),
      blobs_(blobs ? std::move(blobs) : std::make_shared<BlobRegistry>()),
      dir_(std::move(dir)),
      reaper_(reaper),
      lock_(std::move(lock)),
      current_(std::make_shared<Version>()) {}

VersionSegment VersionSet::MapSegment(const ManifestSegment& metadata,
                                      std::shared_ptr<const Segment> segment) {
  auto blob_of_doc = std::make_shared<std::vector<BlobId>>();
  blob_of_doc->reserve(segment->num_docs());
  for (DocId doc = 0; doc < segment->num_docs(); ++doc) {
    blob_of_doc->push_back(blobs_->Intern(segment->doc(doc).oid));
  }
  auto docs_of_blob = std::make_shared<const DocsOfBlob>(*blob_of_doc);
  return VersionSegment{.metadata = metadata,
                        .segment = std::move(segment),
                        .blob_of_doc = std::move(blob_of_doc),
                        .docs_of_blob = std::move(docs_of_blob)};
}

std::string VersionSet::SegmentPath(uint64_t number, bool temporary) const {
  return absl::StrCat(dir_, "/", absl::StrFormat("%06u.fseg", number),
                      temporary ? ".tmp" : "");
}

absl::Status VersionSet::WriteManifest(const Manifest& manifest,
                                       bool* renamed) {
  if (renamed != nullptr) *renamed = false;
  ABSL_ASSIGN_OR_RETURN(std::string data, EncodeManifest(manifest));
  const std::string path = absl::StrCat(dir_, "/MANIFEST.tmp");
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<WritableFile> file,
                        env_.NewWritableFile(path));
  ABSL_RETURN_IF_ERROR(file->Append(data));
  ABSL_RETURN_IF_ERROR(file->Sync());
  ABSL_RETURN_IF_ERROR(file->Close());
  ABSL_RETURN_IF_ERROR(env_.RenameFile(path, absl::StrCat(dir_, "/MANIFEST")));
  if (renamed != nullptr) *renamed = true;
  return env_.SyncDir(dir_);
}

absl::StatusOr<std::shared_ptr<ReapedSegment>> VersionSet::OpenSegment(
    const ManifestSegment& metadata) {
  std::string path = SegmentPath(metadata.file_number, false);
  ABSL_ASSIGN_OR_RETURN(int64_t size, env_.GetFileSize(path));
  if (size < 0 || static_cast<uint64_t>(size) != metadata.file_size) {
    return absl::DataLossError("segment size differs from MANIFEST");
  }
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Segment> segment,
                        Segment::Open(env_, path));
  if (segment->num_docs() != metadata.num_docs) {
    return absl::DataLossError("segment document count differs from MANIFEST");
  }
  return std::make_shared<ReapedSegment>(env_, reaper_, std::move(path),
                                         std::move(segment));
}

absl::StatusOr<std::pair<ManifestSegment, std::shared_ptr<ReapedSegment>>>
VersionSet::InstallSegment(uint64_t number, absl::string_view temp) {
  std::string path = SegmentPath(number, false);
  ABSL_RETURN_IF_ERROR(env_.RenameFile(temp, path));
  ABSL_ASSIGN_OR_RETURN(int64_t size, env_.GetFileSize(path));
  if (size < 0) return absl::DataLossError("negative segment size");
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Segment> segment,
                        Segment::Open(env_, path));
  ManifestSegment metadata{number, static_cast<uint64_t>(size),
                           segment->num_docs()};
  return std::make_pair(
      metadata, std::make_shared<ReapedSegment>(env_, reaper_, std::move(path),
                                                std::move(segment)));
}

absl::StatusOr<std::pair<std::unique_ptr<VersionSet>, RecoveryReport>>
VersionSet::Open(Env& env, absl::string_view dir, Reaper& reaper,
                 std::shared_ptr<BlobRegistry> blobs) {
  ABSL_RETURN_IF_ERROR(env.CreateDir(dir));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<FileLock> lock,
                        env.LockFile(absl::StrCat(dir, "/LOCK")));
  auto set = std::unique_ptr<VersionSet>(new VersionSet(
      env, std::string(dir), reaper, std::move(lock), std::move(blobs)));
  RecoveryReport report;
  ABSL_RETURN_IF_ERROR(set->Recover(report));
  return std::make_pair(std::move(set), std::move(report));
}

absl::Status VersionSet::Recover(RecoveryReport& report) {
  ABSL_ASSIGN_OR_RETURN(std::vector<std::string> children,
                        env_.GetChildren(dir_));
  const std::string path = absl::StrCat(dir_, "/MANIFEST");
  ABSL_ASSIGN_OR_RETURN(bool exists, env_.FileExists(path));
  Manifest manifest;
  if (exists) {
    auto data = env_.ReadFile(path);
    if (!data.ok()) return data.status();
    auto decoded = DecodeManifest(*data);
    if (decoded.ok()) {
      manifest = std::move(*decoded);
    } else if (absl::IsFailedPrecondition(decoded.status())) {
      // A newer format must not be destroyed by an older binary.
      return decoded.status();
    } else {
      report.manifest_reset = true;
    }
  } else {
    uint64_t number = 0;
    for (const std::string& name : children) {
      if (ParseNumber(name, number)) report.manifest_reset = true;
    }
  }
  if (report.manifest_reset) manifest = Manifest{};
  std::vector<std::pair<uint64_t, std::shared_ptr<ReapedSegment>>> handles;
  auto version = std::make_shared<Version>();
  version->manifest_seq = manifest.manifest_seq;
  version->views = manifest.views;
  for (const ManifestSegment& metadata : manifest.segments) {
    auto opened = OpenSegment(metadata);
    if (!opened.ok()) {
      report.dropped_segments.push_back(
          {metadata.file_number, std::string(opened.status().message())});
      continue;
    }
    handles.emplace_back(metadata.file_number, *opened);
    version->segments.push_back(MapSegment(metadata, (*opened)->segment()));
  }
  if (report.manifest_reset || !report.dropped_segments.empty()) {
    ++manifest.manifest_seq;
    manifest.segments.clear();
    for (const VersionSegment& segment : version->segments) {
      manifest.segments.push_back(segment.metadata);
    }
    ABSL_RETURN_IF_ERROR(WriteManifest(manifest));
    version->manifest_seq = manifest.manifest_seq;
    // Writing a corrected MANIFEST may have consumed an old MANIFEST.tmp.
    ABSL_ASSIGN_OR_RETURN(children, env_.GetChildren(dir_));
  }
  // Clean up only after the corrected MANIFEST is installed. Unknown names
  // belong to other users of the directory and are left alone.
  absl::flat_hash_set<uint64_t> listed;
  for (const VersionSegment& segment : version->segments) {
    listed.insert(segment.metadata.file_number);
  }
  for (const std::string& name : children) {
    uint64_t number = 0;
    if (name.ends_with(".tmp") ||
        (ParseNumber(name, number) && !listed.contains(number))) {
      // The reaper of a snapshot from an earlier VersionSet on this directory
      // may still be deleting the same obsolete file.
      const absl::Status deleted =
          env_.DeleteFile(absl::StrCat(dir_, "/", name));
      if (absl::IsNotFound(deleted)) continue;
      ABSL_RETURN_IF_ERROR(deleted);
      report.deleted_files.push_back(name);
    }
  }
  next_file_number_ = manifest.next_file_number;
  live_ = std::move(handles);
  {
    absl::MutexLock lock(current_mu_);
    current_ = std::move(version);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> VersionSet::NewSegmentFile() {
  if (reopen_required_) return absl::FailedPreconditionError("reopen required");
  if (next_file_number_ == std::numeric_limits<uint64_t>::max()) {
    return absl::ResourceExhaustedError("segment file numbers exhausted");
  }
  const uint64_t number = next_file_number_++;
  reserved_.insert(number);
  return SegmentPath(number, true);
}

void VersionSet::DiscardSegmentFile(absl::string_view path) {
  const std::string prefix = absl::StrCat(dir_, "/");
  if (!path.starts_with(prefix) || !path.ends_with(".tmp")) return;
  uint64_t number = 0;
  if (ParseNumber(path.substr(prefix.size(), path.size() - prefix.size() - 4),
                  number)) {
    reserved_.erase(number);
  }
}

bool VersionSet::reopen_required() const { return reopen_required_; }

std::shared_ptr<const Version> VersionSet::current() const {
  absl::MutexLock lock(current_mu_);
  return current_;
}

absl::Status VersionSet::Install(const VersionEdit& edit) {
  if (reopen_required_) return absl::FailedPreconditionError("reopen required");
  std::shared_ptr<const Version> old = current();
  if (old->manifest_seq == std::numeric_limits<uint64_t>::max()) {
    return absl::ResourceExhaustedError("MANIFEST sequence exhausted");
  }
  absl::flat_hash_set<uint64_t> removed;
  for (uint64_t number : edit.remove) {
    if (!removed.insert(number).second ||
        std::none_of(old->segments.begin(), old->segments.end(),
                     [number](const VersionSegment& segment) {
                       return segment.metadata.file_number == number;
                     })) {
      return absl::InvalidArgumentError(
          "remove is not a unique live file number");
    }
  }
  std::vector<std::pair<uint64_t, std::string>> added;
  absl::flat_hash_set<uint64_t> added_numbers;
  for (const std::string& path : edit.add) {
    uint64_t number = 0;
    const std::string prefix = absl::StrCat(dir_, "/");
    if (!absl::string_view(path).starts_with(prefix) ||
        !absl::string_view(path).ends_with(".tmp") ||
        !ParseNumber(absl::string_view(path).substr(
                         prefix.size(), path.size() - prefix.size() - 4),
                     number) ||
        !reserved_.contains(number) || !added_numbers.insert(number).second) {
      return absl::InvalidArgumentError("invalid or unallocated temp segment");
    }
    if (std::any_of(old->segments.begin(), old->segments.end(),
                    [number](const VersionSegment& segment) {
                      return segment.metadata.file_number == number;
                    })) {
      return absl::InvalidArgumentError("cannot replace a live segment");
    }
    added.emplace_back(number, path);
  }
  auto next = std::make_shared<Version>();
  next->manifest_seq = old->manifest_seq + 1;
  next->views = edit.views.value_or(old->views);
  // Replacements inherit the first removed input's position. Equal-key
  // duplicates must not change winners merely because a segment was merged.
  size_t insertion = 0;
  bool found_removed = false;
  for (const VersionSegment& segment : old->segments) {
    if (removed.contains(segment.metadata.file_number)) {
      found_removed = true;
    } else {
      if (!found_removed) ++insertion;
      next->segments.push_back(segment);
    }
  }
  std::vector<std::pair<uint64_t, std::shared_ptr<ReapedSegment>>> new_handles;
  bool renamed_manifest = false;
  absl::Status status = absl::OkStatus();
  for (const auto& [number, temp] : added) {
    auto installed = InstallSegment(number, temp);
    if (!installed.ok()) {
      status = installed.status();
      break;
    }
    auto& [metadata, handle] = *installed;
    new_handles.emplace_back(number, handle);
    next->segments.insert(
        next->segments.begin() + static_cast<std::ptrdiff_t>(insertion++),
        MapSegment(metadata, handle->segment()));
  }
  if (status.ok() && !added.empty()) status = env_.SyncDir(dir_);
  if (status.ok()) {
    Manifest manifest;
    manifest.manifest_seq = next->manifest_seq;
    manifest.next_file_number = next_file_number_;
    manifest.views = next->views;
    for (const VersionSegment& segment : next->segments) {
      manifest.segments.push_back(segment.metadata);
    }
    status = WriteManifest(manifest, &renamed_manifest);
  }
  if (!status.ok()) {
    if (renamed_manifest) {
      reopen_required_ = true;
    } else {
      for (const auto& [number, temp] : added) {
        static_cast<void>(env_.DeleteFile(temp));
        static_cast<void>(env_.DeleteFile(SegmentPath(number, false)));
      }
      static_cast<void>(env_.DeleteFile(absl::StrCat(dir_, "/MANIFEST.tmp")));
    }
    for (const auto& entry : added) reserved_.erase(entry.first);
    return status;
  }
  for (const auto& entry : added) reserved_.erase(entry.first);
  // Order: SegmentWriter writes footer and syncs temp; rename each segment;
  // sync the directory if segments were added; write, sync and rename
  // MANIFEST; sync the directory; publish; mark obsolete. The
  // reaper unmaps and unlinks only after the last reference is released.
  // Holding `old` across the marks prevents a last-release/mark race.
  {
    absl::MutexLock lock(current_mu_);
    current_ = next;
  }
  for (auto& [number, handle] : live_) {
    if (removed.contains(number)) handle->MarkObsolete();
  }
  std::erase_if(live_, [&removed](const auto& entry) {
    return removed.contains(entry.first);
  });
  live_.insert(live_.end(), new_handles.begin(), new_handles.end());
  return absl::OkStatus();
}

}  // namespace ferret

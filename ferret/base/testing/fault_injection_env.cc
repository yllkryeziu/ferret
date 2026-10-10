// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/base/testing/fault_injection_env.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

#include "ferret/base/env.h"
#include "ferret/base/mapped_file.h"

namespace ferret_testing {
namespace {

// `dir` without trailing slashes, except for "/" itself.
std::string NormalizeDir(absl::string_view dir) {
  while (dir.size() > 1 && dir.back() == '/') dir.remove_suffix(1);
  return std::string(dir);
}

// The directory whose SyncDir() makes an entry for `path` durable.
std::string ParentDir(absl::string_view path) {
  const size_t slash = path.rfind('/');
  if (slash == absl::string_view::npos) return ".";
  return NormalizeDir(path.substr(0, slash == 0 ? 1 : slash));
}

}  // namespace

class FaultInjectionEnv::TrackedFile final : public ferret::WritableFile {
 public:
  TrackedFile(FaultInjectionEnv* absl_nonnull env,
              std::unique_ptr<ferret::WritableFile> base, FileId id,
              int64_t epoch)
      : env_(env), base_(std::move(base)), id_(id), epoch_(epoch) {}
  TrackedFile(const TrackedFile&) = delete;
  TrackedFile& operator=(const TrackedFile&) = delete;

  absl::Status Append(absl::string_view data) override {
    absl::MutexLock lock(env_->mu_);
    ABSL_RETURN_IF_ERROR(env_->BeginFileMutationLocked(epoch_));
    return base_->Append(data);
  }

  absl::Status WriteAt(int64_t offset, absl::string_view data) override {
    absl::MutexLock lock(env_->mu_);
    ABSL_RETURN_IF_ERROR(env_->BeginFileMutationLocked(epoch_));
    FileState& state = env_->files_.at(id_);
    // Keep the synced bytes this write is about to overwrite. A deleted file
    // needs nothing, and a replaced one already holds its durable content.
    if (offset < state.synced_size && !state.durable_content.has_value() &&
        !state.path.empty()) {
      ABSL_ASSIGN_OR_RETURN(state.durable_content,
                            env_->DurableContentLocked(state.path));
    }
    return base_->WriteAt(offset, data);
  }

  int64_t size() const override { return base_->size(); }

  absl::Status Sync() override {
    absl::MutexLock lock(env_->mu_);
    ABSL_RETURN_IF_ERROR(env_->BeginFileMutationLocked(epoch_));
    ABSL_RETURN_IF_ERROR(base_->Sync());
    MarkSyncedLocked();
    return absl::OkStatus();
  }

  absl::Status FullSync() override {
    absl::MutexLock lock(env_->mu_);
    ABSL_RETURN_IF_ERROR(env_->BeginFileMutationLocked(epoch_));
    ABSL_RETURN_IF_ERROR(base_->FullSync());
    MarkSyncedLocked();
    return absl::OkStatus();
  }

  absl::Status Close() override {
    absl::MutexLock lock(env_->mu_);
    ABSL_RETURN_IF_ERROR(env_->BeginFileMutationLocked(epoch_));
    return base_->Close();
  }

 private:
  void MarkSyncedLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(env_->mu_) {
    FileState& state = env_->files_.at(id_);
    // A file that a rename replaced has no path to read its content back
    // from, so it keeps the durable content captured when it was replaced.
    if (state.path.empty()) return;
    state.synced_size = base_->size();
    state.durable_content.reset();
  }

  FaultInjectionEnv* absl_nonnull const env_;
  const std::unique_ptr<ferret::WritableFile> base_;
  const FileId id_;
  const int64_t epoch_;
};

FaultInjectionEnv::FaultInjectionEnv(ferret::Env* absl_nonnull base)
    : base_(base) {}

FaultInjectionEnv::~FaultInjectionEnv() = default;

absl::StatusOr<std::unique_ptr<ferret::WritableFile>>
FaultInjectionEnv::NewWritableFile(absl::string_view path) {
  absl::MutexLock lock(mu_);
  ABSL_RETURN_IF_ERROR(BeginMutationLocked());
  ABSL_ASSIGN_OR_RETURN(const bool existed, base_->FileExists(path));
  // Truncating an existing file changes its data but not its directory entry.
  std::optional<std::string> durable_content;
  if (existed) {
    ABSL_ASSIGN_OR_RETURN(durable_content, DurableContentLocked(path));
  }
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<ferret::WritableFile> file,
                        base_->NewWritableFile(path));
  const FileId id = TrackLocked(path);
  FileState& state = files_.at(id);
  state.synced_size = 0;
  state.durable_content = std::move(durable_content);
  if (!existed) {
    pending_ops_.push_back({.kind = PendingEntryOp::Kind::kCreate,
                            .file = id,
                            .unsynced_dirs = {ParentDir(path)}});
  }
  return std::make_unique<TrackedFile>(this, std::move(file), id, epoch_);
}

absl::StatusOr<std::string> FaultInjectionEnv::ReadFile(
    absl::string_view path) {
  return base_->ReadFile(path);
}

absl::StatusOr<std::unique_ptr<ferret::MappedFile>>
FaultInjectionEnv::NewMappedFile(absl::string_view path) {
  return base_->NewMappedFile(path);
}

absl::StatusOr<int64_t> FaultInjectionEnv::GetFileSize(absl::string_view path) {
  return base_->GetFileSize(path);
}

absl::StatusOr<bool> FaultInjectionEnv::FileExists(absl::string_view path) {
  return base_->FileExists(path);
}

absl::StatusOr<std::vector<std::string>> FaultInjectionEnv::GetChildren(
    absl::string_view path) {
  return base_->GetChildren(path);
}

absl::Status FaultInjectionEnv::CreateDir(absl::string_view path) {
  absl::MutexLock lock(mu_);
  ABSL_RETURN_IF_ERROR(BeginMutationLocked());
  return base_->CreateDir(path);
}

absl::Status FaultInjectionEnv::DeleteFile(absl::string_view path) {
  absl::MutexLock lock(mu_);
  ABSL_RETURN_IF_ERROR(BeginMutationLocked());
  ABSL_RETURN_IF_ERROR(base_->DeleteFile(path));
  if (const auto it = file_at_path_.find(path); it != file_at_path_.end()) {
    FileState& state = files_.at(it->second);
    state.deleted = true;
    state.path.clear();
    file_at_path_.erase(it);
  }
  return absl::OkStatus();
}

absl::Status FaultInjectionEnv::RenameFile(absl::string_view from,
                                           absl::string_view to) {
  absl::MutexLock lock(mu_);
  ABSL_RETURN_IF_ERROR(BeginMutationLocked());
  // Renaming a file onto itself changes nothing.
  if (from == to) return base_->RenameFile(from, to);
  ABSL_ASSIGN_OR_RETURN(const bool replacing, base_->FileExists(to));
  std::optional<std::string> replaced_content;
  if (replacing) {
    ABSL_ASSIGN_OR_RETURN(replaced_content, DurableContentLocked(to));
  }
  ABSL_RETURN_IF_ERROR(base_->RenameFile(from, to));

  PendingEntryOp op = {.kind = PendingEntryOp::Kind::kRename,
                       .from = std::string(from),
                       .unsynced_dirs = {ParentDir(from)}};
  if (std::string to_dir = ParentDir(to); to_dir != op.unsynced_dirs[0]) {
    op.unsynced_dirs.push_back(std::move(to_dir));
  }
  if (replacing) {
    op.replaced = TrackLocked(to);
    FileState& replaced = files_.at(*op.replaced);
    replaced.path.clear();
    replaced.durable_content = std::move(replaced_content);
    file_at_path_.erase(to);
  }
  op.file = TrackLocked(from);
  file_at_path_.erase(from);
  files_.at(op.file).path = std::string(to);
  file_at_path_.emplace(to, op.file);
  pending_ops_.push_back(std::move(op));
  return absl::OkStatus();
}

absl::Status FaultInjectionEnv::SyncDir(absl::string_view path) {
  absl::MutexLock lock(mu_);
  ABSL_RETURN_IF_ERROR(BeginMutationLocked());
  ABSL_RETURN_IF_ERROR(base_->SyncDir(path));
  const std::string dir = NormalizeDir(path);
  for (PendingEntryOp& op : pending_ops_) std::erase(op.unsynced_dirs, dir);
  std::erase_if(pending_ops_, [](const PendingEntryOp& op) {
    return op.unsynced_dirs.empty();
  });
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ferret::FileLock>> FaultInjectionEnv::LockFile(
    absl::string_view path) {
  return base_->LockFile(path);
}

void FaultInjectionEnv::FailAfter(int64_t n, absl::Status status) {
  DCHECK_GE(n, 0);
  DCHECK(!status.ok());
  absl::MutexLock lock(mu_);
  fail_after_ = mutation_count_ + n;
  fault_status_ = std::move(status);
}

void FaultInjectionEnv::ClearFault() {
  absl::MutexLock lock(mu_);
  fail_after_.reset();
  fault_status_ = absl::OkStatus();
}

int64_t FaultInjectionEnv::mutation_count() const {
  absl::MutexLock lock(mu_);
  return mutation_count_;
}

absl::Status FaultInjectionEnv::DropUnsyncedData() {
  absl::MutexLock lock(mu_);
  for (auto op = pending_ops_.rbegin(); op != pending_ops_.rend(); ++op) {
    ABSL_RETURN_IF_ERROR(UndoLocked(*op));
  }
  for (const auto& [id, state] : files_) {
    if (state.path.empty()) continue;
    if (state.durable_content.has_value()) {
      ABSL_RETURN_IF_ERROR(OverwriteLocked(state.path, *state.durable_content));
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(const int64_t size, base_->GetFileSize(state.path));
    if (size > state.synced_size) {
      ABSL_ASSIGN_OR_RETURN(std::string content, base_->ReadFile(state.path));
      content.resize(static_cast<size_t>(state.synced_size));
      ABSL_RETURN_IF_ERROR(OverwriteLocked(state.path, content));
    }
  }
  files_.clear();
  file_at_path_.clear();
  pending_ops_.clear();
  fail_after_.reset();
  fault_status_ = absl::OkStatus();
  ++epoch_;
  return absl::OkStatus();
}

absl::Status FaultInjectionEnv::BeginMutationLocked() {
  ++mutation_count_;
  if (fail_after_.has_value() && mutation_count_ > *fail_after_) {
    return fault_status_;
  }
  return absl::OkStatus();
}

absl::Status FaultInjectionEnv::BeginFileMutationLocked(int64_t epoch) {
  if (epoch != epoch_) {
    return absl::FailedPreconditionError(
        "file was opened before a simulated crash");
  }
  return BeginMutationLocked();
}

FaultInjectionEnv::FileId FaultInjectionEnv::TrackLocked(
    absl::string_view path) {
  if (const auto it = file_at_path_.find(path); it != file_at_path_.end()) {
    return it->second;
  }
  const FileId id = next_file_id_++;
  files_.emplace(
      id, FileState{.path = std::string(path), .synced_size = kFullySynced});
  file_at_path_.emplace(path, id);
  return id;
}

absl::StatusOr<std::string> FaultInjectionEnv::DurableContentLocked(
    absl::string_view path) {
  const auto it = file_at_path_.find(path);
  if (it != file_at_path_.end()) {
    const FileState& state = files_.at(it->second);
    if (state.durable_content.has_value()) return *state.durable_content;
  }
  ABSL_ASSIGN_OR_RETURN(std::string content, base_->ReadFile(path));
  if (it != file_at_path_.end()) {
    const int64_t synced_size = files_.at(it->second).synced_size;
    if (static_cast<int64_t>(content.size()) > synced_size) {
      content.resize(static_cast<size_t>(synced_size));
    }
  }
  return content;
}

absl::Status FaultInjectionEnv::UndoLocked(const PendingEntryOp& op) {
  FileState& state = files_.at(op.file);
  // Deletes are never undone. A file without a path was replaced by a later
  // rename that is already durable (newer pending renames were undone
  // first), so it is gone for good.
  if (state.deleted || state.path.empty()) return absl::OkStatus();
  switch (op.kind) {
    case PendingEntryOp::Kind::kCreate:
      ABSL_RETURN_IF_ERROR(base_->DeleteFile(state.path));
      file_at_path_.erase(state.path);
      state.deleted = true;
      state.path.clear();
      return absl::OkStatus();
    case PendingEntryOp::Kind::kRename: {
      // A durable entry has reused the old name: the source directory was
      // synced after this rename, only the target directory was not. Moving
      // the file back would clobber that entry, so the rename is kept.
      if (file_at_path_.contains(op.from)) return absl::OkStatus();
      const std::string to = state.path;
      ABSL_RETURN_IF_ERROR(base_->RenameFile(to, op.from));
      file_at_path_.erase(to);
      state.path = op.from;
      file_at_path_.emplace(op.from, op.file);
      if (op.replaced.has_value()) {
        FileState& replaced = files_.at(*op.replaced);
        // RenameFile() captured it when the rename replaced the file.
        CHECK(replaced.durable_content.has_value());
        ABSL_RETURN_IF_ERROR(OverwriteLocked(to, *replaced.durable_content));
        replaced.path = to;
        replaced.synced_size = kFullySynced;
        replaced.durable_content.reset();
        file_at_path_.emplace(to, *op.replaced);
      }
      return absl::OkStatus();
    }
  }
  LOG(FATAL) << "invalid PendingEntryOp::Kind " << static_cast<int>(op.kind);
}

absl::Status FaultInjectionEnv::OverwriteLocked(absl::string_view path,
                                                absl::string_view content) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<ferret::WritableFile> file,
                        base_->NewWritableFile(path));
  ABSL_RETURN_IF_ERROR(file->Append(content));
  return file->Close();
}

}  // namespace ferret_testing

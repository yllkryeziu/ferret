// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// FaultInjectionEnv: an Env wrapper that simulates power loss, for crash
// tests, in the spirit of LevelDB's FaultInjectionTestEnv and RocksDB's
// FaultInjectionTestFS.
//
// A crash test runs a protocol (flush, compaction, MANIFEST install) over
// this env, stops it at some mutation with FailAfter(), destroys the objects
// of the "crashed" process, calls DropUnsyncedData(), and checks that
// recovery finds a consistent index. Running the protocol once without a
// fault gives its mutation count K; FailAfter(k) for every k in [0, K] then
// visits every crash point.
//
// Crash model. DropUnsyncedData() leaves the files as a power loss would on
// a file system that guarantees nothing beyond explicit syncs:
//
//  * File data. A file's durable content is its content at its last
//    successful Sync() or FullSync() (the two are not distinguished); every
//    later write is lost, including overwrites of synced bytes. Files this
//    env did not write count as fully synced.
//  * Directory entries. Creating a file and renaming one become durable only
//    once SyncDir() has succeeded on the parent directory, or on both
//    parents for a rename across directories. The creates and renames not
//    yet durable are undone newest first: a created file disappears, and a
//    renamed file returns to its old name, bringing back the durable content
//    of any file the rename replaced. The one exception is a rename across
//    directories whose old name a durable entry has since reused (only the
//    source directory was synced): it is kept, as a file system that commits
//    a rename in one transaction would. Paths are compared as strings, so
//    spell each directory the same way in paths and in SyncDir().
//  * Deletes are durable at once and never undone, which is the worst case
//    for a commit protocol: an unlink issued before the MANIFEST that stops
//    referencing the file is durable leaves a dangling reference. An undo
//    step for a file that was deleted later is skipped.
//  * CreateDir() is durable at once.
//
// Mutations are NewWritableFile, CreateDir, DeleteFile, RenameFile and
// SyncDir, plus Append, WriteAt, Sync, FullSync and Close on files from this
// env. Reads (ReadFile, NewMappedFile, GetFileSize, FileExists, GetChildren)
// and LockFile, whose file carries no data, pass straight through and never
// fail by injection.

#ifndef FERRET_BASE_TESTING_FAULT_INJECTION_ENV_H_
#define FERRET_BASE_TESTING_FAULT_INJECTION_ENV_H_

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

#include "ferret/base/env.h"
#include "ferret/base/mapped_file.h"

namespace ferret_testing {

// Mutations are serialized: each one runs under a mutex together with its
// bookkeeping, so the recorded order is the order the base Env saw.
// Thread-safe.
class FaultInjectionEnv final : public ferret::Env {
 public:
  // Wraps `base`, which must outlive this env. Files this env returns must
  // not outlive it.
  explicit FaultInjectionEnv(ferret::Env* absl_nonnull base);
  FaultInjectionEnv(const FaultInjectionEnv&) = delete;
  FaultInjectionEnv& operator=(const FaultInjectionEnv&) = delete;
  ~FaultInjectionEnv() override;

  absl::StatusOr<std::unique_ptr<ferret::WritableFile>> NewWritableFile(
      absl::string_view path) override ABSL_LOCKS_EXCLUDED(mu_);
  absl::StatusOr<std::string> ReadFile(absl::string_view path) override;
  absl::StatusOr<std::unique_ptr<ferret::MappedFile>> NewMappedFile(
      absl::string_view path) override;
  absl::StatusOr<int64_t> GetFileSize(absl::string_view path) override;
  absl::StatusOr<bool> FileExists(absl::string_view path) override;
  absl::StatusOr<std::vector<std::string>> GetChildren(
      absl::string_view path) override;
  absl::Status CreateDir(absl::string_view path) override
      ABSL_LOCKS_EXCLUDED(mu_);
  absl::Status DeleteFile(absl::string_view path) override
      ABSL_LOCKS_EXCLUDED(mu_);
  absl::Status RenameFile(absl::string_view from, absl::string_view to) override
      ABSL_LOCKS_EXCLUDED(mu_);
  absl::Status SyncDir(absl::string_view path) override
      ABSL_LOCKS_EXCLUDED(mu_);
  absl::StatusOr<std::unique_ptr<ferret::FileLock>> LockFile(
      absl::string_view path) override;

  // Lets the next `n` (>= 0) mutations run, then fails every later one with
  // `status` (not OK) without performing it, as if the machine had lost
  // power, until ClearFault() or DropUnsyncedData(). Replaces any earlier
  // fault.
  void FailAfter(int64_t n, absl::Status status) ABSL_LOCKS_EXCLUDED(mu_);

  // Stops failing mutations, for tests of transient errors such as a full
  // disk that frees up.
  void ClearFault() ABSL_LOCKS_EXCLUDED(mu_);

  // The number of mutations attempted since construction, failed ones
  // included.
  int64_t mutation_count() const ABSL_LOCKS_EXCLUDED(mu_);

  // Simulates the power loss described in the file comment, then clears the
  // fault and treats every remaining file as synced, as after a reboot.
  // Files returned earlier fail every later mutation with FailedPrecondition.
  // Mappings of files that lose data must already be destroyed: touching
  // their truncated pages raises SIGBUS.
  absl::Status DropUnsyncedData() ABSL_LOCKS_EXCLUDED(mu_);

 private:
  class TrackedFile;

  // Identifies a file across renames. Stable until DropUnsyncedData().
  using FileId = int64_t;

  // The synced size of a file this env did not write.
  static constexpr int64_t kFullySynced = std::numeric_limits<int64_t>::max();

  struct FileState {
    // Where the file is now; empty once deleted, or after a rename replaced
    // it.
    std::string path;
    // The durable content is the first `synced_size` bytes of the current
    // content, unless `durable_content` holds it verbatim (after an
    // overwrite of synced bytes, a truncation, or a replacement).
    int64_t synced_size = 0;
    std::optional<std::string> durable_content;
    bool deleted = false;
  };

  // A create or rename whose directory entry is not durable yet.
  struct PendingEntryOp {
    enum class Kind : uint8_t { kCreate, kRename };
    Kind kind;
    FileId file;
    // kRename: the file's previous path.
    std::string from;
    // kRename: the file that `to`, the new path, named before the rename.
    std::optional<FileId> replaced;
    // Parents whose SyncDir() is still due.
    std::vector<std::string> unsynced_dirs;
  };

  // Counts a mutation and returns the injected failure, if any.
  absl::Status BeginMutationLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // BeginMutationLocked() for a file created in `epoch`.
  absl::Status BeginFileMutationLocked(int64_t epoch)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Returns the file at `path`, which must exist, and starts tracking it as
  // fully synced if this env has not seen it before.
  FileId TrackLocked(absl::string_view path) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Returns what `path` would contain after a crash right now.
  absl::StatusOr<std::string> DurableContentLocked(absl::string_view path)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Undoes `op`, which is the newest pending entry operation.
  absl::Status UndoLocked(const PendingEntryOp& op)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Replaces the content of `path` through the base env.
  absl::Status OverwriteLocked(absl::string_view path,
                               absl::string_view content)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  ferret::Env* absl_nonnull const base_;
  mutable absl::Mutex mu_;
  int64_t mutation_count_ ABSL_GUARDED_BY(mu_) = 0;
  // Mutations numbered above this fail with fault_status_.
  std::optional<int64_t> fail_after_ ABSL_GUARDED_BY(mu_);
  absl::Status fault_status_ ABSL_GUARDED_BY(mu_);
  // Incremented by DropUnsyncedData() to disown files opened before it.
  int64_t epoch_ ABSL_GUARDED_BY(mu_) = 0;
  FileId next_file_id_ ABSL_GUARDED_BY(mu_) = 0;
  absl::flat_hash_map<FileId, FileState> files_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, FileId> file_at_path_ ABSL_GUARDED_BY(mu_);
  // Oldest first.
  std::vector<PendingEntryOp> pending_ops_ ABSL_GUARDED_BY(mu_);
};

}  // namespace ferret_testing

#endif  // FERRET_BASE_TESTING_FAULT_INJECTION_ENV_H_

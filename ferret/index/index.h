// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// A live index with serialized memtable writes and concurrent snapshot readers.

#ifndef FERRET_INDEX_INDEX_H_
#define FERRET_INDEX_INDEX_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

#include "ferret/base/env.h"
#include "ferret/base/thread_pool.h"
#include "ferret/index/blob_registry.h"
#include "ferret/index/compaction_policy.h"
#include "ferret/index/memtable.h"
#include "ferret/index/read_state.h"
#include "ferret/index/reaper.h"
#include "ferret/index/segment_writer.h"
#include "ferret/index/version_set.h"
#include "ferret/symbols/symbol_extractor.h"

namespace ferret {

// The ingest priority for a memtable append.
enum class IngestLane { kInteractive, kBulk };

struct IndexOptions {
  // Soft flush thresholds and chunking options for the active memtable.
  MemtableOptions memtable;
  // Extractor workers at utility QoS. Must be at least one.
  int symbol_threads = 2;
  // Test-only gate for an extractor. Release it before destroying the Index.
  std::shared_ptr<absl::Notification> symbol_extraction_gate;
  std::shared_ptr<absl::Notification> symbol_extraction_reached;
  // Test-only count of actual memtable parser invocations, across reopen.
  std::shared_ptr<std::atomic<uint64_t>> symbol_extractions;
  // Background compaction thresholds and the test-only disable switch.
  CompactionOptions compaction;
  // Flush an idle memtable once its oldest document reaches this age.
  absl::Duration max_memtable_age = absl::Minutes(5);
  // Bound retained uninstalled memory. One extra slot is reserved for
  // interactive ingest. At most four bulk frozen memtables keep
  // verification of every memtable candidate bounded. Valid range: 1 to 4.
  size_t max_frozen_memtables = 4;
  // A zero count must persist for this long before a document is dropped.
  // GC scans run at most once per tenth of this period, so a dead document
  // becomes drop-eligible at most 10% later than its grace deadline.
  absl::Duration gc_grace_period = absl::Minutes(10);
  // Test-only gate after compaction output is written. Signal before
  // destruction; safe to notify from any thread.
  std::shared_ptr<absl::Notification> compaction_install_gate;
  // Test-only notification after output is written, before the install gate.
  // Safe to wait from any thread; requires the corresponding gate to be set.
  std::shared_ptr<absl::Notification> compaction_install_reached;
  // Test-only gate after flush output is written. Signal before destruction;
  // safe to notify from any thread.
  std::shared_ptr<absl::Notification> flush_install_gate;
  // Test-only notification after output is written, before the install gate.
  // Safe to wait from any thread; requires the corresponding gate to be set.
  std::shared_ptr<absl::Notification> flush_install_reached;
};

struct IndexStats {
  uint64_t pending_symbols = 0;
  uint64_t failed_symbols = 0;
  // Last GC scan's zero-count document count (including grace), and the
  // current published document total minus that count. While GC is suspended,
  // all documents are live and none are dead. Stats() is thread-safe; counts
  // may lag view edits and installs until the next scan.
  uint64_t live_docs = 0;
  uint64_t dead_docs = 0;
  // Longest completed GC scan in this process; zero until the first scan.
  absl::Duration gc_scan_time = absl::ZeroDuration();
  // Number of GC scans completed since Open(); excludes suspended periods.
  uint64_t gc_scans = 0;
  // Cumulative input file bytes no longer occupied after GC compactions.
  uint64_t reclaimed_bytes = 0;
  // Number of successfully installed compaction outputs.
  uint64_t completed_compactions = 0;
  // Number of failed compaction attempts, including retries.
  uint64_t compaction_failures = 0;
  // Cumulative file bytes written by successful flushes, including fixed
  // per-file overhead.
  uint64_t flush_bytes = 0;
  // Cumulative file bytes written by successfully installed compactions,
  // including fixed per-file overhead.
  uint64_t compaction_bytes = 0;
  // Number of times each lane had to wait for frozen space or the writer.
  uint64_t bulk_ingest_waits = 0;
  uint64_t interactive_ingest_waits = 0;
  // Bytes and count of successfully installed direct bulk segments.
  uint64_t bulk_segment_bytes = 0;
  uint64_t bulk_segments = 0;
  // Number of segments in the currently published version.
  size_t segment_count = 0;
  // Last compaction failure; OK until the first failure and after success.
  absl::Status last_compaction_error;
  // Last scheduled retry interval, for diagnostics; zero until a failure.
  absl::Duration last_compaction_retry_delay = absl::ZeroDuration();
  // True if VersionSet requires reopening; no more compactions are scheduled.
  bool compaction_disabled = false;
};

// All public operations are thread-safe. The Env
// must outlive this index and every returned ReadState. Background flush errors
// are sticky: Add() returns the error on every call until Flush() retries
// successfully. Callers must finish all operations except a cooperative
// AddSegment callback before destroying the index.
class Index final {
 public:
  // Opens the persistent segments and reports any recovery losses.
  static absl::StatusOr<std::pair<std::unique_ptr<Index>, RecoveryReport>> Open(
      Env& env, absl::string_view dir, IndexOptions options = {});
  Index(const Index&) = delete;
  Index& operator=(const Index&) = delete;
  ~Index();

  // Adds through the bulk lane, waiting at the frozen memtable cap.
  absl::Status Add(const SegmentDocument& document)
      ABSL_LOCKS_EXCLUDED(state_mu_);
  // Serializes memtable writes. Bulk waits at max_frozen_memtables;
  // interactive may use one reserved extra frozen slot before waiting.
  // Either lane returns a sticky background flush error until Flush retries.
  absl::Status Add(const SegmentDocument& document, IngestLane lane)
      ABSL_LOCKS_EXCLUDED(state_mu_);
  // Streams rank-ordered documents into a new L0 segment without index locks
  // held during the callback. Publishes the entire nonempty segment at once.
  // Duplicate Oids across sources are allowed; queries deduplicate them.
  // A callback may return Cancelled, or poll cancelled() during shutdown.
  // The callback must finish before destruction can complete. Errors discard
  // the partial file; callback errors do not poison memtable ingestion.
  // `finish_pool`, if not null, runs the parallel parts of the writer's
  // Finish() on `finish_threads` threads including the caller (see
  // SegmentWriterOptions); it does not change the bytes written.
  absl::StatusOr<SegmentStats> AddSegment(
      absl::FunctionRef<absl::Status(SegmentWriter&)> write,
      ThreadPool* finish_pool = nullptr, int finish_threads = 1)
      ABSL_LOCKS_EXCLUDED(install_mu_, state_mu_);
  // Becomes true when destruction begins; safe to poll from a callback.
  bool cancelled() const { return cancelled_.load(std::memory_order_relaxed); }
  // Freezes any nonempty active memtable and waits for all installs. A
  // background failure is returned on the first Flush() observing it; the
  // next Flush() retries the retained frozen memtable. Add() keeps returning
  // that failure until the retry succeeds.
  absl::Status Flush() ABSL_LOCKS_EXCLUDED(state_mu_);
  // Captures the active watermark under the state lock. This costs one state
  // allocation per query rather than one publication per Add(). The lock is
  // released before allocating.
  std::shared_ptr<const ReadState> GetReadState() const
      ABSL_LOCKS_EXCLUDED(state_mu_);

  // A single view reference edit. The blob may not yet be indexed.
  struct ReferenceDelta {
    // Process-local blob ID from blobs(); valid while this Index lives.
    BlobId blob;
    // Signed change in the number of views referencing the blob.
    int64_t delta;
  };
  // Applies a batch atomically under the state lock; safe from any thread.
  // Underflow or overflow returns InvalidArgument without changing any counts.
  absl::Status AdjustReferences(absl::Span<const ReferenceDelta> deltas)
      ABSL_LOCKS_EXCLUDED(state_mu_);
  // Runs publication after positive deltas and before negative deltas, under
  // the state lock. The callback must not call back into the index.
  absl::Status AdjustReferences(absl::Span<const ReferenceDelta> deltas,
                                absl::FunctionRef<void()> publish)
      ABSL_LOCKS_EXCLUDED(state_mu_);
  // Returns persisted registrations, including views needing a resync.
  std::vector<ManifestView> Views() const;
  // Durably adds or removes a registration, serialized with segment installs.
  absl::Status RecordView(const ManifestView& view);
  absl::Status ForgetView(absl::string_view canonical_root);

  // Enables collection after restored views have resynced. Repeated
  // calls leave existing grace deadlines unchanged.
  void ResumeGc() ABSL_LOCKS_EXCLUDED(state_mu_);
  // Stops collection before a view resync. Thread-safe; does not erase counts.
  void SuspendGc() ABSL_LOCKS_EXCLUDED(state_mu_);

  // Process-local registry shared by all sources. Valid while *this lives.
  BlobRegistry& blobs() const { return *blobs_; }

  // Returns counters and the live segment count from one published state.
  IndexStats Stats() const ABSL_LOCKS_EXCLUDED(state_mu_);
  // Waits until both lanes have no queued or running job. A failed job is
  // retried with capped exponential backoff. A reopen-required failure
  // abandons queued jobs and unblocks callers.
  void WaitForCompactions() ABSL_LOCKS_EXCLUDED(state_mu_);

 private:
  Index(Env& env, IndexOptions options);
  void FreezeLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  bool WorkerReadyLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  bool FlushDoneLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  bool RetryDoneLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  bool SpaceAvailableLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  bool BulkReadyLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  bool InteractiveReadyLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  bool NoSegmentsLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  void Run();
  void RunCompaction(size_t lane);
  void ScheduleLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  void RegisterBlobLocked(BlobId blob) ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  bool DropLocked(BlobId blob, absl::Time now) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  bool Restorable(const SegmentDoc& doc) const;
  absl::Status RestoreLocked(const SegmentDoc& doc)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  bool IngestIdleLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  void ScanGcLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  void QueueGcLocked(absl::Time now, std::vector<BlobId> blobs)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  void UpdateGcDeadlineLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  bool ZeroCountLocked(BlobId blob) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  uint64_t TotalDocsLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  bool CompactionReadyLocked(size_t lane) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  bool SmallCompactionReadyLocked() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  bool LargeCompactionReadyLocked() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);
  bool CompactionsDoneLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mu_);

  Env& env_;
  IndexOptions options_;
  // Snapshots also retain the reaper after Index destruction.
  std::shared_ptr<Reaper> reaper_;
  std::shared_ptr<BlobRegistry> blobs_ = std::make_shared<BlobRegistry>();
  std::unique_ptr<VersionSet> versions_;
  // Install lock precedes state_mu_. Never take it while holding state_mu_.
  // It serializes VersionSet allocations, installs and state publication; the
  // expensive segment write or merge runs without either lock.
  absl::Mutex install_mu_ ABSL_ACQUIRED_BEFORE(state_mu_);
  mutable absl::Mutex state_mu_;
  std::shared_ptr<const ReadState> current_ ABSL_GUARDED_BY(state_mu_);
  std::shared_ptr<Memtable> active_ ABSL_GUARDED_BY(state_mu_);
  std::deque<std::shared_ptr<Memtable>> pending_ ABSL_GUARDED_BY(state_mu_);
  absl::Status failure_ ABSL_GUARDED_BY(state_mu_);
  bool retry_ ABSL_GUARDED_BY(state_mu_) = false;
  // Advances after each flush attempt, including failed retries.
  uint64_t finished_flushes_ ABSL_GUARDED_BY(state_mu_) = 0;
  uint64_t retry_until_ ABSL_GUARDED_BY(state_mu_) = 0;
  bool failure_reported_ ABSL_GUARDED_BY(state_mu_) = false;
  bool ingest_in_progress_ ABSL_GUARDED_BY(state_mu_) = false;
  size_t active_segments_ ABSL_GUARDED_BY(state_mu_) = 0;
  bool stopping_ ABSL_GUARDED_BY(state_mu_) = false;
  IndexStats stats_ ABSL_GUARDED_BY(state_mu_);
  struct ReferenceState {
    uint64_t count = 0;
    absl::Time dead_since = absl::InfinitePast();
  };
  std::vector<ReferenceState> references_ ABSL_GUARDED_BY(state_mu_);
  bool gc_suspended_ ABSL_GUARDED_BY(state_mu_) = true;
  // A scan caches drop-eligible documents for the compaction policy.
  absl::flat_hash_map<uint64_t, uint32_t> dead_by_segment_
      ABSL_GUARDED_BY(state_mu_);
  uint64_t scanned_dead_docs_ ABSL_GUARDED_BY(state_mu_) = 0;
  struct PendingGc {
    absl::Time deadline;
    std::vector<BlobId> blobs;
  };
  // Batches are appended in time order; stale resurrected entries are cheap
  // to discard at their deadline without scanning documents.
  std::deque<PendingGc> gc_pending_ ABSL_GUARDED_BY(state_mu_);
  // When the deadline worker next drains gc_pending_: the earliest deadline,
  // but no sooner than a tenth of the grace period after the last scan.
  absl::Time gc_deadline_ ABSL_GUARDED_BY(state_mu_) = absl::InfiniteFuture();
  absl::Time last_gc_scan_ ABSL_GUARDED_BY(state_mu_) = absl::InfinitePast();
  std::array<std::optional<CompactionSelection>, 2> jobs_
      ABSL_GUARDED_BY(state_mu_);
  std::array<absl::Duration, 2> retry_delay_ ABSL_GUARDED_BY(state_mu_) = {
      absl::ZeroDuration(), absl::ZeroDuration()};
  std::array<absl::Time, 2> retry_after_ ABSL_GUARDED_BY(state_mu_) = {
      absl::InfinitePast(), absl::InfinitePast()};
  // Relaxed reads are sufficient: a merge only needs to notice shutdown;
  // joining workers establishes visibility of their final writes.
  std::atomic<bool> cancelled_{false};
  struct SymbolWorker {
    absl::Mutex mu;
    SymbolExtractor extractor;
  };
  std::vector<std::unique_ptr<SymbolWorker>> symbol_workers_;
  std::unique_ptr<ThreadPool> symbol_pool_;
  std::atomic<size_t> next_symbol_worker_{0};
  std::thread worker_;
  std::array<std::thread, 2> compaction_workers_;
};

}  // namespace ferret

#endif  // FERRET_INDEX_INDEX_H_

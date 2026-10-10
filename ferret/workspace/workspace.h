// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Several worktrees sharing one live, content-addressed index.
//
// A Workspace owns one Index, one PathTable shared by every worktree, and a
// view (PathId -> BlobId) per registered worktree. Two lanes feed the views:
//  * the bulk lane registers and rescans worktrees on one thread, reading on
//    a pool, and writes new content as rank-ordered segments or through
//    Index::Add(kBulk);
//  * the interactive lane applies Notify batches on one thread through
//    Index::Add(kInteractive), and never waits for the bulk lane.
// A failed index write keeps each affected path's previous entry and holds
// back the worktree's generations; the bulk lane retries Index::Flush with
// backoff and then rereads those paths.
//
// Invariants:
//  * A view edit names a BlobId only after a document with that Oid is in a
//    published ReadState, and Search captures the view before the ReadState,
//    so every visible blob can be found.
//  * Every file read takes a sequence number when it starts, and a result is
//    applied to a path only if its sequence is newer than the path's last
//    applied one: the last read to start wins.
//  * Generations are per worktree and applied in order: once applied()
//    reaches g, every Notify, Rescan, and directory rescan assigned a
//    generation at most g is visible to Search. A generation is not applied
//    while content the worktree read waits for a retried index write.

#ifndef FERRET_WORKSPACE_WORKSPACE_H_
#define FERRET_WORKSPACE_WORKSPACE_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

#include "ferret/base/bitset.h"
#include "ferret/base/env.h"
#include "ferret/base/thread_pool.h"
#include "ferret/git/oid.h"
#include "ferret/index/index.h"
#include "ferret/search/search_types.h"
#include "ferret/search/searcher.h"
#include "ferret/symbols/symbol_extractor.h"
#include "ferret/symbols/symbol_kind.h"
#include "ferret/text/eligibility.h"
#include "ferret/text/trigram_extractor.h"
#include "ferret/watch/watcher.h"
#include "ferret/workspace/path_table.h"

namespace ferret {

// A registered worktree: process-local id, restored with a new id.
enum class WorktreeId : uint32_t {};

// A test-only callback: the repo-relative path being read, and whether the
// bulk lane reads it. Bulk reads call it from several threads at once, so it
// must be thread-safe.
using WorkspaceReadHook = absl::AnyInvocable<void(absl::string_view, bool)>;

// Configures a Workspace; read once by Open. Thread-compatible.
struct WorkspaceOptions {
  // Thresholds, compaction, and segment format of the owned index.
  IndexOptions index;
  // Query parallelism. A null executor selects one owned by the workspace.
  SearcherOptions search;
  // The eligibility policy of both lanes.
  EligibilityOptions eligibility;
  // Threads that stat, read, and hash files for registrations and rescans.
  int reader_threads = 8;
  // The scheduling class of those threads, which also precompute and extract
  // symbols for direct segments. User-initiated, below the query threads, runs
  // a first registration of Linux in 34 s against 84 s at utility, where the
  // threads share the efficiency cores.
  ThreadQos reader_qos = ThreadQos::kUserInitiated;
  // New content per direct segment. A registration or rescan with more new
  // content writes several segments of contiguous rank ranges, so the
  // segment writer's posting memory stays bounded.
  int64_t bulk_segment_bytes = int64_t{256} << 20;
  // File bytes the bulk write pass rereads ahead of the index writer.
  int64_t bulk_prefetch_bytes = int64_t{64} << 20;
  // Test-only reference: write direct segments with the earlier serial path
  // (SegmentWriter::Add on the bulk lane, symbols on a 2-thread pool the
  // readers wait for) instead of precomputing on the readers.
  bool serial_bulk_write = false;
  // After a failed index write, the delay before the first Index::Flush
  // retry. It doubles after each failure up to the maximum, and resets once
  // the maximum passes without one. Positive.
  absl::Duration ingest_retry_initial_delay = absl::Milliseconds(100);
  absl::Duration ingest_retry_max_delay = absl::Seconds(10);
  // Creates a watcher for a canonical root. Defaults to the native watcher;
  // tests can supply a fake. Called only when Register requests watching.
  // The factory must be thread-safe if worktrees are registered concurrently.
  using WatcherFactory =
      absl::AnyInvocable<absl::StatusOr<std::unique_ptr<Watcher>>(
          const std::string&, const WatcherOptions&)>;
  std::shared_ptr<WatcherFactory> watcher_factory =
      std::make_shared<WatcherFactory>(NewWatcher);
  WatcherOptions watcher;
  // Test-only: injects a failure before a restored view is scanned.
  std::shared_ptr<absl::AnyInvocable<absl::Status(absl::string_view)>>
      restore_scan_hook;
  // How often to reconcile a worktree whose watcher failed.
  absl::Duration watch_failure_rescan_interval = absl::Seconds(1);
  // A crashed git can leave index.lock behind. Reconcile after this wait; a
  // later HEAD or index event reconciles any concurrent git update again.
  absl::Duration index_lock_wait = absl::Seconds(2);
  // Test-only: runs just after a file is opened, before its first fstat.
  std::shared_ptr<WorkspaceReadHook> read_hook;
  // Test-only: runs after a file's bytes are read, before its second fstat.
  std::shared_ptr<WorkspaceReadHook> after_read_hook;
};

// What a search waits for before it captures the view. Thread-compatible.
enum class SearchConsistencyMode {
  // Searches the current views without waiting.
  kNone,
  // Waits until the worktree has applied `min_generation`.
  kNotify,
  // Syncs the watcher before waiting for all generations assigned by its
  // callbacks through the sync barrier (and min_generation). A successful
  // result includes writes before the call that the watcher delivered, subject
  // to the file-read stability and eligibility rules. On timeout it returns
  // the current view with stale set. Without a watcher, waits for the latest
  // generation assigned at the start of the search, as kNotify does.
  kSync,
};

// A definition-site lookup within one view.
struct FindSymbolRequest {
  std::string name;
  bool case_insensitive = false;
  std::vector<SymbolKind> kinds;
  bool definitions_only = false;
  std::string path_glob;
  int64_t max_results = 50;
};

struct SymbolLocation {
  std::string path;
  uint32_t line = 0;
  SymbolKind kind = SymbolKind::kFunction;
  bool is_definition = false;
  std::string source_line;

  friend bool operator==(const SymbolLocation&,
                         const SymbolLocation&) = default;
};

struct FindSymbolResponse {
  std::vector<SymbolLocation> occurrences;
  int64_t total_count = 0;
  int64_t pending_files = 0;
  bool stale = false;
  uint64_t generation = 0;
};

// The freshness a search requires. Thread-compatible.
struct SearchConsistency {
  // kNone by default: a search never waits unless asked to.
  SearchConsistencyMode mode = SearchConsistencyMode::kNone;
  // A generation returned by Notify or Rescan for the searched worktree.
  uint64_t min_generation = 0;
  // kNotify and kSync share this total wait budget. Not negative.
  absl::Duration timeout = absl::Milliseconds(50);
};

// A view-scoped search result and the freshness it had. Thread-compatible.
struct WorkspaceSearchResult {
  // The view-scoped result, including its search diagnostics.
  SearchResponse response;
  // True if kNotify missed min_generation, or kSync missed the watcher
  // barrier or an assigned generation. A failed watcher keeps kSync stale;
  // a failed index write holds generations until its retry succeeds.
  bool stale = false;
  // Every generation up to this one is visible in the searched view; later
  // ones may be partly visible.
  uint64_t generation = 0;
};

// Read-only snapshot of a registered worktree's view and ingest state.
struct WorkspaceWorktreeStatus {
  // Visible path count and number of distinct visible content identities.
  int64_t view_paths = 0;
  int64_t view_blobs = 0;
  // Highest applied and assigned generations.
  uint64_t applied_generation = 0;
  uint64_t assigned_generation = 0;
  // Paths waiting in the interactive lane, unstable reads, and skipped paths.
  int64_t queued_paths = 0;
  int64_t unstable_paths = 0;
  int64_t skipped_paths = 0;
  // Watcher health; failed means reconciliation uses periodic rescans.
  bool watcher_running = false;
  bool watcher_failed = false;
  // Published segment bytes and count, excluding unflushed content.
  int64_t index_bytes = 0;
  int64_t segment_count = 0;
  uint64_t pending_symbols = 0;
  uint64_t failed_symbols = 0;
};

// Cumulative counters. Durations and bulk counters cover registrations and
// rescans; skip counts cover both lanes. Thread-compatible.
struct WorkspaceStats {
  // Files read (including those found ineligible or unstable) by Notify
  // batches, by registrations and rescans to learn their Oid, and again by
  // the bulk write pass, which rereads new content in rank order.
  uint64_t interactive_files_read = 0;
  uint64_t bulk_files_read = 0;
  uint64_t bulk_files_reread = 0;
  // Content bytes of those reads: the first read of each file the bulk lane
  // could not resolve from a hint or a stat signature, and the rereads.
  uint64_t bulk_bytes_read = 0;
  uint64_t bulk_bytes_reread = 0;
  // Bulk reads of paths with a git skip hint whose Oid was neither indexed
  // nor known to be ineligible.
  uint64_t hint_reads = 0;
  // Paths resolved without reading: by a hint whose Oid is indexed or known
  // to be ineligible, or by a stat signature equal to the previous read's.
  uint64_t hints_reused = 0;
  uint64_t signatures_reused = 0;
  // Content bytes of the new documents each lane added to the index.
  uint64_t interactive_bytes_ingested = 0;
  uint64_t bulk_bytes_ingested = 0;
  // Reads whose file changed during all three attempts, reads that failed
  // with an I/O error, and bulk rereads whose Oid changed after the first
  // read. The paths keep their previous view entries.
  uint64_t unstable_files = 0;
  uint64_t read_errors = 0;
  uint64_t changed_before_write = 0;
  // Ineligible files, by SkipReason, each time a scan or Notify meets them.
  std::array<uint64_t, kNumSkipReasons> skipped = {};
  // Failed index writes. The paths keep their previous view entries until a
  // retry succeeds; the last failure is OK until the first one.
  uint64_t ingest_failures = 0;
  absl::Status last_ingest_error;
  // Watcher start or sync failures; the last non-timeout error is retained.
  // A failed watcher is replaced by periodic whole-worktree rescans.
  uint64_t watcher_failures = 0;
  // kSync cookie barriers that timed out; each such search returned stale
  // without failing the watcher.
  uint64_t sync_timeouts = 0;
  // Rescans that stopped waiting for index.lock after index_lock_wait.
  uint64_t index_lock_timeouts = 0;
  absl::Status last_watcher_error;
  // Index::Flush calls that retried after a failed write, failed ones
  // included.
  uint64_t ingest_retries = 0;
  // Bulk-lane wall time by phase: git or directory scan; path interning
  // and hint and state lookups; parallel stat, read, eligibility, and hash;
  // direct segment writes (including rereads); Index::Add(kBulk) calls
  // (including rereads); and the per-path state update and view batch.
  absl::Duration scan_time;
  absl::Duration classify_time;
  absl::Duration read_time;
  absl::Duration segment_write_time;
  absl::Duration memtable_add_time;
  absl::Duration publish_time;
  // Where the direct segment writes of a registration spend their time. Wall
  // times are of the bulk lane's thread; thread times add up over the threads
  // that did the work.
  absl::Duration bulk_wait_time;    // Waiting for the next reread.
  absl::Duration bulk_add_time;     // Writer Add and AddSymbols calls.
  absl::Duration bulk_finish_time;  // AddSegment less its callback.
  absl::Duration bulk_reread_thread_time;
  absl::Duration bulk_symbol_wait_thread_time;
  absl::Duration bulk_symbol_parse_thread_time;
  // Flush, compaction, and direct-segment counters of the index.
  IndexStats index;
};

// Owns the index, the lanes, and every registered view. The Env and any
// caller-supplied search executor must outlive the workspace. Calls that take
// a WorktreeId return NotFound for one that is not registered. Destruction
// may overlap Register and Rescan calls, which then return Cancelled; every
// other call must return first. Thread-safe.
class Workspace final {
 public:
  // Opens or creates the index in `index_dir`. Documents recovered from its
  // segments are known to be indexed, so registering a worktree whose blobs
  // they hold reads no hinted file. InvalidArgument for invalid options.
  static absl::StatusOr<std::unique_ptr<Workspace>> Open(
      Env& env, absl::string_view index_dir, WorkspaceOptions options = {});

  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;
  Workspace(Workspace&&) = delete;
  Workspace& operator=(Workspace&&) = delete;

  // Cancels bulk work, joins both lanes and the readers, and flushes the
  // index so that its documents survive a reopen.
  ~Workspace();

  // Canonicalizes `root`, optionally starts a watcher before the scan, scans
  // it on the bulk lane, and returns once its view is published. A git worktree
  // root is scanned with ScanWorktree; any other directory with
  // ListDirectoryFiles. AlreadyExists if the canonical root is registered,
  // InvalidArgument for a directory inside a git worktree other than its top
  // level, and Cancelled during destruction.
  struct RegisterOptions {
    bool watch = false;
  };
  absl::StatusOr<WorktreeId> Register(absl::string_view root) {
    return Register(root, {});
  }
  absl::StatusOr<WorktreeId> Register(absl::string_view root,
                                      RegisterOptions options)
      ABSL_LOCKS_EXCLUDED(mu_);
  // Removes a view. Its documents stay in the index until GC exists. Work
  // queued for the worktree completes without publishing.
  absl::Status Unregister(WorktreeId id) ABSL_LOCKS_EXCLUDED(mu_);
  // Queues repo-relative paths on the interactive lane in batches of at most
  // 256 and returns the generation that makes them visible. Never blocks on
  // the bulk lane, but generations apply in order: a queued or running rescan
  // of the worktree holds this one back until the rescan finishes, and a
  // failed index write until a retry succeeds. A path that is a directory
  // when read, or that is not one but has visible files below it, schedules
  // a Rescan, which completes its generation. InvalidArgument, with nothing
  // queued, if any path is invalid for PathTable.
  absl::StatusOr<uint64_t> Notify(WorktreeId id,
                                  absl::Span<const std::string> paths)
      ABSL_LOCKS_EXCLUDED(mu_);
  // Reconciles a fresh scan with the view on the bulk lane and returns the
  // generation it completed. Unchanged files are not read: git hints cover
  // clean tracked files and stat signatures cover the rest. A rescan that
  // fails returns the error; its generation completes anyway, except after a
  // failed index write, when it waits for the retry.
  absl::StatusOr<uint64_t> Rescan(WorktreeId id) ABSL_LOCKS_EXCLUDED(mu_);
  // Waits at most `timeout` for `generation` to be applied, holding no lock
  // that ingestion needs, and returns the applied generation, which is
  // smaller than `generation` on timeout.
  absl::StatusOr<uint64_t> AwaitGeneration(WorktreeId id, uint64_t generation,
                                           absl::Duration timeout)
      ABSL_LOCKS_EXCLUDED(mu_);
  // Searches the worktree's view: under kNotify, after AwaitGeneration;
  // under kSync, after a watcher barrier and its applied generation (or the
  // latest assigned generation when unwatched). On timeout, searches anyway
  // and marks the result stale.
  // Captures the view before the index ReadState. InvalidArgument for a
  // negative timeout or an invalid request.
  absl::StatusOr<WorkspaceSearchResult> Search(
      WorktreeId id, const SearchRequest& request,
      const SearchConsistency& consistency = {}) ABSL_LOCKS_EXCLUDED(mu_);
  // Does not wait for pending symbol extraction, even under kSync.
  absl::StatusOr<FindSymbolResponse> FindSymbol(
      WorktreeId id, const FindSymbolRequest& request,
      const SearchConsistency& consistency = {}) ABSL_LOCKS_EXCLUDED(mu_);
  // Snapshots already-tracked view and queue state without scanning the
  // filesystem. NotFound if id was removed; safe alongside ingestion.
  absl::StatusOr<WorkspaceWorktreeStatus> WorktreeStatus(WorktreeId id) const;
  // Returns the cumulative counters, including the index's.
  WorkspaceStats Stats() const ABSL_LOCKS_EXCLUDED(stats_mu_);
  // Freezes the index's active memtable and waits until it is a segment on
  // disk, so that the index directory holds everything ingested so far. For
  // benchmarks and tests; ingest flushes by itself when memtables fill.
  absl::Status FlushIndex();
  // Waits until the index has no queued or running compaction, such as the
  // merges that a registration's segments trigger. For benchmarks and tests.
  void WaitForCompactions();
  // Published roots and their process-local ids; excludes pending restores.
  std::vector<std::pair<std::string, WorktreeId>> Roots() const;

  // Number of distinct interned paths, including paths in all worktrees.
  size_t interned_paths() const { return paths_.size(); }

 private:
  struct Worktree;
  void RestoreViews(const std::vector<ManifestView>& views);
  absl::StatusOr<WorktreeId> RegisterView(absl::string_view root,
                                          RegisterOptions options,
                                          bool restoring);
  struct PathState;
  struct FileRead;
  struct Update;
  struct ScanSummary;
  struct NotifyJob;
  struct BulkJob;
  void StartWatcher(const std::shared_ptr<Worktree>& tree);
  void OnWatchEvents(const std::shared_ptr<Worktree>& tree,
                     std::vector<WatchEvent> events);
  void ApplyWatchEvents(const std::shared_ptr<Worktree>& tree,
                        const std::vector<WatchEvent>& events);
  void WatchRescan(const std::shared_ptr<Worktree>& tree);
  void DeferWatchRescan(const std::shared_ptr<Worktree>& tree,
                        uint64_t generation, absl::Time deadline);
  void ScheduleWatchTask(absl::Time due, absl::AnyInvocable<void()> task);
  void RunWatchTimer();
  bool WatchTimerHasWorkLocked() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(watch_timer_mu_);
  bool WatchTimerChangedLocked() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(watch_timer_mu_);
  void WatchFailure(const std::shared_ptr<Worktree>& tree,
                    const absl::Status& error);
  void RunWatchFallback(const std::shared_ptr<Worktree>& tree);
  void FinishWatchFallback(const std::shared_ptr<Worktree>& tree,
                           const std::shared_ptr<BulkJob>& job, bool started,
                           uint64_t failure_version);
  absl::Status RefreshExternalExcludes(const std::shared_ptr<Worktree>& tree);
  absl::Status SyncWatcher(const std::shared_ptr<Worktree>& tree,
                           absl::Time deadline);
  void StopWatcher(const std::shared_ptr<Worktree>& tree);
  bool StoppingLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  Workspace(std::unique_ptr<Index> index, std::string index_dir,
            WorkspaceOptions options);

  // Admission for calls that wait on the bulk lane; see the destructor.
  absl::Status EnterBulkCall() ABSL_LOCKS_EXCLUDED(mu_);
  void ExitBulkCall() ABSL_LOCKS_EXCLUDED(mu_);
  bool NoBulkCallsLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::StatusOr<std::shared_ptr<Worktree>> FindTree(WorktreeId id) const
      ABSL_LOCKS_EXCLUDED(mu_);
  // Generations: assigned by callers, completed in any order by the lanes,
  // and applied in order.
  static uint64_t AssignGeneration(Worktree& tree);
  static void Complete(Worktree& tree, uint64_t generation);
  // Completes `generations` and those held earlier, or holds them all while
  // the worktree has paths whose content failed to reach the index.
  static void CompleteWhenIndexed(Worktree& tree,
                                  std::vector<uint64_t> generations);
  static uint64_t WaitForApplied(Worktree& tree, uint64_t generation,
                                 absl::Duration timeout);
  // Returns a new read sequence, larger than every earlier one.
  uint64_t NextSeq();

  // Queues Notify jobs of at most 256 paths each and returns the generation
  // of the last. Requires !stopping_.
  uint64_t QueueNotifyLocked(const std::shared_ptr<Worktree>& tree,
                             absl::Span<const std::string> paths,
                             absl::Span<const PathId> ids)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Queues a rescan completing `generation`, or joins a queued rescan of the
  // same worktree. Returns null during destruction.
  std::shared_ptr<BulkJob> EnqueueRescan(const std::shared_ptr<Worktree>& tree,
                                         uint64_t generation)
      ABSL_LOCKS_EXCLUDED(mu_);
  void ScheduleBulkLocked(std::shared_ptr<BulkJob> job)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void RunBulkJob(BulkJob& job) ABSL_LOCKS_EXCLUDED(mu_);
  // After a failed index write: schedules RunIngestRecovery on the bulk lane
  // unless it is scheduled already, in which case it tries again.
  void RequestIngestRecovery() ABSL_LOCKS_EXCLUDED(mu_);
  // Retries Index::Flush with backoff until it succeeds and no write failed
  // meanwhile, then rereads every path whose content failed to reach the
  // index. Runs on the bulk lane and yields it to queued jobs while waiting.
  void RunIngestRecovery() ABSL_LOCKS_EXCLUDED(mu_);
  bool RecoveryWaitOverLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Queues a reread of each worktree's failed paths: a Notify job, or a
  // rescan when there are more than one Notify batch holds.
  void ResubmitFailed() ABSL_LOCKS_EXCLUDED(mu_);
  // Runs on the interactive lane until the Notify queue is empty.
  void DrainInteractive() ABSL_LOCKS_EXCLUDED(mu_);
  // Reads, adds, and publishes coalesced Notify jobs of one worktree.
  void ApplyNotifyBatch(const std::shared_ptr<Worktree>& tree,
                        absl::Span<const NotifyJob> jobs);

  // Reads one file without following symlinks, retrying while its fstat
  // signature changes during the read. An interactive read lstats the path
  // first, opens only a regular file, and on a case-insensitive volume
  // treats a path spelled unlike its directory entry as missing; bulk paths
  // come from a scan that did all this. Thread-safe.
  FileRead ReadFile(const Worktree& tree, absl::string_view path, bool bulk);
  // Turns a read into an update: eligibility, metadata, and Oid. `retain`
  // keeps the bytes of new content.
  void ApplyRead(Update& update, FileRead& read, TrigramExtractor& extractor,
                 bool retain, WorkspaceStats& delta) const;

  // One bulk pass over a worktree: scan, classify, read, write, publish.
  // A failed index write is counted in `delta.ingest_failures`.
  absl::Status Reconcile(Worktree& tree, WorkspaceStats& delta);
  // Parallel stat and first read of the updates that need them.
  void ReadForBulk(const Worktree& tree, std::vector<Update>& updates,
                   WorkspaceStats& delta);
  // Adds the documents of new Oids among `updates` and resolves the BlobId of
  // every present update; an update whose content could not be added keeps
  // its previous entry. The interactive lane adds the retained bytes; the
  // bulk lane rereads files in rank order and writes direct segments when
  // the new content fills a memtable.
  absl::Status AddNewContent(const Worktree& tree, std::vector<Update>& updates,
                             bool bulk, WorkspaceStats& delta);
  // Rereads the updates at `order` and writes them, recording what it wrote
  // in `added` and the Oids of files that changed since their first read in
  // `changed`.
  absl::Status WriteBulk(const Worktree& tree, std::vector<Update>& updates,
                         absl::Span<const size_t> order, bool segments,
                         absl::flat_hash_map<Oid, BlobId>& added,
                         std::vector<Oid>& changed, WorkspaceStats& delta);
  // Applies a reread whose Oid changed, adding new content through the
  // memtable.
  absl::Status AddChanged(Update& update, FileRead& read,
                          TrigramExtractor& extractor,
                          absl::flat_hash_map<Oid, BlobId>& added,
                          WorkspaceStats& delta);
  // Adds each `changed` Oid from the next path that still holds it.
  absl::Status AddFromOtherPaths(const Worktree& tree,
                                 std::vector<Update>& updates,
                                 absl::Span<const size_t> unresolved,
                                 absl::Span<const Oid> changed,
                                 absl::flat_hash_map<Oid, BlobId>& added,
                                 WorkspaceStats& delta);
  // Applies updates newer than each path's last applied read, records the
  // paths whose content failed to reach the index, then publishes one view
  // batch. A scan also deletes the paths it omitted and replaces the tracked
  // and nested-repository sets.
  void Publish(Worktree& tree, std::vector<Update>& updates,
               ScanSummary* absl_nullable scan);

  // The BlobId of `oid` if its document is published.
  std::optional<BlobId> FindIndexed(const Oid& oid) const
      ABSL_LOCKS_EXCLUDED(content_mu_);
  bool IndexedLocked(BlobId blob) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(content_mu_);
  // Records published documents, after the call that published them.
  void MarkIndexed(const absl::flat_hash_map<Oid, BlobId>& added)
      ABSL_LOCKS_EXCLUDED(content_mu_);
  // Counts skipped updates and remembers their Oids as ineligible.
  void RecordSkips(absl::Span<const Update> updates, WorkspaceStats& delta)
      ABSL_LOCKS_EXCLUDED(content_mu_);
  void AddStats(const WorkspaceStats& delta) ABSL_LOCKS_EXCLUDED(stats_mu_);
  // Pin documents while a lane resolves content identities against the index.
  void PinGc();
  void UnpinGc();
  void RefreshIndexed();
  void DropReferences(Worktree& tree);

  std::unique_ptr<Index> index_;
  const std::string index_dir_;
  WorkspaceOptions options_;
  PathTable paths_;
  std::unique_ptr<SearchExecutor> executor_;
  // Monotonic source of read sequences; relaxed ordering suffices because
  // results are compared under the worktree mutex that applies them.
  std::atomic<uint64_t> read_seq_{0};
  // Set once destruction begins; polled by bulk work between files.
  std::atomic<bool> cancelling_{false};

  mutable absl::Mutex mu_;
  absl::flat_hash_map<WorktreeId, std::shared_ptr<Worktree>> trees_
      ABSL_GUARDED_BY(mu_);
  // Persisted roots awaiting restoration. A client waits for their scan.
  std::vector<std::string> restore_order_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_set<std::string> restore_requested_ ABSL_GUARDED_BY(mu_);
  // Per pending view: attempts started and finished, and the last failure.
  // A client registering the root waits for one fresh attempt.
  struct RestoreAttempts {
    uint64_t started = 0;
    uint64_t finished = 0;
    absl::Status status;
  };
  absl::flat_hash_map<std::string, RestoreAttempts> restore_attempts_
      ABSL_GUARDED_BY(mu_);
  absl::flat_hash_set<std::string> restoring_ ABSL_GUARDED_BY(mu_);
  // Restored registrations remain idempotent for later client calls.
  absl::flat_hash_set<std::string> restored_roots_ ABSL_GUARDED_BY(mu_);
  // Published roots in registration order.
  std::vector<std::string> registration_order_ ABSL_GUARDED_BY(mu_);
  // Canonical roots registered or registering.
  absl::flat_hash_set<std::string> roots_ ABSL_GUARDED_BY(mu_);
  uint32_t next_id_ ABSL_GUARDED_BY(mu_) = 0;
  // Queued Notify jobs, oldest first.
  std::vector<NotifyJob> notify_jobs_ ABSL_GUARDED_BY(mu_);
  bool interactive_draining_ ABSL_GUARDED_BY(mu_) = false;
  // Rescans queued but not started, by worktree.
  absl::flat_hash_map<const Worktree*, std::shared_ptr<BulkJob>> queued_rescans_
      ABSL_GUARDED_BY(mu_);
  bool stopping_ ABSL_GUARDED_BY(mu_) = false;
  int bulk_calls_ ABSL_GUARDED_BY(mu_) = 0;
  // Bulk jobs scheduled but not started.
  int bulk_queued_ ABSL_GUARDED_BY(mu_) = 0;
  // Ingest recovery: whether RunIngestRecovery is queued or running, whether
  // a write failed since its last attempt began, its backoff, and when the
  // last successful recovery ended.
  bool recovery_scheduled_ ABSL_GUARDED_BY(mu_) = false;
  bool recovery_requested_ ABSL_GUARDED_BY(mu_) = false;
  absl::Duration recovery_delay_ ABSL_GUARDED_BY(mu_);
  absl::Time recovery_due_ ABSL_GUARDED_BY(mu_);
  absl::Time recovery_ended_ ABSL_GUARDED_BY(mu_) = absl::InfinitePast();

  // What the index holds. A BlobId is set once a document with its Oid is in
  // a published ReadState; the registry alone interns Oids earlier.
  absl::Mutex gc_mu_;
  int gc_pins_ ABSL_GUARDED_BY(gc_mu_) = 0;

  mutable absl::Mutex content_mu_;
  Bitset indexed_ ABSL_GUARDED_BY(content_mu_);
  // The MANIFEST sequence `indexed_` was computed from.
  std::optional<uint64_t> indexed_manifest_seq_ ABSL_GUARDED_BY(content_mu_);
  // Oids whose content the eligibility policy rejects, which is a pure
  // function of the content, so a hint with such an Oid is never read.
  absl::flat_hash_map<Oid, SkipReason> ineligible_ ABSL_GUARDED_BY(content_mu_);

  mutable absl::Mutex stats_mu_;
  WorkspaceStats stats_ ABSL_GUARDED_BY(stats_mu_);

  // Scratch space of the interactive lane, used only on its thread.
  TrigramExtractor interactive_extractor_;
  // Destroyed explicitly, in this order, by the destructor.
  std::unique_ptr<ThreadPool> interactive_lane_;
  std::unique_ptr<ThreadPool> bulk_lane_;
  std::thread restore_thread_;
  std::unique_ptr<ThreadPool> readers_;
  struct BulkSymbolWorker {
    absl::Mutex mu;
    SymbolExtractor extractor;
  };
  std::vector<std::unique_ptr<BulkSymbolWorker>> bulk_symbol_workers_;
  std::unique_ptr<ThreadPool> bulk_symbol_pool_;
  std::atomic<size_t> next_bulk_symbol_worker_{0};
  std::unique_ptr<ThreadPool> watch_fallback_lane_;
  struct TimedWatchTask {
    absl::Time due;
    absl::AnyInvocable<void()> run;
  };
  absl::Mutex watch_timer_mu_;
  std::vector<TimedWatchTask> watch_tasks_ ABSL_GUARDED_BY(watch_timer_mu_);
  bool watch_timer_stopping_ ABSL_GUARDED_BY(watch_timer_mu_) = false;
  bool watch_timer_changed_ ABSL_GUARDED_BY(watch_timer_mu_) = false;
  std::thread watch_timer_;
};

}  // namespace ferret

#endif  // FERRET_WORKSPACE_WORKSPACE_H_

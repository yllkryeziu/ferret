// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/index.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

#include "ferret/base/env.h"
#include "ferret/base/thread_pool.h"
#include "ferret/index/blob_registry.h"
#include "ferret/index/compaction_policy.h"
#include "ferret/index/compactor.h"
#include "ferret/index/manifest.h"
#include "ferret/index/memtable.h"
#include "ferret/index/read_state.h"
#include "ferret/index/reaper.h"
#include "ferret/index/segment.h"
#include "ferret/index/segment_writer.h"
#include "ferret/index/version_set.h"
#include "ferret/postings/ids.h"
#include "ferret/symbols/symbol_extractor.h"
#include "ferret/text/chunker.h"
#include "ferret/text/language.h"

namespace ferret {

Index::Index(Env& env, IndexOptions options)
    : env_(env),
      options_(std::move(options)),
      reaper_(std::make_shared<Reaper>()) {
  options_.memtable.async_symbols = options_.memtable.segment.write_symbols;
  if (options_.memtable.segment.write_symbols) {
    for (int i = 0; i < options_.symbol_threads; ++i) {
      symbol_workers_.push_back(std::make_unique<SymbolWorker>());
    }
    symbol_pool_ = std::make_unique<ThreadPool>(options_.symbol_threads,
                                                ThreadQos::kUtility);
  }
}

absl::StatusOr<std::pair<std::unique_ptr<Index>, RecoveryReport>> Index::Open(
    Env& env, absl::string_view dir, IndexOptions options) {
  if (options.symbol_threads < 1 || options.max_frozen_memtables == 0 ||
      options.max_frozen_memtables > 4 ||
      options.max_memtable_age < absl::ZeroDuration() ||
      options.gc_grace_period < absl::ZeroDuration()) {
    return absl::InvalidArgumentError("invalid index flush options");
  }
  const CompactionOptions& compaction = options.compaction;
  if (compaction.enabled &&
      (compaction.tier_base_bytes == 0 || compaction.tier_growth_factor < 2 ||
       compaction.tier_min_segments < 2 ||
       compaction.rewrite_dead_percent == 0 ||
       compaction.rewrite_dead_percent > 100 ||
       compaction.max_live_segments == 0 || compaction.small_lane_bytes == 0 ||
       compaction.forced_merge_segments < 2 ||
       compaction.retry_initial_delay <= absl::ZeroDuration() ||
       compaction.retry_max_delay < compaction.retry_initial_delay)) {
    return absl::InvalidArgumentError("invalid compaction options");
  }
  std::unique_ptr<Index> index(new Index(env, std::move(options)));
  ABSL_ASSIGN_OR_RETURN(
      auto opened, VersionSet::Open(env, dir, *index->reaper_, index->blobs_));
  index->versions_ = std::move(opened.first);
  auto state = std::make_shared<ReadState>();
  state->reaper = index->reaper_;
  state->blobs = index->blobs_;
  state->version = index->versions_->current();
  {
    absl::MutexLock lock(index->state_mu_);
    index->active_ =
        std::make_shared<Memtable>(index->options_.memtable, index->blobs_);
    state->mem = index->active_;
    index->current_ = std::move(state);
  }
  {
    absl::MutexLock lock(index->state_mu_);
    index->ScheduleLocked();
  }
  index->worker_ = std::thread([ptr = index.get()] {
    ApplyThreadQos(ThreadQos::kUtility);
    ptr->Run();
  });
  if (index->options_.compaction.enabled) {
    for (size_t lane = 0; lane < 2; ++lane) {
      index->compaction_workers_[lane] = std::thread([ptr = index.get(), lane] {
        ApplyThreadQos(ThreadQos::kUtility);
        ptr->RunCompaction(lane);
      });
    }
  }
  return std::pair(std::move(index), std::move(opened.second));
}

Index::~Index() {
  {
    absl::MutexLock lock(state_mu_);
    stopping_ = true;
    cancelled_.store(true, std::memory_order_relaxed);
    state_mu_.Await(absl::Condition(this, &Index::NoSegmentsLocked));
  }
  // Open() destroys a partly built index, before any thread starts, when
  // recovery fails.
  symbol_pool_.reset();
  if (worker_.joinable()) worker_.join();
  for (std::thread& thread : compaction_workers_) {
    if (thread.joinable()) thread.join();
  }
  {
    absl::MutexLock lock(state_mu_);
    current_.reset();
    active_.reset();
    pending_.clear();
  }
  versions_.reset();
}

absl::Status Index::AdjustReferences(absl::Span<const ReferenceDelta> deltas) {
  return AdjustReferences(deltas, [] {});
}

absl::Status Index::AdjustReferences(absl::Span<const ReferenceDelta> deltas,
                                     absl::FunctionRef<void()> publish) {
  absl::MutexLock lock(state_mu_);
  absl::flat_hash_map<uint32_t, int64_t> changes;
  for (const ReferenceDelta& change : deltas) {
    const uint32_t id = static_cast<uint32_t>(change.blob);
    int64_t& sum = changes[id];
    if ((change.delta > 0 &&
         sum > std::numeric_limits<int64_t>::max() - change.delta) ||
        (change.delta < 0 &&
         sum < std::numeric_limits<int64_t>::min() - change.delta)) {
      return absl::InvalidArgumentError("reference delta overflow");
    }
    sum += change.delta;
  }
  for (const auto& [id, delta] : changes) {
    const uint64_t count = id < references_.size() ? references_[id].count : 0;
    if ((delta < 0 && (delta == std::numeric_limits<int64_t>::min() ||
                       static_cast<uint64_t>(-delta) > count)) ||
        (delta > 0 && static_cast<uint64_t>(delta) >
                          std::numeric_limits<uint64_t>::max() - count)) {
      // A negative count is a programming error in the view owner.
      return absl::InvalidArgumentError("invalid reference count");
    }
  }
  const absl::Time now = absl::Now();
  std::vector<BlobId> newly_dead;
  for (const auto& [id, delta] : changes) {
    if (delta <= 0) continue;
    if (id >= references_.size()) references_.resize(id + 1);
    references_[id].count += static_cast<uint64_t>(delta);
  }
  // Install cannot drop a newly referenced document between the increment
  // and publication. An install prepared earlier restores it before exposing
  // its output; a reader captures the view before the ReadState.
  publish();
  for (const auto& [id, delta] : changes) {
    if (delta >= 0) continue;
    // A view can precede ingestion. Do not start grace for unindexed IDs.
    if (id >= references_.size()) references_.resize(id + 1);
    ReferenceState& state = references_[id];
    const uint64_t previous = state.count;
    state.count -= static_cast<uint64_t>(-delta);
    if (previous != 0 && state.count == 0) {
      state.dead_since = now;
      if (!gc_suspended_) {
        newly_dead.push_back(static_cast<BlobId>(id));
      }
    }
  }
  if (!newly_dead.empty()) QueueGcLocked(now, std::move(newly_dead));
  return absl::OkStatus();
}

void Index::QueueGcLocked(absl::Time now, std::vector<BlobId> blobs) {
  gc_pending_.push_back({now + options_.gc_grace_period, std::move(blobs)});
  UpdateGcDeadlineLocked();
}

void Index::UpdateGcDeadlineLocked() {
  // A scan walks every published document under state_mu_ (about 2 ms for
  // Linux), and each view batch that zeroes a count queues its own deadline.
  // Coalescing bounds scans to ten per grace period under steady edits.
  gc_deadline_ = gc_pending_.empty()
                     ? absl::InfiniteFuture()
                     : std::max(gc_pending_.front().deadline,
                                last_gc_scan_ + options_.gc_grace_period / 10);
}

std::vector<ManifestView> Index::Views() const {
  return versions_->current()->views;
}

absl::Status Index::RecordView(const ManifestView& view) {
  absl::MutexLock lock(install_mu_);
  std::vector<ManifestView> views = versions_->current()->views;
  for (const ManifestView& existing : views) {
    if (existing.canonical_root == view.canonical_root) {
      return existing == view ? absl::OkStatus()
                              : absl::AlreadyExistsError(
                                    "view root belongs to another repository");
    }
  }
  views.push_back(view);
  return versions_->Install(VersionEdit{.views = std::move(views)});
}

absl::Status Index::ForgetView(absl::string_view canonical_root) {
  absl::MutexLock lock(install_mu_);
  std::vector<ManifestView> views = versions_->current()->views;
  const auto end = std::remove_if(
      views.begin(), views.end(), [canonical_root](const ManifestView& view) {
        return view.canonical_root == canonical_root;
      });
  if (end == views.end()) return absl::OkStatus();
  views.erase(end, views.end());
  return versions_->Install(VersionEdit{.views = std::move(views)});
}

void Index::ResumeGc() {
  absl::MutexLock lock(state_mu_);
  if (!gc_suspended_) return;
  const absl::Time now = absl::Now();
  references_.resize(std::max(references_.size(), blobs_->size()));
  std::vector<BlobId> unreferenced;
  for (size_t i = 0; i < references_.size(); ++i) {
    if (references_[i].count == 0) {
      // Resync starts a fresh grace period, including after suspension.
      references_[i].dead_since = now;
      unreferenced.push_back(static_cast<BlobId>(i));
    }
  }
  gc_suspended_ = false;
  if (!unreferenced.empty()) QueueGcLocked(now, std::move(unreferenced));
  if (options_.gc_grace_period == absl::ZeroDuration() &&
      !gc_pending_.empty()) {
    ScanGcLocked();
    gc_pending_.clear();
    gc_deadline_ = absl::InfiniteFuture();
  }
  ScheduleLocked();
}

void Index::SuspendGc() {
  absl::MutexLock lock(state_mu_);
  gc_suspended_ = true;
  gc_pending_.clear();
  gc_deadline_ = absl::InfiniteFuture();
  dead_by_segment_.clear();
  scanned_dead_docs_ = 0;
}

void Index::RegisterBlobLocked(BlobId blob) {
  const size_t id = static_cast<uint32_t>(blob);
  if (id < references_.size()) {
    ReferenceState& state = references_[id];
    if (!gc_suspended_ && state.count == 0 &&
        state.dead_since == absl::InfinitePast()) {
      state.dead_since = absl::Now();
      QueueGcLocked(state.dead_since, {blob});
    }
    return;
  }
  references_.resize(id + 1);
  if (!gc_suspended_) {
    const absl::Time now = absl::Now();
    references_[id].dead_since = now;
    QueueGcLocked(now, {blob});
  }
}

bool Index::DropLocked(BlobId blob, absl::Time now) const {
  if (gc_suspended_) return false;
  const uint32_t id = static_cast<uint32_t>(blob);
  if (id >= references_.size()) return false;
  const ReferenceState& state = references_[id];
  return state.count == 0 && now - state.dead_since >= options_.gc_grace_period;
}

bool Index::IngestIdleLocked() const {
  return !ingest_in_progress_ || stopping_;
}

bool Index::Restorable(const SegmentDoc& doc) const {
  return !doc.hint_path.empty() &&
         doc.content.size() < std::numeric_limits<uint32_t>::max() &&
         doc.content.size() <= options_.memtable.max_content_bytes &&
         doc.num_chunks <= options_.memtable.max_chunks;
}

absl::Status Index::RestoreLocked(const SegmentDoc& doc) {
  const SegmentDocument source = {
      .hint_path = doc.hint_path,
      .content = doc.content,
      .rank_key = {.tier = doc.tier, .prior = doc.prior, .oid = doc.oid},
      .language = doc.language,
      .long_lines = doc.long_lines};
  absl::Status status = active_->Add(source, /*defer_symbols=*/false).status();
  if (absl::IsResourceExhausted(status)) {
    FreezeLocked();
    status = active_->Add(source, /*defer_symbols=*/false).status();
  }
  if (status.ok() && active_->ShouldFlush()) FreezeLocked();
  return status;
}

std::shared_ptr<const ReadState> Index::GetReadState() const {
  std::shared_ptr<const ReadState> base;
  uint32_t watermark;
  {
    absl::MutexLock lock(state_mu_);
    base = current_;
    watermark = base->mem->published_docs();
  }
  auto snapshot = std::make_shared<ReadState>(*base);
  snapshot->mem_watermark = watermark;
  return snapshot;
}

void Index::FreezeLocked() {
  if (current_->mem->published_docs() == 0) return;
  auto next = std::make_shared<ReadState>(*current_);
  pending_.push_back(active_);
  next->frozen.push_back(current_->mem);
  active_ = std::make_shared<Memtable>(options_.memtable, blobs_);
  next->mem = active_;
  next->mem_watermark = 0;
  current_ = std::move(next);
}

bool Index::SpaceAvailableLocked() const {
  return pending_.size() < options_.max_frozen_memtables || !failure_.ok();
}

bool Index::BulkReadyLocked() const {
  return (!ingest_in_progress_ && SpaceAvailableLocked()) || stopping_;
}

bool Index::InteractiveReadyLocked() const {
  return (!ingest_in_progress_ &&
          (pending_.size() < options_.max_frozen_memtables + 1 ||
           !failure_.ok())) ||
         stopping_;
}

bool Index::NoSegmentsLocked() const { return active_segments_ == 0; }

absl::Status Index::Add(const SegmentDocument& document) {
  return Add(document, IngestLane::kBulk);
}

absl::Status Index::Add(const SegmentDocument& document, IngestLane lane) {
  std::shared_ptr<Memtable> active;
  {
    absl::MutexLock lock(state_mu_);
    if (!failure_.ok()) {
      failure_reported_ = true;
      return failure_;
    }
    const bool ready = lane == IngestLane::kBulk ? BulkReadyLocked()
                                                 : InteractiveReadyLocked();
    if (!ready) {
      ++(lane == IngestLane::kBulk ? stats_.bulk_ingest_waits
                                   : stats_.interactive_ingest_waits);
    }
    state_mu_.Await(
        lane == IngestLane::kBulk
            ? absl::Condition(this, &Index::BulkReadyLocked)
            : absl::Condition(this, &Index::InteractiveReadyLocked));
    if (stopping_) return absl::CancelledError("index stopped");
    if (!failure_.ok()) {
      failure_reported_ = true;
      return failure_;
    }
    ingest_in_progress_ = true;
    active = active_;
  }
  // The worker must not freeze the active memtable during this append.
  absl::StatusOr<DocId> added = active->Add(document);
  absl::Status status = added.status();
  if (status.ok() && symbol_pool_ && HasSymbolGrammar(document.language)) {
    const size_t slot =
        next_symbol_worker_.fetch_add(1, std::memory_order_relaxed) %
        symbol_workers_.size();
    SymbolWorker* worker = symbol_workers_[slot].get();
    const DocId id = *added;
    const auto gate = options_.symbol_extraction_gate;
    const auto reached = options_.symbol_extraction_reached;
    const auto counter = options_.symbol_extractions;
    symbol_pool_->Schedule([this, active, worker, id, gate, reached, counter] {
      if (reached && !reached->HasBeenNotified()) reached->Notify();
      while (gate &&
             !gate->WaitForNotificationWithTimeout(absl::Milliseconds(10))) {
        if (cancelled()) {
          active->AttachSymbols(id, {}, /*failed=*/true);
          return;
        }
      }
      const SegmentDoc doc = active->doc(id);
      absl::MutexLock lock(worker->mu);
      if (counter) counter->fetch_add(1, std::memory_order_relaxed);
      absl::StatusOr<std::vector<ExtractedSymbol>> symbols =
          worker->extractor.Extract(doc.language, doc.content);
      active->AttachSymbols(
          id,
          symbols.ok() ? std::move(*symbols) : std::vector<ExtractedSymbol>{},
          !symbols.ok());
    });
  }
  absl::MutexLock lock(state_mu_);
  ingest_in_progress_ = false;
  if (!status.ok()) return status;
  RegisterBlobLocked(blobs_->Intern(document.rank_key.oid));
  const auto oldest = active->oldest_document_time();
  if (active->ShouldFlush() ||
      (oldest && absl::Now() - *oldest >= options_.max_memtable_age)) {
    FreezeLocked();
  }
  return absl::OkStatus();
}

absl::StatusOr<SegmentStats> Index::AddSegment(
    absl::FunctionRef<absl::Status(SegmentWriter&)> write,
    ThreadPool* finish_pool, int finish_threads) {
  {
    absl::MutexLock lock(state_mu_);
    if (stopping_) return absl::CancelledError("index stopped");
    ++active_segments_;
  }
  // The destructor waits for callbacks to finish. No index lock is held while
  // writing; only file allocation and installation serialize with flushes.
  absl::Status status;
  absl::StatusOr<SegmentStats> written =
      absl::UnknownError("segment not written");
  std::string path;
  {
    absl::MutexLock lock(install_mu_);
    auto reserved = versions_->NewSegmentFile();
    if (reserved.ok()) {
      path = *reserved;
    } else {
      status = reserved.status();
    }
  }
  if (status.ok() && cancelled()) {
    status = absl::CancelledError("index stopped");
  }
  if (status.ok()) {
    SegmentWriterOptions writer_options = options_.memtable.segment;
    if (finish_pool != nullptr) {
      writer_options.finish_pool = finish_pool;
      writer_options.finish_threads = finish_threads;
    }
    auto writer = SegmentWriter::Create(env_, path, writer_options);
    if (!writer.ok()) {
      status = writer.status();
    } else {
      status = write(**writer);
      if (status.ok() && cancelled()) {
        status = absl::CancelledError("index stopped");
      }
      if (status.ok() && (*writer)->num_docs() != 0) {
        written = (*writer)->Finish();
        status = written.status();
      } else if (status.ok()) {
        written = SegmentStats{};
      }
    }
  }
  if (status.ok() && written->num_docs != 0) {
    absl::MutexLock install_lock(install_mu_);
    if (cancelled()) {
      status = absl::CancelledError("index stopped");
    } else {
      status = versions_->Install(VersionEdit{.add = {path}});
      if (status.ok()) {
        absl::MutexLock lock(state_mu_);
        auto next = std::make_shared<ReadState>(*current_);
        next->version = versions_->current();
        for (const VersionSegment& entry : next->version->segments) {
          for (BlobId blob : *entry.blob_of_doc) RegisterBlobLocked(blob);
        }
        current_ = std::move(next);
        stats_.bulk_segment_bytes += written->file_bytes;
        ++stats_.bulk_segments;
        ScheduleLocked();
      }
    }
  }
  if (!status.ok() || written->num_docs == 0) {
    if (!path.empty()) {
      env_.DeleteFile(path).IgnoreError();
      absl::MutexLock lock(install_mu_);
      versions_->DiscardSegmentFile(path);
    }
  }
  {
    absl::MutexLock lock(state_mu_);
    --active_segments_;
  }
  if (!status.ok()) return status;
  return *written;
}

bool Index::RetryDoneLocked() const {
  return finished_flushes_ >= retry_until_ || stopping_;
}

bool Index::FlushDoneLocked() const {
  return pending_.empty() || !failure_.ok();
}

absl::Status Index::Flush() {
  absl::MutexLock lock(state_mu_);
  if (!failure_.ok()) {
    if (!failure_reported_) {
      failure_reported_ = true;
      return failure_;
    }
    // Keep the error sticky for both Add lanes until the worker finishes the
    // retry. Otherwise a concurrent Add could append to a failed memtable.
    retry_until_ = finished_flushes_ + 1;
    retry_ = true;
    state_mu_.Await(absl::Condition(this, &Index::RetryDoneLocked));
    if (stopping_) return absl::CancelledError("index stopped");
    if (!failure_.ok()) {
      failure_reported_ = true;
      return failure_;
    }
  }
  state_mu_.Await(absl::Condition(this, &Index::BulkReadyLocked));
  if (stopping_) return absl::CancelledError("index stopped");
  if (!failure_.ok()) {
    failure_reported_ = true;
    return failure_;
  }
  FreezeLocked();
  state_mu_.Await(absl::Condition(this, &Index::FlushDoneLocked));
  if (!failure_.ok()) failure_reported_ = true;
  return failure_;
}

bool Index::WorkerReadyLocked() const {
  if (stopping_) return true;
  if (!failure_.ok() && !retry_) return false;
  if (!pending_.empty()) return true;
  if (ingest_in_progress_ || pending_.size() >= options_.max_frozen_memtables) {
    return false;
  }
  const auto oldest = current_->mem->oldest_document_time();
  return oldest && absl::Now() - *oldest >= options_.max_memtable_age;
}

void Index::Run() {
  for (;;) {
    std::shared_ptr<Memtable> mem;
    bool retrying = false;
    {
      absl::MutexLock lock(state_mu_);
      state_mu_.AwaitWithTimeout(
          absl::Condition(this, &Index::WorkerReadyLocked), absl::Seconds(1));
      if (stopping_) return;
      if (!WorkerReadyLocked()) continue;
      if (pending_.empty()) FreezeLocked();
      mem = pending_.front();
      retrying = !failure_.ok();
      retry_ = false;
    }
    // Reserve, write, then install. The install lock is released while the
    // memtable writes its segment, so another lane can publish a merge.
    absl::Status status;
    absl::StatusOr<SegmentStats> written =
        absl::UnknownError("segment allocation failed");
    std::string path;
    {
      absl::MutexLock lock(install_mu_);
      auto reserved = versions_->NewSegmentFile();
      if (reserved.ok()) {
        path = *reserved;
      } else {
        status = reserved.status();
      }
    }
    std::vector<DocId> dropped;
    std::vector<bool> keep;
    if (status.ok()) {
      const uint32_t count = mem->published_docs();
      keep.resize(count, true);
      {
        absl::MutexLock lock(state_mu_);
        const absl::Time now = absl::Now();
        for (DocId doc = 0; doc < count; ++doc) {
          if (DropLocked(mem->blob_of_doc(doc), now)) {
            keep[doc] = false;
            dropped.push_back(doc);
          }
        }
      }
      written = mem->WriteSegment(env_, path, count,
                                  [&](DocId doc) { return keep[doc]; });
      status = written.status();
    }
    if (status.ok()) {
      if (options_.flush_install_reached &&
          !options_.flush_install_reached->HasBeenNotified()) {
        options_.flush_install_reached->Notify();
      }
      if (options_.flush_install_gate) {
        options_.flush_install_gate->WaitForNotification();
      }
      absl::MutexLock install_lock(install_mu_);
      // Validate restorations before the install, so an unindexable document
      // leaves the old mapping published and the output can be discarded.
      std::vector<bool> restored(dropped.size(), false);
      {
        absl::MutexLock lock(state_mu_);
        state_mu_.Await(absl::Condition(this, &Index::IngestIdleLocked));
        for (size_t i = 0; i < dropped.size(); ++i) {
          const DocId doc = dropped[i];
          if (!stopping_ && !DropLocked(mem->blob_of_doc(doc), absl::Now())) {
            status = RestoreLocked(mem->doc(doc));
            if (!status.ok()) break;
            restored[i] = true;
          }
        }
      }
      if (status.ok()) status = versions_->Install(VersionEdit{.add = {path}});
      if (status.ok()) {
        absl::MutexLock lock(state_mu_);
        state_mu_.Await(absl::Condition(this, &Index::IngestIdleLocked));
        // The installed output is not visible until publication. The input
        // remains mapped; reference edits cannot interleave this check and
        // publication. Only newly resurrected copies need restoration.
        for (size_t i = 0; i < dropped.size(); ++i) {
          const DocId doc = dropped[i];
          if (!restored[i] && !stopping_ &&
              !DropLocked(mem->blob_of_doc(doc), absl::Now())) {
            // A failed append here cannot be rolled back from the MANIFEST.
            // Fail loudly rather than silently losing a referenced blob.
            CHECK_OK(RestoreLocked(mem->doc(doc)));
          }
        }
        stats_.flush_bytes += written->file_bytes;
        auto next = std::make_shared<ReadState>(*current_);
        next->version = versions_->current();
        next->frozen.erase(next->frozen.begin());
        current_ = std::move(next);
        pending_.pop_front();
        if (retrying) failure_ = absl::OkStatus();
        ++finished_flushes_;
        // A grace-period scan can run while this frozen table is writing.
        // Reclassify newly installed drop-eligible documents as segment
        // deaths. Documents still in grace have a queued deadline; scanning
        // them here would scan the entire index on every flush.
        bool eligible_installed = false;
        const absl::Time now = absl::Now();
        for (DocId doc = 0; doc < mem->published_docs(); ++doc) {
          if (keep[doc] && DropLocked(mem->blob_of_doc(doc), now)) {
            eligible_installed = true;
            break;
          }
        }
        if (!dropped.empty() || eligible_installed) ScanGcLocked();
        ScheduleLocked();
      }
    }
    if (!status.ok()) {
      // Recovery removes any orphan left after an install failure.
      if (!path.empty()) {
        env_.DeleteFile(path).IgnoreError();
        absl::MutexLock lock(install_mu_);
        versions_->DiscardSegmentFile(path);
      }
      absl::MutexLock lock(state_mu_);
      failure_ = status;
      failure_reported_ = false;
      ++finished_flushes_;
    }
  }
}

uint64_t Index::TotalDocsLocked() const {
  uint64_t total = 0;
  for (const VersionSegment& entry : current_->version->segments) {
    total += entry.metadata.num_docs;
  }
  for (const auto& mem : current_->frozen) total += mem->published_docs();
  return total + current_->mem->published_docs();
}

bool Index::ZeroCountLocked(BlobId blob) const {
  const uint32_t id = static_cast<uint32_t>(blob);
  return id >= references_.size() || references_[id].count == 0;
}

void Index::ScanGcLocked() {
  if (gc_suspended_) return;
  const absl::Time start = absl::Now();
  dead_by_segment_.clear();
  scanned_dead_docs_ = 0;
  const absl::Time now = absl::Now();
  for (const VersionSegment& entry : current_->version->segments) {
    uint32_t eligible = 0;
    for (DocId doc = 0; doc < entry.segment->num_docs(); ++doc) {
      const BlobId blob = (*entry.blob_of_doc)[doc];
      if (ZeroCountLocked(blob)) ++scanned_dead_docs_;
      if (DropLocked(blob, now) && Restorable(entry.segment->doc(doc))) {
        ++eligible;
      }
    }
    dead_by_segment_[entry.metadata.file_number] = eligible;
  }
  for (const auto& mem : current_->frozen) {
    for (DocId doc = 0; doc < mem->published_docs(); ++doc) {
      if (ZeroCountLocked(mem->blob_of_doc(doc))) ++scanned_dead_docs_;
    }
  }
  for (DocId doc = 0; doc < current_->mem->published_docs(); ++doc) {
    if (ZeroCountLocked(current_->mem->blob_of_doc(doc))) ++scanned_dead_docs_;
  }
  stats_.gc_scan_time = std::max(stats_.gc_scan_time, absl::Now() - start);
  ++stats_.gc_scans;
  last_gc_scan_ = start;
  UpdateGcDeadlineLocked();
}

IndexStats Index::Stats() const {
  absl::MutexLock lock(state_mu_);
  IndexStats result = stats_;
  result.pending_symbols = active_->pending_symbols();
  result.failed_symbols = active_->failed_symbols();
  for (const std::shared_ptr<Memtable>& mem : pending_) {
    result.pending_symbols += mem->pending_symbols();
    result.failed_symbols += mem->failed_symbols();
  }
  for (const VersionSegment& entry : current_->version->segments) {
    if (!entry.segment->has_symbols()) {
      result.pending_symbols += entry.segment->num_docs();
    } else {
      for (DocId doc = 0; doc < entry.segment->num_docs(); ++doc) {
        result.failed_symbols += entry.segment->doc(doc).symbols_failed;
      }
    }
  }
  result.segment_count = current_->version->segments.size();
  result.dead_docs =
      gc_suspended_ ? 0 : std::min(scanned_dead_docs_, TotalDocsLocked());
  result.live_docs = TotalDocsLocked() - result.dead_docs;
  return result;
}

bool Index::CompactionsDoneLocked() const {
  return !jobs_[0] && !jobs_[1] &&
         (gc_suspended_ || !options_.compaction.enabled ||
          absl::Now() < gc_deadline_);
}

void Index::WaitForCompactions() {
  absl::MutexLock lock(state_mu_);
  state_mu_.Await(absl::Condition(this, &Index::CompactionsDoneLocked));
}

void Index::ScheduleLocked() {
  if (!options_.compaction.enabled || stopping_ || stats_.compaction_disabled) {
    return;
  }
  std::vector<CompactionSegment> candidates;
  for (const VersionSegment& entry : current_->version->segments) {
    bool busy = false;
    for (const auto& job : jobs_) {
      if (job &&
          std::find(job->file_numbers.begin(), job->file_numbers.end(),
                    entry.metadata.file_number) != job->file_numbers.end()) {
        busy = true;
        break;
      }
    }
    if (!busy) {
      const auto it = dead_by_segment_.find(entry.metadata.file_number);
      const uint32_t dead = it == dead_by_segment_.end() ? 0 : it->second;
      candidates.push_back({entry.metadata.file_number,
                            entry.metadata.file_size, entry.metadata.num_docs,
                            dead});
    }
  }
  for (CompactionSelection& selection :
       PickCompactions(candidates, options_.compaction)) {
    const size_t lane =
        selection.lane == CompactionSelection::Lane::kSmall ? 0 : 1;
    if (!jobs_[lane]) jobs_[lane] = std::move(selection);
  }
}

bool Index::CompactionReadyLocked(size_t lane) const {
  return stopping_ ||
         (lane == 0 && !gc_suspended_ && absl::Now() >= gc_deadline_) ||
         (jobs_[lane] && absl::Now() >= retry_after_[lane]);
}

bool Index::SmallCompactionReadyLocked() const {
  return CompactionReadyLocked(0);
}

bool Index::LargeCompactionReadyLocked() const {
  return CompactionReadyLocked(1);
}

void Index::RunCompaction(size_t lane) {
  for (;;) {
    CompactionSelection selection;
    std::shared_ptr<const Version> version;
    {
      absl::MutexLock lock(state_mu_);
      const absl::Condition ready =
          lane == 0 ? absl::Condition(this, &Index::SmallCompactionReadyLocked)
                    : absl::Condition(this, &Index::LargeCompactionReadyLocked);
      // Only a pending grace deadline or a job retry needs a timed wakeup.
      absl::Time wake = lane == 0 ? gc_deadline_ : absl::InfiniteFuture();
      if (jobs_[lane]) wake = std::min(wake, retry_after_[lane]);
      if (wake == absl::InfiniteFuture()) {
        state_mu_.Await(ready);
      } else if (wake > absl::Now()) {
        state_mu_.AwaitWithTimeout(ready, wake - absl::Now());
      }
      if (stopping_) return;
      if (lane == 0 && !gc_suspended_ && absl::Now() >= gc_deadline_) {
        bool due = false;
        const absl::Time now = absl::Now();
        while (!gc_pending_.empty() && gc_pending_.front().deadline <= now) {
          for (BlobId blob : gc_pending_.front().blobs) {
            const uint32_t id = static_cast<uint32_t>(blob);
            if (id < references_.size() && references_[id].count == 0 &&
                now - references_[id].dead_since >= options_.gc_grace_period) {
              due = true;
            }
          }
          gc_pending_.pop_front();
        }
        UpdateGcDeadlineLocked();
        if (due) {
          ScanGcLocked();
          ScheduleLocked();
        }
      }
      if (!CompactionReadyLocked(lane)) continue;
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access): checked just above.
      selection = *jobs_[lane];
      version = current_->version;
    }
    std::vector<const Segment*> inputs;
    std::vector<const VersionSegment*> sources;
    for (const VersionSegment& entry : version->segments) {
      if (std::find(selection.file_numbers.begin(),
                    selection.file_numbers.end(), entry.metadata.file_number) !=
          selection.file_numbers.end()) {
        inputs.push_back(entry.segment.get());
        sources.push_back(&entry);
      }
    }
    absl::Status status;
    absl::StatusOr<SegmentStats> written =
        absl::UnknownError("segment allocation failed");
    std::string path;
    {
      absl::MutexLock lock(install_mu_);
      auto reserved = versions_->NewSegmentFile();
      if (reserved.ok()) {
        path = *reserved;
      } else {
        status = reserved.status();
      }
    }
    std::vector<std::vector<bool>> keep(inputs.size());
    std::vector<std::pair<size_t, DocId>> dropped;
    if (status.ok()) {
      {
        absl::MutexLock lock(state_mu_);
        const absl::Time now = absl::Now();
        for (size_t i = 0; i < inputs.size(); ++i) {
          keep[i].resize(inputs[i]->num_docs(), true);
          for (DocId doc = 0; doc < inputs[i]->num_docs(); ++doc) {
            if (DropLocked((*sources[i]->blob_of_doc)[doc], now) &&
                Restorable(inputs[i]->doc(doc))) {
              keep[i][doc] = false;
              dropped.emplace_back(i, doc);
            }
          }
        }
      }
      // A reopened index may use a different chunk size from its inputs.
      // Validate only the filtered candidates, outside state_mu_, before
      // writing an output that might otherwise be impossible to resurrect.
      std::erase_if(dropped, [&](const auto& candidate) {
        const auto [i, doc] = candidate;
        const absl::string_view content = inputs[i]->doc(doc).content;
        if (content.size() <= options_.memtable.max_chunks ||
            SplitIntoChunks(content, options_.memtable.segment.chunk_bytes)
                    .size() <= options_.memtable.max_chunks) {
          return false;
        }
        keep[i][doc] = true;
        return true;
      });
      written = MergeSegments(
          env_, inputs, [&](size_t i, DocId doc) { return keep[i][doc]; }, path,
          options_.memtable.segment, &cancelled_);
      status = written.status();
    }
    if (status.ok() && options_.compaction_install_reached) {
      absl::MutexLock lock(state_mu_);
      if (!options_.compaction_install_reached->HasBeenNotified()) {
        options_.compaction_install_reached->Notify();
      }
    }
    if (status.ok() && options_.compaction_install_gate) {
      options_.compaction_install_gate->WaitForNotification();
    }
    if (status.ok() && !cancelled_.load(std::memory_order_relaxed)) {
      absl::MutexLock install_lock(install_mu_);
      {
        absl::MutexLock lock(state_mu_);
        if (stats_.compaction_disabled) {
          status = absl::CancelledError("compaction disabled");
        }
      }
      std::vector<bool> restored(dropped.size(), false);
      if (status.ok()) {
        absl::MutexLock lock(state_mu_);
        state_mu_.Await(absl::Condition(this, &Index::IngestIdleLocked));
        for (size_t j = 0; j < dropped.size(); ++j) {
          const auto [i, doc] = dropped[j];
          if (!stopping_ &&
              !DropLocked((*sources[i]->blob_of_doc)[doc], absl::Now())) {
            status = RestoreLocked(inputs[i]->doc(doc));
            if (!status.ok()) break;
            restored[j] = true;
          }
        }
      }
      if (status.ok()) {
        status = versions_->Install(
            VersionEdit{.add = {path}, .remove = selection.file_numbers});
      }
      if (status.ok()) {
        absl::MutexLock lock(state_mu_);
        state_mu_.Await(absl::Condition(this, &Index::IngestIdleLocked));
        // Inputs remain mapped until publication. No reference edit can run
        // between this final check and publishing the installed version.
        for (size_t j = 0; j < dropped.size(); ++j) {
          const auto [i, doc] = dropped[j];
          if (!restored[j] && !stopping_ &&
              !DropLocked((*sources[i]->blob_of_doc)[doc], absl::Now())) {
            CHECK_OK(RestoreLocked(inputs[i]->doc(doc)));
          }
        }
        auto next = std::make_shared<ReadState>(*current_);
        next->version = versions_->current();
        current_ = std::move(next);
        ++stats_.completed_compactions;
        stats_.last_compaction_error = absl::OkStatus();
        retry_delay_[lane] = absl::ZeroDuration();
        retry_after_[lane] = absl::InfinitePast();
        stats_.compaction_bytes += written->file_bytes;
        uint64_t input_bytes = 0;
        for (const VersionSegment* source : sources) {
          input_bytes += source->metadata.file_size;
        }
        if (!dropped.empty() && input_bytes > written->file_bytes) {
          stats_.reclaimed_bytes += input_bytes - written->file_bytes;
        }
        jobs_[lane].reset();
        if (!dropped.empty()) ScanGcLocked();
        ScheduleLocked();
      }
    } else if (status.ok()) {
      status = absl::CancelledError("compaction stopped");
    }
    if (!status.ok()) {
      if (!path.empty()) env_.DeleteFile(path).IgnoreError();
      absl::MutexLock install_lock(install_mu_);
      if (!path.empty()) versions_->DiscardSegmentFile(path);
      const bool reopen_required = versions_->reopen_required();
      absl::MutexLock lock(state_mu_);
      if (stopping_ || stats_.compaction_disabled) return;
      // Ordinary flush errors remain independent of compaction errors.
      if (reopen_required) {
        stats_.compaction_disabled = true;
        jobs_[0].reset();
        jobs_[1].reset();
      }
      ++stats_.compaction_failures;
      stats_.last_compaction_error = status;
      if (!stats_.compaction_disabled) {
        retry_delay_[lane] =
            retry_delay_[lane] == absl::ZeroDuration()
                ? options_.compaction.retry_initial_delay
                : std::min(retry_delay_[lane] * 2,
                           options_.compaction.retry_max_delay);
        stats_.last_compaction_retry_delay = retry_delay_[lane];
        retry_after_[lane] = absl::Now() + retry_delay_[lane];
      }
    }
  }
}

}  // namespace ferret

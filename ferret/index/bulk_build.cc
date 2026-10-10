// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/bulk_build.h"

#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

#include "ferret/base/env.h"
#include "ferret/base/thread_pool.h"
#include "ferret/git/blob_hash.h"
#include "ferret/git/worktree_files.h"
#include "ferret/index/doc_precomputer.h"
#include "ferret/index/internal/code_tokens.h"
#include "ferret/index/internal/ordered_parallel.h"
#include "ferret/index/internal/pack_file.h"
#include "ferret/index/internal/posting_accumulator.h"
#include "ferret/index/prepared_document.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/segment_writer.h"
#include "ferret/postings/ids.h"
#include "ferret/symbols/symbol_extractor.h"
#include "ferret/text/eligibility.h"
#include "ferret/text/language.h"
#include "ferret/text/trigram_extractor.h"

namespace ferret {
namespace {

// What pass 1 learns about one path.
struct ScannedFile {
  enum class Outcome { kIndexed, kSkipped, kUnreadable };

  Outcome outcome = Outcome::kUnreadable;
  SkipReason skip_reason = SkipReason::kNone;
  // Set for kIndexed.
  RankKey rank_key;
  Language language = Language::kUnknown;
  bool long_lines = false;
  int64_t size = 0;
  // Where the content is in the scratch file, and what was computed from it.
  uint64_t pack_offset = 0;
  PrecomputedDocument precomputed;
  int64_t read_ns = 0;
  int64_t prepare_ns = 0;
};

// The process's peak resident size so far, in bytes.
int64_t PeakRssBytes() {
  rusage usage = {};
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#ifdef __APPLE__
  return static_cast<int64_t>(usage.ru_maxrss);  // Bytes.
#else
  return static_cast<int64_t>(usage.ru_maxrss) * 1024;  // Kilobytes.
#endif
}

std::string JoinPath(absl::string_view root, absl::string_view path) {
  return absl::StrCat(root, "/", path);
}

// Reads the file at `path` unless it is larger than `max_bytes`, in which
// case it returns std::nullopt. An error means the file cannot be read.
absl::StatusOr<std::optional<std::string>> ReadIfNotTooLarge(
    Env& env, const std::string& path, int64_t max_bytes) {
  return env.ReadFileIfAtMost(path, max_bytes);
}

constexpr size_t kPackBufferBytes = size_t{4} << 20;

// What the symbol workers extract from one file.
struct SymbolResult {
  bool has_grammar = false;
  bool failed = false;
  std::vector<ExtractedSymbol> symbols;
};

// One file whose symbols are wanted: its content is in the scratch file.
struct SymbolJob {
  size_t index = 0;
  uint64_t offset = 0;
  int64_t size = 0;
  Language language = Language::kUnknown;
};

// Extracts symbols from contents that the scan has copied to the scratch
// file, on its own threads, while the scan, the sort and pass 2 go on. The
// output does not depend on the timing: each file's symbols are a function
// of its bytes, and the caller adds them to the writer in rank order after
// Wait().
class SymbolPipeline {
 public:
  SymbolPipeline(const segment_internal::PackFile& pack, int threads,
                 ThreadQos qos, std::vector<SymbolResult>& results)
      : pack_(pack), results_(results), pool_(threads, qos) {
    start_ = absl::Now();
    {
      absl::MutexLock lock(mu_);
      running_ = threads;
    }
    for (int t = 0; t < threads; ++t) {
      pool_.Schedule([this] { Work(); });
    }
  }

  SymbolPipeline(const SymbolPipeline&) = delete;
  SymbolPipeline& operator=(const SymbolPipeline&) = delete;
  // Drops what is still queued and releases the workers, so that the pool
  // joins them promptly on an early return. After Wait() the queue is empty.
  ~SymbolPipeline() {
    absl::MutexLock lock(mu_);
    batches_.clear();
    closed_ = true;
  }

  // Queues files whose contents are in the scratch file.
  void Push(std::vector<SymbolJob> jobs) ABSL_LOCKS_EXCLUDED(mu_) {
    if (jobs.empty()) return;
    absl::MutexLock lock(mu_);
    batches_.push_back(std::move(jobs));
  }

  // No more files will be pushed. Workers finish what is queued.
  void Close() ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(mu_);
    closed_ = true;
  }

  // Waits for the workers to finish a closed queue. Returns the first error.
  absl::Status Wait() ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(mu_);
    mu_.Await(absl::Condition(this, &SymbolPipeline::AllDone));
    return error_;
  }

  // Valid after Wait().
  absl::Duration wall() const { return end_ - start_; }
  int64_t parse_ns() const { return parse_ns_.load(); }
  int64_t read_ns() const { return read_ns_.load(); }

 private:
  bool AllDone() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return running_ == 0;
  }
  bool HasWork() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return !batches_.empty() || closed_;
  }

  void Work() ABSL_LOCKS_EXCLUDED(mu_) {
    SymbolExtractor extractor;
    std::string content;
    int64_t parse = 0;
    int64_t read = 0;
    while (true) {
      std::vector<SymbolJob> batch;
      {
        absl::MutexLock lock(mu_);
        mu_.Await(absl::Condition(this, &SymbolPipeline::HasWork));
        if (batches_.empty()) break;
        batch = std::move(batches_.front());
        batches_.pop_front();
      }
      for (const SymbolJob& job : batch) {
        const int64_t r0 = absl::GetCurrentTimeNanos();
        const absl::Status status =
            pack_.ReadAt(job.offset, static_cast<size_t>(job.size), content);
        const int64_t r1 = absl::GetCurrentTimeNanos();
        read += r1 - r0;
        if (!status.ok()) {
          absl::MutexLock lock(mu_);
          error_.Update(status);
          continue;
        }
        SymbolResult& result = results_[job.index];
        result.has_grammar = true;
        absl::StatusOr<std::vector<ExtractedSymbol>> extracted =
            extractor.Extract(job.language, content);
        parse += absl::GetCurrentTimeNanos() - r1;
        if (extracted.ok()) {
          result.symbols = std::move(*extracted);
        } else {
          result.failed = true;
        }
      }
    }
    parse_ns_ += parse;
    read_ns_ += read;
    absl::MutexLock lock(mu_);
    if (--running_ == 0) end_ = absl::Now();
  }

  const segment_internal::PackFile& pack_;
  std::vector<SymbolResult>& results_;
  absl::Time start_;
  absl::Time end_;
  std::atomic<int64_t> parse_ns_{0};
  std::atomic<int64_t> read_ns_{0};
  absl::Mutex mu_;
  std::deque<std::vector<SymbolJob>> batches_ ABSL_GUARDED_BY(mu_);
  bool closed_ ABSL_GUARDED_BY(mu_) = false;
  int running_ ABSL_GUARDED_BY(mu_) = 0;
  absl::Status error_ ABSL_GUARDED_BY(mu_);
  // Declared last: its destructor joins the workers first.
  ThreadPool pool_;
};

// One pass 1 thread's state.
struct ScanWorker {
  TrigramExtractor extractor;
  std::unique_ptr<DocumentPrecomputer> precomputer;
  // Contents waiting for the scratch file, and the files they belong to.
  std::string pack_buffer;
  std::vector<std::pair<size_t, uint64_t>> pending;
  int64_t pack_ns = 0;
};

// Shared by the pass 1 threads.
struct ScanShared {
  segment_internal::PackFile* pack = nullptr;
  // Receives the files whose symbols are wanted; null when there are none.
  SymbolPipeline* symbols = nullptr;
  std::atomic<uint32_t> next_chunk{0};
  absl::Mutex mu;
  absl::Status error ABSL_GUARDED_BY(mu);
};

void SetError(ScanShared& shared, const absl::Status& status) {
  absl::MutexLock lock(shared.mu);
  shared.error.Update(status);
}

// Appends the buffered contents to the scratch file and records where each
// went.
void FlushPack(ScanWorker& worker, ScanShared& shared,
               std::vector<ScannedFile>& scanned) {
  if (worker.pending.empty()) return;
  const int64_t start = absl::GetCurrentTimeNanos();
  absl::StatusOr<uint64_t> base = shared.pack->Append(worker.pack_buffer);
  if (base.ok()) {
    std::vector<SymbolJob> jobs;
    for (const auto& [index, offset] : worker.pending) {
      scanned[index].pack_offset = *base + offset;
      if (shared.symbols != nullptr &&
          HasSymbolGrammar(scanned[index].language)) {
        jobs.push_back({.index = index,
                        .offset = *base + offset,
                        .size = scanned[index].size,
                        .language = scanned[index].language});
      }
    }
    if (shared.symbols != nullptr) shared.symbols->Push(std::move(jobs));
  } else {
    SetError(shared, base.status());
  }
  worker.pending.clear();
  worker.pack_buffer.clear();
  worker.pack_ns += absl::GetCurrentTimeNanos() - start;
}

// Reads and ranks one file. With a `worker`, also indexes it under the
// provisional id `index` and copies its content to the scratch file.
void Scan(Env& env, absl::string_view root, absl::string_view path,
          size_t index, const EligibilityOptions& options,
          TrigramExtractor& extractor, ScanWorker* worker, ScanShared& shared,
          std::vector<ScannedFile>& scanned) {
  // The result is built in place: a flush of the scratch buffer sets the
  // pack_offset of every pending file, this one included.
  ScannedFile& file = scanned[index];
  const int64_t t0 = absl::GetCurrentTimeNanos();
  absl::StatusOr<std::optional<std::string>> content =
      ReadIfNotTooLarge(env, JoinPath(root, path), options.max_file_bytes);
  const int64_t t1 = absl::GetCurrentTimeNanos();
  file.read_ns = t1 - t0;
  if (!content.ok()) return;
  if (!content->has_value()) {
    file.outcome = ScannedFile::Outcome::kSkipped;
    file.skip_reason = SkipReason::kTooLarge;
    return;
  }
  const std::string& bytes = **content;
  const PreparedDocument prepared =
      PrepareDocument(path, bytes, options, extractor);
  file.prepare_ns = absl::GetCurrentTimeNanos() - t1;
  if (prepared.skip_reason != SkipReason::kNone) {
    file.outcome = ScannedFile::Outcome::kSkipped;
    file.skip_reason = prepared.skip_reason;
    return;
  }
  file.outcome = ScannedFile::Outcome::kIndexed;
  file.long_lines = prepared.long_lines;
  file.language = prepared.language;
  file.rank_key = prepared.rank_key;
  file.size = static_cast<int64_t>(bytes.size());
  if (worker != nullptr) {
    absl::StatusOr<PrecomputedDocument> precomputed =
        worker->precomputer->Precompute(bytes, file.language,
                                        static_cast<uint32_t>(index));
    if (!precomputed.ok()) {
      SetError(shared, precomputed.status());
      return;
    }
    file.precomputed = std::move(*precomputed);
    worker->pending.emplace_back(index, worker->pack_buffer.size());
    worker->pack_buffer.append(bytes);
    if (worker->pack_buffer.size() >= kPackBufferBytes) {
      FlushPack(*worker, shared, scanned);
    }
  }
}

void Accumulate(SegmentWriterTimings& total, const SegmentWriterTimings& add) {
  total.chunking_ns += add.chunking_ns;
  total.trigrams_ns += add.trigrams_ns;
  total.postings_add_ns += add.postings_add_ns;
  total.code_tokens_ns += add.code_tokens_ns;
  total.content_ns += add.content_ns;
  total.symbols_ns += add.symbols_ns;
  total.tail_sections_ns += add.tail_sections_ns;
  total.symbols_write_ns += add.symbols_write_ns;
  total.code_tokens_write_ns += add.code_tokens_write_ns;
  total.lists_merge_ns += add.lists_merge_ns;
  total.lists_encode_ns += add.lists_encode_ns;
  total.dictionary_ns += add.dictionary_ns;
  total.sync_ns += add.sync_ns;
}

// What a pass 2 worker hands the committer: the contents of a run of
// documents in rank order.
struct ContentBlock {
  std::string bytes;
  // Where each document starts in `bytes`; one more entry ends the last.
  std::vector<size_t> starts;
};

absl::Status WriteSerial(Env& env, absl::string_view root,
                         absl::Span<const std::string> paths,
                         const std::vector<ScannedFile>& scanned,
                         const std::vector<size_t>& order,
                         const BulkBuildOptions& options, SegmentWriter& writer,
                         BulkBuildStats& stats, int64_t& reread_ns,
                         int64_t& verify_ns) {
  for (const size_t i : order) {
    const ScannedFile& file = scanned[i];
    const int64_t r0 = absl::GetCurrentTimeNanos();
    absl::StatusOr<std::optional<std::string>> content = ReadIfNotTooLarge(
        env, JoinPath(root, paths[i]), options.eligibility.max_file_bytes);
    reread_ns += absl::GetCurrentTimeNanos() - r0;
    // A file that changed since pass 1 would break rank order.
    if (!content.ok() || !content->has_value()) {
      ++stats.files_unreadable;
      continue;
    }
    const std::string& bytes = **content;
    const int64_t h0 = absl::GetCurrentTimeNanos();
    const bool same = HashBlob(bytes) == file.rank_key.oid;
    verify_ns += absl::GetCurrentTimeNanos() - h0;
    if (!same) {
      ++stats.files_unreadable;
      continue;
    }
    const SegmentDocument doc = {.hint_path = paths[i],
                                 .content = bytes,
                                 .rank_key = file.rank_key,
                                 .language = file.language,
                                 .long_lines = file.long_lines};
    ABSL_RETURN_IF_ERROR(writer.Add(doc));
    ++stats.files_indexed;
    stats.bytes_indexed += static_cast<int64_t>(bytes.size());
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<BulkBuildStats> BuildSegment(Env& env, absl::string_view root,
                                            absl::Span<const std::string> paths,
                                            absl::string_view segment_path,
                                            const BulkBuildOptions& options) {
  if (options.num_threads < 1) {
    return absl::InvalidArgumentError("num_threads must be at least 1");
  }
  if (options.partition_bytes < 1) {
    return absl::InvalidArgumentError("partition_bytes must be at least 1");
  }
  if (options.window < 0) {
    return absl::InvalidArgumentError("window must not be negative");
  }
  BulkBuildStats stats;
  stats.files_listed = static_cast<int64_t>(paths.size());
  const int threads = options.num_threads;
  // Reading the contents back is limited by the file system, not the CPU, so
  // pass 2 uses twice as many workers to keep requests in flight.
  const int io_threads = threads > 1 ? 2 * threads : 1;
  const int scan_threads =
      options.scan_threads > 0 ? options.scan_threads : threads;
  std::unique_ptr<ThreadPool> pool;
  if (threads > 1 || scan_threads > 1) {
    pool = std::make_unique<ThreadPool>(
        std::max({io_threads, scan_threads, 2}) - 1, options.thread_qos);
  }
  const int io_window = options.window > 0 ? options.window : io_threads + 2;

  // Pass 1: read, rank and index every file in parallel.
  const absl::Time scan_start = absl::Now();
  std::vector<ScannedFile> scanned(paths.size());
  ScanShared shared;
  std::unique_ptr<segment_internal::PackFile> pack;
  if (!options.serial) {
    ABSL_ASSIGN_OR_RETURN(pack, segment_internal::PackFile::Create(
                                    env, absl::StrCat(segment_path, ".pack")));
    shared.pack = pack.get();
  }
  SegmentWriterOptions writer_options = options.writer;
  std::vector<SymbolResult> symbol_results;
  std::unique_ptr<SymbolPipeline> symbol_pipeline;
  if (!options.serial && writer_options.write_symbols) {
    symbol_results.resize(paths.size());
    symbol_pipeline = std::make_unique<SymbolPipeline>(
        *pack, options.symbol_threads > 0 ? options.symbol_threads : threads,
        options.symbol_qos, symbol_results);
    shared.symbols = symbol_pipeline.get();
  }
  writer_options.extract_symbols = options.serial;
  writer_options.finish_pool = pool.get();
  writer_options.finish_threads = threads;
  std::vector<ScanWorker> workers(static_cast<size_t>(scan_threads));
  {
    const auto scan_range = [&](int worker_index, int64_t begin, int64_t end) {
      ScanWorker& worker = workers[static_cast<size_t>(worker_index)];
      if (!options.serial && worker.precomputer == nullptr) {
        worker.precomputer = std::make_unique<DocumentPrecomputer>(
            writer_options, worker.extractor, shared.next_chunk);
      }
      for (int64_t i = begin; i < end; ++i) {
        const auto index = static_cast<size_t>(i);
        Scan(env, root, paths[index], index, options.eligibility,
             worker.extractor, options.serial ? nullptr : &worker, shared,
             scanned);
      }
    };
    if (pool != nullptr) {
      pool->ParallelFor(static_cast<int64_t>(paths.size()), /*grain=*/16,
                        scan_threads, scan_range);
    } else {
      scan_range(0, 0, static_cast<int64_t>(paths.size()));
    }
  }
  int64_t pack_ns = 0;
  for (ScanWorker& worker : workers) {
    if (!options.serial) FlushPack(worker, shared, scanned);
    pack_ns += worker.pack_ns;
  }
  {
    absl::MutexLock lock(shared.mu);
    ABSL_RETURN_IF_ERROR(shared.error);
  }
  if (pack != nullptr) ABSL_RETURN_IF_ERROR(pack->FinishWriting());
  if (symbol_pipeline != nullptr) symbol_pipeline->Close();
  std::vector<size_t> order;
  int64_t read_ns = 0;
  int64_t prepare_ns = 0;
  for (size_t i = 0; i < scanned.size(); ++i) {
    read_ns += scanned[i].read_ns;
    prepare_ns += scanned[i].prepare_ns;
    switch (scanned[i].outcome) {
      case ScannedFile::Outcome::kIndexed:
        order.push_back(i);
        break;
      case ScannedFile::Outcome::kSkipped:
        ++stats.files_skipped[static_cast<size_t>(scanned[i].skip_reason)];
        break;
      case ScannedFile::Outcome::kUnreadable:
        ++stats.files_unreadable;
        break;
    }
  }
  stats.scan_read_thread_time = absl::Nanoseconds(read_ns);
  stats.scan_prepare_thread_time = absl::Nanoseconds(prepare_ns);
  stats.pack_thread_time = absl::Nanoseconds(pack_ns);
  stats.scan_time = absl::Now() - scan_start;
  stats.peak_rss_after_scan = PeakRssBytes();
  const absl::Time sort_start = absl::Now();
  // Paths break ties between copies of one blob, so the order is total.
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return std::tie(scanned[a].rank_key, paths[a]) <
           std::tie(scanned[b].rank_key, paths[b]);
  });
  stats.sort_time = absl::Now() - sort_start;

  // Pass 2: write the files in rank order.
  const absl::Time write_start = absl::Now();
  ABSL_ASSIGN_OR_RETURN(
      const std::unique_ptr<SegmentWriter> writer,
      SegmentWriter::Create(env, segment_path, writer_options));
  int64_t reread_ns = 0;
  int64_t verify_ns = 0;
  SegmentWriterTimings precompute_timings;
  if (options.serial) {
    ABSL_RETURN_IF_ERROR(WriteSerial(env, root, paths, scanned, order, options,
                                     *writer, stats, reread_ns, verify_ns));
  } else {
    // Runs of consecutive documents in rank order, read as one unit.
    std::vector<std::pair<size_t, size_t>> units;
    {
      size_t begin = 0;
      int64_t bytes = 0;
      for (size_t k = 0; k < order.size(); ++k) {
        bytes += scanned[order[k]].size;
        if (bytes >= options.partition_bytes || k + 1 == order.size()) {
          units.emplace_back(begin, k + 1);
          begin = k + 1;
          bytes = 0;
        }
      }
    }
    stats.partitions = static_cast<int64_t>(units.size());
    // Where each provisional chunk and document ended up.
    std::vector<ChunkId> chunk_remap(shared.next_chunk.load());
    std::vector<DocId> doc_remap(paths.size(), 0);
    std::vector<int64_t> read_ns_by_worker(static_cast<size_t>(io_threads), 0);
    const absl::Time assemble_start = absl::Now();
    const auto produce = [&](int worker,
                             int64_t u) -> absl::StatusOr<ContentBlock> {
      ContentBlock block;
      std::string content;
      const int64_t r0 = absl::GetCurrentTimeNanos();
      for (size_t k = units[static_cast<size_t>(u)].first;
           k < units[static_cast<size_t>(u)].second; ++k) {
        const ScannedFile& file = scanned[order[k]];
        block.starts.push_back(block.bytes.size());
        ABSL_RETURN_IF_ERROR(pack->ReadAt(
            file.pack_offset, static_cast<size_t>(file.size), content));
        block.bytes.append(content);
      }
      block.starts.push_back(block.bytes.size());
      read_ns_by_worker[static_cast<size_t>(worker)] +=
          absl::GetCurrentTimeNanos() - r0;
      return block;
    };
    uint32_t next_final_chunk = 0;
    const auto commit = [&](int64_t u, ContentBlock&& block) -> absl::Status {
      const auto [first, last] = units[static_cast<size_t>(u)];
      for (size_t k = first; k < last; ++k) {
        const size_t i = order[k];
        const ScannedFile& file = scanned[i];
        const PrecomputedDocument& pre = file.precomputed;
        for (size_t c = 0; c < pre.chunks.size(); ++c) {
          chunk_remap[pre.first_chunk + c] =
              next_final_chunk + static_cast<uint32_t>(c);
        }
        next_final_chunk += static_cast<uint32_t>(pre.chunks.size());
        doc_remap[i] = static_cast<DocId>(k);
        const absl::string_view content =
            absl::string_view(block.bytes)
                .substr(block.starts[k - first],
                        block.starts[k - first + 1] - block.starts[k - first]);
        const SegmentDocument doc = {.hint_path = paths[i],
                                     .content = content,
                                     .rank_key = file.rank_key,
                                     .language = file.language,
                                     .long_lines = file.long_lines};
        ABSL_RETURN_IF_ERROR(
            writer->AddPrecomputed(doc, pre.num_lines, pre.chunks));
        ++stats.files_indexed;
        stats.bytes_indexed += file.size;
      }
      return absl::OkStatus();
    };
    ABSL_RETURN_IF_ERROR(segment_internal::ParallelOrdered<ContentBlock>(
        pool.get(), io_threads, static_cast<int64_t>(units.size()), io_window,
        produce, commit));
    for (const int64_t ns : read_ns_by_worker) reread_ns += ns;
    stats.assemble_time = absl::Now() - assemble_start;
    std::vector<segment_internal::PostingAccumulator::Run> runs;
    std::vector<segment_internal::CodeTokenMap> token_maps;
    for (ScanWorker& worker : workers) {
      if (worker.precomputer == nullptr) continue;
      Accumulate(precompute_timings, worker.precomputer->timings());
      for (auto& run : worker.precomputer->TakeRuns()) {
        runs.push_back(std::move(run));
      }
      token_maps.push_back(worker.precomputer->TakeCodeTokens());
    }
    writer->SetPostings(std::move(runs), std::move(chunk_remap));
    if (writer_options.write_code_tokens) {
      writer->SetCodeTokenMaps(std::move(token_maps), std::move(doc_remap));
    }

    stats.peak_rss_after_assemble = PeakRssBytes();
    // Symbols: the pipeline has been extracting them since the scan began.
    if (symbol_pipeline != nullptr) {
      const absl::Time wait_start = absl::Now();
      ABSL_RETURN_IF_ERROR(symbol_pipeline->Wait());
      stats.symbols_wait_time = absl::Now() - wait_start;
      stats.symbols_time = symbol_pipeline->wall();
      stats.symbols_thread_time =
          absl::Nanoseconds(symbol_pipeline->parse_ns());
      stats.symbols_read_thread_time =
          absl::Nanoseconds(symbol_pipeline->read_ns());
      for (size_t d = 0; d < order.size(); ++d) {
        SymbolResult& entry = symbol_results[order[d]];
        if (!entry.has_grammar) continue;
        writer->AddSymbolsFor(static_cast<DocId>(d), entry.symbols);
        if (entry.failed) writer->MarkSymbolsFailedFor(static_cast<DocId>(d));
        std::vector<ExtractedSymbol>().swap(entry.symbols);
      }
    }
  }
  stats.reread_thread_time = absl::Nanoseconds(reread_ns);
  stats.verify_hash_thread_time = absl::Nanoseconds(verify_ns);
  const absl::Time finish_start = absl::Now();
  ABSL_ASSIGN_OR_RETURN(stats.segment, writer->Finish());
  stats.finish_time = absl::Now() - finish_start;
  stats.peak_rss_after_finish = PeakRssBytes();
  stats.writer = writer->timings();
  Accumulate(stats.writer, precompute_timings);
  stats.write_time = absl::Now() - write_start;
  return stats;
}

absl::StatusOr<BulkBuildStats> BuildSegmentFromWorktree(
    Env& env, absl::string_view root, absl::string_view segment_path,
    const BulkBuildOptions& options) {
  const absl::Time list_start = absl::Now();
  ABSL_ASSIGN_OR_RETURN(const std::vector<std::string> paths,
                        ListWorktreeFiles(root));
  const absl::Duration list_time = absl::Now() - list_start;
  ABSL_ASSIGN_OR_RETURN(BulkBuildStats stats,
                        BuildSegment(env, root, paths, segment_path, options));
  stats.list_time = list_time;
  return stats;
}

}  // namespace ferret

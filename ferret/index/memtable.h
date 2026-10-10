// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// An append-only, single-writer index of recently ingested blobs. Records and
// content have stable addresses; release/acquire publication makes a captured
// document watermark a lock-free snapshot. Postings are copied under a reader
// lock before query evaluation.

#ifndef FERRET_INDEX_MEMTABLE_H_
#define FERRET_INDEX_MEMTABLE_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

#include "ferret/base/env.h"
#include "ferret/git/oid.h"
#include "ferret/index/blob_registry.h"
#include "ferret/index/segment.h"
#include "ferret/index/segment_writer.h"
#include "ferret/postings/ids.h"
#include "ferret/postings/list_ops.h"
#include "ferret/query/evaluator.h"
#include "ferret/query/trigram_query.h"
#include "ferret/symbols/symbol_extractor.h"
#include "ferret/text/trigram.h"
#include "ferret/text/trigram_extractor.h"

namespace ferret {

struct MemtableOptions {
  // Flush after this many bytes of approximate allocated memory.
  size_t flush_bytes = size_t{8} * 1024 * 1024;
  // Flush after this many documents.
  uint32_t flush_docs = 16 * 1024;
  // Room for one more document after the 16k-document soft threshold.
  uint32_t max_docs = 16 * 1024 + 1;
  // Maximum number of chunks retained in the fixed directory.
  uint32_t max_chunks = 1 << 20;
  // Room for one maximal 1 MiB document after the 8 MiB soft threshold.
  size_t max_content_bytes = size_t{9} * 1024 * 1024;
  // Segment settings shared by ingestion and flushing.
  SegmentWriterOptions segment;
  // Index ingestion uses a pool; standalone tables extract on Add().
  bool async_symbols = false;
};

// Owns the posting copies viewed by its evaluator. Neither this object nor
// its evaluator is shared between threads; it may outlive its memtable.
class MemtableQuery final {
 public:
  MemtableQuery(const MemtableQuery&) = delete;
  MemtableQuery& operator=(const MemtableQuery&) = delete;

  // Requires a nonempty window entirely below the captured chunk count.
  void Evaluate(ChunkRange window, ChunkIdBuffer& out) {
    evaluator_->Evaluate(window, out);
  }
  size_t estimated_size() const { return evaluator_->estimated_size(); }
  bool matches_nothing() const { return evaluator_->matches_nothing(); }

 private:
  friend class Memtable;
  MemtableQuery(const TrigramQuery& query,
                absl::flat_hash_map<Trigram, std::vector<ChunkId>> lists);

  // Declared before evaluator_ so the viewed lists are destroyed last.
  absl::flat_hash_map<Trigram, std::vector<ChunkId>> lists_;
  std::unique_ptr<TrigramQueryEvaluator<absl::Span<const ChunkId>>> evaluator_;
};

// One writer calls Add(); all other methods are safe concurrently with Add().
// The caller keeps the memtable alive while reading records or flushing.
class Memtable final {
 public:
  // Shares the supplied registry with other sources, or creates a private one.
  explicit Memtable(const MemtableOptions& options = {},
                    std::shared_ptr<BlobRegistry> blobs = nullptr);
  Memtable(const Memtable&) = delete;
  Memtable& operator=(const Memtable&) = delete;

  // Appends one blob in arrival order. InvalidArgument for invalid input;
  // ResourceExhausted if any hard limit would be crossed (no bytes published).
  // O(content bytes + extracted trigrams); holds the postings lock once per
  // document, only while appending postings and recording its first Oid.
  absl::StatusOr<DocId> Add(const SegmentDocument& document,
                            bool defer_symbols = true);

  uint32_t published_docs() const {
    return published_docs_.load(std::memory_order_acquire);
  }
  // Requires watermark <= published_docs(). O(1).
  uint32_t num_chunks(uint32_t watermark) const;
  // Requires id < watermark for some watermark captured from published_docs().
  // Returned views remain valid for the lifetime of this memtable. O(1).
  SegmentDoc doc(DocId id) const;
  // Requires id < num_chunks(watermark) for some watermark captured from
  // published_docs(). Returned views remain valid for the lifetime of this
  // memtable. O(1).
  SegmentChunk chunk(ChunkId id) const;

  // Requires id below an acquired published_docs() watermark. The parallel
  // record array is written before release publication; readers take no lock.
  BlobId blob_of_doc(DocId id) const;

  // A published document starts pending until its extraction attaches.
  // The mutex also publishes the extracted symbols to concurrent readers.
  void AttachSymbols(DocId id, std::vector<ExtractedSymbol> symbols,
                     bool failed);
  bool SymbolsPending(DocId id) const;
  void FindSymbols(
      absl::string_view name, bool case_insensitive, uint32_t watermark,
      absl::FunctionRef<void(DocId, const ExtractedSymbol&)> fn) const;
  uint32_t pending_symbols() const;
  uint32_t failed_symbols() const;

  // Copies only postings below watermark; the result owns its lists and can
  // be evaluated without holding a lock. O(query postings copied).
  std::unique_ptr<MemtableQuery> NewEvaluator(const TrigramQuery& query,
                                              uint32_t watermark) const;
  // Returns the first published document with this Oid below watermark. O(1).
  std::optional<DocId> FindDoc(const Oid& oid, uint32_t watermark) const;

  // Checks the configured soft thresholds for bytes or documents.
  bool ShouldFlush() const;
  // Nullopt until a document has been published.
  std::optional<absl::Time> oldest_document_time() const;
  // Approximate allocated content, record sizes, paths, and posting storage.
  size_t memory_bytes() const;

  // Writes kept, distinct Oids below watermark in RankKey order. Uses the
  // earliest-ranked copy of each Oid. Uses the segment options from Add().
  // keep() may run concurrently with Add(), but must itself be safe to call
  // here.
  absl::StatusOr<SegmentStats> WriteSegment(
      Env& env, absl::string_view path, uint32_t watermark,
      absl::FunctionRef<bool(DocId)> keep) const;

 private:
  template <typename T>
  class RecordArray final {
   public:
    explicit RecordArray(uint32_t capacity)
        : blocks_(std::make_unique<std::unique_ptr<T[]>[]>(
              (static_cast<size_t>(capacity) + kBlockSize - 1) / kBlockSize)) {}
    void Set(uint32_t id, T value) {
      std::unique_ptr<T[]>& block = blocks_[id / kBlockSize];
      if (!block) block = std::make_unique<T[]>(kBlockSize);
      block[id % kBlockSize] = std::move(value);
    }
    const T& Get(uint32_t id) const {
      return blocks_[id / kBlockSize][id % kBlockSize];
    }
    static constexpr uint32_t kBlockSize = 4096;

   private:
    // The directory is fixed at construction; each slot is initialized only
    // before its first record is published.
    const std::unique_ptr<std::unique_ptr<T[]>[]> blocks_;
  };

  struct DocRecord {
    SegmentDoc doc;
    RankKey rank_key;
    absl::Time arrival;
    std::unique_ptr<char[]> path;
  };
  struct ChunkRecord {
    SegmentChunk chunk;
  };

  // Writer-only fields are never read by readers. Records are read only below
  // an acquired published watermark.
  const MemtableOptions options_;
  RecordArray<DocRecord> docs_;
  RecordArray<BlobId> blob_of_doc_;
  // Shared with the Index and its VersionSet, or private for standalone tables.
  std::shared_ptr<BlobRegistry> blobs_;
  RecordArray<ChunkRecord> chunks_;
  const std::unique_ptr<std::unique_ptr<char[]>[]> arena_;
  size_t next_block_ = 0;
  size_t used_in_block_ = 0;
  size_t allocated_content_bytes_ = 0;
  uint32_t next_doc_ = 0;
  uint32_t next_chunk_ = 0;
  size_t content_bytes_ = 0;
  size_t path_bytes_ = 0;
  size_t posting_bytes_ = 0;
  TrigramExtractor extractor_;
  std::vector<Trigram> chunk_trigrams_;
  std::vector<std::pair<Trigram, ChunkId>> posting_scratch_;

  mutable absl::Mutex postings_mu_;
  absl::flat_hash_map<Trigram, std::vector<ChunkId>> postings_
      ABSL_GUARDED_BY(postings_mu_);
  absl::flat_hash_map<Oid, DocId> first_docs_ ABSL_GUARDED_BY(postings_mu_);
  mutable absl::Mutex symbols_mu_;
  // Indexed by DocId. Access requires symbols_mu_ after publication.
  std::vector<std::vector<ExtractedSymbol>> symbols_
      ABSL_GUARDED_BY(symbols_mu_);
  std::vector<uint8_t> symbols_state_ ABSL_GUARDED_BY(symbols_mu_);
  // A balanced tree avoids copying all prior symbols when one blob attaches.
  absl::btree_multimap<std::string, std::pair<DocId, uint32_t>> names_
      ABSL_GUARDED_BY(symbols_mu_);
  uint32_t pending_symbols_ ABSL_GUARDED_BY(symbols_mu_) = 0;
  uint32_t failed_symbols_ ABSL_GUARDED_BY(symbols_mu_) = 0;
  bool SymbolsReady() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(symbols_mu_) {
    return pending_symbols_ == 0;
  }
  std::atomic<uint32_t> published_docs_{0};
  std::atomic<size_t> published_memory_bytes_{0};
};

}  // namespace ferret

#endif  // FERRET_INDEX_MEMTABLE_H_

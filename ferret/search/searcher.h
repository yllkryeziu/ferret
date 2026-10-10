// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Text search over segments, frozen memtables, and the active memtable below
// its per-query watermark. The pattern is compiled into a trigram query;
// candidates are filtered by path and verified line by line.
//
// Segment candidates are verified in global RankKey order, so a truncated
// response keeps the best-ranked files. Memtable candidates are always
// verified and do not consume the segment budget. Each Oid is reported once.
// Top-K segment verification stops at a safe score bound.

#ifndef FERRET_SEARCH_SEARCHER_H_
#define FERRET_SEARCH_SEARCHER_H_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"

#include "ferret/base/system_info.h"
#include "ferret/base/thread_pool.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/read_state.h"
#include "ferret/index/segment.h"
#include "ferret/search/search_types.h"
#include "ferret/workspace/path_table.h"
#include "ferret/workspace/view_state.h"

namespace ferret {

// Upper bound on SearchRequest::context_lines.
inline constexpr int kMaxContextLines = 100;

// Shared by concurrent searches. The caller also participates in verification.
// Defaults to the number of performance cores at user-interactive QoS.
class SearchExecutor final {
 public:
  // Creates a shared executor with a positive number of worker threads.
  explicit SearchExecutor(int threads = PerformanceCoreCount())
      : pool_(threads, ThreadQos::kUserInteractive) {}
  SearchExecutor(const SearchExecutor&) = delete;
  SearchExecutor& operator=(const SearchExecutor&) = delete;

  // Holds a query's share of the pool until destruction. The caller counts
  // as a worker; admission is safe to use from concurrent searches.
  class Admission final {
   public:
    explicit Admission(SearchExecutor& executor)
        : executor_(executor), parallelism_(executor_.Enter()) {}
    Admission(const Admission&) = delete;
    Admission& operator=(const Admission&) = delete;
    // Releases the admitted share after all of this query's tasks finish.
    ~Admission() { executor_.Leave(); }

    // Returns the number of admitted workers, including the caller.
    int parallelism() const { return parallelism_; }
    // Returns the pool, which must outlive the admission and its tasks.
    ThreadPool& pool() const { return executor_.pool_; }

   private:
    SearchExecutor& executor_;
    int parallelism_;
  };

 private:
  // Admission changes only a count; no query data is published by it.
  int Enter() {
    const int active = active_.fetch_add(1, std::memory_order_relaxed) + 1;
    return std::max(1, pool_.num_threads() / active);
  }
  void Leave() { active_.fetch_sub(1, std::memory_order_relaxed); }

  ThreadPool pool_;
  // Relaxed ordering suffices: admission shares only a count, not query data.
  std::atomic<int> active_{0};
};

// What blocked (or allowed) a rank stop at one test of the strict score bound:
// at the first document `next` of a new file, with the heap holding K complete
// files whose worst is `worst_tier`/`worst_score`. For tests and
// `ferret_bench search --diagnose_stop`.
struct StopProbe {
  RankTier next_tier = RankTier::kSource;
  uint16_t next_prior = 0;
  RankTier worst_tier = RankTier::kSource;
  int64_t worst_score = 0;
  // Whether the bound passed: nothing at or after `next` can outrank.
  bool stop = false;
  // Whether the query has name-aware bounds; if not, only kMaxBoost applies.
  bool name_aware = false;
  // Whether the heap held K files, and the lines kept so far. When the global
  // line cap is spent, `stop` holds whatever the heap holds.
  bool heap_full = false;
  int64_t kept_lines = 0;
  bool lines_exhausted = false;
  // The ceiling of a document outside the priority set and, for an eligible
  // query, the code-token lists still reaches `worst`.
  bool unlisted_blocks = false;
  // Some segment has no code-token list and its documents keep a coarser
  // ceiling.
  bool untokenized_blocks = false;
  // The first remaining priority document, of a source, outside its token
  // list (or any when there is no key), still reaches `worst`.
  bool priority_blocks = false;
  // ...and the first remaining document in both lists, or any memtable
  // candidate.
  bool priority_in_tokens_blocks = false;
  // The first remaining code-token document still reaches `worst`.
  bool code_token_blocks = false;
  // Priority documents across sources at or after `next`, and how many of
  // them precede the first one whose ceiling cannot reach `worst`.
  int64_t priority_remaining = 0;
  int64_t priority_ahead = 0;
};

// One verified document of a search, reported by SearcherOptions::doc_probe
// when its last chunk has been verified, for diagnosis.
struct DocTrace {
  size_t source = 0;
  uint32_t doc = 0;
  int64_t chunks = 0;
  int64_t bytes = 0;
  // All matching lines of the document, and the index among them of the first
  // whose match is a code identifier (-1: none). Meaningful only when the
  // search keeps every line (exact counts and a large line limit).
  int64_t matching_lines = 0;
  int64_t first_identifier_line = -1;
};

// How the priority scan limits itself to documents that can become results.
enum class PriorityRestriction : uint8_t {
  // Per segment, whichever of the basename scan and the candidate documents
  // is estimated to cost less.
  kAuto,
  // Always scan the basenames. The unrestricted set.
  kNever,
  // Always take the segment's candidate documents. Test knob.
  kAlways,
};

// The priority set a search built, for tests: per source, the RankKeys of
// its priority documents, and of those also in its token list.
struct PriorityProbe {
  std::vector<std::vector<RankKey>> keys;
  std::vector<std::vector<RankKey>> both;
};

struct SearcherOptions {
  // Whether a top-K search stops once the global line cap is spent, since no
  // later document can keep a line. Test and ablation knob.
  bool stop_at_line_cap = true;
  // Called by the commit at each rank-stop test, when set. Slow; for
  // diagnosis only.
  const absl::FunctionRef<void(const StopProbe&)>* stop_probe = nullptr;
  // Called by the commit for each document it verified, in commit order, when
  // set. For diagnosis only; use one thread.
  const absl::FunctionRef<void(const DocTrace&)>* doc_probe = nullptr;
  // Whether a top-K search stops once no later document can outrank the K files
  // it holds. The line-cap stop is separate (stop_at_line_cap). Off, every
  // candidate is verified up to the byte budget, so a response can be
  // incomplete where the default is not. Ablation knob.
  bool early_termination = true;
  // Whether the rank-stop bound separates documents whose basename can match
  // (the name-aware bound). Off, every document has the full ceiling,
  // so stops come later; results do not change. Ablation knob.
  bool name_aware_bound = true;
  // Whether the calling thread accumulates the decode and intersect times of
  // posting-list operations, and the pipeline its scan time, into
  // SearchStats. Costs two clock reads per list operation. Benchmark knob.
  bool stage_timing = false;
  // Whether a top-K search counts the matching lines of a segment document that
  // cannot reach the top K, instead of classifying and keeping them
  // (counting-only verification). Results do not depend on it; test and
  // ablation knob.
  bool count_only_excluded = true;
  // Counting-only verification starts after this many verified bytes, so that
  // small queries never pay for the document tests.
  int64_t count_only_min_bytes = int64_t{256} << 10;
  // A pattern that needs two folded bytes and has no trigram selects its
  // candidates from the posting lists of its bigram's trigrams when they hold
  // at most this share of a segment's chunk count in total.
  // Tuning and test knob: 0 never uses them, and a large value always does.
  double max_bigram_posting_share = 8;
  // How the priority scan finds the documents whose basename can match. The
  // set of priority documents that can become results, and so every response,
  // does not depend on it; test and ablation knob.
  PriorityRestriction priority_restriction = PriorityRestriction::kAuto;
  // Whether an alternation's priority scan searches the names for each
  // branch's required string. Off, it tests every name with the matcher
  // (the original scan). Ablation knob.
  bool priority_alternatives = true;
  // Whether the priority scan may use the executor's workers. Off, one
  // thread scans; the set is the same. Ablation knob.
  bool priority_parallel = true;
  // Documents per task of the basename scan, which an executor spreads over
  // its workers. Positive tuning and test knob.
  int64_t priority_range_docs = 32768;
  // Called once with the priority set when the search builds it, if set. For
  // tests.
  const absl::FunctionRef<void(const PriorityProbe&)>* priority_probe = nullptr;
  // Maximum candidate documents used to estimate counts after a rank stop.
  int count_sample_size = 64;
  // How the sample is drawn; every level draws the same one.
  // 0 is the reference selection, kept for tests and ablation: every stream
  // from its first chunk, a full sort per candidate. 1 keeps a bounded heap.
  // 2 also resumes each segment stream after the committed prefix and skips
  // the rest of a document. 3 also splits a segment's frame over the
  // executor's workers and keeps the Oid set small.
  int count_sample_level = 3;
  // Whether a search of one segment without copies of an Oid skips the Oid sets
  // of deduplication, which could not find a copy. Results do not depend on it;
  // test and ablation knob.
  bool lone_segment_skips_dedup = true;
  // Level 3 splits a frame over workers only when it spans at least this many
  // chunk ids; below it one thread is quicker. Positive tuning and test knob.
  int64_t count_sample_parallel_chunks = int64_t{1} << 16;
  // Whether counting-only verification also runs with a count sample, the
  // classes of the counted documents being estimated from a sample of them.
  // Needs count_only_excluded. Ablation knob.
  bool count_only_with_sample = true;
  // The executor must outlive every search using it. Null verifies inline.
  SearchExecutor* executor = nullptr;
  // Candidate bytes scanned inline before starting parallel verification.
  int64_t parallel_threshold_bytes = int64_t{2} << 20;
  // Minimum chunks and bytes per document-aligned grain. Positive tuning and
  // test knobs; a single document can exceed either minimum.
  int64_t grain_chunks = 64;
  int64_t grain_bytes = int64_t{256} << 10;
  // Ring slots per admitted worker. Positive tuning and test knob.
  int ring_slots_per_worker = 4;
  // Run-ahead, in grains per admitted worker, once the top-K heap is full and
  // a rank stop can come at any commit. At most ring_slots_per_worker matters.
  // Positive tuning and test knob.
  int full_ring_slots_per_worker = 4;
};

// Retains a worktree snapshot and the table that names its paths. The table
// must outlive the search; paths may be interned concurrently.
struct SearchView {
  std::shared_ptr<const ViewState> state;
  const PathTable& paths;
};

// Searches one immutable snapshot of segments and memtables. InvalidArgument
// for an invalid pattern, glob, or limit; DataLoss for corrupt postings.
// Thread-safe.
absl::StatusOr<SearchResponse> Search(const ReadState& state,
                                      const SearchRequest& request);

absl::StatusOr<SearchResponse> Search(const ReadState& state,
                                      const SearchRequest& request,
                                      const SearcherOptions& options);

// Searches only blobs in `view`, naming results by matching view paths.
// Callers capture the view before the ReadState: ingestion publishes documents
// before view edits that reference them. A live blob absent from the captured
// ReadState is simply not found. Thread-safe.
absl::StatusOr<SearchResponse> Search(const ReadState& state,
                                      const SearchView& view,
                                      const SearchRequest& request);

absl::StatusOr<SearchResponse> Search(const ReadState& state,
                                      const SearchView& view,
                                      const SearchRequest& request,
                                      const SearcherOptions& options);

// Searches a single segment with the same filtering, verification, and
// ranking as Search(). Files are named by their hint paths. Thread-safe.
absl::StatusOr<SearchResponse> SearchSegment(const Segment& segment,
                                             const SearchRequest& request);

absl::StatusOr<SearchResponse> SearchSegment(const Segment& segment,
                                             const SearchRequest& request,
                                             const SearcherOptions& options);

}  // namespace ferret

#endif  // FERRET_SEARCH_SEARCHER_H_

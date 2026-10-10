// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/search/searcher.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/crc/crc32c.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "re2/re2.h"
#include "re2/stringpiece.h"

#include "ferret/base/stage_clock.h"
#include "ferret/base/thread_pool.h"
#include "ferret/git/oid.h"
#include "ferret/git/path_glob.h"
#include "ferret/index/blob_registry.h"
#include "ferret/index/memtable.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/read_state.h"
#include "ferret/index/segment.h"
#include "ferret/index/version_set.h"
#include "ferret/postings/ids.h"
#include "ferret/postings/list_ops.h"
#include "ferret/postings/posting_list.h"
#include "ferret/query/evaluator.h"
#include "ferret/query/trigram_query.h"
#include "ferret/regex/compile.h"
#include "ferret/search/kept_lines.h"
#include "ferret/search/ranking.h"
#include "ferret/search/search_types.h"
#include "ferret/search/verifier.h"
#include "ferret/text/language.h"
#include "ferret/text/lexer.h"
#include "ferret/text/line_counter.h"
#include "ferret/text/literal_finder.h"
#include "ferret/text/trigram.h"
#include "ferret/text/trigram_extractor.h"
#include "ferret/workspace/path_table.h"

namespace ferret {
namespace {

// Query windows grow from the first size to the maximum.
constexpr uint64_t kFirstWindowChunks = 8192;
constexpr uint64_t kMaxWindowChunks = 131072;

absl::Status ValidateRequest(const SearchRequest& request) {
  if (request.context_lines < 0 || request.context_lines > kMaxContextLines) {
    return absl::InvalidArgumentError("context_lines must be in [0, 100]");
  }
  if (request.max_lines_per_file < 1) {
    return absl::InvalidArgumentError("max_lines_per_file must be positive");
  }
  if (request.max_lines < 0) {
    return absl::InvalidArgumentError("max_lines must not be negative");
  }
  if (request.max_files < 0) {
    return absl::InvalidArgumentError("max_files must not be negative");
  }
  if (request.max_verify_bytes < 1) {
    return absl::InvalidArgumentError("max_verify_bytes must be positive");
  }
  return absl::OkStatus();
}

// Turns on this thread's StageClock for one search, when asked to, and puts
// it back as it was, whichever way the search returns.
class StageClockScope final {
 public:
  explicit StageClockScope(bool enabled)
      : clock_(ThisThreadStageClock()), saved_(clock_) {
    if (enabled) clock_ = StageClock{.enabled = true};
  }
  StageClockScope(const StageClockScope&) = delete;
  StageClockScope& operator=(const StageClockScope&) = delete;
  ~StageClockScope() { clock_ = saved_; }

  // Records the decode and intersect times so far.
  void Snapshot(SearchStats& stats) const {
    if (!clock_.enabled) return;
    stats.decode_time = absl::Nanoseconds(clock_.decode_ns);
    stats.intersect_time = absl::Nanoseconds(clock_.intersect_ns);
  }

 private:
  StageClock& clock_;
  const StageClock saved_;
};

absl::string_view Basename(absl::string_view path) {
  const size_t slash = path.rfind('/');
  return slash == absl::string_view::npos ? path : path.substr(slash + 1);
}

// A single source of document and chunk records. Exactly one pointer is set.
struct Source {
  const Segment* segment = nullptr;
  const Memtable* mem = nullptr;
  uint32_t watermark = 0;
  const std::vector<BlobId>* blob_of_doc = nullptr;
  // The inverse of blob_of_doc, or null.
  const DocsOfBlob* docs_of_blob = nullptr;

  BlobId blob(DocId id) const {
    return mem ? mem->blob_of_doc(id) : (*blob_of_doc)[id];
  }

  uint32_t num_chunks() const {
    return mem ? mem->num_chunks(watermark) : segment->num_chunks();
  }
  SegmentDoc doc(DocId id) const {
    return mem ? mem->doc(id) : segment->doc(id);
  }
  SegmentChunk chunk(ChunkId id) const {
    return mem ? mem->chunk(id) : segment->chunk(id);
  }
  RankKey rank(ChunkId id) const {
    const SegmentDoc record = doc(chunk(id).doc);
    return RankKey{
        .tier = record.tier, .prior = record.prior, .oid = record.oid};
  }
};

// The trigram query that selects `source`'s candidates. It is the pattern's own
// query, except for a pattern that needs two folded bytes "ab" and has no
// trigram: then a source that indexes short lines answers with the union of the
// posting lists of "abX", "Xab" and "ab\n", a superset of the chunks with a
// matching line, when that union is small enough to pay for itself.
absl::StatusOr<TrigramQuery> SourceQuery(const Source& source,
                                         const CompiledPattern& pattern,
                                         double max_posting_share) {
  const absl::string_view literal = pattern.scan_literal();
  if (pattern.query().op() != TrigramQuery::Op::kAll || literal.size() != 2) {
    return pattern.query();
  }
  std::vector<Trigram> trigrams = TrigramsCoveringBigram(
      static_cast<uint8_t>(literal[0]), static_cast<uint8_t>(literal[1]));
  // A case-sensitive literal with a capital is named exactly by its raw
  // bigram's list, which is far shorter than the folded union.
  const PatternSpec& spec = pattern.spec();
  const auto is_upper = [](char c) { return c >= 'A' && c <= 'Z'; };
  if (spec.literal && spec.case_sensitive && spec.pattern.size() == 2 &&
      (is_upper(spec.pattern[0]) || is_upper(spec.pattern[1])) &&
      spec.pattern[0] != '\n' && spec.pattern[1] != '\n') {
    trigrams = {RawBigramTrigram(static_cast<uint8_t>(spec.pattern[0]),
                                 static_cast<uint8_t>(spec.pattern[1]))};
  }
  if (trigrams.empty()) return pattern.query();
  if (source.mem == nullptr) {
    if (!source.segment->has_short_line_trigrams()) return pattern.query();
    uint64_t postings = 0;
    for (const Trigram trigram : trigrams) {
      ABSL_ASSIGN_OR_RETURN(const PostingListView list,
                            source.segment->FindPostingList(trigram));
      postings += list.size();
    }
    if (static_cast<double>(postings) >
        max_posting_share * source.segment->num_chunks()) {
      return pattern.query();
    }
  }
  return TrigramQuery::AnyOf(trigrams);
}

// Evaluates lazily in bounded windows. Memtable chunks arrive in ingestion
// order, so their candidates are sorted by rank before verification.
class CandidateStream final {
 public:
  static absl::StatusOr<CandidateStream> Create(Source source,
                                                const TrigramQuery& query,
                                                SearchStats& stats) {
    Evaluator evaluator;
    // The dictionary lookups of every trigram of the query happen here.
    const absl::Time lookup_start = absl::Now();
    if (source.mem) {
      evaluator.emplace<std::unique_ptr<MemtableQuery>>(
          source.mem->NewEvaluator(query, source.watermark));
    } else {
      ABSL_ASSIGN_OR_RETURN(auto segment_evaluator,
                            source.segment->NewEvaluator(query));
      evaluator.emplace<TrigramQueryEvaluator<PostingListView>>(
          std::move(segment_evaluator));
    }
    stats.dictionary_time += absl::Now() - lookup_start;
    CandidateStream stream(source, std::move(evaluator));
    stream.query_ = query;
    stream.Init(stats);
    stream.estimated_chunks_ =
        source.mem ? stream.ranked_ids_.size()
                   : std::get<TrigramQueryEvaluator<PostingListView>>(
                         stream.evaluator_)
                         .estimated_size();
    return stream;
  }

  bool Next(SearchStats& stats, bool merge_sources = true) {
    if (matches_nothing_) return false;
    while (position_ == size() && first_ < limit_) {
      const absl::Time start = absl::Now();
      EvaluateWindow();
      stats.evaluate_time += absl::Now() - start;
    }
    if (position_ == size()) return false;
    if (merge_sources) {
      rank_ = source_.mem ? ranked_ids_[position_].first : source_.rank(id());
    }
    return true;
  }
  ChunkId id() const {
    return source_.mem ? ranked_ids_[position_].second
                       : candidates_.ids()[position_];
  }
  const RankKey& rank() const { return rank_; }
  void Pop() { ++position_; }
  // Segment streams only, before the first Next(): the stream yields only
  // chunks below `end`.
  void SetLimit(ChunkId end) { limit_ = std::min<uint64_t>(end, limit_); }
  // Segment streams only. Positions a stream that has not yielded yet at the
  // first candidate chunk at or after `chunk`, without evaluating the chunks
  // before it.
  void SeekTo(ChunkId chunk) {
    candidates_.clear();
    position_ = 0;
    first_ = chunk;
  }
  // Segment streams only, with the stream at a candidate: moves past every
  // chunk before `end`. A document's chunks are contiguous, so this skips the
  // rest of a document whose chunks end before `end`.
  void SkipPast(ChunkId end) {
    if (position_ < size()) {
      // Documents have few candidate chunks: look at the next few before
      // searching the rest of the window.
      const absl::Span<const ChunkId> ids = candidates_.ids();
      constexpr size_t kLinear = 8;
      const size_t linear_end = std::min(ids.size(), position_ + kLinear);
      while (position_ < linear_end && ids[position_] < end) ++position_;
      if (position_ == linear_end && position_ < ids.size()) {
        position_ = std::lower_bound(ids.begin() + position_, ids.end(), end) -
                    ids.begin();
      }
    }
    if (position_ == size() && end > first_) first_ = end;
  }
  const Source& source() const { return source_; }
  // The trigram query that selects this stream's candidates.
  const TrigramQuery& query() const { return query_; }
  // An upper bound on the number of chunks the stream yields in all: the
  // evaluator's estimate for a segment (SIZE_MAX when every chunk is a
  // candidate), and the exact count for a memtable. Thread-safe once the
  // stream is created: it reads only what Create() settled.
  size_t estimated_chunks() const { return estimated_chunks_; }
  // A memtable stream's candidate chunks with their documents' RankKeys, in
  // RankKey order and including those already taken; empty for a segment.
  absl::Span<const std::pair<RankKey, ChunkId>> memtable_candidates() const {
    return ranked_ids_;
  }

 private:
  using Evaluator =
      std::variant<std::monostate, TrigramQueryEvaluator<PostingListView>,
                   std::unique_ptr<MemtableQuery>>;

  CandidateStream(Source source, Evaluator evaluator)
      : source_(source), evaluator_(std::move(evaluator)) {
    matches_nothing_ =
        source_.mem
            ? std::get<std::unique_ptr<MemtableQuery>>(evaluator_)
                  ->matches_nothing()
            : std::get<TrigramQueryEvaluator<PostingListView>>(evaluator_)
                  .matches_nothing();
  }

  void Init(SearchStats& stats) {
    if (matches_nothing_ || !source_.mem) return;
    const absl::Time evaluate_start = absl::Now();
    while (first_ < source_.num_chunks()) EvaluateWindow();
    stats.evaluate_time += absl::Now() - evaluate_start;
    std::sort(ranked_ids_.begin(), ranked_ids_.end());
  }

  size_t size() const {
    return source_.mem ? ranked_ids_.size() : candidates_.size();
  }

  void EvaluateWindow() {
    const uint64_t last =
        std::min<uint64_t>(first_ + window_chunks_, limit_) - 1;
    const ChunkRange range{.first = static_cast<ChunkId>(first_),
                           .last = static_cast<ChunkId>(last)};
    candidates_.clear();
    if (source_.mem) {
      std::get<std::unique_ptr<MemtableQuery>>(evaluator_)
          ->Evaluate(range, candidates_);
      for (ChunkId id : candidates_.ids()) {
        ranked_ids_.emplace_back(source_.rank(id), id);
      }
    } else {
      std::get<TrigramQueryEvaluator<PostingListView>>(evaluator_)
          .Evaluate(range, candidates_);
      position_ = 0;
    }
    first_ = last + 1;
    window_chunks_ = std::min(2 * window_chunks_, kMaxWindowChunks);
  }

  Source source_;
  TrigramQuery query_ = TrigramQuery::All();
  size_t estimated_chunks_ = 0;
  bool matches_nothing_ = false;
  Evaluator evaluator_;
  ChunkIdBuffer candidates_;
  std::vector<std::pair<RankKey, ChunkId>> ranked_ids_;
  RankKey rank_;
  size_t position_ = 0;
  uint64_t first_ = 0;
  // The stream yields no chunk at or past this one.
  uint64_t limit_ = source_.num_chunks();
  uint64_t window_chunks_ = kFirstWindowChunks;
};

// The needles a literal prefilter of names scans for: a plain literal pattern
// scans for its own bytes (a case-sensitive one skips the many case variants
// that only the lowercase literal would hit), otherwise the required literal,
// otherwise the alternation's branches, otherwise nothing.
struct NameNeedles {
  LiteralFinder::Mode mode = LiteralFinder::Mode::kAsciiCaseInsensitive;
  std::vector<std::string> strings;

  NameNeedles(const CompiledPattern& pattern, bool alternatives) {
    const bool exact = pattern.exact_literal() != nullptr;
    mode = exact && pattern.spec().case_sensitive
               ? LiteralFinder::Mode::kExact
               : LiteralFinder::Mode::kAsciiCaseInsensitive;
    if (exact) {
      strings.push_back(pattern.spec().pattern);
    } else if (!pattern.scan_literal().empty()) {
      strings.push_back(std::string(pattern.scan_literal()));
    } else if (alternatives) {
      strings.assign(pattern.scan_alternatives().begin(),
                     pattern.scan_alternatives().end());
    }
  }
};

// Confirms a name with the shared matcher, or with a copy of RE2's own so
// that workers do not contend on one DFA cache.
class NameMatcher final {
 public:
  NameMatcher(const CompiledPattern& pattern, bool shared)
      : pattern_(pattern),
        own_(shared || pattern.exact_literal() != nullptr
                 ? nullptr
                 : pattern.NewMatcher()) {}
  bool operator()(absl::string_view text) const {
    return own_ ? RE2::PartialMatch(text, *own_) : pattern_.MatchesWithin(text);
  }

 private:
  const CompiledPattern& pattern_;
  std::unique_ptr<RE2> own_;
};

// The view blobs that have a path passing the glob whose basename matches the
// pattern: the only blobs whose documents can earn the name boost in a view
// search. The view's packed names are scanned once per query, in runs of
// directory chunks shared among the workers, so the set depends on neither the
// partition nor the parallelism. Immutable after construction; thread-safe.
class NamedBlobs final {
 public:
  struct Options {
    ThreadPool* pool = nullptr;
    int parallelism = 1;
    bool alternatives = true;
  };

  NamedBlobs(const CompiledPattern& pattern, const PathGlob* glob,
             const SearchView& view, const Options& options) {
    const NameNeedles needles(pattern, options.alternatives);
    constexpr size_t kChunksPerTask = 64;
    const size_t chunks = view.state->num_chunks();
    const size_t count = (chunks + kChunksPerTask - 1) / kChunksPerTask;
    struct NameTask {
      std::vector<BlobId> named;
      int64_t tested = 0;
    };
    std::vector<NameTask> results(count);
    const int workers =
        options.pool == nullptr
            ? 1
            : static_cast<int>(std::clamp<size_t>(
                  static_cast<size_t>(std::max(options.parallelism, 1)), 1,
                  std::max<size_t>(count, 1)));
    std::vector<std::unique_ptr<NameMatcher>> matchers(
        static_cast<size_t>(workers));
    auto run = [&](int worker, int64_t begin, int64_t end) {
      std::unique_ptr<NameMatcher>& matches =
          matchers[static_cast<size_t>(worker)];
      if (!matches) {
        matches = std::make_unique<NameMatcher>(pattern, worker == 0);
      }
      for (int64_t task = begin; task < end; ++task) {
        NameTask& out = results[static_cast<size_t>(task)];
        const size_t first = static_cast<size_t>(task) * kChunksPerTask;
        for (size_t i = 0; i < std::max<size_t>(needles.strings.size(), 1);
             ++i) {
          view.state->ForEachNameContaining(
              needles.strings.empty() ? absl::string_view()
                                      : absl::string_view(needles.strings[i]),
              needles.mode, view.paths,
              [&](BlobId blob, PathId path_id, absl::string_view basename) {
                if (!out.named.empty() && out.named.back() == blob) return;
                ++out.tested;
                // Without a glob the packed basename is the whole test, so
                // no path string is built.
                if (glob == nullptr ? (*matches)(basename)
                                    : glob->Matches(view.paths.path(path_id)) &&
                                          (*matches)(basename)) {
                  out.named.push_back(blob);
                }
              },
              first, first + kChunksPerTask);
        }
      }
    };
    if (workers > 1 && count > 1) {
      options.pool->ParallelFor(static_cast<int64_t>(count), 1, workers, run);
    } else {
      run(0, 0, static_cast<int64_t>(count));
    }
    // Every BlobId in the view is below the directory's capacity.
    bits_.assign((chunks * ViewState::kChunkSize + 63) / 64, 0);
    for (const NameTask& task : results) {
      tested_ += task.tested;
      for (const BlobId blob : task.named) {
        const size_t id = static_cast<size_t>(blob);
        uint64_t& word = bits_[id / 64];
        if ((word >> (id % 64) & 1) == 0) {
          word |= uint64_t{1} << (id % 64);
          blobs_.push_back(blob);
        }
      }
    }
  }

  bool Contains(BlobId blob) const {
    const size_t id = static_cast<size_t>(blob);
    return id / 64 < bits_.size() && (bits_[id / 64] >> (id % 64) & 1) != 0;
  }
  int64_t size() const { return std::ssize(blobs_); }
  // Names the matcher confirmed (SearchStats::priority_basenames_tested).
  int64_t tested() const { return tested_; }
  // The blobs, each once, in no particular order.
  const std::vector<BlobId>& blobs() const { return blobs_; }

 private:
  std::vector<uint64_t> bits_;
  std::vector<BlobId> blobs_;
  int64_t tested_ = 0;
};

// The priority documents whose result path passes the glob and whose basename
// matches the pattern, which are the only documents that can earn kNameBoost,
// among documents that can become results: those with a candidate chunk. Every
// match contains one of the pattern's scan strings once ASCII letters are
// lowercased, so a literal scan finds them all: of a segment's packed
// basenames, or of the basenames of a memtable's candidates. A segment whose
// candidate chunks are few is instead restricted to its candidate documents'
// basenames. View paths use the cached snapshot names instead of the segment
// hint path.
//
// With a code-token key, it also tracks the documents that can earn
// kIdentifierBoost: those in the key's list in each segment, and all candidates
// of each memtable. Documents outside those lists and outside the priority set
// earn at most kMaxMatchCountBoost. Thread-compatible after construction; the
// constructor itself fans out over `pool`.
class PriorityDocuments final {
 public:
  struct ScanOptions {
    PriorityRestriction restriction = PriorityRestriction::kAuto;
    int64_t range_docs = 32768;
    // Null, or the pool whose threads may help the constructor's caller scan.
    ThreadPool* pool = nullptr;
    int parallelism = 1;
    // Whether to search for an alternation's branch strings.
    bool alternatives = true;
  };

  // Uses a literal prefilter for unscoped names when available; otherwise
  // tests every basename with the compiled matcher. Requires streams from
  // every source, created as CandidateStream::Create() does. `glob` filters
  // result paths. The scan costs O(min(packed basename bytes, candidate
  // chunks of the segment) + memtable candidate chunks), plus matcher
  // confirmations, shared among up to `parallelism` threads. The set does not
  // depend on `parallelism`.
  PriorityDocuments(absl::Span<const CandidateStream> streams,
                    const CompiledPattern& pattern, const PathGlob* glob,
                    const SearchView* view, const NamedBlobs* named,
                    const std::optional<std::string>& token_key,
                    const ScanOptions& options)
      : pattern_(pattern), glob_(glob), view_(view), named_(named) {
    // A plain literal pattern scans for its own bytes; a case-sensitive one
    // skips the many case variants that only the lowercase literal would hit.
    const bool exact = pattern.exact_literal() != nullptr;
    mode_ = exact && pattern.spec().case_sensitive
                ? LiteralFinder::Mode::kExact
                : LiteralFinder::Mode::kAsciiCaseInsensitive;
    if (exact) {
      needles_.push_back(pattern.spec().pattern);
    } else if (!pattern.scan_literal().empty()) {
      needles_.push_back(pattern.scan_literal());
    } else if (options.alternatives) {
      needles_.assign(pattern.scan_alternatives().begin(),
                      pattern.scan_alternatives().end());
    }
    for (const std::string& needle : needles_) {
      needle_views_.push_back(needle);
      finders_.emplace_back(needle, mode_);
    }
    if (named != nullptr) basenames_tested_ += named->tested();
    std::vector<ScanTask> tasks;
    candidate_docs_.resize(streams.size());
    for (size_t source_index = 0; source_index < streams.size();
         ++source_index) {
      const CandidateStream& stream = streams[source_index];
      const Source& source = stream.source();
      SourceKeys& keys = sources_.emplace_back();
      keys.segment = source.segment;
      if (token_key.has_value() && source.mem == nullptr) {
        // A segment that cannot answer keeps the coarser ceiling for all of
        // its documents.
        if (source.segment->FindCodeTokenDocs(*token_key, keys.token_ids)) {
          keys.has_token_ids = true;
          code_token_documents_ += std::ssize(keys.token_ids);
        } else {
          untokenized_ = true;
        }
      }
      if (source.mem != nullptr) {
        ScanMemtable(stream, token_key.has_value(), keys);
        continue;
      }
      const int64_t step = std::max<int64_t>(options.range_docs, 1);
      if (std::optional<std::vector<DocId>> docs =
              CandidateDocs(stream, options.restriction)) {
        ++restricted_sources_;
        candidate_docs_[source_index] = *std::move(docs);
        // A candidate costs more to confirm than a name to scan, so slices
        // are smaller.
        const int64_t candidates = std::ssize(candidate_docs_[source_index]);
        const int64_t slice = std::max<int64_t>(step / 16, 1);
        for (int64_t begin = 0; begin < candidates; begin += slice) {
          tasks.push_back(
              {.source = source_index,
               .begin = static_cast<DocId>(begin),
               .end = static_cast<DocId>(std::min(candidates, begin + slice)),
               .candidates = true});
        }
        continue;
      }
      // A view with few named blobs finds their documents directly instead of
      // testing every document's blob.
      if (UsesInverse(source)) {
        ScanNamedBlobs(stream, keys, options);
        continue;
      }
      // DocIds follow RankKey order, so the keys are nondecreasing; copies
      // of a blob at equally ranked paths have equal keys. Open() does not
      // check the order, but a corrupt segment only makes results wrong.
      const int64_t docs = source.segment->num_docs();
      for (int64_t begin = 0; begin < docs; begin += step) {
        tasks.push_back(
            {.source = source_index,
             .begin = static_cast<DocId>(begin),
             .end = static_cast<DocId>(std::min(docs, begin + step))});
      }
    }
    std::vector<TaskResult> results(tasks.size());
    RunTasks(streams, tasks, results, options);
    // Tasks of one source are in DocId order, so appending keeps the keys
    // nondecreasing, exactly as one thread scanning in order would.
    for (size_t i = 0; i < tasks.size(); ++i) {
      SourceKeys& keys = sources_[tasks[i].source];
      TaskResult& result = results[i];
      keys.keys.insert(keys.keys.end(), result.keys.begin(), result.keys.end());
      keys.both.insert(keys.both.end(), result.both.begin(), result.both.end());
      basenames_tested_ += result.tested;
      basenames_scanned_ += result.scanned;
    }
    for (const SourceKeys& keys : sources_) size_ += std::ssize(keys.keys);
  }

  PriorityDocuments(const PriorityDocuments&) = delete;
  PriorityDocuments& operator=(const PriorityDocuments&) = delete;

  // The number of priority documents across sources.
  int64_t size() const { return size_; }
  // Basenames confirmed with the matcher, and basenames covered by literal
  // search passes (SearchStats::priority_basenames_tested and _scanned).
  int64_t basenames_tested() const { return basenames_tested_; }
  int64_t basenames_scanned() const { return basenames_scanned_; }
  // The number of segments restricted to their candidate documents.
  int64_t restricted_sources() const { return restricted_sources_; }

  // Copies the set into `probe`, per source.
  void Describe(PriorityProbe& probe) const {
    for (const SourceKeys& source : sources_) {
      probe.keys.push_back(source.keys);
      probe.both.push_back(source.both);
    }
  }
  // The number of documents in code-token lists across sources, plus memtable
  // candidates.
  int64_t code_token_documents() const { return code_token_documents_; }
  // Whether some segment could not supply a token list. Its documents can
  // earn the identifier boost, so the caller must also bound them by
  // kMaxBoostWithoutName from `next`.
  bool untokenized() const { return untokenized_; }

  // Fills the priority-related fields of `probe` for the test at `next`.
  void Explain(const RankKey& next, const FileResult& worst,
               StopProbe& probe) const {
    probe.untokenized_blocks =
        untokenized_ && !CannotOutrank(next, kMaxBoostWithoutName, worst);
    for (const SourceKeys& source : sources_) {
      const int64_t ceiling =
          source.has_token_ids ? kMaxBoostWithoutIdentifier : kMaxBoost;
      probe.priority_blocks |= Remaining(source.keys, next, worst, ceiling);
      probe.priority_in_tokens_blocks |=
          Remaining(source.both, next, worst, kMaxBoost) ||
          Remaining(source.all, next, worst, kMaxBoostWithoutName);
      if (source.has_token_ids) {
        const auto first = std::partition_point(
            source.token_ids.begin(), source.token_ids.end(),
            [&](DocId id) { return KeyOf(source, id) < next; });
        probe.code_token_blocks |=
            first != source.token_ids.end() &&
            !CannotOutrank(KeyOf(source, *first), kMaxBoostWithoutName, worst);
      }
      const auto first =
          std::lower_bound(source.keys.begin(), source.keys.end(), next);
      const auto past = std::partition_point(
          first, source.keys.end(), [&](const RankKey& key) {
            return !CannotOutrank(key, ceiling, worst);
          });
      probe.priority_remaining += source.keys.end() - first;
      probe.priority_ahead += past - first;
    }
  }

  // Whether a priority or token document ranked at or after `next`, in any
  // source, might tie or outrank `worst` with its ceiling. Const and
  // thread-safe: the pipeline's workers call it concurrently with the commit,
  // each with its own `next`. O(sources * log keys) per call.
  //
  // Without a token key a priority document's ceiling is kMaxBoost. With
  // one, a document in a segment's token list earns at most
  // kMaxBoostWithoutName, a priority document outside it at most
  // kMaxBoostWithoutIdentifier, one in both kMaxBoost. A memtable's
  // candidates count as token documents.
  bool MayOutrank(const RankKey& next, const FileResult& worst) const {
    if (untokenized_ && !CannotOutrank(next, kMaxBoostWithoutName, worst)) {
      return true;
    }
    for (const SourceKeys& source : sources_) {
      if (Remaining(
              source.keys, next, worst,
              source.has_token_ids ? kMaxBoostWithoutIdentifier : kMaxBoost) ||
          Remaining(source.both, next, worst, kMaxBoost) ||
          Remaining(source.all, next, worst, kMaxBoostWithoutName)) {
        return true;
      }
      // Token ids follow RankKey order like every DocId; the first one not
      // before `next` has the highest ceiling of the rest.
      if (source.has_token_ids) {
        const auto first = std::partition_point(
            source.token_ids.begin(), source.token_ids.end(),
            [&](DocId id) { return KeyOf(source, id) < next; });
        if (first != source.token_ids.end() &&
            !CannotOutrank(KeyOf(source, *first), kMaxBoostWithoutName,
                           worst)) {
          return true;
        }
      }
    }
    return false;
  }

 private:
  struct SourceKeys;
  // One unit of the scan: a range of a segment's documents, or all of a
  // segment's candidate documents.
  struct ScanTask {
    size_t source = 0;
    DocId begin = 0;
    DocId end = 0;
    bool candidates = false;
  };
  struct TaskResult {
    std::vector<RankKey> keys;
    std::vector<RankKey> both;
    int64_t tested = 0;
    int64_t scanned = 0;
  };
  // Estimated costs, in the time of one literal-search step over a basename
  // (calibrated on Chromium, where a step is about 1.8 ns): a literal pass over
  // a segment's names costs 1 per document; confirming a name, or looking up a
  // view blob, 25; finding the document of a candidate chunk 30; and confirming
  // a candidate document 60.
  static constexpr double kMatchCost = 25;
  static constexpr double kChunkCost = 30;
  // A view scan looks documents up by blob when at most one in this many
  // documents has a named blob; a lookup costs about as much as testing the
  // blob of a few documents, and the lookups are then sorted.
  static constexpr int64_t kInverseRatio = 4;
  // Scans cheaper than this many literal passes are never restricted.
  static constexpr double kMinRestrictedScan = 2;
  // The candidate chunks that pay: finding their documents costs at most this
  // share of a scan. Evaluation stops, and the scan runs, once the chunks
  // found exceed it, which bounds the evaluation wasted on a query with many
  // candidates to the work of finding that many.
  static constexpr double kRestrictShare = 0.1;
  // Chunks per window of the evaluation, so that it can stop early.
  static constexpr uint64_t kRestrictWindow = 1 << 16;

  // Whether a view scan finds a segment's priority documents through the
  // inverse blob table: it has one, and few documents have a named blob.
  bool UsesInverse(const Source& source) const {
    return view_ != nullptr && source.docs_of_blob != nullptr &&
           named_->size() * kInverseRatio <=
               static_cast<int64_t>(source.segment->num_docs());
  }

  // The cost of scanning every name of a segment.
  double ScanCost(const Source& source) const {
    const double docs = source.segment->num_docs();
    // A view scan with the inverse table looks up its named blobs only.
    if (UsesInverse(source)) {
      return static_cast<double>(named_->size()) * kMatchCost;
    }
    return view_ != nullptr || needles_.empty()
               ? docs * kMatchCost
               : docs * static_cast<double>(needles_.size());
  }

  // Returns the sorted documents with a candidate chunk when the scan should
  // be limited to them, and nothing when it should not.
  std::optional<std::vector<DocId>> CandidateDocs(
      const CandidateStream& stream, PriorityRestriction restriction) const {
    if (restriction == PriorityRestriction::kNever) return std::nullopt;
    const Source& source = stream.source();
    const bool always = restriction == PriorityRestriction::kAlways;
    const double scan_cost = ScanCost(source);
    // One literal pass over the names is cheaper than evaluating the query's
    // postings (measured on Chromium: 0.3 to 1.1 ms against 0.8), so only a
    // scan of several literals or of every name can pay.
    if (!always &&
        scan_cost < kMinRestrictedScan * source.segment->num_docs()) {
      return std::nullopt;
    }
    const double max_chunks = always ? std::numeric_limits<double>::infinity()
                                     : kRestrictShare * scan_cost / kChunkCost;
    // A query that selects every chunk finds more than the limit at once.
    if (!always &&
        stream.estimated_chunks() == std::numeric_limits<size_t>::max()) {
      return std::nullopt;
    }
    std::vector<DocId> docs;
    if (stream.estimated_chunks() != 0 && source.num_chunks() != 0) {
      absl::StatusOr<TrigramQueryEvaluator<PostingListView>> evaluator =
          source.segment->NewEvaluator(stream.query());
      // The stream opened the same lists, so this does not happen; if it
      // does, the full scan is always sound.
      if (!evaluator.ok()) return std::nullopt;
      ChunkIdBuffer window;
      uint64_t found = 0;
      for (uint64_t first = 0; first < source.num_chunks();
           first += kRestrictWindow) {
        const uint64_t last =
            std::min<uint64_t>(first + kRestrictWindow, source.num_chunks()) -
            1;
        evaluator->Evaluate({.first = static_cast<ChunkId>(first),
                             .last = static_cast<ChunkId>(last)},
                            window);
        found += window.size();
        if (static_cast<double>(found) > max_chunks) return std::nullopt;
        for (const ChunkId chunk : window.ids()) {
          const DocId doc = source.chunk(chunk).doc;
          if (docs.empty() || docs.back() != doc) docs.push_back(doc);
        }
      }
      std::sort(docs.begin(), docs.end());
      docs.erase(std::unique(docs.begin(), docs.end()), docs.end());
    }
    return docs;
  }

  bool ContainsNeedle(absl::string_view name) const {
    for (const LiteralFinder& finder : finders_) {
      if (finder.Find(name) != absl::string_view::npos) return true;
    }
    return false;
  }

  // Adds document `id` of `source` to `out` if its name can match. A segment
  // document is read through the packed basenames and a lone RankKey; a
  // memtable document is `*doc`, which the caller has read.
  void ConfirmDoc(const Source& source, const SourceKeys& keys, DocId id,
                  const SegmentDoc* doc, const NameMatcher& matches,
                  TaskResult& out) const {
    if (view_ != nullptr) {
      if (!named_->Contains(source.blob(id))) return;
    } else {
      ++out.tested;
      if (source.segment != nullptr) {
        if (!matches(source.segment->basename(id))) return;
        if (glob_ != nullptr &&
            !glob_->Matches(source.segment->doc(id).hint_path)) {
          return;
        }
      } else if (!matches(Basename(doc->hint_path)) ||
                 (glob_ != nullptr && !glob_->Matches(doc->hint_path))) {
        return;
      }
    }
    const RankKey key =
        doc != nullptr
            ? RankKey{.tier = doc->tier, .prior = doc->prior, .oid = doc->oid}
            : source.segment->rank_key(id);
    out.keys.push_back(key);
    if (keys.has_token_ids &&
        std::binary_search(keys.token_ids.begin(), keys.token_ids.end(), id)) {
      out.both.push_back(key);
    }
  }

  // Candidates are in RankKey order, and a document's chunks are adjacent.
  void ScanMemtable(const CandidateStream& stream, bool has_token_key,
                    SourceKeys& keys) {
    const Source& source = stream.source();
    const NameMatcher matches(pattern_, /*shared=*/true);
    TaskResult out;
    std::optional<DocId> last;
    for (const std::pair<RankKey, ChunkId>& candidate :
         stream.memtable_candidates()) {
      const DocId id = source.chunk(candidate.second).doc;
      if (id == last) continue;
      last = id;
      // A memtable keeps no token lists: every candidate may earn the
      // identifier boost.
      if (has_token_key) {
        keys.all.push_back(candidate.first);
        ++code_token_documents_;
      }
      const SegmentDoc doc = source.doc(id);
      if (view_ == nullptr && !finders_.empty()) {
        ++basenames_scanned_;
        if (!ContainsNeedle(Basename(doc.hint_path))) continue;
      }
      ConfirmDoc(source, keys, id, &doc, matches, out);
    }
    keys.keys = std::move(out.keys);
    basenames_tested_ += out.tested;
  }

  // Adds the documents of the view's named blobs in a segment, in DocId order.
  // This is the set that testing every document's blob finds, in
  // O(named blobs + documents / 64) instead of O(documents): the documents
  // are marked in a bitmap, which yields them in order, and their keys are
  // read in slices shared among the workers.
  void ScanNamedBlobs(const CandidateStream& stream, SourceKeys& keys,
                      const ScanOptions& options) {
    const Source& source = stream.source();
    std::vector<uint64_t> marked((source.segment->num_docs() + 63) / 64, 0);
    for (const BlobId blob : named_->blobs()) {
      for (const DocId doc : source.docs_of_blob->Docs(blob)) {
        marked[doc / 64] |= uint64_t{1} << (doc % 64);
      }
    }
    std::vector<DocId> docs;
    for (size_t word = 0; word < marked.size(); ++word) {
      for (uint64_t bits = marked[word]; bits != 0; bits &= bits - 1) {
        docs.push_back(static_cast<DocId>(word * 64 + std::countr_zero(bits)));
      }
    }
    constexpr size_t kDocsPerTask = 4096;
    const size_t count = (docs.size() + kDocsPerTask - 1) / kDocsPerTask;
    std::vector<TaskResult> results(count);
    const NameMatcher matches(pattern_, /*shared=*/true);
    auto run = [&](int, int64_t begin, int64_t end) {
      for (int64_t task = begin; task < end; ++task) {
        const size_t first = static_cast<size_t>(task) * kDocsPerTask;
        const size_t last = std::min(docs.size(), first + kDocsPerTask);
        for (size_t i = first; i < last; ++i) {
          ConfirmDoc(source, keys, docs[i], nullptr, matches,
                     results[static_cast<size_t>(task)]);
        }
      }
    };
    const int workers =
        options.pool == nullptr
            ? 1
            : static_cast<int>(std::clamp<size_t>(
                  static_cast<size_t>(std::max(options.parallelism, 1)), 1,
                  std::max<size_t>(count, 1)));
    if (workers > 1 && count > 1) {
      options.pool->ParallelFor(static_cast<int64_t>(count), 1, workers, run);
    } else {
      run(0, 0, static_cast<int64_t>(count));
    }
    for (const TaskResult& result : results) {
      keys.keys.insert(keys.keys.end(), result.keys.begin(), result.keys.end());
      keys.both.insert(keys.both.end(), result.both.begin(), result.both.end());
    }
  }

  // Scans the names of documents [begin, end) of a segment.
  void ScanRange(const CandidateStream& stream, const SourceKeys& keys,
                 DocId begin, DocId end, const NameMatcher& matches,
                 TaskResult& out) const {
    const Source& source = stream.source();
    // A view names documents through its blobs, so every document is looked
    // up; otherwise the needles select the names to confirm.
    const bool literal = view_ == nullptr && !needle_views_.empty();
    if (literal) {
      out.scanned += std::min(end, source.segment->num_docs()) - begin;
    }
    source.segment->ForEachBasenameContainingAny(
        literal ? absl::MakeConstSpan(needle_views_)
                : absl::Span<const absl::string_view>(),
        mode_, begin, end,
        [&](DocId id) { ConfirmDoc(source, keys, id, nullptr, matches, out); });
  }

  // Confirms the names of `docs`, documents that have a candidate chunk.
  void ScanCandidates(const CandidateStream& stream, const SourceKeys& keys,
                      absl::Span<const DocId> docs, const NameMatcher& matches,
                      TaskResult& out) const {
    const Source& source = stream.source();
    for (const DocId id : docs) {
      if (view_ == nullptr && !finders_.empty()) {
        ++out.scanned;
        if (!ContainsNeedle(source.segment->basename(id))) continue;
      }
      ConfirmDoc(source, keys, id, nullptr, matches, out);
    }
  }

  void RunTasks(absl::Span<const CandidateStream> streams,
                const std::vector<ScanTask>& tasks,
                std::vector<TaskResult>& results, const ScanOptions& options) {
    const int workers =
        options.pool == nullptr
            ? 1
            : static_cast<int>(std::clamp<size_t>(
                  static_cast<size_t>(std::max(options.parallelism, 1)), 1,
                  std::max<size_t>(tasks.size(), 1)));
    std::vector<std::unique_ptr<NameMatcher>> matchers(
        static_cast<size_t>(workers));
    auto run = [&](int worker, int64_t begin, int64_t end) {
      std::unique_ptr<NameMatcher>& matches =
          matchers[static_cast<size_t>(worker)];
      if (!matches) {
        matches = std::make_unique<NameMatcher>(pattern_, worker == 0);
      }
      for (int64_t i = begin; i < end; ++i) {
        const ScanTask& task = tasks[static_cast<size_t>(i)];
        const CandidateStream& stream = streams[task.source];
        const SourceKeys& keys = sources_[task.source];
        TaskResult& out = results[static_cast<size_t>(i)];
        if (task.candidates) {
          ScanCandidates(stream, keys,
                         absl::MakeConstSpan(candidate_docs_[task.source])
                             .subspan(task.begin, task.end - task.begin),
                         *matches, out);
        } else {
          ScanRange(stream, keys, task.begin, task.end, *matches, out);
        }
      }
    };
    const int64_t count = std::ssize(tasks);
    if (workers > 1 && count > 1) {
      options.pool->ParallelFor(count, 1, workers, run);
    } else {
      run(0, 0, count);
    }
  }

  struct SourceKeys {
    // The nondecreasing RankKeys of one source's priority documents.
    std::vector<RankKey> keys;
    // A segment's priority documents that are also in its token list.
    std::vector<RankKey> both;
    // Every candidate of a memtable, when there is a token key.
    std::vector<RankKey> all;
    // A segment's token list, in DocId order.
    const Segment* segment = nullptr;
    bool has_token_ids = false;
    std::vector<DocId> token_ids;
  };

  static RankKey KeyOf(const SourceKeys& source, DocId id) {
    const SegmentDoc doc = source.segment->doc(id);
    return RankKey{.tier = doc.tier, .prior = doc.prior, .oid = doc.oid};
  }

  // Whether the first of `keys` not before `next` might tie or outrank
  // `worst` with `ceiling`. The merge has taken every key strictly before
  // `next`; equal keys remain: `next` itself, or a copy of its Oid in another
  // source. The first remaining key has the highest ceiling in its list.
  static bool Remaining(const std::vector<RankKey>& keys, const RankKey& next,
                        const FileResult& worst, int64_t ceiling) {
    const auto first = std::lower_bound(keys.begin(), keys.end(), next);
    return first != keys.end() && !CannotOutrank(*first, ceiling, worst);
  }

  std::vector<SourceKeys> sources_;
  const CompiledPattern& pattern_;
  const PathGlob* glob_;
  const SearchView* view_;
  LiteralFinder::Mode mode_ = LiteralFinder::Mode::kAsciiCaseInsensitive;
  // The strings every name that can match contains: one, or the alternatives
  // of an alternation, or none.
  std::vector<std::string> needles_;
  std::vector<absl::string_view> needle_views_;
  std::vector<LiteralFinder> finders_;
  // The view's named blobs, when scoped to a view.
  const NamedBlobs* named_;
  // Per source, the documents with a candidate chunk of a restricted segment.
  std::vector<std::vector<DocId>> candidate_docs_;
  int64_t restricted_sources_ = 0;
  int64_t size_ = 0;
  int64_t basenames_tested_ = 0;
  int64_t basenames_scanned_ = 0;
  int64_t code_token_documents_ = 0;
  bool untokenized_ = false;
};

// The committed top-K heap's worst file, once the heap holds K files, for the
// pipeline's workers. Only the commit publishes. The worst file only improves
// (a replacement ranks before the old worst, and the heap never shrinks), so
// the published (tier, score) pair only decreases in tier or increases in
// score, and a worker's stale read is never better than the commit's own. The
// pair packs into one word, so no seqlock is needed. Thread-safe.
class PublishedWorst final {
 public:
  // Publishes `worst`, the heap's current worst file. Commit only.
  void Publish(const FileResult& worst) {
    word_.store(Pack(worst.tier, worst.score), std::memory_order_release);
  }
  // Returns the packed pair; zero when nothing is published.
  uint64_t Load() const { return word_.load(std::memory_order_acquire); }
  static bool Valid(uint64_t packed) { return packed != 0; }
  // A file carrying only the tier and score, which is all that the bound reads.
  static FileResult Unpack(uint64_t packed) {
    FileResult worst;
    worst.tier = static_cast<RankTier>((packed >> kScoreBits) & 0xff);
    worst.score =
        static_cast<int64_t>(packed & ((uint64_t{1} << kScoreBits) - 1));
    return worst;
  }

 private:
  // Scores are a uint16 prior plus a boost below 2^15 * 3: they fit easily.
  static constexpr int kScoreBits = 48;
  static uint64_t Pack(RankTier tier, int64_t score) {
    return (uint64_t{1} << 63) |
           (uint64_t{static_cast<uint8_t>(tier)} << kScoreBits) |
           static_cast<uint64_t>(score);
  }

  std::atomic<uint64_t> word_{0};
};

// Verifies one source of candidate chunks and builds the response's files
// and counts. A document's chunks are contiguous in every source.
// Resolves a scoped query's view paths on the calling thread, which both
// produces candidates and commits them. Without a glob every live blob is
// admitted; with one, the decision is memoized per blob as the matching
// PathIds. Path strings are built only for files that have matches.
class ViewPaths final {
 public:
  ViewPaths(const SearchView& view, const PathGlob* glob)
      : view_(view), glob_(glob) {}

  // Requires a blob live in the view. True if a view path passes the glob.
  bool Admits(BlobId blob) { return glob_ == nullptr || AdmitsGlob(blob); }

  // The number of Paths(blob), without building the strings.
  int64_t PathCount(BlobId blob) const {
    return glob_ == nullptr ? static_cast<int64_t>(view_.state->PathCount(blob))
                            : std::ssize(matching_.at(blob));
  }

  // Returns an admitted blob's view paths that pass the glob, in PathId order.
  std::vector<std::string> Paths(BlobId blob) const {
    std::vector<std::string> paths;
    const std::vector<PathId> all =
        glob_ == nullptr ? view_.state->PathsFor(blob) : std::vector<PathId>();
    const std::vector<PathId>& ids =
        glob_ == nullptr ? all : matching_.at(blob);
    paths.reserve(ids.size());
    for (PathId id : ids) paths.emplace_back(view_.paths.path(id));
    return paths;
  }

 private:
  // Memoizes the blob's view paths that pass the glob.
  bool AdmitsGlob(BlobId blob) {
    auto [it, inserted] = matching_.try_emplace(blob);
    if (inserted) {
      for (PathId id : view_.state->PathsFor(blob)) {
        if (glob_->Matches(view_.paths.path(id))) it->second.push_back(id);
      }
    }
    return !it->second.empty();
  }

  const SearchView& view_;
  const PathGlob* glob_;
  absl::flat_hash_map<BlobId, std::vector<PathId>> matching_;
};

// Counts every non-overlapping occurrence on a matching line. The verifier
// reports only the first occurrence because result counts are per line.
void CountMatchClasses(const CompiledPattern& pattern,
                       absl::string_view content, const LineMatch& match,
                       MatchClassifier& classifier,
                       std::array<int64_t, 4>& classes) {
  ++classes[static_cast<size_t>(
      classifier.Classify(match.match_begin, match.match_end))];
  size_t offset = std::max(match.match_end, match.match_begin + 1);
  while (offset < match.line_end) {
    re2::StringPiece found;
    if (!pattern.matcher().Match(content, offset, match.line_end,
                                 RE2::UNANCHORED, &found, 1)) {
      break;
    }
    const size_t begin = static_cast<size_t>(found.data() - content.data());
    const size_t end = begin + found.size();
    ++classes[static_cast<size_t>(classifier.Classify(begin, end))];
    offset = std::max(end, begin + 1);
  }
}

class MatchCollector final {
 public:
  MatchCollector(const Source& source, const SearchRequest& request,
                 const CompiledPattern& pattern, SearchResponse& response,
                 int64_t& kept_lines, const ViewPaths* view_paths)
      : source_(source),
        request_(request),
        pattern_(pattern),
        verifier_(pattern),
        response_(response),
        kept_lines_(kept_lines),
        view_paths_(view_paths) {}

  // Whether `chunk` belongs to a document other than the previous chunk's.
  bool StartsNewFile(const SegmentChunk& chunk) const {
    return !doc_.has_value() || chunk.doc != *doc_;
  }

  // `count_only` applies from a document's first chunk to its last: the
  // collector then counts the document's matching lines and charges the line
  // limits exactly as a full verification would, but classifies and keeps
  // nothing and builds no file.
  void Verify(const SegmentChunk& chunk, bool count_only = false) {
    if (StartsNewFile(chunk)) {
      FinishFile();
      doc_ = chunk.doc;
      language_ = source_.doc(chunk.doc).language;
      count_only_ = count_only;
    }
    if (count_only_) {
      int64_t lines = 0;
      verifier_.ForEachMatchingLine(chunk.content, [&](const LineMatch&) {
        ++lines;
        return true;
      });
      ChargeCountedLines(lines);
      return;
    }
    MatchClassifier classifier(language_, chunk.content, chunk.entry_state);
    int64_t line_number = chunk.first_line;
    size_t counted_through = 0;
    // The verifier reports lines in increasing offset order for every scan
    // path.
    verifier_.ForEachMatchingLine(chunk.content, [&](const LineMatch& match) {
      line_number += CountNewlines(chunk.content.substr(
          counted_through, match.line_begin - counted_through));
      counted_through = match.line_begin;
      ++file_matching_lines_;
      const bool keep = std::ssize(kept_) < request_.max_lines_per_file &&
                        kept_lines_ < request_.max_lines;
      const MatchClass match_class =
          keep ? classifier.Classify(match.match_begin, match.match_end)
               : MatchClass::kCodeIdentifier;
      if (request_.count_sample) {
        CountMatchClasses(pattern_, chunk.content, match, classifier,
                          file_classes_);
      }
      if (std::ssize(kept_) < request_.max_lines_per_file &&
          kept_lines_ < request_.max_lines) {
        ++kept_lines_;
        kept_.push_back(
            KeptLine{.number = line_number,
                     .begin = chunk.offset_in_doc + match.line_begin,
                     .end = chunk.offset_in_doc + match.line_end,
                     .match_begin = match.match_begin - match.line_begin,
                     .match_end = match.match_end - match.line_begin,
                     .match_class = match_class});
      }
      return true;
    });
  }

  // Commits a worker's count and its per-file bounded prefix of matches.
  void VerifyMatches(const SegmentChunk& chunk, int64_t matching_lines,
                     absl::Span<const KeptLine> matches,
                     const std::array<int64_t, 4>& classes,
                     bool count_only = false) {
    if (StartsNewFile(chunk)) {
      FinishFile();
      doc_ = chunk.doc;
      language_ = source_.doc(chunk.doc).language;
      count_only_ = count_only;
    }
    if (count_only_) {
      ChargeCountedLines(matching_lines);
      return;
    }
    file_matching_lines_ += matching_lines;
    for (size_t i = 0; i < classes.size(); ++i) file_classes_[i] += classes[i];
    for (const KeptLine& match : matches) {
      if (std::ssize(kept_) >= request_.max_lines_per_file ||
          kept_lines_ >= request_.max_lines) {
        break;
      }
      ++kept_lines_;
      kept_.push_back(match);
    }
  }

  // Completes the current file, if any. Call after the last Verify().
  void FinishFile() {
    if (!doc_.has_value()) return;
    const SegmentDoc doc = source_.doc(*doc_);
    // The path strings are built only for a file that is kept; a counted
    // file needs only how many paths it has.
    const bool keeps_file = !count_only_ && !kept_.empty();
    std::vector<std::string> paths;
    int64_t num_paths = 0;
    if (file_matching_lines_ > 0) {
      if (view_paths_ != nullptr) {
        if (keeps_file) {
          paths = view_paths_->Paths(source_.blob(*doc_));
          num_paths = std::ssize(paths);
        } else {
          num_paths = view_paths_->PathCount(source_.blob(*doc_));
        }
      } else {
        num_paths = 1;
        if (keeps_file) paths.emplace_back(doc.hint_path);
      }
    }
    for (size_t i = 0; i < file_classes_.size(); ++i) {
      response_.match_classes[i] +=
          static_cast<double>(file_classes_[i] * num_paths);
    }
    response_.num_matching_files += num_paths;
    response_.num_matching_lines += file_matching_lines_ * num_paths;
    if (keeps_file) {
      const std::vector<ResultLine> lines = WithContext(
          doc.content, doc.num_lines, kept_, request_.context_lines);
      for (const std::string& path : paths) {
        response_.files.push_back(
            FileResult{.path = path,
                       .oid = doc.oid,
                       .tier = doc.tier,
                       .prior = doc.prior,
                       .name_matches = pattern_.MatchesWithin(Basename(path)),
                       .num_matching_lines = file_matching_lines_,
                       .lines = lines});
      }
    }
    doc_.reset();
    file_matching_lines_ = 0;
    file_classes_ = {};
    kept_.clear();
    count_only_ = false;
    counted_kept_ = 0;
  }

 private:
  // Counts `lines` more matching lines of a counting-only document, and keeps
  // as many as the per-file and global limits would have kept.
  void ChargeCountedLines(int64_t lines) {
    file_matching_lines_ += lines;
    const int64_t kept =
        std::min({lines, request_.max_lines_per_file - counted_kept_,
                  std::max<int64_t>(request_.max_lines - kept_lines_, 0)});
    if (kept > 0) {
      counted_kept_ += kept;
      kept_lines_ += kept;
    }
  }

  const Source& source_;
  const SearchRequest& request_;
  const CompiledPattern& pattern_;
  LineVerifier verifier_;
  SearchResponse& response_;
  // Matching lines kept so far, across files.
  int64_t& kept_lines_;
  const ViewPaths* view_paths_;

  // The current file.
  std::optional<DocId> doc_;
  Language language_ = Language::kUnknown;
  int64_t file_matching_lines_ = 0;
  std::array<int64_t, 4> file_classes_ = {};
  std::vector<KeptLine> kept_;
  // Whether the current file is counting-only, and the lines it has charged.
  bool count_only_ = false;
  int64_t counted_kept_ = 0;
};

// Whether a document cannot reach the top K, from facts that need no
// verification. Above its prior a document earns at most the match-count boost,
// plus the identifier boost unless the query has a code-token key and the
// document's segment lists the key's documents without this one, plus the name
// boost unless a view-free search finds no match in the document's basename.
// The cheap tests come first: most documents are decided by their prior alone.
// Thread-safe.
class ExcludedDocuments final {
 public:
  ExcludedDocuments(absl::Span<const CandidateStream> streams,
                    const CompiledPattern& pattern,
                    const std::optional<std::string>& token_key, bool view,
                    absl::FunctionRef<const NamedBlobs&()> named,
                    bool name_aware)
      : streams_(streams),
        pattern_(pattern),
        token_key_(token_key),
        view_(view),
        named_(named),
        name_aware_(name_aware) {
    tokens_.reserve(streams.size());
    for (size_t i = 0; i < streams.size(); ++i) {
      tokens_.push_back(std::make_unique<Tokens>());
    }
  }

  // Whether `doc` of source `source_index`, ranked `key`, ranks strictly after
  // `worst` whatever lines it keeps.
  bool Excludes(size_t source_index, DocId doc, const RankKey& key,
                const FileResult& worst) const {
    if (!CannotOutrank(key, kMaxMatchCountBoost, worst)) return false;
    if (CannotOutrank(key, kMaxBoost, worst)) return true;
    const int64_t without_name =
        kMaxMatchCountBoost +
        (IdentifierPossible(source_index, doc) ? kIdentifierBoost : 0);
    if (!CannotOutrank(key, without_name, worst)) return false;
    if (CannotOutrank(key, without_name + kNameBoost, worst)) return true;
    return !NamePossible(source_index, doc);
  }

 private:
  struct Tokens {
    absl::once_flag once;
    bool listed = false;
    std::vector<DocId> ids;
  };

  bool IdentifierPossible(size_t source_index, DocId doc) const {
    const Source& source = streams_[source_index].source();
    if (!token_key_.has_value() || source.mem != nullptr) return true;
    Tokens& tokens = *tokens_[source_index];
    absl::call_once(tokens.once, [&] {
      tokens.listed =
          source.segment->FindCodeTokenDocs(*token_key_, tokens.ids);
    });
    return !tokens.listed ||
           std::binary_search(tokens.ids.begin(), tokens.ids.end(), doc);
  }

  // Whether the document can have a result path whose basename matches. A
  // view search decides by the blob's view paths (NamedBlobs), when the
  // caller supplied them, and otherwise assumes it can.
  bool NamePossible(size_t source_index, DocId doc) const {
    const Source& source = streams_[source_index].source();
    if (view_) return !name_aware_ || named_().Contains(source.blob(doc));
    return pattern_.MatchesWithin(Basename(source.doc(doc).hint_path));
  }

  absl::Span<const CandidateStream> streams_;
  const CompiledPattern& pattern_;
  const std::optional<std::string>& token_key_;
  const bool view_;
  const absl::FunctionRef<const NamedBlobs&()> named_;
  const bool name_aware_;
  std::vector<std::unique_ptr<Tokens>> tokens_;
};

}  // namespace

namespace {

struct Candidate {
  size_t source_index;
  SegmentChunk chunk;
  std::vector<KeptLine> matches;
  int64_t matching_lines = 0;
  std::array<int64_t, 4> match_classes = {};
  int64_t view_filtered_before = 0;
  int64_t view_filtered_segments_before = 0;
  // Set by a worker on a segment document's first chunk. `observed_worst` is
  // the published worst the worker read (zero: none); `scanned` says whether
  // the worker verified the document rather than skipping it. Commit uses
  // them only to count violations of the skip rule.
  uint64_t observed_worst = 0;
  bool scanned = false;
  // Set by a worker when it counted the document's lines without keeping
  // them, because the published worst excluded the document from the top K.
  bool count_only = false;
};

// The caller alone produces and commits grains in rank order, and commit is
// the only writer of the response. Workers only scan claimed slots; release /
// acquire publishes candidates, and the mutex hands completed scans to commit.
// A slot is reused only after commit clears it. Helpers capture the heap-owned
// window: a late task checks stopped under its mutex before touching this
// object (or any caller stack), and active helpers finish before Run returns.
template <typename Next, typename Commit, typename Stop, typename CountOnly>
class VerificationPipeline final {
 private:
  struct Grain {
    std::vector<Candidate> candidates;
    bool done = false;  // Protected by Window::mu.
  };
  struct Window {
    explicit Window(size_t capacity) : slots(capacity) {}

    // A worker can claim only published grains. Release/acquire publishes all
    // candidate data; the cursor hands out each grain exactly once.
    bool Claim(size_t& index) {
      size_t current = next.load(std::memory_order_relaxed);
      while (current < published.load(std::memory_order_acquire)) {
        if (next.compare_exchange_weak(current, current + 1,
                                       std::memory_order_relaxed)) {
          index = current;
          return true;
        }
      }
      return false;
    }
    struct PublicationWaiter {
      const Window& window;
      size_t epoch;
      bool Ready() const { return window.stopped || window.epoch != epoch; }
    };
    bool HelpersDone() const { return helpers == 0; }

    std::vector<Grain> slots;
    std::atomic<bool> discard_segments{false};
    std::atomic<size_t> next{0};
    std::atomic<size_t> published{0};
    absl::Mutex mu;
    size_t epoch = 0;      // Protected by mu.
    bool stopped = false;  // Protected by mu.
    int helpers = 0;       // Protected by mu.
  };

 public:
  // `stop(next, worst)` is the commit's own rank-stop bound: it must be
  // thread-safe and monotone in `worst` (a better worst never turns a true
  // result false). `published` carries the committed heap's worst file.
  VerificationPipeline(Next& next, Commit& commit, Stop& stop,
                       CountOnly& count_only, const PublishedWorst& published,
                       absl::Span<const CandidateStream> streams,
                       const CompiledPattern& pattern,
                       const SearchRequest& request, SearchResponse& response,
                       const SearcherOptions& options,
                       SearchExecutor::Admission& admission)
      : next_(next),
        commit_(commit),
        stop_(stop),
        count_only_(count_only),
        published_(published),
        streams_(streams),
        pattern_(pattern),
        request_(request),
        response_(response),
        options_(options),
        admission_(admission) {}

  // Segment documents workers skipped under the published worst. Valid after
  // Run() returns; varies with timing, so it is only a statistic.
  int64_t skipped_documents() const {
    return skipped_documents_.load(std::memory_order_relaxed);
  }
  // Nanoseconds all threads spent scanning grains; zero unless
  // SearcherOptions::stage_timing. Valid after Run() returns.
  int64_t scan_ns() const { return scan_ns_.load(std::memory_order_relaxed); }

  void Run(std::optional<Candidate> pending) {
    pending_ = std::move(pending);
    window_ = std::make_shared<Window>(
        static_cast<size_t>(options_.ring_slots_per_worker) *
        admission_.parallelism());
    window_->discard_segments.store(
        !response_.complete || response_.terminated_early,
        std::memory_order_relaxed);
    LineVerifier caller_verifier(pattern_);
    for (int i = 1; i < admission_.parallelism(); ++i) {
      admission_.pool().Schedule([this, lifetime = window_] {
        {
          absl::MutexLock lock(lifetime->mu);
          if (lifetime->stopped) return;
          ++lifetime->helpers;
        }
        LineVerifier verifier(pattern_);
        while (true) {
          if (Work(verifier)) continue;
          absl::MutexLock lock(lifetime->mu);
          if (lifetime->stopped) {
            --lifetime->helpers;
            return;
          }
          if (lifetime->next.load(std::memory_order_relaxed) <
              lifetime->published.load(std::memory_order_acquire)) {
            continue;
          }
          typename Window::PublicationWaiter waiter{*lifetime, lifetime->epoch};
          lifetime->mu.Await(
              absl::Condition(&waiter, &Window::PublicationWaiter::Ready));
          if (lifetime->stopped) {
            --lifetime->helpers;
            return;
          }
        }
      });
    }
    size_t produced = 0;
    size_t committed = 0;
    bool exhausted = false;
    while (true) {
      // Fill free slots while workers scan older grains.
      // Once the heap is full a rank stop can come at any grain, and every
      // grain past it is wasted: run ahead by fewer grains.
      const size_t limit =
          PublishedWorst::Valid(published_.Load())
              ? std::min(
                    window_->slots.size(),
                    static_cast<size_t>(options_.full_ring_slots_per_worker) *
                        static_cast<size_t>(admission_.parallelism()))
              : window_->slots.size();
      while (produced - committed < limit) {
        if (!pending_ && !exhausted) {
          pending_ = next_();
          exhausted = !pending_.has_value();
        }
        if (!pending_) break;
        Grain& grain = window_->slots[produced % window_->slots.size()];
        NextGrain(grain);
        window_->published.store(++produced, std::memory_order_release);
        {
          absl::MutexLock lock(window_->mu);
          ++window_->epoch;
        }
      }
      if (committed == produced) {
        if (exhausted) break;
        continue;
      }
      Grain& grain = window_->slots[committed % window_->slots.size()];
      while (true) {
        {
          absl::MutexLock lock(window_->mu);
          if (grain.done) break;
        }
        if (!Work(caller_verifier)) {
          absl::MutexLock lock(window_->mu);
          window_->mu.Await(absl::Condition(&grain.done));
          break;
        }
      }
      for (Candidate& candidate : grain.candidates) {
        if (response_.terminated_early &&
            !streams_[candidate.source_index].source().mem) {
          continue;
        }
        commit_(candidate, true);
        if (!response_.complete || response_.terminated_early) {
          window_->discard_segments.store(true, std::memory_order_relaxed);
        }
      }
      ++response_.stats.parallel_grains;
      grain.candidates.clear();
      {
        absl::MutexLock lock(window_->mu);
        grain.done = false;
      }
      ++committed;
      // A rank stop discards speculative segment candidates from the merge.
      // Remaining memtable candidates still run.
      if (response_.terminated_early && pending_ &&
          !streams_[pending_->source_index].source().mem) {
        pending_.reset();
      }
    }
    absl::MutexLock lock(window_->mu);
    window_->stopped = true;
    window_->mu.Await(absl::Condition(window_.get(), &Window::HelpersDone));
  }

 private:
  // Grains contain whole documents, so a match prefix can be capped without
  // knowing what earlier grains will commit.
  void ScanGrain(Grain& grain, LineVerifier& verifier) {
    size_t source_index = streams_.size();
    DocId doc = 0;
    int64_t kept = 0;
    bool skipping = false;
    bool counting = false;
    bool first_chunk = false;
    for (Candidate& candidate : grain.candidates) {
      const SegmentChunk& chunk = candidate.chunk;
      const bool segment = !streams_[candidate.source_index].source().mem;
      if (segment &&
          window_->discard_segments.load(std::memory_order_relaxed)) {
        continue;
      }
      if (source_index != candidate.source_index || doc != chunk.doc) {
        source_index = candidate.source_index;
        doc = chunk.doc;
        kept = 0;
        skipping = false;
        counting = false;
        first_chunk = false;
        if (segment) {
          // The commit's stop test, on a worst that is never better than the
          // commit's own by the time it reaches this document: if it stops
          // here, the commit stops at this document (or the document is
          // ineligible or the budget already cut segments; either way it is
          // not verified), so its lines are never read.
          const SegmentDoc segment_doc =
              streams_[source_index].source().doc(doc);
          candidate.observed_worst = published_.Load();
          skipping = PublishedWorst::Valid(candidate.observed_worst) &&
                     stop_(RankKey{.tier = segment_doc.tier,
                                   .prior = segment_doc.prior,
                                   .oid = segment_doc.oid},
                           PublishedWorst::Unpack(candidate.observed_worst));
          first_chunk = true;
          if (skipping) {
            skipped_documents_.fetch_add(1, std::memory_order_relaxed);
          } else if (PublishedWorst::Valid(candidate.observed_worst)) {
            // The commit's own test, on a worst no better than the commit's by
            // the time it reaches the document: a document that cannot reach
            // the top K under the earlier worst cannot under the later one.
            counting =
                count_only_(source_index, doc,
                            RankKey{.tier = segment_doc.tier,
                                    .prior = segment_doc.prior,
                                    .oid = segment_doc.oid},
                            PublishedWorst::Unpack(candidate.observed_worst));
          }
        }
      }
      if (skipping) continue;
      candidate.count_only = counting;
      candidate.scanned = first_chunk;
      first_chunk = false;
      const Language language =
          streams_[source_index].source().doc(doc).language;
      MatchClassifier classifier(language, chunk.content, chunk.entry_state);
      int64_t line_number = chunk.first_line;
      size_t counted_through = 0;
      verifier.ForEachMatchingLine(chunk.content, [&](const LineMatch& match) {
        if (segment &&
            window_->discard_segments.load(std::memory_order_relaxed)) {
          return false;
        }
        ++candidate.matching_lines;
        if (counting) return true;
        const MatchClass match_class =
            kept < request_.max_lines_per_file
                ? classifier.Classify(match.match_begin, match.match_end)
                : MatchClass::kCodeIdentifier;
        if (request_.count_sample) {
          CountMatchClasses(pattern_, chunk.content, match, classifier,
                            candidate.match_classes);
        }
        if (kept < request_.max_lines_per_file) {
          line_number += CountNewlines(chunk.content.substr(
              counted_through, match.line_begin - counted_through));
          counted_through = match.line_begin;
          candidate.matches.push_back(
              KeptLine{.number = line_number,
                       .begin = chunk.offset_in_doc + match.line_begin,
                       .end = chunk.offset_in_doc + match.line_end,
                       .match_begin = match.match_begin - match.line_begin,
                       .match_end = match.match_end - match.line_begin,
                       .match_class = match_class});
          ++kept;
        }
        return true;
      });
    }
  }

  bool Work(LineVerifier& verifier) {
    size_t index;
    if (!window_->Claim(index)) return false;
    Grain& grain = window_->slots[index % window_->slots.size()];
    if (options_.stage_timing) {
      const auto start = std::chrono::steady_clock::now();
      ScanGrain(grain, verifier);
      scan_ns_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count(),
                         std::memory_order_relaxed);
    } else {
      ScanGrain(grain, verifier);
    }
    {
      absl::MutexLock lock(window_->mu);
      grain.done = true;
    }
    return true;
  }

  void NextGrain(Grain& grain) {
    int64_t bytes = 0;
    while (pending_) {
      const size_t i = pending_->source_index;
      const DocId doc = pending_->chunk.doc;
      do {
        bytes += std::ssize(pending_->chunk.content);
        grain.candidates.push_back(std::move(*pending_));
        pending_ = next_();
      } while (pending_ && pending_->source_index == i &&
               pending_->chunk.doc == doc);
      if (grain.candidates.size() >=
              static_cast<size_t>(options_.grain_chunks) &&
          bytes >= options_.grain_bytes) {
        break;
      }
    }
  }

  Next& next_;
  Commit& commit_;
  Stop& stop_;
  CountOnly& count_only_;
  const PublishedWorst& published_;
  std::atomic<int64_t> skipped_documents_{0};
  std::atomic<int64_t> scan_ns_{0};
  absl::Span<const CandidateStream> streams_;
  const CompiledPattern& pattern_;
  const SearchRequest& request_;
  SearchResponse& response_;
  const SearcherOptions& options_;
  SearchExecutor::Admission& admission_;
  std::shared_ptr<Window> window_;
  std::optional<Candidate> pending_;
};

// One document of the count sample's frame, and its place in the selection.
struct SampleDoc {
  uint32_t hash;
  Oid oid;
  Source source;
  DocId id;
  int64_t paths;
  size_t source_index;
};

// The count sample's selection order: the lowest CRC32C of the Oid, then the
// Oid, then the source and DocId.
bool SampleBefore(const SampleDoc& a, const SampleDoc& b) {
  if (a.hash != b.hash) return a.hash < b.hash;
  if (a.oid != b.oid) return a.oid < b.oid;
  return a.source_index != b.source_index ? a.source_index < b.source_index
                                          : a.id < b.id;
}

// Whether `candidate` would enter the bounded sample of the `capacity` lowest
// keys held as a max-heap in `sample`.
bool WouldEnterSample(const std::vector<SampleDoc>& sample, size_t capacity,
                      const SampleDoc& candidate) {
  return sample.size() < capacity || SampleBefore(candidate, sample.front());
}

// Inserts `candidate` into `sample`, which holds at most `capacity` documents
// as a max-heap under SampleBefore, if it is among the lowest keys.
void OfferToSample(std::vector<SampleDoc>& sample, size_t capacity,
                   const SampleDoc& candidate) {
  if (!WouldEnterSample(sample, capacity, candidate)) return;
  if (sample.size() < capacity) {
    sample.push_back(candidate);
    std::push_heap(sample.begin(), sample.end(), SampleBefore);
  } else {
    std::pop_heap(sample.begin(), sample.end(), SampleBefore);
    sample.back() = candidate;
    std::push_heap(sample.begin(), sample.end(), SampleBefore);
  }
}

// What a search had decided when it stopped on rank, as the sample's frame
// needs it.
struct SampleInput {
  absl::Span<const Source> sources;
  const CompiledPattern& pattern;
  const SearcherOptions& options;
  bool deduplicate;
  const PathGlob* glob;
  const SearchView* view;
  ViewPaths* view_paths;
  // Oids verified (deduplicating search) and (source, DocId) pairs verified
  // (otherwise).
  const absl::flat_hash_set<Oid>& verified_documents;
  const absl::flat_hash_set<std::pair<size_t, DocId>>& verified_ids;
  // Per source, the first DocId the commit had not finished when the search
  // stopped; zero for a memtable.
  absl::Span<const DocId> frontier;
  // The executor's pool and this query's share of it, if any.
  ThreadPool* pool = nullptr;
  int parallelism = 1;
};

constexpr int kMaxFrameParts = 8;

struct SampleFrame {
  // The selected documents in SampleBefore order, at most count_sample_size.
  std::vector<SampleDoc> sample;
  // The unverified candidate documents, each once.
  int64_t documents = 0;
  // Candidate chunks the draw yielded.
  int64_t chunks = 0;
};

// How a count-sample scan of one source treats Oids.
enum class FrameDedup {
  // Documents are keyed by (source, DocId): no deduplication.
  kNone,
  // A document whose Oid is in the shared set is a copy; the set is only read.
  kCheck,
  // A document whose Oid is in the shared set is a copy; the others join it.
  kInsert
};

// A scan's results; scans over parts of a source merge by adding the counts
// and offering each part's sample to the whole.
struct FrameScan {
  std::vector<SampleDoc> sample;
  int64_t documents = 0;
  int64_t chunks = 0;
  SearchStats stats;
};

// Scans the candidate chunks of `source_index` in [begin, end) (whole
// documents only: both bounds fall on document starts) and offers the frame's
// documents to `out`. `unverified` holds the Oids of earlier frame documents
// (kCheck, kInsert); `unverified_ids` the (source, DocId) pairs of a memtable
// without deduplication.
absl::Status ScanFrame(const SampleInput& in, size_t source_index,
                       uint64_t begin, uint64_t end, FrameDedup dedup,
                       absl::flat_hash_set<Oid>& unverified,
                       absl::flat_hash_set<std::pair<size_t, DocId>>& ids,
                       FrameScan& out) {
  const SearcherOptions& options = in.options;
  const size_t capacity = static_cast<size_t>(options.count_sample_size);
  const int level = options.count_sample_level;
  const Source& source = in.sources[source_index];
  SearchStats& ignored = out.stats;
  ABSL_ASSIGN_OR_RETURN(
      const TrigramQuery source_query,
      SourceQuery(source, in.pattern, options.max_bigram_posting_share));
  ABSL_ASSIGN_OR_RETURN(CandidateStream stream,
                        CandidateStream::Create(source, source_query, ignored));
  const bool seekable = level >= 2 && !source.mem;
  if (seekable) {
    if (begin > 0) stream.SeekTo(static_cast<ChunkId>(begin));
    stream.SetLimit(static_cast<ChunkId>(end));
  }
  std::optional<DocId> last_doc;
  while (stream.Next(ignored, false)) {
    ++out.chunks;
    // A segment's document is read field by field: this runs for every
    // candidate document.
    const DocId id = source.mem ? source.mem->chunk(stream.id()).doc
                                : source.segment->doc_of_chunk(stream.id());
    stream.Pop();
    const Oid oid =
        source.mem ? source.doc(id).oid : source.segment->doc_oid(id);
    if (seekable) {
      // The rest of the document's candidate chunks are the same document.
      const auto [first, count] = source.segment->doc_chunks(id);
      stream.SkipPast(first + count);
    } else if (!source.mem) {
      if (last_doc == id) continue;
      last_doc = id;
    }
    // Without deduplication a segment document is verified only if the commit
    // reached it, which is at or before the frontier.
    const bool maybe_verified =
        in.deduplicate || !seekable || id <= in.frontier[source_index];
    if ((in.deduplicate ? in.verified_documents.contains(oid)
                        : maybe_verified &&
                              in.verified_ids.contains({source_index, id})) ||
        (in.glob && !in.view && !in.glob->Matches(source.doc(id).hint_path)) ||
        (in.view_paths && (!in.view->state->Contains(source.blob(id)) ||
                           !in.view_paths->Admits(source.blob(id))))) {
      continue;
    }
    // A memtable orders chunks by rank, so a document's chunks need not be
    // adjacent; and a deduplicating search counts an Oid once.
    if (dedup == FrameDedup::kInsert) {
      if (!unverified.insert(oid).second) continue;
    } else if (dedup == FrameDedup::kCheck) {
      // A copy within the segment is the same frame document as its first.
      if ((!source.mem && source.segment->is_repeated_oid(id)) ||
          unverified.contains(oid)) {
        continue;
      }
    } else if (source.mem) {
      if (!ids.insert({source_index, id}).second) continue;
    }
    ++out.documents;
    const uint32_t hash = static_cast<uint32_t>(absl::ComputeCrc32c(oid.raw()));
    SampleDoc candidate{hash, oid, source, id, 0, source_index};
    if (level >= 1 && !WouldEnterSample(out.sample, capacity, candidate)) {
      continue;
    }
    candidate.paths =
        in.view_paths ? in.view_paths->PathCount(source.blob(id)) : 1;
    if (level == 0) {
      out.sample.push_back(candidate);
      std::sort(out.sample.begin(), out.sample.end(), SampleBefore);
      if (out.sample.size() > capacity) out.sample.pop_back();
    } else {
      OfferToSample(out.sample, capacity, candidate);
    }
  }
  return absl::OkStatus();
}

// Draws the count sample: the count_sample_size documents of the frame with the
// lowest SampleBefore keys, and the frame's size. Level 0 is the reference
// selection: every stream from its first chunk, a full sort per push. Level 1
// keeps a bounded heap. Level 2 also resumes each segment stream at the
// commit's frontier and skips the rest of a document. Level 3 also keeps the
// Oid set small and splits a segment's range over the executor's workers. All
// levels return a sample of the same documents and the same size; copies of one
// Oid are interchangeable.
absl::StatusOr<SampleFrame> DrawSampleFrame(const SampleInput& in,
                                            SearchStats& stats) {
  StageClock& clock = ThisThreadStageClock();
  const int64_t decode_before = clock.decode_ns;
  const int64_t intersect_before = clock.intersect_ns;
  const SearcherOptions& options = in.options;
  const size_t capacity = static_cast<size_t>(options.count_sample_size);
  const int level = options.count_sample_level;
  SampleFrame frame;
  // Oids already in the frame. A source's documents can share an Oid.
  absl::flat_hash_set<Oid> unverified;
  absl::flat_hash_set<std::pair<size_t, DocId>> unverified_ids;
  // Level 3 takes the memtables first, so that a segment's documents need only
  // be looked up among them, not inserted.
  std::vector<size_t> order(in.sources.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  if (level >= 3 && in.deduplicate) {
    std::stable_partition(order.begin(), order.end(), [&](size_t i) {
      return in.sources[i].mem != nullptr;
    });
  }
  FrameScan whole;
  int64_t max_parts = 1;
  for (size_t position = 0; position < order.size(); ++position) {
    const size_t source_index = order[position];
    const Source& source = in.sources[source_index];
    bool later_segment = false;
    for (size_t later = position + 1; later < order.size(); ++later) {
      later_segment |= in.sources[order[later]].mem == nullptr;
    }
    FrameDedup dedup = FrameDedup::kNone;
    if (in.deduplicate) {
      // A glob on the hint path can admit one copy and not another.
      dedup = level < 3 || source.mem || later_segment ||
                      (in.glob != nullptr && in.view == nullptr &&
                       source.segment->has_duplicate_oids())
                  ? FrameDedup::kInsert
                  : FrameDedup::kCheck;
    }
    uint64_t begin = 0;
    uint64_t end = source.num_chunks();
    if (level >= 2 && !source.mem && in.frontier[source_index] > 0) {
      if (in.frontier[source_index] >= source.segment->num_docs()) continue;
      begin = source.segment->doc(in.frontier[source_index]).first_chunk;
    }
    // Parts split at document starts; Admits memoizes under a glob, which is
    // not thread-safe.
    int parts = 1;
    if (level >= 3 && !source.mem && dedup != FrameDedup::kInsert &&
        in.pool != nullptr && in.parallelism > 1 &&
        !(in.glob != nullptr && in.view_paths != nullptr) &&
        end - begin >=
            static_cast<uint64_t>(options.count_sample_parallel_chunks)) {
      parts = std::min<int>(in.parallelism, kMaxFrameParts);
    }
    max_parts = std::max<int64_t>(max_parts, parts);
    if (parts == 1) {
      ABSL_RETURN_IF_ERROR(ScanFrame(in, source_index, begin, end, dedup,
                                     unverified, unverified_ids, whole));
      continue;
    }
    // Boundaries: the first chunk of the document at each equal share of the
    // documents past `begin`.
    const uint32_t num_docs = source.segment->num_docs();
    const DocId first_doc =
        begin == 0 ? 0
                   : source.segment->doc_of_chunk(static_cast<ChunkId>(begin));
    std::vector<uint64_t> bounds = {begin};
    for (int part = 1; part < parts; ++part) {
      const DocId doc = static_cast<DocId>(
          first_doc + (uint64_t{num_docs} - first_doc) * part / parts);
      const uint64_t chunk =
          doc < num_docs ? source.segment->doc_chunks(doc).first : end;
      bounds.push_back(std::max(chunk, bounds.back()));
    }
    bounds.push_back(end);
    std::vector<FrameScan> scans(static_cast<size_t>(parts));
    std::vector<absl::Status> statuses(static_cast<size_t>(parts));
    in.pool->ParallelFor(
        parts, 1, parts, [&](int, int64_t first, int64_t last) {
          for (int64_t part = first; part < last; ++part) {
            absl::flat_hash_set<std::pair<size_t, DocId>> none;
            statuses[part] =
                ScanFrame(in, source_index, bounds[part], bounds[part + 1],
                          dedup, unverified, none, scans[part]);
          }
        });
    for (int part = 0; part < parts; ++part) {
      ABSL_RETURN_IF_ERROR(statuses[part]);
      FrameScan& scan = scans[part];
      whole.documents += scan.documents;
      whole.chunks += scan.chunks;
      whole.stats.dictionary_time += scan.stats.dictionary_time;
      whole.stats.evaluate_time += scan.stats.evaluate_time;
      for (const SampleDoc& doc : scan.sample) {
        OfferToSample(whole.sample, capacity, doc);
      }
    }
  }
  frame.sample = std::move(whole.sample);
  frame.documents = whole.documents;
  frame.chunks = whole.chunks;
  std::sort(frame.sample.begin(), frame.sample.end(), SampleBefore);
  stats.count_sample_parts = max_parts;
  stats.count_sample_dictionary_time += whole.stats.dictionary_time;
  stats.count_sample_evaluate_time += whole.stats.evaluate_time;
  stats.count_sample_decode_time =
      absl::Nanoseconds(clock.decode_ns - decode_before);
  stats.count_sample_intersect_time =
      absl::Nanoseconds(clock.intersect_ns - intersect_before);
  return frame;
}

// All sources share the verifier and path filter. Memtable candidates are
// verified even after the segment budget has been exhausted.
absl::StatusOr<SearchResponse> SearchSources(absl::Span<const Source> sources,
                                             const SearchRequest& request,
                                             bool deduplicate,
                                             const SearcherOptions& options,
                                             const SearchView* view,
                                             const BlobRegistry* blobs) {
  ABSL_RETURN_IF_ERROR(ValidateRequest(request));
  // A lone segment without copies of an Oid has no Oid to deduplicate: the
  // search is the same as one that does not deduplicate, without the Oid sets.
  if (deduplicate && options.lone_segment_skips_dedup && sources.size() == 1 &&
      sources[0].mem == nullptr && !sources[0].segment->has_duplicate_oids()) {
    deduplicate = false;
  }
  if (options.parallel_threshold_bytes < 0 || options.grain_chunks < 1 ||
      options.grain_bytes < 1 || options.ring_slots_per_worker < 1 ||
      options.full_ring_slots_per_worker < 1 ||
      options.priority_range_docs < 1) {
    return absl::InvalidArgumentError("invalid parallel verification options");
  }
  SearchResponse response;
  SearchStats& stats = response.stats;
  StageClockScope stage_clock(options.stage_timing);
  const absl::Time compile_start = absl::Now();
  ABSL_ASSIGN_OR_RETURN(
      const std::unique_ptr<CompiledPattern> pattern,
      CompilePattern({.pattern = request.pattern,
                      .literal = !request.regex,
                      .case_sensitive = request.case_sensitive}));
  std::optional<PathGlob> glob;
  if (!request.path_glob.empty()) {
    ABSL_ASSIGN_OR_RETURN(glob, PathGlob::Parse(request.path_glob));
  }
  std::vector<CandidateStream> streams;
  streams.reserve(sources.size());
  for (const Source& source : sources) {
    ABSL_ASSIGN_OR_RETURN(
        const TrigramQuery source_query,
        SourceQuery(source, *pattern, options.max_bigram_posting_share));
    ABSL_ASSIGN_OR_RETURN(CandidateStream stream,
                          CandidateStream::Create(source, source_query, stats));
    streams.push_back(std::move(stream));
  }
  stats.compile_time = absl::Now() - compile_start - stats.evaluate_time;

  std::optional<ViewPaths> view_paths;
  if (view != nullptr) view_paths.emplace(*view, glob ? &*glob : nullptr);
  absl::flat_hash_set<Oid> seen;
  // The count sample's verified Oids, which only a search without
  // deduplication leaves empty: with it, `seen` is that set.
  const absl::flat_hash_set<Oid> no_verified_documents;
  absl::flat_hash_set<std::pair<size_t, DocId>> verified_ids;
  // Skipped segment copies suppress other segment copies, but not memtables.
  absl::flat_hash_set<Oid> skipped_segments;
  int64_t kept_lines = 0;
  std::vector<std::unique_ptr<MatchCollector>> collectors;
  collectors.reserve(streams.size());
  for (const CandidateStream& stream : streams) {
    collectors.push_back(std::make_unique<MatchCollector>(
        stream.source(), request, *pattern, response, kept_lines,
        view_paths ? &*view_paths : nullptr));
  }
  // Segment streams are already sorted by RankKey; memtables were sorted
  // in Init(). The heap merges documents, not just windows, by rank.
  auto worse = [&](size_t a, size_t b) {
    const RankKey& left = streams[a].rank();
    const RankKey& right = streams[b].rank();
    return left == right ? a > b : left > right;
  };
  std::priority_queue<size_t, std::vector<size_t>, decltype(worse)> heap(worse);
  // Keep the best finished files; the heap's top is the worst of those K.
  auto better_file = [&](size_t a, size_t b) {
    return RanksBefore(response.files[a], response.files[b]);
  };
  std::priority_queue<size_t, std::vector<size_t>, decltype(better_file)>
      best_files(better_file);
  PublishedWorst published_worst;
  std::vector<std::optional<DocId>> last_doc(streams.size());
  // Per segment source, the document the commit last reached before the rank
  // stop, for the count sample.
  std::vector<DocId> frontier(streams.size(), 0);
  std::vector<bool> eligible(streams.size(), false);
  std::vector<DocTrace> traces(options.doc_probe ? streams.size() : 0);
  auto finish_file = [&](size_t source_index) {
    MatchCollector& collector = *collectors[source_index];
    const size_t index = response.files.size();
    collector.FinishFile();
    if (options.doc_probe && traces[source_index].chunks > 0) {
      DocTrace& trace = traces[source_index];
      trace.source = source_index;
      trace.doc = *last_doc[source_index];
      if (response.files.size() > index) {
        const FileResult& file = response.files[index];
        int64_t line_index = 0;
        for (const ResultLine& line : file.lines) {
          if (!line.is_match) continue;
          if (line.match_class == MatchClass::kCodeIdentifier &&
              trace.first_identifier_line < 0) {
            trace.first_identifier_line = line_index;
          }
          ++line_index;
        }
        trace.matching_lines = file.num_matching_lines;
      }
      (*options.doc_probe)(trace);
      trace = DocTrace{};
    }
    if (response.files.size() == index || request.max_files == 0 ||
        request.exact_counts) {
      return;
    }
    for (size_t file = index; file < response.files.size(); ++file) {
      response.files[file].score = ScoreFile(response.files[file]);
      if (static_cast<int64_t>(best_files.size()) < request.max_files) {
        best_files.push(file);
      } else if (better_file(file, best_files.top())) {
        best_files.pop();
        best_files.push(file);
      }
    }
    if (static_cast<int64_t>(best_files.size()) == request.max_files) {
      published_worst.Publish(response.files[best_files.top()]);
    }
  };
  int64_t segment_bytes = 0;
  // Bound names separately. View basenames live in shared snapshot chunks;
  // unscoped basenames live in the segment. Without a required literal,
  // test every basename with the compiled matcher only when needed.
  const bool name_aware_bound =
      options.name_aware_bound && (!view || blobs != nullptr);
  // The query's key in the segments' code-token lists, when it has one.
  const std::optional<std::string> token_key =
      request.exact_counts ? std::nullopt : CodeTokenBoundKey(request);
  // The ceiling of a document outside the priority set and, for an eligible
  // query, outside its segment's code-token list.
  const int64_t unlisted_ceiling =
      token_key.has_value() ? kMaxMatchCountBoost : kMaxBoostWithoutName;
  if (streams.size() > 1) {
    for (size_t i = 0; i < streams.size(); ++i) {
      if (streams[i].Next(stats)) heap.push(i);
    }
  }
  // The rank-stop bound: whether no document at or after `next` in any
  // segment can outrank `worst`. The commit and the workers both call it.
  //
  // Safety of the workers' use. Let W be the worst the commit holds when it
  // reaches a segment document D, and W' the worst a worker read earlier from
  // PublishedWorst. W' is a worst the commit held at some earlier point, and
  // the worst only improves: tier never rises and, within a tier, score never
  // falls. Every disjunct of the bound, CannotOutrank(k, b, worst), is
  // monotone in that order (a smaller tier or larger score can only turn a
  // true result true), so stop_at(D, W') implies stop_at(D, W). The priority
  // set depends only on the query and snapshot, not on the worst. So if a
  // worker skips D, the commit either stops at D (D is verified by nobody, as
  // in one thread), or finds D ineligible (glob, Oid already seen), or has had
  // its segment verification cut by the byte budget; in all three D's lines are
  // never read. Candidates skipped by a worker therefore are exactly ones the
  // ordered commit would not have kept.
  // The query's share of the executor, which the priority scan and the
  // verification pipeline both draw on.
  std::optional<SearchExecutor::Admission> admission;
  if (options.executor) admission.emplace(*options.executor);
  // The view's named blobs, built when the priority scan or the counting-only
  // test first needs them.
  std::optional<NamedBlobs> named_blobs;
  absl::once_flag named_once;
  auto get_named = [&]() -> const NamedBlobs& {
    absl::call_once(named_once, [&] {
      named_blobs.emplace(
          *pattern, glob ? &*glob : nullptr, *view,
          NamedBlobs::Options{
              .pool = admission && options.priority_parallel
                          ? &admission->pool()
                          : nullptr,
              .parallelism = admission ? admission->parallelism() : 1,
              .alternatives = options.priority_alternatives});
    });
    return *named_blobs;
  };
  std::optional<PriorityDocuments> priority;
  absl::once_flag priority_once;
  absl::Duration priority_scan;
  std::atomic<bool> priority_ready{false};
  bool priority_accounted = false;
  auto stop_at = [&](const RankKey& next, const FileResult& worst) {
    if (!options.early_termination) return false;
    if (!name_aware_bound) return CannotOutrank(next, kMaxBoost, worst);
    if (!CannotOutrank(next, unlisted_ceiling, worst)) return false;
    absl::call_once(priority_once, [&] {
      const absl::Time scan_start = absl::Now();
      priority.emplace(
          streams, *pattern, glob ? &*glob : nullptr, view,
          view ? &get_named() : nullptr, token_key,
          PriorityDocuments::ScanOptions{
              .restriction = options.priority_restriction,
              .range_docs = options.priority_range_docs,
              .pool = admission && options.priority_parallel
                          ? &admission->pool()
                          : nullptr,
              .parallelism = admission ? admission->parallelism() : 1,
              .alternatives = options.priority_alternatives});
      if (options.priority_probe != nullptr) {
        PriorityProbe probe;
        priority->Describe(probe);
        (*options.priority_probe)(probe);
      }
      priority_scan = absl::Now() - scan_start;
      priority_ready.store(true, std::memory_order_release);
    });
    return !priority->MayOutrank(next, worst);
  };
  // Statistics come from the commit, whichever thread built the set, and
  // only once the commit's own test needs it.
  auto account_priority = [&] {
    if (priority_accounted || !priority_ready.load(std::memory_order_acquire)) {
      return;
    }
    priority_accounted = true;
    stats.priority_documents = priority->size();
    stats.priority_basenames_tested = priority->basenames_tested();
    stats.priority_basenames_scanned = priority->basenames_scanned();
    stats.priority_restricted_sources = priority->restricted_sources();
    stats.code_token_documents = priority->code_token_documents();
    stats.priority_scan_time += priority_scan;
    stats.evaluate_time += stats.priority_scan_time;
  };
  // Counting-only verification: a segment document whose ceiling cannot reach
  // the heap's worst file is counted, not classified or kept. Off for exact
  // counts, class counts and a top K of zero files.
  const bool count_only_enabled =
      options.count_only_excluded && !request.exact_counts &&
      (!request.count_sample || options.count_only_with_sample) &&
      request.max_files > 0;
  // With a count sample the classes of counted documents are estimated from a
  // sample of them, and the commit alone decides which documents are counted,
  // so that the set does not depend on the thread count.
  const bool sample_counted = count_only_enabled && request.count_sample;
  std::vector<SampleDoc> counted_sample;
  int64_t counted_documents = 0;
  std::optional<ExcludedDocuments> excluded_documents;
  if (count_only_enabled) {
    excluded_documents.emplace(streams, *pattern, token_key, view != nullptr,
                               get_named, view != nullptr && name_aware_bound);
  }
  auto count_only = [&](size_t source_index, DocId doc, const RankKey& key,
                        const FileResult& worst) {
    if (!count_only_enabled) return false;
    // Without name-aware bounds only the coarse ceiling is known.
    return name_aware_bound
               ? excluded_documents->Excludes(source_index, doc, key, worst)
               : CannotOutrank(key, kMaxBoost, worst);
  };
  std::vector<bool> counting(streams.size(), false);
  auto verify = [&](Candidate& candidate, bool preverified) {
    stats.view_filtered_chunks += candidate.view_filtered_before;
    if (!response.terminated_early) {
      stats.view_filtered_chunks += candidate.view_filtered_segments_before;
    }
    const size_t i = candidate.source_index;
    CandidateStream& stream = streams[i];
    const Source& source = stream.source();
    MatchCollector& collector = *collectors[i];
    const SegmentChunk& chunk = candidate.chunk;
    if (!source.mem && !response.terminated_early) frontier[i] = chunk.doc;
    // A document's chunks are contiguous in every candidate stream.
    const bool new_doc = !last_doc[i].has_value() || *last_doc[i] != chunk.doc;
    if (new_doc) {
      finish_file(i);
      last_doc[i] = chunk.doc;
      eligible[i] = true;
      if ((glob && !view) || deduplicate) {
        const SegmentDoc doc = source.doc(chunk.doc);
        eligible[i] = (!glob || view || glob->Matches(doc.hint_path)) &&
                      (!deduplicate ||
                       (!seen.contains(doc.oid) &&
                        (source.mem || !skipped_segments.contains(doc.oid))));
      }
    }
    if (!eligible[i]) return;
    ++stats.candidate_chunks;
    // Once the global line cap is spent, no later document keeps a line and so
    // none can become a file: the heap's files are final, whatever their
    // scores. A cap of zero is the caller counting matches.
    const bool lines_exhausted = options.stop_at_line_cap &&
                                 request.max_lines > 0 &&
                                 kept_lines >= request.max_lines;
    if (!source.mem && !request.exact_counts && response.complete &&
        !response.terminated_early && new_doc && request.max_files > 0 &&
        static_cast<int64_t>(best_files.size()) == request.max_files) {
      const RankKey next = RankKey{.tier = source.doc(chunk.doc).tier,
                                   .prior = source.doc(chunk.doc).prior,
                                   .oid = source.doc(chunk.doc).oid};
      // A stop leaves the exhaustive top K. The merge takes documents in
      // RankKey order, so every document before `next` is verified as in an
      // exhaustive run, and the stop affects only documents at or after it:
      // later segment documents, which are no longer verified, and later
      // memtable candidates, which still are. Those may now keep lines that
      // skipped files would have used, and are no longer hidden by a skipped
      // segment copy of their Oid. It suffices that none of these documents
      // can outrank `worst`, whatever lines it keeps: the top K then holds
      // only files verified before `next`. Boosts (identifier, name and match
      // count) are nonnegative and sum to at most kMaxBoost under any line
      // caps; the match-count boost also counts unkept matches, but remains
      // capped. An expanded blob contributes one file per view path. All its
      // paths share the ingest RankKey and each name boost is at most
      // kNameBoost; therefore the same bound covers every path. Strict
      // inequality is necessary because paths break equal-score ties. The
      // check runs only while the byte budget has cut nothing, so a stop
      // leaves the unlimited-budget top K.
      //
      // The name-aware bound splits the ceiling. A document at or after
      // `next` outside the priority set fails the glob and is never kept, or
      // has no basename match and earns at most kMaxBoostWithoutName on a
      // RankKey no better than `next`. A priority document keeps kMaxBoost,
      // and in each source the first one not before `next` bounds the rest.
      // A memtable's priority documents are its candidates with a matching
      // name: only candidates are verified. Keys equal to `next` still count:
      // `next` itself, and copies of its Oid in later sources, which the
      // merge takes after it. The set ignores deduplication and matching: a
      // duplicate, or a document that matches no line, can only delay the
      // stop. View searches scan shared view-path basenames instead of hints.
      //
      // The code-token bound splits the ceiling again for an eligible query. A
      // document that is neither a priority document nor in its segment's list
      // of documents where the query's token is a code identifier cannot earn
      // either boost and keeps at most kMaxMatchCountBoost. Listed documents
      // keep kMaxBoostWithoutName, and priority ones kMaxBoost; a memtable's
      // candidates all count as listed. A segment without the section keeps the
      // previous ceilings. Views need no change: the lists cover every document
      // of the segment, and a document outside the view can only delay the
      // stop.
      const FileResult& worst = response.files[best_files.top()];
      const bool bound_stop = !lines_exhausted && stop_at(next, worst);
      const bool stop = lines_exhausted || bound_stop;
      if (options.stop_probe) {
        StopProbe probe{.next_tier = next.tier,
                        .next_prior = next.prior,
                        .worst_tier = worst.tier,
                        .worst_score = worst.score,
                        .stop = stop,
                        .name_aware = name_aware_bound,
                        .heap_full = true,
                        .kept_lines = kept_lines,
                        .lines_exhausted = lines_exhausted};
        if (!lines_exhausted && name_aware_bound) {
          probe.unlisted_blocks = !CannotOutrank(next, unlisted_ceiling, worst);
          if (!probe.unlisted_blocks && priority.has_value()) {
            priority->Explain(next, worst, probe);
          }
        }
        (*options.stop_probe)(probe);
      }
      // Attribute the priority scan only where one thread would have built
      // the set: a worker may have built it for a document this test skips.
      if (!lines_exhausted && name_aware_bound &&
          CannotOutrank(next, unlisted_ceiling, worst)) {
        account_priority();
      }
      if (stop) {
        response.terminated_early = true;
        // A worker that verified this document although the worst it read
        // already excluded it broke the skip rule.
        if (bound_stop && candidate.scanned &&
            PublishedWorst::Valid(candidate.observed_worst) &&
            stop_at(next, PublishedWorst::Unpack(candidate.observed_worst))) {
          ++stats.excluded_scans;
        }
      }
    }
    if (response.terminated_early && !source.mem) return;
    if ((!response.complete && !source.mem) ||
        (!source.mem && collector.StartsNewFile(chunk) &&
         segment_bytes >= request.max_verify_bytes)) {
      response.complete = false;
      if (deduplicate && new_doc) {
        skipped_segments.insert(source.doc(chunk.doc).oid);
      }
    } else {
      // A skipped segment document must not hide a later memtable copy: the
      // memtable is always verified, even after the segment budget runs out.
      if (deduplicate && new_doc) seen.insert(source.doc(chunk.doc).oid);
      // A deduplicating search's `seen` is the set of verified Oids, which
      // the count sample reads as `verified_documents`; otherwise it reads
      // the (source, DocId) pairs, once per document.
      if (request.count_sample && !deduplicate && new_doc) {
        verified_ids.insert({i, chunk.doc});
      }
      if (new_doc) {
        const bool commit_may_count =
            count_only_enabled && !source.mem &&
            stats.verified_bytes >= options.count_only_min_bytes &&
            static_cast<int64_t>(best_files.size()) == request.max_files;
        const auto excluded = [&] {
          const SegmentDoc counted_doc = source.doc(chunk.doc);
          return count_only(i, chunk.doc,
                            RankKey{.tier = counted_doc.tier,
                                    .prior = counted_doc.prior,
                                    .oid = counted_doc.oid},
                            response.files[best_files.top()]);
        };
        if (sample_counted) {
          // A worker's flag counts only where the commit may count, and
          // implies the commit's own test (the safety argument above): the set
          // of counted documents is the commit's, whatever the thread count.
          counting[i] =
              commit_may_count && (candidate.count_only || excluded());
        } else {
          counting[i] = candidate.count_only;
          if (commit_may_count && !counting[i]) counting[i] = excluded();
        }
        if (counting[i]) {
          ++stats.count_only_documents;
          if (sample_counted) {
            ++counted_documents;
            const SegmentDoc doc = source.doc(chunk.doc);
            OfferToSample(
                counted_sample, static_cast<size_t>(options.count_sample_size),
                SampleDoc{
                    static_cast<uint32_t>(absl::ComputeCrc32c(doc.oid.raw())),
                    doc.oid, source, chunk.doc, 0, i});
          }
        }
      }
      if (preverified &&
          !(sample_counted && candidate.count_only && !counting[i])) {
        collector.VerifyMatches(chunk, candidate.matching_lines,
                                candidate.matches, candidate.match_classes,
                                counting[i]);
      } else {
        collector.Verify(chunk, counting[i]);
      }
      ++stats.verified_chunks;
      stats.verified_bytes += std::ssize(chunk.content);
      if (counting[i]) stats.count_only_bytes += std::ssize(chunk.content);
      if (options.doc_probe) {
        ++traces[i].chunks;
        traces[i].bytes += std::ssize(chunk.content);
      }
      if (!source.mem) segment_bytes += std::ssize(chunk.content);
    }
  };
  const absl::Duration earlier_evaluate = stats.evaluate_time;
  const absl::Time verify_start = absl::Now();
  // The stream is advanced on the caller. Only commit mutates the response.
  auto next_unfiltered = [&]() -> std::optional<Candidate> {
    if (streams.size() == 1) {
      CandidateStream& stream = streams.front();
      if (response.terminated_early || !stream.Next(stats, false)) {
        return std::nullopt;
      }
      Candidate candidate{.source_index = 0,
                          .chunk = stream.source().chunk(stream.id())};
      stream.Pop();
      return candidate;
    }
    while (!heap.empty()) {
      const size_t i = heap.top();
      heap.pop();
      CandidateStream& stream = streams[i];
      if (response.terminated_early && !stream.source().mem) continue;
      Candidate candidate{.source_index = i,
                          .chunk = stream.source().chunk(stream.id())};
      stream.Pop();
      if (stream.Next(stats)) heap.push(i);
      return candidate;
    }
    return std::nullopt;
  };
  // Filter on the producer, before either inline verification or publishing
  // work to the pipeline. Invisible chunks never enter its candidate grains.
  // The pipeline can produce past an early rank stop. Attach skipped counts
  // to the next published candidate and account for them only at commit;
  // speculative segment skips must not change statistics after a rank stop.
  int64_t filtered_before = 0;
  int64_t filtered_segments_before = 0;
  auto next_candidate = [&]() -> std::optional<Candidate> {
    while (auto candidate = next_unfiltered()) {
      if (!view) return candidate;
      const Source& source = streams[candidate->source_index].source();
      const BlobId blob = source.blob(candidate->chunk.doc);
      if (!view->state->Contains(blob)) {
        ++(source.mem ? filtered_before : filtered_segments_before);
        continue;
      }
      if (view_paths->Admits(blob)) {
        candidate->view_filtered_before = std::exchange(filtered_before, 0);
        candidate->view_filtered_segments_before =
            std::exchange(filtered_segments_before, 0);
        return candidate;
      }
    }
    return std::nullopt;
  };
  int64_t candidate_bytes = 0;
  std::optional<Candidate> pending;
  while ((pending = next_candidate())) {
    candidate_bytes += std::ssize(pending->chunk.content);
    if (admission && admission->parallelism() > 1 &&
        candidate_bytes > options.parallel_threshold_bytes) {
      VerificationPipeline pipeline(next_candidate, verify, stop_at, count_only,
                                    published_worst,
                                    absl::MakeConstSpan(streams), *pattern,
                                    request, response, options, *admission);
      pipeline.Run(std::move(pending));
      stats.worker_skipped_documents = pipeline.skipped_documents();
      stats.worker_scan_ns = pipeline.scan_ns();
      break;
    }
    verify(*pending, false);
  }
  stats.view_filtered_chunks += filtered_before;
  if (!response.terminated_early) {
    stats.view_filtered_chunks += filtered_segments_before;
  }
  stats.verify_time +=
      absl::Now() - verify_start - (stats.evaluate_time - earlier_evaluate);
  for (size_t i = 0; i < collectors.size(); ++i) finish_file(i);
  // The count sample below evaluates streams again; it is not evaluation.
  stage_clock.Snapshot(stats);
  // Verifies each document in full and adds its weighted totals: matching
  // files (view paths), matching lines and MatchClass occurrences.
  auto classify_sample = [&](const std::vector<SampleDoc>& docs, double& files,
                             double& lines, std::array<double, 4>& classes) {
    struct DocCounts {
      int64_t lines = 0;
      std::array<int64_t, 4> classes = {};
    };
    std::vector<DocCounts> counts(docs.size());
    auto classify = [&](LineVerifier& verifier, size_t index) {
      const SampleDoc& item = docs[index];
      const SegmentDoc doc = item.source.doc(item.id);
      DocCounts& out = counts[index];
      for (uint32_t i = 0; i < doc.num_chunks; ++i) {
        const SegmentChunk chunk = item.source.chunk(doc.first_chunk + i);
        MatchClassifier classifier(doc.language, chunk.content,
                                   chunk.entry_state);
        verifier.ForEachMatchingLine(
            chunk.content, [&](const LineMatch& match) {
              ++out.lines;
              CountMatchClasses(*pattern, chunk.content, match, classifier,
                                out.classes);
              return true;
            });
      }
    };
    // The documents are independent, and the sums below run in document
    // order whoever classified them.
    const int workers =
        admission && options.count_sample_level >= 3
            ? static_cast<int>(std::min<size_t>(
                  static_cast<size_t>(admission->parallelism()), docs.size()))
            : 1;
    if (workers > 1) {
      std::vector<std::unique_ptr<LineVerifier>> verifiers(
          static_cast<size_t>(workers));
      admission->pool().ParallelFor(
          std::ssize(docs), 1, workers,
          [&](int worker, int64_t begin, int64_t end) {
            std::unique_ptr<LineVerifier>& verifier =
                verifiers[static_cast<size_t>(worker)];
            if (!verifier) verifier = std::make_unique<LineVerifier>(*pattern);
            for (int64_t i = begin; i < end; ++i) {
              classify(*verifier, static_cast<size_t>(i));
            }
          });
    } else {
      LineVerifier verifier(*pattern);
      for (size_t i = 0; i < docs.size(); ++i) classify(verifier, i);
    }
    for (size_t i = 0; i < docs.size(); ++i) {
      const double paths = static_cast<double>(docs[i].paths);
      if (counts[i].lines > 0) files += paths;
      lines += static_cast<double>(counts[i].lines) * paths;
      for (size_t cls = 0; cls < classes.size(); ++cls) {
        classes[cls] += static_cast<double>(counts[i].classes[cls]) * paths;
      }
    }
  };
  const absl::Time sample_start = absl::Now();
  if (stats.candidate_chunks > 0) {
    response.verified_fraction = static_cast<double>(stats.verified_chunks) /
                                 static_cast<double>(stats.candidate_chunks);
  }
  if (request.count_sample && response.terminated_early &&
      options.count_sample_size > 0) {
    // The parallel producer may have advanced past the committed prefix:
    // never use its speculative position as the sampling frame. Oids make the
    // selection independent of thread count.
    ABSL_ASSIGN_OR_RETURN(
        SampleFrame frame,
        DrawSampleFrame(
            {.sources = sources,
             .pattern = *pattern,
             .options = options,
             .deduplicate = deduplicate,
             .glob = glob ? &*glob : nullptr,
             .view = view,
             .view_paths = view_paths ? &*view_paths : nullptr,
             .verified_documents = deduplicate ? seen : no_verified_documents,
             .verified_ids = verified_ids,
             .frontier = frontier,
             .pool = admission ? &admission->pool() : nullptr,
             .parallelism = admission ? admission->parallelism() : 1},
            stats));
    std::vector<SampleDoc>& sample = frame.sample;
    const int64_t unverified_count = frame.documents;
    stats.count_sample_chunks = frame.chunks;
    stats.count_sample_documents = frame.documents;
    int64_t sample_bytes = 0;
    for (const SampleDoc& item : sample) {
      sample_bytes += std::ssize(item.source.doc(item.id).content);
    }
    if (sample_bytes <= request.max_verify_bytes - stats.verified_bytes) {
      response.count_sample_size = std::ssize(sample);
      stats.count_sample_bytes = sample_bytes;
    } else {
      sample.clear();
    }
    if (!sample.empty()) {
      const absl::Time verify_sample_start = absl::Now();
      double files = 0;
      double lines = 0;
      std::array<double, 4> classes = {};
      classify_sample(sample, files, lines, classes);
      stats.count_sample_verify_time = absl::Now() - verify_sample_start;
      const double weight = static_cast<double>(unverified_count) /
                            static_cast<double>(sample.size());
      response.num_matching_files +=
          static_cast<int64_t>(std::llround(files * weight));
      response.num_matching_lines +=
          static_cast<int64_t>(std::llround(lines * weight));
      for (size_t i = 0; i < classes.size(); ++i) {
        response.match_classes[i] += classes[i] * weight;
      }
      response.counts_estimated = true;
    }
  }
  if (sample_counted && counted_documents > 0) {
    // The classes of the counted documents: their lines are exact, their
    // classes are the expansion of a sample of them.
    std::sort(counted_sample.begin(), counted_sample.end(), SampleBefore);
    for (SampleDoc& item : counted_sample) {
      item.paths =
          view_paths ? view_paths->PathCount(item.source.blob(item.id)) : 1;
    }
    double files = 0;
    double lines = 0;
    std::array<double, 4> classes = {};
    classify_sample(counted_sample, files, lines, classes);
    const double weight = static_cast<double>(counted_documents) /
                          static_cast<double>(counted_sample.size());
    for (size_t i = 0; i < classes.size(); ++i) {
      response.match_classes[i] += classes[i] * weight;
    }
    stats.counted_sample_documents = std::ssize(counted_sample);
  }
  stats.count_sample_time = absl::Now() - sample_start;
  const absl::Time rank_start = absl::Now();
  const double classified =
      response.match_classes[0] + response.match_classes[1] +
      response.match_classes[2] + response.match_classes[3];
  if (classified > 0) {
    response.token_precision = response.match_classes[0] / classified;
  }
  RankFiles(response.files);
  if (std::ssize(response.files) > request.max_files) {
    response.files.erase(
        response.files.begin() + static_cast<ptrdiff_t>(request.max_files),
        response.files.end());
  }
  stats.rank_time = absl::Now() - rank_start;
  return response;
}

}  // namespace

absl::StatusOr<SearchResponse> SearchSegment(const Segment& segment,
                                             const SearchRequest& request,
                                             const SearcherOptions& options) {
  const Source source{.segment = &segment};
  return SearchSources(absl::MakeConstSpan(&source, 1), request, false, options,
                       nullptr, nullptr);
}

namespace {
absl::StatusOr<SearchResponse> SearchState(const ReadState& state,
                                           const SearchView* view,
                                           const SearchRequest& request,
                                           const SearcherOptions& options) {
  std::vector<Source> sources;
  sources.reserve(state.frozen.size() + state.version->segments.size() + 1);
  // Sources are oldest first, breaking equal-RankKey ties by arrival order.
  // The heap still chooses by rank, not by source order.
  for (const VersionSegment& entry : state.version->segments) {
    sources.push_back(Source{.segment = entry.segment.get(),
                             .blob_of_doc = entry.blob_of_doc.get(),
                             .docs_of_blob = entry.docs_of_blob.get()});
  }
  for (const auto& frozen : state.frozen) {
    sources.push_back(
        Source{.mem = frozen.get(), .watermark = frozen->published_docs()});
  }
  if (state.mem && state.mem_watermark) {
    sources.push_back(
        Source{.mem = state.mem.get(), .watermark = state.mem_watermark});
  }
  return SearchSources(sources, request, true, options, view,
                       view ? state.blobs.get() : nullptr);
}
}  // namespace

absl::StatusOr<SearchResponse> Search(const ReadState& state,
                                      const SearchRequest& request,
                                      const SearcherOptions& options) {
  return SearchState(state, nullptr, request, options);
}

absl::StatusOr<SearchResponse> Search(const ReadState& state,
                                      const SearchView& view,
                                      const SearchRequest& request,
                                      const SearcherOptions& options) {
  return SearchState(state, &view, request, options);
}

absl::StatusOr<SearchResponse> Search(const ReadState& state,
                                      const SearchView& view,
                                      const SearchRequest& request) {
  return Search(state, view, request, SearcherOptions{});
}

absl::StatusOr<SearchResponse> SearchSegment(const Segment& segment,
                                             const SearchRequest& request) {
  return SearchSegment(segment, request, SearcherOptions{});
}

absl::StatusOr<SearchResponse> Search(const ReadState& state,
                                      const SearchRequest& request) {
  return Search(state, request, SearcherOptions{});
}

}  // namespace ferret

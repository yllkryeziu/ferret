// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The request and response of a text search, shared by every search backend
// and by the pure functions over responses: ranking (ranking.h) and
// formatting (formatter.h). Files are named by path and Oid; segment-local
// ids never appear here.

#ifndef FERRET_SEARCH_SEARCH_TYPES_H_
#define FERRET_SEARCH_SEARCH_TYPES_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/time/time.h"

#include "ferret/git/oid.h"
#include "ferret/index/rank_key.h"
#include "ferret/text/lexer.h"

namespace ferret {

struct SearchRequest {
  // A literal string, or an RE2 regular expression if `regex` is set. Always
  // matched within one line.
  std::string pattern;
  bool regex = false;
  bool case_sensitive = false;
  // Only files matching this PathGlob; empty for all files.
  std::string path_glob;
  // Lines of context kept before and after each matching line.
  int context_lines = 2;
  // At most this many matching lines are kept per file and in total, in
  // verification order, and at most this many files, the best after
  // ranking. The counts in the response still include every verified match.
  int max_lines_per_file = 5;
  int64_t max_lines = 10'000;
  int64_t max_files = 10;
  // Verification stops, and the response is incomplete, once this many bytes
  // of candidate chunks have been searched.
  int64_t max_verify_bytes = int64_t{128} << 20;
  // If set, verification never stops at the top-K bound, so the counts are
  // exact up to `max_verify_bytes`.
  bool exact_counts = false;
  // Estimate counts and token precision from at most 64 unverified documents.
  bool count_sample = false;
  // Raw callers need only path, number, and text; the index ignores this hint.
  bool text_only = false;
};

// One line of a result file: a matching line or a context line.
struct ResultLine {
  // 1-based.
  int64_t number = 0;
  // The line's bytes without its '\n', possibly invalid UTF-8.
  std::string text;
  bool is_match = false;
  // For a matching line: the first match in the line, as a byte range of
  // `text`, and what kind of token it starts in.
  size_t match_begin = 0;
  size_t match_end = 0;
  MatchClass match_class = MatchClass::kCodeIdentifier;

  friend bool operator==(const ResultLine&, const ResultLine&) = default;
};

struct FileResult {
  std::string path;
  Oid oid;
  RankTier tier = RankTier::kSource;
  uint16_t prior = 0;
  // Whether the file's name contains a match of the pattern.
  bool name_matches = false;
  // Set by RankFiles().
  int64_t score = 0;
  // Every matching line found in the file, kept in `lines` or not.
  int64_t num_matching_lines = 0;
  // The kept matching lines and their context, in increasing line order and
  // without repeats.
  std::vector<ResultLine> lines;
};

struct SearchStats {
  // Evaluated candidate chunks that pass filtering and deduplication, and
  // chunks verified (including memtables). Unevaluated segment chunks are not
  // included after ranking termination.
  int64_t candidate_chunks = 0;
  // Evaluated chunks rejected because their blob is not live in the view.
  int64_t view_filtered_chunks = 0;
  int64_t verified_chunks = 0;
  int64_t verified_bytes = 0;
  // Segment documents and memtable candidates whose result-path basename
  // matches the pattern and whose path passes the glob, found once the
  // residual top-K bound first allows a stop.
  int64_t priority_documents = 0;
  // The work of that scan, which is deterministic:
  // basenames the compiled matcher (or, for a view, the glob and matcher)
  // confirmed, and basenames covered by a literal search pass over packed
  // names (once per literal per segment or view chunk set).
  int64_t priority_basenames_tested = 0;
  int64_t priority_basenames_scanned = 0;
  // Segments whose priority documents came from their candidate chunks
  // instead of a scan of every basename.
  int64_t priority_restricted_sources = 0;
  // Segment documents in the code-token lists of the query's token, plus
  // memtable candidates, found with the priority documents; zero when the
  // query is not eligible for the code-token bound.
  int64_t code_token_documents = 0;
  absl::Duration priority_scan_time = absl::ZeroDuration();
  // Grains verified by the pipeline; zero for inline verification.
  int64_t parallel_grains = 0;
  // Segment documents the pipeline's workers skipped because the committed
  // heap's worst file already excluded them. Varies with timing.
  int64_t worker_skipped_documents = 0;
  // Documents a worker verified although the worst it read excluded them.
  // Always zero; tests assert it.
  int64_t excluded_scans = 0;
  // Segment documents, and their verified bytes, whose matching lines the
  // commit counted without classifying or keeping them because they could not
  // reach the top K. Included in the verified counts.
  int64_t count_only_documents = 0;
  int64_t count_only_bytes = 0;
  absl::Duration compile_time;
  absl::Duration evaluate_time;
  absl::Duration verify_time;
  // The stage breakdown; every duration is the calling thread's wall time.
  // Nested in compile_time: the dictionary lookups that build each source's
  // evaluator.
  absl::Duration dictionary_time;
  // Nested in evaluate_time, only with SearcherOptions::stage_timing: the
  // time decoding posting lists, and intersecting or merging id lists.
  absl::Duration decode_time;
  absl::Duration intersect_time;
  // After verification: the sampled count estimate, then ranking and trimming
  // the files.
  absl::Duration count_sample_time;
  // The count sample's work, deterministic: candidate chunks its streams
  // yielded, the documents of its frame, and the bytes of the documents it
  // verified. Zero without a sample.
  int64_t count_sample_chunks = 0;
  int64_t count_sample_documents = 0;
  int64_t count_sample_bytes = 0;
  // With counting-only verification and a count sample: the counted documents
  // whose classes were verified to estimate the rest's.
  int64_t counted_sample_documents = 0;
  // The most parts a segment's frame was split into over workers; 1 when
  // every scan ran on the calling thread.
  int64_t count_sample_parts = 0;
  // Where the sample's time goes, for profiling: the dictionary lookups that
  // open its streams, evaluating windows (which holds decode and intersect,
  // the latter two only with stage_timing), and verifying the selected
  // documents. What remains of count_sample_time is the per-candidate tests.
  absl::Duration count_sample_dictionary_time;
  absl::Duration count_sample_evaluate_time;
  absl::Duration count_sample_decode_time;
  absl::Duration count_sample_intersect_time;
  absl::Duration count_sample_verify_time;
  absl::Duration rank_time;
  // Only with stage_timing: the time the pipeline's threads, the caller
  // included, spent scanning grains. Summed over threads, so it can exceed
  // the wall time. Zero for inline verification.
  int64_t worker_scan_ns = 0;
  // Files the ripgrep backend left out of the response because the bytes rg
  // printed for one of their lines differ from the bytes read afterwards,
  // typically because the file changed in between. Their matches are in no
  // count. Always zero for the index.
  int64_t inconsistent_files = 0;
};

struct SearchResponse {
  // The files with at least one kept line, best first after RankFiles().
  std::vector<FileResult> files;
  // Over the verified candidates, including files and lines not kept.
  int64_t num_matching_files = 0;
  int64_t num_matching_lines = 0;
  // True when the files equal an unlimited-budget search, even if counts
  // cover only the verified prefix because ranking stopped early. False when
  // the byte budget stopped segment verification; files then equal the
  // budget-limited top K, not necessarily the unlimited top K.
  bool complete = true;
  // True when ranking stopped segment evaluation and verification. Memtables
  // are still verified; counts include their matches and the verified prefix.
  bool terminated_early = false;
  bool counts_estimated = false;
  int64_t count_sample_size = 0;
  // Weighted occurrence counts in code identifiers, partial tokens,
  // comments, and strings, in MatchClass order.
  std::array<double, 4> match_classes = {};
  double token_precision = 0;
  // Verified candidate chunks / evaluated candidate chunks (after path
  // filtering and deduplication). Evaluation stops at the rank bound, so this
  // fraction is not an estimate of matching files, lines, or skipped chunks.
  double verified_fraction = 1.0;
  SearchStats stats;
};

}  // namespace ferret

#endif  // FERRET_SEARCH_SEARCH_TYPES_H_

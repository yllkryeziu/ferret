// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The single definition of which files Ferret indexes.
//
// The indexer, the brute-force oracle, and the benchmark file lists all call
// CheckEligibility, so a file the index skips is skipped everywhere, and a
// benchmark can state exactly which files each engine saw. Whether a path is
// a regular, tracked or unignored file is decided earlier, by the worktree
// scanner; this function sees only content.
//
// Two policies:
//  * kAgent, the default, skips files that are larger than max_file_bytes,
//    empty, contain a NUL byte, or have more than max_distinct_trigrams
//    distinct indexable trigrams as TrigramExtractor defines them (folded,
//    no '\n').
//  * kZoekt reproduces zoekt-git-index at the pinned commit 153817f643cd run
//    with -file_limit and -max_trigram_count (index/builder.go Builder.Add
//    and index/shard_builder.go DocChecker.Check), so that benchmarks index
//    the same files in both engines. Zoekt also skips files of one or two
//    bytes, and counts trigrams of consecutive runes, case-sensitively and
//    including '\n', where an ill-formed byte decodes to U+FFFD and advances
//    one byte as in Go's utf8.DecodeRune.
// Empty files are kEmpty under both policies: nothing can match them, so they
// are not ingested, but they still appear in views. Zoekt keeps them as
// documents without content, which matches no query either.
//
// An eligible file with a line longer than long_line_bytes is flagged
// long_lines: it goes to the last rank tier, and its displayed lines are cut
// around the match.

#ifndef FERRET_TEXT_ELIGIBILITY_H_
#define FERRET_TEXT_ELIGIBILITY_H_

#include <cstdint>

#include "absl/strings/string_view.h"

#include "ferret/text/trigram_extractor.h"

namespace ferret {

enum class EligibilityPolicy {
  kAgent,
  kZoekt,
};

struct EligibilityOptions {
  EligibilityPolicy policy = EligibilityPolicy::kAgent;
  // Larger files are skipped: 1 MiB.
  int64_t max_file_bytes = int64_t{1} << 20;
  // Files with more distinct trigrams are skipped. This is the Zoekt and
  // codesearch rule; it catches generated tables and binary-like data.
  int max_distinct_trigrams = 20000;
  // An eligible file with a longer line, not counting its '\n', is flagged
  // long_lines.
  int64_t long_line_bytes = 1000;
};

// Why a file is not indexed, in the order the checks run. The values are
// dense from 0 so that reports can count them in an array of
// kNumSkipReasons.
enum class SkipReason {
  kNone = 0,  // Eligible.
  kTooLarge = 1,
  kEmpty = 2,
  kTooSmall = 3,  // kZoekt only: one or two bytes.
  kContainsNul = 4,
  kTooManyTrigrams = 5,
};
inline constexpr int kNumSkipReasons = 6;

// Returns a stable snake_case name for reports, such as "too_many_trigrams".
absl::string_view SkipReasonName(SkipReason reason);

struct Eligibility {
  SkipReason skip_reason = SkipReason::kNone;
  // Set only when skip_reason is kNone.
  bool long_lines = false;
};

// The checks of CheckEligibility that depend only on the content's size, in
// the same order: kTooLarge, kEmpty, or kTooSmall, else kNone. A caller that
// knows a file's size can skip it without reading it. O(1).
SkipReason SkipReasonForSize(int64_t size, const EligibilityOptions& options);

// Decides whether a file with this `content` is indexed under `options`,
// reporting the first reason that applies in SkipReason order. `extractor`
// is scratch space for counting trigrams under kAgent; reuse one per thread.
// O(|content|). Concurrent calls need separate extractors.
Eligibility CheckEligibility(absl::string_view content,
                             const EligibilityOptions& options,
                             TrigramExtractor& extractor);

}  // namespace ferret

#endif  // FERRET_TEXT_ELIGIBILITY_H_

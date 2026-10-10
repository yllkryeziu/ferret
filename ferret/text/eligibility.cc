// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/eligibility.h"

#include <cstddef>
#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"

#include "ferret/text/line_counter.h"
#include "ferret/text/trigram_extractor.h"
#include "ferret/text/utf8.h"

namespace ferret {
namespace {

// Zoekt's trigram count (DocChecker.Check): distinct triples of consecutive
// runes, where an ill-formed byte decodes to U+FFFD and advances one byte.
// Stops once the count exceeds `limit`.
size_t CountDistinctRuneTrigrams(absl::string_view content, size_t limit) {
  constexpr uint64_t kReplacementCharacter = 0xFFFD;
  constexpr uint64_t kRuneBits = 21;
  constexpr uint64_t kWindowMask = (uint64_t{1} << (3 * kRuneBits)) - 1;
  absl::flat_hash_set<uint64_t> trigrams;
  // The last three runes, packed as Zoekt's runesToNGram packs them.
  uint64_t window = 0;
  size_t runes = 0;
  for (size_t i = 0; i < content.size() && trigrams.size() <= limit;) {
    const Utf8Sequence sequence = DecodeUtf8(content.substr(i));
    const bool ill_formed = sequence.code_point < 0;
    const uint64_t rune = ill_formed
                              ? kReplacementCharacter
                              : static_cast<uint64_t>(sequence.code_point);
    i += ill_formed ? 1 : sequence.length;
    window = ((window << kRuneBits) | rune) & kWindowMask;
    if (++runes >= 3) trigrams.insert(window);
  }
  return trigrams.size();
}

bool HasTooManyTrigrams(absl::string_view content,
                        const EligibilityOptions& options,
                        TrigramExtractor& extractor) {
  DCHECK_GE(options.max_distinct_trigrams, 0);
  const size_t limit = static_cast<size_t>(options.max_distinct_trigrams);
  // Content of n bytes has at most n - 2 trigrams of bytes or of runes.
  if (content.size() <= limit + 2) return false;
  const size_t count = options.policy == EligibilityPolicy::kZoekt
                           ? CountDistinctRuneTrigrams(content, limit)
                           : extractor.CountDistinct(content, limit);
  return count > limit;
}

bool HasLineLongerThan(absl::string_view content, int64_t max_line_bytes) {
  for (size_t begin = 0; begin <= content.size();) {
    const size_t end = LineEnd(content, begin);
    if (static_cast<int64_t>(end - begin) > max_line_bytes) return true;
    begin = end + 1;
  }
  return false;
}

}  // namespace

absl::string_view SkipReasonName(SkipReason reason) {
  switch (reason) {
    case SkipReason::kNone:
      return "none";
    case SkipReason::kTooLarge:
      return "too_large";
    case SkipReason::kEmpty:
      return "empty";
    case SkipReason::kTooSmall:
      return "too_small";
    case SkipReason::kContainsNul:
      return "contains_nul";
    case SkipReason::kTooManyTrigrams:
      return "too_many_trigrams";
  }
  return "unknown";
}

SkipReason SkipReasonForSize(int64_t size, const EligibilityOptions& options) {
  if (size > options.max_file_bytes) return SkipReason::kTooLarge;
  if (size == 0) return SkipReason::kEmpty;
  if (options.policy == EligibilityPolicy::kZoekt && size < 3) {
    return SkipReason::kTooSmall;
  }
  return SkipReason::kNone;
}

Eligibility CheckEligibility(absl::string_view content,
                             const EligibilityOptions& options,
                             TrigramExtractor& extractor) {
  const SkipReason by_size =
      SkipReasonForSize(static_cast<int64_t>(content.size()), options);
  if (by_size != SkipReason::kNone) return {.skip_reason = by_size};
  if (absl::StrContains(content, '\0')) {
    return {.skip_reason = SkipReason::kContainsNul};
  }
  if (HasTooManyTrigrams(content, options, extractor)) {
    return {.skip_reason = SkipReason::kTooManyTrigrams};
  }
  return {.skip_reason = SkipReason::kNone,
          .long_lines = HasLineLongerThan(content, options.long_line_bytes)};
}

}  // namespace ferret

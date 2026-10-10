// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/prepared_document.h"

#include <cstddef>

#include "absl/strings/string_view.h"

#include "ferret/git/blob_hash.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/segment_writer.h"
#include "ferret/text/eligibility.h"
#include "ferret/text/language.h"
#include "ferret/text/trigram_extractor.h"

namespace ferret {
namespace {

constexpr size_t kLanguagePrefixBytes = 4096;

}  // namespace

SegmentDocument PreparedDocument::Bind(absl::string_view path,
                                       absl::string_view bytes) const {
  return {.hint_path = path,
          .content = bytes,
          .rank_key = rank_key,
          .language = language,
          .long_lines = long_lines};
}

PreparedDocument PrepareDocument(absl::string_view path,
                                 absl::string_view bytes,
                                 const EligibilityOptions& options,
                                 TrigramExtractor& extractor) {
  PreparedDocument result;
  const Eligibility eligibility = CheckEligibility(bytes, options, extractor);
  result.skip_reason = eligibility.skip_reason;
  if (result.skip_reason != SkipReason::kNone) return result;
  result.long_lines = eligibility.long_lines;
  result.language = DetectLanguage(path, bytes.substr(0, kLanguagePrefixBytes));
  result.rank_key =
      ComputeRankKey(path, bytes.size(), result.long_lines, HashBlob(bytes));
  return result;
}

}  // namespace ferret

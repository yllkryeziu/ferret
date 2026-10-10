// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/search/ranking.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "ferret/index/rank_key.h"
#include "ferret/search/router.h"
#include "ferret/search/search_types.h"
#include "ferret/text/fold.h"
#include "ferret/text/lexer.h"

namespace ferret {

std::optional<std::string> CodeTokenBoundKey(const SearchRequest& request) {
  if (!IsIdentifierQuery(request)) return std::nullopt;
  // A case-insensitive non-ASCII query matches other case variants that
  // folding does not equate, so only ASCII queries have a sound key.
  for (const char c : request.pattern) {
    if (static_cast<unsigned char>(c) >= 0x80) return std::nullopt;
  }
  return Fold(request.pattern);
}

int64_t ScoreFile(const FileResult& file) {
  int64_t boost = 0;
  if (std::any_of(file.lines.begin(), file.lines.end(),
                  [](const ResultLine& line) {
                    return line.is_match &&
                           line.match_class == MatchClass::kCodeIdentifier;
                  })) {
    boost += kIdentifierBoost;
  }
  if (file.name_matches) boost += kNameBoost;
  if (file.num_matching_lines > 1) {
    const auto doublings =
        static_cast<int64_t>(
            std::bit_width(static_cast<uint64_t>(file.num_matching_lines))) -
        1;
    boost += std::min(doublings * kMatchCountBoostStep, kMaxMatchCountBoost);
  }
  return int64_t{file.prior} + boost;
}

bool CannotOutrank(const RankKey& next, int64_t max_boost,
                   const FileResult& worst) {
  return next.tier > worst.tier ||
         (next.tier == worst.tier &&
          int64_t{next.prior} + max_boost < worst.score);
}

bool RanksBefore(const FileResult& a, const FileResult& b) {
  return std::forward_as_tuple(a.tier, -a.score, a.path) <
         std::forward_as_tuple(b.tier, -b.score, b.path);
}

void RankFiles(std::vector<FileResult>& files) {
  for (FileResult& file : files) file.score = ScoreFile(file);
  // Stable: files equal in tier, score and path (two blobs that a segment
  // holds under one hint path) keep their verification order, so the result
  // does not depend on which other files were verified.
  std::stable_sort(files.begin(), files.end(), RanksBefore);
}

}  // namespace ferret

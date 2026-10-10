// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Ranks the files of a search response. A pure function of the response, so
// every backend ranks alike.
//
// A file's score is its prior (rank_key.h) plus a boost in [0, kMaxBoost]:
//   * kIdentifierBoost if a kept match is a whole identifier in code
//     (MatchClass::kCodeIdentifier), not part of a longer token, a comment
//     or a string;
//   * kNameBoost if the file name contains a match;
//   * kMatchCountBoostStep for every doubling of the file's matching lines,
//     up to kMaxMatchCountBoost.
// Files are ordered by tier, then by descending score, then by path. The
// tier dominates: no boost lifts a test above first-party source.

#ifndef FERRET_SEARCH_RANKING_H_
#define FERRET_SEARCH_RANKING_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ferret/index/rank_key.h"
#include "ferret/search/search_types.h"

namespace ferret {

inline constexpr int64_t kIdentifierBoost = 8192;
inline constexpr int64_t kNameBoost = 8192;
inline constexpr int64_t kMatchCountBoostStep = 1024;
inline constexpr int64_t kMaxMatchCountBoost = 4096;
inline constexpr int64_t kMaxBoost =
    kIdentifierBoost + kNameBoost + kMaxMatchCountBoost;
// The most a file whose name has no match can earn.
inline constexpr int64_t kMaxBoostWithoutName = kMaxBoost - kNameBoost;

// The most a file that cannot earn the identifier boost can earn. Without a
// name match it can earn only kMaxMatchCountBoost.
inline constexpr int64_t kMaxBoostWithoutIdentifier =
    kMaxBoost - kIdentifierBoost;

// Returns the key that a segment's code-token index (kCodeTokens) uses to bound
// `request`'s identifier boost, or nullopt when the query is not eligible.
// Eligible queries are non-regex literals of a single ASCII identifier: those
// are the only patterns whose every kCodeIdentifier match is a whole code
// identifier equal to the query up to case, and so folds to the folded query.
// The key is Fold(pattern).
std::optional<std::string> CodeTokenBoundKey(const SearchRequest& request);

// Returns the score of `file` as defined above.
int64_t ScoreFile(const FileResult& file);

// Whether a file with rank `next` and a boost of at most `max_boost` ranks
// strictly after `worst`. Equal scores must be verified because paths break
// ties.
bool CannotOutrank(const RankKey& next, int64_t max_boost,
                   const FileResult& worst);

// Whether `a` ranks ahead of `b` by tier, descending score, then path.
// Scores must already be set.
bool RanksBefore(const FileResult& a, const FileResult& b);

// Sets every file's score and sorts the files best first. Files equal in
// tier, score and path keep their input order.
void RankFiles(std::vector<FileResult>& files);

}  // namespace ferret

#endif  // FERRET_SEARCH_RANKING_H_

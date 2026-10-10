// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The static rank of a blob. Every segment assigns DocIds and ChunkIds in
// RankKey order, so candidates stream out best-first and the searcher can
// stop early without depending on how the index was built.
//
// The rank is a pure function of the blob's hint path (the path it was first
// indexed under), its size, whether it has long lines, and its Oid.

#ifndef FERRET_INDEX_RANK_KEY_H_
#define FERRET_INDEX_RANK_KEY_H_

#include <compare>
#include <cstdint>

#include "absl/strings/string_view.h"

#include "ferret/git/oid.h"

namespace ferret {

// What kind of file a blob is, best first. Persisted in .fseg: renumbering
// requires a new segment format version.
enum class RankTier : uint8_t {
  kSource = 0,         // First-party code.
  kTest = 1,           // Tests and test data.
  kDocumentation = 2,  // Prose: *.md, *.rst, README, LICENSE...
  kThirdParty = 3,     // Vendored code: third_party/, vendor/, node_modules/...
  kGenerated = 4,      // Generated code and lock files: *.pb.cc, *.min.js...
  kLongLines = 5,      // Files with a line over the long-line limit.
};

inline constexpr int kNumRankTiers = 6;

// Returns a stable lowercase name, such as "third_party".
absl::string_view RankTierName(RankTier tier);

// Returns the tier of the file at repo-relative `path`: the worst tier whose
// rule matches, or kLongLines if `long_lines` is set.
RankTier ClassifyTier(absl::string_view path, bool long_lines);

// Returns the prior of a file at repo-relative `path` with `size` bytes;
// higher is better. Shallower paths and smaller files score higher:
//   65535 - 4096 * min(depth, 8) - 2048 * min(max(bit_width(size) - 12, 0), 15)
// where depth is the number of '/' in `path`. Always at least 2047, so it
// leaves room below for per-query adjustments.
uint16_t ComputePrior(absl::string_view path, uint64_t size);

// A total order on blobs: by tier, then by descending prior, then by Oid.
// Two keys compare equal only for the same content at an equally ranked
// path.
struct RankKey {
  RankTier tier = RankTier::kSource;
  uint16_t prior = 0;
  Oid oid;

  friend bool operator==(const RankKey&, const RankKey&) = default;
  friend std::strong_ordering operator<=>(const RankKey& a, const RankKey& b) {
    if (a.tier != b.tier) return a.tier <=> b.tier;
    if (a.prior != b.prior) return b.prior <=> a.prior;
    return a.oid <=> b.oid;
  }
};

// Returns the RankKey of a blob with Oid `oid` and `size` bytes, first seen
// at `hint_path`.
RankKey ComputeRankKey(absl::string_view hint_path, uint64_t size,
                       bool long_lines, const Oid& oid);

}  // namespace ferret

#endif  // FERRET_INDEX_RANK_KEY_H_

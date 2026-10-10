// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Extracts the set of indexable trigrams of a piece of text.
//
// The indexable trigrams of t are the 3-byte windows of F(t) (fold.h) that
// contain no '\n'. This is the only index-side definition of that set: the
// segment writer, the memtable, the eligibility check, and the soundness
// property test all go through TrigramExtractor.

#ifndef FERRET_TEXT_TRIGRAM_EXTRACTOR_H_
#define FERRET_TEXT_TRIGRAM_EXTRACTOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"

#include "ferret/text/trigram.h"

namespace ferret {

// Deduplicates with a 2^24-bit bitmap (2 MiB) plus a "touched" list, so each
// call costs O(|text|) regardless of the bitmap size. Reuse one extractor per
// thread. Not thread-safe.
class TrigramExtractor final {
 public:
  TrigramExtractor();
  TrigramExtractor(const TrigramExtractor&) = delete;
  TrigramExtractor& operator=(const TrigramExtractor&) = delete;

  // Appends the distinct indexable trigrams of `text` to `out`, in order of
  // first occurrence. Trigrams already in `out` are not deduplicated against.
  void Extract(absl::string_view text, std::vector<Trigram>& out);

  // Appends the short-line keys of `text`: for each distinct line (bounded by
  // '\n' or the ends of `text`) whose folded image F(line) is exactly two bytes
  // "ab", the pseudo-trigram MakeTrigram('a', 'b', '\n'), in order of first
  // occurrence. Trigrams of Extract() never contain '\n', and lines of fewer
  // than three folded bytes have no indexable trigram, so these keys are what
  // lets a two-byte literal find a line made of just that literal. A
  // pseudo-trigram is stored in the posting lists like any other, and no query
  // planned from an RE2 tree contains one, so older readers ignore it. Not
  // counted by CountDistinct().
  //
  // It also appends, for every distinct pair of adjacent bytes "ab" of a line
  // as written (neither '\n') in which a or b is in A-Z, the pseudo-trigram
  // MakeTrigram('a', 'b', 0). F rewrites those bytes, so no folded trigram
  // says which case the text had; these keys let a case-sensitive two-byte
  // literal with a capital find its chunks exactly. A real trigram of folded
  // text has no A-Z, so the keys cannot collide with one.
  void ExtractShortLines(absl::string_view text, std::vector<Trigram>& out);

  // Returns the number of distinct indexable trigrams of `text`, stopping
  // early once the count exceeds `limit`, in which case it returns limit + 1.
  size_t CountDistinct(absl::string_view text, size_t limit);

 private:
  // Folds `text` into folded_ and calls `emit(trigram)` for every first
  // occurrence, stopping when `emit` returns false. Clears the bitmap
  // afterwards.
  template <typename Emit>
  void ForEachDistinct(absl::string_view text, Emit emit);

  std::unique_ptr<uint64_t[]> seen_;
  // One bit per two-byte line image, for ExtractShortLines().
  std::unique_ptr<uint64_t[]> seen_short_;
  std::vector<Trigram> touched_;
  std::string folded_;
};

// The trigrams whose posting lists, over a segment or memtable that indexes
// short lines (ExtractShortLines), jointly name every chunk with a line that
// contains the two folded bytes `a` `b`: "abX" and "Xab" for every byte X
// other than '\n' that F does not rewrite (A-Z), and the pseudo-trigram
// "ab\n". Every occurrence of "ab" in a line of at least three folded bytes
// lies in a window "abX" or "Xab" of that line, and a window never crosses a
// line, so it never crosses a chunk. Empty, meaning no such guarantee, if a
// or b is '\n'. Sorted and unique.
std::vector<Trigram> TrigramsCoveringBigram(uint8_t a, uint8_t b);

// The raw-bigram pseudo-trigram of the bytes `a` `b` as written, one of which
// must be in A-Z (see ExtractShortLines): the posting list of this trigram
// names exactly the chunks that contain "ab" in a line.
inline Trigram RawBigramTrigram(uint8_t a, uint8_t b) {
  return MakeTrigram(a, b, 0);
}

}  // namespace ferret

#endif  // FERRET_TEXT_TRIGRAM_EXTRACTOR_H_

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/trigram_extractor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/strings/string_view.h"

#include "ferret/text/fold.h"
#include "ferret/text/trigram.h"

namespace ferret {
namespace {

constexpr size_t kBitmapWords = (size_t{1} << 24) / 64;
constexpr size_t kShortBitmapWords = (size_t{1} << 16) / 64;
// F maps at most 3 bytes to 1, so a line whose image has 2 bytes has at most
// 6 bytes.
constexpr size_t kMaxRawBytesOfShortLine = 6;

}  // namespace

TrigramExtractor::TrigramExtractor()
    : seen_(std::make_unique<uint64_t[]>(kBitmapWords)),
      seen_short_(std::make_unique<uint64_t[]>(kShortBitmapWords)) {}

template <typename Emit>
void TrigramExtractor::ForEachDistinct(absl::string_view text, Emit emit) {
  folded_.clear();
  AppendFolded(text, folded_);
  const size_t n = folded_.size();
  for (size_t i = 0; i + 3 <= n; ++i) {
    const Trigram trigram = TrigramAt(folded_, i);
    if (TrigramHasNewline(trigram)) continue;
    uint64_t& word = seen_[trigram >> 6];
    const uint64_t bit = uint64_t{1} << (trigram & 63);
    if ((word & bit) != 0) continue;
    word |= bit;
    touched_.push_back(trigram);
    if (!emit(trigram)) break;
  }
  for (const Trigram trigram : touched_) {
    seen_[trigram >> 6] &= ~(uint64_t{1} << (trigram & 63));
  }
  touched_.clear();
}

void TrigramExtractor::Extract(absl::string_view text,
                               std::vector<Trigram>& out) {
  ForEachDistinct(text, [&out](Trigram trigram) {
    out.push_back(trigram);
    return true;
  });
}

void TrigramExtractor::ExtractShortLines(absl::string_view text,
                                         std::vector<Trigram>& out) {
  const size_t first_new = out.size();
  size_t begin = 0;
  while (begin < text.size()) {
    const void* newline =
        std::memchr(text.data() + begin, '\n', text.size() - begin);
    const size_t end =
        newline == nullptr
            ? text.size()
            : static_cast<size_t>(static_cast<const char*>(newline) -
                                  text.data());
    const size_t length = end - begin;
    if (length >= 2 && length <= kMaxRawBytesOfShortLine) {
      uint8_t image[2];
      bool two_bytes = false;
      if (length == 2) {
        const auto b0 = static_cast<uint8_t>(text[begin]);
        const auto b1 = static_cast<uint8_t>(text[begin + 1]);
        // C5 BF folds to the single byte 's'.
        two_bytes = !(b0 == 0xC5 && b1 == 0xBF);
        image[0] = FoldAsciiByte(b0);
        image[1] = FoldAsciiByte(b1);
      } else {
        folded_.clear();
        AppendFolded(text.substr(begin, length), folded_);
        two_bytes = folded_.size() == 2;
        if (two_bytes) {
          image[0] = static_cast<uint8_t>(folded_[0]);
          image[1] = static_cast<uint8_t>(folded_[1]);
        }
      }
      if (two_bytes) {
        const unsigned key = (unsigned{image[0]} << 8) | image[1];
        uint64_t& word = seen_short_[key >> 6];
        const uint64_t bit = uint64_t{1} << (key & 63);
        if ((word & bit) == 0) {
          word |= bit;
          out.push_back(MakeTrigram(image[0], image[1], '\n'));
        }
      }
    }
    begin = end + 1;
  }
  for (size_t i = first_new; i < out.size(); ++i) {
    const Trigram key = out[i] >> 8;
    seen_short_[key >> 6] &= ~(uint64_t{1} << (key & 63));
  }
  // Raw bigrams with an upper-case letter: every adjacent pair of bytes of a
  // line, as written, in which F rewrites a byte.
  const size_t first_raw = out.size();
  for (size_t i = 0; i + 1 < text.size(); ++i) {
    const auto a = static_cast<uint8_t>(text[i]);
    const auto b = static_cast<uint8_t>(text[i + 1]);
    if (a == '\n' || b == '\n') continue;
    if (!((a >= 'A' && a <= 'Z') || (b >= 'A' && b <= 'Z'))) continue;
    const unsigned key = (unsigned{a} << 8) | b;
    uint64_t& word = seen_short_[key >> 6];
    const uint64_t bit = uint64_t{1} << (key & 63);
    if ((word & bit) == 0) {
      word |= bit;
      out.push_back(MakeTrigram(a, b, 0));
    }
  }
  for (size_t i = first_raw; i < out.size(); ++i) {
    const Trigram key = out[i] >> 8;
    seen_short_[key >> 6] &= ~(uint64_t{1} << (key & 63));
  }
}

size_t TrigramExtractor::CountDistinct(absl::string_view text, size_t limit) {
  size_t count = 0;
  ForEachDistinct(
      text, [&count, limit](Trigram /*trigram*/) { return ++count <= limit; });
  return count;
}

std::vector<Trigram> TrigramsCoveringBigram(uint8_t a, uint8_t b) {
  std::vector<Trigram> out;
  if (a == '\n' || b == '\n') return out;
  out.reserve(size_t{2} * 256);
  for (unsigned x = 0; x < 256; ++x) {
    // Folded text has no A-Z, and no window of an indexed line has '\n'.
    if (x == '\n' || (x >= 'A' && x <= 'Z')) continue;
    out.push_back(MakeTrigram(a, b, static_cast<uint8_t>(x)));
    out.push_back(MakeTrigram(static_cast<uint8_t>(x), a, b));
  }
  out.push_back(MakeTrigram(a, b, '\n'));
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

}  // namespace ferret

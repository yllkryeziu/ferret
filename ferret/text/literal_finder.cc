// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/literal_finder.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"

#include "ferret/text/fold.h"
#include "ferret/text/internal/literal_finder_kernels.h"

namespace ferret {
namespace {

// Occurrences of each byte value per 10^9 bytes of source code. Measured on
// every 8th tracked Linux file (fe2ec83746e5; 11,971 files, 140 MiB) and every
// 16th tracked Chromium file (d7bd2de3cbd6; 28,852 files, 155 MiB), skipping
// files over 1 MiB or containing NUL, with the two corpora weighted equally.
// Only the order matters, so the exact values are not load-bearing.
constexpr std::array<uint32_t, 256> kByteFrequency = {
    0,         34,       22,       15,        // 0x00
    25,        9,        12,       12,        // 0x04
    3,         18173167, 28286662, 0,         // 0x08
    73,        122893,   6,        6,         // 0x0C
    9,         6,        9,        0,         // 0x10
    6,         0,        3,        3,         // 0x14
    9,         9,        0,        3,         // 0x18
    3,         0,        0,        9,         // 0x1C
    137287659, 514470,   8264247,  2102492,   // 0x20
    211353,    375460,   1504591,  1343951,   // 0x24
    9661591,   9668682,  5391363,  865112,    // 0x28
    11059414,  6468616,  7172289,  7870952,   // 0x2C
    13579712,  6453283,  5407891,  3996800,   // 0x30
    3339636,   2932581,  3083748,  2554920,   // 0x34
    3353610,   2527017,  6589843,  8015255,   // 0x38
    3247054,   6536615,  5453981,  117898,    // 0x3C
    388607,    7038169,  2920337,  7514352,   // 0x40
    5247321,   9109363,  3687413,  3023541,   // 0x44
    2086472,   6870926,  208478,   1368379,   // 0x48
    5060203,   4737399,  5048104,  4758458,   // 0x4C
    5608563,   726274,   6707533,  7964773,   // 0x50
    8309390,   3167809,  1826373,  1541555,   // 0x54
    1593267,   913749,   368579,   1316061,   // 0x58
    2377979,   1314957,  32200,    32560380,  // 0x5C
    374292,    33946818, 8485749,  22876094,  // 0x60
    22113921,  60186866, 12670784, 9597698,   // 0x64
    10795896,  35948447, 792561,   5171231,   // 0x68
    21566529,  13927316, 33159190, 30956902,  // 0x6C
    15805866,  1546227,  35614583, 33359260,  // 0x70
    47652285,  16847172, 7362298,  4907649,   // 0x74
    7752314,   5435242,  1213276,  2770573,   // 0x78
    613185,    2770084,  104868,   19,        // 0x7C
    298710,    283581,   219921,   256876,    // 0x80
    81591,     84663,    79475,    146114,    // 0x84
    179034,    101631,   70446,    169071,    // 0x88
    150577,    258531,   25108,    60693,     // 0x8C
    51222,     37530,    32533,    43164,     // 0x90
    73922,     198653,   55046,    62167,     // 0x94
    48105,     80286,    71882,    59128,     // 0x98
    59879,     34952,    34340,    136740,    // 0x9C
    70609,     130368,   33698,    50914,     // 0xA0
    734025,    270353,   420757,   250534,    // 0xA4
    181728,    80575,    248703,   126157,    // 0xA8
    69386,     99273,    161239,   155049,    // 0xAC
    316436,    107828,   122865,   93710,     // 0xB0
    192030,    203572,   66822,    62242,     // 0xB4
    189687,    98861,    236281,   147181,    // 0xB8
    91566,     127584,   274833,   148558,    // 0xBC
    154,       342,      21268,    223820,    // 0xC0
    55819,     40282,    2837,     160,       // 0xC4
    214,       9188,     200,      210,       // 0xC8
    20781,     173,      115350,   63071,     // 0xCC
    688430,    349866,   25618,    12314,     // 0xD0
    1856,      131948,   31116,    76969,     // 0xD4
    160234,    101493,   30382,    43359,     // 0xD8
    148,       136,      176,      166,       // 0xDC
    2048725,   482454,   64022,    230036,    // 0xE0
    70011,     149572,   95632,    84144,     // 0xE4
    64601,     44095,    1650,     3234,      // 0xE8
    4875,      1884,     123,      23591,     // 0xEC
    54014,     123,      157,      160,       // 0xF0
    170,       126,      126,      145,       // 0xF4
    133,       114,      654,      694,       // 0xF8
    166,       18,       166,      49,        // 0xFC
};

// How often the filter accepts byte `b` of a needle: a letter of a folded
// needle matches either case.
uint64_t FilterFrequency(uint8_t b, bool fold_ascii) {
  uint64_t frequency = kByteFrequency[b];
  if (fold_ascii && b >= 'a' && b <= 'z') {
    frequency += kByteFrequency[b - ('a' - 'A')];
  }
  return frequency;
}

std::string FoldAscii(absl::string_view text) {
  std::string folded(text);
  for (char& c : folded) {
    c = static_cast<char>(FoldAsciiByte(static_cast<uint8_t>(c)));
  }
  return folded;
}

}  // namespace

LiteralFinder::LiteralFinder(absl::string_view needle, Mode mode)
    : needle_(mode == Mode::kAsciiCaseInsensitive ? FoldAscii(needle)
                                                  : std::string(needle)),
      mode_(mode) {
  if (needle_.size() < 2) return;
  const bool fold_ascii = mode == Mode::kAsciiCaseInsensitive;
  auto frequency = [&](size_t i) {
    return FilterFrequency(static_cast<uint8_t>(needle_[i]), fold_ascii);
  };
  for (size_t i = 1; i < needle_.size(); ++i) {
    if (frequency(i) < frequency(rare1_offset_)) rare1_offset_ = i;
  }
  // The second filter byte is the rarest at another offset, preferring a
  // value different from the first: two equal bytes pass the filter together
  // far more often than their frequencies suggest.
  auto second_key = [&](size_t i) {
    return std::pair(needle_[i] == needle_[rare1_offset_], frequency(i));
  };
  rare2_offset_ = rare1_offset_ == 0 ? 1 : 0;
  for (size_t i = 0; i < needle_.size(); ++i) {
    if (i != rare1_offset_ && second_key(i) < second_key(rare2_offset_)) {
      rare2_offset_ = i;
    }
  }
}

size_t LiteralFinder::Find(absl::string_view haystack, size_t from) const {
  if (needle_.empty()) {
    return from <= haystack.size() ? from : absl::string_view::npos;
  }
  const text_internal::LiteralNeedle needle = {
      .bytes = needle_,
      .fold_ascii = mode_ == Mode::kAsciiCaseInsensitive,
      .rare1_offset = rare1_offset_,
      .rare2_offset = rare2_offset_,
  };
#if defined(__aarch64__)
  return text_internal::FindLiteralNeon(needle, haystack, from);
#else
  return text_internal::FindLiteralScalar(needle, haystack, from);
#endif
}

}  // namespace ferret

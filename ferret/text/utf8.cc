// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/utf8.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"

namespace ferret {
namespace {

constexpr absl::string_view kReplacementCharacter = "\xEF\xBF\xBD";  // U+FFFD
constexpr absl::string_view kEllipsis = "\xE2\x80\xA6";              // U+2026

bool IsContinuationByte(char c) {
  return (static_cast<uint8_t>(c) & 0xC0) == 0x80;
}

// Returns the length of the longest well-formed prefix of `text`.
size_t ValidPrefixLength(absl::string_view text) {
  size_t i = 0;
  while (i < text.size()) {
    // Skip ASCII eight bytes at a time.
    uint64_t word;
    if (text.size() - i >= sizeof(word)) {
      std::memcpy(&word, text.data() + i, sizeof(word));
      if ((word & 0x8080808080808080u) == 0) {
        i += sizeof(word);
        continue;
      }
    }
    const Utf8Sequence sequence = DecodeUtf8(text.substr(i));
    if (sequence.code_point < 0) break;
    i += sequence.length;
  }
  return i;
}

// The size of `sequence` after sanitizing.
size_t DisplaySize(const Utf8Sequence& sequence) {
  return sequence.code_point < 0 ? kReplacementCharacter.size()
                                 : sequence.length;
}

// Sequences (well-formed ones and maximal subparts of ill-formed ones) tile a
// text, and only the first byte of a sequence can be a non-continuation byte,
// so decoding from any non-continuation byte stays in step with decoding from
// the start. This returns the start of the sequence that contains text[pos].
size_t SequenceStart(absl::string_view text, size_t pos) {
  DCHECK_LT(pos, text.size());
  // A sequence spans at most four bytes.
  const size_t lowest = pos >= 3 ? pos - 3 : 0;
  for (size_t start = pos;; --start) {
    if (!IsContinuationByte(text[start])) {
      // A continuation byte past the end of the sequence at `start` is a
      // one-byte ill-formed sequence of its own.
      return start + DecodeUtf8(text.substr(start)).length > pos ? start : pos;
    }
    if (start == lowest) return pos;
  }
}

// A window [begin, end) of whole sequences of a line, grown one sequence at a
// time while its display form, including an ellipsis for each cut end, fits
// the budget.
class DisplayWindow {
 public:
  DisplayWindow(absl::string_view line, size_t start, size_t max_bytes)
      : line_(line), max_bytes_(max_bytes), begin_(start), end_(start) {}
  DisplayWindow(const DisplayWindow&) = delete;
  DisplayWindow& operator=(const DisplayWindow&) = delete;

  size_t begin() const { return begin_; }
  size_t end() const { return end_; }

  // Adds the sequence before begin() if it fits.
  bool GrowLeft() {
    if (begin_ == 0) return false;
    const size_t start = SequenceStart(line_, begin_ - 1);
    const size_t size = DisplaySize(DecodeUtf8(line_.substr(start)));
    if (!Fits(content_bytes_ + size, start, end_)) return false;
    content_bytes_ += size;
    left_bytes_ += size;
    begin_ = start;
    return true;
  }

  // Adds the sequence at end() if it fits. `context` says whether it counts
  // as context after the match rather than as part of the match.
  bool GrowRight(bool context) {
    if (end_ == line_.size()) return false;
    const Utf8Sequence sequence = DecodeUtf8(line_.substr(end_));
    const size_t size = DisplaySize(sequence);
    if (!Fits(content_bytes_ + size, begin_, end_ + sequence.length)) {
      return false;
    }
    content_bytes_ += size;
    if (context) right_bytes_ += size;
    end_ += sequence.length;
    return true;
  }

  // Grows whichever side of the match has less context, falling back to the
  // other side. Returns false once neither side can grow.
  bool GrowBalanced() {
    if (left_bytes_ <= right_bytes_) return GrowLeft() || GrowRight(true);
    return GrowRight(true) || GrowLeft();
  }

  void AppendTo(std::string& out) const {
    if (begin_ > 0) out.append(kEllipsis);
    AppendSanitizedUtf8(line_.substr(begin_, end_ - begin_), out);
    if (end_ < line_.size()) out.append(kEllipsis);
  }

 private:
  bool Fits(size_t content_bytes, size_t begin, size_t end) const {
    const size_t markers = (begin > 0 ? kEllipsis.size() : 0) +
                           (end < line_.size() ? kEllipsis.size() : 0);
    return content_bytes + markers <= max_bytes_;
  }

  const absl::string_view line_;
  const size_t max_bytes_;
  size_t begin_;
  size_t end_;
  // Display bytes in the window, and the parts of them before and after the
  // match.
  size_t content_bytes_ = 0;
  size_t left_bytes_ = 0;
  size_t right_bytes_ = 0;
};

}  // namespace

Utf8Sequence DecodeUtf8(absl::string_view text) {
  DCHECK(!text.empty());
  const uint8_t lead = static_cast<uint8_t>(text[0]);
  if (lead < 0x80) return {.code_point = lead, .length = 1};
  // Table 3-7: the number of continuation bytes, and the range allowed for
  // the first of them, which is what excludes overlong forms, surrogates, and
  // code points above U+10FFFF.
  size_t trailing;
  uint8_t low = 0x80;
  uint8_t high = 0xBF;
  int32_t code_point;
  if (lead >= 0xC2 && lead <= 0xDF) {
    trailing = 1;
    code_point = lead & 0x1F;
  } else if (lead >= 0xE0 && lead <= 0xEF) {
    trailing = 2;
    code_point = lead & 0x0F;
    if (lead == 0xE0) low = 0xA0;
    if (lead == 0xED) high = 0x9F;
  } else if (lead >= 0xF0 && lead <= 0xF4) {
    trailing = 3;
    code_point = lead & 0x07;
    if (lead == 0xF0) low = 0x90;
    if (lead == 0xF4) high = 0x8F;
  } else {
    return {.code_point = -1, .length = 1};
  }
  for (size_t i = 1; i <= trailing; ++i) {
    if (i == text.size()) return {.code_point = -1, .length = i};
    const uint8_t byte = static_cast<uint8_t>(text[i]);
    if (byte < low || byte > high) return {.code_point = -1, .length = i};
    code_point = (code_point << 6) | (byte & 0x3F);
    low = 0x80;
    high = 0xBF;
  }
  return {.code_point = code_point, .length = trailing + 1};
}

bool IsValidUtf8(absl::string_view text) {
  return ValidPrefixLength(text) == text.size();
}

void AppendSanitizedUtf8(absl::string_view text, std::string& out) {
  while (true) {
    const size_t valid = ValidPrefixLength(text);
    out.append(text.substr(0, valid));
    if (valid == text.size()) return;
    out.append(kReplacementCharacter);
    text.remove_prefix(valid + DecodeUtf8(text.substr(valid)).length);
  }
}

void AppendDisplayLine(absl::string_view line, size_t match_begin,
                       size_t match_end, size_t max_bytes, std::string& out) {
  DCHECK_LE(match_begin, match_end);
  DCHECK_LE(match_end, line.size());
  DCHECK_GE(max_bytes, 2 * kEllipsis.size());
  if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
  match_end = std::min(match_end, line.size());
  match_begin = std::min(match_begin, match_end);
  // Sanitizing never shrinks a text and at most triples it, since each
  // replaced byte becomes three.
  if (line.size() <= max_bytes / kReplacementCharacter.size()) {
    AppendSanitizedUtf8(line, out);
    return;
  }
  if (line.size() <= max_bytes) {
    size_t display_bytes = 0;
    for (size_t i = 0; i < line.size();) {
      const Utf8Sequence sequence = DecodeUtf8(line.substr(i));
      display_bytes += DisplaySize(sequence);
      i += sequence.length;
    }
    if (display_bytes <= max_bytes) {
      AppendSanitizedUtf8(line, out);
      return;
    }
  }
  const size_t start = match_begin < line.size()
                           ? SequenceStart(line, match_begin)
                           : line.size();
  DisplayWindow window(line, start, max_bytes);
  while (window.end() < match_end && window.GrowRight(/*context=*/false)) {
  }
  while (window.GrowBalanced()) {
  }
  window.AppendTo(out);
}

}  // namespace ferret

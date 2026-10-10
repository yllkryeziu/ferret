// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
#include "ferret/search/kept_lines.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "ferret/search/search_types.h"
#include "ferret/text/line_counter.h"

namespace ferret {

// Returns the kept lines of a document with `num_lines` lines and their
// context: every line within `context_lines` of a kept line, in order and
// without repeats. `kept` is in increasing line order.
std::vector<ResultLine> WithContext(absl::string_view content,
                                    int64_t num_lines,
                                    absl::Span<const KeptLine> kept,
                                    int64_t context_lines) {
  std::vector<ResultLine> lines;
  // The last line appended, by number and by the offset of its end.
  int64_t last_number = 0;
  size_t last_end = 0;
  auto append_context_line = [&](int64_t number, size_t begin, size_t end) {
    lines.push_back(
        ResultLine{.number = number,
                   .text = std::string(content.substr(begin, end - begin))});
  };
  // Appends the lines after the last one appended, through line `limit`.
  auto extend_through = [&](int64_t limit) {
    while (last_number < limit) {
      const size_t begin = last_end + 1;
      const size_t end = LineEnd(content, begin);
      append_context_line(++last_number, begin, end);
      last_end = end;
    }
  };

  for (const KeptLine& line : kept) {
    if (!lines.empty()) {
      // The context after the previous kept line.
      extend_through(std::min(last_number + context_lines, line.number - 1));
    }
    // The context before this one, found by walking back from it.
    const int64_t first =
        std::max(line.number - context_lines, last_number + 1);
    const size_t mark = lines.size();
    size_t begin = line.begin;
    for (int64_t number = line.number - 1; number >= first; --number) {
      const size_t end = begin - 1;
      begin = LineBegin(content, end);
      append_context_line(number, begin, end);
    }
    std::reverse(lines.begin() + static_cast<ptrdiff_t>(mark), lines.end());
    lines.push_back(ResultLine{
        .number = line.number,
        .text = std::string(content.substr(line.begin, line.end - line.begin)),
        .is_match = true,
        .match_begin = line.match_begin,
        .match_end = line.match_end,
        .match_class = line.match_class});
    last_number = line.number;
    last_end = line.end;
  }
  if (!lines.empty()) {
    extend_through(std::min(last_number + context_lines, num_lines));
  }
  return lines;
}

}  // namespace ferret

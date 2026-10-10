// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Renders a search response as text for an agent. Pure and thread-safe;
// every backend's responses go through here.
//
// Shapes:
//   kRaw        "path:line:text" for each kept matching line, files in path
//               order: what an agent gets from grep.
//   kLine       The same lines, files in rank order.
//   kContext    Files in rank order under a count header, in ripgrep's
//               heading style: the path, then "12:text" for a matching line,
//               "13-text" for context, "--" between runs of lines, and
//               "+N more matching lines" when the file has more.
//   kLocations  "path:line" for each kept matching line, in rank order.
//
// Output is valid UTF-8: text and paths are sanitized (utf8.h), and each
// line of text is cut to a window around its match. Output stops at a line
// boundary before `max_bytes` and ends with a trailer that says how many
// matching lines were not shown and, for the ranked shapes, how to narrow
// the search. An incomplete search says so, too.

#ifndef FERRET_SEARCH_FORMATTER_H_
#define FERRET_SEARCH_FORMATTER_H_

#include <cstddef>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "ferret/search/search_types.h"

namespace ferret {

enum class OutputShape { kRaw, kLine, kContext, kLocations };

// "raw", "line", "context" or "locations".
absl::string_view OutputShapeName(OutputShape shape);
// The inverse of OutputShapeName(); InvalidArgument for any other name.
absl::StatusOr<OutputShape> ParseOutputShape(absl::string_view name);

struct FormatOptions {
  OutputShape shape = OutputShape::kContext;
  // The output before the trailer stays within this many bytes. The trailer
  // adds at most two lines of under 200 bytes each.
  size_t max_bytes = 8 * 1024;
  // Each line of text is cut to this many bytes, markers included. At least
  // 6.
  size_t max_line_bytes = 256;
};

// Renders `response`, whose files must be in rank order (RankFiles()).
std::string FormatSearchResponse(const SearchResponse& response,
                                 const FormatOptions& options);

}  // namespace ferret

#endif  // FERRET_SEARCH_FORMATTER_H_

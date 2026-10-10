// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/search/formatter.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "ferret/search/search_types.h"
#include "ferret/text/utf8.h"

namespace ferret {
namespace {

// Collects output lines until the first one that does not fit the budget,
// and counts the matching lines shown.
class OutputWriter final {
 public:
  explicit OutputWriter(size_t max_bytes) : max_bytes_(max_bytes) {}

  // Appends the lines, each followed by '\n', if all of them fit; otherwise
  // appends nothing, now or later, and returns false.
  bool Append(std::initializer_list<absl::string_view> lines) {
    if (full_) return false;
    size_t bytes = 0;
    for (const absl::string_view line : lines) bytes += line.size() + 1;
    if (out_.size() + bytes > max_bytes_) {
      full_ = true;
      return false;
    }
    for (const absl::string_view line : lines) {
      absl::StrAppend(&out_, line, "\n");
    }
    return true;
  }

  std::string& out() { return out_; }

 private:
  const size_t max_bytes_;
  std::string out_;
  bool full_ = false;
};

// The matching lines accounted for by the output, shown or announced by a
// "+N more" line, so that the trailer reports only the rest.
struct Accounted {
  int64_t matching_lines = 0;
  // Files all of whose matching lines are accounted for.
  int64_t complete_files = 0;

  void AddFile(const FileResult& file, int64_t lines) {
    matching_lines += lines;
    if (lines == file.num_matching_lines) ++complete_files;
  }
};

// "1 file", "2 files".
std::string Count(int64_t n, absl::string_view noun) {
  return absl::StrCat(n, " ", noun, n == 1 ? "" : "s");
}

std::string DisplayPath(absl::string_view path) {
  std::string out;
  AppendSanitizedUtf8(path, out);
  return out;
}

std::string DisplayText(const ResultLine& line, size_t max_line_bytes) {
  std::string out;
  if (line.is_match) {
    AppendDisplayLine(line.text, line.match_begin, line.match_end,
                      max_line_bytes, out);
  } else {
    AppendDisplayLine(line.text, 0, 0, max_line_bytes, out);
  }
  return out;
}

std::string Percent(double fraction) {
  return absl::StrCat(static_cast<int>(fraction * 100), "%");
}

// One line per kept matching line: "path:line:text", or "path:line" without
// text.
Accounted FormatFlat(const std::vector<const FileResult*>& files,
                     bool with_text, const FormatOptions& options,
                     OutputWriter& writer) {
  Accounted accounted;
  for (const FileResult* file : files) {
    const std::string path = DisplayPath(file->path);
    int64_t file_shown = 0;
    for (const ResultLine& line : file->lines) {
      if (!line.is_match) continue;
      const std::string entry =
          with_text ? absl::StrCat(path, ":", line.number, ":",
                                   DisplayText(line, options.max_line_bytes))
                    : absl::StrCat(path, ":", line.number);
      if (!writer.Append({entry})) break;
      ++file_shown;
    }
    accounted.AddFile(*file, file_shown);
  }
  return accounted;
}

Accounted FormatContext(const SearchResponse& response,
                        const FormatOptions& options, OutputWriter& writer) {
  std::string header =
      absl::StrCat(Count(response.num_matching_lines, "matching line"), " in ",
                   Count(response.num_matching_files, "file"));
  if (!response.complete) {
    absl::StrAppend(&header, " (incomplete: verified ",
                    Percent(response.verified_fraction),
                    " of candidate chunks)");
  }
  Accounted accounted;
  if (!writer.Append({header})) return accounted;
  for (const FileResult& file : response.files) {
    int64_t file_shown = 0;
    int64_t previous_number = 0;
    for (const ResultLine& line : file.lines) {
      const std::string entry =
          absl::StrCat(line.number, line.is_match ? ":" : "-",
                       DisplayText(line, options.max_line_bytes));
      // The path goes out with the file's first line, never alone.
      const bool appended =
          previous_number == 0
              ? writer.Append({"", DisplayPath(file.path), entry})
          : line.number == previous_number + 1 ? writer.Append({entry})
                                               : writer.Append({"--", entry});
      if (!appended) break;
      previous_number = line.number;
      file_shown += line.is_match;
    }
    const int64_t kept =
        std::count_if(file.lines.begin(), file.lines.end(),
                      [](const ResultLine& line) { return line.is_match; });
    if (file_shown == kept && kept < file.num_matching_lines &&
        writer.Append({absl::StrCat("+", Count(file.num_matching_lines - kept,
                                               "more matching line"))})) {
      file_shown = file.num_matching_lines;
    }
    accounted.AddFile(file, file_shown);
  }
  return accounted;
}

}  // namespace

absl::string_view OutputShapeName(OutputShape shape) {
  switch (shape) {
    case OutputShape::kRaw:
      return "raw";
    case OutputShape::kLine:
      return "line";
    case OutputShape::kContext:
      return "context";
    case OutputShape::kLocations:
      return "locations";
  }
  return "unknown";
}

absl::StatusOr<OutputShape> ParseOutputShape(absl::string_view name) {
  for (const OutputShape shape :
       {OutputShape::kRaw, OutputShape::kLine, OutputShape::kContext,
        OutputShape::kLocations}) {
    if (name == OutputShapeName(shape)) return shape;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("unknown output shape \"", name,
                   "\"; expected raw, line, context or locations"));
}

std::string FormatSearchResponse(const SearchResponse& response,
                                 const FormatOptions& options) {
  CHECK_GE(options.max_line_bytes, size_t{6});
  OutputWriter writer(options.max_bytes);
  std::vector<const FileResult*> files;
  files.reserve(response.files.size());
  for (const FileResult& file : response.files) files.push_back(&file);

  Accounted accounted;
  switch (options.shape) {
    case OutputShape::kRaw:
      // Sorted by path (unique within a response), never by address.
      // NOLINTNEXTLINE(bugprone-nondeterministic-pointer-iteration-order): Key.
      std::sort(files.begin(), files.end(),
                [](const FileResult* a, const FileResult* b) {
                  return a->path < b->path;
                });
      accounted = FormatFlat(files, /*with_text=*/true, options, writer);
      break;
    case OutputShape::kLine:
      accounted = FormatFlat(files, /*with_text=*/true, options, writer);
      break;
    case OutputShape::kLocations:
      accounted = FormatFlat(files, /*with_text=*/false, options, writer);
      break;
    case OutputShape::kContext:
      accounted = FormatContext(response, options, writer);
      break;
  }

  std::string& out = writer.out();
  const int64_t hidden_lines =
      response.num_matching_lines - accounted.matching_lines;
  const int64_t hidden_files =
      response.num_matching_files - accounted.complete_files;
  if (hidden_lines > 0) {
    if (options.shape == OutputShape::kRaw) {
      absl::StrAppend(&out, "[truncated: ", Count(hidden_lines, "more line"),
                      " in ", Count(hidden_files, "file"), "]\n");
    } else {
      absl::StrAppend(&out, "[", Count(hidden_lines, "more matching line"),
                      " in ", Count(hidden_files, "file"),
                      " not shown; narrow the search with a path glob or a "
                      "more specific pattern]\n");
    }
  }
  if (!response.complete && options.shape != OutputShape::kContext) {
    absl::StrAppend(&out, "[incomplete: the search verified ",
                    Percent(response.verified_fraction),
                    " of candidate chunks; counts cover only those]\n");
  }
  return std::move(out);
}

}  // namespace ferret

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
#include "ferret/search/structural_response.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "ferret/search/formatter.h"
#include "ferret/search/search_types.h"
#include "ferret/text/lexer.h"
#include "ferret/text/utf8.h"
#include "ferret/workspace/workspace.h"

namespace ferret {
namespace {
std::string Display(absl::string_view value) {
  std::string text;
  AppendSanitizedUtf8(value, text);
  return text;
}

// A code identifier followed by whitespace and an opening parenthesis.
bool IsCall(const ResultLine& line) {
  size_t next = line.match_end;
  while (next < line.text.size() &&
         (line.text[next] == ' ' || line.text[next] == '\t')) {
    ++next;
  }
  return next < line.text.size() && line.text[next] == '(';
}

struct ReferenceFile {
  const FileResult* file;
  int64_t calls;
  std::vector<const ResultLine*> lines;
};

void AppendBudgeted(std::string& output, const std::string& text,
                    size_t max_bytes, bool& truncated) {
  if (output.size() + text.size() <= max_bytes) {
    output += text;
  } else {
    truncated = true;
  }
}
}  // namespace

std::string FormatStructuralResponse(const SearchResponse& search,
                                     const FindSymbolResponse& definitions,
                                     const SearchRequest& request,
                                     const FormatOptions& options) {
  std::vector<ReferenceFile> references;
  for (const FileResult& file : search.files) {
    ReferenceFile entry{.file = &file, .calls = 0};
    for (const ResultLine& line : file.lines) {
      if (!line.is_match || line.match_class != MatchClass::kCodeIdentifier) {
        continue;
      }
      entry.lines.push_back(&line);
      entry.calls += IsCall(line);
    }
    if (!entry.lines.empty()) references.push_back(std::move(entry));
  }
  std::sort(references.begin(), references.end(),
            [](const ReferenceFile& a, const ReferenceFile& b) {
              return a.calls != b.calls ? a.calls > b.calls
                                        : a.file->path < b.file->path;
            });
  const std::string header = absl::StrCat(
      "Structural route: ", search.counts_estimated ? "~" : "",
      search.num_matching_lines, " text matches in ",
      search.counts_estimated ? "~" : "", search.num_matching_files, " files",
      search.counts_estimated ? " (estimated, sample " : "",
      search.counts_estimated ? absl::StrCat(search.count_sample_size, ")")
                              : "",
      "; token_precision ", search.token_precision, "; ",
      definitions.total_count, " definitions",
      definitions.pending_files > 0
          ? absl::StrCat(" (", definitions.pending_files,
                         " symbol files pending)")
          : "",
      "; reference ranking ",
      search.terminated_early || !search.complete || search.counts_estimated ||
              search.num_matching_files > std::ssize(search.files)
          ? "incomplete (verified text results only)"
          : "complete",
      ".\n");
  std::string output;
  bool truncated = false;
  AppendBudgeted(output, header, options.max_bytes, truncated);
  AppendBudgeted(output, "Definitions:\n", options.max_bytes, truncated);
  for (const SymbolLocation& location : definitions.occurrences) {
    const std::string definition =
        absl::StrCat(Display(location.path), ":", location.line, ":",
                     Display(location.source_line), "\n");
    std::string before;
    std::string after;
    for (const FileResult& file : search.files) {
      if (file.path != location.path) continue;
      for (const ResultLine& line : file.lines) {
        if (line.number <
                static_cast<int64_t>(location.line) - request.context_lines ||
            line.number >
                static_cast<int64_t>(location.line) + request.context_lines ||
            line.number == location.line) {
          continue;
        }
        std::string& context = line.number < location.line ? before : after;
        absl::StrAppend(&context, "  ", line.number, ":", Display(line.text),
                        "\n");
      }
      break;
    }
    AppendBudgeted(output, absl::StrCat(before, definition, after),
                   options.max_bytes, truncated);
  }
  AppendBudgeted(output, "References:\n", options.max_bytes, truncated);
  for (const ReferenceFile& entry : references) {
    std::string lines = absl::StrCat(Display(entry.file->path), " (",
                                     entry.calls, " call sites):\n");
    for (const ResultLine* line : entry.lines) {
      std::string text;
      AppendDisplayLine(line->text, line->match_begin, line->match_end,
                        options.max_line_bytes, text);
      absl::StrAppend(&lines, "  ", line->number, ":", text, "\n");
    }
    AppendBudgeted(output, lines, options.max_bytes, truncated);
  }
  if (truncated) {
    const std::string trailer = "[truncated: narrow with path_glob]\n";
    if (options.max_bytes >= trailer.size()) {
      const size_t boundary =
          std::min(output.size(), options.max_bytes - trailer.size());
      const size_t newline =
          boundary == 0 ? std::string::npos : output.rfind('\n', boundary - 1);
      output.resize(newline == std::string::npos ? 0 : newline + 1);
      output += trailer;
    }
  }
  return output;
}

}  // namespace ferret

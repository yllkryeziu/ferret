// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"

#include "ferret/cli/command_util.h"
#include "ferret/cli/commands.h"
#include "ferret/git/path_glob.h"
#include "ferret/index/segment.h"
#include "ferret/search/formatter.h"
#include "ferret/search/router.h"
#include "ferret/search/search_types.h"
#include "ferret/search/searcher.h"
#include "ferret/search/structural_response.h"
#include "ferret/text/line_counter.h"
#include "ferret/workspace/workspace.h"

namespace ferret {
namespace {

std::string Millis(absl::Duration d) {
  return absl::FormatDuration(absl::Trunc(d, absl::Microseconds(10)));
}

}  // namespace

absl::StatusOr<bool> RunSearchCommand(const SearchCommandOptions& options,
                                      std::ostream& out, std::ostream& err) {
  ABSL_ASSIGN_OR_RETURN(const std::string index_path,
                        ResolveIndexPath(options.root, options.index_path));
  ABSL_ASSIGN_OR_RETURN(const std::unique_ptr<Segment> segment,
                        OpenIndex(index_path));
  ABSL_ASSIGN_OR_RETURN(const SearchResponse response,
                        SearchSegment(*segment, options.request));
  std::string text = FormatSearchResponse(response, options.format);
  if (options.router && IsIdentifierQuery(options.request) &&
      response.num_matching_files > RouterOptions().route_min_files) {
    FindSymbolResponse definitions;
    std::optional<PathGlob> glob;
    if (!options.request.path_glob.empty()) {
      ABSL_ASSIGN_OR_RETURN(glob, PathGlob::Parse(options.request.path_glob));
    }
    segment->FindSymbols(
        options.request.pattern, !options.request.case_sensitive,
        [&](const SegmentSymbol& symbol) {
          if (!symbol.is_definition) return;
          const SegmentDoc doc = segment->doc(symbol.doc);
          if (glob && !glob->Matches(doc.hint_path)) return;
          ++definitions.total_count;
          if (definitions.occurrences.size() >= 50) return;
          size_t begin = 0;
          for (uint32_t i = 1; i < symbol.line; ++i) {
            begin = LineEnd(doc.content, begin) + 1;
          }
          const size_t end = LineEnd(doc.content, begin);
          definitions.occurrences.push_back(
              {std::string(doc.hint_path), symbol.line, symbol.kind, true,
               std::string(doc.content.substr(begin, end - begin))});
        });
    if (DecideRoute(true, response.num_matching_files, response.token_precision,
                    definitions.total_count > 0) == SearchRoute::kStructural) {
      text = FormatStructuralResponse(response, definitions, options.request,
                                      options.format);
    }
  }
  out << text;
  if (options.print_stats) {
    // One write after the results, so a terminal shows the two apart.
    out.flush();
    const SearchStats& stats = response.stats;
    err << absl::StrCat("candidates ", stats.candidate_chunks,
                        " chunks, verified ", stats.verified_chunks, " (",
                        HumanBytes(stats.verified_bytes), "); compile ",
                        Millis(stats.compile_time), ", evaluate ",
                        Millis(stats.evaluate_time), ", verify ",
                        Millis(stats.verify_time), "\n");
  }
  return response.num_matching_lines > 0;
}

}  // namespace ferret

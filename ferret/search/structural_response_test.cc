// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
#include "ferret/search/structural_response.h"

#include <cstddef>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "ferret/search/search_types.h"
#include "ferret/symbols/symbol_kind.h"
#include "ferret/text/lexer.h"
#include "ferret/workspace/workspace.h"

namespace ferret {
namespace {
TEST(StructuralResponseTest, DefinitionsAndRankedCodeOnlyReferences) {
  for (const std::string& language : {"c", "cc", "py"}) {
    SearchRequest request{.pattern = "needle", .context_lines = 0};
    SearchResponse search;
    search.counts_estimated = true;
    search.count_sample_size = 64;
    search.num_matching_lines = 25;
    search.num_matching_files = 23;
    search.token_precision = 0.5;
    search.terminated_early = true;
    FileResult comments;
    comments.path = "a." + language;
    comments.lines = {
        {.number = 2,
         .text = "// needle()",
         .is_match = true,
         .match_begin = 3,
         .match_end = 9,
         .match_class = MatchClass::kComment},
        {.number = 3,
         .text = "needleExtra()",
         .is_match = true,
         .match_begin = 0,
         .match_end = 6,
         .match_class = MatchClass::kPartialToken},
        {.number = 4,
         .text = "\"needle()\"",
         .is_match = true,
         .match_begin = 1,
         .match_end = 7,
         .match_class = MatchClass::kString},
        {.number = 5,
         .text = "needle;",
         .is_match = true,
         .match_begin = 0,
         .match_end = 6},
        {.number = 6,
         .text = language == "py" ? "def needle(x): ..." : "int needle(int x);",
         .is_match = true,
         .match_begin = 4,
         .match_end = 10}};
    FileResult calls;
    calls.path = "b." + language;
    calls.lines = {{.number = 8,
                    .text = "needle ();",
                    .is_match = true,
                    .match_begin = 0,
                    .match_end = 6},
                   {.number = 9,
                    .text = "needle();",
                    .is_match = true,
                    .match_begin = 0,
                    .match_end = 6}};
    search.files = {comments, calls};
    FindSymbolResponse symbols;
    symbols.total_count = 1;
    symbols.pending_files = 2;
    symbols.occurrences.push_back(
        {"def." + language, 1, SymbolKind::kFunction, true, "needle() {"});
    const std::string got =
        FormatStructuralResponse(search, symbols, request, {.max_bytes = 8192});
    const std::string want = absl::StrCat(
        "Structural route: ~25 text matches in ~23 files (estimated, sample "
        "64); token_precision 0.5; 1 definitions (2 symbol files pending); "
        "reference ranking incomplete (verified text results only).\n"
        "Definitions:\ndef.",
        language, ":1:needle() {\nReferences:\nb.", language,
        " (2 call sites):\n  8:needle ();\n  9:needle();\na.", language,
        " (1 call sites):\n  5:needle;\n  6:",
        language == "py" ? "def needle(x): ...\n" : "int needle(int x);\n");
    EXPECT_EQ(got, want);
    SearchResponse with_context = search;
    FileResult definition_file;
    definition_file.path = "def." + language;
    definition_file.lines = {{.number = 1, .text = "before"},
                             {.number = 2,
                              .text = "needle() {",
                              .is_match = true,
                              .match_begin = 0,
                              .match_end = 6},
                             {.number = 3, .text = "after"}};
    with_context.files.push_back(definition_file);
    FindSymbolResponse contextual_symbols = symbols;
    contextual_symbols.occurrences[0].line = 2;
    SearchRequest contextual_request = request;
    contextual_request.context_lines = 1;
    EXPECT_NE(FormatStructuralResponse(with_context, contextual_symbols,
                                       contextual_request, {.max_bytes = 8192})
                  .find(absl::StrCat("  1:before\ndef.", language,
                                     ":2:needle() {\n  3:after\n")),
              std::string::npos);
    EXPECT_LE(
        FormatStructuralResponse(search, symbols, request, {.max_bytes = 100})
            .size(),
        size_t{100});
  }
}
}  // namespace
}  // namespace ferret

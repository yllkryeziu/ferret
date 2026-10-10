// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
#include "ferret/mcp/search_tools.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "nlohmann/json.hpp"
#include "re2/re2.h"

#include "ferret/engine/engine.h"
#include "ferret/git/path_glob.h"
#include "ferret/mcp/mcp_tool.h"
#include "ferret/regex/compile.h"
#include "ferret/rpc/json_codec.h"
#include "ferret/search/formatter.h"
#include "ferret/search/router.h"
#include "ferret/search/search_types.h"
#include "ferret/search/structural_response.h"
#include "ferret/symbols/symbol_kind.h"
#include "ferret/text/lexer.h"
#include "ferret/text/utf8.h"
#include "ferret/workspace/workspace.h"

namespace ferret {
namespace {

using Json = nlohmann::json;

Json Property(const char* type, const char* description) {
  return {{"type", type}, {"description", description}};
}

Json SearchOutputSchema() {
  Json line = {{"type", "object"},
               {"properties",
                {{"number", Property("integer", "1-based line.")},
                 {"text", Property("string", "Line text.")},
                 {"is_match", Property("boolean", "True for a match.")}}},
               {"required", {"number", "text", "is_match"}}};
  Json file = {{"type", "object"},
               {"required", {"path", "matching_lines", "lines"}}};
  file["properties"] = {
      {"path", Property("string", "Repository-relative path.")},
      {"matching_lines", Property("integer", "Matches in file.")},
      {"lines", {{"type", "array"}, {"items", line}}}};
  Json schema = {{"type", "object"},
                 {"anyOf",
                  {{{"type", "object"},
                    {"required",
                     {"text", "worktree", "shape", "matching_files",
                      "matching_lines", "complete", "stale", "files"}}},
                   {{"type", "object"}, {"required", {"error"}}}}}};
  schema["properties"] = {
      {"error", Property("string", "Correctable pattern or glob error.")},
      {"text", Property("string", "Formatted output.")},
      {"worktree", Property("string", "Searched root.")},
      {"shape", Property("string", "Output shape.")},
      {"matching_files", Property("integer", "Verified files.")},
      {"matching_lines", Property("integer", "Verified lines.")},
      {"complete", Property("boolean", "Complete top results.")},
      {"stale", Property("boolean", "Recent edits may be missing.")},
      {"files", {{"type", "array"}, {"items", file}}}};
  return schema;
}

McpToolResult ModelError(const absl::Status& status) {
  std::string message;
  AppendSanitizedUtf8(status.message(), message);
  return {.text = message,
          .structured = {{"error", std::move(message)}},
          .is_error = true};
}

absl::StatusOr<std::string> SelectRoot(const SearchBackend& backend,
                                       const JsonView& arguments) {
  ABSL_ASSIGN_OR_RETURN(std::optional<std::string> requested,
                        arguments.GetOptionalString("worktree"));
  std::vector<std::string> roots = backend.Roots();
  if (requested.has_value()) {
    for (const std::string& root : roots) {
      if (root == *requested) return root;
    }
    return absl::InvalidArgumentError("worktree must be a registered root");
  }
  if (roots.size() != 1) {
    return absl::InvalidArgumentError(
        "worktree is required when multiple roots are registered");
  }
  return roots.front();
}

// One call takes up to this many patterns besides `pattern`.
constexpr int64_t kMaxPatterns = 16;

// An empty scoped search tries up to this many parent directories of its
// path_glob before the whole worktree.
constexpr int kMaxParentScopes = 3;

// The call's other patterns. A model that fills every property sends "" or []
// when it has none, and may send a single name as a string.
absl::StatusOr<std::vector<std::string>> OtherPatterns(
    const JsonView& arguments) {
  std::vector<std::string> patterns;
  const auto it = arguments.value().find("patterns");
  if (it == arguments.value().end() || it->is_null()) return patterns;
  if (it->is_string()) {
    patterns.push_back(it->get<std::string>());
    return patterns;
  }
  const absl::Status malformed = absl::InvalidArgumentError(absl::StrCat(
      "patterns must be an array of at most ", kMaxPatterns, " strings"));
  if (!it->is_array() || std::ssize(*it) > kMaxPatterns) return malformed;
  for (const Json& element : *it) {
    if (!element.is_string()) return malformed;
    patterns.push_back(element.get<std::string>());
  }
  return patterns;
}

// Where to look when nothing matches under `glob`, narrowest first: the
// nearest parent directories of an anchored glob, then the whole worktree
// (""). A glob without a slash already matches at any depth, and a negated
// one already covers most of the tree, so both widen straight to the whole
// worktree. Trailing * and ** components select everything below their
// directory, so "src/*" widens past "src", which is the same scope.
std::vector<std::string> WiderScopes(absl::string_view glob) {
  std::vector<std::string> scopes;
  if (!absl::StartsWith(glob, "!")) {
    std::vector<absl::string_view> parts =
        absl::StrSplit(absl::StripPrefix(glob, "/"), '/', absl::SkipEmpty());
    const auto trim = [&parts] {
      while (!parts.empty() && (parts.back() == "*" || parts.back() == "**")) {
        parts.pop_back();
      }
    };
    trim();
    while (parts.size() > 1 && std::ssize(scopes) < kMaxParentScopes) {
      parts.pop_back();
      trim();
      if (parts.empty()) break;
      // A one-component glob would match at any depth; a leading slash keeps
      // it at the root.
      scopes.push_back(absl::StrCat(parts.size() == 1 ? "/" : "",
                                    absl::StrJoin(parts, "/")));
    }
  }
  scopes.push_back("");
  return scopes;
}

// Whether a literal pattern reads as a regex an agent meant: an alternation
// without an empty branch (which would match every line), or a wildcard.
bool LooksLikeRegex(absl::string_view pattern) {
  if (absl::StrContains(pattern, '|')) {
    return !absl::StrContains(pattern, "||") &&
           !absl::StartsWith(pattern, "|") && !absl::EndsWith(pattern, "|");
  }
  return absl::StrContains(pattern, ".*") || absl::StrContains(pattern, ".+");
}

class SearchMcpTool final : public McpTool {
 public:
  SearchMcpTool(const SearchBackend& backend, OutputShape shape,
                SearchConsistencyMode consistency, bool fixed_shape,
                bool router)
      : backend_(backend),
        shape_(shape),
        fixed_shape_(fixed_shape),
        router_(router),
        consistency_(consistency) {}
  absl::string_view name() const override { return "search"; }
  absl::string_view description() const override {
    // Fixed-shape servers are experiment arms with one root, so they neither
    // advertise nor accept worktree: a model that fills every property would
    // otherwise pass a relative path and fail every search.
    constexpr absl::string_view kDescription =
        "Search indexed worktree lines. Pattern is required; defaults: "
        "literal, case-insensitive, all paths, 10 files, 2 context lines, "
        "8192 output bytes. One call can try several names: list the others "
        "in patterns, or set regex to true and join them with | "
        "(FooBar|foo_bar|FooBarImpl). Search finds where code is; to look "
        "inside a file you already found, read the file instead of searching "
        "it again. path_glob uses gitignore syntax; a pattern matching a "
        "directory includes everything below it (dir/* matches dir/a/b.c). "
        "When nothing matches under path_glob, the reply shows the matches in "
        "the nearest wider directory. Set worktree when more than one root is "
        "registered.";
    constexpr absl::string_view kWorktreeSentence =
        " Set worktree when more than one root is registered.";
    return fixed_shape_ ? absl::StripSuffix(kDescription, kWorktreeSentence)
                        : kDescription;
  }
  Json input_schema() const override {
    Json schema = {
        {"type", "object"},
        {"additionalProperties", false},
        {"properties",
         {{"pattern", Property("string", "Required search pattern.")},
          {"patterns",
           {{"type", "array"},
            {"items", {{"type", "string"}}},
            {"default", Json::array()},
            {"maxItems", kMaxPatterns},
            {"description",
             "More patterns for the same call; a line matches if it matches "
             "pattern or any of these, with the same regex and case "
             "settings."}}},
          {"regex",
           {{"type", "boolean"},
            {"default", false},
            {"description",
             "Use RE2 regex instead of literal; default false."}}},
          {"case_sensitive",
           {{"type", "boolean"},
            {"default", false},
            {"description", "Exact case; default false."}}},
          {"path_glob",
           Property("string",
                    "Gitignore-style path filter; default all. Prefer a "
                    "directory to a single file.")},
          {"max_files",
           {{"type", "integer"},
            {"description", "Maximum result files; 0 means no limit."},
            {"default", 10},
            {"minimum", 0}}},
          {"context_lines",
           {{"type", "integer"},
            {"description", "Lines before and after matches."},
            {"default", 2},
            {"minimum", 0},
            {"maximum", 100}}},
          {"shape",
           {{"type", "string"},
            {"enum", {"raw", "line", "context", "locations"}},
            {"default", OutputShapeName(shape_)},
            {"description", absl::StrCat("Output shape; default ",
                                         OutputShapeName(shape_), ".")}}},
          {"max_bytes",
           {{"type", "integer"},
            {"description", "Text body budget before trailer."},
            {"default", 8192},
            {"minimum", 0},
            {"maximum", 1048576}}},
          {"worktree",
           Property("string", "Registered root; defaults to the only root.")}}},
        {"required", {"pattern"}}};
    if (fixed_shape_) {
      schema["properties"].erase("shape");
      schema["properties"].erase("worktree");
    }
    return schema;
  }
  Json output_schema() const override {
    Json schema = SearchOutputSchema();
    return schema;
  }
  absl::StatusOr<McpToolResult> Call(
      JsonView arguments, const McpToolContext& context) const override {
    constexpr absl::string_view kKeys[] = {
        "pattern",   "patterns",      "regex", "case_sensitive", "path_glob",
        "max_files", "context_lines", "shape", "max_bytes",      "worktree"};
    for (auto it = arguments.value().begin(); it != arguments.value().end();
         ++it) {
      if ((fixed_shape_ && (it.key() == "shape" || it.key() == "worktree")) ||
          absl::c_find(kKeys, it.key()) == std::end(kKeys)) {
        return absl::InvalidArgumentError(
            absl::StrCat("unknown search argument: ", it.key()));
      }
    }
    ABSL_ASSIGN_OR_RETURN(std::string pattern, arguments.GetString("pattern"));
    ABSL_ASSIGN_OR_RETURN(std::vector<std::string> others,
                          OtherPatterns(arguments));
    ABSL_ASSIGN_OR_RETURN(std::optional<bool> regex,
                          arguments.GetOptionalBool("regex"));
    ABSL_ASSIGN_OR_RETURN(std::optional<bool> case_sensitive,
                          arguments.GetOptionalBool("case_sensitive"));
    ABSL_ASSIGN_OR_RETURN(std::optional<std::string> glob,
                          arguments.GetOptionalString("path_glob"));
    ABSL_ASSIGN_OR_RETURN(std::optional<int64_t> max_files,
                          arguments.GetOptionalInt64("max_files"));
    ABSL_ASSIGN_OR_RETURN(std::optional<int64_t> context_lines,
                          arguments.GetOptionalInt64("context_lines"));
    ABSL_ASSIGN_OR_RETURN(std::optional<int64_t> max_bytes,
                          arguments.GetOptionalInt64("max_bytes"));
    ABSL_ASSIGN_OR_RETURN(std::optional<std::string> shape_name,
                          arguments.GetOptionalString("shape"));
    if (max_files.value_or(0) < 0) {
      return absl::InvalidArgumentError(
          "max_files must not be negative; 0 means no limit");
    }
    if (context_lines.value_or(0) < 0 || context_lines.value_or(0) > 100) {
      return absl::InvalidArgumentError("context_lines must be in [0, 100]");
    }
    if (max_bytes.value_or(0) < 0 || max_bytes.value_or(0) > (1 << 20)) {
      return absl::InvalidArgumentError("max_bytes must be in [0, 1048576]");
    }
    OutputShape shape = shape_;
    if (shape_name) {
      absl::StatusOr<OutputShape> parsed = ParseOutputShape(*shape_name);
      if (!parsed.ok()) {
        return absl::InvalidArgumentError(
            "shape must be raw, line, context, or locations");
      }
      shape = *parsed;
    }
    if (router_ && shape != OutputShape::kContext) {
      return absl::InvalidArgumentError("router requires context output shape");
    }
    ABSL_ASSIGN_OR_RETURN(std::string root, SelectRoot(backend_, arguments));
    SearchRequest request;
    request.pattern = std::move(pattern);
    request.text_only = shape == OutputShape::kRaw;
    request.count_sample = router_;
    request.regex = regex.value_or(false);
    request.case_sensitive = case_sensitive.value_or(false);
    request.path_glob = glob.value_or("");
    // The schema's 0 means no limit; the engine's 0 keeps no files.
    request.max_files = max_files == 0 ? std::numeric_limits<int64_t>::max()
                                       : max_files.value_or(request.max_files);
    request.context_lines =
        static_cast<int>(context_lines.value_or(request.context_lines));
    // Several patterns become one alternation, which the index plans and the
    // engine verifies in a single pass.
    std::vector<std::string> patterns;
    others.insert(others.begin(), request.pattern);
    for (std::string& other : others) {
      if (!other.empty() && !absl::c_linear_search(patterns, other)) {
        patterns.push_back(std::move(other));
      }
    }
    if (patterns.size() == 1) request.pattern = std::move(patterns[0]);
    if (patterns.size() > 1) {
      for (std::string& one : patterns) {
        const absl::Status compiled =
            CompilePattern({.pattern = one,
                            .literal = !request.regex,
                            .case_sensitive = request.case_sensitive})
                .status();
        if (!compiled.ok()) return ModelError(compiled);
        one =
            request.regex ? absl::StrCat("(?:", one, ")") : RE2::QuoteMeta(one);
      }
      request.pattern = absl::StrJoin(patterns, "|");
      request.regex = true;
    }
    const absl::Status compiled =
        CompilePattern({.pattern = request.pattern,
                        .literal = !request.regex,
                        .case_sensitive = request.case_sensitive})
            .status();
    if (!compiled.ok()) return ModelError(compiled);
    if (!request.path_glob.empty()) {
      const absl::Status parsed = PathGlob::Parse(request.path_glob).status();
      if (!parsed.ok()) return ModelError(parsed);
    }
    if (context.cancelled()) return absl::CancelledError("search cancelled");
    SearchConsistency consistency;
    consistency.mode = consistency_;
    consistency.timeout = kServingSyncTimeout;
    absl::StatusOr<WorkspaceSearchResult> result =
        backend_.Search(root, request, consistency);
    if (!result.ok()) {
      if (absl::IsInvalidArgument(result.status())) {
        return ModelError(result.status());
      }
      return result.status();
    }
    SearchResponse response = std::move(result->response);
    // A miss costs an agent a turn, so say what would have matched.
    std::string hint;
    if (response.num_matching_lines == 0 && !request.regex &&
        LooksLikeRegex(request.pattern) &&
        CompilePattern({.pattern = request.pattern,
                        .literal = false,
                        .case_sensitive = request.case_sensitive})
            .ok()) {
      SearchRequest as_regex = request;
      as_regex.regex = true;
      absl::StatusOr<WorkspaceSearchResult> retried =
          backend_.Search(root, as_regex, consistency);
      if (retried.ok() && retried->response.num_matching_lines > 0) {
        hint = absl::StrCat("No literal match for \"", request.pattern,
                            "\"; these are its matches as a regex.\n");
        request = std::move(as_regex);
        response = std::move(retried->response);
        result->stale |= retried->stale;
      }
    }
    // Agents scope a search to the one file they expect a name in, and when
    // it is not there they rephrase the pattern rather than drop the scope,
    // so an empty scope answers with the nearest scope that has matches.
    if (response.num_matching_lines == 0 && !request.path_glob.empty()) {
      const std::string missed = absl::StrCat("No matches under path_glob \"",
                                              request.path_glob, "\"");
      hint = absl::StrCat(missed, " or anywhere else in the worktree.\n");
      for (std::string& scope : WiderScopes(request.path_glob)) {
        if (context.cancelled()) {
          return absl::CancelledError("search cancelled");
        }
        SearchRequest wider = request;
        wider.path_glob = std::move(scope);
        absl::StatusOr<WorkspaceSearchResult> widened =
            backend_.Search(root, wider, consistency);
        if (!widened.ok()) {
          hint.clear();
          break;
        }
        result->stale |= widened->stale;
        if (widened->response.num_matching_lines == 0) continue;
        hint =
            absl::StrCat(missed, "; these are the matches ",
                         wider.path_glob.empty()
                             ? std::string("in the whole worktree")
                             : absl::StrCat("under \"", wider.path_glob, "\""),
                         " instead.\n");
        request = std::move(wider);
        response = std::move(widened->response);
        break;
      }
    }
    if (router_ && !backend_.SupportsFindSymbol()) {
      int64_t code = 0;
      int64_t total = 0;
      for (const FileResult& file : response.files) {
        for (const ResultLine& line : file.lines) {
          if (!line.is_match) continue;
          ++total;
          code += line.match_class == MatchClass::kCodeIdentifier;
        }
      }
      if (total > 0) {
        response.token_precision =
            static_cast<double>(code) / static_cast<double>(total);
      }
    }
    FormatOptions format = {
        .shape = shape,
        .max_bytes = static_cast<size_t>(max_bytes.value_or(8192))};
    std::string text;
    bool structural = false;
    if (router_ && IsIdentifierQuery(request) &&
        response.num_matching_files > RouterOptions().route_min_files) {
      FindSymbolResponse definitions;
      if (backend_.SupportsFindSymbol()) {
        FindSymbolRequest symbol_request;
        symbol_request.name = request.pattern;
        symbol_request.definitions_only = true;
        symbol_request.case_insensitive = !request.case_sensitive;
        symbol_request.path_glob = request.path_glob;
        ABSL_ASSIGN_OR_RETURN(
            definitions,
            backend_.FindSymbol(root, symbol_request, consistency));
      }
      if (DecideRoute(true, response.num_matching_files,
                      response.token_precision, definitions.total_count > 0) ==
          SearchRoute::kStructural) {
        text = FormatStructuralResponse(response, definitions, request, format);
        structural = true;
      }
    }
    if (!structural) text = FormatSearchResponse(response, format);
    // The note comes first: the lines below it answer a different question
    // than the one asked.
    text.insert(0, hint);
    if (result->stale) {
      absl::StrAppend(
          &text,
          "\nStale: recent edits may be missing (the file watcher did not "
          "confirm them in time).\n");
    }
    Json files = Json::array();
    for (const FileResult& file : response.files) {
      Json lines = Json::array();
      for (const ResultLine& line : file.lines) {
        std::string sanitized;
        AppendDisplayLine(line.text, line.match_begin, line.match_end, 256,
                          sanitized);
        lines.push_back({{"number", line.number},
                         {"text", std::move(sanitized)},
                         {"is_match", line.is_match}});
      }
      std::string path;
      AppendSanitizedUtf8(file.path, path);
      files.push_back({{"path", std::move(path)},
                       {"matching_lines", file.num_matching_lines},
                       {"lines", std::move(lines)}});
    }
    return McpToolResult{
        .text = text,
        .structured = {{"text", text},
                       {"worktree", root},
                       {"shape", OutputShapeName(shape)},
                       {"matching_files", response.num_matching_files},
                       {"matching_lines", response.num_matching_lines},
                       {"complete", response.complete},
                       {"stale", result->stale},
                       {"files", std::move(files)}}};
  }

 private:
  const SearchBackend& backend_;
  const OutputShape shape_;
  const bool fixed_shape_;
  const bool router_;
  const SearchConsistencyMode consistency_;
};

class FindSymbolMcpTool final : public McpTool {
 public:
  FindSymbolMcpTool(const SearchBackend& backend,
                    SearchConsistencyMode consistency)
      : backend_(backend), consistency_(consistency) {}
  absl::string_view name() const override { return "find_symbol"; }
  absl::string_view description() const override {
    return "Find definitions and declarations by exact symbol name. Results "
           "are scoped to a registered worktree; pending files may not yet "
           "appear. Set worktree when multiple roots are registered.";
  }
  Json input_schema() const override {
    return {{"type", "object"},
            {"additionalProperties", false},
            {"properties",
             {{"name", Property("string", "Required exact symbol name.")},
              {"case_insensitive", {{"type", "boolean"}, {"default", false}}},
              {"kinds",
               {{"type", "array"},
                {"items",
                 {{"type", "string"},
                  {"enum",
                   {"function", "method", "class", "struct", "union", "enum",
                    "enumerator", "type_alias", "macro", "variable"}}}}}},
              {"definitions_only", {{"type", "boolean"}, {"default", false}}},
              {"path_glob", Property("string", "Gitignore-style path filter.")},
              {"max_results",
               {{"type", "integer"}, {"minimum", 0}, {"default", 50}}},
              {"max_bytes",
               {{"type", "integer"},
                {"minimum", 0},
                {"maximum", 1048576},
                {"default", 8192}}},
              {"worktree", Property("string", "Registered root.")}}},
            {"required", {"name"}}};
  }
  Json output_schema() const override {
    Json occurrence = {
        {"type", "object"},
        {"properties",
         {{"path", Property("string", "View path.")},
          {"line", Property("integer", "Line number.")},
          {"kind", Property("string", "Symbol kind.")},
          {"is_definition", Property("boolean", "Definition.")},
          {"source_line", Property("string", "Source text.")}}},
        {"required", {"path", "line", "kind", "is_definition", "source_line"}}};
    Json schema = {
        {"type", "object"},
        {"properties",
         {{"text", Property("string", "Budgeted result text.")},
          {"worktree", Property("string", "Searched root.")},
          {"total_count", Property("integer", "All matches.")},
          {"pending_files", Property("integer", "Not extracted yet.")},
          {"stale", Property("boolean", "Edits may be missing.")},
          {"occurrences", {{"type", "array"}, {"items", occurrence}}}}},
        {"required",
         {"text", "worktree", "total_count", "pending_files", "stale",
          "occurrences"}}};
    schema.erase("required");
    schema["properties"]["error"] =
        Property("string", "Correctable symbol or glob error.");
    schema["anyOf"] = {{{"type", "object"},
                        {"required",
                         {"text", "worktree", "total_count", "pending_files",
                          "stale", "occurrences"}}},
                       {{"type", "object"}, {"required", {"error"}}}};
    return schema;
  }
  absl::StatusOr<McpToolResult> Call(
      JsonView arguments, const McpToolContext& context) const override {
    constexpr absl::string_view kKeys[] = {
        "name",      "case_insensitive", "kinds",     "definitions_only",
        "path_glob", "max_results",      "max_bytes", "worktree"};
    for (auto it = arguments.value().begin(); it != arguments.value().end();
         ++it) {
      if (absl::c_find(kKeys, it.key()) == std::end(kKeys)) {
        return absl::InvalidArgumentError(
            absl::StrCat("unknown find_symbol argument: ", it.key()));
      }
    }
    FindSymbolRequest request;
    ABSL_ASSIGN_OR_RETURN(request.name, arguments.GetString("name"));
    ABSL_ASSIGN_OR_RETURN(std::optional<bool> insensitive,
                          arguments.GetOptionalBool("case_insensitive"));
    ABSL_ASSIGN_OR_RETURN(std::optional<bool> definitions,
                          arguments.GetOptionalBool("definitions_only"));
    ABSL_ASSIGN_OR_RETURN(std::optional<std::string> glob,
                          arguments.GetOptionalString("path_glob"));
    ABSL_ASSIGN_OR_RETURN(std::optional<int64_t> max_results,
                          arguments.GetOptionalInt64("max_results"));
    ABSL_ASSIGN_OR_RETURN(std::optional<int64_t> max_bytes,
                          arguments.GetOptionalInt64("max_bytes"));
    ABSL_ASSIGN_OR_RETURN(std::optional<JsonView> kinds,
                          arguments.GetOptionalArray("kinds"));
    if (kinds) {
      for (size_t i = 0; i < kinds->size(); ++i) {
        ABSL_ASSIGN_OR_RETURN(std::string name, kinds->Element(i).AsString());
        const std::optional<SymbolKind> kind = ParseSymbolKind(name);
        if (!kind) {
          return ModelError(absl::InvalidArgumentError("unknown symbol kind"));
        }
        request.kinds.push_back(*kind);
      }
    }
    request.case_insensitive = insensitive.value_or(false);
    request.definitions_only = definitions.value_or(false);
    request.path_glob = glob.value_or("");
    request.max_results = max_results.value_or(50);
    if (max_bytes.value_or(8192) < 0 || max_bytes.value_or(8192) > (1 << 20)) {
      return ModelError(absl::InvalidArgumentError("max_bytes out of range"));
    }
    ABSL_ASSIGN_OR_RETURN(std::string root, SelectRoot(backend_, arguments));
    if (context.cancelled()) {
      return absl::CancelledError("symbol lookup cancelled");
    }
    SearchConsistency consistency;
    consistency.mode = consistency_;
    consistency.timeout = kServingSyncTimeout;
    absl::StatusOr<FindSymbolResponse> response =
        backend_.FindSymbol(root, request, consistency);
    if (!response.ok()) {
      if (absl::IsInvalidArgument(response.status())) {
        return ModelError(response.status());
      }
      return response.status();
    }
    Json occurrences = Json::array();
    std::string text;
    const size_t budget = static_cast<size_t>(max_bytes.value_or(8192));
    for (const SymbolLocation& item : response->occurrences) {
      std::string path;
      AppendSanitizedUtf8(item.path, path);
      std::string line;
      AppendSanitizedUtf8(item.source_line, line);
      occurrences.push_back({{"path", path},
                             {"line", item.line},
                             {"kind", SymbolKindName(item.kind)},
                             {"is_definition", item.is_definition},
                             {"source_line", line}});
      const std::string row =
          absl::StrCat(path, ":", item.line, ": ",
                       item.is_definition ? "definition " : "declaration ",
                       SymbolKindName(item.kind), " ", line, "\n");
      if (text.size() + row.size() <= budget) text.append(row);
    }
    if (text.empty() && response->total_count == 0 && budget >= 18) {
      text = "No symbols found.\n";
    }
    if (response->pending_files > 0) {
      const std::string pending =
          absl::StrCat(response->pending_files, " files pending extraction.\n");
      if (text.size() + pending.size() <= budget) text.append(pending);
    }
    return McpToolResult{
        .text = text,
        .structured = {{"text", text},
                       {"worktree", root},
                       {"total_count", response->total_count},
                       {"pending_files", response->pending_files},
                       {"stale", response->stale},
                       {"occurrences", std::move(occurrences)}}};
  }

 private:
  const SearchBackend& backend_;
  SearchConsistencyMode consistency_;
};

class IndexStatusMcpTool final : public McpTool {
 public:
  explicit IndexStatusMcpTool(const SearchBackend& backend)
      : backend_(backend) {}
  absl::string_view name() const override { return "index_status"; }
  absl::string_view description() const override {
    return "Report published view paths and blobs, ingest generations and "
           "queues, "
           "watcher health, and published index segments. No arguments.";
  }
  Json input_schema() const override {
    return {{"type", "object"}, {"additionalProperties", false}};
  }
  Json output_schema() const override {
    Json status = {
        {"type", "object"},
        {"properties",
         {{"root", Property("string", "Registered root.")},
          {"file_count", Property("integer", "Visible paths.")},
          {"blob_count", Property("integer", "Distinct visible blobs.")},
          {"index_bytes", Property("integer", "Published segment bytes.")},
          {"segment_count", Property("integer", "Segments.")},
          {"pending_symbols", Property("integer", "Pending symbol files.")},
          {"failed_symbols", Property("integer", "Failed symbol files.")},
          {"applied_generation", Property("integer", "Applied generation.")},
          {"assigned_generation", Property("integer", "Assigned generation.")},
          {"queued_paths", Property("integer", "Interactive paths queued.")},
          {"unstable_paths", Property("integer", "Unstable paths.")},
          {"skipped_paths", Property("integer", "Skipped paths.")},
          {"watcher_running", Property("boolean", "Watcher running.")},
          {"watcher_failed", Property("boolean", "Watcher failed over.")}}},
        {"required",
         {"root", "file_count", "blob_count", "index_bytes", "segment_count",
          "applied_generation", "assigned_generation", "pending_symbols",
          "failed_symbols", "queued_paths", "unstable_paths", "skipped_paths",
          "watcher_running", "watcher_failed"}}};
    return {
        {"type", "object"},
        {"properties", {{"worktrees", {{"type", "array"}, {"items", status}}}}},
        {"required", {"worktrees"}}};
  }
  absl::StatusOr<McpToolResult> Call(
      JsonView arguments, const McpToolContext& context) const override {
    if (!arguments.value().is_object() || !arguments.value().empty()) {
      return absl::InvalidArgumentError("index_status takes no arguments");
    }
    if (context.cancelled()) return absl::CancelledError("status cancelled");
    ABSL_ASSIGN_OR_RETURN(std::vector<WorktreeIndexStatus> statuses,
                          backend_.IndexStatus());
    Json worktrees = Json::array();
    std::string text;
    for (const WorktreeIndexStatus& status : statuses) {
      worktrees.push_back({{"root", status.root},
                           {"file_count", status.file_count},
                           {"blob_count", status.blob_count},
                           {"index_bytes", status.index_bytes},
                           {"segment_count", status.segment_count},
                           {"pending_symbols", status.pending_symbols},
                           {"failed_symbols", status.failed_symbols},
                           {"applied_generation", status.applied_generation},
                           {"assigned_generation", status.assigned_generation},
                           {"queued_paths", status.queued_paths},
                           {"unstable_paths", status.unstable_paths},
                           {"skipped_paths", status.skipped_paths},
                           {"watcher_running", status.watcher_running},
                           {"watcher_failed", status.watcher_failed}});
      absl::StrAppend(&text, status.root, ": ", status.file_count, " files, ",
                      status.blob_count, " blobs, ", status.index_bytes,
                      " bytes, ", status.segment_count, " segments, queued ",
                      status.queued_paths, ", symbols pending ",
                      status.pending_symbols, ", failed ",
                      status.failed_symbols, "\n");
    }
    return McpToolResult{.text = std::move(text),
                         .structured = {{"worktrees", std::move(worktrees)}}};
  }

 private:
  const SearchBackend& backend_;
};

}  // namespace

std::vector<std::unique_ptr<McpTool>> SearchTools(
    const SearchBackend& backend, OutputShape default_shape,
    SearchConsistencyMode consistency, bool fixed_shape, bool router) {
  std::vector<std::unique_ptr<McpTool>> tools;
  tools.push_back(std::make_unique<SearchMcpTool>(
      backend, default_shape, consistency, fixed_shape, router));
  if (backend.SupportsFindSymbol()) {
    tools.push_back(std::make_unique<FindSymbolMcpTool>(backend, consistency));
  }
  tools.push_back(std::make_unique<IndexStatusMcpTool>(backend));
  return tools;
}

}  // namespace ferret

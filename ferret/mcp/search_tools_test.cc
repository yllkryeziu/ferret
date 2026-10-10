// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
#include "ferret/mcp/search_tools.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

#include "ferret/engine/engine.h"
#include "ferret/mcp/mcp_server.h"
#include "ferret/mcp/mcp_tool.h"
#include "ferret/rpc/framing.h"
#include "ferret/rpc/json_codec.h"
#include "ferret/rpc/testing/test_pipe.h"
#include "ferret/search/formatter.h"
#include "ferret/search/search_types.h"
#include "ferret/symbols/symbol_kind.h"
#include "ferret/text/lexer.h"
#include "ferret/text/utf8.h"
#include "ferret/workspace/workspace.h"

namespace ferret {
namespace {

class FakeSearchBackend final : public SearchBackend {
 public:
  std::vector<std::string> roots = {"/repo"};
  bool invalid_utf8 = false;
  bool stale = false;
  bool supports_symbols = false;
  int64_t symbol_pending = 0;
  bool many_matches = false;
  double precision = 0.0;
  mutable int64_t seen_max_files = -1;
  // When set, answers every search instead of the fixed "needle" reply, and
  // `requests` records them.
  absl::AnyInvocable<SearchResponse(const SearchRequest&) const> respond;
  mutable std::vector<SearchRequest> requests;
  std::vector<std::string> Roots() const override { return roots; }
  absl::StatusOr<WorkspaceSearchResult> Search(
      absl::string_view root, const SearchRequest& request,
      const SearchConsistency& consistency) const override {
    if (respond != nullptr) {
      requests.push_back(request);
      return WorkspaceSearchResult{.response = respond(request)};
    }
    if (root != "/repo" || request.pattern != "needle" ||
        consistency.mode != kServingConsistency) {
      return absl::InvalidArgumentError("unexpected request");
    }
    seen_max_files = request.max_files;
    SearchResponse response;
    response.num_matching_files = many_matches ? 25 : 1;
    response.num_matching_lines = many_matches ? 25 : 1;
    response.token_precision = precision;
    FileResult file;
    file.path = "file.cc";
    file.num_matching_lines = 1;
    file.lines.push_back({.number = 4,
                          .text = "needle and more",
                          .is_match = true,
                          .match_begin = 0,
                          .match_end = 6,
                          .match_class = many_matches && precision < 0.7
                                             ? MatchClass::kComment
                                             : MatchClass::kCodeIdentifier});
    if (invalid_utf8) {
      file.path = "bad\xff.cc";
      file.lines[0].text = "needle\xff";
    }
    response.files.push_back(std::move(file));
    return WorkspaceSearchResult{.response = std::move(response),
                                 .stale = stale};
  }
  bool SupportsFindSymbol() const override { return supports_symbols; }
  absl::StatusOr<FindSymbolResponse> FindSymbol(
      absl::string_view root, const FindSymbolRequest& request,
      const SearchConsistency& consistency) const override {
    if (root != "/repo" || consistency.mode != kServingConsistency) {
      return absl::InvalidArgumentError("unexpected symbol lookup");
    }
    FindSymbolResponse response;
    response.pending_files = request.name == "Pending" ? symbol_pending : 0;
    if (request.name == "Found") {
      response.total_count = 1;
      if (request.max_results > 0) {
        response.occurrences.push_back(
            {"file.cc", 4, SymbolKind::kFunction, true, "int Found() {}"});
      }
    }
    return response;
  }
  absl::StatusOr<std::vector<WorktreeIndexStatus>> IndexStatus()
      const override {
    return std::vector<WorktreeIndexStatus>{{.root = "/repo",
                                             .file_count = 2,
                                             .blob_count = 1,
                                             .index_bytes = 42,
                                             .segment_count = 1,
                                             .applied_generation = 0}};
  }
};

// Checks the object, array, scalar, and required-property subset used by the
// published output schemas, rather than merely checking field names.
bool MatchesSchema(const nlohmann::json& value, const nlohmann::json& schema) {
  const std::string type = schema.at("type").get<std::string>();
  if (type == "object") {
    if (!value.is_object()) return false;
    if (schema.contains("anyOf")) {
      bool valid = false;
      for (const auto& option : schema.at("anyOf")) {
        valid |= MatchesSchema(value, option);
      }
      if (!valid) return false;
    }
    for (const auto& key : schema.value("required", nlohmann::json::array())) {
      if (!value.contains(key.get<std::string>())) return false;
    }
    const nlohmann::json properties =
        schema.value("properties", nlohmann::json::object());
    for (auto it = properties.begin(); it != properties.end(); ++it) {
      if (value.contains(it.key()) &&
          !MatchesSchema(value.at(it.key()), it.value())) {
        return false;
      }
    }
    return true;
  }
  if (type == "array") {
    if (!value.is_array()) return false;
    for (const auto& item : value) {
      if (!MatchesSchema(item, schema.at("items"))) return false;
    }
    return true;
  }
  if (type == "integer") return value.is_number_integer();
  if (type == "boolean") return value.is_boolean();
  if (type == "string") return value.is_string();
  return false;
}

TEST(SearchToolsTest, FixedShapeKeepsSchemaAcrossShapesAndRouter) {
  FakeSearchBackend backend;
  const nlohmann::json baseline =
      SearchTools(backend, OutputShape::kRaw, kServingConsistency, true)[0]
          ->input_schema();
  EXPECT_FALSE(baseline["properties"].contains("shape"));
  EXPECT_FALSE(baseline["properties"].contains("worktree"));
  for (OutputShape shape : {OutputShape::kRaw, OutputShape::kLine,
                            OutputShape::kContext, OutputShape::kLocations}) {
    for (bool router : {false, true}) {
      auto tools =
          SearchTools(backend, shape, kServingConsistency, true, router);
      EXPECT_EQ(tools[0]->input_schema(), baseline);
      EXPECT_NE(std::string(tools[0]->description()).find("dir/a/b.c"),
                std::string::npos);
      EXPECT_EQ(std::string(tools[0]->description()).find("worktree when"),
                std::string::npos);
      absl::Notification cancellation;
      McpToolContext context(cancellation);
      for (const char* key : {"shape", "worktree"}) {
        EXPECT_EQ(tools[0]
                      ->Call(JsonView(nlohmann::json{{"pattern", "needle"},
                                                     {key, "raw"}}),
                             context)
                      .status()
                      .code(),
                  absl::StatusCode::kInvalidArgument)
            << key;
      }
      if (router && shape != OutputShape::kContext) continue;
      // gpt-6-luna fills every advertised property, with "" for strings that
      // have no default; that must still search.
      nlohmann::json every_property;
      for (auto it = baseline["properties"].begin();
           it != baseline["properties"].end(); ++it) {
        every_property[it.key()] =
            it.value().contains("default") ? it.value()["default"] : "";
      }
      every_property["pattern"] = "needle";
      auto result = tools[0]->Call(JsonView(every_property), context);
      ASSERT_TRUE(result.ok()) << result.status();
      EXPECT_FALSE(result->is_error) << result->text;
      EXPECT_EQ(result->structured["matching_files"], 1);
    }
  }
}

TEST(SearchToolsTest, UnfixedShapeKeepsWorktree) {
  FakeSearchBackend backend;
  auto tools = SearchTools(backend, OutputShape::kRaw);
  EXPECT_TRUE(tools[0]->input_schema()["properties"].contains("worktree"));
  EXPECT_NE(std::string(tools[0]->description()).find("worktree when"),
            std::string::npos);
  absl::Notification cancellation;
  McpToolContext context(cancellation);
  auto result = tools[0]->Call(
      JsonView(nlohmann::json{{"pattern", "needle"}, {"worktree", "/repo"}}),
      context);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->structured["worktree"], "/repo");
}

TEST(SearchToolsTest, RouterSelectsStructuralAndTextWithoutSymbols) {
  FakeSearchBackend backend;
  backend.many_matches = true;
  auto tools = SearchTools(backend, OutputShape::kContext, kServingConsistency,
                           true, true);
  absl::Notification cancellation;
  McpToolContext context(cancellation);
  nlohmann::json args = {{"pattern", "needle"}};
  auto structural = tools[0]->Call(JsonView(args), context);
  ASSERT_TRUE(structural.ok()) << structural.status();
  EXPECT_NE(structural->text.find("Structural route:"), std::string::npos);
  backend.precision = 0.7;
  auto text = tools[0]->Call(JsonView(args), context);
  ASSERT_TRUE(text.ok()) << text.status();
  EXPECT_EQ(text->text.find("Structural route:"), std::string::npos);
  backend.supports_symbols = true;
  // A definition is sufficient even at the precision boundary.
  backend.precision = 0.69;
  auto still_structural = tools[0]->Call(JsonView(args), context);
  ASSERT_TRUE(still_structural.ok()) << still_structural.status();
  EXPECT_NE(still_structural->text.find("Structural route:"),
            std::string::npos);
}

TEST(SearchToolsTest, SearchDefaultRootAndBudget) {
  FakeSearchBackend backend;
  auto tools = SearchTools(backend, OutputShape::kRaw);
  absl::Notification cancellation;
  McpToolContext context(cancellation);
  nlohmann::json args = {{"pattern", "needle"}, {"max_bytes", 0}};
  auto result = tools[0]->Call(JsonView(args), context);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_FALSE(result->is_error);
  EXPECT_EQ(result->structured["worktree"], "/repo");
  EXPECT_EQ(result->structured["shape"], "raw");
  EXPECT_EQ(result->structured["text"], result->text);
  EXPECT_EQ(result->structured["files"][0]["lines"][0]["number"], 4);
  EXPECT_EQ(result->structured["matching_files"], 1);
  EXPECT_TRUE(MatchesSchema(result->structured, tools[0]->output_schema()));
  EXPECT_EQ(result->text.find("needle and more"), std::string::npos);
}

TEST(SearchToolsTest, SchemaBoundsAndStaleResult) {
  FakeSearchBackend backend;
  backend.stale = true;
  auto tools = SearchTools(backend);
  const nlohmann::json schema = tools[0]->input_schema()["properties"];
  for (const auto& [key, default_value, min_value, max_value] :
       {std::tuple{"context_lines", 2, 0, int64_t{100}},
        std::tuple{"max_bytes", 8192, 0, int64_t{1048576}}}) {
    EXPECT_EQ(schema[key]["default"], default_value);
    EXPECT_EQ(schema[key]["minimum"], min_value);
    EXPECT_EQ(schema[key]["maximum"], max_value);
  }
  // Zero means no limit, so max_files has no maximum.
  EXPECT_EQ(schema["max_files"]["default"], 10);
  EXPECT_EQ(schema["max_files"]["minimum"], 0);
  EXPECT_FALSE(schema["max_files"].contains("maximum"));
  absl::Notification cancellation;
  McpToolContext context(cancellation);
  nlohmann::json args = {{"pattern", "needle"}};
  auto result = tools[0]->Call(JsonView(args), context);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->structured["stale"], true);
  EXPECT_NE(result->text.find("Stale:"), std::string::npos);
  EXPECT_TRUE(MatchesSchema(result->structured, tools[0]->output_schema()));
}

TEST(SearchToolsTest, ZeroMaxFilesMeansNoLimit) {
  // The engine keeps no files for a top K of zero, so the schema's zero must
  // not reach it as zero. Agents asked for 0 and got only the counts.
  FakeSearchBackend backend;
  auto tools = SearchTools(backend);
  absl::Notification cancellation;
  McpToolContext context(cancellation);
  for (const auto& [max_files, expected] :
       {std::pair{nlohmann::json(nullptr), int64_t{10}},
        std::pair{nlohmann::json(3), int64_t{3}},
        std::pair{nlohmann::json(0), std::numeric_limits<int64_t>::max()}}) {
    nlohmann::json args = {{"pattern", "needle"}};
    if (!max_files.is_null()) args["max_files"] = max_files;
    auto result = tools[0]->Call(JsonView(args), context);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(backend.seen_max_files, expected) << args;
  }
}

SearchResponse Matches(std::vector<std::string> paths) {
  SearchResponse response;
  for (std::string& path : paths) {
    FileResult file;
    file.path = std::move(path);
    file.num_matching_lines = 1;
    file.lines.push_back({.number = 1,
                          .text = "match",
                          .is_match = true,
                          .match_begin = 0,
                          .match_end = 5});
    response.files.push_back(std::move(file));
  }
  response.num_matching_files = std::ssize(response.files);
  response.num_matching_lines = std::ssize(response.files);
  return response;
}

TEST(SearchToolsTest, DescriptionTeachesOneCallForSeveralNames) {
  FakeSearchBackend backend;
  auto tools = SearchTools(backend);
  const std::string description(tools[0]->description());
  for (const char* phrase :
       {"in patterns", "FooBar|foo_bar", "read the file instead"}) {
    EXPECT_NE(description.find(phrase), std::string::npos) << phrase;
  }
}

// Agents averaged 18 calls a task against ripgrep's 11, largely because one
// rg command line tries several names and one MCP call tried one.
TEST(SearchToolsTest, SeveralPatternsSearchInOneRequest) {
  FakeSearchBackend backend;
  backend.respond = [](const SearchRequest&) { return Matches({"a.cc"}); };
  auto tools = SearchTools(backend);
  absl::Notification cancellation;
  McpToolContext context(cancellation);
  struct Case {
    nlohmann::json args;
    std::string pattern;
    bool regex;
  };
  for (const Case& c : {
           // Literals are quoted; empty and repeated patterns are dropped.
           Case{{{"pattern", "Foo.Bar"},
                 {"patterns", {"foo_bar", "", "Foo.Bar", "a|b"}}},
                "Foo\\.Bar|foo_bar|a\\|b",
                true},
           Case{{{"pattern", "Open"}, {"patterns", {"Close"}}, {"regex", true}},
                "(?:Open)|(?:Close)",
                true},
           Case{{{"pattern", "a|b"}, {"patterns", {"c+"}}, {"regex", true}},
                "(?:a|b)|(?:c+)",
                true},
           // A lone pattern searches as itself, wherever it was given.
           Case{{{"pattern", ""}, {"patterns", {"needle"}}}, "needle", false},
           Case{{{"pattern", "needle"}, {"patterns", ""}}, "needle", false},
           Case{{{"pattern", "needle"}, {"patterns", nlohmann::json::array()}},
                "needle",
                false},
           // A model may send one extra name as a string.
           Case{{{"pattern", "x"}, {"patterns", "y"}}, "x|y", true},
       }) {
    backend.requests.clear();
    auto result = tools[0]->Call(JsonView(c.args), context);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_FALSE(result->is_error) << c.args << result->text;
    ASSERT_EQ(backend.requests.size(), 1u) << c.args;
    EXPECT_EQ(backend.requests[0].pattern, c.pattern) << c.args;
    EXPECT_EQ(backend.requests[0].regex, c.regex) << c.args;
  }

  backend.requests.clear();
  auto result = tools[0]->Call(
      JsonView(nlohmann::json{
          {"pattern", "a"}, {"patterns", {"b("}}, {"regex", true}}),
      context);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->is_error);
  EXPECT_TRUE(backend.requests.empty());

  nlohmann::json too_many = nlohmann::json::array();
  for (int i = 0; i < 17; ++i) too_many.push_back(absl::StrCat("p", i));
  for (const nlohmann::json& patterns :
       {too_many, nlohmann::json{1}, nlohmann::json(true)}) {
    EXPECT_EQ(tools[0]
                  ->Call(JsonView(nlohmann::json{{"pattern", "a"},
                                                 {"patterns", patterns}}),
                         context)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument)
        << patterns;
  }
}

// Agents carry rg's habit of writing A|B into Ferret's literal default.
TEST(SearchToolsTest, LiteralMissWithRegexSyntaxRetriesAsRegex) {
  FakeSearchBackend backend;
  backend.respond = [](const SearchRequest& request) {
    return request.regex ? Matches({"a.cc"}) : SearchResponse();
  };
  auto tools = SearchTools(backend);
  absl::Notification cancellation;
  McpToolContext context(cancellation);
  auto result = tools[0]->Call(
      JsonView(nlohmann::json{{"pattern", "Open|Close"}}), context);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_NE(result->text.find("No literal match"), std::string::npos)
      << result->text;
  EXPECT_NE(result->text.find("a.cc"), std::string::npos) << result->text;
  EXPECT_EQ(result->structured["matching_files"], 1);
  ASSERT_EQ(backend.requests.size(), 2u);
  EXPECT_TRUE(backend.requests[1].regex);

  // An empty alternative would match every line, so `a || b` stays literal.
  for (const char* pattern : {"a || b", "|x", "x|", "plain"}) {
    backend.requests.clear();
    result =
        tools[0]->Call(JsonView(nlohmann::json{{"pattern", pattern}}), context);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(backend.requests.size(), 1u) << pattern;
    EXPECT_EQ(result->text.find("No literal match"), std::string::npos);
  }
}

// After an empty scoped search, agents rephrased the pattern rather than drop
// the scope, though every such scope had matches nearby.
TEST(SearchToolsTest, EmptyScopeWidensToNearestMatches) {
  FakeSearchBackend backend;
  std::string matching_scope;
  backend.respond = [&matching_scope](const SearchRequest& request) {
    return request.path_glob == matching_scope ? Matches({"src/a.cc"})
                                               : SearchResponse();
  };
  auto tools = SearchTools(backend);
  absl::Notification cancellation;
  McpToolContext context(cancellation);
  struct Case {
    std::string glob;
    std::string matching_scope;
    std::vector<std::string> searched;
    std::string note;
  };
  for (const Case& c : {
           Case{"src/foo/bar.cc",
                "/src",
                {"src/foo/bar.cc", "src/foo", "/src"},
                "these are the matches under \"/src\" instead."},
           // src/* and src are the same scope.
           Case{"src/*",
                "",
                {"src/*", ""},
                "these are the matches in the whole worktree instead."},
           Case{"src/**/*.cc", "", {"src/**/*.cc", "/src", ""}, "whole"},
           // Unanchored and negated globs already span the tree.
           Case{"*.cc", "", {"*.cc", ""}, "whole"},
           Case{"!vendor", "", {"!vendor", ""}, "whole"},
           // Only the nearest three parents, then everything.
           Case{"a/b/c/d/e.cc",
                "",
                {"a/b/c/d/e.cc", "a/b/c/d", "a/b/c", "a/b", ""},
                "whole"},
           Case{"src/foo.cc",
                "nowhere",
                {"src/foo.cc", "/src", ""},
                "or anywhere else in the worktree."},
       }) {
    matching_scope = c.matching_scope;
    backend.requests.clear();
    auto result = tools[0]->Call(
        JsonView(nlohmann::json{
            {"pattern", "needle"}, {"path_glob", c.glob}, {"max_files", 3}}),
        context);
    ASSERT_TRUE(result.ok()) << result.status();
    std::vector<std::string> searched;
    for (const SearchRequest& request : backend.requests) {
      searched.push_back(request.path_glob);
      EXPECT_EQ(request.max_files, 3) << c.glob;
    }
    EXPECT_EQ(searched, c.searched) << c.glob;
    const std::string missed =
        absl::StrCat("No matches under path_glob \"", c.glob, "\"");
    EXPECT_EQ(result->text.rfind(missed, 0), 0u) << result->text;
    EXPECT_NE(result->text.find(c.note), std::string::npos) << result->text;
    const bool found = c.matching_scope != "nowhere";
    EXPECT_EQ(result->structured["matching_files"], found ? 1 : 0);
    EXPECT_EQ(result->text.find("src/a.cc") != std::string::npos, found);
  }
}

TEST(SearchToolsTest, StructuredContentSanitizesInvalidUtf8) {
  FakeSearchBackend backend;
  backend.invalid_utf8 = true;
  auto tools = SearchTools(backend);
  absl::Notification cancellation;
  McpToolContext context(cancellation);
  nlohmann::json args = {{"pattern", "needle"}};
  auto result = tools[0]->Call(JsonView(args), context);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(IsValidUtf8(result->text));
  EXPECT_TRUE(
      IsValidUtf8(result->structured["files"][0]["path"].get<std::string>()));
  EXPECT_TRUE(IsValidUtf8(
      result->structured["files"][0]["lines"][0]["text"].get<std::string>()));
  EXPECT_TRUE(MatchesSchema(result->structured, tools[0]->output_schema()));
}

TEST(SearchToolsTest, RejectsMalformedArgumentsAndCorrectablePatterns) {
  FakeSearchBackend backend;
  auto tools = SearchTools(backend);
  absl::Notification cancellation;
  McpToolContext context(cancellation);
  for (const nlohmann::json& args :
       {nlohmann::json::object(),
        nlohmann::json{{"pattern", "needle"}, {"max_files", -1}},
        nlohmann::json{{"pattern", "needle"}, {"shape", "other"}},
        nlohmann::json{{"pattern", "needle"}, {"context_lines", -1}},
        nlohmann::json{{"pattern", "needle"}, {"surprise", true}}}) {
    auto result = tools[0]->Call(JsonView(args), context);
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  }
  for (const nlohmann::json& args :
       {nlohmann::json{{"pattern", "("}, {"regex", true}},
        nlohmann::json{{"pattern", "needle"}, {"path_glob", "!"}}}) {
    auto result = tools[0]->Call(JsonView(args), context);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_TRUE(result->is_error);
    EXPECT_FALSE(result->text.empty());
    EXPECT_TRUE(MatchesSchema(result->structured, tools[0]->output_schema()));
  }
  backend.roots.push_back("/another");
  nlohmann::json args = {{"pattern", "needle"}};
  EXPECT_EQ(tools[0]->Call(JsonView(args), context).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(SearchToolsTest, StatusAndConcurrentCalls) {
  FakeSearchBackend backend;
  auto tools = SearchTools(backend);
  absl::Notification cancellation;
  McpToolContext context(cancellation);
  nlohmann::json empty = nlohmann::json::object();
  auto status = tools[1]->Call(JsonView(empty), context);
  ASSERT_TRUE(status.ok()) << status.status();
  EXPECT_EQ(status->structured["worktrees"][0]["index_bytes"], 42);
  nlohmann::json invalid = {{"surprise", true}};
  EXPECT_EQ(tools[1]->Call(JsonView(invalid), context).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(MatchesSchema(status->structured, tools[1]->output_schema()));
  EXPECT_NE(status->text.find("2 files"), std::string::npos);
  std::vector<std::thread> threads;
  threads.reserve(8);
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&] {
      nlohmann::json args = {{"pattern", "needle"}};
      auto result = tools[0]->Call(JsonView(args), context);
      EXPECT_TRUE(result.ok()) << result.status();
    });
  }
  for (std::thread& thread : threads) thread.join();
}

TEST(SearchToolsTest, FindSymbolFoundMissingAndPending) {
  FakeSearchBackend backend;
  backend.supports_symbols = true;
  backend.symbol_pending = 2;
  auto tools = SearchTools(backend);
  ASSERT_EQ(tools.size(), 3u);
  EXPECT_EQ(tools[1]->name(), "find_symbol");
  EXPECT_EQ(tools[1]->input_schema()["properties"]["max_results"]["default"],
            50);
  absl::Notification cancellation;
  McpToolContext context(cancellation);
  nlohmann::json found = {{"name", "Found"}};
  auto result = tools[1]->Call(JsonView(found), context);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->structured["total_count"], 1);
  EXPECT_EQ(result->structured["pending_files"], 0);
  EXPECT_EQ(result->structured["occurrences"][0]["kind"], "function");
  EXPECT_TRUE(MatchesSchema(result->structured, tools[1]->output_schema()));
  nlohmann::json missing = {{"name", "Missing"}};
  result = tools[1]->Call(JsonView(missing), context);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->structured["total_count"], 0);
  EXPECT_TRUE(result->structured["occurrences"].empty());
  EXPECT_EQ(result->structured["pending_files"], 0);
  nlohmann::json pending = {{"name", "Pending"}};
  result = tools[1]->Call(JsonView(pending), context);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->structured["pending_files"], 2);
  EXPECT_TRUE(MatchesSchema(result->structured, tools[1]->output_schema()));
  nlohmann::json invalid = {{"name", "Found"}, {"kinds", {"not_a_kind"}}};
  result = tools[1]->Call(JsonView(invalid), context);
  ASSERT_TRUE(result.ok());
  EXPECT_TRUE(result->is_error);
  EXPECT_TRUE(MatchesSchema(result->structured, tools[1]->output_schema()));
}

TEST(SearchToolsTest, FixedSearchEntryMatchesAcrossServerFlagsInBothEras) {
  using Json = nlohmann::json;
  FakeSearchBackend backend;
  for (bool modern : {false, true}) {
    std::string baseline;
    for (OutputShape shape : {OutputShape::kRaw, OutputShape::kLine,
                              OutputShape::kContext, OutputShape::kLocations}) {
      for (bool router : {false, true}) {
        auto registry = McpToolRegistry::FromTools(
            SearchTools(backend, shape, kServingConsistency, true, router));
        ASSERT_TRUE(registry.ok());
        McpServer server(*registry, {});
        ferret_testing::TestPipe input;
        ferret_testing::TestPipe output;
        RpcFrameWriter writer(input.write_fd());
        RpcFrameReader reader(output.read_fd());
        absl::Status status;
        std::thread worker(
            [&] { status = server.Serve(input.read_fd(), output.write_fd()); });
        if (!modern) {
          EXPECT_TRUE(writer
                          .Write(SerializeJson(
                              {{"jsonrpc", "2.0"},
                               {"id", 1},
                               {"method", "initialize"},
                               {"params",
                                {{"protocolVersion", "2025-11-25"},
                                 {"capabilities", Json::object()},
                                 {"clientInfo",
                                  {{"name", "test"}, {"version", "1"}}}}}}))
                          .ok());
          EXPECT_TRUE(reader.Read().ok());
        }
        Json params = Json::object();
        if (modern) {
          params["_meta"] = {
              {"io.modelcontextprotocol/protocolVersion", "2026-07-28"},
              {"io.modelcontextprotocol/clientInfo",
               {{"name", "test"}, {"version", "1"}}},
              {"io.modelcontextprotocol/clientCapabilities", Json::object()}};
        }
        EXPECT_TRUE(writer
                        .Write(SerializeJson({{"jsonrpc", "2.0"},
                                              {"id", 2},
                                              {"method", "tools/list"},
                                              {"params", params}}))
                        .ok());
        auto frame = reader.Read();
        ASSERT_TRUE(frame.ok()) << frame.status();
        const Json result = Json::parse(*frame);
        const std::string search = SerializeJson(result["result"]["tools"][0]);
        if (baseline.empty()) baseline = search;
        EXPECT_EQ(search, baseline);
        input.CloseWrite();
        worker.join();
        EXPECT_TRUE(status.ok()) << status;
      }
    }
  }
}

TEST(SearchToolsTest, RouterThroughMcpInBothProtocolEras) {
  using Json = nlohmann::json;
  FakeSearchBackend backend;
  backend.many_matches = true;
  for (bool modern : {false, true}) {
    auto registry = McpToolRegistry::FromTools(SearchTools(
        backend, OutputShape::kContext, kServingConsistency, true, true));
    ASSERT_TRUE(registry.ok());
    McpServer server(*registry, {});
    ferret_testing::TestPipe input;
    ferret_testing::TestPipe output;
    RpcFrameWriter writer(input.write_fd());
    RpcFrameReader reader(output.read_fd());
    absl::Status status;
    std::thread worker(
        [&] { status = server.Serve(input.read_fd(), output.write_fd()); });
    const Json meta = {
        {"io.modelcontextprotocol/protocolVersion", "2026-07-28"},
        {"io.modelcontextprotocol/clientInfo",
         {{"name", "test"}, {"version", "1"}}},
        {"io.modelcontextprotocol/clientCapabilities", Json::object()}};
    auto exchange = [&](const Json& message) {
      EXPECT_TRUE(writer.Write(SerializeJson(message)).ok());
      auto frame = reader.Read();
      EXPECT_TRUE(frame.ok()) << frame.status();
      return Json::parse(*frame);
    };
    if (!modern) {
      exchange({{"jsonrpc", "2.0"},
                {"id", 1},
                {"method", "initialize"},
                {"params",
                 {{"protocolVersion", "2025-11-25"},
                  {"capabilities", Json::object()},
                  {"clientInfo", {{"name", "test"}, {"version", "1"}}}}}});
    }
    const Json params = modern ? Json{{"name", "search"},
                                      {"arguments", {{"pattern", "needle"}}},
                                      {"_meta", meta}}
                               : Json{{"name", "search"},
                                      {"arguments", {{"pattern", "needle"}}}};
    Json result = exchange({{"jsonrpc", "2.0"},
                            {"id", 2},
                            {"method", "tools/call"},
                            {"params", params}});
    EXPECT_NE(SerializeJson(result).find("Structural route:"),
              std::string::npos);
    backend.precision = 0.7;
    result = exchange({{"jsonrpc", "2.0"},
                       {"id", 3},
                       {"method", "tools/call"},
                       {"params", params}});
    EXPECT_EQ(SerializeJson(result).find("Structural route:"),
              std::string::npos);
    backend.precision = 0;
    input.CloseWrite();
    worker.join();
    EXPECT_TRUE(status.ok()) << status;
  }
}

TEST(SearchToolsTest, GoldenTranscriptsForModernAndLegacy) {
  using Json = nlohmann::json;
  FakeSearchBackend backend;
  backend.supports_symbols = true;
  backend.symbol_pending = 2;
  for (const std::string era : {"modern", "legacy"}) {
    auto registry = McpToolRegistry::FromTools(SearchTools(backend));
    ASSERT_TRUE(registry.ok());
    McpServer server(*registry, {.server_version = "0.0.0-test"});
    ferret_testing::TestPipe input;
    ferret_testing::TestPipe output;
    RpcFrameWriter writer(input.write_fd());
    RpcFrameReader reader(output.read_fd());
    absl::Status status;
    std::thread worker(
        [&] { status = server.Serve(input.read_fd(), output.write_fd()); });
    const Json meta = {
        {"io.modelcontextprotocol/protocolVersion", "2026-07-28"},
        {"io.modelcontextprotocol/clientInfo",
         {{"name", "test"}, {"version", "1"}}},
        {"io.modelcontextprotocol/clientCapabilities", Json::object()}};
    const std::vector<Json> messages = {
        {{"jsonrpc", "2.0"},
         {"id", 1},
         {"method", "initialize"},
         {"params",
          {{"protocolVersion", "2025-11-25"},
           {"capabilities", Json::object()},
           {"clientInfo", {{"name", "test"}, {"version", "1"}}}}}},
        {{"jsonrpc", "2.0"},
         {"id", 2},
         {"method", "tools/list"},
         {"params", era == "modern" ? Json{{"_meta", meta}} : Json::object()}},
        {{"jsonrpc", "2.0"},
         {"id", 3},
         {"method", "tools/call"},
         {"params", era == "modern"
                        ? Json{{"name", "search"},
                               {"arguments", {{"pattern", "needle"}}},
                               {"_meta", meta}}
                        : Json{{"name", "search"},
                               {"arguments", {{"pattern", "needle"}}}}}},
        {{"jsonrpc", "2.0"},
         {"id", 4},
         {"method", "tools/call"},
         {"params", era == "modern"
                        ? Json{{"name", "index_status"}, {"_meta", meta}}
                        : Json{{"name", "index_status"}}}},
        {{"jsonrpc", "2.0"},
         {"id", 5},
         {"method", "tools/call"},
         {"params", era == "modern"
                        ? Json{{"name", "find_symbol"},
                               {"arguments", {{"name", "Found"}}},
                               {"_meta", meta}}
                        : Json{{"name", "find_symbol"},
                               {"arguments", {{"name", "Found"}}}}}},
        {{"jsonrpc", "2.0"},
         {"id", 6},
         {"method", "tools/call"},
         {"params", era == "modern"
                        ? Json{{"name", "find_symbol"},
                               {"arguments", {{"name", "Missing"}}},
                               {"_meta", meta}}
                        : Json{{"name", "find_symbol"},
                               {"arguments", {{"name", "Missing"}}}}}},
        {{"jsonrpc", "2.0"},
         {"id", 7},
         {"method", "tools/call"},
         {"params", era == "modern"
                        ? Json{{"name", "find_symbol"},
                               {"arguments", {{"name", "Pending"}}},
                               {"_meta", meta}}
                        : Json{{"name", "find_symbol"},
                               {"arguments", {{"name", "Pending"}}}}}}};
    const char* runfiles = std::getenv("TEST_SRCDIR");
    ASSERT_NE(runfiles, nullptr);
    std::ifstream expected(absl::StrCat(
        runfiles, "/_main/ferret/mcp/testdata/search_", era, ".jsonl"));
    for (const Json& message : messages) {
      EXPECT_TRUE(writer.Write(SerializeJson(message)).ok());
      auto frame = reader.Read();
      ASSERT_TRUE(frame.ok()) << frame.status();
      auto actual = ParseJson(*frame);
      ASSERT_TRUE(actual.ok()) << actual.status();
      std::string send_line;
      std::string expect_line;
      ASSERT_TRUE(static_cast<bool>(std::getline(expected, send_line)));
      ASSERT_TRUE(static_cast<bool>(std::getline(expected, expect_line)));
      EXPECT_EQ(Json::parse(send_line)["send"], message);
      EXPECT_EQ(Json::parse(expect_line)["expect"], *actual);
    }
    input.CloseWrite();
    worker.join();
    EXPECT_TRUE(status.ok()) << status;
  }
}

}  // namespace
}  // namespace ferret

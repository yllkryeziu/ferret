// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The ferret command-line tool: flag parsing and dispatch to commands.h.
//
// Exit status: 0 on success (for search, when a line matched), 1 when a
// search matched nothing, 2 on any error, as grep does.

#include <signal.h>

#include <csignal>
#include <cstring>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

#include "ferret/base/env.h"
#include "ferret/cli/commands.h"
#include "ferret/engine/engine.h"
#include "ferret/engine/ripgrep_backend.h"
#include "ferret/index/bulk_build.h"
#include "ferret/mcp/mcp_server.h"
#include "ferret/mcp/mcp_tool.h"
#include "ferret/mcp/search_tools.h"
#include "ferret/regex/compile.h"
#include "ferret/rpc/daemon.h"
#include "ferret/rpc/rpc_client.h"
#include "ferret/search/formatter.h"
#include "ferret/search/router.h"
#include "ferret/search/search_types.h"
#include "ferret/search/structural_response.h"
#include "ferret/text/lexer.h"
#include "ferret/workspace/workspace.h"

ABSL_FLAG(bool, fixed_shape, false,
          "mcp: do not expose or accept a model-supplied output shape.");
ABSL_FLAG(bool, spawn_daemon, true,
          "mcp: start a daemon if none is serving the root.");
ABSL_FLAG(bool, router, false,
          "mcp/search: enable structural routing (context only).");
ABSL_FLAG(bool, count_sample, false,
          "search: estimate totals from unverified candidates.");
ABSL_FLAG(bool, standalone, false, "mcp: run in-process over stdio.");
ABSL_FLAG(std::string, backend, "index",
          "mcp/search: index or ripgrep (mcp requires --standalone).");
ABSL_FLAG(std::string, rg_path, "",
          "mcp: path to ripgrep (then FERRET_RG, PATH).");
ABSL_FLAG(absl::Duration, idle_timeout, absl::Minutes(30),
          "serve: exit when no clients remain for this long.");
ABSL_FLAG(std::string, test_build_id, "",
          "serve: override the build id for compatibility testing.");
ABSL_FLAG(
    std::string, index_dir, "",
    "mcp: directory for per-repository indexes (default: the platform cache).");
ABSL_FLAG(int, max_concurrent_calls, 4, "mcp: concurrent tools/call limit.");
ABSL_FLAG(
    std::string, serve_consistency, "sync",
    "mcp: sync (watcher barrier) or none (benchmark without freshness wait).");
ABSL_FLAG(bool, measure_latency, false,
          "mcp/serve: log per-call server latency to stderr.");
ABSL_FLAG(std::string, index, "",
          "Index file to write or read. Default: a file per worktree under "
          "$XDG_CACHE_HOME/ferret (or ~/.cache/ferret).");
ABSL_FLAG(std::string, root, ".",
          "The git worktree to index, or whose default index to use.");
ABSL_FLAG(int, threads, ferret::BulkBuildOptions().num_threads,
          "index: threads that read and hash files.");
ABSL_FLAG(bool, regex, false,
          "search, plan: the pattern is an RE2 regular expression, not a "
          "literal string.");
ABSL_FLAG(bool, case_sensitive, false,
          "search, plan: match case exactly. Default: ASCII and Unicode "
          "simple case folding.");
ABSL_FLAG(std::string, path, "",
          "search: only files whose path matches this glob, in .gitignore "
          "syntax (\"*.h\", \"src/**/net\"); a leading '!' excludes.");
ABSL_FLAG(std::string, shape, "context",
          "search: output shape: raw (grep-like, path order), line (ranked "
          "matching lines), context (ranked, with context) or locations "
          "(ranked path:line).");
ABSL_FLAG(int, context, ferret::SearchRequest().context_lines,
          "search: lines of context before and after each match.");
ABSL_FLAG(std::optional<int>, max_lines_per_file, std::nullopt,
          absl::StrCat("search: matching lines shown per file. Default: ",
                       ferret::SearchRequest().max_lines_per_file,
                       ", or unlimited with --shape=raw."));
ABSL_FLAG(std::optional<int64_t>, max_files, std::nullopt,
          absl::StrCat("search: files shown, the best after ranking. "
                       "Default: ",
                       ferret::SearchRequest().max_files,
                       ", or unlimited with --shape=raw."));
ABSL_FLAG(std::optional<int64_t>, max_lines, std::nullopt,
          absl::StrCat("search: matching lines kept in all, the first in "
                       "verification order; the counts still include every "
                       "match. Default: ",
                       ferret::SearchRequest().max_lines,
                       ". Pass a larger value to remove the cap."));
ABSL_FLAG(int64_t, max_bytes,
          static_cast<int64_t>(ferret::FormatOptions().max_bytes),
          "search: output budget in bytes; the output says what it left out.");
ABSL_FLAG(bool, stats, false, "search: print timings to stderr.");
ABSL_FLAG(bool, explain, false,
          "plan: also show posting list sizes and candidates from the index.");

namespace ferret {
namespace {

constexpr int kExitSuccess = 0;
constexpr int kExitNoMatch = 1;
constexpr int kExitError = 2;

constexpr absl::string_view kUsage = R"(trigram code search for coding agents

Usage:
  ferret index [flags] [ROOT]    index a git worktree (default: .)
  ferret search [flags] PATTERN  search the index of --root or --index
  ferret plan [flags] PATTERN    show how a pattern becomes a trigram query
  ferret mcp [ROOT ...]     serve MCP on stdio through the daemon
  ferret mcp --standalone [ROOT ...]  run MCP in process
  ferret serve [ROOT]        serve one repository over a Unix socket

Put -- before a pattern that starts with '-'. Run `ferret --help` for flags.)";

int Fail(const absl::Status& status) {
  std::cerr << "ferret: " << status.message() << "\n";
  return kExitError;
}

int UsageError(absl::string_view message) {
  std::cerr << "ferret: " << message << "\n\n" << kUsage << "\n";
  return kExitError;
}

PatternSpec PatternFromFlags(absl::string_view pattern) {
  return PatternSpec{.pattern = std::string(pattern),
                     .literal = !absl::GetFlag(FLAGS_regex),
                     .case_sensitive = absl::GetFlag(FLAGS_case_sensitive)};
}

int RunIndex(absl::Span<char* const> args) {
  if (args.size() > 1) return UsageError("index takes at most one ROOT");
  const IndexCommandOptions options = {
      .root = args.empty() ? absl::GetFlag(FLAGS_root) : std::string(args[0]),
      .index_path = absl::GetFlag(FLAGS_index),
      .num_threads = absl::GetFlag(FLAGS_threads)};
  if (absl::Status s = RunIndexCommand(options, std::cout); !s.ok()) {
    return Fail(s);
  }
  return kExitSuccess;
}

int RunSearch(absl::Span<char* const> args) {
  if (args.size() != 1) return UsageError("search takes one PATTERN");
  absl::StatusOr<OutputShape> shape =
      ParseOutputShape(absl::GetFlag(FLAGS_shape));
  if (!shape.ok()) return Fail(shape.status());
  if (absl::GetFlag(FLAGS_max_bytes) < 0) {
    return Fail(absl::InvalidArgumentError("--max_bytes must not be negative"));
  }
  if (absl::GetFlag(FLAGS_router) && *shape != OutputShape::kContext) {
    return UsageError("--router requires --shape=context");
  }
  if (absl::GetFlag(FLAGS_max_lines).value_or(0) < 0) {
    return Fail(absl::InvalidArgumentError("--max_lines must not be negative"));
  }
  // Raw output is what grep prints: every match, in path order.
  const bool raw = *shape == OutputShape::kRaw;
  const PatternSpec pattern = PatternFromFlags(args[0]);
  const SearchRequest defaults;
  const SearchCommandOptions options = {
      .root = absl::GetFlag(FLAGS_root),
      .index_path = absl::GetFlag(FLAGS_index),
      .request =
          {.pattern = pattern.pattern,
           .regex = !pattern.literal,
           .case_sensitive = pattern.case_sensitive,
           .path_glob = absl::GetFlag(FLAGS_path),
           .context_lines = absl::GetFlag(FLAGS_context),
           .max_lines_per_file =
               absl::GetFlag(FLAGS_max_lines_per_file)
                   .value_or(raw ? std::numeric_limits<int>::max()
                                 : defaults.max_lines_per_file),
           .max_lines =
               absl::GetFlag(FLAGS_max_lines).value_or(defaults.max_lines),
           .max_files = absl::GetFlag(FLAGS_max_files)
                            .value_or(raw ? std::numeric_limits<int64_t>::max()
                                          : defaults.max_files),
           .text_only = raw,
           .count_sample = absl::GetFlag(FLAGS_count_sample) ||
                           absl::GetFlag(FLAGS_router)},
      .format = {.shape = *shape,
                 .max_bytes =
                     static_cast<size_t>(absl::GetFlag(FLAGS_max_bytes))},
      .print_stats = absl::GetFlag(FLAGS_stats),
      .router = absl::GetFlag(FLAGS_router)};
  const std::string backend_name = absl::GetFlag(FLAGS_backend);
  if (backend_name != "index" && backend_name != "ripgrep") {
    return UsageError("--backend must be index or ripgrep");
  }
  if (backend_name == "ripgrep") {
    RipgrepBackend rg(absl::GetFlag(FLAGS_rg_path));
    if (absl::Status status = rg.Register(options.root); !status.ok()) {
      return Fail(status);
    }
    absl::StatusOr<WorkspaceSearchResult> result =
        rg.Search(rg.Roots().front(), options.request, {});
    if (!result.ok()) return Fail(result.status());
    if (options.router) {
      int64_t code = 0;
      int64_t total = 0;
      for (const FileResult& file : result->response.files) {
        for (const ResultLine& line : file.lines) {
          if (!line.is_match) continue;
          ++total;
          code += line.match_class == MatchClass::kCodeIdentifier;
        }
      }
      if (total > 0) {
        result->response.token_precision =
            static_cast<double>(code) / static_cast<double>(total);
      }
    }
    std::string text = FormatSearchResponse(result->response, options.format);
    if (options.router && IsIdentifierQuery(options.request) &&
        DecideRoute(true, result->response.num_matching_files,
                    result->response.token_precision,
                    false) == SearchRoute::kStructural) {
      text = FormatStructuralResponse(result->response, {}, options.request,
                                      options.format);
    }
    std::cout << text;
    if (result->response.stats.inconsistent_files > 0) {
      std::cerr << "ferret: left out "
                << result->response.stats.inconsistent_files
                << " files that changed during the search\n";
    }
    return result->response.num_matching_lines > 0 ? kExitSuccess
                                                   : kExitNoMatch;
  }
  absl::StatusOr<bool> matched =
      RunSearchCommand(options, std::cout, std::cerr);
  if (!matched.ok()) return Fail(matched.status());
  return *matched ? kExitSuccess : kExitNoMatch;
}

// The path passed to spawn and hashed for the build handshake.
std::string ExecutablePath() {
#if defined(__APPLE__)
  uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  std::string path(size, '\0');
  if (_NSGetExecutablePath(path.data(), &size) == 0) {
    path.resize(std::strlen(path.c_str()));
    return path;
  }
  return "";
#else
  return "/proc/self/exe";
#endif
}

int RunServe(absl::Span<char* const> args) {
  if (args.size() > 1) return UsageError("serve takes at most one ROOT");
  const absl::string_view root = args.empty() ? "." : args[0];
  absl::StatusOr<std::string> build = DaemonBuildId(ExecutablePath());
  if (!build.ok()) return Fail(build.status());
  if (!absl::GetFlag(FLAGS_test_build_id).empty()) {
    *build = absl::GetFlag(FLAGS_test_build_id);
  }
  const absl::Status status = RunDaemon(
      root, absl::GetFlag(FLAGS_index_dir), *build,
      absl::GetFlag(FLAGS_idle_timeout), absl::GetFlag(FLAGS_measure_latency));
  if (absl::IsAlreadyExists(status)) return kExitSuccess;
  return status.ok() ? kExitSuccess : Fail(status);
}

int RunMcp(absl::Span<char* const> args) {
  if (absl::GetFlag(FLAGS_max_concurrent_calls) < 1) {
    return UsageError("--max_concurrent_calls must be positive");
  }
  absl::StatusOr<OutputShape> shape =
      ParseOutputShape(absl::GetFlag(FLAGS_shape));
  if (!shape.ok()) return Fail(shape.status());
  if (absl::GetFlag(FLAGS_router) && *shape != OutputShape::kContext) {
    return UsageError("--router requires --shape=context");
  }
  const std::string backend_name = absl::GetFlag(FLAGS_backend);
  if (backend_name != "index" && backend_name != "ripgrep") {
    return UsageError("--backend must be index or ripgrep");
  }
  if (backend_name == "ripgrep" && !absl::GetFlag(FLAGS_standalone)) {
    return UsageError("--backend=ripgrep requires --standalone");
  }
  Engine engine(*Env::Default(), absl::GetFlag(FLAGS_index_dir));
  RipgrepBackend ripgrep(absl::GetFlag(FLAGS_rg_path));
  RpcClient client(ExecutablePath(), absl::GetFlag(FLAGS_index_dir),
                   {.spawn_daemon = absl::GetFlag(FLAGS_spawn_daemon)});
  std::vector<std::string> roots;
  for (char* root : args) roots.emplace_back(root);
  if (roots.empty()) roots.emplace_back(".");
  for (const std::string& root : roots) {
    absl::Status status = backend_name == "ripgrep" ? ripgrep.Register(root)
                          : absl::GetFlag(FLAGS_standalone)
                              ? engine.Register(root)
                              : client.Register(root);
    if (!status.ok()) {
      return Fail(status);
    }
  }
  const std::string serve_consistency = absl::GetFlag(FLAGS_serve_consistency);
  if (serve_consistency != "sync" && serve_consistency != "none") {
    return UsageError("--serve_consistency must be sync or none");
  }
  EngineBackend backend(engine);
  const SearchBackend& selected =
      backend_name == "ripgrep" ? static_cast<const SearchBackend&>(ripgrep)
      : absl::GetFlag(FLAGS_standalone)
          ? static_cast<const SearchBackend&>(backend)
          : static_cast<const SearchBackend&>(client);
  absl::StatusOr<McpToolRegistry> tools =
      McpToolRegistry::FromTools(SearchTools(
          selected, *shape,
          serve_consistency == "sync" ? SearchConsistencyMode::kSync
                                      : SearchConsistencyMode::kNone,
          absl::GetFlag(FLAGS_fixed_shape), absl::GetFlag(FLAGS_router)));
  if (!tools.ok()) return Fail(tools.status());
  McpServerOptions options;
  options.max_concurrent_calls = absl::GetFlag(FLAGS_max_concurrent_calls);
  options.log_call_latency = absl::GetFlag(FLAGS_measure_latency);
  McpServer server(*tools, options);
  if (absl::Status status = server.Serve(0, 1); !status.ok()) {
    return Fail(status);
  }
  return kExitSuccess;
}

int RunPlan(absl::Span<char* const> args) {
  if (args.size() != 1) return UsageError("plan takes one PATTERN");
  const PlanCommandOptions options = {.pattern = PatternFromFlags(args[0]),
                                      .explain = absl::GetFlag(FLAGS_explain),
                                      .root = absl::GetFlag(FLAGS_root),
                                      .index_path = absl::GetFlag(FLAGS_index)};
  if (absl::Status s = RunPlanCommand(options, std::cout); !s.ok()) {
    return Fail(s);
  }
  return kExitSuccess;
}

// `args` are the positional arguments after the program name.
int Main(absl::Span<char* const> args) {
  if (args.empty()) return UsageError("missing command");
  const absl::string_view command = args[0];
  const absl::Span<char* const> rest = args.subspan(1);
  if (command == "index") return RunIndex(rest);
  if (command == "search") return RunSearch(rest);
  if (command == "plan") return RunPlan(rest);
  if (command == "mcp") return RunMcp(rest);
  if (command == "serve") return RunServe(rest);
  return UsageError(absl::StrCat("unknown command \"", command, "\""));
}

}  // namespace
}  // namespace ferret

int main(int argc, char** argv) {
  std::signal(SIGPIPE, SIG_IGN);  // NOLINT(misc-include-cleaner)
  absl::SetProgramUsageMessage(ferret::kUsage);
  std::vector<char*> args = absl::ParseCommandLine(argc, argv);
  return ferret::Main(absl::MakeConstSpan(args).subspan(1));
}

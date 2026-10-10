// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The ferret subcommands, separate from flag parsing (main.cc) so that tests
// can run them in-process. Each writes its normal output to `out`.

#ifndef FERRET_CLI_COMMANDS_H_
#define FERRET_CLI_COMMANDS_H_

#include <ostream>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "ferret/index/bulk_build.h"
#include "ferret/regex/compile.h"
#include "ferret/search/formatter.h"
#include "ferret/search/search_types.h"

namespace ferret {

struct IndexCommandOptions {
  // The git worktree to index.
  std::string root = ".";
  // Where to write the index; empty for DefaultIndexPath(root).
  std::string index_path;
  int num_threads = BulkBuildOptions().num_threads;
};

// `ferret index`: indexes the worktree and atomically replaces the index
// file, so concurrent searches see the old index or the new one. Writes a
// summary.
absl::Status RunIndexCommand(const IndexCommandOptions& options,
                             std::ostream& out);

struct SearchCommandOptions {
  // The worktree whose default index to search, when `index_path` is empty.
  std::string root = ".";
  std::string index_path;
  SearchRequest request;
  FormatOptions format;
  // Writes the search's statistics to `err`.
  bool print_stats = false;
  bool router = false;
};

// `ferret search`: writes the formatted response. Returns whether any line
// matched.
absl::StatusOr<bool> RunSearchCommand(const SearchCommandOptions& options,
                                      std::ostream& out, std::ostream& err);

struct PlanCommandOptions {
  PatternSpec pattern;
  // Also shows each trigram's posting list size and the candidates, from the
  // index of `root` or at `index_path`.
  bool explain = false;
  std::string root = ".";
  std::string index_path;
};

// `ferret plan`: shows how a pattern compiles: its trigram query, required
// literal, and flags.
absl::Status RunPlanCommand(const PlanCommandOptions& options,
                            std::ostream& out);

}  // namespace ferret

#endif  // FERRET_CLI_COMMANDS_H_

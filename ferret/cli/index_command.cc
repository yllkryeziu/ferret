// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "ferret/base/env.h"
#include "ferret/cli/command_util.h"
#include "ferret/cli/commands.h"
#include "ferret/index/bulk_build.h"
#include "ferret/text/eligibility.h"

namespace ferret {
namespace {

absl::string_view Dirname(absl::string_view path) {
  const size_t slash = path.rfind('/');
  if (slash == absl::string_view::npos) return ".";
  return slash == 0 ? "/" : path.substr(0, slash);
}

void PrintSummary(const BulkBuildStats& stats, absl::string_view index_path,
                  absl::Duration elapsed, std::ostream& out) {
  out << "Indexed " << stats.files_indexed << " of " << stats.files_listed
      << " files (" << HumanBytes(stats.bytes_indexed) << ") into "
      << index_path << " ("
      << HumanBytes(static_cast<int64_t>(stats.segment.file_bytes)) << ") in "
      << absl::FormatDuration(absl::Trunc(elapsed, absl::Milliseconds(1)))
      << "\n";
  std::vector<std::string> skipped;
  for (int i = 0; i < kNumSkipReasons; ++i) {
    const auto reason = static_cast<SkipReason>(i);
    if (reason == SkipReason::kNone) continue;
    if (const int64_t n = stats.files_skipped[static_cast<size_t>(i)]; n > 0) {
      skipped.push_back(absl::StrCat(n, " ", SkipReasonName(reason)));
    }
  }
  if (stats.files_unreadable > 0) {
    skipped.push_back(absl::StrCat(stats.files_unreadable, " unreadable"));
  }
  if (!skipped.empty()) {
    out << "  skipped: " << absl::StrJoin(skipped, ", ") << "\n";
  }
  out << "  segment: " << stats.segment.num_docs << " docs, "
      << stats.segment.num_chunks << " chunks, " << stats.segment.num_trigrams
      << " trigrams, " << stats.segment.num_postings << " postings ("
      << HumanBytes(static_cast<int64_t>(stats.segment.postings_bytes))
      << ")\n";
}

}  // namespace

absl::Status RunIndexCommand(const IndexCommandOptions& options,
                             std::ostream& out) {
  ABSL_ASSIGN_OR_RETURN(const std::string index_path,
                        ResolveIndexPath(options.root, options.index_path));
  Env& env = *Env::Default();
  const absl::string_view dir = Dirname(index_path);
  ABSL_RETURN_IF_ERROR(env.CreateDir(dir));

  const absl::Time start = absl::Now();
  const std::string temp_path = absl::StrCat(index_path, ".tmp.", getpid());
  absl::StatusOr<BulkBuildStats> stats = BuildSegmentFromWorktree(
      env, options.root, temp_path, {.num_threads = options.num_threads});
  if (!stats.ok()) {
    // Best effort: the build failed, and the partial file is garbage.
    env.DeleteFile(temp_path).IgnoreError();
    return stats.status();
  }
  ABSL_RETURN_IF_ERROR(env.RenameFile(temp_path, index_path));
  ABSL_RETURN_IF_ERROR(env.SyncDir(dir));
  PrintSummary(*stats, index_path, absl::Now() - start, out);
  return absl::OkStatus();
}

}  // namespace ferret

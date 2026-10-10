// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Symbol index measurements on small corpora. The input root is data, never
// built in place.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "ferret/base/env.h"
#include "ferret/index/bulk_build.h"
#include "ferret/search/search_types.h"
#include "ferret/workspace/workspace.h"

namespace ferret {
namespace {

int64_t Percentile(std::vector<int64_t> samples, int hundredths) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() * hundredths / 100];
}

absl::Status Measure(absl::string_view root, absl::string_view output,
                     absl::string_view name) {
  Env& env = *Env::Default();
  ABSL_RETURN_IF_ERROR(env.CreateDir(output));
  for (bool symbols : {false, true}) {
    const std::string section_path =
        absl::StrCat(output, symbols ? "/with.fseg" : "/without.fseg");
    BulkBuildOptions bulk;
    bulk.writer.write_symbols = symbols;
    const absl::Time started = absl::Now();
    ABSL_ASSIGN_OR_RETURN(
        BulkBuildStats built,
        BuildSegmentFromWorktree(env, root, section_path, bulk));
    std::printf(
        "bulk symbols=%d files=%lld bytes=%lld size=%llu wall_us=%lld "
        "scan_us=%lld threads=%d\n",
        symbols, static_cast<long long>(built.files_indexed),
        static_cast<long long>(built.bytes_indexed),
        static_cast<unsigned long long>(built.segment.file_bytes),
        static_cast<long long>(
            absl::ToInt64Microseconds(absl::Now() - started)),
        static_cast<long long>(absl::ToInt64Microseconds(built.scan_time)),
        bulk.num_threads);
  }
  for (bool symbols : {false, true}) {
    WorkspaceOptions options;
    options.index.memtable.segment.write_symbols = symbols;
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<Workspace> workspace,
        Workspace::Open(env,
                        absl::StrCat(output, symbols ? "/with" : "/without"),
                        options));
    ABSL_ASSIGN_OR_RETURN(WorktreeId id, workspace->Register(root));
    if (symbols) {
      const absl::Time deadline = absl::Now() + absl::Minutes(2);
      while (workspace->Stats().index.pending_symbols != 0 &&
             absl::Now() < deadline) {
        absl::SleepFor(absl::Milliseconds(5));
      }
      if (workspace->Stats().index.pending_symbols != 0) {
        return absl::DeadlineExceededError("bulk symbols still pending");
      }
      std::vector<int64_t> lookups;
      lookups.reserve(1000);
      for (int i = 0; i < 1000; ++i) {
        const absl::Time begin = absl::Now();
        ABSL_RETURN_IF_ERROR(
            workspace
                ->FindSymbol(id, {.name = std::string(name)},
                             {.mode = SearchConsistencyMode::kNone})
                .status());
        lookups.push_back(absl::ToInt64Nanoseconds(absl::Now() - begin));
      }
      std::printf("find_symbol kNone calls=1000 p50_ns=%lld p99_ns=%lld\n",
                  static_cast<long long>(Percentile(lookups, 50)),
                  static_cast<long long>(Percentile(lookups, 99)));
    }

    // The independent Notify path measures only publication + text visibility.
    std::vector<int64_t> notify;
    notify.reserve(30);
    for (int i = 0; i < 30; ++i) {
      const std::string token = absl::StrCat("notify_symbol_", symbols, "_", i);
      const std::string content =
          absl::StrCat("int ", token, "() { return 1; }\n");
      const std::string path = absl::StrCat(root, "/symbol_latency.cc");
      FILE* output_file = std::fopen(path.c_str(), "wb");
      if (output_file == nullptr) return absl::InternalError("open failed");
      const size_t written =
          std::fwrite(content.data(), 1, content.size(), output_file);
      const int closed = std::fclose(output_file);
      if (written != content.size() || closed != 0) {
        return absl::InternalError("write failed");
      }
      const absl::Time notify_start = absl::Now();
      ABSL_ASSIGN_OR_RETURN(uint64_t generation,
                            workspace->Notify(id, {"symbol_latency.cc"}));
      const SearchRequest search{.pattern = token, .context_lines = 0};
      const SearchConsistency consistency{
          .mode = SearchConsistencyMode::kNotify,
          .min_generation = generation,
          .timeout = absl::Seconds(10)};
      ABSL_ASSIGN_OR_RETURN(WorkspaceSearchResult result,
                            workspace->Search(id, search, consistency));
      if (result.stale || result.response.num_matching_files != 1) {
        return absl::DataLossError("notified content was not searchable");
      }
      notify.push_back(absl::ToInt64Microseconds(absl::Now() - notify_start));
    }
    std::printf("notify symbols=%d calls=30 p50_us=%lld p99_us=%lld\n", symbols,
                static_cast<long long>(Percentile(notify, 50)),
                static_cast<long long>(Percentile(notify, 99)));
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace ferret

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr,
                 "usage: index_symbol_benchmark ROOT OUTPUT_DIR NAME\n");
    return 2;
  }
  const absl::Status status = ferret::Measure(argv[1], argv[2], argv[3]);
  if (!status.ok()) {
    std::fprintf(stderr, "symbol benchmark failed: %s\n",
                 status.ToString().c_str());
    return 1;
  }
  return 0;
}

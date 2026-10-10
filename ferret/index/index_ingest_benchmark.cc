// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Measures three ingest routes and an optional reference-count GC workload.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

#include "ferret/base/env.h"
#include "ferret/git/worktree_files.h"
#include "ferret/index/blob_registry.h"
#include "ferret/index/bulk_build.h"
#include "ferret/index/index.h"
#include "ferret/index/prepared_document.h"
#include "ferret/index/segment_writer.h"
#include "ferret/text/eligibility.h"
#include "ferret/text/trigram_extractor.h"

namespace ferret {
namespace {

constexpr int64_t kBatchBytes = int64_t{256} * 1024 * 1024;

struct InputFile {
  std::string path;
  PreparedDocument prepared;
};

absl::StatusOr<std::vector<InputFile>> ScanFiles(Env& env,
                                                 absl::string_view root) {
  ABSL_ASSIGN_OR_RETURN(std::vector<std::string> paths,
                        ListWorktreeFiles(root));
  TrigramExtractor extractor;
  std::vector<InputFile> files;
  for (const std::string& path : paths) {
    auto bytes = env.ReadFile(absl::StrCat(root, "/", path));
    if (!bytes.ok()) continue;
    PreparedDocument prepared = PrepareDocument(path, *bytes, {}, extractor);
    if (prepared.skip_reason == SkipReason::kNone) {
      files.push_back({path, prepared});
    }
  }
  return files;
}

absl::Status IngestMemtables(Env& env, Index& index, absl::string_view root,
                             const std::vector<InputFile>& files,
                             int64_t& content_bytes) {
  for (const InputFile& file : files) {
    ABSL_ASSIGN_OR_RETURN(std::string bytes,
                          env.ReadFile(absl::StrCat(root, "/", file.path)));
    ABSL_RETURN_IF_ERROR(index.Add(file.prepared.Bind(file.path, bytes)));
    content_bytes += static_cast<int64_t>(bytes.size());
  }
  return index.Flush();
}

absl::Status IngestSegments(Env& env, Index& index, absl::string_view root,
                            std::vector<InputFile> files,
                            int64_t& content_bytes) {
  std::sort(files.begin(), files.end(),
            [](const InputFile& a, const InputFile& b) {
              return std::tie(a.prepared.rank_key, a.path) <
                     std::tie(b.prepared.rank_key, b.path);
            });
  size_t begin = 0;
  while (begin < files.size()) {
    int64_t batch_bytes = 0;
    const auto written = index.AddSegment([&](SegmentWriter& writer)
                                              -> absl::Status {
      while (begin < files.size() && batch_bytes < kBatchBytes) {
        const InputFile& file = files[begin];
        ABSL_ASSIGN_OR_RETURN(std::string bytes,
                              env.ReadFile(absl::StrCat(root, "/", file.path)));
        ABSL_RETURN_IF_ERROR(writer.Add(file.prepared.Bind(file.path, bytes)));
        batch_bytes += static_cast<int64_t>(bytes.size());
        ++begin;
      }
      return absl::OkStatus();
    });
    if (!written.ok()) return written.status();
    content_bytes += batch_bytes;
  }
  return absl::OkStatus();
}

void Report(absl::string_view name, int64_t content_bytes, uint64_t flush_bytes,
            uint64_t bulk_bytes, uint64_t compaction_bytes,
            absl::Duration wall) {
  const uint64_t installed = flush_bytes + bulk_bytes + compaction_bytes;
  std::printf(
      "%.*s: wall=%.2fs content=%lld installed=%llu "
      "flush=%llu bulk=%llu compaction=%llu write_amplification=%.3f\n",
      static_cast<int>(name.size()), name.data(), absl::ToDoubleSeconds(wall),
      static_cast<long long>(content_bytes),
      static_cast<unsigned long long>(installed),
      static_cast<unsigned long long>(flush_bytes),
      static_cast<unsigned long long>(bulk_bytes),
      static_cast<unsigned long long>(compaction_bytes),
      content_bytes == 0 ? 0.0
                         : static_cast<double>(installed) /
                               static_cast<double>(content_bytes));
}

// Measures 1,000 real reference edits without making any blob drop-eligible.
absl::Status MeasureReferenceLatency(Index& index,
                                     absl::Span<const BlobId> ids) {
  for (size_t batch_size : {size_t{1}, size_t{256}}) {
    if (ids.size() < batch_size) continue;
    std::vector<Index::ReferenceDelta> edits;
    edits.reserve(batch_size);
    std::vector<int64_t> nanos;
    nanos.reserve(1000);
    for (int call = 0; call < 1000; ++call) {
      edits.clear();
      for (size_t i = 0; i < batch_size; ++i) {
        edits.push_back({ids[i], call % 2 == 0 ? 1 : -1});
      }
      const absl::Time start = absl::Now();
      ABSL_RETURN_IF_ERROR(index.AdjustReferences(edits));
      nanos.push_back(absl::ToInt64Nanoseconds(absl::Now() - start));
    }
    std::sort(nanos.begin(), nanos.end());
    std::printf("adjust: batch=%zu calls=1000 p50=%lldns p99=%lldns\n",
                batch_size, static_cast<long long>(nanos[500]),
                static_cast<long long>(nanos[990]));
  }
  std::vector<int64_t> nanos;
  nanos.reserve(1000);
  for (int call = 0; call < 1000; ++call) {
    const absl::Time start = absl::Now();
    static_cast<void>(index.Stats());
    nanos.push_back(absl::ToInt64Nanoseconds(absl::Now() - start));
  }
  std::sort(nanos.begin(), nanos.end());
  std::printf("stats: calls=1000 p50=%lldns p99=%lldns\n",
              static_cast<long long>(nanos[500]),
              static_cast<long long>(nanos[990]));
  return absl::OkStatus();
}

absl::Status MeasureGc(absl::string_view root, absl::string_view output) {
  Env& env = *Env::Default();
  ABSL_RETURN_IF_ERROR(env.CreateDir(output));
  ABSL_ASSIGN_OR_RETURN(const std::vector<InputFile> files,
                        ScanFiles(env, root));
  IndexOptions options;
  options.gc_grace_period = absl::ZeroDuration();
  ABSL_ASSIGN_OR_RETURN(auto opened,
                        Index::Open(env, absl::StrCat(output, "/gc"), options));
  std::unique_ptr<Index> index = std::move(opened.first);
  int64_t content_bytes = 0;
  ABSL_RETURN_IF_ERROR(IngestSegments(env, *index, root, files, content_bytes));
  index->WaitForCompactions();
  const IndexStats before = index->Stats();
  std::vector<BlobId> ids;
  ids.reserve(files.size());
  for (const InputFile& file : files) {
    ids.push_back(index->blobs().Intern(file.prepared.rank_key.oid));
  }
  std::vector<Index::ReferenceDelta> changes;
  changes.reserve(ids.size());
  for (BlobId id : ids) changes.push_back({id, 1});
  ABSL_RETURN_IF_ERROR(index->AdjustReferences(changes));
  index->ResumeGc();
  ABSL_RETURN_IF_ERROR(MeasureReferenceLatency(*index, ids));
  std::vector<size_t> order(ids.size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937_64 rng(0x9a73);
  std::shuffle(order.begin(), order.end(), rng);
  changes.clear();
  for (size_t i = 0; i < order.size() * 30 / 100; ++i) {
    changes.push_back({ids[order[i]], -1});
  }
  const absl::Time start = absl::Now();
  ABSL_RETURN_IF_ERROR(index->AdjustReferences(changes));
  index->WaitForCompactions();
  const IndexStats after = index->Stats();
  std::printf(
      "gc: files=%zu dropped=%zu reclaimed_bytes=%llu "
      "compaction_bytes=%llu write_amplification=%.3f time=%.3fs "
      "remaining_segments=%zu dead_docs=%llu scan=%.3fms\n",
      files.size(), changes.size(),
      static_cast<unsigned long long>(after.reclaimed_bytes),
      static_cast<unsigned long long>(after.compaction_bytes -
                                      before.compaction_bytes),
      before.bulk_segment_bytes == 0
          ? 0.0
          : static_cast<double>(after.compaction_bytes -
                                before.compaction_bytes) /
                static_cast<double>(before.bulk_segment_bytes),
      absl::ToDoubleSeconds(absl::Now() - start), after.segment_count,
      static_cast<unsigned long long>(after.dead_docs),
      absl::ToDoubleMilliseconds(after.gc_scan_time));
  return absl::OkStatus();
}

absl::Status Measure(absl::string_view root, absl::string_view output) {
  Env& env = *Env::Default();
  ABSL_RETURN_IF_ERROR(env.CreateDir(output));
  ABSL_ASSIGN_OR_RETURN(const std::vector<InputFile> files,
                        ScanFiles(env, root));
  for (int mode = 0; mode < 2; ++mode) {
    const char* name = mode == 0 ? "memtable" : "AddSegment";
    ABSL_ASSIGN_OR_RETURN(auto opened,
                          Index::Open(env, absl::StrCat(output, "/", name)));
    std::unique_ptr<Index> index = std::move(opened.first);
    int64_t content_bytes = 0;
    const absl::Time start = absl::Now();
    ABSL_RETURN_IF_ERROR(
        mode == 0 ? IngestMemtables(env, *index, root, files, content_bytes)
                  : IngestSegments(env, *index, root, files, content_bytes));
    index->WaitForCompactions();
    const IndexStats stats = index->Stats();
    if (!stats.last_compaction_error.ok()) return stats.last_compaction_error;
    Report(name, content_bytes, stats.flush_bytes, stats.bulk_segment_bytes,
           stats.compaction_bytes, absl::Now() - start);
  }
  const absl::Time start = absl::Now();
  ABSL_ASSIGN_OR_RETURN(
      const BulkBuildStats stats,
      BuildSegmentFromWorktree(env, root, absl::StrCat(output, "/build.fseg")));
  Report("BuildSegmentFromWorktree", stats.bytes_indexed, 0,
         stats.segment.file_bytes, 0, absl::Now() - start);
  return absl::OkStatus();
}

}  // namespace
}  // namespace ferret

int main(int argc, char** argv) {
  if ((argc != 3 && argc != 4) ||
      (argc == 4 && absl::string_view(argv[3]) != "gc")) {
    std::fprintf(stderr,
                 "usage: index_ingest_benchmark ROOT OUTPUT_DIR [gc]\n");
    return 2;
  }
  const absl::Status status = argc == 4 ? ferret::MeasureGc(argv[1], argv[2])
                                        : ferret::Measure(argv[1], argv[2]);
  if (!status.ok()) {
    std::fprintf(stderr, "benchmark failed: %s\n", status.ToString().c_str());
    return 1;
  }
  return 0;
}

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Builds one segment from the files of a directory.
//
// Opening a file costs far more than reading it when there are hundreds of
// thousands of them, so every file is read once. Pass 1 reads the files on
// all threads. For each eligible one it computes the RankKey, then chunks
// the content and extracts its trigrams and code tokens under provisional
// ids (DocumentPrecomputer), and copies the content to a scratch file. The
// files are then sorted into rank order. Pass 2 reads the contents back from
// the scratch file, in rank order, into the SegmentWriter; the writer maps the
// provisional ids to the ranked ones. Pass 3 extracts symbols from the same
// contents. Finish merges the workers' postings and code tokens, in parallel.
// The segment has the bytes a serial SegmentWriter::Add() loop writes. Peak
// memory is the postings (about 4 bytes each), the code-token maps and a
// window of contents; the scratch file is as large as the content.
//
// Each path becomes its own document, even when two paths hold the same
// blob. Sharing blobs between paths and worktrees is the job of views.

#ifndef FERRET_INDEX_BULK_BUILD_H_
#define FERRET_INDEX_BULK_BUILD_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

#include "ferret/base/env.h"
#include "ferret/base/thread_pool.h"
#include "ferret/index/segment_writer.h"
#include "ferret/text/eligibility.h"

namespace ferret {

struct BulkBuildOptions {
  EligibilityOptions eligibility;
  SegmentWriterOptions writer;
  // Threads for every pass, including the caller. At least 1. Pass 2 reads
  // from a file and uses twice as many.
  int num_threads = 4;
  // The scheduling class of the worker threads. The default keeps them on
  // the performance cores; kUtility yields to interactive work but runs the
  // build several times slower on a machine with efficiency cores.
  ThreadQos thread_qos = ThreadQos::kUserInteractive;
  // Content bytes that pass 2 and pass 3 read as one unit (they always read
  // at least one file). At least 1.
  int64_t partition_bytes = int64_t{16} << 20;
  // Units in flight or finished and waiting for the writer; 0 means two more
  // than the workers.
  int window = 0;
  // Scan workers (pass 1); 0 means `num_threads`.
  int scan_threads = 0;
  // Threads that extract symbols while the scan and pass 2 run; 0 means
  // `num_threads`.
  int symbol_threads = 0;
  // Their scheduling class. Utility puts them on the efficiency cores, where
  // the symbols took about twice the thread time and the build ran longer.
  ThreadQos symbol_qos = ThreadQos::kUserInteractive;
  // Reads every file twice and writes the documents one at a time through
  // SegmentWriter::Add(), as the build once did: the reference the default
  // build must match byte for byte. A file that changes between the passes
  // is then counted as unreadable.
  bool serial = false;
};

struct BulkBuildStats {
  // Paths given or listed.
  int64_t files_listed = 0;
  int64_t files_indexed = 0;
  // Files skipped by the eligibility rules, indexed by SkipReason (entry
  // kNone stays 0).
  std::array<int64_t, kNumSkipReasons> files_skipped = {};
  // Files that could not be read, such as tracked files deleted from the
  // worktree, or that changed between the two passes.
  int64_t files_unreadable = 0;
  int64_t bytes_indexed = 0;
  SegmentStats segment;
  absl::Duration list_time;
  absl::Duration scan_time;
  absl::Duration write_time;
  // Phase breakdown. Wall time unless noted; "thread" rows sum the time every
  // thread spent, so they can exceed the wall time.
  absl::Duration sort_time;
  // Symbols, wall time from the first file queued to the last extracted;
  // zero when symbols are off. They run beside the other passes.
  absl::Duration symbols_time;
  // How long the build waited for the symbols after pass 2 had finished.
  absl::Duration symbols_wait_time;
  // Finish, wall time.
  absl::Duration finish_time;
  int64_t partitions = 0;
  // Pass 2, wall time: reading the contents back and appending them.
  absl::Duration assemble_time;
  // Pass 1, thread time: copying contents to the scratch file.
  absl::Duration pack_thread_time;
  // Pass 1, thread time: reading, eligibility + hash + rank key.
  absl::Duration scan_read_thread_time;
  absl::Duration scan_prepare_thread_time;
  // Pass 3, thread time: rereading and parsing.
  absl::Duration symbols_thread_time;
  absl::Duration symbols_read_thread_time;
  // Pass 2, thread time: rereading and hashing (serial build), or reading
  // the scratch file (default build).
  absl::Duration reread_thread_time;
  absl::Duration verify_hash_thread_time;
  // The process's peak resident size (bytes) after pass 1, after pass 2 and
  // after Finish. It includes what the process held before the build.
  int64_t peak_rss_after_scan = 0;
  int64_t peak_rss_after_assemble = 0;
  int64_t peak_rss_after_finish = 0;
  // Pass 2: the writer's per-document and Finish phases.
  SegmentWriterTimings writer;
};

// Indexes the files at `paths`, relative to directory `root`, into a new
// segment at `segment_path`. Returns the first error from the writer or the
// file system other than a file that cannot be read.
absl::StatusOr<BulkBuildStats> BuildSegment(
    Env& env, absl::string_view root, absl::Span<const std::string> paths,
    absl::string_view segment_path, const BulkBuildOptions& options = {});

// Indexes the git worktree at `root`: the files ListWorktreeFiles() returns.
absl::StatusOr<BulkBuildStats> BuildSegmentFromWorktree(
    Env& env, absl::string_view root, absl::string_view segment_path,
    const BulkBuildOptions& options = {});

}  // namespace ferret

#endif  // FERRET_INDEX_BULK_BUILD_H_

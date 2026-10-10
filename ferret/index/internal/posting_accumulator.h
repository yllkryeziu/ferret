// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Collects the postings of a segment under construction and returns them as
// posting lists in trigram order.
//
// Chunks arrive in increasing ChunkId order. Their (trigram, chunk) pairs are
// buffered in a partition; a full partition is sorted by trigram with a
// stable two-pass radix sort, which keeps each trigram's chunks in arrival,
// hence increasing, order, and is kept as a run of lists, each stored as
// varint deltas (a bulk build holds hundreds of millions of postings, and the
// deltas of the dense lists are one byte). At the end, the runs are merged by
// trigram, and each list is the concatenation of its pieces in run order, which
// is already sorted.

#ifndef FERRET_INDEX_INTERNAL_POSTING_ACCUMULATOR_H_
#define FERRET_INDEX_INTERNAL_POSTING_ACCUMULATOR_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

#include "ferret/postings/ids.h"
#include "ferret/text/trigram.h"

namespace ferret::segment_internal {

// Thread-compatible.
class PostingAccumulator final {
 public:
  // The posting lists of a contiguous range of chunks.
  struct Run {
    // Strictly increasing.
    std::vector<Trigram> trigrams;
    // List i has ends[i] - ends[i - 1] ids (ends[-1] is 0); ends[i] is the
    // total of lists 0..i. Its bytes in `packed` are
    // [byte_ends[i - 1], byte_ends[i]): the first id, then each id minus the
    // one before it, as LEB128 varints.
    std::vector<uint32_t> ends;
    std::vector<uint32_t> byte_ends;
    std::string packed;

    size_t list_size(size_t i) const {
      return ends[i] - (i == 0 ? 0 : ends[i - 1]);
    }
    // Writes the ids of list `i` to `dst` (room for list_size(i)) and returns
    // the end of what it wrote.
    ChunkId* DecodeList(size_t i, ChunkId* dst) const;
  };

  // Sorts a partition once it holds `max_partition_postings` postings; peak
  // memory for the partition is 16 bytes per posting. Requires
  // 0 < max_partition_postings <= 2^28.
  explicit PostingAccumulator(size_t max_partition_postings);

  PostingAccumulator(const PostingAccumulator&) = delete;
  PostingAccumulator& operator=(const PostingAccumulator&) = delete;

  // Adds a posting for each of `trigrams`, which are distinct, in chunk
  // `chunk`. Requires `chunk` to be larger than every chunk added before
  // (DCHECKed).
  void AddChunk(ChunkId chunk, absl::Span<const Trigram> trigrams);

  // The number of postings added.
  uint64_t num_postings() const { return num_postings_; }

  // Sorts the pending postings and returns every run, in increasing chunk
  // order, leaving the accumulator empty. A run holds the lists of a
  // contiguous range of chunks; concatenating the lists of one trigram over
  // the runs in order gives its posting list. For a caller that merges runs
  // itself instead of calling ForEachList().
  std::vector<Run> TakeRuns();

  // Calls `fn(trigram, ids)` for every trigram that has postings, in
  // increasing trigram order, with its strictly increasing chunk ids. Stops
  // at and returns the first error `fn` returns. Consumes the postings: call
  // it once, after the last AddChunk().
  absl::Status ForEachList(
      absl::FunctionRef<absl::Status(Trigram, absl::Span<const ChunkId>)> fn);

 private:
  // Sorts the pending pairs into a new run.
  void FlushPartition();

  const size_t max_partition_postings_;
  // Pending postings as (trigram << 32) | chunk, in chunk order.
  std::vector<uint64_t> pairs_;
  std::vector<Run> runs_;
  uint64_t num_postings_ = 0;
  bool has_chunk_ = false;
  ChunkId last_chunk_ = 0;
};

}  // namespace ferret::segment_internal

#endif  // FERRET_INDEX_INTERNAL_POSTING_ACCUMULATOR_H_

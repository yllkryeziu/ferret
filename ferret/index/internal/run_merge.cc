// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/internal/run_merge.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "ferret/base/thread_pool.h"
#include "ferret/index/internal/ordered_parallel.h"
#include "ferret/index/internal/posting_accumulator.h"
#include "ferret/postings/ids.h"
#include "ferret/postings/posting_list.h"
#include "ferret/postings/posting_list_encoder.h"
#include "ferret/text/trigram.h"

namespace ferret::segment_internal {
namespace {

// Trigrams per task. Wide enough to amortize the search of every run, narrow
// enough to balance the dense ASCII region.
constexpr uint32_t kRangeBits = 10;
constexpr uint32_t kRangeWidth = uint32_t{1} << kRangeBits;

struct ListMeta {
  Trigram trigram;
  uint32_t count;
  uint64_t inline_postings;
  // The encoding is bytes[begin, end) of the task's output.
  uint32_t begin;
  uint32_t end;
};

struct TaskOutput {
  std::vector<ListMeta> lists;
  std::string bytes;
};

struct Scratch {
  std::vector<ChunkId> ids;
  std::vector<ChunkId> temp;
};

constexpr size_t kRadixThreshold = 1024;
constexpr int kRadixBits = 11;

// Sorts `ids`, all below 2^id_bits. Short lists use std::sort; long ones an
// LSD radix sort on kRadixBits digits, using `temp` as scratch.
void SortIds(absl::Span<ChunkId> ids, int id_bits, std::vector<ChunkId>& temp) {
  if (ids.size() <= kRadixThreshold) {
    std::sort(ids.begin(), ids.end());
    return;
  }
  temp.resize(ids.size());
  ChunkId* src = ids.data();
  ChunkId* dst = temp.data();
  const int passes = (id_bits + kRadixBits - 1) / kRadixBits;
  for (int pass = 0; pass < passes; ++pass) {
    const int shift = pass * kRadixBits;
    size_t starts[(1 << kRadixBits) + 1] = {};
    for (size_t i = 0; i < ids.size(); ++i) {
      ++starts[((src[i] >> shift) & ((1 << kRadixBits) - 1)) + 1];
    }
    for (size_t b = 1; b <= (1 << kRadixBits); ++b) starts[b] += starts[b - 1];
    for (size_t i = 0; i < ids.size(); ++i) {
      dst[starts[(src[i] >> shift) & ((1 << kRadixBits) - 1)]++] = src[i];
    }
    std::swap(src, dst);
  }
  if (src != ids.data()) std::copy(src, src + ids.size(), ids.data());
}

}  // namespace

absl::Status MergeRuns(absl::Span<const PostingAccumulator::Run> runs,
                       absl::Span<const ChunkId> remap, ThreadPool* pool,
                       int threads, ListSink sink) {
  const int id_bits = std::max(1, std::bit_width(remap.size()));
  std::vector<Scratch> scratch(static_cast<size_t>(std::max(threads, 1)));
  // Skip the ranges below the smallest and above the largest trigram.
  Trigram min_trigram = kTrigramMask;
  Trigram max_trigram = 0;
  bool any = false;
  for (const PostingAccumulator::Run& run : runs) {
    if (run.trigrams.empty()) continue;
    any = true;
    min_trigram = std::min(min_trigram, run.trigrams.front());
    max_trigram = std::max(max_trigram, run.trigrams.back());
  }
  if (!any) return absl::OkStatus();
  const int64_t first_range = min_trigram >> kRangeBits;
  const int64_t last_range = max_trigram >> kRangeBits;

  const auto produce = [&](int worker,
                           int64_t index) -> absl::StatusOr<TaskOutput> {
    TaskOutput out;
    const Trigram lo =
        static_cast<Trigram>((first_range + index) << kRangeBits);
    const Trigram hi = lo + kRangeWidth;  // exclusive, at most 2^24
    // Per run, the lists with trigram in [lo, hi).
    struct Span {
      size_t begin;
      size_t end;
    };
    std::vector<Span> spans(runs.size());
    uint32_t counts[kRangeWidth + 1] = {};
    bool has_postings = false;
    for (size_t r = 0; r < runs.size(); ++r) {
      const PostingAccumulator::Run& run = runs[r];
      const auto first =
          std::lower_bound(run.trigrams.begin(), run.trigrams.end(), lo);
      const auto last = std::lower_bound(first, run.trigrams.end(), hi);
      spans[r] = {static_cast<size_t>(first - run.trigrams.begin()),
                  static_cast<size_t>(last - run.trigrams.begin())};
      for (size_t j = spans[r].begin; j < spans[r].end; ++j) {
        counts[run.trigrams[j] - lo + 1] += run.list_size(j);
        has_postings = true;
      }
    }
    if (!has_postings) return out;
    for (uint32_t t = 1; t <= kRangeWidth; ++t) counts[t] += counts[t - 1];
    std::vector<ChunkId>& ids = scratch[static_cast<size_t>(worker)].ids;
    ids.resize(counts[kRangeWidth]);
    std::vector<uint32_t> cursor(counts, counts + kRangeWidth);
    for (size_t r = 0; r < runs.size(); ++r) {
      const PostingAccumulator::Run& run = runs[r];
      for (size_t j = spans[r].begin; j < spans[r].end; ++j) {
        ChunkId* const first = ids.data() + cursor[run.trigrams[j] - lo];
        ChunkId* const last = run.DecodeList(j, first);
        for (ChunkId* id = first; id != last; ++id) *id = remap[*id];
        cursor[run.trigrams[j] - lo] += static_cast<uint32_t>(last - first);
      }
    }
    for (uint32_t t = 0; t < kRangeWidth; ++t) {
      const size_t count = counts[t + 1] - counts[t];
      if (count == 0) continue;
      if (count > kMaxPostingListSize) {
        return absl::InvalidArgumentError("invalid posting list");
      }
      const absl::Span<ChunkId> list(ids.data() + counts[t], count);
      SortIds(list, id_bits, scratch[static_cast<size_t>(worker)].temp);
      ListMeta meta = {.trigram = lo + t,
                       .count = static_cast<uint32_t>(count),
                       .inline_postings = 0,
                       .begin = 0,
                       .end = 0};
      if (IsInlinedPostingList(count)) {
        meta.inline_postings = list[0];
        if (count == 2) meta.inline_postings |= uint64_t{list[1]} << 32;
      } else {
        meta.begin = static_cast<uint32_t>(out.bytes.size());
        AppendEncodedPostingList(list, out.bytes);
        meta.end = static_cast<uint32_t>(out.bytes.size());
      }
      out.lists.push_back(meta);
    }
    return out;
  };
  const auto commit = [&](int64_t, TaskOutput&& out) -> absl::Status {
    for (const ListMeta& meta : out.lists) {
      ABSL_RETURN_IF_ERROR(sink(meta.trigram, meta.count, meta.inline_postings,
                                absl::string_view(out.bytes).substr(
                                    meta.begin, meta.end - meta.begin)));
    }
    return absl::OkStatus();
  };
  return ParallelOrdered<TaskOutput>(pool, threads,
                                     last_range - first_range + 1, threads + 2,
                                     produce, commit);
}

}  // namespace ferret::segment_internal

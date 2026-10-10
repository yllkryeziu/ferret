// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/internal/posting_accumulator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/types/span.h"

#include "ferret/postings/ids.h"
#include "ferret/text/trigram.h"

namespace ferret::segment_internal {
namespace {

constexpr int kRadixBits = 12;
constexpr size_t kRadixBuckets = size_t{1} << kRadixBits;

Trigram PairTrigram(uint64_t pair) { return static_cast<Trigram>(pair >> 32); }
ChunkId PairChunk(uint64_t pair) { return static_cast<ChunkId>(pair); }

// Sorts `pairs` by trigram, keeping the order of pairs with equal trigrams:
// an LSD radix sort on the trigram's two 12-bit halves. `temp` is scratch.
void StableSortByTrigram(std::vector<uint64_t>& pairs,
                         std::vector<uint64_t>& temp) {
  temp.resize(pairs.size());
  for (const int shift : {32, 32 + kRadixBits}) {
    std::array<size_t, kRadixBuckets + 1> starts = {};
    for (const uint64_t pair : pairs) {
      ++starts[((pair >> shift) & (kRadixBuckets - 1)) + 1];
    }
    for (size_t b = 1; b <= kRadixBuckets; ++b) starts[b] += starts[b - 1];
    for (const uint64_t pair : pairs) {
      temp[starts[(pair >> shift) & (kRadixBuckets - 1)]++] = pair;
    }
    pairs.swap(temp);
  }
}

size_t VarintSize(uint32_t value) {
  size_t n = 1;
  while (value >= 0x80) {
    value >>= 7;
    ++n;
  }
  return n;
}

}  // namespace

ChunkId* PostingAccumulator::Run::DecodeList(size_t i, ChunkId* dst) const {
  const auto* p = reinterpret_cast<const uint8_t*>(packed.data()) +
                  (i == 0 ? 0 : byte_ends[i - 1]);
  const size_t count = list_size(i);
  uint32_t previous = 0;
  for (size_t k = 0; k < count; ++k) {
    uint32_t delta = *p & 0x7F;
    for (int shift = 7; *p++ >= 0x80; shift += 7) {
      delta |= static_cast<uint32_t>(*p & 0x7F) << shift;
    }
    previous += delta;
    *dst++ = previous;
  }
  return dst;
}

PostingAccumulator::PostingAccumulator(size_t max_partition_postings)
    : max_partition_postings_(max_partition_postings) {
  CHECK_GT(max_partition_postings, 0u);
  // Run::ends and byte_ends hold offsets into one partition as uint32_t; an
  // id takes at most 5 bytes.
  CHECK_LE(max_partition_postings, size_t{1} << 28);
}

void PostingAccumulator::AddChunk(ChunkId chunk,
                                  absl::Span<const Trigram> trigrams) {
  DCHECK(!has_chunk_ || chunk > last_chunk_) << "chunks out of order";
  has_chunk_ = true;
  last_chunk_ = chunk;
  for (const Trigram trigram : trigrams) {
    DCHECK_EQ(trigram & ~kTrigramMask, 0u);
    pairs_.push_back((uint64_t{trigram} << 32) | chunk);
  }
  num_postings_ += trigrams.size();
  if (pairs_.size() >= max_partition_postings_) FlushPartition();
}

void PostingAccumulator::FlushPartition() {
  if (pairs_.empty()) return;
  std::vector<uint64_t> temp;
  StableSortByTrigram(pairs_, temp);
  Run& run = runs_.emplace_back();
  // Size the packed bytes exactly, so that no capacity is wasted.
  size_t bytes = 0;
  for (size_t i = 0; i < pairs_.size(); ++i) {
    const bool first =
        i == 0 || PairTrigram(pairs_[i]) != PairTrigram(pairs_[i - 1]);
    bytes +=
        VarintSize(first ? PairChunk(pairs_[i])
                         : PairChunk(pairs_[i]) - PairChunk(pairs_[i - 1]));
  }
  run.packed.reserve(bytes);
  ChunkId previous = 0;
  for (size_t i = 0; i < pairs_.size(); ++i) {
    const Trigram trigram = PairTrigram(pairs_[i]);
    if (i == 0 || trigram != run.trigrams.back()) {
      if (i != 0) {
        run.ends.push_back(static_cast<uint32_t>(i));
        run.byte_ends.push_back(static_cast<uint32_t>(run.packed.size()));
      }
      run.trigrams.push_back(trigram);
      previous = 0;
    }
    uint32_t delta = PairChunk(pairs_[i]) - previous;
    previous = PairChunk(pairs_[i]);
    while (delta >= 0x80) {
      run.packed.push_back(static_cast<char>(0x80 | (delta & 0x7F)));
      delta >>= 7;
    }
    run.packed.push_back(static_cast<char>(delta));
  }
  run.ends.push_back(static_cast<uint32_t>(pairs_.size()));
  run.byte_ends.push_back(static_cast<uint32_t>(run.packed.size()));
  // Release the partition's memory rather than keeping its capacity.
  std::vector<uint64_t>().swap(pairs_);
}

std::vector<PostingAccumulator::Run> PostingAccumulator::TakeRuns() {
  FlushPartition();
  std::vector<Run> runs = std::move(runs_);
  runs_.clear();
  return runs;
}

absl::Status PostingAccumulator::ForEachList(
    absl::FunctionRef<absl::Status(Trigram, absl::Span<const ChunkId>)> fn) {
  FlushPartition();
  // A min-heap of (trigram, run) over each run's next list. Runs cover
  // increasing chunk ranges, so popping equal trigrams in run order yields
  // each list's pieces in id order.
  using Head = std::pair<Trigram, size_t>;
  std::priority_queue<Head, std::vector<Head>, std::greater<Head>> heads;
  std::vector<size_t> next(runs_.size(), 0);
  for (size_t r = 0; r < runs_.size(); ++r) {
    heads.emplace(runs_[r].trigrams[0], r);
  }
  std::vector<ChunkId> list;
  while (!heads.empty()) {
    const Trigram trigram = heads.top().first;
    list.clear();
    while (!heads.empty() && heads.top().first == trigram) {
      const size_t r = heads.top().second;
      heads.pop();
      const Run& run = runs_[r];
      const size_t i = next[r]++;
      const size_t old_size = list.size();
      list.resize(old_size + run.list_size(i));
      run.DecodeList(i, list.data() + old_size);
      if (next[r] < run.trigrams.size()) {
        heads.emplace(run.trigrams[next[r]], r);
      }
    }
    ABSL_RETURN_IF_ERROR(fn(trigram, list));
  }
  runs_.clear();
  return absl::OkStatus();
}

}  // namespace ferret::segment_internal

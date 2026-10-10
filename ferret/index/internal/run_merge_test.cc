// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/internal/run_merge.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/thread_pool.h"
#include "ferret/index/internal/posting_accumulator.h"
#include "ferret/postings/ids.h"
#include "ferret/postings/posting_list.h"
#include "ferret/postings/posting_list_encoder.h"
#include "ferret/testing/test_seed.h"
#include "ferret/text/trigram.h"

namespace ferret::segment_internal {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

struct Received {
  size_t count;
  uint64_t inline_postings;
  std::string encoded;
};

// Workers number chunks in the order they process them, so a chunk's
// provisional id says nothing about its final place: `remap` is a permutation.
struct Scenario {
  std::vector<PostingAccumulator::Run> runs;
  std::vector<ChunkId> remap;
  // Trigram to the final ids that contain it, sorted.
  std::map<Trigram, std::vector<ChunkId>> expected;
};

Scenario MakeScenario(uint64_t seed, uint32_t num_chunks, int num_workers,
                      size_t max_partition_postings) {
  std::mt19937_64 rng(seed);
  Scenario scenario;
  scenario.remap.resize(num_chunks);
  std::iota(scenario.remap.begin(), scenario.remap.end(), 0);
  std::shuffle(scenario.remap.begin(), scenario.remap.end(), rng);
  // Trigrams around the task range boundaries (1024 wide) and the ends of
  // the space, a few common ones with long lists, many rare ones.
  std::vector<Trigram> trigrams = {
      0, 1, 1023, 1024, 1025, 2047, 2048, 0x7FFFFF, 0xFFFFFE, kTrigramMask};
  for (int i = 0; i < 300; ++i) {
    trigrams.push_back(static_cast<Trigram>(rng() & kTrigramMask));
  }
  std::sort(trigrams.begin(), trigrams.end());
  trigrams.erase(std::unique(trigrams.begin(), trigrams.end()), trigrams.end());
  std::vector<std::unique_ptr<PostingAccumulator>> workers;
  workers.reserve(static_cast<size_t>(num_workers));
  for (int w = 0; w < num_workers; ++w) {
    workers.push_back(
        std::make_unique<PostingAccumulator>(max_partition_postings));
  }
  for (ChunkId chunk = 0; chunk < num_chunks; ++chunk) {
    std::vector<Trigram> in_chunk;
    for (size_t t = 0; t < trigrams.size(); ++t) {
      // The first three trigrams are in most chunks, every tenth in a few,
      // the rest in about one in 500.
      const uint64_t odds = t < 3 ? 3 : (t % 10 == 0 ? 40 : 500);
      if (rng() % odds == 0) in_chunk.push_back(trigrams[t]);
    }
    workers[rng() % workers.size()]->AddChunk(chunk, in_chunk);
    for (const Trigram trigram : in_chunk) {
      scenario.expected[trigram].push_back(scenario.remap[chunk]);
    }
  }
  for (auto& worker : workers) {
    for (auto& run : worker->TakeRuns()) {
      scenario.runs.push_back(std::move(run));
    }
  }
  for (auto& [trigram, ids] : scenario.expected) {
    std::sort(ids.begin(), ids.end());
  }
  return scenario;
}

std::map<Trigram, Received> Merge(const Scenario& scenario, ThreadPool* pool,
                                  int threads) {
  std::map<Trigram, Received> received;
  Trigram previous = 0;
  bool first = true;
  EXPECT_THAT(
      MergeRuns(scenario.runs, scenario.remap, pool, threads,
                [&](Trigram trigram, size_t count, uint64_t inline_postings,
                    absl::string_view encoded) {
                  EXPECT_TRUE(first || trigram > previous);
                  first = false;
                  previous = trigram;
                  received[trigram] = {count, inline_postings,
                                       std::string(encoded)};
                  return absl::OkStatus();
                }),
      IsOk());
  return received;
}

void ExpectMatches(const Scenario& scenario,
                   const std::map<Trigram, Received>& received) {
  ASSERT_EQ(received.size(), scenario.expected.size());
  for (const auto& [trigram, ids] : scenario.expected) {
    const auto it = received.find(trigram);
    ASSERT_TRUE(it != received.end()) << trigram;
    ASSERT_EQ(it->second.count, ids.size()) << trigram;
    if (IsInlinedPostingList(ids.size())) {
      uint64_t inline_postings = ids[0];
      if (ids.size() == 2) inline_postings |= uint64_t{ids[1]} << 32;
      EXPECT_EQ(it->second.inline_postings, inline_postings) << trigram;
      EXPECT_TRUE(it->second.encoded.empty());
    } else {
      std::string encoded;
      AppendEncodedPostingList(ids, encoded);
      EXPECT_TRUE(it->second.encoded == encoded) << trigram;
    }
  }
}

TEST(RunMergeTest, MapsAndSortsLists) {
  const uint64_t seed = ferret_testing::TestSeed(11);
  // Lists longer than the radix threshold, single runs, and many runs.
  const Scenario scenario = MakeScenario(seed, 6000, 3, 2000);
  EXPECT_GT(scenario.runs.size(), 3u);
  size_t longest = 0;
  for (const auto& [trigram, ids] : scenario.expected) {
    longest = std::max(longest, ids.size());
  }
  EXPECT_GT(longest, 1500u);
  ExpectMatches(scenario, Merge(scenario, nullptr, 1));
}

TEST(RunMergeTest, ParallelMatchesSerial) {
  const Scenario scenario =
      MakeScenario(ferret_testing::TestSeed(5), 4000, 4, size_t{1} << 20);
  ThreadPool pool(4, ThreadQos::kUtility);
  for (const int threads : {2, 5}) {
    ExpectMatches(scenario, Merge(scenario, &pool, threads));
  }
}

TEST(RunMergeTest, NoRuns) {
  Scenario scenario;
  EXPECT_THAT(MergeRuns(scenario.runs, scenario.remap, nullptr, 1,
                        [](Trigram, size_t, uint64_t, absl::string_view) {
                          ADD_FAILURE();
                          return absl::OkStatus();
                        }),
              IsOk());
}

TEST(RunMergeTest, StopsAtASinkError) {
  const Scenario scenario = MakeScenario(3, 1000, 2, 1 << 20);
  ThreadPool pool(3, ThreadQos::kUtility);
  int calls = 0;
  EXPECT_THAT(MergeRuns(scenario.runs, scenario.remap, &pool, 4,
                        [&](Trigram, size_t, uint64_t, absl::string_view) {
                          return ++calls == 5 ? absl::AbortedError("sink")
                                              : absl::OkStatus();
                        }),
              StatusIs(absl::StatusCode::kAborted));
  EXPECT_EQ(calls, 5);
}

}  // namespace
}  // namespace ferret::segment_internal

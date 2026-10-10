// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/postings/list_ops.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/ablation.h"
#include "ferret/base/stage_clock.h"
#include "ferret/postings/ids.h"
#include "ferret/postings/posting_list.h"
#include "ferret/postings/posting_list_encoder.h"
#include "ferret/postings/testing/random_posting_lists.h"
#include "ferret/simd/dispatch.h"
#include "ferret/testing/test_seed.h"

namespace ferret {
namespace {

using ::ferret_testing::RandomPostingList;
using ::ferret_testing::UniformBelow;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

using IdVector = std::vector<ChunkId>;

constexpr ChunkId kMaxId = std::numeric_limits<ChunkId>::max();

IdVector ToVector(const ChunkIdBuffer& buffer) {
  return IdVector(buffer.ids().begin(), buffer.ids().end());
}

IdVector ReferenceIntersection(absl::Span<const ChunkId> a,
                               absl::Span<const ChunkId> b) {
  IdVector out;
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(out));
  return out;
}

IdVector ReferenceUnion(absl::Span<const IdVector> lists) {
  IdVector out;
  for (const IdVector& list : lists) {
    IdVector merged;
    std::set_union(out.begin(), out.end(), list.begin(), list.end(),
                   std::back_inserter(merged));
    out = std::move(merged);
  }
  return out;
}

// A posting list in both representations the operations accept. The encoded
// bytes are owned here, so `view` stays valid while *this is alive; do not
// copy or move it after construction.
class TestList final {
 public:
  explicit TestList(IdVector ids) : ids_(std::move(ids)) {
    absl::StatusOr<PostingListView> view;
    if (IsInlinedPostingList(ids_.size())) {
      view = PostingListView::FromInlined(ids_);
    } else {
      AppendEncodedPostingList(ids_, encoded_);
      view = PostingListView::FromEncoded(encoded_);
    }
    CHECK_OK(view.status());
    view_ = *view;
  }
  TestList(const TestList&) = delete;
  TestList& operator=(const TestList&) = delete;

  const IdVector& ids() const { return ids_; }
  const PostingListView& view() const { return view_; }

 private:
  IdVector ids_;
  std::string encoded_;
  PostingListView view_;
};

// Returns `count` distinct ids drawn from [first, last], sorted.
IdVector RandomSubsetOfRange(std::mt19937_64& rng, size_t count, ChunkId first,
                             ChunkId last) {
  IdVector out;
  out.reserve(count);
  const uint64_t span = uint64_t{last} - first + 1;
  count = std::min<uint64_t>(count, span);
  while (out.size() < count) {
    out.push_back(static_cast<ChunkId>(first + UniformBelow(rng, span)));
    if (out.size() == count) {
      std::sort(out.begin(), out.end());
      out.erase(std::unique(out.begin(), out.end()), out.end());
    }
  }
  return out;
}

// Returns a random sample of `list`: each id is kept with probability
// `keep`, and ids from `noise` are mixed in, so some candidates miss.
IdVector Candidates(std::mt19937_64& rng, const IdVector& list, double keep,
                    const IdVector& noise) {
  std::bernoulli_distribution keep_id(keep);
  IdVector out;
  for (const ChunkId id : list) {
    if (keep_id(rng)) out.push_back(id);
  }
  out.insert(out.end(), noise.begin(), noise.end());
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

TEST(ChunkIdBufferTest, StartsEmpty) {
  const ChunkIdBuffer buffer;
  EXPECT_TRUE(buffer.empty());
  EXPECT_EQ(buffer.size(), size_t{0});
  EXPECT_THAT(buffer.ids(), IsEmpty());
}

TEST(ChunkIdBufferTest, WritesUpToCapacityPlusSlack) {
  ChunkIdBuffer buffer;
  for (const size_t capacity : {0u, 1u, 3u, 100u, 1000u, 7u}) {
    ChunkId* const ids = buffer.ResetForOverwrite(capacity);
    EXPECT_TRUE(buffer.empty());
    // Under ASan, writing the slack proves it was allocated.
    for (size_t i = 0; i < capacity + kSimdSlack; ++i) {
      ids[i] = static_cast<ChunkId>(i);
    }
    buffer.set_size(capacity);
    EXPECT_EQ(buffer.size(), capacity);
    for (size_t i = 0; i < capacity; ++i) EXPECT_EQ(buffer.ids()[i], i);
  }
}

TEST(ChunkIdBufferTest, AssignCopies) {
  ChunkIdBuffer buffer;
  const IdVector ids = {3, 5, 8};
  buffer.Assign(ids);
  EXPECT_THAT(ToVector(buffer), ElementsAre(3, 5, 8));
  buffer.Assign({});
  EXPECT_TRUE(buffer.empty());
}

TEST(ChunkIdBufferTest, ClearKeepsStorage) {
  ChunkIdBuffer buffer;
  buffer.Assign(IdVector{1, 2, 3});
  const ChunkId* const storage = buffer.ids().data();
  buffer.clear();
  EXPECT_TRUE(buffer.empty());
  EXPECT_EQ(buffer.ResetForOverwrite(3), storage);
}

TEST(ChunkIdBufferTest, MoveLeavesSourceEmpty) {
  ChunkIdBuffer a;
  a.Assign(IdVector{1, 2});
  ChunkIdBuffer b(std::move(a));
  EXPECT_THAT(ToVector(b), ElementsAre(1, 2));
  EXPECT_TRUE(a.empty());  // NOLINT(bugprone-use-after-move)
  // A moved-from buffer is usable.
  a.Assign(IdVector{7});
  EXPECT_THAT(ToVector(a), ElementsAre(7));

  ChunkIdBuffer c;
  c = std::move(b);
  EXPECT_THAT(ToVector(c), ElementsAre(1, 2));
  EXPECT_TRUE(b.empty());  // NOLINT(bugprone-use-after-move)
}

TEST(ChunkIdBufferTest, Swap) {
  ChunkIdBuffer a;
  ChunkIdBuffer b;
  a.Assign(IdVector{1});
  b.Assign(IdVector{2, 3});
  swap(a, b);
  EXPECT_THAT(ToVector(a), ElementsAre(2, 3));
  EXPECT_THAT(ToVector(b), ElementsAre(1));
}

TEST(DecodeRangeTest, EdgeRanges) {
  const TestList list(IdVector{0, 5, 6, 200, kMaxId});
  ChunkIdBuffer out;
  const auto check = [&](ChunkRange range, const IdVector& expected) {
    SCOPED_TRACE(absl::StrCat("[", range.first, ", ", range.last, "]"));
    DecodeRange(list.view(), range, out);
    EXPECT_THAT(ToVector(out), ElementsAreArray(expected));
    DecodeRange(absl::MakeConstSpan(list.ids()), range, out);
    EXPECT_THAT(ToVector(out), ElementsAreArray(expected));
  };
  check({0, kMaxId}, list.ids());
  check({0, 0}, {0});
  check({1, 4}, {});
  check({5, 6}, {5, 6});
  check({6, 199}, {6});
  check({201, kMaxId - 1}, {});
  check({kMaxId, kMaxId}, {kMaxId});
}

TEST(DecodeRangeTest, EmptyList) {
  ChunkIdBuffer out;
  out.Assign(IdVector{1});
  DecodeRange(PostingListView(), ChunkRange{0, kMaxId}, out);
  EXPECT_TRUE(out.empty());
  out.Assign(IdVector{1});
  DecodeRange(absl::Span<const ChunkId>(), ChunkRange{0, kMaxId}, out);
  EXPECT_TRUE(out.empty());
}

TEST(DecodeRangeTest, RandomRangesMatchReference) {
  std::mt19937_64 rng(ferret_testing::TestSeed(1));
  ChunkIdBuffer out;
  for (int iteration = 0; iteration < 300; ++iteration) {
    const size_t size = UniformBelow(rng, 3000);
    const int gap_bits = static_cast<int>(UniformBelow(rng, 12));
    const TestList list(RandomPostingList(rng, size, gap_bits, 0));
    const ChunkId limit = list.ids().empty() ? 100 : list.ids().back() + 1;
    ChunkId first = static_cast<ChunkId>(UniformBelow(rng, limit + 1));
    ChunkId last = static_cast<ChunkId>(UniformBelow(rng, limit + 1));
    if (first > last) std::swap(first, last);
    IdVector expected;
    for (const ChunkId id : list.ids()) {
      if (first <= id && id <= last) expected.push_back(id);
    }
    DecodeRange(list.view(), ChunkRange{first, last}, out);
    ASSERT_THAT(ToVector(out), ElementsAreArray(expected));
    DecodeRange(absl::MakeConstSpan(list.ids()), ChunkRange{first, last}, out);
    ASSERT_THAT(ToVector(out), ElementsAreArray(expected));
  }
}

TEST(IntersectWithPostingListTest, EmptyInputs) {
  const TestList list(IdVector{1, 2, 3});
  ChunkIdBuffer scratch;
  ChunkIdBuffer out;
  out.Assign(IdVector{9});
  IntersectWithPostingList({}, list.view(), scratch, out);
  EXPECT_TRUE(out.empty());
  out.Assign(IdVector{9});
  IntersectWithPostingList(IdVector{1, 2}, PostingListView(), scratch, out);
  EXPECT_TRUE(out.empty());
  out.Assign(IdVector{9});
  IntersectWithPostingList({}, absl::MakeConstSpan(list.ids()), scratch, out);
  EXPECT_TRUE(out.empty());
  out.Assign(IdVector{9});
  IntersectWithPostingList(IdVector{1, 2}, absl::Span<const ChunkId>(), scratch,
                           out);
  EXPECT_TRUE(out.empty());
}

// Sweeps candidate-set sizes across the skip-probe threshold
// (|candidates| < num_blocks / 2) for lists from inlined to many blocks.
TEST(IntersectWithPostingListTest, BothStrategiesMatchReference) {
  std::mt19937_64 rng(ferret_testing::TestSeed(2));
  ChunkIdBuffer scratch;
  ChunkIdBuffer out;
  for (const size_t list_size :
       {size_t{1}, size_t{2}, size_t{3}, size_t{127}, size_t{128}, size_t{129},
        size_t{1000}, size_t{20000}}) {
    for (const int gap_bits : {0, 3, 10}) {
      const TestList list(RandomPostingList(rng, list_size, gap_bits, 17));
      const ChunkId last_id = list.ids().back();
      const size_t num_blocks = static_cast<size_t>(list.view().num_blocks());
      for (const size_t noise_size :
           {size_t{0}, size_t{1}, num_blocks / 2 - num_blocks / 4, num_blocks,
            4 * num_blocks}) {
        for (const double keep : {0.0, 0.001, 0.05, 0.5, 1.0}) {
          const IdVector noise =
              RandomSubsetOfRange(rng, noise_size, 0, last_id + 50);
          const IdVector candidates = Candidates(rng, list.ids(), keep, noise);
          SCOPED_TRACE(absl::StrCat(
              "list_size=", list_size, " gap_bits=", gap_bits,
              " candidates=", candidates.size(), " blocks=", num_blocks));
          const IdVector expected =
              ReferenceIntersection(candidates, list.ids());
          IntersectWithPostingList(candidates, list.view(), scratch, out);
          ASSERT_THAT(ToVector(out), ElementsAreArray(expected));
          IntersectWithPostingList(candidates, absl::MakeConstSpan(list.ids()),
                                   scratch, out);
          ASSERT_THAT(ToVector(out), ElementsAreArray(expected));
        }
      }
    }
  }
}

// Puts the ablation switches back when a test ends, whichever way it ends.
class AblationGuard final {
 public:
  AblationGuard() = default;
  AblationGuard(const AblationGuard&) = delete;
  AblationGuard& operator=(const AblationGuard&) = delete;
  ~AblationGuard() {
    Ablation::scalar_decode = false;
    Ablation::scalar_intersect = false;
    Ablation::no_skip_probe = false;
  }
};

TEST(AblationTest, SkipProbeSwitchChangesThePathNotTheResult) {
  const AblationGuard guard;
  std::mt19937_64 rng(ferret_testing::TestSeed(11));
  const TestList list(RandomPostingList(rng, 20000, 4, 17));
  ASSERT_GT(list.view().num_blocks(), 100);
  // Few candidates against many blocks: the probe pays.
  const IdVector candidates =
      Candidates(rng, list.ids(), 0.001,
                 RandomSubsetOfRange(rng, 20, 0, list.ids().back()));
  ASSERT_LT(candidates.size(),
            static_cast<size_t>(list.view().num_blocks()) / 2);
  ChunkIdBuffer scratch;
  ChunkIdBuffer probed;
  ChunkIdBuffer decoded;
  StageClock& clock = ThisThreadStageClock();
  clock = StageClock{.enabled = true};
  const AblationCounts before = ThisThreadAblationCounts();
  IntersectWithPostingList(candidates, list.view(), scratch, probed);
  // The probe decodes on its own, outside DecodeRange.
  EXPECT_EQ(clock.decode_ns, 0);
  EXPECT_EQ(ThisThreadAblationCounts().unprobed_intersections,
            before.unprobed_intersections);
  Ablation::no_skip_probe = true;
  IntersectWithPostingList(candidates, list.view(), scratch, decoded);
  clock = StageClock{};
  EXPECT_THAT(ToVector(decoded), ElementsAreArray(ToVector(probed)));
  EXPECT_THAT(ToVector(decoded),
              ElementsAreArray(ReferenceIntersection(candidates, list.ids())));
  EXPECT_EQ(ThisThreadAblationCounts().unprobed_intersections,
            before.unprobed_intersections + 1);
}

TEST(AblationTest, ScalarKernelsGiveTheSameListsAndAreCounted) {
  const AblationGuard guard;
  std::mt19937_64 rng(ferret_testing::TestSeed(12));
  const TestList a(RandomPostingList(rng, 5000, 3, 17));
  const TestList b(RandomPostingList(rng, 6000, 3, 17));
  ChunkIdBuffer scratch;
  ChunkIdBuffer neon;
  ChunkIdBuffer scalar;
  const IdVector expected = ReferenceIntersection(a.ids(), b.ids());
  const AblationCounts before = ThisThreadAblationCounts();
  // Equal sizes: the merge path, decoded blocks and all.
  IntersectWithPostingList(a.ids(), b.view(), scratch, neon);
  EXPECT_THAT(ToVector(neon), ElementsAreArray(expected));
  EXPECT_EQ(ThisThreadAblationCounts().scalar_blocks, before.scalar_blocks);
  EXPECT_EQ(ThisThreadAblationCounts().scalar_merges, before.scalar_merges);
  Ablation::scalar_decode = true;
  Ablation::scalar_intersect = true;
  IntersectWithPostingList(a.ids(), b.view(), scratch, scalar);
  EXPECT_THAT(ToVector(scalar), ElementsAreArray(expected));
  EXPECT_GT(ThisThreadAblationCounts().scalar_blocks, before.scalar_blocks);
  EXPECT_GT(ThisThreadAblationCounts().scalar_merges, before.scalar_merges);
  ChunkIdBuffer decoded;
  DecodeRange(b.view(), ChunkRange{0, kMaxId}, decoded);
  EXPECT_THAT(ToVector(decoded), ElementsAreArray(b.ids()));
}

TEST(StageClockTest, CountsDecodeAndIntersectOnlyWhenEnabled) {
  std::mt19937_64 rng(ferret_testing::TestSeed(13));
  const TestList list(RandomPostingList(rng, 20000, 3, 17));
  ChunkIdBuffer scratch;
  ChunkIdBuffer out;
  StageClock& clock = ThisThreadStageClock();
  clock = StageClock{};
  DecodeRange(list.view(), ChunkRange{0, kMaxId}, out);
  EXPECT_EQ(clock.decode_ns, 0);
  clock.enabled = true;
  DecodeRange(list.view(), ChunkRange{0, kMaxId}, out);
  EXPECT_GT(clock.decode_ns, 0);
  EXPECT_EQ(clock.intersect_ns, 0);
  const IdVector candidates = Candidates(rng, list.ids(), 0.5, {});
  IntersectWithPostingList(candidates, list.view(), scratch, out);
  EXPECT_GT(clock.intersect_ns, 0);
  const int64_t decode_before = clock.decode_ns;
  const absl::Span<const ChunkId> lists[] = {list.ids(), candidates};
  UnionAll(lists, scratch, out);
  EXPECT_GE(clock.intersect_ns, 0);
  EXPECT_EQ(clock.decode_ns, decode_before);
  clock = StageClock{};
}

TEST(IntersectWithPostingListTest, CandidatesOutsideTheList) {
  std::mt19937_64 rng(ferret_testing::TestSeed(3));
  const TestList list(RandomPostingList(rng, 1000, 2, 1000));
  ChunkIdBuffer scratch;
  ChunkIdBuffer out;
  const IdVector below = {0, 1, 999};
  const IdVector above = {list.ids().back() + 1, kMaxId};
  IntersectWithPostingList(below, list.view(), scratch, out);
  EXPECT_TRUE(out.empty());
  IntersectWithPostingList(above, list.view(), scratch, out);
  EXPECT_TRUE(out.empty());
  const IdVector ends = {0, list.ids().front(), list.ids().back(), kMaxId};
  IntersectWithPostingList(ends, list.view(), scratch, out);
  EXPECT_THAT(ToVector(out),
              ElementsAre(list.ids().front(), list.ids().back()));
}

TEST(UnionAllTest, SmallCases) {
  ChunkIdBuffer scratch;
  ChunkIdBuffer out;
  const auto run = [&](const std::vector<IdVector>& lists) {
    std::vector<absl::Span<const ChunkId>> spans(lists.begin(), lists.end());
    UnionAll(spans, scratch, out);
    return ToVector(out);
  };
  EXPECT_THAT(run({}), IsEmpty());
  EXPECT_THAT(run({{}, {}}), IsEmpty());
  EXPECT_THAT(run({{1, 4}}), ElementsAre(1, 4));
  EXPECT_THAT(run({{}, {1, 4}, {}}), ElementsAre(1, 4));
  EXPECT_THAT(run({{1, 4}, {2, 4}}), ElementsAre(1, 2, 4));
  EXPECT_THAT(run({{1}, {2}, {1, 3}}), ElementsAre(1, 2, 3));
  EXPECT_THAT(run({{0}, {kMaxId}, {kMaxId / 2}}),
              ElementsAre(0, kMaxId / 2, kMaxId));
  EXPECT_THAT(run({{kMaxId - 2}, {kMaxId}, {kMaxId - 1, kMaxId}}),
              ElementsAre(kMaxId - 2, kMaxId - 1, kMaxId));
}

// Varies the number of lists and their density, so that both the bitmap and
// the pairwise path run, with overlapping and disjoint lists.
TEST(UnionAllTest, RandomListsMatchReference) {
  std::mt19937_64 rng(ferret_testing::TestSeed(4));
  ChunkIdBuffer scratch;
  ChunkIdBuffer out;
  for (int iteration = 0; iteration < 2000; ++iteration) {
    const size_t num_lists = UniformBelow(rng, 9);
    std::vector<IdVector> lists;
    for (size_t i = 0; i < num_lists; ++i) {
      const size_t size =
          UniformBelow(rng, 4) == 0 ? 0 : UniformBelow(rng, 600);
      const int gap_bits = static_cast<int>(UniformBelow(rng, 16));
      const auto first = static_cast<ChunkId>(UniformBelow(rng, 1 << 20));
      lists.push_back(RandomPostingList(rng, size, gap_bits, first));
    }
    std::vector<absl::Span<const ChunkId>> spans(lists.begin(), lists.end());
    UnionAll(spans, scratch, out);
    ASSERT_THAT(ToVector(out), ElementsAreArray(ReferenceUnion(lists)))
        << "iteration " << iteration;
  }
}

}  // namespace
}  // namespace ferret

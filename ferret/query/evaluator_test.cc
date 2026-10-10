// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/query/evaluator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
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

#include "ferret/postings/ids.h"
#include "ferret/postings/list_ops.h"
#include "ferret/postings/posting_list.h"
#include "ferret/postings/posting_list_encoder.h"
#include "ferret/query/trigram_query.h"
#include "ferret/testing/test_seed.h"
#include "ferret/text/trigram.h"

namespace ferret {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

using IdVector = std::vector<ChunkId>;

Trigram T(int i) {
  return MakeTrigram('t', 'r', static_cast<uint8_t>('a' + i));
}

IdVector ToVector(const ChunkIdBuffer& buffer) {
  return IdVector(buffer.ids().begin(), buffer.ids().end());
}

// A toy index: every chunk has a trigram set, and every trigram a posting
// list, kept both as an array and as an encoded PostingListView.
class ToyIndex final {
 public:
  ToyIndex() = default;
  ToyIndex(const ToyIndex&) = delete;
  ToyIndex& operator=(const ToyIndex&) = delete;

  // Adds the next chunk, with the given trigrams.
  void AddChunk(const std::vector<Trigram>& trigrams) {
    const auto id = static_cast<ChunkId>(chunks_.size());
    chunks_.push_back(trigrams);
    for (const Trigram trigram : trigrams) lists_[trigram].ids.push_back(id);
  }

  // Encodes the lists; call once, after the last AddChunk().
  void Finish() {
    for (auto& [trigram, list] : lists_) {
      absl::StatusOr<PostingListView> view;
      if (IsInlinedPostingList(list.ids.size())) {
        view = PostingListView::FromInlined(list.ids);
      } else {
        AppendEncodedPostingList(list.ids, list.encoded);
        view = PostingListView::FromEncoded(list.encoded);
      }
      CHECK_OK(view.status());
      list.view = *view;
    }
  }

  size_t num_chunks() const { return chunks_.size(); }

  PostingListView FindView(Trigram trigram) const {
    const auto it = lists_.find(trigram);
    return it == lists_.end() ? PostingListView() : it->second.view;
  }
  absl::Span<const ChunkId> FindSpan(Trigram trigram) const {
    const auto it = lists_.find(trigram);
    return it == lists_.end() ? absl::Span<const ChunkId>()
                              : absl::MakeConstSpan(it->second.ids);
  }

  // The ids in `window` whose chunks satisfy `query` by its reference
  // semantics.
  IdVector Reference(const TrigramQuery& query, ChunkRange window) const {
    IdVector out;
    for (size_t id = window.first; id <= window.last; ++id) {
      const std::vector<Trigram>& trigrams = chunks_[id];
      const bool match = query.Evaluate([&](Trigram trigram) {
        return std::find(trigrams.begin(), trigrams.end(), trigram) !=
               trigrams.end();
      });
      if (match) out.push_back(static_cast<ChunkId>(id));
    }
    return out;
  }

 private:
  struct List {
    IdVector ids;
    std::string encoded;
    PostingListView view;
  };

  std::vector<std::vector<Trigram>> chunks_;
  // std::map, so that references to the encoded bytes stay valid.
  std::map<Trigram, List> lists_;
};

// Evaluates `query` over `window` with both list types, checks that they
// agree, and returns the result.
IdVector EvaluateBoth(const ToyIndex& index, const TrigramQuery& query,
                      ChunkRange window) {
  TrigramQueryEvaluator<PostingListView> encoded(
      query, [&](Trigram trigram) { return index.FindView(trigram); });
  TrigramQueryEvaluator<absl::Span<const ChunkId>> plain(
      query, [&](Trigram trigram) { return index.FindSpan(trigram); });
  ChunkIdBuffer out;
  encoded.Evaluate(window, out);
  IdVector result = ToVector(out);
  plain.Evaluate(window, out);
  EXPECT_THAT(ToVector(out), ElementsAreArray(result)) << query.ToString();
  EXPECT_EQ(encoded.estimated_size(), plain.estimated_size());
  return result;
}

// Chunks 0..9; chunk c contains T(0) if c is even, T(1) if c % 3 == 0, and
// T(2) if c == 7. T(3) is in no chunk.
class SmallIndexTest : public ::testing::Test {
 protected:
  SmallIndexTest() {
    for (int c = 0; c < 10; ++c) {
      std::vector<Trigram> trigrams;
      if (c % 2 == 0) trigrams.push_back(T(0));
      if (c % 3 == 0) trigrams.push_back(T(1));
      if (c == 7) trigrams.push_back(T(2));
      index_.AddChunk(trigrams);
    }
    index_.Finish();
  }

  static constexpr ChunkRange kAll = {0, 9};
  ToyIndex index_;
};

TEST_F(SmallIndexTest, Constants) {
  EXPECT_THAT(EvaluateBoth(index_, TrigramQuery::All(), {2, 5}),
              ElementsAre(2, 3, 4, 5));
  EXPECT_THAT(EvaluateBoth(index_, TrigramQuery::None(), kAll), IsEmpty());
}

TEST_F(SmallIndexTest, SingleTrigram) {
  EXPECT_THAT(EvaluateBoth(index_, TrigramQuery::AllOf({T(0)}), kAll),
              ElementsAre(0, 2, 4, 6, 8));
  EXPECT_THAT(EvaluateBoth(index_, TrigramQuery::AllOf({T(0)}), {3, 6}),
              ElementsAre(4, 6));
  EXPECT_THAT(EvaluateBoth(index_, TrigramQuery::AllOf({T(3)}), kAll),
              IsEmpty());
}

TEST_F(SmallIndexTest, AndOr) {
  const TrigramQuery t0 = TrigramQuery::AllOf({T(0)});
  const TrigramQuery t1 = TrigramQuery::AllOf({T(1)});
  const TrigramQuery t2 = TrigramQuery::AllOf({T(2)});
  EXPECT_THAT(EvaluateBoth(index_, TrigramQuery::And(t0, t1), kAll),
              ElementsAre(0, 6));
  EXPECT_THAT(EvaluateBoth(index_, TrigramQuery::Or(t0, t1), kAll),
              ElementsAre(0, 2, 3, 4, 6, 8, 9));
  EXPECT_THAT(
      EvaluateBoth(index_, TrigramQuery::Or(TrigramQuery::And(t0, t1), t2),
                   kAll),
      ElementsAre(0, 6, 7));
  EXPECT_THAT(
      EvaluateBoth(index_, TrigramQuery::And(TrigramQuery::Or(t1, t2), t0),
                   {1, 9}),
      ElementsAre(6));
}

TEST_F(SmallIndexTest, MissingTrigramsSimplify) {
  const TrigramQuery t0 = TrigramQuery::AllOf({T(0)});
  const TrigramQuery t3 = TrigramQuery::AllOf({T(3)});
  const auto evaluator = [&](const TrigramQuery& query) {
    return TrigramQueryEvaluator<absl::Span<const ChunkId>>(
        query, [&](Trigram trigram) { return index_.FindSpan(trigram); });
  };
  // T(3) has no postings, so an AND with it is None ...
  EXPECT_TRUE(evaluator(TrigramQuery::And(t0, t3)).matches_nothing());
  EXPECT_EQ(evaluator(TrigramQuery::And(t0, t3)).estimated_size(), 0u);
  // ... and it drops out of an OR.
  EXPECT_FALSE(evaluator(TrigramQuery::Or(t0, t3)).matches_nothing());
  EXPECT_EQ(evaluator(TrigramQuery::Or(t0, t3)).estimated_size(), 5u);
  EXPECT_THAT(EvaluateBoth(index_, TrigramQuery::Or(t0, t3), kAll),
              ElementsAre(0, 2, 4, 6, 8));
  EXPECT_EQ(evaluator(TrigramQuery::All()).estimated_size(),
            std::numeric_limits<size_t>::max());
}

TEST_F(SmallIndexTest, ReusesBuffersAcrossWindows) {
  const TrigramQuery query = TrigramQuery::Or(TrigramQuery::AllOf({T(0), T(1)}),
                                              TrigramQuery::AllOf({T(2)}));
  TrigramQueryEvaluator<PostingListView> evaluator(
      query, [&](Trigram trigram) { return index_.FindView(trigram); });
  ChunkIdBuffer out;
  for (const ChunkRange window :
       {ChunkRange{0, 9}, ChunkRange{1, 5}, ChunkRange{6, 7}, ChunkRange{0, 9},
        ChunkRange{8, 9}}) {
    evaluator.Evaluate(window, out);
    EXPECT_THAT(ToVector(out),
                ElementsAreArray(index_.Reference(query, window)));
  }
}

// Returns a random query over trigrams T(0) .. T(num_trigrams - 1), built
// with the factories so that it is canonical.
TrigramQuery RandomQuery(std::mt19937_64& rng, int num_trigrams, int depth) {
  std::uniform_int_distribution<int> pick(0, 9);
  const int choice = pick(rng);
  if (depth == 0 || choice < 3) {
    if (choice == 0 && depth > 0) {
      return rng() % 2 == 0 ? TrigramQuery::All() : TrigramQuery::None();
    }
    std::vector<Trigram> trigrams;
    const int count = 1 + static_cast<int>(rng() % 3);
    trigrams.reserve(count);
    for (int i = 0; i < count; ++i) {
      trigrams.push_back(T(static_cast<int>(rng() % num_trigrams)));
    }
    return TrigramQuery::AllOf(trigrams);
  }
  TrigramQuery q = RandomQuery(rng, num_trigrams, depth - 1);
  const int arity = 1 + static_cast<int>(rng() % 3);
  for (int i = 0; i < arity; ++i) {
    TrigramQuery r = RandomQuery(rng, num_trigrams, depth - 1);
    q = choice % 2 == 0 ? TrigramQuery::And(std::move(q), std::move(r))
                        : TrigramQuery::Or(std::move(q), std::move(r));
  }
  return q;
}

// Random corpora whose trigram densities range from a handful of chunks
// (inlined lists) to nearly all of them (multi-block lists), so both
// intersection strategies and all union paths run, against the reference
// semantics of TrigramQuery::Evaluate.
TEST(TrigramQueryEvaluatorTest, RandomQueriesMatchReference) {
  std::mt19937_64 rng(ferret_testing::TestSeed(5));
  constexpr int kNumTrigrams = 12;
  constexpr double kDensities[kNumTrigrams] = {
      0.0005, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 0.0};
  for (int corpus = 0; corpus < 4; ++corpus) {
    const size_t num_chunks = corpus == 0 ? 40 : 6000 * corpus;
    ToyIndex index;
    for (size_t c = 0; c < num_chunks; ++c) {
      std::vector<Trigram> trigrams;
      trigrams.reserve(kNumTrigrams);
      for (int t = 0; t < kNumTrigrams; ++t) {
        if (std::bernoulli_distribution(kDensities[t])(rng)) {
          trigrams.push_back(T(t));
        }
      }
      index.AddChunk(trigrams);
    }
    index.Finish();
    const auto last = static_cast<ChunkId>(num_chunks - 1);
    for (int iteration = 0; iteration < 300; ++iteration) {
      // One more trigram than the corpus has, so some are missing.
      const TrigramQuery query = RandomQuery(rng, kNumTrigrams + 1, 4);
      auto first = static_cast<ChunkId>(rng() % num_chunks);
      auto window_last = static_cast<ChunkId>(rng() % num_chunks);
      if (first > window_last) std::swap(first, window_last);
      for (const ChunkRange window :
           {ChunkRange{0, last}, ChunkRange{first, window_last}}) {
        SCOPED_TRACE(absl::StrCat("corpus ", corpus, " iteration ", iteration,
                                  " window [", window.first, ", ", window.last,
                                  "] query ", query.ToString()));
        ASSERT_THAT(EvaluateBoth(index, query, window),
                    ElementsAreArray(index.Reference(query, window)));
      }
    }
  }
}

}  // namespace
}  // namespace ferret

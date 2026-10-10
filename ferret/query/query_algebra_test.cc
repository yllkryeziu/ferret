// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Exhaustive semantic checks of the TrigramQuery factories. The universe has
// six trigrams, so a query's meaning is a 64-bit truth table: bit s is
// Evaluate() on the trigram set whose members are the set bits of s. Every
// identity is checked on all 64 sets for many random query trees.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"

#include "ferret/query/trigram_query.h"
#include "ferret/text/trigram.h"

namespace ferret {
namespace {

using Op = TrigramQuery::Op;

constexpr int kUniverse = 6;
constexpr uint64_t kDefaultSeed = 20260923;

uint64_t TestSeed() {
  const char* env = std::getenv("FERRET_TEST_SEED");
  uint64_t seed = 0;
  if (env != nullptr && absl::SimpleAtoi(env, &seed)) return seed;
  return kDefaultSeed;
}

// Uniform draw from [0, n); the modulo bias is irrelevant here, and a
// hand-written draw is reproducible across standard libraries.
size_t Below(std::mt19937_64& rng, size_t n) {
  return static_cast<size_t>(rng() % n);
}

Trigram UniverseTrigram(int i) {
  return MakeTrigram('t', 'r', static_cast<uint8_t>('0' + i));
}

int UniverseIndex(Trigram trigram) {
  for (int i = 0; i < kUniverse; ++i) {
    if (UniverseTrigram(i) == trigram) return i;
  }
  ADD_FAILURE() << "trigram outside the universe: " << TrigramToString(trigram);
  return 0;
}

// Bit s of the result is Evaluate() on trigram set s.
uint64_t TruthTable(const TrigramQuery& query) {
  uint64_t table = 0;
  for (uint64_t set = 0; set < (uint64_t{1} << kUniverse); ++set) {
    const bool value = query.Evaluate([set](Trigram trigram) {
      return ((set >> UniverseIndex(trigram)) & 1) != 0;
    });
    if (value) table |= uint64_t{1} << set;
  }
  return table;
}

// An arbitrary, possibly non-canonical query tree built without the
// factories: empty nodes, single-child nodes, and same-op nesting all occur.
struct RawQuery {
  Op op = Op::kAll;
  std::vector<int> trigrams;  // Universe indices, possibly repeated.
  std::vector<RawQuery> subs;
};

uint64_t RawTruthTable(const RawQuery& raw) {
  constexpr uint64_t kAllSets = ~uint64_t{0};
  switch (raw.op) {
    case Op::kAll:
      return kAllSets;
    case Op::kNone:
      return 0;
    case Op::kAnd:
    case Op::kOr:
      break;
  }
  const bool is_and = raw.op == Op::kAnd;
  uint64_t table = is_and ? kAllSets : 0;
  for (const int i : raw.trigrams) {
    uint64_t has = 0;
    for (uint64_t set = 0; set < 64; ++set) {
      if (((set >> i) & 1) != 0) has |= uint64_t{1} << set;
    }
    table = is_and ? (table & has) : (table | has);
  }
  for (const RawQuery& sub : raw.subs) {
    table =
        is_and ? (table & RawTruthTable(sub)) : (table | RawTruthTable(sub));
  }
  return table;
}

// Canonicalizes through the public factories only.
TrigramQuery Canonicalize(const RawQuery& raw) {
  switch (raw.op) {
    case Op::kAll:
      return TrigramQuery::All();
    case Op::kNone:
      return TrigramQuery::None();
    case Op::kAnd:
    case Op::kOr:
      break;
  }
  const bool is_and = raw.op == Op::kAnd;
  TrigramQuery result = is_and ? TrigramQuery::All() : TrigramQuery::None();
  const auto combine = [is_and](TrigramQuery q, TrigramQuery r) {
    return is_and ? TrigramQuery::And(std::move(q), std::move(r))
                  : TrigramQuery::Or(std::move(q), std::move(r));
  };
  if (is_and) {
    std::vector<Trigram> trigrams;
    trigrams.reserve(raw.trigrams.size());
    for (const int i : raw.trigrams) trigrams.push_back(UniverseTrigram(i));
    result = TrigramQuery::AllOf(trigrams);
  } else {
    for (const int i : raw.trigrams) {
      result =
          combine(std::move(result), TrigramQuery::AllOf({UniverseTrigram(i)}));
    }
  }
  for (const RawQuery& sub : raw.subs) {
    result = combine(std::move(result), Canonicalize(sub));
  }
  return result;
}

RawQuery ToRaw(const TrigramQuery& query) {
  RawQuery raw;
  raw.op = query.op();
  for (const Trigram trigram : query.trigrams()) {
    raw.trigrams.push_back(UniverseIndex(trigram));
  }
  for (const TrigramQuery& sub : query.subs()) raw.subs.push_back(ToRaw(sub));
  return raw;
}

RawQuery RandomRaw(std::mt19937_64& rng, int depth) {
  RawQuery raw;
  const size_t kind = Below(rng, 20);
  if (kind == 0) return RawQuery{Op::kAll, {}, {}};
  if (kind == 1) return RawQuery{Op::kNone, {}, {}};
  raw.op = kind % 2 == 0 ? Op::kAnd : Op::kOr;
  const size_t num_trigrams = Below(rng, 4);
  for (size_t i = 0; i < num_trigrams; ++i) {
    raw.trigrams.push_back(static_cast<int>(Below(rng, kUniverse)));
  }
  if (depth > 0) {
    const size_t num_subs = Below(rng, 4);
    for (size_t i = 0; i < num_subs; ++i) {
      raw.subs.push_back(RandomRaw(rng, depth - 1));
    }
  }
  return raw;
}

TrigramQuery RandomQuery(std::mt19937_64& rng, int depth) {
  return Canonicalize(RandomRaw(rng, depth));
}

// Checks the structural invariants documented in trigram_query.h.
void ExpectCanonical(const TrigramQuery& query) {
  SCOPED_TRACE(query.ToString());
  if (query.op() == Op::kAll || query.op() == Op::kNone) {
    EXPECT_TRUE(query.trigrams().empty());
    EXPECT_TRUE(query.subs().empty());
    return;
  }
  for (size_t i = 1; i < query.trigrams().size(); ++i) {
    EXPECT_LT(query.trigrams()[i - 1], query.trigrams()[i]);
  }
  const size_t elements = query.trigrams().size() + query.subs().size();
  if (elements == 1) {
    EXPECT_EQ(query.op(), Op::kAnd);
    EXPECT_EQ(query.trigrams().size(), 1u);
  }
  for (const TrigramQuery& sub : query.subs()) {
    EXPECT_NE(sub.op(), query.op());
    EXPECT_TRUE(sub.op() == Op::kAnd || sub.op() == Op::kOr);
    EXPECT_GE(sub.trigrams().size() + sub.subs().size(), 2u);
    ExpectCanonical(sub);
  }
}

size_t MaxOrFanout(const TrigramQuery& query) {
  size_t max =
      query.op() == Op::kOr ? query.trigrams().size() + query.subs().size() : 0;
  for (const TrigramQuery& sub : query.subs()) {
    max = std::max(max, MaxOrFanout(sub));
  }
  return max;
}

class QueryAlgebraTest : public ::testing::Test {
 protected:
  QueryAlgebraTest() : seed_(TestSeed()), rng_(seed_) {}

  const uint64_t seed_;
  std::mt19937_64 rng_;
};

constexpr int kIterations = 20000;

TEST_F(QueryAlgebraTest, CanonicalizationPreservesSemantics) {
  SCOPED_TRACE(absl::StrCat("seed=", seed_));
  for (int i = 0; i < kIterations; ++i) {
    const RawQuery raw = RandomRaw(rng_, 3);
    const TrigramQuery query = Canonicalize(raw);
    ASSERT_EQ(TruthTable(query), RawTruthTable(raw))
        << "iteration " << i << ": " << query.ToString();
    ExpectCanonical(query);
    if (HasFailure()) return;
  }
}

TEST_F(QueryAlgebraTest, CanonicalizationIsIdempotent) {
  SCOPED_TRACE(absl::StrCat("seed=", seed_));
  for (int i = 0; i < kIterations; ++i) {
    const TrigramQuery query = RandomQuery(rng_, 3);
    ASSERT_EQ(Canonicalize(ToRaw(query)), query)
        << "iteration " << i << ": rebuilding " << query.ToString() << " gives "
        << Canonicalize(ToRaw(query)).ToString();
  }
}

TEST_F(QueryAlgebraTest, AndIsConjunctionAndOrIsDisjunction) {
  SCOPED_TRACE(absl::StrCat("seed=", seed_));
  for (int i = 0; i < kIterations; ++i) {
    const TrigramQuery q = RandomQuery(rng_, 3);
    const TrigramQuery r = RandomQuery(rng_, 3);
    const uint64_t tq = TruthTable(q);
    const uint64_t tr = TruthTable(r);
    const TrigramQuery q_and_r = TrigramQuery::And(q, r);
    const TrigramQuery q_or_r = TrigramQuery::Or(q, r);
    ASSERT_EQ(TruthTable(q_and_r), tq & tr)
        << q.ToString() << " AND " << r.ToString();
    ASSERT_EQ(TruthTable(q_or_r), tq | tr)
        << q.ToString() << " OR " << r.ToString();
    ExpectCanonical(q_and_r);
    ExpectCanonical(q_or_r);
    if (HasFailure()) return;
  }
}

TEST_F(QueryAlgebraTest, ImpliesIsSound) {
  SCOPED_TRACE(absl::StrCat("seed=", seed_));
  int proven = 0;
  for (int i = 0; i < kIterations; ++i) {
    const TrigramQuery q = RandomQuery(rng_, 2);
    // Relate r to q often enough that implications are common.
    const TrigramQuery r = Below(rng_, 2) == 0
                               ? TrigramQuery::Or(q, RandomQuery(rng_, 1))
                               : RandomQuery(rng_, 2);
    if (!q.Implies(r)) continue;
    ++proven;
    ASSERT_EQ(TruthTable(q) & ~TruthTable(r), 0u)
        << q.ToString() << " => " << r.ToString();
  }
  // Guards against a vacuous pass.
  EXPECT_GT(proven, kIterations / 10);
}

TEST_F(QueryAlgebraTest, WeakeningIsSoundAndBoundsOrFanout) {
  SCOPED_TRACE(absl::StrCat("seed=", seed_));
  int weakened = 0;
  for (int i = 0; i < kIterations; ++i) {
    const TrigramQuery query = RandomQuery(rng_, 3);
    const int max_or_fanout = 1 + static_cast<int>(Below(rng_, 4));
    const TrigramQuery weak = WeakenExpensiveOrs(query, max_or_fanout);
    // Weakening may only add trigram sets.
    ASSERT_EQ(TruthTable(query) & ~TruthTable(weak), 0u)
        << query.ToString() << " weakened to " << weak.ToString();
    ASSERT_LE(MaxOrFanout(weak), static_cast<size_t>(max_or_fanout))
        << query.ToString() << " weakened to " << weak.ToString();
    ASSERT_EQ(WeakenExpensiveOrs(weak, max_or_fanout), weak);
    ExpectCanonical(weak);
    if (HasFailure()) return;
    if (weak != query) ++weakened;
  }
  EXPECT_GT(weakened, kIterations / 10);
}

}  // namespace
}  // namespace ferret

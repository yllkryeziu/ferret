// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/query/trigram_query.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "ferret/text/trigram.h"

namespace ferret {
namespace {

using Op = TrigramQuery::Op;

// Single-trigram queries named after their bytes.
TrigramQuery T(const char (&s)[4]) {
  return TrigramQuery::AllOf(
      {MakeTrigram(static_cast<uint8_t>(s[0]), static_cast<uint8_t>(s[1]),
                   static_cast<uint8_t>(s[2]))});
}

TrigramQuery And(TrigramQuery q, TrigramQuery r) {
  return TrigramQuery::And(std::move(q), std::move(r));
}

TrigramQuery Or(TrigramQuery q, TrigramQuery r) {
  return TrigramQuery::Or(std::move(q), std::move(r));
}

TEST(TrigramQueryTest, DefaultIsAll) {
  EXPECT_EQ(TrigramQuery(), TrigramQuery::All());
  EXPECT_EQ(TrigramQuery().op(), Op::kAll);
  EXPECT_EQ(TrigramQuery::All().ToString(), "All");
  EXPECT_EQ(TrigramQuery::None().ToString(), "None");
}

TEST(TrigramQueryTest, AllOfSortsAndDeduplicates) {
  const Trigram abc = MakeTrigram('a', 'b', 'c');
  const Trigram xyz = MakeTrigram('x', 'y', 'z');
  const TrigramQuery q = TrigramQuery::AllOf({xyz, abc, xyz});
  EXPECT_EQ(q.op(), Op::kAnd);
  EXPECT_EQ(std::vector<Trigram>(q.trigrams().begin(), q.trigrams().end()),
            (std::vector<Trigram>{abc, xyz}));
  EXPECT_TRUE(q.subs().empty());
  EXPECT_EQ(TrigramQuery::AllOf({}), TrigramQuery::All());
}

TEST(TrigramQueryTest, SingleTrigramHasOneSpelling) {
  EXPECT_EQ(T("abc").op(), Op::kAnd);
  EXPECT_EQ(Or(T("abc"), TrigramQuery::None()), T("abc"));
  EXPECT_EQ(And(TrigramQuery::All(), T("abc")), T("abc"));
  EXPECT_EQ(T("abc").ToString(), "\"abc\"");
}

TEST(TrigramQueryTest, ConstantsFold) {
  const TrigramQuery q = Or(T("abc"), T("def"));
  EXPECT_EQ(And(q, TrigramQuery::All()), q);
  EXPECT_EQ(And(TrigramQuery::None(), q), TrigramQuery::None());
  EXPECT_EQ(Or(q, TrigramQuery::None()), q);
  EXPECT_EQ(Or(TrigramQuery::All(), q), TrigramQuery::All());
}

TEST(TrigramQueryTest, SameOpNodesMerge) {
  EXPECT_EQ(And(And(T("def"), T("abc")), T("ghi")).ToString(),
            "\"abc\" & \"def\" & \"ghi\"");
  EXPECT_EQ(Or(Or(T("ghi"), T("abc")), Or(T("def"), T("abc"))).ToString(),
            "\"abc\" | \"def\" | \"ghi\"");
}

TEST(TrigramQueryTest, OppositeOpBecomesSubquery) {
  const TrigramQuery q = And(T("abc"), Or(T("def"), T("ghi")));
  EXPECT_EQ(q.ToString(), "\"abc\" & (\"def\" | \"ghi\")");
  ASSERT_EQ(q.subs().size(), 1u);
  EXPECT_EQ(q.subs()[0].op(), Op::kOr);
}

TEST(TrigramQueryTest, ImpliedSiblingIsAbsorbed) {
  const TrigramQuery abc_def = And(T("abc"), T("def"));
  EXPECT_EQ(Or(T("abc"), abc_def), T("abc"));
  EXPECT_EQ(And(T("abc"), Or(T("abc"), T("def"))), T("abc"));
  // Merging brings "abc" next to a subquery it implies.
  EXPECT_EQ(And(And(T("xyz"), Or(T("abc"), T("def"))), T("abc")).ToString(),
            "\"abc\" & \"xyz\"");
}

TEST(TrigramQueryTest, CommonTrigramsAreFactored) {
  EXPECT_EQ(Or(And(T("abc"), T("def")), And(T("abc"), T("ghi"))).ToString(),
            "\"abc\" & (\"def\" | \"ghi\")");
  EXPECT_EQ(And(Or(T("abc"), T("def")), Or(T("abc"), T("ghi"))).ToString(),
            "\"abc\" | (\"def\" & \"ghi\")");
  // After "abc" and "def" are factored out, the rests merge into one OR.
  EXPECT_EQ(Or(And(And(T("abc"), T("def")), Or(T("jkl"), T("mno"))),
               And(And(T("abc"), T("def")), T("ghi")))
                .ToString(),
            "\"abc\" & \"def\" & (\"ghi\" | \"jkl\" | \"mno\")");
}

TEST(TrigramQueryTest, ImpliesProvableCases) {
  const TrigramQuery abc = T("abc");
  const TrigramQuery abc_or_def = Or(T("abc"), T("def"));
  const TrigramQuery abc_and_def = And(T("abc"), T("def"));
  EXPECT_TRUE(TrigramQuery::None().Implies(abc));
  EXPECT_TRUE(abc.Implies(TrigramQuery::All()));
  EXPECT_FALSE(TrigramQuery::All().Implies(abc));
  EXPECT_FALSE(abc.Implies(TrigramQuery::None()));
  EXPECT_TRUE(abc.Implies(abc_or_def));
  EXPECT_TRUE(abc_and_def.Implies(abc));
  EXPECT_TRUE(abc_and_def.Implies(abc_or_def));
  EXPECT_TRUE(abc_or_def.Implies(Or(abc_or_def, T("ghi"))));
  EXPECT_FALSE(abc_or_def.Implies(abc));
  EXPECT_FALSE(abc.Implies(abc_and_def));
}

TEST(TrigramQueryTest, EvaluateFollowsTheOps) {
  const TrigramQuery q = And(T("abc"), Or(T("def"), T("ghi")));
  const auto has = [](const std::vector<std::string>& present) {
    return [present](Trigram t) {
      for (const std::string& s : present) {
        if (TrigramAt(s, 0) == t) return true;
      }
      return false;
    };
  };
  EXPECT_TRUE(q.Evaluate(has({"abc", "ghi"})));
  EXPECT_FALSE(q.Evaluate(has({"abc"})));
  EXPECT_FALSE(q.Evaluate(has({"def", "ghi"})));
  EXPECT_TRUE(TrigramQuery::All().Evaluate(has({})));
  EXPECT_FALSE(TrigramQuery::None().Evaluate(has({"abc"})));
}

TEST(TrigramQueryTest, ToStringEscapesBytes) {
  const TrigramQuery kelvin =
      TrigramQuery::AllOf({MakeTrigram(0xE2, 0x84, 0xAA)});
  EXPECT_EQ(kelvin.ToString(), "\"\\xe2\\x84\\xaa\"");
  EXPECT_EQ(absl::StrCat(kelvin), kelvin.ToString());
}

TEST(TrigramQueryTest, WeakenExpensiveOrsReplacesWideOrs) {
  TrigramQuery wide = TrigramQuery::None();
  for (int i = 0; i < 33; ++i) {
    wide = Or(std::move(wide), TrigramQuery::AllOf({MakeTrigram(
                                   'a', 'b', static_cast<uint8_t>('A' + i))}));
  }
  ASSERT_EQ(wide.trigrams().size(), 33u);
  EXPECT_EQ(WeakenExpensiveOrs(wide, 33), wide);
  EXPECT_EQ(WeakenExpensiveOrs(wide, 32), TrigramQuery::All());
  // Only the wide OR goes; its AND parent keeps the other conjuncts.
  EXPECT_EQ(WeakenExpensiveOrs(And(T("xyz"), wide), 32), T("xyz"));
}

TEST(TrigramQueryTest, WeakenExpensiveOrsRecanonicalizes) {
  // "abc" | ("def" & (wide OR)): weakening leaves "def", which then merges
  // into the parent OR.
  TrigramQuery wide = TrigramQuery::None();
  for (int i = 0; i < 4; ++i) {
    wide = Or(std::move(wide), TrigramQuery::AllOf({MakeTrigram(
                                   'w', 'w', static_cast<uint8_t>('0' + i))}));
  }
  const TrigramQuery q = Or(T("abc"), And(T("def"), wide));
  EXPECT_EQ(WeakenExpensiveOrs(q, 3).ToString(), "\"abc\" | \"def\"");
  EXPECT_EQ(WeakenExpensiveOrs(q, 4), q);
}

}  // namespace
}  // namespace ferret

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The planner's soundness property: whenever RE2 matches a line, the plan
// accepts the trigrams of the match, and the ASCII-lowercased match contains
// the required literal. Checking the match rather than the whole text is the
// stronger oracle: queries are monotone, so a plan that accepts the match's
// trigrams accepts every text containing it.
//
// Everything runs through production code: CompilePattern(), the analyzer,
// and TrigramExtractor, the index-side definition of a text's trigrams.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "re2/re2.h"
#include "re2/regexp.h"
#include "util/utf.h"

#include "ferret/regex/analyze.h"
#include "ferret/regex/compile.h"
#include "ferret/regex/testing/regex_gen.h"
#include "ferret/testing/test_seed.h"
#include "ferret/text/fold.h"
#include "ferret/text/trigram.h"
#include "ferret/text/trigram_extractor.h"

namespace ferret {
namespace {

constexpr uint64_t kDefaultSeed = 20260923;
constexpr int kNumShards = 8;
constexpr int kPatternsPerShard = 25000;
constexpr int kTextsPerPattern = 8;
// Per shard, so the whole test visits each op at least 8x this often.
constexpr int kMinVisitsPerOpPerShard = 250;

// Plans `compiled`'s parse tree with `analyzer`, so that seeded-bug analyzers
// see exactly the tree the production analyzer saw.
RegexPlan Plan(TrigramAnalyzer& analyzer, const CompiledPattern& compiled) {
  return analyzer.Analyze(compiled.matcher().Regexp());
}

class SoundnessChecker {
 public:
  // Returns a description of how `plan` violates the contract on `text`, or
  // nullopt if it does not.
  std::optional<std::string> Violation(const CompiledPattern& compiled,
                                       const RegexPlan& plan,
                                       absl::string_view text) {
    absl::string_view match;
    if (!compiled.matcher().Match(text, 0, text.size(), RE2::UNANCHORED, &match,
                                  1)) {
      return std::nullopt;
    }
    ++matches_;
    trigrams_.clear();
    extractor_.Extract(match, trigrams_);
    const absl::flat_hash_set<Trigram> present(trigrams_.begin(),
                                               trigrams_.end());
    if (!plan.query.Evaluate(
            [&present](Trigram t) { return present.contains(t); })) {
      return absl::StrCat("plan ", plan.query.ToString(), " rejects match \"",
                          absl::CHexEscape(match), "\"");
    }
    // scan_literal is the longest, so it covers required_literal too.
    if (!plan.scan_literal.empty() &&
        !absl::StrContains(absl::AsciiStrToLower(match), plan.scan_literal)) {
      return absl::StrCat("match \"", absl::CHexEscape(match),
                          "\" lacks required literal \"",
                          absl::CHexEscape(plan.scan_literal), "\"");
    }
    if (!plan.scan_alternatives.empty()) {
      const std::string lowered = absl::AsciiStrToLower(match);
      bool any = false;
      for (const std::string& alternative : plan.scan_alternatives) {
        any |= absl::StrContains(lowered, alternative);
      }
      if (!any) {
        return absl::StrCat(
            "match \"", absl::CHexEscape(match), "\" contains none of ",
            plan.scan_alternatives.size(), " scan alternatives, the first \"",
            absl::CHexEscape(plan.scan_alternatives[0]), "\"");
      }
    }
    return std::nullopt;
  }

  int64_t matches() const { return matches_; }

 private:
  TrigramExtractor extractor_;
  std::vector<Trigram> trigrams_;
  int64_t matches_ = 0;
};

// Texts for one pattern: 60% samples of its language, 20% mutated samples,
// 20% random strings.
std::string MakeText(const CompiledPattern& compiled, std::mt19937_64& rng) {
  const uint64_t roll = rng() % 100;
  if (roll < 80) {
    std::optional<std::string> sample =
        ferret_testing::SampleMatch(compiled.matcher().Regexp(), rng);
    if (sample.has_value()) {
      return roll < 60 ? ferret_testing::Embed(*sample, rng)
                       : ferret_testing::Mutate(*std::move(sample), rng);
    }
  }
  return ferret_testing::RandomText(rng);
}

// Greedily deletes bytes from the pattern, then the text, while `fails`
// still holds, and returns a ready-to-paste regression line.
std::string Shrink(
    PatternSpec spec, std::string text,
    absl::FunctionRef<bool(const PatternSpec&, const std::string&)> fails) {
  for (bool progress = true; progress;) {
    progress = false;
    for (size_t i = 0; i < spec.pattern.size(); ++i) {
      PatternSpec smaller = spec;
      smaller.pattern.erase(i, 1);
      if (fails(smaller, text)) {
        spec = std::move(smaller);
        progress = true;
      }
    }
    for (size_t i = 0; i < text.size(); ++i) {
      std::string smaller = text;
      smaller.erase(i, 1);
      if (fails(spec, smaller)) {
        text = std::move(smaller);
        progress = true;
      }
    }
  }
  return absl::StrCat("{\"", absl::CHexEscape(spec.pattern), "\", ",
                      spec.case_sensitive ? "true" : "false", ", \"",
                      absl::CHexEscape(text), "\"},");
}

// True if the production planner violates the contract on (spec, text).
bool ProductionFails(const PatternSpec& spec, const std::string& text) {
  absl::StatusOr<std::unique_ptr<CompiledPattern>> compiled =
      CompilePattern(spec);
  if (!compiled.ok()) return false;
  SoundnessChecker checker;
  const RegexPlan plan{(*compiled)->query(), (*compiled)->required_literal(),
                       (*compiled)->plan_complete(),
                       (*compiled)->scan_literal(),
                       (*compiled)->scan_alternatives()};
  return checker.Violation(**compiled, plan, text).has_value();
}

void CountOps(re2::Regexp* re, std::vector<int64_t>& visits) {
  std::vector<re2::Regexp*> stack = {re};
  while (!stack.empty()) {
    re2::Regexp* node = stack.back();
    stack.pop_back();
    ++visits[static_cast<size_t>(node->op())];
    for (int i = 0; i < node->nsub(); ++i) stack.push_back(node->sub()[i]);
  }
}

// Checks every text of every pattern against the production planner.
// Returns the number of patterns that compiled.
int64_t CheckPatterns(const std::vector<PatternSpec>& specs,
                      std::mt19937_64& rng, SoundnessChecker& checker,
                      std::vector<int64_t>& op_visits) {
  int64_t compiled_count = 0;
  for (const PatternSpec& spec : specs) {
    absl::StatusOr<std::unique_ptr<CompiledPattern>> compiled =
        CompilePattern(spec);
    if (!compiled.ok()) continue;
    ++compiled_count;
    CountOps((*compiled)->matcher().Regexp(), op_visits);
    const RegexPlan plan{(*compiled)->query(), (*compiled)->required_literal(),
                         (*compiled)->plan_complete(),
                         (*compiled)->scan_literal(),
                         (*compiled)->scan_alternatives()};
    for (int t = 0; t < kTextsPerPattern; ++t) {
      const std::string text = MakeText(**compiled, rng);
      if (std::optional<std::string> violation =
              checker.Violation(**compiled, plan, text)) {
        ADD_FAILURE() << *violation << "\n  pattern: \""
                      << absl::CHexEscape(spec.pattern)
                      << "\" case_sensitive=" << spec.case_sensitive
                      << "\n  shrunk regression:\n    "
                      << Shrink(spec, text, ProductionFails);
        return compiled_count;
      }
    }
  }
  return compiled_count;
}

class AnalyzeSoundnessTest : public ::testing::TestWithParam<int> {
 protected:
  uint64_t Seed() const {
    return ferret_testing::TestSeed(kDefaultSeed) +
           static_cast<uint64_t>(GetParam());
  }
};

TEST_P(AnalyzeSoundnessTest, RandomPatternsAreSound) {
  const uint64_t seed = Seed();
  SCOPED_TRACE(absl::StrCat("seed ", seed));
  std::mt19937_64 rng(seed);
  std::vector<PatternSpec> specs;
  specs.reserve(kPatternsPerShard);
  for (int i = 0; i < kPatternsPerShard; ++i) {
    specs.push_back({.pattern = ferret_testing::RandomPattern(rng),
                     .case_sensitive = rng() % 100 >= 30});
  }
  SoundnessChecker checker;
  std::vector<int64_t> op_visits(re2::kMaxRegexpOp + 1, 0);
  const int64_t compiled = CheckPatterns(specs, rng, checker, op_visits);

  // Vacuity guards: the oracle must fire often, and every op the planner
  // sees must be exercised. AnyChar is absent because NeverNL parses "." as
  // the class [^\n]; AnyByte (\C) is rejected; HaveMatch comes only from
  // RE2::Set; Capture is absent because patterns compile with NeverCapture.
  EXPECT_GE(compiled, kPatternsPerShard / 2);
  EXPECT_GE(checker.matches(), compiled * kTextsPerPattern * 3 / 10)
      << "RE2 matched too few texts for the oracle to mean much";
  for (const re2::RegexpOp op :
       {re2::kRegexpNoMatch, re2::kRegexpEmptyMatch, re2::kRegexpLiteral,
        re2::kRegexpLiteralString, re2::kRegexpConcat, re2::kRegexpAlternate,
        re2::kRegexpStar, re2::kRegexpPlus, re2::kRegexpQuest,
        re2::kRegexpRepeat, re2::kRegexpBeginLine, re2::kRegexpEndLine,
        re2::kRegexpWordBoundary, re2::kRegexpNoWordBoundary,
        re2::kRegexpBeginText, re2::kRegexpEndText, re2::kRegexpCharClass}) {
    EXPECT_GE(op_visits[static_cast<size_t>(op)], kMinVisitsPerOpPerShard)
        << "RE2 op " << static_cast<int>(op) << " is under-exercised";
  }
}

// Every tree with at most three leaves over a small alphabet of fold-orbit
// and UTF-8 leaves, split across shards.
TEST_P(AnalyzeSoundnessTest, SmallTreesAreSound) {
  static constexpr absl::string_view kLeaves[] = {"a", "K",    "\\x{212A}",
                                                  ".", "[bk]", "\xC3\xA9"};
  static constexpr absl::string_view kSuffixes[] = {"", "*", "?", "{0}", "+"};
  std::vector<std::string> atoms;
  for (const absl::string_view leaf : kLeaves) {
    for (const absl::string_view suffix : kSuffixes) {
      atoms.push_back(absl::StrCat("(?:", leaf, ")", suffix));
    }
  }
  std::vector<std::string> patterns = atoms;
  for (const std::string& x : atoms) {
    for (const std::string& y : atoms) {
      patterns.push_back(x + y);
      patterns.push_back(absl::StrCat(x, "|", y));
      for (const absl::string_view leaf : kLeaves) {
        const std::string z = absl::StrCat("(?:", leaf, ")");
        patterns.push_back(absl::StrCat(x, y, z));
        patterns.push_back(absl::StrCat("(?:", x, "|", y, ")", z));
        patterns.push_back(absl::StrCat(x, "(?:", y, "|", z, ")"));
        patterns.push_back(absl::StrCat(x, "|", y, z));
      }
    }
  }
  std::vector<PatternSpec> specs;
  for (size_t i = static_cast<size_t>(GetParam()); i < patterns.size();
       i += kNumShards) {
    specs.push_back({.pattern = patterns[i], .case_sensitive = i % 3 != 0});
  }
  std::mt19937_64 rng(Seed());
  SoundnessChecker checker;
  std::vector<int64_t> op_visits(re2::kMaxRegexpOp + 1, 0);
  CheckPatterns(specs, rng, checker, op_visits);
}

INSTANTIATE_TEST_SUITE_P(Shards, AnalyzeSoundnessTest,
                         ::testing::Range(0, kNumShards));

// Patterns that once failed, or that pin a subtle rule.
TEST(AnalyzeSoundnessRegressionTest, KnownCasesAreSound) {
  struct Regression {
    absl::string_view pattern;
    bool case_sensitive;
    absl::string_view text;
  };
  static constexpr Regression kRegressions[] = {
      // (?i)k matches U+212A, which F folds to 'k'.
      {"(?i)black", true, "blac\xE2\x84\xAA"},
      {"black", false, "blac\xE2\x84\xAA"},
      // (?i)s matches U+017F, which F folds to 's'.
      {"(?i)class", true, "cla\xC5\xBF\xC5\xBF"},
      // x{0} stays in RE2's tree and matches only "".
      {"x{0}abc", true, "abc"},
      // A literal Kelvin sign plans the trigram "akc" but requires no literal.
      {"a\\x{212A}c", true,
       "a\xE2\x84\xAA"
       "c"},
      // \s must not be rejected, and matches within a line only.
      {"foo\\s+bar", true, "foo \t bar"},
      // RE2 factors foo|foobar into foo(?:|bar).
      {"(?:foo|foobar)baz", true, "foobarbaz"},
  };
  for (const Regression& r : kRegressions) {
    SCOPED_TRACE(r.pattern);
    const PatternSpec spec{.pattern = std::string(r.pattern),
                           .case_sensitive = r.case_sensitive};
    EXPECT_FALSE(ProductionFails(spec, std::string(r.text)));
    // Each regression must exercise the oracle, not pass vacuously.
    absl::StatusOr<std::unique_ptr<CompiledPattern>> compiled =
        CompilePattern(spec);
    ASSERT_TRUE(compiled.ok()) << compiled.status();
    EXPECT_TRUE(RE2::PartialMatch(r.text, (*compiled)->matcher()));
  }
}

// Mutation guard: planners with a seeded bug must be caught quickly, which
// shows that the generator and oracle can see each class of mistake.

std::string EncodeRune(re2::Rune rune) {
  char buffer[re2::UTFmax];
  return std::string(buffer,
                     static_cast<size_t>(re2::runetochar(buffer, &rune)));
}

// Plans only the lowest rune of each small character class.
class ClassLowestRuneOnlyAnalyzer final : public TrigramAnalyzer {
 public:
  using TrigramAnalyzer::TrigramAnalyzer;

 protected:
  RegexInfo PostVisit(re2::Regexp* re, RegexInfo parent_arg, RegexInfo pre_arg,
                      RegexInfo* child_args, int nchild_args) override {
    if (re->op() == re2::kRegexpCharClass && !re->cc()->empty() &&
        re->cc()->size() <= options().class_limit) {
      return Leaf({Fold(EncodeRune(re->cc()->begin()->lo))});
    }
    return TrigramAnalyzer::PostVisit(
        re, std::move(parent_arg), std::move(pre_arg), child_args, nchild_args);
  }
};

// Forgets to fold literals.
class UnfoldedLiteralAnalyzer final : public TrigramAnalyzer {
 public:
  using TrigramAnalyzer::TrigramAnalyzer;

 protected:
  RegexInfo PostVisit(re2::Regexp* re, RegexInfo parent_arg, RegexInfo pre_arg,
                      RegexInfo* child_args, int nchild_args) override {
    if (re->op() == re2::kRegexpLiteral) return Leaf({EncodeRune(re->rune())});
    if (re->op() == re2::kRegexpLiteralString) {
      std::string s;
      for (int i = 0; i < re->nrunes(); ++i) s += EncodeRune(re->runes()[i]);
      return Leaf({std::move(s)});
    }
    return TrigramAnalyzer::PostVisit(
        re, std::move(parent_arg), std::move(pre_arg), child_args, nchild_args);
  }
};

// Plans an alternation as its first branch.
class FirstAlternativeOnlyAnalyzer final : public TrigramAnalyzer {
 public:
  using TrigramAnalyzer::TrigramAnalyzer;

 protected:
  RegexInfo PostVisit(re2::Regexp* re, RegexInfo parent_arg, RegexInfo pre_arg,
                      RegexInfo* child_args, int nchild_args) override {
    if (re->op() == re2::kRegexpAlternate) return std::move(child_args[0]);
    return TrigramAnalyzer::PostVisit(
        re, std::move(parent_arg), std::move(pre_arg), child_args, nchild_args);
  }
};

// Plans x* as x, forgetting that it can match "".
class StarAsOperandAnalyzer final : public TrigramAnalyzer {
 public:
  using TrigramAnalyzer::TrigramAnalyzer;

 protected:
  RegexInfo PostVisit(re2::Regexp* re, RegexInfo parent_arg, RegexInfo pre_arg,
                      RegexInfo* child_args, int nchild_args) override {
    if (re->op() == re2::kRegexpStar) return std::move(child_args[0]);
    return TrigramAnalyzer::PostVisit(
        re, std::move(parent_arg), std::move(pre_arg), child_args, nchild_args);
  }
};

// Returns the number of (pattern, text) cases it took to catch `mutant`, or
// -1 if it survived `max_cases`.
int64_t CasesToCatch(TrigramAnalyzer& mutant, std::mt19937_64& rng,
                     int64_t max_cases) {
  SoundnessChecker checker;
  int64_t cases = 0;
  while (cases < max_cases) {
    const PatternSpec spec{.pattern = ferret_testing::RandomPattern(rng),
                           .case_sensitive = rng() % 100 >= 30};
    absl::StatusOr<std::unique_ptr<CompiledPattern>> compiled =
        CompilePattern(spec);
    if (!compiled.ok()) continue;
    const RegexPlan plan = Plan(mutant, **compiled);
    for (int t = 0; t < kTextsPerPattern; ++t, ++cases) {
      if (checker.Violation(**compiled, plan, MakeText(**compiled, rng))) {
        return cases;
      }
    }
  }
  return -1;
}

TEST(AnalyzeSoundnessMutationTest, SeededPlannerBugsAreCaught) {
  constexpr int64_t kMaxCases = 200000;
  const RegexPlanOptions options;
  ClassLowestRuneOnlyAnalyzer class_lowest_rune_only(options);
  UnfoldedLiteralAnalyzer unfolded_literal(options);
  FirstAlternativeOnlyAnalyzer first_alternative_only(options);
  StarAsOperandAnalyzer star_as_operand(options);
  const struct {
    absl::string_view name;
    TrigramAnalyzer* mutant;
  } k_mutants[] = {
      {"ClassLowestRuneOnly", &class_lowest_rune_only},
      {"UnfoldedLiteral", &unfolded_literal},
      {"FirstAlternativeOnly", &first_alternative_only},
      {"StarAsOperand", &star_as_operand},
  };
  std::mt19937_64 rng(ferret_testing::TestSeed(kDefaultSeed));
  for (const auto& [name, mutant] : k_mutants) {
    const int64_t cases = CasesToCatch(*mutant, rng, kMaxCases);
    EXPECT_GE(cases, 0) << name << " survived " << kMaxCases << " cases";
  }
}

}  // namespace
}  // namespace ferret

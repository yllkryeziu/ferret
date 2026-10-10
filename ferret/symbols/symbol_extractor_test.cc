// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Tests the public extractor contract independently of query coverage.

#include "ferret/symbols/symbol_extractor.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/symbols/testing/sample_sources.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/text/language.h"

namespace ferret {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

TEST(SymbolExtractorTest, RejectsUnsupportedLanguages) {
  SymbolExtractor extractor;
  for (Language language :
       {Language::kUnknown, Language::kGo, Language::kObjectiveCpp}) {
    EXPECT_THAT(extractor.Extract(language, ""),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(SymbolExtractorTest, InputLimitIsInclusive) {
  SymbolExtractor extractor({.max_input_bytes = 8});
  EXPECT_THAT(extractor.Extract(Language::kC, "int f();"), IsOk());
  EXPECT_THAT(extractor.Extract(Language::kC, "int f(); "),
              StatusIs(absl::StatusCode::kResourceExhausted));
}

std::string ManyFunctions(int count) {
  std::string source;
  for (int i = 0; i < count; ++i) {
    absl::StrAppend(&source, "static int f", i, "(int x) { return x + ", i,
                    "; }\n");
  }
  return source;
}

TEST(SymbolExtractorTest, StopsAtTheWorkBudgetAndCanBeReused) {
  SymbolExtractor extractor({.max_progress_checks = 0});
  EXPECT_THAT(extractor.Extract(Language::kC, ManyFunctions(2000)),
              StatusIs(absl::StatusCode::kResourceExhausted));
  // Too little work for a progress check.
  EXPECT_THAT(extractor.Extract(Language::kC, "int alive(void) {}"), IsOk());
}

// The budget counts work, not time, so the outcome at the boundary is the
// same on every run.
TEST(SymbolExtractorTest, BudgetBoundaryIsExactAndRepeatable) {
  const std::string source = ManyFunctions(2000);
  SymbolExtractor unlimited({.max_progress_checks = int64_t{1} << 40});
  FERRET_ASSERT_OK_AND_ASSIGN(const std::vector<ExtractedSymbol> expected,
                              unlimited.Extract(Language::kC, source));
  ASSERT_EQ(expected.size(), 2000u);
  // The smallest budget that suffices.
  int64_t low = 0;
  int64_t high = int64_t{1} << 20;
  while (low < high) {
    const int64_t mid = low + (high - low) / 2;
    SymbolExtractor probe({.max_progress_checks = mid});
    if (probe.Extract(Language::kC, source).ok()) {
      high = mid;
    } else {
      low = mid + 1;
    }
  }
  ASSERT_GT(low, 0);
  ASSERT_LT(low, SymbolExtractorOptions().max_progress_checks);
  SymbolExtractor exact({.max_progress_checks = low});
  SymbolExtractor short_by_one({.max_progress_checks = low - 1});
  for (int run = 0; run < 3; ++run) {
    EXPECT_THAT(short_by_one.Extract(Language::kC, source),
                StatusIs(absl::StatusCode::kResourceExhausted))
        << run;
    EXPECT_THAT(exact.Extract(Language::kC, source),
                IsOkAndHolds(ElementsAreArray(expected)))
        << run;
  }
}

TEST(SymbolExtractorTest, ReturnsNameOrderAndOwnsResults) {
  SymbolExtractor extractor;
  std::string source = "int z(void);\nint a(void) {}\n";
  FERRET_ASSERT_OK_AND_ASSIGN(std::vector<ExtractedSymbol> symbols,
                              extractor.Extract(Language::kC, source));
  ASSERT_EQ(symbols.size(), size_t{2});
  EXPECT_THAT(std::vector<std::string>({symbols[0].name, symbols[1].name}),
              ElementsAre("z", "a"));
  EXPECT_FALSE(symbols[0].is_definition);
  EXPECT_TRUE(symbols[1].is_definition);
  source.clear();
  EXPECT_EQ(symbols[0].name, "z");
}

TEST(SymbolExtractorTest, DeterministicAcrossCallsInstancesAndLanguages) {
  SymbolExtractor first;
  SymbolExtractor second;
  for (int repeat = 0; repeat < 10; ++repeat) {
    for (Language language :
         {Language::kC, Language::kCpp, Language::kPython}) {
      const absl::string_view source = ferret_testing::SampleSource(language);
      FERRET_ASSERT_OK_AND_ASSIGN(std::vector<ExtractedSymbol> expected,
                                  second.Extract(language, source));
      FERRET_ASSERT_OK_AND_ASSIGN(std::vector<ExtractedSymbol> actual,
                                  first.Extract(language, source));
      EXPECT_EQ(actual, expected);
      EXPECT_FALSE(actual.empty());
    }
  }
}

TEST(SymbolExtractorTest, ReusesParserAcrossManyDifferentInputs) {
  SymbolExtractor extractor;
  for (int i = 0; i < 100; ++i) {
    const std::string name = absl::StrCat("symbol", i);
    const std::string c = absl::StrCat("int ", name, "(void) {}\n");
    const std::string cpp = absl::StrCat("class FILE_EXPORT ", name, " {};\n");
    const std::string python = absl::StrCat("def ", name, "(): pass\n");
    for (const auto& [language, source] :
         {std::pair{Language::kC, c}, std::pair{Language::kCpp, cpp},
          std::pair{Language::kPython, python}}) {
      FERRET_ASSERT_OK_AND_ASSIGN(std::vector<ExtractedSymbol> symbols,
                                  extractor.Extract(language, source));
      ASSERT_EQ(symbols.size(), size_t{1});
      EXPECT_EQ(symbols[0].name, name);
      EXPECT_TRUE(symbols[0].is_definition);
    }
  }
}

TEST(SymbolExtractorTest, RecoversBeforeAndAfterSyntaxError) {
  SymbolExtractor extractor;
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::vector<ExtractedSymbol> symbols,
      extractor.Extract(
          Language::kC,
          "int before(void) {}\n@ ! broken ;\nint after(void) {}\n"));
  EXPECT_THAT(symbols, ::testing::Contains(
                           ::testing::Field(&ExtractedSymbol::name, "before")));
  EXPECT_THAT(symbols, ::testing::Contains(
                           ::testing::Field(&ExtractedSymbol::name, "after")));
}

TEST(SymbolExtractorTest, EmptyCrLfAndNoFinalNewline) {
  SymbolExtractor extractor;
  for (Language language : {Language::kC, Language::kCpp, Language::kPython}) {
    FERRET_ASSERT_OK_AND_ASSIGN(std::vector<ExtractedSymbol> empty,
                                extractor.Extract(language, ""));
    EXPECT_THAT(empty, IsEmpty());
  }
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::vector<ExtractedSymbol> symbols,
      extractor.Extract(Language::kC,
                        "\r\nint first(void);\r\nint last(void) {}"));
  ASSERT_EQ(symbols.size(), size_t{2});
  EXPECT_EQ(symbols[0].line, uint32_t{2});
  EXPECT_EQ(symbols[1].line, uint32_t{3});
  EXPECT_EQ(symbols[1].name, "last");
}

TEST(SymbolExtractorTest, NonUtf8BytesDoNotPreventExtraction) {
  SymbolExtractor extractor;
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::vector<ExtractedSymbol> symbols,
      extractor.Extract(Language::kC, "/* \xff */\nint valid(void) {}"));
  ASSERT_EQ(symbols.size(), size_t{1});
  EXPECT_EQ(symbols[0].name, "valid");
}

}  // namespace
}  // namespace ferret

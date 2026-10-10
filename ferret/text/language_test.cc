// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/language.h"

#include "gtest/gtest.h"

namespace ferret {
namespace {

TEST(LanguageTest, DetectsByExtension) {
  EXPECT_EQ(DetectLanguage("net/http/http_cache.cc", ""), Language::kCpp);
  EXPECT_EQ(DetectLanguage("kernel/sched/core.c", ""), Language::kC);
  EXPECT_EQ(DetectLanguage("django/db/models/query.py", ""), Language::kPython);
  EXPECT_EQ(DetectLanguage("ui/cocoa/window.mm", ""), Language::kObjectiveCpp);
  EXPECT_EQ(DetectLanguage("SRC/MAIN.CPP", ""), Language::kCpp);
}

TEST(LanguageTest, UnknownWithoutExtension) {
  EXPECT_EQ(DetectLanguage("Makefile", ""), Language::kUnknown);
  EXPECT_EQ(DetectLanguage(".gitignore", ""), Language::kUnknown);
  EXPECT_EQ(DetectLanguage("dir.v2/README", ""), Language::kUnknown);
}

TEST(LanguageTest, HeaderIsCppOnlyWithCppConstructs) {
  EXPECT_EQ(DetectLanguage("include/linux/list.h", "struct list_head {"),
            Language::kC);
  EXPECT_EQ(DetectLanguage("base/strings/string_util.h",
                           "namespace base {\nclass Foo;"),
            Language::kCpp);
}

TEST(LanguageTest, SymbolGrammarsCoverCCppPythonOnly) {
  EXPECT_TRUE(HasSymbolGrammar(Language::kC));
  EXPECT_TRUE(HasSymbolGrammar(Language::kCpp));
  EXPECT_TRUE(HasSymbolGrammar(Language::kPython));
  EXPECT_FALSE(HasSymbolGrammar(Language::kObjectiveCpp));
  EXPECT_FALSE(HasSymbolGrammar(Language::kGo));
}

TEST(LanguageTest, NamesAreStable) {
  EXPECT_EQ(LanguageName(Language::kCpp), "cpp");
  EXPECT_EQ(LanguageName(Language::kUnknown), "unknown");
}

}  // namespace
}  // namespace ferret

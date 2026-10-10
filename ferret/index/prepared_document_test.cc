// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/prepared_document.h"

#include <string>

#include "gtest/gtest.h"

#include "ferret/git/blob_hash.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/segment_writer.h"
#include "ferret/text/eligibility.h"
#include "ferret/text/language.h"
#include "ferret/text/trigram_extractor.h"

namespace ferret {
namespace {

TEST(PreparedDocumentTest, EligibleFileBindsMetadata) {
  TrigramExtractor extractor;
  const std::string bytes = "int main() {}\n";
  const PreparedDocument prepared =
      PrepareDocument("src/main.cc", bytes, {}, extractor);
  EXPECT_EQ(prepared.skip_reason, SkipReason::kNone);
  EXPECT_EQ(prepared.language, Language::kCpp);
  EXPECT_EQ(prepared.rank_key, ComputeRankKey("src/main.cc", bytes.size(),
                                              false, HashBlob(bytes)));
  const SegmentDocument bound = prepared.Bind("src/main.cc", bytes);
  EXPECT_EQ(bound.content, bytes);
  EXPECT_EQ(bound.hint_path, "src/main.cc");
  EXPECT_EQ(bound.rank_key, prepared.rank_key);
}

TEST(PreparedDocumentTest, SkipsIneligibleFiles) {
  TrigramExtractor extractor;
  EXPECT_EQ(PrepareDocument("empty.cc", "", {}, extractor).skip_reason,
            SkipReason::kEmpty);
  EXPECT_EQ(PrepareDocument("nul.cc", std::string("x\0y", 3), {}, extractor)
                .skip_reason,
            SkipReason::kContainsNul);
}

TEST(PreparedDocumentTest, LongLinesChangeRankAndLanguage) {
  TrigramExtractor extractor;
  EligibilityOptions options;
  options.long_line_bytes = 10;
  const std::string bytes = "#!/usr/bin/env python\nprint('hello')\n";
  const PreparedDocument prepared =
      PrepareDocument("scripts/run", bytes, options, extractor);
  EXPECT_EQ(prepared.skip_reason, SkipReason::kNone);
  EXPECT_TRUE(prepared.long_lines);
  EXPECT_EQ(prepared.language, DetectLanguage("scripts/run", bytes));
  EXPECT_EQ(prepared.rank_key,
            ComputeRankKey("scripts/run", bytes.size(), true, HashBlob(bytes)));
}

}  // namespace
}  // namespace ferret

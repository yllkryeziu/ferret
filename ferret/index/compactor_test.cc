// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/compactor.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/git/blob_hash.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/segment.h"
#include "ferret/index/segment_writer.h"
#include "ferret/postings/ids.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/testing/test_seed.h"
#include "ferret/text/language.h"
#include "ferret/text/trigram.h"

namespace ferret {
namespace {

struct CorpusDoc {
  std::string path;
  std::string content;
  bool long_lines = false;
};

SegmentDocument AsDocument(const CorpusDoc& doc) {
  return {.hint_path = doc.path,
          .content = doc.content,
          .rank_key = ComputeRankKey(doc.path, doc.content.size(),
                                     doc.long_lines, HashBlob(doc.content)),
          .language = Language::kCpp,
          .long_lines = doc.long_lines};
}

void Sort(std::vector<CorpusDoc>& docs) {
  std::stable_sort(docs.begin(), docs.end(), [](const auto& a, const auto& b) {
    return AsDocument(a).rank_key < AsDocument(b).rank_key;
  });
}

void CheckIdentity(std::vector<std::vector<CorpusDoc>> partitions,
                   const std::vector<std::vector<bool>>& kept,
                   const SegmentWriterOptions& options = {}) {
  ferret_testing::TempDir dir;
  std::vector<std::unique_ptr<Segment>> opened;
  std::vector<const Segment*> inputs;
  std::vector<CorpusDoc> expected;
  for (size_t i = 0; i < partitions.size(); ++i) {
    Sort(partitions[i]);
    const std::string path = dir.Join(absl::StrCat(i, ".fseg"));
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto writer, SegmentWriter::Create(*Env::Default(), path, options));
    for (const CorpusDoc& doc : partitions[i]) {
      ASSERT_THAT(writer->Add(AsDocument(doc)), absl_testing::IsOk());
    }
    ASSERT_THAT(writer->Finish().status(), absl_testing::IsOk());
    FERRET_ASSERT_OK_AND_ASSIGN(auto segment,
                                Segment::Open(*Env::Default(), path));
    inputs.push_back(segment.get());
    opened.push_back(std::move(segment));
    for (size_t d = 0; d < partitions[i].size(); ++d) {
      if (kept[i][d]) expected.push_back(partitions[i][d]);
    }
  }
  Sort(expected);
  std::vector<CorpusDoc> unique;
  for (const CorpusDoc& doc : expected) {
    bool seen = false;
    for (const CorpusDoc& old : unique) {
      if (AsDocument(old).rank_key.oid == AsDocument(doc).rank_key.oid) {
        seen = true;
      }
    }
    if (!seen) unique.push_back(doc);
  }
  // The first path at the best RankKey is the first input's copy.
  const std::string direct = dir.Join("direct.fseg");
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto writer, SegmentWriter::Create(*Env::Default(), direct, options));
  for (const CorpusDoc& doc : unique) {
    ASSERT_THAT(writer->Add(AsDocument(doc)), absl_testing::IsOk());
  }
  ASSERT_THAT(writer->Finish().status(), absl_testing::IsOk());
  const std::string merged = dir.Join("merged.fseg");
  FERRET_ASSERT_OK_AND_ASSIGN(
      const SegmentStats stats,
      MergeSegments(
          *Env::Default(), inputs,
          [&](size_t i, DocId d) { return kept[i][d]; }, merged, options));
  EXPECT_EQ(stats.num_docs, unique.size());
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment,
                              Segment::Open(*Env::Default(), merged));
  EXPECT_THAT(segment->VerifyChecksums(), absl_testing::IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(std::string actual,
                              Env::Default()->ReadFile(merged));
  FERRET_ASSERT_OK_AND_ASSIGN(std::string reference,
                              Env::Default()->ReadFile(direct));
  EXPECT_EQ(actual, reference);
}

TEST(CompactorTest, MergedSegmentHasShortLineTrigramsOnlyIfEveryInputDoes) {
  ferret_testing::TempDir dir;
  const std::vector<CorpusDoc> docs = {{"a.cc", "x\n};\n"},
                                       {"b.cc", "ab\nyy\n"}};
  auto write = [&](const std::string& name, const CorpusDoc& doc,
                   bool short_lines) {
    SegmentWriterOptions options;
    options.write_short_line_trigrams = short_lines;
    auto writer =
        SegmentWriter::Create(*Env::Default(), dir.Join(name), options);
    EXPECT_THAT(writer.status(), absl_testing::IsOk());
    EXPECT_THAT((*writer)->Add(AsDocument(doc)), absl_testing::IsOk());
    EXPECT_THAT((*writer)->Finish().status(), absl_testing::IsOk());
    auto segment = Segment::Open(*Env::Default(), dir.Join(name));
    EXPECT_THAT(segment.status(), absl_testing::IsOk());
    return *std::move(segment);
  };
  const auto modern_a = write("a.fseg", docs[0], true);
  const auto modern_b = write("b.fseg", docs[1], true);
  const auto legacy_b = write("c.fseg", docs[1], false);
  EXPECT_TRUE(modern_a->has_short_line_trigrams());
  EXPECT_FALSE(legacy_b->has_short_line_trigrams());
  auto merge = [&](const std::vector<const Segment*>& inputs) {
    const std::string out = dir.Join("merged.fseg");
    EXPECT_THAT(MergeSegments(*Env::Default(), inputs,
                              [](size_t, DocId) { return true; }, out, {})
                    .status(),
                absl_testing::IsOk());
    auto segment = Segment::Open(*Env::Default(), out);
    EXPECT_THAT(segment.status(), absl_testing::IsOk());
    return *std::move(segment);
  };
  const auto both_modern = merge({modern_a.get(), modern_b.get()});
  EXPECT_TRUE(both_modern->has_short_line_trigrams());
  const Trigram braces = MakeTrigram('}', ';', '\n');
  FERRET_ASSERT_OK_AND_ASSIGN(auto list, both_modern->FindPostingList(braces));
  EXPECT_EQ(list.size(), 1u);
  const auto mixed = merge({modern_a.get(), legacy_b.get()});
  EXPECT_FALSE(mixed->has_short_line_trigrams());
}

TEST(CompactorTest, CancellationDoesNotCreateOutput) {
  ferret_testing::TempDir dir;
  const std::string input_path = dir.Join("input.fseg");
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto writer, SegmentWriter::Create(*Env::Default(), input_path));
  ASSERT_THAT(writer->Add(AsDocument({"a.cc", "needle\n"})),
              absl_testing::IsOk());
  ASSERT_THAT(writer->Finish().status(), absl_testing::IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment,
                              Segment::Open(*Env::Default(), input_path));
  const std::vector<const Segment*> inputs = {segment.get()};
  std::atomic<bool> cancelled{true};
  const std::string output = dir.Join("partial.fseg");
  EXPECT_THAT(MergeSegments(
                  *Env::Default(), inputs, [](size_t, DocId) { return true; },
                  output, {}, &cancelled),
              absl_testing::StatusIs(absl::StatusCode::kCancelled));
  FERRET_ASSERT_OK_AND_ASSIGN(bool exists, Env::Default()->FileExists(output));
  EXPECT_FALSE(exists);
}

TEST(CompactorTest, RandomCorporaAreByteIdentical) {
  const uint64_t seed = ferret_testing::TestSeed(90210);
  SCOPED_TRACE(seed);
  std::mt19937_64 random(seed);
  for (int trial = 0; trial < 12; ++trial) {
    const size_t count = 2 + random() % 4;
    std::vector<std::vector<CorpusDoc>> parts(count);
    std::vector<std::vector<bool>> kept(count);
    for (size_t i = 0; i < count; ++i) {
      for (int j = 0; j < 8; ++j) {
        const int blob = static_cast<int>(random() % 15);
        parts[i].push_back(
            {absl::StrCat("src/", i, "/", j, ".cc"),
             absl::StrCat("int token", blob, "() { return ", blob, "; }\n")});
      }
      Sort(parts[i]);
      for (size_t j = 0; j < parts[i].size(); ++j) {
        kept[i].push_back(random() % 3 != 0);
      }
    }
    CheckIdentity(std::move(parts), kept, {.chunk_bytes = 12});
  }
}

TEST(CompactorTest, SingleInputAndAllDropped) {
  const std::vector<std::vector<CorpusDoc>> parts = {
      {{"foo.cc", "unique trigram only here\n"}}};
  CheckIdentity(parts, {{true}});
  CheckIdentity(parts, {{false}});
}

TEST(CompactorTest, KeepsLongLineAndLanguageMetadata) {
  CheckIdentity({{{"a.cc", "/* comment\n continues */\n"},
                  {"long.cc", std::string(10000, 'x'), true}},
                 {{"b.cc", "only xyz in this input\n"}}},
                {{true, true}, {true}}, {.chunk_bytes = 32});
}

TEST(CompactorTest, PicksLowestRankKeyAcrossInputs) {
  CheckIdentity(
      {{{"deep/vendor/a.cc", "same blob\n"}, {"src/other.cc", "only first\n"}},
       {{"a.cc", "same blob\n"}, {"src/more.cc", "only second\n"}}},
      {{true, true}, {true, true}});
}

TEST(CompactorTest, NoInputsProducesEmptySegment) { CheckIdentity({}, {}); }

TEST(CompactorTest, MergedWriterRejectsInvalidLists) {
  ferret_testing::TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto source,
      SegmentWriter::Create(*Env::Default(), dir.Join("source.fseg"),
                            {.chunk_bytes = 4}));
  ASSERT_THAT(source->Add(AsDocument({"a.cc", "abcd\nefgh\nijkl\nmnop\n"})),
              absl_testing::IsOk());
  ASSERT_THAT(source->Finish().status(), absl_testing::IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto input, Segment::Open(*Env::Default(), dir.Join("source.fseg")));
  const SegmentDoc doc = input->doc(0);
  ASSERT_GE(doc.num_chunks, 3u);
  std::vector<SegmentChunk> chunks;
  chunks.reserve(doc.num_chunks);
  for (ChunkId id = 0; id < doc.num_chunks; ++id) {
    chunks.push_back(input->chunk(id));
  }
  struct Case {
    const char* name;
    Trigram first_trigram;
    std::vector<ChunkId> first_ids;
    Trigram second_trigram = 0;
    // NOLINTNEXTLINE(readability-redundant-member-init): Required by build.
    std::vector<ChunkId> second_ids = {};
  };
  const std::vector<Case> cases = {
      {"empty", 1, {}},
      {"duplicateIds", 1, {0, 0}},
      {"decreasingIds", 1, {1, 0}},
      {"outOfRange", 1, {doc.num_chunks}},
      {"invalidTrigram", 1u << 24, {0}},
      {"duplicateTrigram", 1, {0}, 1, {1}},
      {"decreasingTrigram", 2, {0}, 1, {1}},
  };
  for (const Case& test : cases) {
    SCOPED_TRACE(test.name);
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto writer,
        SegmentWriter::Create(*Env::Default(), dir.Join("invalid.fseg"),
                              {.chunk_bytes = 4}));
    ASSERT_THAT(writer->AddChunked(doc, chunks), absl_testing::IsOk());
    int calls = 0;
    EXPECT_THAT(writer->FinishMerged(
                    [&](Trigram& trigram,
                        std::vector<ChunkId>& ids) -> absl::StatusOr<bool> {
                      if (calls++ == 0) {
                        trigram = test.first_trigram;
                        ids = test.first_ids;
                      } else {
                        trigram = test.second_trigram;
                        ids = test.second_ids;
                      }
                      return true;
                    }),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_LE(calls, 2);
  }
}

TEST(CompactorTest, MergedWriterPropagatesListErrors) {
  ferret_testing::TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto writer,
      SegmentWriter::Create(*Env::Default(), dir.Join("error.fseg")));
  EXPECT_THAT(writer->FinishMerged(
                  [](Trigram&, std::vector<ChunkId>&) -> absl::StatusOr<bool> {
                    return absl::CancelledError("stopped");
                  }),
              absl_testing::StatusIs(absl::StatusCode::kCancelled));
}

TEST(CompactorTest, RejectsDifferentChunkSizes) {
  ferret_testing::TempDir dir;
  std::vector<std::unique_ptr<Segment>> opened;
  std::vector<const Segment*> inputs;
  for (int size : {16, 32}) {
    const std::string path = dir.Join(absl::StrCat(size, ".fseg"));
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto writer,
        SegmentWriter::Create(*Env::Default(), path, {.chunk_bytes = size}));
    ASSERT_THAT(writer->Finish().status(), absl_testing::IsOk());
    FERRET_ASSERT_OK_AND_ASSIGN(auto segment,
                                Segment::Open(*Env::Default(), path));
    inputs.push_back(segment.get());
    opened.push_back(std::move(segment));
  }
  EXPECT_THAT(
      MergeSegments(*Env::Default(), inputs, [](size_t, DocId) { return true; },
                    dir.Join("out.fseg"), {.chunk_bytes = 16}),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CompactorTest, LegacySegmentGainsSymbolsWithoutLosingContent) {
  ferret_testing::TempDir dir;
  Env& env = *Env::Default();
  const CorpusDoc file{"main.cc", "int Legacy() { return 1; }\n"};
  auto writer = SegmentWriter::Create(env, dir.Join("legacy.fseg"),
                                      {.write_symbols = false});
  ASSERT_TRUE(writer.ok());
  ASSERT_TRUE((*writer)->Add(AsDocument(file)).ok());
  ASSERT_TRUE((*writer)->Finish().ok());
  auto legacy = Segment::Open(env, dir.Join("legacy.fseg"));
  ASSERT_TRUE(legacy.ok());
  ASSERT_FALSE((*legacy)->has_symbols());
  const Segment* input = legacy->get();
  auto merged = MergeSegments(
      env, {input}, [](size_t, DocId) { return true; },
      dir.Join("merged.fseg"));
  ASSERT_TRUE(merged.ok()) << merged.status();
  auto result = Segment::Open(env, dir.Join("merged.fseg"));
  ASSERT_TRUE(result.ok());
  EXPECT_EQ((*result)->doc(0).content, file.content);
  EXPECT_TRUE((*result)->has_symbols());
  int definitions = 0;
  (*result)->FindSymbols("Legacy", false, [&](SegmentSymbol symbol) {
    if (symbol.is_definition) ++definitions;
  });
  EXPECT_EQ(definitions, 1);
}

// Compaction re-lexes the chunks it copies with their stored entry states, so
// the merged section equals a fresh build's, byte for byte, even when the
// inputs are legacy segments without one and when a block comment crosses the
// chunk boundaries of a copied document.
TEST(CompactorTest, CodeTokensMatchAFreshBuildAndUpgradeLegacyInputs) {
  ferret_testing::TempDir dir;
  Env& env = *Env::Default();
  std::vector<CorpusDoc> all;
  for (int i = 0; i < 30; ++i) {
    std::string content = absl::StrCat("int id", i, " = 1;\n/* comment flow\n");
    for (int line = 0; line < 8 + i % 5; ++line) {
      absl::StrAppend(&content, "  still comment word", line, " flow\n");
    }
    absl::StrAppend(&content, "*/\nint flow", i % 4, " = 2; // flow\n");
    all.push_back({absl::StrCat("d", i % 3, "/f", i, ".cc"), content});
  }
  Sort(all);
  SegmentWriterOptions options;
  options.chunk_bytes = 48;
  SegmentWriterOptions legacy_options = options;
  legacy_options.write_code_tokens = false;
  std::vector<std::unique_ptr<Segment>> opened;
  std::vector<const Segment*> inputs;
  for (int part = 0; part < 3; ++part) {
    const std::string path = dir.Join(absl::StrCat("in", part, ".fseg"));
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto writer,
        SegmentWriter::Create(env, path, part == 1 ? legacy_options : options));
    for (size_t i = part; i < all.size(); i += 3) {
      ASSERT_THAT(writer->Add(AsDocument(all[i])), absl_testing::IsOk());
    }
    ASSERT_THAT(writer->Finish().status(), absl_testing::IsOk());
    FERRET_ASSERT_OK_AND_ASSIGN(auto segment, Segment::Open(env, path));
    EXPECT_EQ(segment->has_code_tokens(), part != 1);
    inputs.push_back(segment.get());
    opened.push_back(std::move(segment));
  }
  const std::string merged_path = dir.Join("merged.fseg");
  ASSERT_THAT(
      MergeSegments(
          env, inputs, [](size_t, DocId) { return true; }, merged_path, options)
          .status(),
      absl_testing::IsOk());
  const std::string direct_path = dir.Join("direct.fseg");
  FERRET_ASSERT_OK_AND_ASSIGN(auto writer,
                              SegmentWriter::Create(env, direct_path, options));
  for (const CorpusDoc& doc : all) {
    ASSERT_THAT(writer->Add(AsDocument(doc)), absl_testing::IsOk());
  }
  ASSERT_THAT(writer->Finish().status(), absl_testing::IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(auto merged, Segment::Open(env, merged_path));
  FERRET_ASSERT_OK_AND_ASSIGN(auto direct, Segment::Open(env, direct_path));
  ASSERT_TRUE(merged->has_code_tokens());
  EXPECT_THAT(merged->VerifyChecksums(), absl_testing::IsOk());
  EXPECT_EQ(merged->code_tokens_bytes(), direct->code_tokens_bytes());
  std::vector<std::pair<std::string, std::vector<DocId>>> got;
  std::vector<std::pair<std::string, std::vector<DocId>>> want;
  merged->ForEachCodeToken(
      [&](absl::string_view token, absl::Span<const DocId> docs) {
        got.emplace_back(token, std::vector<DocId>(docs.begin(), docs.end()));
      });
  direct->ForEachCodeToken(
      [&](absl::string_view token, absl::Span<const DocId> docs) {
        want.emplace_back(token, std::vector<DocId>(docs.begin(), docs.end()));
      });
  EXPECT_EQ(got, want);
  EXPECT_GT(got.size(), 30u);
  // The comment words never appear: they are inside block comments that
  // start in an earlier chunk.
  for (const auto& [token, docs] : got) {
    EXPECT_FALSE(token.starts_with("word")) << token;
    EXPECT_NE(token, "comment");
  }
  FERRET_ASSERT_OK_AND_ASSIGN(std::string a, env.ReadFile(merged_path));
  FERRET_ASSERT_OK_AND_ASSIGN(std::string b, env.ReadFile(direct_path));
  EXPECT_EQ(a, b);
}

}  // namespace
}  // namespace ferret

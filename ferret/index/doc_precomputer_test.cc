// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Precomputed documents must give the segment SegmentWriter::Add() writes.

#include "ferret/index/doc_precomputer.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/git/blob_hash.h"
#include "ferret/index/internal/code_tokens.h"
#include "ferret/index/internal/posting_accumulator.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/segment_writer.h"
#include "ferret/postings/ids.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/testing/test_seed.h"
#include "ferret/text/language.h"
#include "ferret/text/trigram.h"
#include "ferret/text/trigram_extractor.h"

namespace ferret {
namespace {

using ::absl_testing::StatusIs;

struct Doc {
  std::string path;
  std::string content;
  Language language;
  RankKey key;
};

std::vector<Doc> MakeDocs(uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<Doc> docs;
  for (int i = 0; i < 60; ++i) {
    Doc doc;
    doc.path = absl::StrCat("d", i % 7, "/f", i, i % 2 ? ".cc" : ".py");
    doc.language = i % 2 ? Language::kCpp : Language::kPython;
    const int lines = static_cast<int>(rng() % (i % 5 == 0 ? 400 : 20));
    for (int l = 0; l < lines; ++l) {
      absl::StrAppend(&doc.content, "int Fn", rng() % 90, "(Arg", rng() % 30,
                      ") { return Val", rng() % 50, "; }  // Note ", l, "\n");
    }
    if (i % 11 == 0) doc.content.clear();  // zero chunks
    doc.key = ComputeRankKey(doc.path, doc.content.size(), false,
                             HashBlob(doc.content));
    docs.push_back(std::move(doc));
  }
  std::sort(docs.begin(), docs.end(),
            [](const Doc& a, const Doc& b) { return a.key < b.key; });
  return docs;
}

class DocPrecomputerTest : public ::testing::Test {
 protected:
  std::string Bytes(const std::string& path) {
    return *Env::Default()->ReadFile(path);
  }
  ferret_testing::TempDir dir_;
};

TEST_F(DocPrecomputerTest, MatchesAdd) {
  const std::vector<Doc> docs = MakeDocs(ferret_testing::TestSeed(4));
  SegmentWriterOptions options;
  options.chunk_bytes = 400;
  options.write_symbols = false;

  FERRET_ASSERT_OK_AND_ASSIGN(
      auto reference,
      SegmentWriter::Create(*Env::Default(), dir_.Join("ref.fseg"), options));
  for (const Doc& doc : docs) {
    ABSL_ASSERT_OK(reference->Add({.hint_path = doc.path,
                                   .content = doc.content,
                                   .rank_key = doc.key,
                                   .language = doc.language}));
  }
  ABSL_ASSERT_OK(reference->Finish().status());

  // Three "workers" take the documents in a scrambled order; provisional
  // chunk and document ids follow processing order.
  std::vector<size_t> processing(docs.size());
  std::iota(processing.begin(), processing.end(), 0);
  std::mt19937_64 rng(1);
  std::shuffle(processing.begin(), processing.end(), rng);
  TrigramExtractor extractors[3];
  std::atomic<uint32_t> next_chunk{0};
  std::vector<std::unique_ptr<DocumentPrecomputer>> workers;
  workers.reserve(3);
  for (int w = 0; w < 3; ++w) {
    workers.push_back(std::make_unique<DocumentPrecomputer>(
        options, extractors[w], next_chunk));
  }
  std::vector<PrecomputedDocument> precomputed(docs.size());
  for (size_t k = 0; k < processing.size(); ++k) {
    const size_t i = processing[k];
    FERRET_ASSERT_OK_AND_ASSIGN(
        precomputed[i],
        workers[k % 3]->Precompute(docs[i].content, docs[i].language,
                                   static_cast<uint32_t>(i)));
  }
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto writer,
      SegmentWriter::Create(*Env::Default(), dir_.Join("pre.fseg"), options));
  std::vector<ChunkId> chunk_remap(next_chunk.load());
  std::vector<DocId> doc_remap(docs.size());
  uint32_t final_chunk = 0;
  for (size_t d = 0; d < docs.size(); ++d) {
    const PrecomputedDocument& pre = precomputed[d];
    for (size_t c = 0; c < pre.chunks.size(); ++c) {
      chunk_remap[pre.first_chunk + c] = final_chunk++;
    }
    doc_remap[d] = static_cast<DocId>(d);
    ABSL_ASSERT_OK(writer->AddPrecomputed({.hint_path = docs[d].path,
                                           .content = docs[d].content,
                                           .rank_key = docs[d].key,
                                           .language = docs[d].language},
                                          pre.num_lines, pre.chunks));
  }
  std::vector<segment_internal::PostingAccumulator::Run> runs;
  std::vector<segment_internal::CodeTokenMap> maps;
  for (auto& worker : workers) {
    for (auto& run : worker->TakeRuns()) runs.push_back(std::move(run));
    maps.push_back(worker->TakeCodeTokens());
  }
  writer->SetPostings(std::move(runs), std::move(chunk_remap));
  writer->SetCodeTokenMaps(std::move(maps), std::move(doc_remap));
  ABSL_ASSERT_OK(writer->Finish().status());
  EXPECT_TRUE(Bytes(dir_.Join("ref.fseg")) == Bytes(dir_.Join("pre.fseg")));
}

TEST_F(DocPrecomputerTest, ModesDoNotMix) {
  SegmentWriterOptions options;
  TrigramExtractor extractor;
  std::atomic<uint32_t> next_chunk{0};
  DocumentPrecomputer precomputer(options, extractor, next_chunk);
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto pre, precomputer.Precompute("int a;\n", Language::kCpp, 0));
  const RankKey low = {
      .tier = RankTier::kSource, .prior = 9, .oid = HashBlob("int a;\n")};
  const RankKey high = {
      .tier = RankTier::kTest, .prior = 9, .oid = HashBlob("int a;\n")};
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto writer,
      SegmentWriter::Create(*Env::Default(), dir_.Join("m.fseg"), options));
  ABSL_ASSERT_OK(writer->AddPrecomputed({.hint_path = "b.cc",
                                         .content = "int a;\n",
                                         .rank_key = high,
                                         .language = Language::kCpp},
                                        pre.num_lines, pre.chunks));
  EXPECT_THAT(writer->AddPrecomputed({.hint_path = "a.cc",
                                      .content = "int a;\n",
                                      .rank_key = low,
                                      .language = Language::kCpp},
                                     pre.num_lines, pre.chunks),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(writer->Add({.hint_path = "c.cc",
                           .content = "x",
                           .rank_key = high,
                           .language = Language::kCpp}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(writer->FinishMerged(
                  [](Trigram&, std::vector<ChunkId>&) { return false; }),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto other,
      SegmentWriter::Create(*Env::Default(), dir_.Join("n.fseg"), options));
  ABSL_ASSERT_OK(other->Add({.hint_path = "a.cc",
                             .content = "int a;\n",
                             .rank_key = low,
                             .language = Language::kCpp}));
  EXPECT_THAT(other->AddPrecomputed({.hint_path = "b.cc",
                                     .content = "int a;\n",
                                     .rank_key = high,
                                     .language = Language::kCpp},
                                    pre.num_lines, pre.chunks),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace ferret

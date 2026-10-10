// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Checks stable records, publication, exact query evaluation, rank-ordered
// flushing, capacity, and one-writer/many-reader operation.

#include "ferret/index/memtable.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/git/blob_hash.h"
#include "ferret/git/oid.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/segment.h"
#include "ferret/index/segment_writer.h"
#include "ferret/postings/ids.h"
#include "ferret/postings/list_ops.h"
#include "ferret/query/trigram_query.h"
#include "ferret/regex/compile.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/testing/test_seed.h"
#include "ferret/text/language.h"
#include "ferret/text/trigram.h"
#include "ferret/text/trigram_extractor.h"

namespace ferret {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

SegmentDocument Document(absl::string_view path, absl::string_view content) {
  return {.hint_path = path,
          .content = content,
          .rank_key =
              ComputeRankKey(path, content.size(), false, HashBlob(content))};
}

void CompareRecords(const SegmentDoc& a, const SegmentDoc& b) {
  EXPECT_EQ(a.hint_path, b.hint_path);
  EXPECT_EQ(a.content, b.content);
  EXPECT_EQ(a.oid, b.oid);
  EXPECT_EQ(a.tier, b.tier);
  EXPECT_EQ(a.prior, b.prior);
  EXPECT_EQ(a.language, b.language);
  EXPECT_EQ(a.long_lines, b.long_lines);
  EXPECT_EQ(a.num_chunks, b.num_chunks);
  EXPECT_EQ(a.num_lines, b.num_lines);
}

void CompareChunks(const SegmentChunk& a, const SegmentChunk& b) {
  EXPECT_EQ(a.content, b.content);
  EXPECT_EQ(a.offset_in_doc, b.offset_in_doc);
  EXPECT_EQ(a.first_line, b.first_line);
  EXPECT_EQ(a.entry_state, b.entry_state);
}

TEST(MemtableTest, RecordsAndChunkingMatchSegmentWriter) {
  MemtableOptions options;
  options.segment.chunk_bytes = 6;
  Memtable mem(options);
  std::vector<std::string> contents = {"", "/* hello\nworld */\nabc\nxyz\n",
                                       "first long line without breaks\nabc"};
  std::vector<std::string> paths = {"a.cc", "b.cc", "c.cc"};
  ferret_testing::TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SegmentWriter> writer,
      SegmentWriter::Create(*Env::Default(), dir.Join("reference.fseg"),
                            options.segment));
  // SegmentWriter requires rank order; compare each document by Oid.
  std::vector<SegmentDocument> documents;
  for (size_t i = 0; i < contents.size(); ++i) {
    documents.push_back(Document(paths[i], contents[i]));
    documents.back().language = Language::kCpp;
    ASSERT_THAT(mem.Add(documents.back()), IsOk());
  }
  std::sort(documents.begin(), documents.end(),
            [](const SegmentDocument& a, const SegmentDocument& b) {
              return a.rank_key < b.rank_key;
            });
  for (const SegmentDocument& document : documents) {
    ASSERT_THAT(writer->Add(document), IsOk());
  }
  ASSERT_THAT(writer->Finish(), IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Segment> segment,
      Segment::Open(*Env::Default(), dir.Join("reference.fseg")));
  for (DocId id = 0; id < mem.published_docs(); ++id) {
    const SegmentDoc a = mem.doc(id);
    bool found = false;
    for (DocId j = 0; j < segment->num_docs(); ++j) {
      const SegmentDoc b = segment->doc(j);
      if (a.oid != b.oid) continue;
      found = true;
      CompareRecords(a, b);
      for (uint32_t k = 0; k < a.num_chunks; ++k) {
        CompareChunks(mem.chunk(a.first_chunk + k),
                      segment->chunk(b.first_chunk + k));
      }
    }
    EXPECT_TRUE(found);
  }
}

TEST(MemtableTest, EvaluatorMatchesTrigramSetsAndLiteralMatches) {
  const uint64_t seed = ferret_testing::TestSeed(908177);
  SCOPED_TRACE(seed);
  std::mt19937_64 random(seed);
  MemtableOptions options;
  options.segment.chunk_bytes = 13;
  Memtable mem(options);
  for (int i = 0; i < 80; ++i) {
    std::string content;
    for (int j = 0; j < 90; ++j) {
      content.push_back(j % 17 == 16 ? '\n' : "abcdXYZ"[random() % 7]);
    }
    const std::string path = absl::StrCat(i, ".cc");
    ASSERT_THAT(mem.Add(Document(path, content)), IsOk());
  }
  const uint32_t count = mem.num_chunks(mem.published_docs());
  TrigramExtractor extractor;
  for (int trial = 0; trial < 80; ++trial) {
    std::string literal;
    for (int j = 0; j < 3 + trial % 7; ++j) {
      literal.push_back("abcdXYZ"[random() % 7]);
    }
    FERRET_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<CompiledPattern> compiled,
        CompilePattern({.pattern = literal, .literal = true}));
    auto evaluator = mem.NewEvaluator(compiled->query(), mem.published_docs());
    ChunkIdBuffer candidates;
    evaluator->Evaluate({0, count - 1}, candidates);
    for (ChunkId id = 0; id < count; ++id) {
      const SegmentChunk chunk = mem.chunk(id);
      std::vector<Trigram> trigrams;
      extractor.Extract(chunk.content, trigrams);
      const bool expected = compiled->query().Evaluate([&](Trigram trigram) {
        return std::find(trigrams.begin(), trigrams.end(), trigram) !=
               trigrams.end();
      });
      const bool actual = std::binary_search(candidates.ids().begin(),
                                             candidates.ids().end(), id);
      EXPECT_EQ(actual, expected) << literal << " chunk " << id;
      if (absl::StrContains(chunk.content, literal)) {
        EXPECT_TRUE(actual) << literal << " chunk " << id;
      }
    }
  }
}

TEST(MemtableTest, WatermarkHidesLaterDocuments) {
  Memtable mem;
  ASSERT_THAT(mem.Add(Document("a.cc", "abc abc")), IsOk());
  const uint32_t watermark = mem.published_docs();
  const uint32_t chunks = mem.num_chunks(watermark);
  auto evaluator = mem.NewEvaluator(TrigramQuery::All(), watermark);
  const Oid later = HashBlob("later");
  const TrigramQuery later_query =
      TrigramQuery::AllOf(std::vector<Trigram>{0x6c6174});
  auto before = mem.NewEvaluator(later_query, watermark);
  ASSERT_THAT(mem.Add(Document("b.cc", "later")), IsOk());
  EXPECT_TRUE(before->matches_nothing());
  EXPECT_TRUE(mem.NewEvaluator(later_query, watermark)->matches_nothing());
  EXPECT_FALSE(
      mem.NewEvaluator(later_query, mem.published_docs())->matches_nothing());
  EXPECT_EQ(mem.num_chunks(watermark), chunks);
  EXPECT_EQ(mem.doc(0).content, "abc abc");
  EXPECT_EQ(mem.FindDoc(later, watermark), std::nullopt);
  EXPECT_EQ(mem.FindDoc(later, mem.published_docs()), std::optional<DocId>(1));
  ChunkIdBuffer out;
  evaluator->Evaluate({0, chunks - 1}, out);
  ASSERT_EQ(out.size(), chunks);
}

TEST(MemtableTest, FindDocReturnsFirstDuplicateBelowWatermark) {
  Memtable mem;
  const Oid oid = HashBlob("same");
  EXPECT_EQ(mem.FindDoc(oid, 0), std::nullopt);
  ASSERT_THAT(mem.Add(Document("first.cc", "same")), IsOk());
  const uint32_t first_watermark = mem.published_docs();
  ASSERT_THAT(mem.Add(Document("other.cc", "other")), IsOk());
  ASSERT_THAT(mem.Add(Document("duplicate.cc", "same")), IsOk());
  EXPECT_EQ(mem.FindDoc(oid, 0), std::nullopt);
  EXPECT_EQ(mem.FindDoc(oid, first_watermark), std::optional<DocId>(0));
  EXPECT_EQ(mem.FindDoc(oid, mem.published_docs()), std::optional<DocId>(0));
  EXPECT_EQ(mem.FindDoc(HashBlob("other"), first_watermark), std::nullopt);
  EXPECT_EQ(mem.FindDoc(HashBlob("other"), mem.published_docs()),
            std::optional<DocId>(1));
}

TEST(MemtableTest, ConfiguredSoftFlushThresholds) {
  MemtableOptions options;
  options.flush_docs = 2;
  Memtable by_docs(options);
  ASSERT_THAT(by_docs.Add(Document("first.cc", "")), IsOk());
  EXPECT_FALSE(by_docs.ShouldFlush());
  ASSERT_THAT(by_docs.Add(Document("second.cc", "")), IsOk());
  EXPECT_TRUE(by_docs.ShouldFlush());

  options.flush_docs = 3;
  options.flush_bytes = 1024;
  Memtable by_bytes(options);
  EXPECT_FALSE(by_bytes.ShouldFlush());
  ASSERT_THAT(by_bytes.Add(Document("one.cc", "x")), IsOk());
  EXPECT_TRUE(by_bytes.ShouldFlush());
}

TEST(MemtableTest, FlushOrdersDropsAndDeduplicates) {
  MemtableOptions options;
  options.segment.chunk_bytes = 5;
  Memtable mem(options);
  ASSERT_THAT(mem.Add(Document("z.cc", "/* abc\ndef */\n")), IsOk());
  ASSERT_THAT(mem.Add(Document("a.cc", "aaa\nbbb\n")), IsOk());
  const uint32_t watermark = mem.published_docs();
  ASSERT_THAT(mem.Add(Document("a.cc", "aaa\nbbb\n")), IsOk());
  ASSERT_THAT(mem.Add(Document("drop.cc", "drop")), IsOk());
  ferret_testing::TempDir dir;
  FERRET_ASSERT_OK_AND_ASSIGN(
      SegmentStats stats,
      mem.WriteSegment(*Env::Default(), dir.Join("flush.fseg"),
                       mem.published_docs(), [](DocId id) { return id != 3; }));
  EXPECT_EQ(stats.num_docs, 2u);
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Segment> segment,
      Segment::Open(*Env::Default(), dir.Join("flush.fseg")));
  EXPECT_EQ(segment->num_docs(), 2u);
  const SegmentDoc first = segment->doc(0);
  const SegmentDoc second = segment->doc(1);
  EXPECT_FALSE(ComputeRankKey(first.hint_path, first.content.size(),
                              first.long_lines, first.oid) >
               ComputeRankKey(second.hint_path, second.content.size(),
                              second.long_lines, second.oid));
  for (DocId id = 0; id < 2; ++id) {
    const SegmentDoc b = segment->doc(id);
    const SegmentDoc a = mem.doc(b.oid == mem.doc(0).oid ? 0 : 1);
    CompareRecords(a, b);
    for (uint32_t k = 0; k < a.num_chunks; ++k) {
      CompareChunks(mem.chunk(a.first_chunk + k),
                    segment->chunk(b.first_chunk + k));
    }
  }
  EXPECT_EQ(mem.num_chunks(watermark),
            mem.doc(1).first_chunk + mem.doc(1).num_chunks);
}

TEST(MemtableTest, CapacityFailuresLeaveEverythingUnchanged) {
  MemtableOptions options;
  options.max_docs = 2;
  options.max_chunks = 1;
  options.max_content_bytes = 4;
  options.flush_docs = 1;
  options.flush_bytes = 3;
  Memtable mem(options);
  ASSERT_THAT(mem.Add(Document("a", "abc")), IsOk());
  const size_t bytes = mem.memory_bytes();
  EXPECT_THAT(mem.Add(Document("b", "toolong")),
              StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_THAT(mem.Add(Document("b", "x")),
              StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_EQ(mem.published_docs(), 1u);
  EXPECT_EQ(mem.num_chunks(1), 1u);
  EXPECT_EQ(mem.memory_bytes(), bytes);
  EXPECT_EQ(mem.doc(0).content, "abc");
}

TEST(MemtableTest, FlushThresholdAndStableOversizeContent) {
  Memtable mem;
  EXPECT_FALSE(mem.ShouldFlush());
  EXPECT_FALSE(mem.oldest_document_time().has_value());
  const absl::Time before = absl::Now();
  std::string large(size_t{1024} * 1024 + 1, 'x');
  ASSERT_THAT(mem.Add(Document("large.cc", large)), IsOk());
  ASSERT_THAT(mem.Add(Document("small.cc", "small")), IsOk());
  EXPECT_EQ(mem.doc(0).content, large);
  EXPECT_EQ(mem.doc(1).content, "small");
  ASSERT_TRUE(mem.oldest_document_time().has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): Asserted above.
  EXPECT_GE(*mem.oldest_document_time(), before);
  for (int i = 0; i < 7; ++i) {
    const std::string path = absl::StrCat(i, ".cc");
    ASSERT_THAT(mem.Add(Document(path, std::string(size_t{1024} * 1024, 'a'))),
                IsOk());
  }
  EXPECT_TRUE(mem.ShouldFlush());
  EXPECT_GT(mem.memory_bytes(), 8u * 1024 * 1024);
}

TEST(MemtableTest, DocumentCountTriggersSoftFlush) {
  Memtable mem;
  const SegmentDocument empty = Document("empty.cc", "");
  for (uint32_t i = 0; i < 16 * 1024 - 1; ++i) {
    ASSERT_THAT(mem.Add(empty), IsOk());
  }
  EXPECT_FALSE(mem.ShouldFlush());
  EXPECT_EQ(mem.num_chunks(mem.published_docs()), 0u);
  ASSERT_THAT(mem.Add(empty), IsOk());
  EXPECT_TRUE(mem.ShouldFlush());
}

TEST(MemtableTest, DocumentCapacityIsIndependentOfBytes) {
  MemtableOptions options;
  options.max_docs = 2;
  options.flush_docs = 1;
  Memtable mem(options);
  ASSERT_THAT(mem.Add(Document("first", "")), IsOk());
  ASSERT_THAT(mem.Add(Document("another", "")), IsOk());
  const size_t memory = mem.memory_bytes();
  EXPECT_THAT(mem.Add(Document("second", "")),
              StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_EQ(mem.memory_bytes(), memory);
  EXPECT_EQ(mem.published_docs(), 2u);
}

TEST(MemtableTest, ConcurrentReadersUseCapturedWatermarks) {
  Memtable mem;
  std::atomic<bool> done = false;
  ferret_testing::TempDir dir;
  std::thread writer([&] {
    for (int i = 0; i < 300; ++i) {
      const std::string path = absl::StrCat(i, ".cc");
      const absl::StatusOr<DocId> id = mem.Add(Document(path, "abc\nxyz\n"));
      ASSERT_THAT(id, IsOk());
      ASSERT_EQ(*id, static_cast<DocId>(i));
    }
    done.store(true, std::memory_order_release);
  });
  std::vector<std::thread> readers;
  readers.reserve(4);
  for (int i = 0; i < 4; ++i) {
    readers.emplace_back([&, i] {
      bool flushed = false;
      do {
        const uint32_t w = mem.published_docs();
        const uint32_t chunks = mem.num_chunks(w);
        if (w == 0) continue;
        if (i == 0 && !flushed && w >= 20) {
          const absl::StatusOr<SegmentStats> stats =
              mem.WriteSegment(*Env::Default(), dir.Join("concurrent.fseg"), w,
                               [](DocId) { return true; });
          ASSERT_THAT(stats, IsOk());
          EXPECT_LE(stats->num_docs, w);
          FERRET_ASSERT_OK_AND_ASSIGN(
              std::unique_ptr<Segment> segment,
              Segment::Open(*Env::Default(), dir.Join("concurrent.fseg")));
          EXPECT_EQ(segment->num_docs(), stats->num_docs);
          flushed = true;
        }
        auto query = mem.NewEvaluator(
            TrigramQuery::AllOf(std::vector<Trigram>{0x616263}), w);
        ChunkIdBuffer out;
        query->Evaluate({0, chunks - 1}, out);
        ASSERT_EQ(out.size(), chunks);
        for (DocId id = 0; id < w; ++id) {
          const SegmentDoc doc = mem.doc(id);
          EXPECT_EQ(mem.blob_of_doc(id), mem.blob_of_doc(0));
          EXPECT_EQ(doc.content, "abc\nxyz\n");
          EXPECT_EQ(doc.num_chunks, 1u);
          EXPECT_EQ(mem.FindDoc(doc.oid, w), std::optional<DocId>(0));
          const SegmentChunk chunk = mem.chunk(doc.first_chunk);
          EXPECT_EQ(chunk.doc, id);
          EXPECT_EQ(chunk.content, doc.content);
        }
      } while (!done.load(std::memory_order_acquire));
    });
  }
  writer.join();
  for (std::thread& reader : readers) reader.join();
}

}  // namespace
}  // namespace ferret

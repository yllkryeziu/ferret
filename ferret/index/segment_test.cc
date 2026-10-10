// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/segment.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/git/blob_hash.h"
#include "ferret/index/internal/segment_format.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/segment_writer.h"
#include "ferret/postings/ids.h"
#include "ferret/postings/list_ops.h"
#include "ferret/postings/posting_list.h"
#include "ferret/query/trigram_query.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/testing/test_seed.h"
#include "ferret/text/chunker.h"
#include "ferret/text/language.h"
#include "ferret/text/lexer.h"
#include "ferret/text/literal_finder.h"
#include "ferret/text/trigram.h"
#include "ferret/text/trigram_extractor.h"

namespace ferret {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::AnyOf;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;

struct TestDoc {
  std::string path;
  std::string content;
  Language language = Language::kUnknown;
  bool long_lines = false;
};

SegmentDocument ToSegmentDocument(const TestDoc& doc) {
  return SegmentDocument{
      .hint_path = doc.path,
      .content = doc.content,
      .rank_key = ComputeRankKey(doc.path, doc.content.size(), doc.long_lines,
                                 HashBlob(doc.content)),
      .language = doc.language,
      .long_lines = doc.long_lines,
  };
}

// Sorts `docs` into rank order.
void SortByRank(std::vector<TestDoc>& docs) {
  std::stable_sort(
      docs.begin(), docs.end(), [](const TestDoc& a, const TestDoc& b) {
        return ToSegmentDocument(a).rank_key < ToSegmentDocument(b).rank_key;
      });
}

absl::StatusOr<SegmentStats> WriteSegment(
    const std::string& path, const std::vector<TestDoc>& docs,
    const SegmentWriterOptions& options = {}) {
  absl::StatusOr<std::unique_ptr<SegmentWriter>> writer =
      SegmentWriter::Create(*Env::Default(), path, options);
  if (!writer.ok()) return writer.status();
  for (const TestDoc& doc : docs) {
    if (absl::Status s = (*writer)->Add(ToSegmentDocument(doc)); !s.ok()) {
      return s;
    }
  }
  return (*writer)->Finish();
}

std::string ReadAll(const std::string& path) {
  absl::StatusOr<std::string> bytes = Env::Default()->ReadFile(path);
  EXPECT_THAT(bytes.status(), IsOk());
  return bytes.ok() ? *bytes : std::string();
}

// The expected posting lists: every chunk's trigrams, computed the way the
// writer is specified to, independently of the segment.
std::map<Trigram, std::vector<ChunkId>> ReferencePostings(
    const std::vector<TestDoc>& docs, int64_t chunk_bytes) {
  std::map<Trigram, std::vector<ChunkId>> lists;
  TrigramExtractor extractor;
  ChunkId next = 0;
  for (const TestDoc& doc : docs) {
    for (const ChunkSpan& chunk : SplitIntoChunks(doc.content, chunk_bytes)) {
      std::vector<Trigram> trigrams;
      extractor.Extract(absl::string_view(doc.content)
                            .substr(static_cast<size_t>(chunk.offset),
                                    static_cast<size_t>(chunk.size)),
                        trigrams);
      extractor.ExtractShortLines(absl::string_view(doc.content)
                                      .substr(static_cast<size_t>(chunk.offset),
                                              static_cast<size_t>(chunk.size)),
                                  trigrams);
      for (const Trigram t : trigrams) lists[t].push_back(next);
      ++next;
    }
  }
  return lists;
}

std::vector<ChunkId> Decode(const PostingListView& list) {
  std::vector<ChunkId> ids(list.size());
  if (!ids.empty()) list.DecodeAll(ids.data());
  return ids;
}

// Random lines over a small alphabet, mixed with a line most chunks share
// (long, blocked posting lists) and rare tokens (inlined lists).
std::string RandomContent(std::mt19937_64& rng, size_t size) {
  static constexpr char kAlphabet[] = "abcdeABC  _(){}\n";
  std::string out;
  while (out.size() < size) {
    switch (rng() % 8) {
      case 0:
        out.append("hello world\n");
        break;
      case 1:
        absl::StrAppend(&out, "id", rng() % 1000000, "\n");
        break;
      default:
        out.push_back(kAlphabet[rng() % (sizeof(kAlphabet) - 1)]);
        break;
    }
  }
  out.resize(size);
  return out;
}

class SegmentTest : public ::testing::Test {
 protected:
  std::string Path(absl::string_view name) const { return dir_.Join(name); }

  ferret_testing::TempDir dir_;
};

TEST_F(SegmentTest, ReportsCurrentFormatVersion) {
  EXPECT_EQ(Segment::FormatVersion(), 2u);
}

TEST_F(SegmentTest, EmptySegment) {
  FERRET_ASSERT_OK_AND_ASSIGN(const SegmentStats stats,
                              WriteSegment(Path("empty.fseg"), {}));
  EXPECT_EQ(stats.num_docs, 0u);
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto segment, Segment::Open(*Env::Default(), Path("empty.fseg")));
  EXPECT_EQ(segment->num_docs(), 0u);
  EXPECT_EQ(segment->num_chunks(), 0u);
  EXPECT_EQ(segment->num_trigrams(), 0u);
  FERRET_ASSERT_OK_AND_ASSIGN(
      const PostingListView list,
      segment->FindPostingList(MakeTrigram('a', 'b', 'c')));
  EXPECT_EQ(list.size(), 0u);
  EXPECT_THAT(segment->VerifyChecksums(), IsOk());
}

TEST_F(SegmentTest, DocsAndChunksRoundTrip) {
  std::vector<TestDoc> docs = {
      {"src/main.c", "int main() {\n  /* a\ncomment */ return 0;\n}\n",
       Language::kC, false},
      {"README", "no trailing newline", Language::kUnknown, false},
      {"lib/data.min.js", "x=1;y=2;\n", Language::kJavaScript, true},
  };
  SortByRank(docs);
  const SegmentWriterOptions options = {.chunk_bytes = 16};
  FERRET_ASSERT_OK_AND_ASSIGN(const SegmentStats stats,
                              WriteSegment(Path("s.fseg"), docs, options));
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment,
                              Segment::Open(*Env::Default(), Path("s.fseg")));
  ASSERT_EQ(segment->num_docs(), docs.size());
  EXPECT_EQ(stats.num_docs, docs.size());
  EXPECT_EQ(segment->chunk_bytes(), 16u);
  EXPECT_EQ(segment->num_chunks(), stats.num_chunks);
  EXPECT_EQ(segment->num_trigrams(), stats.num_trigrams);

  ChunkId next_chunk = 0;
  for (DocId d = 0; d < docs.size(); ++d) {
    SCOPED_TRACE(docs[d].path);
    const SegmentDoc doc = segment->doc(d);
    const SegmentDocument expected = ToSegmentDocument(docs[d]);
    EXPECT_EQ(doc.hint_path, docs[d].path);
    EXPECT_EQ(doc.content, docs[d].content);
    EXPECT_EQ(doc.oid, expected.rank_key.oid);
    EXPECT_EQ(doc.tier, expected.rank_key.tier);
    EXPECT_EQ(doc.prior, expected.rank_key.prior);
    EXPECT_EQ(doc.language, docs[d].language);
    EXPECT_EQ(doc.long_lines, docs[d].long_lines);
    EXPECT_EQ(doc.first_chunk, next_chunk);

    const std::vector<ChunkSpan> spans = SplitIntoChunks(docs[d].content, 16);
    ASSERT_EQ(doc.num_chunks, spans.size());
    LexState state = LexState::kCode;
    for (size_t i = 0; i < spans.size(); ++i) {
      const SegmentChunk chunk =
          segment->chunk(next_chunk + static_cast<ChunkId>(i));
      const absl::string_view text =
          absl::string_view(docs[d].content)
              .substr(static_cast<size_t>(spans[i].offset),
                      static_cast<size_t>(spans[i].size));
      EXPECT_EQ(chunk.doc, d);
      EXPECT_EQ(chunk.content, text);
      EXPECT_EQ(chunk.offset_in_doc, static_cast<uint64_t>(spans[i].offset));
      EXPECT_EQ(chunk.first_line, spans[i].first_line);
      EXPECT_EQ(chunk.entry_state, state);
      state = AdvanceLexState(docs[d].language, text, state);
    }
    next_chunk += doc.num_chunks;
  }
  // "comment */" starts a chunk inside the C block comment.
  bool saw_block_comment = false;
  for (ChunkId c = 0; c < segment->num_chunks(); ++c) {
    saw_block_comment |=
        segment->chunk(c).entry_state == LexState::kBlockComment;
  }
  EXPECT_TRUE(saw_block_comment);
}

// The single-field accessors read what doc() and chunk() read, and copies of
// an Oid are found: the later documents of a repeated Oid, by DocId.
TEST_F(SegmentTest, SingleFieldAccessorsAndRepeatedOids) {
  std::vector<TestDoc> docs = {
      {"a/one.c", "int a;\nint b;\nint c;\n"},
      {"b/two.c", "same\ncontent\n"},
      {"c/three.c", "other\n"},
      {"d/four.c", "same\ncontent\n"},
      {"e/five.c", "same\ncontent\n"},
  };
  SortByRank(docs);
  const SegmentWriterOptions options = {.chunk_bytes = 8};
  FERRET_ASSERT_OK_AND_ASSIGN(const SegmentStats stats,
                              WriteSegment(Path("r.fseg"), docs, options));
  ASSERT_EQ(stats.num_docs, docs.size());
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment,
                              Segment::Open(*Env::Default(), Path("r.fseg")));
  std::map<Oid, DocId> first;
  int repeated = 0;
  for (DocId d = 0; d < segment->num_docs(); ++d) {
    const SegmentDoc doc = segment->doc(d);
    EXPECT_EQ(segment->doc_oid(d), doc.oid);
    EXPECT_EQ(segment->doc_chunks(d),
              std::make_pair(doc.first_chunk, doc.num_chunks));
    for (uint32_t i = 0; i < doc.num_chunks; ++i) {
      EXPECT_EQ(segment->doc_of_chunk(doc.first_chunk + i), d);
    }
    const bool seen_before = !first.try_emplace(doc.oid, d).second;
    EXPECT_EQ(segment->is_repeated_oid(d), seen_before) << d;
    repeated += seen_before;
  }
  EXPECT_EQ(repeated, 2);
  EXPECT_TRUE(segment->has_duplicate_oids());

  std::vector<TestDoc> distinct = {{"x", "1\n"}, {"y", "2\n"}};
  SortByRank(distinct);
  ASSERT_THAT(WriteSegment(Path("d.fseg"), distinct).status(), IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(auto plain,
                              Segment::Open(*Env::Default(), Path("d.fseg")));
  EXPECT_FALSE(plain->has_duplicate_oids());
  EXPECT_FALSE(plain->is_repeated_oid(1));
}

TEST_F(SegmentTest, LineCounts) {
  std::vector<TestDoc> docs = {{"a", "x\ny\n"}, {"b", "x\ny"}, {"c", "\n"}};
  SortByRank(docs);
  ASSERT_THAT(WriteSegment(Path("s.fseg"), docs).status(), IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment,
                              Segment::Open(*Env::Default(), Path("s.fseg")));
  for (DocId d = 0; d < docs.size(); ++d) {
    const uint32_t expected = docs[d].path == "c" ? 1 : 2;
    EXPECT_EQ(segment->doc(d).num_lines, expected) << docs[d].path;
  }
}

std::vector<TestDoc> RandomDocs(std::mt19937_64& rng, int count) {
  std::vector<TestDoc> docs;
  docs.reserve(count);
  for (int i = 0; i < count; ++i) {
    docs.push_back({absl::StrCat("dir", i % 3, "/file", i, ".txt"),
                    RandomContent(rng, 1 + rng() % 3000)});
  }
  SortByRank(docs);
  return docs;
}

// Basename search reports exactly the documents whose last path component
// contains the needle: never for a hit in a directory or one that spans two
// neighboring basenames, which the packing places side by side.
TEST_F(SegmentTest, BasenameSearchMatchesReference) {
  std::mt19937_64 rng(ferret_testing::TestSeed(25));
  constexpr absl::string_view kAlphabet = "abAB.";
  std::vector<TestDoc> docs;
  for (int i = 0; i < 200; ++i) {
    std::string path;
    for (int depth = static_cast<int>(rng() % 3); depth > 0; --depth) {
      absl::StrAppend(&path, "ab", rng() % 3, "/");
    }
    for (int n = 1 + static_cast<int>(rng() % 5); n > 0; --n) {
      path.push_back(kAlphabet[rng() % kAlphabet.size()]);
    }
    absl::StrAppend(&path, i % 7);
    docs.push_back({path, absl::StrCat("content ", i, "\n")});
  }
  SortByRank(docs);
  ASSERT_THAT(WriteSegment(Path("names.fseg"), docs).status(), IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto segment, Segment::Open(*Env::Default(), Path("names.fseg")));
  std::vector<std::string> needles = {"",    "a",  "ab",  "aB", "b.",
                                      "ab0", "0a", "3ab", "b/a"};
  for (int i = 0; i < 40; ++i) {
    std::string needle;
    for (int n = 1 + static_cast<int>(rng() % 4); n > 0; --n) {
      needle.push_back(kAlphabet[rng() % kAlphabet.size()]);
    }
    if (rng() % 3 == 0) absl::StrAppend(&needle, rng() % 7);
    needles.push_back(std::move(needle));
  }
  // No path ends in '/', and npos + 1 is 0 for a path without one.
  const auto basename = [](absl::string_view path) {
    return path.substr(path.rfind('/') + 1);
  };
  int64_t basename_length = 0;
  for (const TestDoc& doc : docs) {
    basename_length += std::ssize(basename(doc.path));
  }
  int64_t reported = 0;
  for (const std::string& needle : needles) {
    SCOPED_TRACE(needle);
    for (const LiteralFinder::Mode mode :
         {LiteralFinder::Mode::kExact,
          LiteralFinder::Mode::kAsciiCaseInsensitive}) {
      const bool fold = mode == LiteralFinder::Mode::kAsciiCaseInsensitive;
      std::vector<DocId> expected;
      for (DocId d = 0; d < segment->num_docs(); ++d) {
        std::string name(basename(docs[d].path));
        std::string wanted = needle;
        if (fold) {
          absl::AsciiStrToLower(&name);
          absl::AsciiStrToLower(&wanted);
        }
        if (absl::StrContains(name, wanted)) expected.push_back(d);
      }
      std::vector<DocId> actual;
      segment->ForEachBasenameContaining(needle, mode,
                                         [&](DocId d) { actual.push_back(d); });
      EXPECT_THAT(actual, ElementsAreArray(expected)) << "fold " << fold;
      reported += std::ssize(actual);
    }
  }
  EXPECT_GT(reported, 1000);
  EXPECT_GE(segment->basename_bytes(), basename_length + 4 * std::ssize(docs));

  // The accessors that skip the record reads agree with doc().
  for (DocId d = 0; d < segment->num_docs(); ++d) {
    const SegmentDoc record = segment->doc(d);
    EXPECT_EQ(segment->basename(d), basename(record.hint_path));
    const RankKey key = segment->rank_key(d);
    EXPECT_EQ(key.tier, record.tier);
    EXPECT_EQ(key.prior, record.prior);
    EXPECT_EQ(key.oid, record.oid);
  }

  // Several needles over a range of documents report the union of the
  // single-needle scans in DocId order, and any partition into ranges
  // reports exactly what the whole segment does.
  int64_t union_hits = 0;
  for (int round = 0; round < 60; ++round) {
    std::vector<absl::string_view> chosen;
    for (int n = 1 + static_cast<int>(rng() % 4); n > 0; --n) {
      chosen.push_back(needles[rng() % needles.size()]);
    }
    for (const LiteralFinder::Mode mode :
         {LiteralFinder::Mode::kExact,
          LiteralFinder::Mode::kAsciiCaseInsensitive}) {
      std::vector<DocId> expected;
      for (const absl::string_view needle : chosen) {
        segment->ForEachBasenameContaining(
            needle, mode, [&](DocId d) { expected.push_back(d); });
      }
      std::sort(expected.begin(), expected.end());
      expected.erase(std::unique(expected.begin(), expected.end()),
                     expected.end());
      std::vector<DocId> whole;
      segment->ForEachBasenameContainingAny(
          chosen, mode, 0, segment->num_docs(),
          [&](DocId d) { whole.push_back(d); });
      EXPECT_THAT(whole, ElementsAreArray(expected));
      union_hits += std::ssize(whole);
      std::vector<DocId> pieces;
      const DocId step = 1 + static_cast<DocId>(rng() % 37);
      for (DocId begin = 0; begin < segment->num_docs(); begin += step) {
        segment->ForEachBasenameContainingAny(
            chosen, mode, begin, begin + step,
            [&](DocId d) { pieces.push_back(d); });
      }
      EXPECT_THAT(pieces, ElementsAreArray(expected)) << "step " << step;
    }
  }
  EXPECT_GT(union_hits, 500);
}

// Every trigram's list, with lengths from inlined to multi-block, matches
// the reference, and absent trigrams have empty lists.
TEST_F(SegmentTest, PostingListsMatchReference) {
  std::mt19937_64 rng(ferret_testing::TestSeed(21));
  const std::vector<TestDoc> docs = RandomDocs(rng, 60);
  const SegmentWriterOptions options = {.chunk_bytes = 64};
  FERRET_ASSERT_OK_AND_ASSIGN(const SegmentStats stats,
                              WriteSegment(Path("s.fseg"), docs, options));
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment,
                              Segment::Open(*Env::Default(), Path("s.fseg")));
  const auto reference = ReferencePostings(docs, 64);
  EXPECT_EQ(segment->num_trigrams(), reference.size());
  EXPECT_TRUE(segment->has_short_line_trigrams());
  // The alphabet makes lines such as "{}" and "a(" common, so the lists
  // include pseudo-trigrams.
  EXPECT_TRUE(reference.contains(MakeTrigram('{', '}', '\n')));
  uint64_t postings = 0;
  bool saw_inlined = false;
  bool saw_blocked = false;
  for (const auto& [trigram, ids] : reference) {
    FERRET_ASSERT_OK_AND_ASSIGN(const PostingListView list,
                                segment->FindPostingList(trigram));
    ASSERT_THAT(Decode(list), ElementsAreArray(ids))
        << TrigramToString(trigram);
    postings += ids.size();
    saw_inlined |= IsInlinedPostingList(ids.size());
    saw_blocked |= ids.size() >= size_t{2} * kPostingBlockSize;
  }
  EXPECT_EQ(stats.num_postings, postings);
  EXPECT_TRUE(saw_inlined);
  EXPECT_TRUE(saw_blocked);
  for (const Trigram absent : {MakeTrigram('z', 'z', 'z'), MakeTrigram(0, 0, 0),
                               kTrigramMask, Trigram{0x01000000}}) {
    FERRET_ASSERT_OK_AND_ASSIGN(const PostingListView list,
                                segment->FindPostingList(absent));
    EXPECT_EQ(list.size(), 0u);
  }
}

TEST_F(SegmentTest, EvaluatorOverSegment) {
  std::mt19937_64 rng(ferret_testing::TestSeed(22));
  const std::vector<TestDoc> docs = RandomDocs(rng, 40);
  ASSERT_THAT(WriteSegment(Path("s.fseg"), docs, {.chunk_bytes = 64}).status(),
              IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment,
                              Segment::Open(*Env::Default(), Path("s.fseg")));
  const auto reference = ReferencePostings(docs, 64);
  const auto has = [&](Trigram t, ChunkId c) {
    const auto it = reference.find(t);
    return it != reference.end() &&
           std::binary_search(it->second.begin(), it->second.end(), c);
  };
  const Trigram abc = MakeTrigram('a', 'b', 'c');
  const Trigram cde = MakeTrigram('c', 'd', 'e');
  const Trigram paren = MakeTrigram('(', ')', '{');
  const TrigramQuery query = TrigramQuery::Or(TrigramQuery::AllOf({abc, cde}),
                                              TrigramQuery::AllOf({paren}));
  FERRET_ASSERT_OK_AND_ASSIGN(auto evaluator, segment->NewEvaluator(query));
  ChunkIdBuffer out;
  const ChunkRange all = {0, segment->num_chunks() - 1};
  evaluator.Evaluate(all, out);
  std::vector<ChunkId> expected;
  for (ChunkId c = 0; c < segment->num_chunks(); ++c) {
    if ((has(abc, c) && has(cde, c)) || has(paren, c)) expected.push_back(c);
  }
  EXPECT_FALSE(expected.empty());
  EXPECT_THAT(std::vector<ChunkId>(out.ids().begin(), out.ids().end()),
              ElementsAreArray(expected));
}

// The file depends only on the documents, not on how postings were
// partitioned while building.
TEST_F(SegmentTest, OutputIndependentOfPartitioning) {
  std::mt19937_64 rng(ferret_testing::TestSeed(23));
  const std::vector<TestDoc> docs = RandomDocs(rng, 30);
  ASSERT_THAT(WriteSegment(Path("one.fseg"), docs,
                           {.chunk_bytes = 128,
                            .max_partition_postings = size_t{1} << 24})
                  .status(),
              IsOk());
  ASSERT_THAT(WriteSegment(Path("many.fseg"), docs,
                           {.chunk_bytes = 128, .max_partition_postings = 50})
                  .status(),
              IsOk());
  EXPECT_TRUE(ReadAll(Path("one.fseg")) == ReadAll(Path("many.fseg")));
}

TEST_F(SegmentTest, WriterRejectsBadInput) {
  EXPECT_THAT(
      SegmentWriter::Create(*Env::Default(), Path("x.fseg"), {.chunk_bytes = 0})
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument));
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto writer, SegmentWriter::Create(*Env::Default(), Path("x.fseg")));
  EXPECT_THAT(writer->Add(ToSegmentDocument({"", "content"})),
              StatusIs(absl::StatusCode::kInvalidArgument));
  const TestDoc deep = {"a/b/c/d.txt", "content"};
  const TestDoc shallow = {"a.txt", "content"};
  ASSERT_THAT(writer->Add(ToSegmentDocument(deep)), IsOk());
  EXPECT_THAT(
      writer->Add(ToSegmentDocument(shallow)),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("rank order")));
  // A rejected document leaves the writer usable.
  ASSERT_THAT(writer->Add(ToSegmentDocument(deep)), IsOk());
  ASSERT_THAT(writer->Finish().status(), IsOk());
  EXPECT_THAT(writer->Finish().status(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment,
                              Segment::Open(*Env::Default(), Path("x.fseg")));
  EXPECT_EQ(segment->num_docs(), 2u);
}

TEST_F(SegmentTest, SymbolSectionLookupAndLegacyCompatibility) {
  const TestDoc file{"main.cc", "int Foo() { return 1; }\nint foo();\n",
                     Language::kCpp};
  ASSERT_THAT(WriteSegment(Path("symbols.fseg"), {file}).status(), IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto segment, Segment::Open(*Env::Default(), Path("symbols.fseg")));
  EXPECT_TRUE(segment->has_symbols());
  std::vector<SegmentSymbol> found;
  segment->FindSymbols("Foo", false,
                       [&](SegmentSymbol symbol) { found.push_back(symbol); });
  ASSERT_EQ(found.size(), 1u);
  EXPECT_EQ(found[0].name, "Foo");
  EXPECT_TRUE(found[0].is_definition);
  found.clear();
  segment->FindSymbols("FOO", true,
                       [&](SegmentSymbol symbol) { found.push_back(symbol); });
  ASSERT_EQ(found.size(), 2u);
  EXPECT_THAT(segment->VerifyChecksums(), IsOk());
  ASSERT_THAT(
      WriteSegment(Path("legacy.fseg"), {file}, {.write_symbols = false})
          .status(),
      IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto legacy, Segment::Open(*Env::Default(), Path("legacy.fseg")));
  EXPECT_FALSE(legacy->has_symbols());
  EXPECT_EQ(legacy->doc(0).content, file.content);
}

TEST_F(SegmentTest, RejectsInvalidSymbolBoundsAndDetectsCorruptBytes) {
  ASSERT_THAT(
      WriteSegment(Path("valid.fseg"),
                   {{"main.cc", "int Foo() { return 1; }\n", Language::kCpp}})
          .status(),
      IsOk());
  std::string bytes = ReadAll(Path("valid.fseg"));
  namespace sf = segment_format;
  const auto header = sf::ReadRecord<sf::SegmentHeader>(bytes, 0);
  uint64_t offset = 0;
  for (uint32_t i = 0; i < header.num_sections; ++i) {
    const auto entry = sf::ReadRecord<sf::SectionEntry>(
        absl::string_view(bytes).substr(sizeof(sf::SegmentHeader)), i);
    if (entry.kind == static_cast<uint32_t>(sf::SectionKind::kSymbols)) {
      offset = entry.offset;
    }
  }
  ASSERT_GT(offset, 0u);
  std::string bad = bytes;
  // Replace the name pool length; the section no longer fits its records.
  bad[offset + 4] = '\x7f';
  ASSERT_THAT(dir_.WriteFile("bad_size.fseg", bad), IsOk());
  EXPECT_THAT(Segment::Open(*Env::Default(), Path("bad_size.fseg")).status(),
              StatusIs(absl::StatusCode::kDataLoss, HasSubstr("symbols")));
  bad = bytes;
  const size_t record = offset + 8;
  // SymbolRecord::name_offset exceeds the pool; Open must not dereference it.
  bad[record + offsetof(sf::SymbolRecord, name_offset) + 3] = '\x7f';
  ASSERT_THAT(dir_.WriteFile("bad_offset.fseg", bad), IsOk());
  EXPECT_THAT(Segment::Open(*Env::Default(), Path("bad_offset.fseg")).status(),
              StatusIs(absl::StatusCode::kDataLoss, HasSubstr("symbols")));
  bad = bytes;
  bad[record + offsetof(sf::SymbolRecord, pad)] ^= 1;
  ASSERT_THAT(dir_.WriteFile("bad_crc.fseg", bad), IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto segment, Segment::Open(*Env::Default(), Path("bad_crc.fseg")));
  EXPECT_THAT(segment->VerifyChecksums(),
              StatusIs(absl::StatusCode::kDataLoss));
}

TEST_F(SegmentTest, OpenRejectsNonSegments) {
  ASSERT_THAT(dir_.WriteFile("empty", ""), IsOk());
  EXPECT_THAT(Segment::Open(*Env::Default(), Path("empty")).status(),
              StatusIs(absl::StatusCode::kDataLoss));
  ASSERT_THAT(dir_.WriteFile("junk", std::string(40000, 'x')), IsOk());
  EXPECT_THAT(Segment::Open(*Env::Default(), Path("junk")).status(),
              StatusIs(absl::StatusCode::kDataLoss, HasSubstr("magic")));
  EXPECT_THAT(Segment::Open(*Env::Default(), Path("missing")).status(),
              StatusIs(absl::StatusCode::kNotFound));
}

// Reads everything a searcher would read, so that ASan sees any
// out-of-bounds access a corrupt file could provoke.
void TouchEverything(const Segment& segment,
                     const std::vector<Trigram>& trigrams) {
  size_t bytes = 0;
  for (DocId d = 0; d < segment.num_docs(); ++d) {
    const SegmentDoc doc = segment.doc(d);
    bytes += doc.hint_path.size() + doc.content.size();
    for (const char c : doc.content) bytes += static_cast<unsigned char>(c);
  }
  for (ChunkId c = 0; c < segment.num_chunks(); ++c) {
    for (const char ch : segment.chunk(c).content) {
      bytes += static_cast<unsigned char>(ch);
    }
  }
  for (const Trigram t : trigrams) {
    absl::StatusOr<PostingListView> list = segment.FindPostingList(t);
    if (!list.ok()) continue;
    for (const ChunkId id : Decode(*list)) ASSERT_LT(id, segment.num_chunks());
  }
  segment.ForEachBasenameContaining(
      "e", LiteralFinder::Mode::kAsciiCaseInsensitive, [&](DocId d) {
        EXPECT_LT(d, segment.num_docs());
        ++bytes;
      });
  (void)segment.VerifyChecksums();
  volatile size_t sink = bytes;
  (void)sink;
}

// Expects `actual` to present exactly the documents, chunks and posting
// lists of `expected`.
void ExpectSameContents(const Segment& expected, const Segment& actual,
                        const std::vector<Trigram>& trigrams) {
  ASSERT_EQ(actual.num_docs(), expected.num_docs());
  ASSERT_EQ(actual.num_chunks(), expected.num_chunks());
  for (DocId d = 0; d < expected.num_docs(); ++d) {
    const SegmentDoc a = actual.doc(d);
    const SegmentDoc e = expected.doc(d);
    EXPECT_EQ(a.hint_path, e.hint_path);
    EXPECT_EQ(a.content, e.content);
    EXPECT_EQ(a.oid, e.oid);
    EXPECT_EQ(a.tier, e.tier);
    EXPECT_EQ(a.prior, e.prior);
  }
  for (ChunkId c = 0; c < expected.num_chunks(); ++c) {
    EXPECT_EQ(actual.chunk(c).content, expected.chunk(c).content);
    EXPECT_EQ(actual.chunk(c).first_line, expected.chunk(c).first_line);
  }
  for (const Trigram t : trigrams) {
    absl::StatusOr<PostingListView> a = actual.FindPostingList(t);
    absl::StatusOr<PostingListView> e = expected.FindPostingList(t);
    ASSERT_TRUE(a.ok() && e.ok());
    EXPECT_EQ(Decode(*a), Decode(*e));
  }
}

// Flips random bytes and truncates the file. Open() must either reject the
// result as DataLoss (or FailedPrecondition, for a changed version), or
// accept it and then never read out of bounds. VerifyChecksums() must then
// catch the change unless it is harmless: a flip in padding or a reserved
// field that leaves every document, chunk and posting list as it was.
TEST_F(SegmentTest, CorruptionNeverReadsOutOfBounds) {
  std::mt19937_64 rng(ferret_testing::TestSeed(24));
  const std::vector<TestDoc> docs = RandomDocs(rng, 12);
  ASSERT_THAT(
      WriteSegment(Path("good.fseg"), docs, {.chunk_bytes = 64}).status(),
      IsOk());
  const std::string good = ReadAll(Path("good.fseg"));
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto good_segment, Segment::Open(*Env::Default(), Path("good.fseg")));
  std::vector<Trigram> trigrams;
  for (const auto& [t, ids] : ReferencePostings(docs, 64)) {
    trigrams.push_back(t);
  }

  int accepted = 0;
  int detected = 0;
  for (int iteration = 0; iteration < 400; ++iteration) {
    SCOPED_TRACE(absl::StrCat("iteration ", iteration));
    std::string bad = good;
    if (iteration % 10 == 0) {
      bad.resize(rng() % bad.size());
    } else {
      for (int flips = 1 + static_cast<int>(rng() % 3); flips > 0; --flips) {
        // Bias half of the flips toward the metadata page and the footer.
        const size_t pos = rng() % 2 == 0   ? rng() % bad.size()
                           : rng() % 2 == 0 ? rng() % 1024
                                            : bad.size() - 1 - rng() % 32;
        bad[pos] = static_cast<char>(bad[pos] ^ (1 + rng() % 255));
      }
    }
    const std::string name = absl::StrCat("bad", iteration, ".fseg");
    ASSERT_THAT(dir_.WriteFile(name, bad), IsOk());
    absl::StatusOr<std::unique_ptr<Segment>> segment =
        Segment::Open(*Env::Default(), Path(name));
    if (!segment.ok()) {
      EXPECT_THAT(segment.status(),
                  AnyOf(StatusIs(absl::StatusCode::kDataLoss),
                        StatusIs(absl::StatusCode::kFailedPrecondition)));
      continue;
    }
    ++accepted;
    TouchEverything(**segment, trigrams);
    if (!(*segment)->VerifyChecksums().ok()) {
      ++detected;
      continue;
    }
    ExpectSameContents(*good_segment, **segment, trigrams);
  }
  // Most flips land in content or postings, which Open() does not check.
  EXPECT_GT(accepted, 50);
  EXPECT_GT(detected, 10);
}

}  // namespace
}  // namespace ferret

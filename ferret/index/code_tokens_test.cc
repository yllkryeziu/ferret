// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The kCodeTokens section: the lists must contain every document where
// MatchClassifier labels a match of the token kCodeIdentifier, and must
// survive corruption checks.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <set>
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
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/git/blob_hash.h"
#include "ferret/index/internal/segment_format.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/segment.h"
#include "ferret/index/segment_writer.h"
#include "ferret/postings/ids.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/testing/test_seed.h"
#include "ferret/text/fold.h"
#include "ferret/text/language.h"
#include "ferret/text/lexer.h"

namespace ferret {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

namespace sf = segment_format;

struct TestDoc {
  std::string path;
  std::string content;
  Language language = Language::kUnknown;
};

SegmentDocument ToSegmentDocument(const TestDoc& doc) {
  return SegmentDocument{
      .hint_path = doc.path,
      .content = doc.content,
      .rank_key = ComputeRankKey(doc.path, doc.content.size(),
                                 /*long_lines=*/false, HashBlob(doc.content)),
      .language = doc.language,
  };
}

// Writes `docs` in rank order to `path` and returns the order.
absl::StatusOr<std::vector<TestDoc>> WriteSegment(
    const std::string& path, std::vector<TestDoc> docs,
    const SegmentWriterOptions& options = {}) {
  std::stable_sort(
      docs.begin(), docs.end(), [](const TestDoc& a, const TestDoc& b) {
        return ToSegmentDocument(a).rank_key < ToSegmentDocument(b).rank_key;
      });
  absl::StatusOr<std::unique_ptr<SegmentWriter>> writer =
      SegmentWriter::Create(*Env::Default(), path, options);
  if (!writer.ok()) return writer.status();
  for (const TestDoc& doc : docs) {
    if (absl::Status s = (*writer)->Add(ToSegmentDocument(doc)); !s.ok()) {
      return s;
    }
  }
  if (absl::Status s = (*writer)->Finish().status(); !s.ok()) return s;
  return docs;
}

std::vector<DocId> Lookup(const Segment& segment, absl::string_view query) {
  std::vector<DocId> docs;
  EXPECT_TRUE(segment.FindCodeTokenDocs(Fold(query), docs)) << query;
  return docs;
}

class CodeTokensTest : public ::testing::Test {
 protected:
  std::string Path(absl::string_view name) const { return dir_.Join(name); }

  absl::StatusOr<std::unique_ptr<Segment>> Build(
      std::vector<TestDoc> docs, const SegmentWriterOptions& options = {},
      std::vector<TestDoc>* ordered = nullptr) {
    absl::StatusOr<std::vector<TestDoc>> written =
        WriteSegment(Path("test.fseg"), std::move(docs), options);
    if (!written.ok()) return written.status();
    if (ordered != nullptr) *ordered = *std::move(written);
    return Segment::Open(*Env::Default(), Path("test.fseg"));
  }

  ferret_testing::TempDir dir_;
};

TEST_F(CodeTokensTest, ListsDocumentsWhereTheTokenIsACodeIdentifier) {
  std::vector<TestDoc> order;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto segment,
      Build({{"a.cc", "int Flow = 1;\n", Language::kCpp},
             {"b.cc", "int overflow = 1;\n// flow\nconst char* s = \"flow\";\n",
              Language::kCpp},
             {"c.cc", "flow(); flow_2(); x.flow;\n", Language::kCpp},
             {"d.txt", "flow of text\n", Language::kUnknown},
             {"e.py", "# flow\n\"\"\"\nflow\n\"\"\"\n", Language::kPython}},
            {}, &order));
  ASSERT_TRUE(segment->has_code_tokens());
  std::vector<std::string> listed;
  for (const DocId doc : Lookup(*segment, "FLOW")) {
    listed.push_back(order[doc].path);
  }
  // Case variants share one list; a substring, comment, string and Python
  // docstring do not put a document in it. An unknown language is lexed as
  // plain code, exactly as the verifier lexes it.
  std::sort(listed.begin(), listed.end());
  EXPECT_THAT(listed, ElementsAre("a.cc", "c.cc", "d.txt"));
  EXPECT_THAT(Lookup(*segment, "flow_2").size(), 1u);
  EXPECT_THAT(Lookup(*segment, "overflow").size(), 1u);
  EXPECT_THAT(Lookup(*segment, "absent"), IsEmpty());
  EXPECT_THAT(segment->VerifyChecksums(), IsOk());
}

TEST_F(CodeTokensTest, KelvinSignAndLongSFoldToAsciiTokens) {
  std::vector<TestDoc> order;
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto segment, Build({{"a.cc", "int \xE2\x84\xAAit;\n", Language::kCpp},
                           {"b.cc", "int \xC5\xBFtop;\n", Language::kCpp},
                           {"c.cc", "int caf\xC3\xA9;\n", Language::kCpp}},
                          {}, &order));
  // RE2's (?i) matches U+212A for `k` and U+017F for `s`, so a
  // case-insensitive query "kit" matches the first document's token, which
  // must fold into the list. Tokens with other non-ASCII bytes cannot equal
  // an ASCII query and are not stored.
  ASSERT_EQ(Lookup(*segment, "kit").size(), 1u);
  EXPECT_EQ(order[Lookup(*segment, "kit")[0]].path, "a.cc");
  ASSERT_EQ(Lookup(*segment, "stop").size(), 1u);
  EXPECT_EQ(order[Lookup(*segment, "stop")[0]].path, "b.cc");
  segment->ForEachCodeToken(
      [](absl::string_view token, absl::Span<const DocId>) {
        for (const char c : token) EXPECT_LT(static_cast<uint8_t>(c), 0x80u);
      });
}

TEST_F(CodeTokensTest, DropsTokensWithDigitFirstDollarOrTooLong) {
  const std::string long_token(sf::kMaxCodeTokenBytes + 1, 'a');
  const std::string edge_token(sf::kMaxCodeTokenBytes, 'b');
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto segment,
      Build({{"a.js",
              absl::StrCat("var $jq = 1; var 9lives = 2; var ", long_token,
                           " = 3; var ", edge_token, " = 4;\n"),
              Language::kJavaScript}}));
  std::set<std::string> tokens;
  ASSERT_TRUE(segment->ForEachCodeToken(
      [&](absl::string_view token, absl::Span<const DocId>) {
        tokens.emplace(token);
      }));
  EXPECT_THAT(tokens, ::testing::UnorderedElementsAre("var", edge_token));
  // A query longer than the limit cannot be answered, so the reader says so
  // instead of reporting an empty list.
  std::vector<DocId> docs;
  EXPECT_FALSE(segment->FindCodeTokenDocs(long_token, docs));
  EXPECT_FALSE(segment->FindCodeTokenDocs("", docs));
  EXPECT_TRUE(segment->FindCodeTokenDocs(edge_token, docs));
  EXPECT_EQ(docs.size(), 1u);
}

TEST_F(CodeTokensTest, LegacySegmentHasNoSection) {
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment,
                              Build({{"a.cc", "int flow;\n", Language::kCpp}},
                                    {.write_code_tokens = false}));
  EXPECT_FALSE(segment->has_code_tokens());
  EXPECT_EQ(segment->code_tokens_bytes(), 0);
  std::vector<DocId> docs;
  EXPECT_FALSE(segment->FindCodeTokenDocs("flow", docs));
  EXPECT_FALSE(segment->ForEachCodeToken(
      [](absl::string_view, absl::Span<const DocId>) {}));
  EXPECT_THAT(segment->VerifyChecksums(), IsOk());
  EXPECT_EQ(segment->doc(0).content, "int flow;\n");
}

TEST_F(CodeTokensTest, EmptySegmentHasAnEmptySection) {
  FERRET_ASSERT_OK_AND_ASSIGN(auto segment, Build({}));
  EXPECT_TRUE(segment->has_code_tokens());
  std::vector<DocId> docs;
  EXPECT_TRUE(segment->FindCodeTokenDocs("flow", docs));
  EXPECT_THAT(docs, IsEmpty());
}

// Returns the offset and length of the kCodeTokens section in `bytes`.
std::pair<size_t, size_t> CodeTokensSection(const std::string& bytes) {
  const auto header = sf::ReadRecord<sf::SegmentHeader>(bytes, 0);
  for (uint32_t i = 0; i < header.num_sections; ++i) {
    const auto entry = sf::ReadRecord<sf::SectionEntry>(
        absl::string_view(bytes).substr(sizeof(sf::SegmentHeader)), i);
    if (entry.kind == static_cast<uint32_t>(sf::SectionKind::kCodeTokens)) {
      return {entry.offset, entry.length};
    }
  }
  return {0, 0};
}

TEST_F(CodeTokensTest, RejectsMalformedSectionsAndDetectsCorruptBytes) {
  ASSERT_THAT(WriteSegment(Path("valid.fseg"),
                           {{"a.cc", "int alpha; int beta; int gamma;\n",
                             Language::kCpp},
                            {"b.cc", "int beta;\n", Language::kCpp}})
                  .status(),
              IsOk());
  absl::StatusOr<std::string> valid =
      Env::Default()->ReadFile(Path("valid.fseg"));
  ASSERT_THAT(valid.status(), IsOk());
  const auto [offset, length] = CodeTokensSection(*valid);
  ASSERT_GT(length, sizeof(sf::CodeTokensHeader));
  const auto check_open = [&](const std::string& name, const std::string& bad,
                              bool fails) {
    ASSERT_THAT(dir_.WriteFile(name, bad), IsOk());
    const absl::StatusOr<std::unique_ptr<Segment>> segment =
        Segment::Open(*Env::Default(), Path(name));
    if (fails) {
      EXPECT_THAT(segment.status(), StatusIs(absl::StatusCode::kDataLoss,
                                             HasSubstr("code token")))
          << name;
    } else {
      ASSERT_THAT(segment.status(), IsOk()) << name;
      EXPECT_THAT((*segment)->VerifyChecksums(),
                  StatusIs(absl::StatusCode::kDataLoss))
          << name;
    }
  };
  std::string bad = *valid;
  // The token count no longer fits the section.
  bad[offset + offsetof(sf::CodeTokensHeader, num_tokens)] = '\x7f';
  check_open("bad_count.fseg", bad, true);
  bad = *valid;
  bad[offset + offsetof(sf::CodeTokensHeader, pool_bytes)] ^= 1;
  check_open("bad_pool.fseg", bad, true);
  // An empty first key, a first key longer than the limit, and a first list
  // that ends past the posting bytes.
  const size_t first_entry = offset + sizeof(sf::CodeTokensHeader);
  bad = *valid;
  bad[first_entry + offsetof(sf::CodeTokenEntry, key_end)] = '\0';
  check_open("empty_key.fseg", bad, true);
  bad = *valid;
  bad[first_entry + offsetof(sf::CodeTokenEntry, key_end)] =
      static_cast<char>(sf::kMaxCodeTokenBytes + 1);
  check_open("long_key.fseg", bad, true);
  bad = *valid;
  bad[first_entry + offsetof(sf::CodeTokenEntry, postings_end)] = '\x7f';
  check_open("bad_list_end.fseg", bad, true);
  // The keys are alpha, beta, gamma, int: "zlpha" breaks the sort order.
  bad = *valid;
  const size_t pool =
      offset + sizeof(sf::CodeTokensHeader) + 4 * sizeof(sf::CodeTokenEntry);
  bad[pool] = 'z';
  check_open("unsorted.fseg", bad, true);
  // A flipped bit in a list is caught by the checksum, and by decoding.
  bad = *valid;
  bad[offset + length - 1] ^= 0x40;
  ASSERT_THAT(dir_.WriteFile("bad_list.fseg", bad), IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto segment, Segment::Open(*Env::Default(), Path("bad_list.fseg")));
  EXPECT_THAT(segment->VerifyChecksums(),
              StatusIs(absl::StatusCode::kDataLoss));
  std::vector<DocId> docs;
  // Whatever the flip decoded to, no id is out of range, and a corrupt list
  // says "cannot tell" rather than "absent".
  for (const char* token : {"alpha", "beta", "gamma"}) {
    if (segment->FindCodeTokenDocs(token, docs)) {
      for (const DocId doc : docs) EXPECT_LT(doc, segment->num_docs());
    }
  }
}

// Random single-byte damage anywhere in the section never crashes a lookup
// and never yields an id outside the segment.
TEST_F(CodeTokensTest, CorruptSectionNeverReadsOutOfBounds) {
  std::mt19937_64 rng(ferret_testing::TestSeed(9051));
  ASSERT_THAT(WriteSegment(Path("valid.fseg"),
                           {{"a.cc", "int alpha; int beta; int gamma;\n",
                             Language::kCpp},
                            {"b.cc", "int beta; int delta;\n", Language::kCpp},
                            {"c.cc", "delta(alpha);\n", Language::kCpp}})
                  .status(),
              IsOk());
  absl::StatusOr<std::string> valid =
      Env::Default()->ReadFile(Path("valid.fseg"));
  ASSERT_THAT(valid.status(), IsOk());
  const auto [offset, length] = CodeTokensSection(*valid);
  for (int trial = 0; trial < 300; ++trial) {
    std::string bad = *valid;
    const size_t at = offset + rng() % length;
    bad[at] = static_cast<char>(bad[at] ^ static_cast<char>(1 + rng() % 255));
    ASSERT_THAT(dir_.WriteFile("bad.fseg", bad), IsOk());
    const absl::StatusOr<std::unique_ptr<Segment>> segment =
        Segment::Open(*Env::Default(), Path("bad.fseg"));
    if (!segment.ok()) continue;
    std::vector<DocId> docs;
    for (const char* token : {"alpha", "beta", "gamma", "delta", "int", "zz"}) {
      if ((*segment)->FindCodeTokenDocs(token, docs)) {
        for (const DocId doc : docs) EXPECT_LT(doc, (*segment)->num_docs());
      }
    }
    (*segment)->ForEachCodeToken(
        [&](absl::string_view, absl::Span<const DocId> ids) {
          for (const DocId doc : ids) EXPECT_LT(doc, (*segment)->num_docs());
        });
  }
}

// Tokens for the random documents: an identifier, its case variants and
// substrings, alone and inside longer identifiers.
const std::vector<std::string>& Identifiers() {
  static const auto* identifiers = new std::vector<std::string>{
      "foo", "Foo", "FOO", "foobar", "barfoo", "overflow", "flow", "x1", "_id",
      "a_b", "fo",  "oo",  "Flow",   "o",      "f",        "Ab",   "b1"};
  return *identifiers;
}

struct Vocabulary {
  Language language;
  const char* extension;
  std::vector<std::string> constructs;
};

std::vector<Vocabulary> Vocabularies() {
  return {
      {Language::kC,
       ".c",
       {"/*", "*/", "//", "\"", "'", "\\", "\\\"", "#", "\"s\"", "'a'"}},
      {Language::kCpp,
       ".cc",
       {"/*", "*/", "//", "\"", "'", "\\", "R\"x(", ")x\"", "1'000", "\"s\""}},
      {Language::kPython,
       ".py",
       {"#", "\"", "'", "\"\"\"", "'''", "\\", "r'", "f\"", "\"\""}},
      {Language::kGo, ".go", {"//", "/*", "*/", "\"", "'", "`", "\\"}},
      {Language::kJava, ".java", {"//", "/*", "*/", "\"", "'", "\"\"\"", "$x"}},
      {Language::kJavaScript,
       ".js",
       {"//", "/*", "*/", "\"", "'", "`", "${", "}", "= /re[/]x/g", "$q"}},
      {Language::kRust,
       ".rs",
       {"//", "/* c */", "/*", "*/", "\"", "'a'", "'a", "r\"", "r#\"", "\"#"}},
      {Language::kShell,
       ".sh",
       {"#", "a#b", "$#", "'", "\"", "\\", "`", "$(", "${x}", "<<EOF", "EOF"}},
      {Language::kMarkup, ".html", {"<!--", "-->", "\"", "'", "#", "//"}},
      {Language::kUnknown, ".txt", {"/*", "*/", "//", "\"", "#"}},
  };
}

// Every occurrence of `query` in `text` (compared exactly or ignoring ASCII
// case) as a half-open range.
std::vector<std::pair<size_t, size_t>> Occurrences(absl::string_view text,
                                                   absl::string_view query,
                                                   bool case_sensitive) {
  std::vector<std::pair<size_t, size_t>> found;
  for (size_t begin = 0; begin + query.size() <= text.size(); ++begin) {
    const absl::string_view candidate = text.substr(begin, query.size());
    if (case_sensitive ? candidate == query
                       : absl::EqualsIgnoreCase(candidate, query)) {
      found.emplace_back(begin, begin + query.size());
    }
  }
  return found;
}

// The soundness requirement of the code-token bound: for every document and
// query where MatchClassifier labels any match of the query kCodeIdentifier,
// with the entry states the segment stores, fold(query)'s list contains the
// document. Documents mix identifiers, their substrings and case variants,
// line and block comments (crossing chunk boundaries at small chunk sizes),
// strings, raw strings, and unknown languages. The lists must also be
// selective, or the bound would gain nothing.
TEST_F(CodeTokensTest, ListsAreASupersetOfClassifiedCodeIdentifiers) {
  const uint64_t seed = ferret_testing::TestSeed(20261005);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  const std::vector<std::string>& identifiers = Identifiers();
  int64_t code_matches = 0;
  int64_t excluded_pairs = 0;
  int64_t chunks_in_open_construct = 0;
  int64_t chunks_in_block_comment = 0;
  for (int round = 0; round < 40; ++round) {
    std::vector<TestDoc> docs;
    const int num_docs = 4 + static_cast<int>(rng() % 12);
    const std::vector<Vocabulary> vocabularies = Vocabularies();
    for (int d = 0; d < num_docs; ++d) {
      const Vocabulary& vocabulary = vocabularies[rng() % vocabularies.size()];
      std::string content;
      const size_t tokens = 1 + rng() % 120;
      for (size_t i = 0; i < tokens; ++i) {
        const uint64_t pick = rng() % 10;
        if (pick < 5) {
          content += identifiers[rng() % identifiers.size()];
        } else if (pick < 8) {
          content +=
              vocabulary.constructs[rng() % vocabulary.constructs.size()];
        } else {
          content += std::string(1, "(){};=+.,"[rng() % 9]);
        }
        const uint64_t gap = rng() % 6;
        content += gap == 0 ? "\n" : gap == 1 ? "" : " ";
      }
      docs.push_back({absl::StrCat("d", d, "/f", d, vocabulary.extension),
                      std::move(content), vocabulary.language});
    }
    const int64_t chunk_bytes =
        std::vector<int64_t>{8, 24, 64, 300, 8192}[rng() % 5];
    std::vector<TestDoc> order;
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto segment, Build(docs, {.chunk_bytes = chunk_bytes}, &order));
    ASSERT_TRUE(segment->has_code_tokens());
    // Queries: every identifier, its case variants, and its substrings.
    std::set<std::string> queries;
    for (const std::string& id : identifiers) {
      queries.insert(id);
      queries.insert(absl::AsciiStrToUpper(id));
      queries.insert(absl::AsciiStrToLower(id));
      for (size_t b = 0; b < id.size(); ++b) {
        for (size_t l = 1; b + l <= id.size(); ++l) {
          queries.insert(id.substr(b, l));
        }
      }
    }
    for (const std::string& query : queries) {
      // An eligible query is an identifier: it does not start with a digit.
      if (absl::ascii_isdigit(static_cast<unsigned char>(query[0]))) continue;
      const std::vector<DocId> listed = Lookup(*segment, query);
      const std::set<DocId> listed_set(listed.begin(), listed.end());
      ASSERT_TRUE(std::is_sorted(listed.begin(), listed.end()));
      for (DocId d = 0; d < segment->num_docs(); ++d) {
        const SegmentDoc doc = segment->doc(d);
        bool has_code_identifier = false;
        bool has_occurrence = false;
        for (ChunkId c = doc.first_chunk; c < doc.first_chunk + doc.num_chunks;
             ++c) {
          const SegmentChunk chunk = segment->chunk(c);
          for (const bool case_sensitive : {true, false}) {
            const auto occurrences =
                Occurrences(chunk.content, query, case_sensitive);
            has_occurrence |= !occurrences.empty();
            MatchClassifier classifier(doc.language, chunk.content,
                                       chunk.entry_state);
            for (const auto& [begin, end] : occurrences) {
              if (classifier.Classify(begin, end) ==
                  MatchClass::kCodeIdentifier) {
                has_code_identifier = true;
              }
            }
          }
        }
        if (has_code_identifier) {
          ++code_matches;
          EXPECT_TRUE(listed_set.contains(d))
              << "missing " << order[d].path << " for query " << query
              << " chunk_bytes " << chunk_bytes << "\n"
              << order[d].content;
        } else if (has_occurrence && !listed_set.contains(d)) {
          ++excluded_pairs;
        }
      }
    }
    for (ChunkId c = 0; c < segment->num_chunks(); ++c) {
      const LexState state = segment->chunk(c).entry_state;
      if (state != LexState::kCode) ++chunks_in_open_construct;
      if (state == LexState::kBlockComment) ++chunks_in_block_comment;
    }
    if (::testing::Test::HasFailure()) return;
  }
  // The generator reaches what it claims to: matches in code, matches in
  // comments and strings that the lists leave out, and chunks that start
  // inside a block comment or another construct.
  EXPECT_GT(code_matches, 2000);
  EXPECT_GT(excluded_pairs, 500);
  EXPECT_GT(chunks_in_open_construct, 30);
  EXPECT_GT(chunks_in_block_comment, 10);
}

}  // namespace
}  // namespace ferret

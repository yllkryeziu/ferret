// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The default bulk build (one read per file, rank-ordered assembly, parallel
// finish) must write the bytes of the serial build, which feeds a
// SegmentWriter one document at a time.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/base/mapped_file.h"
#include "ferret/base/thread_pool.h"
#include "ferret/index/bulk_build.h"
#include "ferret/index/segment.h"
#include "ferret/postings/ids.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/text/eligibility.h"

namespace ferret {
namespace {

using ::absl_testing::StatusIs;
using ::ferret_testing::TempDir;

// Reads through to the real file system and counts the reads of each path.
// `on_read` may replace the content a read returns.
class CountingEnv final : public Env {
 public:
  using Hook = absl::AnyInvocable<std::optional<std::string>(
      const std::string& path, int read_number) const>;

  void set_hook(Hook hook) { hook_ = std::move(hook); }

  int reads(const std::string& path) const {
    absl::MutexLock lock(mu_);
    const auto it = reads_.find(path);
    return it == reads_.end() ? 0 : it->second;
  }
  int total_reads() const {
    absl::MutexLock lock(mu_);
    int total = 0;
    for (const auto& entry : reads_) total += entry.second;
    return total;
  }
  void ResetReads() {
    absl::MutexLock lock(mu_);
    reads_.clear();
  }

  absl::StatusOr<std::unique_ptr<WritableFile>> NewWritableFile(
      absl::string_view path) override {
    return Env::Default()->NewWritableFile(path);
  }
  absl::StatusOr<std::string> ReadFile(absl::string_view path) override {
    int number = 0;
    {
      absl::MutexLock lock(mu_);
      number = ++reads_[std::string(path)];
    }
    if (hook_) {
      std::optional<std::string> replaced = hook_(std::string(path), number);
      if (replaced.has_value()) return *std::move(replaced);
    }
    return Env::Default()->ReadFile(path);
  }
  absl::StatusOr<std::unique_ptr<MappedFile>> NewMappedFile(
      absl::string_view path) override {
    return Env::Default()->NewMappedFile(path);
  }
  absl::StatusOr<int64_t> GetFileSize(absl::string_view path) override {
    return Env::Default()->GetFileSize(path);
  }
  absl::StatusOr<bool> FileExists(absl::string_view path) override {
    return Env::Default()->FileExists(path);
  }
  absl::StatusOr<std::vector<std::string>> GetChildren(
      absl::string_view path) override {
    return Env::Default()->GetChildren(path);
  }
  absl::Status CreateDir(absl::string_view path) override {
    return Env::Default()->CreateDir(path);
  }
  absl::Status DeleteFile(absl::string_view path) override {
    return Env::Default()->DeleteFile(path);
  }
  absl::Status RenameFile(absl::string_view from,
                          absl::string_view to) override {
    return Env::Default()->RenameFile(from, to);
  }
  absl::Status SyncDir(absl::string_view path) override {
    return Env::Default()->SyncDir(path);
  }
  absl::StatusOr<std::unique_ptr<FileLock>> LockFile(
      absl::string_view path) override {
    return Env::Default()->LockFile(path);
  }

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, int> reads_ ABSL_GUARDED_BY(mu_);
  Hook hook_;
};

constexpr const char* kWords[] = {
    "Alpha",   "beta", "GAMMA", "delta_x", "Epsilon9", "zeta",   "eta",
    "Theta",   "iota", "kappa", "Lambda",  "mu",       "nu",     "Xi",
    "omicron", "pi",   "Rho",   "sigma",   "tau",      "Upsilon"};

std::string Lines(std::mt19937& rng, int count, const char* prefix) {
  std::string out;
  for (int i = 0; i < count; ++i) {
    const auto pick = [&] { return kWords[rng() % std::size(kWords)]; };
    absl::StrAppend(&out, "int ", prefix, pick(), i, "(int ", pick(),
                    ") { return ", pick(), "(", rng() % 1000, ") + $", pick(),
                    " + 0x", rng() % 100, "; }  // ", pick(),
                    " \xE2\x84\xAA\n");
  }
  return out;
}

// A corpus with every kind of file the build treats differently.
class Corpus {
 public:
  Corpus() {
    std::mt19937 rng(7);
    Write("src/a.cc", Lines(rng, 30, "A"));
    Write("src/b.cc", Lines(rng, 5, "B"));
    Write("src/big.cc", Lines(rng, 1200, "Big"));  // many chunks
    Write("src/deep/er/c.cc", Lines(rng, 60, "C"));
    Write("src/foo_test.cc", Lines(rng, 12, "T"));
    Write("third_party/z/z.c", Lines(rng, 12, "Z"));
    Write("src/gen.pb.cc", Lines(rng, 12, "G"));
    Write("src/oneline.js", std::string(30000, 'q') + "\nfunction f() {}\n");
    Write("src/long2.js", "var x = '" + std::string(1500, 'w') + "';\n");
    Write("README.md", "# Title\n\nSome text about Alpha and beta.\n");
    Write("docs/guide.md", "Guide\n=====\n\nUse the `Foo` class.\n");
    Write("tools/s.py",
          "def hello(name):\n    return name\n\nclass K:\n    pass\n");
    Write("tools/s.go", "package main\n\nfunc Hello() {}\n\ntype K struct{}\n");
    Write("tools/s.rs", "pub fn hello() {}\n\nstruct K;\n");
    Write("tools/s.java", "class K { void hello() {} }\n");
    Write("tools/s.ts", "export function hello(): void {}\nclass K {}\n");
    Write("tools/s.h", "#ifndef S_H_\n#define S_H_\nint Hello();\n#endif\n");
    // Lines of two folded bytes and bigrams with a capital.
    Write("docs/short.txt",
          "ab\nXY\nok\nAb\nQ\nzz\n\xC5\xBF"
          "a\nBc d\n");
    Write("single_line.txt", "x");  // kTooSmall only under kZoekt
    Write("data/empty.txt", "");
    Write("data/bin.dat", std::string("ab\0cd", 5));
    Write("data/huge.txt", std::string(300000, 'h'));
    std::string noise;
    for (int i = 0; i < 40000; ++i) {
      noise.push_back(static_cast<char>(' ' + rng() % 90));
    }
    Write("data/noise.txt", noise);
    for (const char* dir : {"dup/a", "dup/b", "dup/c"}) {
      Write(absl::StrCat(dir, "/LICENSE"), "License text, same everywhere.\n");
    }
    Write("x/same.py", "def same():\n    pass\n");
    Write("y/same.py", "def same():\n    pass\n");
    paths_.push_back("gone/missing.cc");  // listed but absent
    std::shuffle(paths_.begin(), paths_.end(), rng);
  }

  const std::string& root() const { return dir_.path(); }
  const std::vector<std::string>& paths() const { return paths_; }
  absl::Status Rewrite(const std::string& path, absl::string_view content) {
    return dir_.WriteFile(path, content);
  }

 private:
  void Write(const std::string& path, absl::string_view content) {
    CHECK_OK(dir_.WriteFile(path, content));
    paths_.push_back(path);
  }

  TempDir dir_;
  std::vector<std::string> paths_;
};

class BulkBuildEquivalenceTest : public ::testing::Test {
 protected:
  std::string OutPath(absl::string_view name) { return out_.Join(name); }

  std::string Bytes(const std::string& path) {
    absl::StatusOr<std::string> bytes = Env::Default()->ReadFile(path);
    CHECK_OK(bytes.status());
    return *std::move(bytes);
  }

  // Both builds must agree on the bytes and on every statistic.
  void ExpectSame(Env& env, const Corpus& corpus,
                  const BulkBuildOptions& reference_options,
                  const BulkBuildOptions& options, const std::string& label) {
    const std::string expected_path = OutPath("expected.fseg");
    const std::string actual_path = OutPath("actual.fseg");
    FERRET_ASSERT_OK_AND_ASSIGN(const BulkBuildStats expected,
                                BuildSegment(env, corpus.root(), corpus.paths(),
                                             expected_path, reference_options));
    FERRET_ASSERT_OK_AND_ASSIGN(
        const BulkBuildStats actual,
        BuildSegment(env, corpus.root(), corpus.paths(), actual_path, options));
    const std::string expected_bytes = Bytes(expected_path);
    const std::string actual_bytes = Bytes(actual_path);
    EXPECT_EQ(expected_bytes.size(), actual_bytes.size()) << label;
    EXPECT_TRUE(expected_bytes == actual_bytes) << label;
    EXPECT_EQ(expected.files_indexed, actual.files_indexed) << label;
    EXPECT_EQ(expected.files_unreadable, actual.files_unreadable) << label;
    EXPECT_EQ(expected.files_skipped, actual.files_skipped) << label;
    EXPECT_EQ(expected.bytes_indexed, actual.bytes_indexed) << label;
    EXPECT_EQ(expected.segment.num_docs, actual.segment.num_docs) << label;
    EXPECT_EQ(expected.segment.num_chunks, actual.segment.num_chunks) << label;
    EXPECT_EQ(expected.segment.num_trigrams, actual.segment.num_trigrams)
        << label;
    EXPECT_EQ(expected.segment.num_postings, actual.segment.num_postings)
        << label;
    EXPECT_EQ(expected.segment.file_bytes, actual.segment.file_bytes) << label;
    FERRET_ASSERT_OK_AND_ASSIGN(auto segment, Segment::Open(env, actual_path));
    ABSL_ASSERT_OK(segment->VerifyChecksums());
    // The scratch file is gone.
    FERRET_ASSERT_OK_AND_ASSIGN(const bool pack_exists,
                                env.FileExists(actual_path + ".pack"));
    EXPECT_FALSE(pack_exists) << label;
  }

  BulkBuildOptions Options(int threads, bool serial) {
    BulkBuildOptions options;
    options.num_threads = threads;
    options.serial = serial;
    options.eligibility.max_file_bytes = 200000;
    options.eligibility.max_distinct_trigrams = 3000;
    return options;
  }

  TempDir out_;
};

// Every thread count, unit size and window, for the default writer options.
TEST_F(BulkBuildEquivalenceTest, MatchesTheSerialBuildAtEveryShape) {
  const Corpus corpus;
  BulkBuildOptions reference = Options(1, /*serial=*/true);
  reference.writer.chunk_bytes = 300;
  for (const int threads : {1, 2, 3, 8}) {
    for (const int64_t partition_bytes :
         {int64_t{1}, int64_t{5000}, int64_t{1} << 30}) {
      for (const int window : {0, 1}) {
        BulkBuildOptions options = reference;
        options.serial = false;
        options.num_threads = threads;
        options.partition_bytes = partition_bytes;
        options.window = window;
        ExpectSame(*Env::Default(), corpus, reference, options,
                   absl::StrCat("threads=", threads, " partition_bytes=",
                                partition_bytes, " window=", window));
        if (::testing::Test::HasFailure()) return;
      }
    }
  }
}

// Symbols, code tokens, short-line trigrams and chunk size on and off.
TEST_F(BulkBuildEquivalenceTest, MatchesTheSerialBuildForEveryWriterOption) {
  const Corpus corpus;
  for (const bool symbols : {true, false}) {
    for (const bool code_tokens : {true, false}) {
      for (const bool short_lines : {true, false}) {
        for (const int64_t chunk_bytes : {int64_t{300}, int64_t{8192}}) {
          BulkBuildOptions reference = Options(1, /*serial=*/true);
          reference.writer.write_symbols = symbols;
          reference.writer.write_code_tokens = code_tokens;
          reference.writer.write_short_line_trigrams = short_lines;
          reference.writer.chunk_bytes = chunk_bytes;
          BulkBuildOptions options = reference;
          options.serial = false;
          options.num_threads = 3;
          options.partition_bytes = 5000;
          ExpectSame(*Env::Default(), corpus, reference, options,
                     absl::StrCat("symbols=", symbols, " tokens=", code_tokens,
                                  " short_lines=", short_lines,
                                  " chunk_bytes=", chunk_bytes));
          if (::testing::Test::HasFailure()) return;
        }
      }
    }
  }
}

// The scan and symbol workers, and the size of the sorted runs of postings
// (tiny runs put one trigram's list in many runs).
TEST_F(BulkBuildEquivalenceTest, MatchesTheSerialBuildForEveryWorkerShape) {
  const Corpus corpus;
  BulkBuildOptions reference = Options(1, /*serial=*/true);
  reference.writer.chunk_bytes = 300;
  for (const int scan_threads : {1, 3, 16}) {
    for (const int symbol_threads : {1, 5}) {
      for (const size_t run_postings :
           {size_t{1}, size_t{7}, size_t{1} << 24}) {
        for (const ThreadQos qos :
             {ThreadQos::kUserInteractive, ThreadQos::kUtility}) {
          BulkBuildOptions options = reference;
          options.serial = false;
          options.num_threads = 2;
          options.scan_threads = scan_threads;
          options.symbol_threads = symbol_threads;
          options.symbol_qos = qos;
          options.writer.max_partition_postings = run_postings;
          ExpectSame(
              *Env::Default(), corpus, reference, options,
              absl::StrCat("scan=", scan_threads, " symbols=", symbol_threads,
                           " run_postings=", run_postings));
          if (::testing::Test::HasFailure()) return;
        }
      }
    }
  }
}

TEST_F(BulkBuildEquivalenceTest, SerialBuildIsDeterministicAcrossThreads) {
  const Corpus corpus;
  ExpectSame(*Env::Default(), corpus, Options(1, true), Options(8, true),
             "serial 1 vs 8");
}

TEST_F(BulkBuildEquivalenceTest, ZoektPolicyMatches) {
  const Corpus corpus;
  BulkBuildOptions reference = Options(1, true);
  reference.eligibility.policy = EligibilityPolicy::kZoekt;
  BulkBuildOptions options = reference;
  options.serial = false;
  options.num_threads = 4;
  options.partition_bytes = 20000;
  ExpectSame(*Env::Default(), corpus, reference, options, "zoekt");
}

TEST_F(BulkBuildEquivalenceTest, EmptyAndFullySkippedInputs) {
  const Corpus corpus;
  for (const bool serial : {true, false}) {
    FERRET_ASSERT_OK_AND_ASSIGN(
        const BulkBuildStats stats,
        BuildSegment(*Env::Default(), corpus.root(), {}, OutPath("none.fseg"),
                     Options(3, serial)));
    EXPECT_EQ(stats.files_indexed, 0);
    FERRET_ASSERT_OK_AND_ASSIGN(
        auto segment, Segment::Open(*Env::Default(), OutPath("none.fseg")));
    EXPECT_EQ(segment->num_docs(), 0u);
  }
  const std::vector<std::string> skipped = {"data/empty.txt", "data/bin.dat",
                                            "data/huge.txt", "gone/missing.cc"};
  FERRET_ASSERT_OK_AND_ASSIGN(
      const BulkBuildStats a,
      BuildSegment(*Env::Default(), corpus.root(), skipped, OutPath("a.fseg"),
                   Options(1, true)));
  FERRET_ASSERT_OK_AND_ASSIGN(
      const BulkBuildStats b,
      BuildSegment(*Env::Default(), corpus.root(), skipped, OutPath("b.fseg"),
                   Options(4, false)));
  EXPECT_EQ(a.files_indexed, 0);
  EXPECT_EQ(b.files_indexed, 0);
  EXPECT_EQ(a.files_unreadable, 1);
  EXPECT_EQ(b.files_unreadable, 1);
  EXPECT_EQ(Bytes(OutPath("a.fseg")), Bytes(OutPath("b.fseg")));
}

// The default build opens each file once: reading is the cost on a large
// tree. The serial build reads indexed files twice.
TEST_F(BulkBuildEquivalenceTest, DefaultBuildReadsEachFileOnce) {
  const Corpus corpus;
  CountingEnv env;
  FERRET_ASSERT_OK_AND_ASSIGN(
      const BulkBuildStats stats,
      BuildSegment(env, corpus.root(), corpus.paths(), OutPath("once.fseg"),
                   Options(4, false)));
  ASSERT_GT(stats.files_indexed, 10);
  for (const std::string& path : corpus.paths()) {
    EXPECT_LE(env.reads(absl::StrCat(corpus.root(), "/", path)), 1) << path;
  }
  env.ResetReads();
  FERRET_ASSERT_OK_AND_ASSIGN(
      const BulkBuildStats serial,
      BuildSegment(env, corpus.root(), corpus.paths(), OutPath("twice.fseg"),
                   Options(4, true)));
  EXPECT_EQ(env.reads(absl::StrCat(corpus.root(), "/src/a.cc")), 2);
  EXPECT_EQ(serial.files_indexed, stats.files_indexed);
}

// The serial build counts a file that changes between its two reads as
// unreadable. The default build has no second read, so the file is indexed as
// the first read saw it.
TEST_F(BulkBuildEquivalenceTest, FileThatChangesBetweenReads) {
  const Corpus corpus;
  CountingEnv env;
  const std::string changing = absl::StrCat(corpus.root(), "/src/a.cc");
  env.set_hook(
      [&](const std::string& path, int number) -> std::optional<std::string> {
        if (path == changing && number >= 2) return std::string("changed\n");
        return std::nullopt;
      });
  FERRET_ASSERT_OK_AND_ASSIGN(
      const BulkBuildStats serial,
      BuildSegment(env, corpus.root(), corpus.paths(), OutPath("s.fseg"),
                   Options(3, true)));
  env.ResetReads();
  FERRET_ASSERT_OK_AND_ASSIGN(
      const BulkBuildStats fresh,
      BuildSegment(env, corpus.root(), corpus.paths(), OutPath("d.fseg"),
                   Options(3, false)));
  // Missing file + the changed file.
  EXPECT_EQ(serial.files_unreadable, 2);
  EXPECT_EQ(fresh.files_unreadable, 1);
  EXPECT_EQ(fresh.files_indexed, serial.files_indexed + 1);
  FERRET_ASSERT_OK_AND_ASSIGN(
      auto segment, Segment::Open(*Env::Default(), OutPath("d.fseg")));
  bool found = false;
  for (DocId d = 0; d < segment->num_docs(); ++d) {
    if (segment->doc(d).hint_path == "src/a.cc") {
      found = true;
      EXPECT_NE(segment->doc(d).content, "changed\n");
    }
  }
  EXPECT_TRUE(found);
  // The same as a build of the unchanged tree.
  CountingEnv plain;
  ABSL_ASSERT_OK(BuildSegment(plain, corpus.root(), corpus.paths(),
                              OutPath("p.fseg"), Options(3, false))
                     .status());
  EXPECT_EQ(Bytes(OutPath("d.fseg")), Bytes(OutPath("p.fseg")));
}

TEST_F(BulkBuildEquivalenceTest, RejectsBadOptions) {
  const Corpus corpus;
  BulkBuildOptions options = Options(2, false);
  options.partition_bytes = 0;
  EXPECT_THAT(BuildSegment(*Env::Default(), corpus.root(), corpus.paths(),
                           OutPath("x.fseg"), options)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  options = Options(2, false);
  options.window = -1;
  EXPECT_THAT(BuildSegment(*Env::Default(), corpus.root(), corpus.paths(),
                           OutPath("x.fseg"), options)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace ferret

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/workspace/view_state.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"

#include "ferret/git/blob_hash.h"
#include "ferret/git/oid.h"
#include "ferret/index/blob_registry.h"
#include "ferret/testing/test_seed.h"
#include "ferret/text/literal_finder.h"
#include "ferret/workspace/path_table.h"

namespace ferret {
namespace {

using Model = absl::btree_map<PathId, Oid>;

void Compare(const ViewState& state, const Model& model,
             const BlobRegistry& blobs) {
  for (size_t i = 0; i <= blobs.size(); ++i) {
    const BlobId id = static_cast<BlobId>(i);
    std::vector<PathId> expected;
    if (i < blobs.size()) {
      for (const auto& [path, oid] : model) {
        if (oid == blobs.oid(id)) expected.push_back(path);
      }
    }
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(state.Contains(id), !expected.empty());
    EXPECT_EQ(state.PathsFor(id), expected);
  }
}

TEST(ViewStateTest, SeededBatchesAndIndependentWorktrees) {
  const uint64_t seed = ferret_testing::TestSeed(7221);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  BlobRegistry registry;
  PathTable paths;
  std::vector<Oid> oids;
  for (int i = 0; i < 18; ++i) {
    oids.push_back(HashBlob(absl::StrCat(i)));
    registry.Intern(oids.back());
  }
  std::vector<PathId> names;
  for (int i = 0; i < 60; ++i) {
    auto id = paths.Intern(absl::StrCat("dir/", i));
    ASSERT_TRUE(id.ok());
    names.push_back(*id);
  }
  ViewWriter writers[2];
  Model models[2];
  std::vector<std::pair<std::shared_ptr<const ViewState>, Model>> history[2];
  for (int batch = 0; batch < 150; ++batch) {
    const int worktree = batch % 2;
    std::vector<ViewEdit> edits;
    for (int i = 0; i < 1 + batch % 16; ++i) {
      const PathId path = names[rng() % names.size()];
      const int blob = static_cast<int>(rng() % 22);
      if (blob >= static_cast<int>(oids.size())) {
        models[worktree].erase(path);
        edits.push_back({path, std::nullopt});
      } else {
        const BlobId id = registry.Intern(oids[blob]);
        models[worktree][path] = oids[blob];
        edits.push_back({path, id});
      }
    }
    auto snapshot = writers[worktree].ApplyBatch(edits);
    EXPECT_EQ(snapshot, writers[worktree].Snapshot());
    Compare(*snapshot, models[worktree], registry);
    history[worktree].emplace_back(snapshot, models[worktree]);
    for (int w = 0; w < 2; ++w) {
      for (const auto& [state, model] : history[w]) {
        Compare(*state, model, registry);
      }
    }
  }
  const BlobId newer = registry.Intern(HashBlob("after snapshot"));
  EXPECT_FALSE(writers[0].Snapshot()->Contains(newer));
  EXPECT_TRUE(writers[0].Snapshot()->PathsFor(newer).empty());
}

TEST(ViewStateTest, OneBlobAtManyPathsSharesUntouchedNodes) {
  ViewWriter writer;
  std::vector<ViewEdit> files;
  files.reserve(10000);
  for (uint32_t i = 0; i < 10000; ++i) {
    files.push_back({static_cast<PathId>(i), BlobId{0}});
  }
  const auto before = writer.ApplyBatch(files);
  const auto after = writer.ApplyBatch({ViewEdit{PathId{0}, std::nullopt}});
  EXPECT_EQ(before->PathsFor(BlobId{0}).size(), size_t{10000});
  EXPECT_EQ(after->PathsFor(BlobId{0}).size(), size_t{9999});
  EXPECT_EQ(before->PathsFor(BlobId{0}).front(), PathId{0});
  EXPECT_EQ(after->PathsFor(BlobId{0}).front(), PathId{1});
}

// A copied chunk must discard cached names; untouched chunks remain valid.
// Literal hits must not straddle two adjacent basenames in the packed buffer.
TEST(ViewStateTest, PackedNamesFollowCopiedChunks) {
  PathTable paths;
  ViewWriter writer;
  auto intern = [&](absl::string_view path) {
    auto id = paths.Intern(path);
    EXPECT_TRUE(id.ok());
    return id.value();
  };
  const PathId left = intern("dir/ab"), right = intern("dir/cd");
  const PathId folded = intern("other/NEEDLE.cc");
  const PathId untouched = intern("other/needle.h");
  const PathId replacement = intern("dir/needle.rs");
  const auto first = writer.ApplyBatch(
      {ViewEdit{left, BlobId{0}}, ViewEdit{right, BlobId{0}},
       ViewEdit{folded, BlobId{1}}, ViewEdit{untouched, BlobId{256}}});
  auto find = [&](const ViewState& state, absl::string_view literal,
                  LiteralFinder::Mode mode) {
    std::vector<std::pair<BlobId, PathId>> found;
    state.ForEachBasenameContaining(
        literal, mode, paths,
        [&](BlobId blob, PathId path) { found.emplace_back(blob, path); });
    return found;
  };
  const auto folded_mode = LiteralFinder::Mode::kAsciiCaseInsensitive;
  EXPECT_TRUE(find(*first, "bc", folded_mode).empty());
  EXPECT_EQ(find(*first, "needle", folded_mode),
            (std::vector<std::pair<BlobId, PathId>>{{BlobId{1}, folded},
                                                    {BlobId{256}, untouched}}));
  const auto second = writer.ApplyBatch(
      {ViewEdit{folded, std::nullopt}, ViewEdit{replacement, BlobId{1}}});
  EXPECT_EQ(find(*first, "needle", LiteralFinder::Mode::kExact),
            (std::vector<std::pair<BlobId, PathId>>{{BlobId{256}, untouched}}));
  EXPECT_EQ(find(*second, "needle", folded_mode),
            (std::vector<std::pair<BlobId, PathId>>{{BlobId{1}, replacement},
                                                    {BlobId{256}, untouched}}));
  EXPECT_EQ(find(*second, "", folded_mode).size(), size_t{4});
  std::atomic<bool> ok{true};
  std::vector<std::thread> readers;
  readers.reserve(8);
  for (int i = 0; i < 8; ++i) {
    readers.emplace_back([&] {
      for (int j = 0; j < 100; ++j) {
        if (find(*second, "needle", folded_mode).size() != 2) ok = false;
      }
    });
  }
  for (auto& reader : readers) reader.join();
  EXPECT_TRUE(ok);
}

TEST(ViewStateTest, ReadersRetainCompleteBatches) {
  ViewWriter writer;
  const BlobId blob{0};
  const PathId first{0}, second{1};
  std::atomic<bool> done{false};
  std::thread reader([&] {
    do {
      auto state = writer.Snapshot();
      const auto paths = state->PathsFor(blob);
      EXPECT_EQ(state->Contains(blob), paths.size() == 2);
      EXPECT_TRUE(paths.empty() || (paths.size() == 2 && paths[0] == first &&
                                    paths[1] == second));
    } while (!done.load(std::memory_order_acquire));
  });
  for (int i = 0; i < 200; ++i) {
    if (i % 2 == 0) {
      writer.ApplyBatch({ViewEdit{first, blob}, ViewEdit{second, blob}});
    } else {
      writer.ApplyBatch(
          {ViewEdit{first, std::nullopt}, ViewEdit{second, std::nullopt}});
    }
  }
  done.store(true, std::memory_order_release);
  reader.join();
}

TEST(ViewStateTest, DuplicatePathsAndMoves) {
  ViewWriter writer;
  const BlobId a{0}, b{1};
  const PathId x{0}, y{1};
  auto first = writer.ApplyBatch({ViewEdit{x, a}, ViewEdit{y, a}});
  auto second = writer.ApplyBatch({ViewEdit{x, b}});
  EXPECT_TRUE(first->Contains(a));
  EXPECT_FALSE(first->Contains(b));
  EXPECT_EQ(first->PathsFor(a).size(), size_t{2});
  EXPECT_EQ(second->PathsFor(a).size(), size_t{1});
  EXPECT_EQ(second->PathsFor(b).size(), size_t{1});
  auto third = writer.ApplyBatch(
      {ViewEdit{y, std::nullopt}, ViewEdit{x, a}, ViewEdit{x, std::nullopt}});
  EXPECT_FALSE(third->Contains(a));
  EXPECT_FALSE(third->Contains(b));
  EXPECT_EQ(second->PathsFor(a).size(), size_t{1});
}

TEST(ViewStateTest, ReconcileReportsOnlyMembershipChangesOncePerBatch) {
  ViewWriter writer;
  const BlobId a{0}, b{1};
  const PathId x{0}, y{1}, z{2};
  int batches = 0;
  auto reconcile = [&](absl::Span<const BlobId> entered,
                       absl::Span<const BlobId> left,
                       absl::FunctionRef<void()> publish) {
    ++batches;
    const auto previous = writer.Snapshot();
    for (BlobId blob : entered) EXPECT_FALSE(previous->Contains(blob));
    for (BlobId blob : left) EXPECT_TRUE(previous->Contains(blob));
    publish();
    const auto next = writer.Snapshot();
    for (BlobId blob : entered) EXPECT_TRUE(next->Contains(blob));
    for (BlobId blob : left) EXPECT_FALSE(next->Contains(blob));
  };
  writer.ApplyBatch({ViewEdit{x, a}, ViewEdit{y, a}}, reconcile);
  EXPECT_EQ(batches, 1);
  writer.ApplyBatch(
      {ViewEdit{x, std::nullopt}, ViewEdit{z, a}},
      [&](absl::Span<const BlobId> entered, absl::Span<const BlobId> left,
          absl::FunctionRef<void()> publish) {
        EXPECT_TRUE(entered.empty());
        EXPECT_TRUE(left.empty());
        reconcile(entered, left, publish);
      });
  EXPECT_EQ(batches, 2);
  writer.ApplyBatch(
      {ViewEdit{y, b}, ViewEdit{z, std::nullopt}},
      [&](absl::Span<const BlobId> entered, absl::Span<const BlobId> left,
          absl::FunctionRef<void()> publish) {
        EXPECT_EQ(entered, (std::vector<BlobId>{b}));
        EXPECT_EQ(left, (std::vector<BlobId>{a}));
        reconcile(entered, left, publish);
      });
  EXPECT_EQ(batches, 3);
}

// A view whose blobs sit in two distant ranges has no chunk between them. A
// search asks about every blob of the index, including the ones in the gap.
TEST(ViewStateTest, BlobsInAnUnallocatedChunkAreAbsent) {
  PathTable paths;
  auto first = paths.Intern("a/first.cc");
  auto second = paths.Intern("b/second.cc");
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  const BlobId low = static_cast<BlobId>(1);
  const BlobId high = static_cast<BlobId>(3 * ViewState::kChunkSize + 1);
  ViewWriter writer;
  const auto state =
      writer.ApplyBatch({ViewEdit{*first, low}, ViewEdit{*second, high}});
  const BlobId gap = static_cast<BlobId>(ViewState::kChunkSize + 7);
  EXPECT_TRUE(state->Contains(low));
  EXPECT_TRUE(state->Contains(high));
  EXPECT_FALSE(state->Contains(gap));
  EXPECT_TRUE(state->PathsFor(gap).empty());
  int visited = 0;
  state->ForEachBasenameContaining("", LiteralFinder::Mode::kExact, paths,
                                   [&visited](BlobId, PathId) { ++visited; });
  EXPECT_EQ(visited, 2);
  // Editing a blob in the gap allocates its chunk without disturbing others.
  const auto next = writer.ApplyBatch({ViewEdit{*first, gap}});
  EXPECT_TRUE(next->Contains(gap));
  EXPECT_FALSE(next->Contains(low));
  EXPECT_TRUE(next->Contains(high));
}

// PathCount is the size of PathsFor, and the ranges of a partition of the
// directory chunks visit exactly the names that one call over all of them
// does, each with its basename.
TEST(ViewStateTest, PathCountAndNameRangesAgreeWithTheWholeScan) {
  const uint64_t seed = ferret_testing::TestSeed(0x9a7c);
  SCOPED_TRACE(seed);
  std::mt19937_64 rng(seed);
  PathTable paths;
  ViewWriter writer;
  std::vector<ViewEdit> edits;
  for (int i = 0; i < 900; ++i) {
    const BlobId blob =
        static_cast<BlobId>(rng() % (5 * ViewState::kChunkSize));
    // Several paths for some blobs, and names that share the literal.
    for (int copy = 0; copy < (rng() % 4 == 0 ? 3 : 1); ++copy) {
      auto path = paths.Intern(absl::StrCat("d", rng() % 7, "/",
                                            rng() % 3 == 0 ? "needle" : "other",
                                            i, "_", copy, ".cc"));
      ASSERT_TRUE(path.ok());
      edits.push_back({*path, blob});
    }
  }
  const auto state = writer.ApplyBatch(edits);
  for (size_t b = 0; b < 5 * ViewState::kChunkSize + 3; ++b) {
    const BlobId blob = static_cast<BlobId>(b);
    EXPECT_EQ(state->PathCount(blob), state->PathsFor(blob).size()) << b;
  }
  using Hit = std::tuple<BlobId, PathId, std::string>;
  for (const char* literal : {"needle", "NEEDLE", "", "_1.cc"}) {
    SCOPED_TRACE(literal);
    auto collect = [&](size_t first, size_t last) {
      std::vector<Hit> hits;
      state->ForEachNameContaining(
          literal, LiteralFinder::Mode::kAsciiCaseInsensitive, paths,
          [&](BlobId blob, PathId path, absl::string_view basename) {
            hits.emplace_back(blob, path, std::string(basename));
          },
          first, last);
      return hits;
    };
    const std::vector<Hit> whole = collect(0, state->num_chunks());
    EXPECT_FALSE(whole.empty());
    for (const auto& [blob, path, basename] : whole) {
      const std::string full = paths.path(path);
      EXPECT_EQ(basename, full.substr(full.rfind('/') + 1));
    }
    std::vector<Hit> parts;
    for (size_t first = 0; first < state->num_chunks(); first += 2) {
      const std::vector<Hit> part = collect(first, first + 2);
      parts.insert(parts.end(), part.begin(), part.end());
    }
    EXPECT_EQ(parts, whole);
    EXPECT_TRUE(collect(state->num_chunks(), state->num_chunks() + 5).empty());
  }
}

}  // namespace
}  // namespace ferret

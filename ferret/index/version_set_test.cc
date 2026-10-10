// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/version_set.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/base/mapped_file.h"
#include "ferret/git/blob_hash.h"
#include "ferret/index/manifest.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/reaper.h"
#include "ferret/index/segment_writer.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"

namespace ferret {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::ferret_testing::TempDir;

absl::Status WriteDoc(Env& env, const std::string& path,
                      const std::string& name, const std::string& content) {
  auto writer = SegmentWriter::Create(env, path);
  if (!writer.ok()) return writer.status();
  absl::Status status =
      (*writer)->Add({.hint_path = name,
                      .content = content,
                      .rank_key = ComputeRankKey(name, content.size(), false,
                                                 HashBlob(content))});
  if (!status.ok()) return status;
  auto stats = (*writer)->Finish();
  return stats.ok() ? absl::OkStatus() : stats.status();
}

// Deletes through the default Env but reports NotFound, as if another
// deleter (an earlier VersionSet's reaper) removed the file first.
class LosesDeleteRacesEnv final : public Env {
 public:
  absl::StatusOr<std::unique_ptr<WritableFile>> NewWritableFile(
      absl::string_view path) override {
    return base_.NewWritableFile(path);
  }
  absl::StatusOr<std::string> ReadFile(absl::string_view path) override {
    return base_.ReadFile(path);
  }
  absl::StatusOr<std::unique_ptr<MappedFile>> NewMappedFile(
      absl::string_view path) override {
    return base_.NewMappedFile(path);
  }
  absl::StatusOr<int64_t> GetFileSize(absl::string_view path) override {
    return base_.GetFileSize(path);
  }
  absl::StatusOr<bool> FileExists(absl::string_view path) override {
    return base_.FileExists(path);
  }
  absl::StatusOr<std::vector<std::string>> GetChildren(
      absl::string_view path) override {
    return base_.GetChildren(path);
  }
  absl::Status CreateDir(absl::string_view path) override {
    return base_.CreateDir(path);
  }
  absl::Status DeleteFile(absl::string_view path) override {
    absl::Status status = base_.DeleteFile(path);
    if (!status.ok()) return status;
    return absl::NotFoundError(path);
  }
  absl::Status RenameFile(absl::string_view from,
                          absl::string_view to) override {
    return base_.RenameFile(from, to);
  }
  absl::Status SyncDir(absl::string_view path) override {
    return base_.SyncDir(path);
  }
  absl::StatusOr<std::unique_ptr<FileLock>> LockFile(
      absl::string_view path) override {
    return base_.LockFile(path);
  }

 private:
  Env& base_ = *Env::Default();
};

TEST(VersionSetTest, InstallsRemovesAndPersistsViewsAndNumbers) {
  TempDir dir;
  Env& env = *Env::Default();
  Reaper reaper;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              VersionSet::Open(env, dir.Join("index"), reaper));
  auto& set = opened.first;
  FERRET_ASSERT_OK_AND_ASSIGN(std::string a, set->NewSegmentFile());
  EXPECT_EQ(a, dir.Join("index/000001.fseg.tmp"));
  ASSERT_THAT(WriteDoc(env, a, "a.cc", "hello A\n"), IsOk());
  EXPECT_THAT(
      set->Install(
          {.add = {a}, .views = std::vector<ManifestView>{{"/root", "/git"}}}),
      IsOk());
  auto old = set->current();
  ASSERT_EQ(old->segments.size(), 1u);
  EXPECT_EQ(old->segments[0].segment->doc(0).content, "hello A\n");
  FERRET_ASSERT_OK_AND_ASSIGN(std::string b, set->NewSegmentFile());
  ASSERT_THAT(WriteDoc(env, b, "b.cc", "hello B\n"), IsOk());
  EXPECT_THAT(set->Install({.add = {b}, .remove = {1}}), IsOk());
  EXPECT_EQ(set->current()->views,
            (std::vector<ManifestView>{{"/root", "/git"}}));
  FERRET_ASSERT_OK_AND_ASSIGN(bool exists,
                              env.FileExists(dir.Join("index/000001.fseg")));
  EXPECT_TRUE(exists);
  EXPECT_EQ(old->segments[0].segment->doc(0).hint_path, "a.cc");
  auto held_segment = old->segments[0].segment;
  old.reset();
  // Reaper work is ordered behind this barrier task.
  EXPECT_THAT(env.FileExists(dir.Join("index/000001.fseg")),
              absl_testing::IsOkAndHolds(true));
  held_segment.reset();
  // The barrier waits for any destruction task submitted by the last release.
  absl::Notification done;
  reaper.Submit([&done] { done.Notify(); });
  done.WaitForNotification();
  EXPECT_THAT(env.FileExists(dir.Join("index/000001.fseg")),
              IsOkAndHolds(false));
  set.reset();
  FERRET_ASSERT_OK_AND_ASSIGN(auto reopened,
                              VersionSet::Open(env, dir.Join("index"), reaper));
  EXPECT_TRUE(reopened.second.deleted_files.empty());
  EXPECT_EQ(reopened.first->current()->views,
            (std::vector<ManifestView>{{"/root", "/git"}}));
  FERRET_ASSERT_OK_AND_ASSIGN(std::string c, reopened.first->NewSegmentFile());
  EXPECT_EQ(c, dir.Join("index/000003.fseg.tmp"));
}

TEST(VersionSetTest, DiscardedReservationCannotBeInstalled) {
  TempDir dir;
  Reaper reaper;
  Env& env = *Env::Default();
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              VersionSet::Open(env, dir.Join("index"), reaper));
  auto& set = opened.first;
  FERRET_ASSERT_OK_AND_ASSIGN(std::string path, set->NewSegmentFile());
  ASSERT_THAT(WriteDoc(env, path, "discard.cc", "discard\n"), IsOk());
  set->DiscardSegmentFile(path);
  EXPECT_THAT(set->Install({.add = {path}}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(set->current()->segments.size(), 0u);
  ASSERT_THAT(env.DeleteFile(path), IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(std::string next, set->NewSegmentFile());
  EXPECT_EQ(next, dir.Join("index/000002.fseg.tmp"));
}

TEST(VersionSetTest, HoldsExclusiveLock) {
  TempDir dir;
  Reaper reaper;
  Env& env = *Env::Default();
  FERRET_ASSERT_OK_AND_ASSIGN(auto first,
                              VersionSet::Open(env, dir.Join("index"), reaper));
  EXPECT_THAT(VersionSet::Open(env, dir.Join("index"), reaper).status(),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
  first.first.reset();
  EXPECT_THAT(VersionSet::Open(env, dir.Join("index"), reaper),
              absl_testing::IsOk());
}

TEST(VersionSetTest, RecoveryDeletesOrphansAndCorrectsManifestWithOldTemp) {
  TempDir dir;
  Reaper reaper;
  Env& env = *Env::Default();
  const std::string index = dir.Join("index");
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              VersionSet::Open(env, index, reaper));
  FERRET_ASSERT_OK_AND_ASSIGN(std::string path, opened.first->NewSegmentFile());
  ASSERT_THAT(WriteDoc(env, path, "keep.cc", "keep\n"), IsOk());
  ASSERT_THAT(opened.first->Install({.add = {path}}), IsOk());
  opened.first.reset();
  // An interrupted commit can leave both a corrupt listed segment and a
  // MANIFEST.tmp that correction will overwrite and consume.
  FERRET_ASSERT_OK_AND_ASSIGN(auto temp,
                              env.NewWritableFile(index + "/MANIFEST.tmp"));
  ASSERT_THAT(temp->Append("incomplete"), IsOk());
  ASSERT_THAT(temp->Close(), IsOk());
  ASSERT_THAT(env.RenameFile(index + "/000001.fseg", index + "/000009.fseg"),
              IsOk());
  ASSERT_THAT(
      WriteDoc(env, index + "/000008.fseg.tmp", "orphan.cc", "orphan\n"),
      IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(auto recovered,
                              VersionSet::Open(env, index, reaper));
  EXPECT_EQ(recovered.second.dropped_segments.size(), 1u);
  EXPECT_EQ(recovered.second.deleted_files.size(), 2u);
  EXPECT_TRUE(recovered.first->current()->segments.empty());
  recovered.first.reset();
  FERRET_ASSERT_OK_AND_ASSIGN(auto second,
                              VersionSet::Open(env, index, reaper));
  EXPECT_TRUE(second.second.deleted_files.empty());
  EXPECT_TRUE(second.second.dropped_segments.empty());
}

TEST(VersionSetTest, RecoveryToleratesOrphansDeletedConcurrently) {
  TempDir dir;
  Reaper reaper;
  LosesDeleteRacesEnv env;
  const std::string index = dir.Join("index");
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              VersionSet::Open(env, index, reaper));
  opened.first.reset();
  ASSERT_THAT(WriteDoc(env, index + "/000007.fseg", "orphan.cc", "orphan\n"),
              IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(auto recovered,
                              VersionSet::Open(env, index, reaper));
  EXPECT_TRUE(recovered.second.deleted_files.empty());
  EXPECT_THAT(env.FileExists(index + "/000007.fseg"), IsOkAndHolds(false));
}

TEST(VersionSetTest, RejectsUnallocatedOrReusedSegmentFiles) {
  TempDir dir;
  Reaper reaper;
  Env& env = *Env::Default();
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              VersionSet::Open(env, dir.Join("index"), reaper));
  FERRET_ASSERT_OK_AND_ASSIGN(std::string path, opened.first->NewSegmentFile());
  ASSERT_THAT(WriteDoc(env, path, "a.cc", "a\n"), IsOk());
  ASSERT_THAT(opened.first->Install({.add = {path}}), IsOk());
  EXPECT_THAT(opened.first->Install({.add = {path}}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      opened.first->Install({.add = {dir.Join("index/000999.fseg.tmp")}}),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(opened.first->current()->segments.size(), 1u);
}

TEST(VersionSetTest, NumbersBeyondSixDigitsAndUint32) {
  TempDir dir;
  Env& env = *Env::Default();
  const std::string index = dir.Join("index");
  ASSERT_THAT(env.CreateDir(index), IsOk());
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::string manifest,
      EncodeManifest(Manifest{.next_file_number = uint64_t{1} << 32}));
  FERRET_ASSERT_OK_AND_ASSIGN(auto file,
                              env.NewWritableFile(index + "/MANIFEST"));
  ASSERT_THAT(file->Append(manifest), IsOk());
  ASSERT_THAT(file->Close(), IsOk());
  Reaper reaper;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              VersionSet::Open(env, index, reaper));
  FERRET_ASSERT_OK_AND_ASSIGN(std::string path, opened.first->NewSegmentFile());
  EXPECT_EQ(path, index + "/4294967296.fseg.tmp");
}

// The inverse table lists each blob's documents in DocId order, for dense,
// sparse and repeated BlobIds, and nothing for any other blob.
TEST(DocsOfBlobTest, MatchesABruteForceInverse) {
  std::mt19937_64 rng(0x0d0c);
  struct Shape {
    uint32_t docs;
    uint32_t blobs;
    uint32_t base;
    uint32_t stride;
  };
  // Dense with copies; dense with an offset; sparse (a large stride); empty;
  // one document.
  const Shape shapes[] = {{200, 60, 0, 1}, {200, 200, 5000, 1},
                          {150, 90, 7, 1}, {120, 100, 3, 100000},
                          {0, 1, 0, 1},    {1, 1, 9, 1}};
  for (const Shape& shape : shapes) {
    SCOPED_TRACE(shape.docs);
    std::vector<BlobId> blob_of_doc;
    for (uint32_t doc = 0; doc < shape.docs; ++doc) {
      blob_of_doc.push_back(static_cast<BlobId>(
          shape.base + shape.stride * (rng() % shape.blobs)));
    }
    const DocsOfBlob inverse(blob_of_doc);
    // Every blob present, its neighbours, and ids far outside the range.
    std::vector<uint64_t> probes = {
        0, 1, uint64_t{shape.base} + 7,
        uint64_t{shape.base} + shape.stride * shape.blobs + 3,
        uint64_t{0xfffffff0}};
    for (const BlobId blob : blob_of_doc) {
      const uint64_t id = static_cast<uint32_t>(blob);
      probes.insert(probes.end(), {id, id + 1, id > 0 ? id - 1 : 0});
    }
    for (const uint64_t probe : probes) {
      const BlobId blob = static_cast<BlobId>(probe);
      std::vector<DocId> expected;
      for (DocId doc = 0; doc < blob_of_doc.size(); ++doc) {
        if (blob_of_doc[doc] == blob) expected.push_back(doc);
      }
      const auto docs = inverse.Docs(blob);
      EXPECT_EQ(std::vector<DocId>(docs.begin(), docs.end()), expected)
          << probe;
    }
    if (shape.docs > 1) EXPECT_EQ(inverse.dense(), shape.stride == 1);
  }
}

}  // namespace
}  // namespace ferret

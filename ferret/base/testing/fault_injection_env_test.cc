// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/base/testing/fault_injection_env.h"

#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"

namespace ferret {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::ferret_testing::FaultInjectionEnv;
using ::ferret_testing::TempDir;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Not;

constexpr absl::string_view kMissing = "<missing>";

// Returns the content of `path` as the base file system holds it, or
// kMissing.
std::string ReadOrMissing(const std::string& path) {
  absl::StatusOr<std::string> content = Env::Default()->ReadFile(path);
  if (absl::IsNotFound(content.status())) return std::string(kMissing);
  EXPECT_THAT(content, IsOk());
  return content.value_or("<error>");
}

// Writes `content` to `path` through `env`, syncing it if `sync`.
absl::Status WriteThrough(Env& env, const std::string& path,
                          absl::string_view content, bool sync) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<WritableFile> file,
                        env.NewWritableFile(path));
  ABSL_RETURN_IF_ERROR(file->Append(content));
  if (sync) ABSL_RETURN_IF_ERROR(file->Sync());
  return file->Close();
}

class FaultInjectionEnvTest : public ::testing::Test {
 protected:
  std::string Path(absl::string_view name) const { return dir_.Join(name); }
  std::string Read(absl::string_view name) const {
    return ReadOrMissing(Path(name));
  }

  TempDir dir_;
  FaultInjectionEnv env_{Env::Default()};
};

TEST_F(FaultInjectionEnvTest, KeepsSyncedDataAndDropsLaterWrites) {
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WritableFile> file,
                              env_.NewWritableFile(Path("f")));
  ABSL_ASSERT_OK(file->Append("abc"));
  ABSL_ASSERT_OK(file->Sync());
  ABSL_ASSERT_OK(file->Append("def"));
  ABSL_ASSERT_OK(file->Close());
  ABSL_ASSERT_OK(env_.SyncDir(dir_.path()));
  EXPECT_EQ(Read("f"), "abcdef");
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_EQ(Read("f"), "abc");
}

TEST_F(FaultInjectionEnvTest, NeverSyncedFileLosesAllData) {
  ABSL_ASSERT_OK(WriteThrough(env_, Path("f"), "data", /*sync=*/false));
  ABSL_ASSERT_OK(env_.SyncDir(dir_.path()));
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_EQ(Read("f"), "");
}

TEST_F(FaultInjectionEnvTest, FullSyncCountsAsSync) {
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WritableFile> file,
                              env_.NewWritableFile(Path("f")));
  ABSL_ASSERT_OK(file->Append("abc"));
  ABSL_ASSERT_OK(file->FullSync());
  ABSL_ASSERT_OK(file->Append("def"));
  ABSL_ASSERT_OK(file->Close());
  ABSL_ASSERT_OK(env_.SyncDir(dir_.path()));
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_EQ(Read("f"), "abc");
}

TEST_F(FaultInjectionEnvTest, UnsyncedOverwriteOfSyncedBytesIsRolledBack) {
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WritableFile> file,
                              env_.NewWritableFile(Path("f")));
  ABSL_ASSERT_OK(file->Append("0123456789"));
  ABSL_ASSERT_OK(file->Sync());
  ABSL_ASSERT_OK(file->WriteAt(2, "xx"));
  ABSL_ASSERT_OK(file->Append("tail"));
  ABSL_ASSERT_OK(file->Close());
  ABSL_ASSERT_OK(env_.SyncDir(dir_.path()));
  EXPECT_EQ(Read("f"), "01xx456789tail");
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_EQ(Read("f"), "0123456789");
}

TEST_F(FaultInjectionEnvTest, SyncedOverwriteIsDurable) {
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WritableFile> file,
                              env_.NewWritableFile(Path("f")));
  ABSL_ASSERT_OK(file->Append("header...body"));
  ABSL_ASSERT_OK(file->Sync());
  ABSL_ASSERT_OK(file->WriteAt(0, "HEADER"));
  ABSL_ASSERT_OK(file->Sync());
  ABSL_ASSERT_OK(file->Close());
  ABSL_ASSERT_OK(env_.SyncDir(dir_.path()));
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_EQ(Read("f"), "HEADER...body");
}

TEST_F(FaultInjectionEnvTest, TruncationOfAnExistingFileIsRolledBack) {
  ABSL_ASSERT_OK(dir_.WriteFile("f", "durable"));
  ABSL_ASSERT_OK(WriteThrough(env_, Path("f"), "new", /*sync=*/false));
  EXPECT_EQ(Read("f"), "new");
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_EQ(Read("f"), "durable");

  // The directory entry already existed, so syncing the data suffices.
  ABSL_ASSERT_OK(WriteThrough(env_, Path("f"), "new", /*sync=*/true));
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_EQ(Read("f"), "new");
}

TEST_F(FaultInjectionEnvTest, CreateWithoutSyncDirIsUndone) {
  ABSL_ASSERT_OK(WriteThrough(env_, Path("f"), "data", /*sync=*/true));
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_EQ(Read("f"), kMissing);
}

TEST_F(FaultInjectionEnvTest, SyncDirAcceptsATrailingSlash) {
  ABSL_ASSERT_OK(WriteThrough(env_, Path("f"), "data", /*sync=*/true));
  ABSL_ASSERT_OK(env_.SyncDir(absl::StrCat(dir_.path(), "/")));
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_EQ(Read("f"), "data");
}

TEST_F(FaultInjectionEnvTest, RenameWithoutSyncDirIsUndone) {
  ABSL_ASSERT_OK(WriteThrough(env_, Path("a"), "A", /*sync=*/true));
  ABSL_ASSERT_OK(env_.SyncDir(dir_.path()));
  ABSL_ASSERT_OK(env_.RenameFile(Path("a"), Path("b")));
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_EQ(Read("a"), "A");
  EXPECT_EQ(Read("b"), kMissing);
}

TEST_F(FaultInjectionEnvTest, UndoneRenameBringsBackTheReplacedFile) {
  ABSL_ASSERT_OK(dir_.WriteFile("MANIFEST", "old"));
  ABSL_ASSERT_OK(
      WriteThrough(env_, Path("MANIFEST.tmp"), "new", /*sync=*/true));
  ABSL_ASSERT_OK(env_.RenameFile(Path("MANIFEST.tmp"), Path("MANIFEST")));
  EXPECT_EQ(Read("MANIFEST"), "new");
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_EQ(Read("MANIFEST"), "old");
  EXPECT_EQ(Read("MANIFEST.tmp"), kMissing);
}

TEST_F(FaultInjectionEnvTest, RenameIsDurableAfterSyncDir) {
  ABSL_ASSERT_OK(dir_.WriteFile("MANIFEST", "old"));
  ABSL_ASSERT_OK(
      WriteThrough(env_, Path("MANIFEST.tmp"), "new", /*sync=*/true));
  ABSL_ASSERT_OK(env_.RenameFile(Path("MANIFEST.tmp"), Path("MANIFEST")));
  ABSL_ASSERT_OK(env_.SyncDir(dir_.path()));
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_EQ(Read("MANIFEST"), "new");
  EXPECT_EQ(Read("MANIFEST.tmp"), kMissing);
}

TEST_F(FaultInjectionEnvTest, ReplacedFileComesBackWithItsDurableContent) {
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WritableFile> target,
                              env_.NewWritableFile(Path("b")));
  ABSL_ASSERT_OK(target->Append("synced"));
  ABSL_ASSERT_OK(target->Sync());
  ABSL_ASSERT_OK(env_.SyncDir(dir_.path()));
  ABSL_ASSERT_OK(target->Append("+unsynced"));
  ABSL_ASSERT_OK(target->Close());
  ABSL_ASSERT_OK(WriteThrough(env_, Path("a"), "A", /*sync=*/true));
  ABSL_ASSERT_OK(env_.RenameFile(Path("a"), Path("b")));
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_EQ(Read("b"), "synced");
  EXPECT_EQ(Read("a"), kMissing);
}

TEST_F(FaultInjectionEnvTest, RenameAcrossDirectoriesNeedsBothSyncs) {
  ABSL_ASSERT_OK(env_.CreateDir(Path("d1")));
  ABSL_ASSERT_OK(env_.CreateDir(Path("d2")));
  ABSL_ASSERT_OK(dir_.WriteFile("d1/f", "F"));
  ABSL_ASSERT_OK(env_.RenameFile(Path("d1/f"), Path("d2/f")));
  ABSL_ASSERT_OK(env_.SyncDir(Path("d2")));
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_EQ(Read("d1/f"), "F");
  EXPECT_EQ(Read("d2/f"), kMissing);

  ABSL_ASSERT_OK(env_.RenameFile(Path("d1/f"), Path("d2/f")));
  ABSL_ASSERT_OK(env_.SyncDir(Path("d2")));
  ABSL_ASSERT_OK(env_.SyncDir(Path("d1")));
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_EQ(Read("d1/f"), kMissing);
  EXPECT_EQ(Read("d2/f"), "F");
}

TEST_F(FaultInjectionEnvTest, RenameWhoseOldNameWasDurablyReusedIsKept) {
  ABSL_ASSERT_OK(env_.CreateDir(Path("d1")));
  ABSL_ASSERT_OK(env_.CreateDir(Path("d2")));
  ABSL_ASSERT_OK(dir_.WriteFile("d1/a", "moved"));
  ABSL_ASSERT_OK(env_.RenameFile(Path("d1/a"), Path("d2/b")));
  ABSL_ASSERT_OK(WriteThrough(env_, Path("d1/a"), "reused", /*sync=*/true));
  // Makes the new d1/a durable, but not the rename's entry in d2.
  ABSL_ASSERT_OK(env_.SyncDir(Path("d1")));
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_EQ(Read("d1/a"), "reused");
  EXPECT_EQ(Read("d2/b"), "moved");
}

TEST_F(FaultInjectionEnvTest, DeleteIsDurableAtOnce) {
  ABSL_ASSERT_OK(dir_.WriteFile("f", "x"));
  ABSL_ASSERT_OK(env_.DeleteFile(Path("f")));
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_EQ(Read("f"), kMissing);
}

TEST_F(FaultInjectionEnvTest, DeletedFileIsNotBroughtBackByAnUndoneRename) {
  ABSL_ASSERT_OK(dir_.WriteFile("a", "A"));
  ABSL_ASSERT_OK(env_.RenameFile(Path("a"), Path("b")));
  ABSL_ASSERT_OK(env_.DeleteFile(Path("b")));
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_EQ(Read("a"), kMissing);
  EXPECT_EQ(Read("b"), kMissing);
}

TEST_F(FaultInjectionEnvTest, UndoFollowsFilesNotNames) {
  // Two different files are created at "x"; deleting the second must not
  // stop the first, renamed to "y", from being undone.
  ABSL_ASSERT_OK(WriteThrough(env_, Path("x"), "first", /*sync=*/true));
  ABSL_ASSERT_OK(env_.RenameFile(Path("x"), Path("y")));
  ABSL_ASSERT_OK(WriteThrough(env_, Path("x"), "second", /*sync=*/true));
  ABSL_ASSERT_OK(env_.DeleteFile(Path("x")));
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_THAT(env_.GetChildren(dir_.path()), IsOkAndHolds(IsEmpty()));
}

TEST_F(FaultInjectionEnvTest, FilesItDidNotWriteCountAsSynced) {
  ABSL_ASSERT_OK(dir_.WriteFile("a", "A"));
  ABSL_ASSERT_OK(dir_.WriteFile("sub/b", "B"));
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_EQ(Read("a"), "A");
  EXPECT_EQ(Read("sub/b"), "B");
}

TEST_F(FaultInjectionEnvTest, FaultFailsLaterMutationsWithoutPerformingThem) {
  env_.FailAfter(1, absl::UnavailableError("simulated crash"));
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WritableFile> file,
                              env_.NewWritableFile(Path("a")));
  const auto crashed =
      StatusIs(absl::StatusCode::kUnavailable, "simulated crash");
  EXPECT_THAT(file->Append("x"), crashed);
  EXPECT_THAT(env_.NewWritableFile(Path("b")), crashed);
  EXPECT_THAT(env_.CreateDir(Path("d")), crashed);
  EXPECT_THAT(env_.RenameFile(Path("a"), Path("c")), crashed);
  EXPECT_THAT(env_.SyncDir(dir_.path()), crashed);
  EXPECT_THAT(env_.DeleteFile(Path("a")), crashed);
  // Nothing reached the disk, and reads still work.
  EXPECT_THAT(env_.GetChildren(dir_.path()), IsOkAndHolds(ElementsAre("a")));
  EXPECT_THAT(env_.ReadFile(Path("a")), IsOkAndHolds(""));
  EXPECT_THAT(env_.LockFile(Path("LOCK")), IsOk());

  env_.ClearFault();
  EXPECT_THAT(file->Append("x"), IsOk());
  EXPECT_THAT(file->Close(), IsOk());
}

TEST_F(FaultInjectionEnvTest, CountsEveryMutationAndNothingElse) {
  EXPECT_EQ(env_.mutation_count(), 0);
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WritableFile> file,
                              env_.NewWritableFile(Path("a")));
  ABSL_ASSERT_OK(file->Append("0123"));
  ABSL_ASSERT_OK(file->WriteAt(0, "x"));
  ABSL_ASSERT_OK(file->Sync());
  ABSL_ASSERT_OK(file->FullSync());
  ABSL_ASSERT_OK(file->Close());
  ABSL_ASSERT_OK(env_.CreateDir(Path("d")));
  ABSL_ASSERT_OK(env_.RenameFile(Path("a"), Path("b")));
  ABSL_ASSERT_OK(env_.SyncDir(dir_.path()));
  ABSL_ASSERT_OK(env_.DeleteFile(Path("b")));
  EXPECT_EQ(env_.mutation_count(), 10);

  // A mutation that fails in the base env still counts.
  EXPECT_THAT(env_.DeleteFile(Path("b")),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_EQ(env_.mutation_count(), 11);

  // Reads, locks and size() do not.
  ABSL_ASSERT_OK(dir_.WriteFile("r", "r"));
  ABSL_ASSERT_OK(env_.ReadFile(Path("r")).status());
  ABSL_ASSERT_OK(env_.NewMappedFile(Path("r")).status());
  ABSL_ASSERT_OK(env_.GetFileSize(Path("r")).status());
  ABSL_ASSERT_OK(env_.FileExists(Path("r")).status());
  ABSL_ASSERT_OK(env_.GetChildren(dir_.path()).status());
  ABSL_ASSERT_OK(env_.LockFile(Path("LOCK")).status());
  EXPECT_EQ(file->size(), 4);
  EXPECT_EQ(env_.mutation_count(), 11);
}

TEST_F(FaultInjectionEnvTest, DropUnsyncedDataEndsTheFault) {
  env_.FailAfter(0, absl::UnavailableError("simulated crash"));
  EXPECT_THAT(env_.CreateDir(Path("d")), Not(IsOk()));
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_THAT(env_.CreateDir(Path("d")), IsOk());
}

TEST_F(FaultInjectionEnvTest, FilesOpenedBeforeACrashAreDisowned) {
  FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WritableFile> file,
                              env_.NewWritableFile(Path("f")));
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  EXPECT_THAT(file->Append("x"),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       "file was opened before a simulated crash"));
  EXPECT_THAT(file->Close(), StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(FaultInjectionEnvTest, ConcurrentWritersAreAllTracked) {
  constexpr int kThreads = 4;
  constexpr int kFilesPerThread = 20;
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([this, t] {
      for (int i = 0; i < kFilesPerThread; ++i) {
        const std::string name = absl::StrCat(t, "-", i);
        // Even files are synced, odd ones are not.
        EXPECT_THAT(WriteThrough(env_, Path(name), name, /*sync=*/i % 2 == 0),
                    IsOk());
      }
    });
  }
  for (std::thread& thread : threads) thread.join();
  ABSL_ASSERT_OK(env_.SyncDir(dir_.path()));
  ABSL_ASSERT_OK(env_.DropUnsyncedData());
  for (int t = 0; t < kThreads; ++t) {
    for (int i = 0; i < kFilesPerThread; ++i) {
      const std::string name = absl::StrCat(t, "-", i);
      EXPECT_EQ(Read(name), i % 2 == 0 ? name : "") << name;
    }
  }
}

// A miniature of the index's commit protocol (ferret/base/env.h), with two
// broken variants that crash enumeration has to catch.
enum class Protocol : uint8_t {
  kCorrect,
  kSegmentNotSynced,
  kDeleteBeforeSyncDir,
};

// Installs segment 2 over the durable state {MANIFEST: "1.seg", 1.seg}.
absl::Status InstallSegmentTwo(Env& env, const TempDir& dir,
                               Protocol protocol) {
  const bool sync_segment = protocol != Protocol::kSegmentNotSynced;
  ABSL_RETURN_IF_ERROR(
      WriteThrough(env, dir.Join("2.seg.tmp"), "segment two", sync_segment));
  ABSL_RETURN_IF_ERROR(
      env.RenameFile(dir.Join("2.seg.tmp"), dir.Join("2.seg")));
  ABSL_RETURN_IF_ERROR(
      WriteThrough(env, dir.Join("MANIFEST.tmp"), "2.seg", /*sync=*/true));
  ABSL_RETURN_IF_ERROR(
      env.RenameFile(dir.Join("MANIFEST.tmp"), dir.Join("MANIFEST")));
  if (protocol == Protocol::kDeleteBeforeSyncDir) {
    ABSL_RETURN_IF_ERROR(env.DeleteFile(dir.Join("1.seg")));
  }
  ABSL_RETURN_IF_ERROR(env.SyncDir(dir.path()));
  if (protocol != Protocol::kDeleteBeforeSyncDir) {
    ABSL_RETURN_IF_ERROR(env.DeleteFile(dir.Join("1.seg")));
  }
  return absl::OkStatus();
}

// Writes the durable state InstallSegmentTwo() starts from.
void WriteSegmentOneState(const TempDir& dir) {
  EXPECT_THAT(dir.WriteFile("MANIFEST", "1.seg"), IsOk());
  EXPECT_THAT(dir.WriteFile("1.seg", "segment one"), IsOk());
}

struct CrashOutcome {
  std::vector<int64_t> inconsistent_crash_points;
  std::vector<std::string> recovered_manifests;
};

// Crashes the protocol after each of its k mutations, for every k, and
// checks that the durable MANIFEST names a segment with intact content.
CrashOutcome EnumerateCrashes(Protocol protocol) {
  int64_t num_mutations = 0;
  {
    TempDir dir;
    WriteSegmentOneState(dir);
    FaultInjectionEnv env(Env::Default());
    EXPECT_THAT(InstallSegmentTwo(env, dir, protocol), IsOk());
    num_mutations = env.mutation_count();
  }
  CrashOutcome outcome;
  for (int64_t k = 0; k <= num_mutations; ++k) {
    TempDir dir;
    WriteSegmentOneState(dir);
    FaultInjectionEnv env(Env::Default());
    env.FailAfter(k, absl::UnavailableError("simulated crash"));
    // Fails unless the crash comes after the last mutation.
    InstallSegmentTwo(env, dir, protocol).IgnoreError();
    EXPECT_THAT(env.DropUnsyncedData(), IsOk());

    const std::string manifest = ReadOrMissing(dir.Join("MANIFEST"));
    outcome.recovered_manifests.push_back(manifest);
    const bool consistent =
        (manifest == "1.seg" &&
         ReadOrMissing(dir.Join("1.seg")) == "segment one") ||
        (manifest == "2.seg" &&
         ReadOrMissing(dir.Join("2.seg")) == "segment two");
    if (!consistent) outcome.inconsistent_crash_points.push_back(k);
  }
  return outcome;
}

TEST(FaultInjectionEnvCrashTest, CorrectProtocolSurvivesEveryCrashPoint) {
  const CrashOutcome outcome = EnumerateCrashes(Protocol::kCorrect);
  EXPECT_THAT(outcome.inconsistent_crash_points, IsEmpty());
  // Early crashes recover the old MANIFEST and the last one the new one, so
  // the enumeration exercised both sides of the commit point.
  ASSERT_FALSE(outcome.recovered_manifests.empty());
  EXPECT_EQ(outcome.recovered_manifests.front(), "1.seg");
  EXPECT_EQ(outcome.recovered_manifests.back(), "2.seg");
}

TEST(FaultInjectionEnvCrashTest, CatchesASegmentThatWasNotSynced) {
  EXPECT_THAT(
      EnumerateCrashes(Protocol::kSegmentNotSynced).inconsistent_crash_points,
      Not(IsEmpty()));
}

TEST(FaultInjectionEnvCrashTest, CatchesADeleteBeforeTheManifestIsDurable) {
  EXPECT_THAT(EnumerateCrashes(Protocol::kDeleteBeforeSyncDir)
                  .inconsistent_crash_points,
              Not(IsEmpty()));
}

}  // namespace
}  // namespace ferret

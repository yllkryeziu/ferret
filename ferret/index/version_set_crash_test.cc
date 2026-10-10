// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/base/mapped_file.h"
#include "ferret/base/testing/fault_injection_env.h"
#include "ferret/git/blob_hash.h"
#include "ferret/index/manifest.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/reaper.h"
#include "ferret/index/segment.h"
#include "ferret/index/segment_writer.h"
#include "ferret/index/version_set.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"

namespace ferret {
namespace {

using ::absl_testing::IsOk;
using ::ferret_testing::FaultInjectionEnv;
using ::ferret_testing::TempDir;

const std::vector<ManifestView> kFirstViews{{"/root/a", "/git/a"}};
const std::vector<ManifestView> kLastViews{{"/root/b", "/git/b"}};

absl::Status AddDoc(Env& env, VersionSet& set, const std::string& name,
                    const std::string& content, std::string& path) {
  auto allocated = set.NewSegmentFile();
  if (!allocated.ok()) return allocated.status();
  path = *allocated;
  auto writer = SegmentWriter::Create(env, path);
  if (!writer.ok()) return writer.status();
  absl::Status status =
      (*writer)->Add({.hint_path = name,
                      .content = content,
                      .rank_key = ComputeRankKey(name, content.size(), false,
                                                 HashBlob(content))});
  if (!status.ok()) return status;
  auto finished = (*writer)->Finish();
  return finished.ok() ? absl::OkStatus() : finished.status();
}

absl::Status Step(Env& env, VersionSet& set, int step) {
  if (step == 3) return set.Install({.views = kLastViews});
  std::string path;
  const std::string name = absl::StrCat("doc", step, ".cc");
  const std::string content = absl::StrCat("unique content ", step, "\n");
  absl::Status status = AddDoc(env, set, name, content, path);
  if (!status.ok()) return status;
  if (step == 0) return set.Install({.add = {path}, .views = kFirstViews});
  if (step == 1) return set.Install({.add = {path}});
  return set.Install({.add = {path}, .remove = {1, 2}});
}

// Own strings rather than comparing file numbers or views into a mapping.
using Docs = std::vector<std::pair<std::string, std::string>>;

Docs Contents(const Version& version) {
  Docs docs;
  for (const VersionSegment& entry : version.segments) {
    for (uint32_t d = 0; d < entry.segment->num_docs(); ++d) {
      SegmentDoc doc = entry.segment->doc(d);
      docs.emplace_back(doc.hint_path, doc.content);
    }
  }
  std::sort(docs.begin(), docs.end());
  return docs;
}

Docs Expected(int completed_steps) {
  Docs docs;
  if (completed_steps == 1 || completed_steps == 2) {
    docs.emplace_back("doc0.cc", "unique content 0\n");
  }
  if (completed_steps == 2) {
    docs.emplace_back("doc1.cc", "unique content 1\n");
  }
  if (completed_steps >= 3) {
    docs.emplace_back("doc2.cc", "unique content 2\n");
  }
  return docs;
}

std::vector<ManifestView> ExpectedViews(int completed_steps) {
  if (completed_steps == 0) return {};
  return completed_steps == 4 ? kLastViews : kFirstViews;
}

void AssertNoOrphans(Env& env, const std::string& index) {
  Manifest manifest;
  FERRET_ASSERT_OK_AND_ASSIGN(bool exists, env.FileExists(index + "/MANIFEST"));
  if (exists) {
    FERRET_ASSERT_OK_AND_ASSIGN(std::string data,
                                env.ReadFile(index + "/MANIFEST"));
    FERRET_ASSERT_OK_AND_ASSIGN(manifest, DecodeManifest(data));
  }
  FERRET_ASSERT_OK_AND_ASSIGN(auto children, env.GetChildren(index));
  for (const std::string& name : children) {
    EXPECT_FALSE(absl::string_view(name).ends_with(".tmp")) << name;
    if (absl::string_view(name).ends_with(".fseg")) {
      bool listed = false;
      for (const ManifestSegment& segment : manifest.segments) {
        if (name == absl::StrFormat("%06u.fseg", segment.file_number)) {
          listed = true;
        }
      }
      EXPECT_TRUE(listed) << name;
    }
  }
}

// Creates the state preceding `interrupted_step` without injecting faults.
void Prepare(Env& env, const std::string& index, int interrupted_step) {
  Reaper reaper;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              VersionSet::Open(env, index, reaper));
  for (int i = 0; i < interrupted_step; ++i) {
    ASSERT_THAT(Step(env, *opened.first, i), IsOk()) << i;
  }
}

// Records the fault env's mutation count immediately before a chosen MANIFEST
// operation. File mutations still run through the fault env's tracked files.
class ManifestMutationEnv final : public Env {
 public:
  enum class Target { kRename, kSyncDir, kSegmentSyncDir };

  ManifestMutationEnv(FaultInjectionEnv& fault, std::string index,
                      Target target)
      : fault_(fault), index_(std::move(index)), target_(target) {}

  int64_t target_before() const { return target_before_; }
  bool target_failed() const { return target_failed_; }
  int sync_dir_calls() const { return sync_dir_calls_; }
  bool manifest_written_before_target() const {
    return manifest_written_before_target_;
  }

  absl::StatusOr<std::unique_ptr<WritableFile>> NewWritableFile(
      absl::string_view path) override {
    if (path == absl::StrCat(index_, "/MANIFEST.tmp")) {
      saw_manifest_write_ = true;
    }
    return fault_.NewWritableFile(path);
  }
  absl::StatusOr<std::string> ReadFile(absl::string_view path) override {
    return fault_.ReadFile(path);
  }
  absl::StatusOr<std::unique_ptr<MappedFile>> NewMappedFile(
      absl::string_view path) override {
    return fault_.NewMappedFile(path);
  }
  absl::StatusOr<int64_t> GetFileSize(absl::string_view path) override {
    return fault_.GetFileSize(path);
  }
  absl::StatusOr<bool> FileExists(absl::string_view path) override {
    return fault_.FileExists(path);
  }
  absl::StatusOr<std::vector<std::string>> GetChildren(
      absl::string_view path) override {
    return fault_.GetChildren(path);
  }
  absl::Status CreateDir(absl::string_view path) override {
    return fault_.CreateDir(path);
  }
  absl::Status DeleteFile(absl::string_view path) override {
    return fault_.DeleteFile(path);
  }
  absl::Status RenameFile(absl::string_view from,
                          absl::string_view to) override {
    const bool targeted =
        target_ == Target::kRename && to == absl::StrCat(index_, "/MANIFEST");
    if (targeted) target_before_ = fault_.mutation_count();
    absl::Status status = fault_.RenameFile(from, to);
    if (targeted) target_failed_ = !status.ok();
    return status;
  }
  absl::Status SyncDir(absl::string_view path) override {
    if (path == index_) ++sync_dir_calls_;
    const bool targeted =
        path == index_ &&
        (target_ == Target::kSyncDir ||
         (target_ == Target::kSegmentSyncDir && target_before_ < 0));
    if (targeted) {
      target_before_ = fault_.mutation_count();
      manifest_written_before_target_ = saw_manifest_write_;
    }
    absl::Status status = fault_.SyncDir(path);
    if (targeted) target_failed_ = !status.ok();
    return status;
  }
  absl::StatusOr<std::unique_ptr<FileLock>> LockFile(
      absl::string_view path) override {
    return fault_.LockFile(path);
  }

 private:
  FaultInjectionEnv& fault_;
  std::string index_;
  Target target_;
  int64_t target_before_ = -1;
  bool target_failed_ = false;
  bool saw_manifest_write_ = false;
  bool manifest_written_before_target_ = false;
  int sync_dir_calls_ = 0;
};

// Count a successful install on an independent directory, so the failure
// point follows the protocol if earlier mutations are added.
int64_t MutationBefore(ManifestMutationEnv::Target target,
                       bool add_segment = false) {
  TempDir dir;
  const std::string index = dir.Join("index");
  Prepare(*Env::Default(), index, 1);
  FaultInjectionEnv fault(Env::Default());
  ManifestMutationEnv tracing(fault, index, target);
  Reaper reaper;
  auto opened = VersionSet::Open(tracing, index, reaper);
  EXPECT_THAT(opened, IsOk());
  if (!opened.ok()) return -1;
  std::string segment_path;
  if (add_segment) {
    EXPECT_THAT(
        AddDoc(tracing, *opened->first, "extra.cc", "extra\n", segment_path),
        IsOk());
  }
  const int64_t starting_count = fault.mutation_count();
  EXPECT_THAT(opened->first->Install(
                  {.add = add_segment ? std::vector<std::string>{segment_path}
                                      : std::vector<std::string>{},
                   .views = kLastViews}),
              IsOk());
  EXPECT_GE(tracing.target_before(), starting_count);
  EXPECT_FALSE(tracing.target_failed());
  EXPECT_EQ(tracing.sync_dir_calls(), add_segment ? 2 : 1);
  return tracing.target_before() - starting_count;
}

TEST(VersionSetCrashTest, EveryInstallMutationRecoversAnEntireVersion) {
  for (int step = 0; step < 4; ++step) {
    // Count the actual operations, including cleanup after returning from
    // Install, before enumerating every failure boundary.
    int64_t mutations = 0;
    {
      TempDir dir;
      const std::string index = dir.Join("index");
      Prepare(*Env::Default(), index, step);
      FaultInjectionEnv fault(Env::Default());
      {
        Reaper reaper;
        FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                                    VersionSet::Open(fault, index, reaper));
        ASSERT_THAT(Step(fault, *opened.first, step), IsOk());
      }
      mutations = fault.mutation_count();
    }
    for (int64_t k = 0; k <= mutations; ++k) {
      SCOPED_TRACE(absl::StrCat("step=", step, " crash=", k));
      TempDir dir;
      const std::string index = dir.Join("index");
      Prepare(*Env::Default(), index, step);
      FaultInjectionEnv fault(Env::Default());
      {
        Reaper reaper;
        FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                                    VersionSet::Open(fault, index, reaper));
        fault.FailAfter(k, absl::UnavailableError("simulated power loss"));
        static_cast<void>(Step(fault, *opened.first, step));
      }
      ASSERT_THAT(fault.DropUnsyncedData(), IsOk());
      Reaper recovery_reaper;
      FERRET_ASSERT_OK_AND_ASSIGN(
          auto recovered,
          VersionSet::Open(*Env::Default(), index, recovery_reaper));
      EXPECT_TRUE(recovered.second.dropped_segments.empty());
      const Docs docs = Contents(*recovered.first->current());
      const auto views = recovered.first->current()->views;
      const bool before =
          docs == Expected(step) && views == ExpectedViews(step);
      const bool after =
          docs == Expected(step + 1) && views == ExpectedViews(step + 1);
      EXPECT_TRUE(before || after) << "inconsistent recovered snapshot";
      AssertNoOrphans(*Env::Default(), index);
      recovered.first.reset();
      FERRET_ASSERT_OK_AND_ASSIGN(
          auto second,
          VersionSet::Open(*Env::Default(), index, recovery_reaper));
      EXPECT_TRUE(second.second.dropped_segments.empty());
      EXPECT_TRUE(second.second.deleted_files.empty());
      EXPECT_FALSE(second.second.manifest_reset);
    }
  }
}

TEST(VersionSetCrashTest, CorruptSegmentsAreDroppedAndReported) {
  // Each corruption is checked by Segment::Open, not by a section CRC scan.
  for (int corruption = 0; corruption < 3; ++corruption) {
    TempDir dir;
    const std::string index = dir.Join("index");
    Prepare(*Env::Default(), index, 2);
    Env& env = *Env::Default();
    const std::string file = index + "/000001.fseg";
    FERRET_ASSERT_OK_AND_ASSIGN(std::string bytes, env.ReadFile(file));
    if (corruption == 0) {
      bytes.resize(bytes.size() - 1);
    } else if (corruption == 1) {
      bytes[bytes.size() - 32] ^= 1;  // Footer magic.
    } else {
      bytes[0] ^= 1;  // Header magic.
    }
    FERRET_ASSERT_OK_AND_ASSIGN(auto writer, env.NewWritableFile(file));
    ASSERT_THAT(writer->Append(bytes), IsOk());
    ASSERT_THAT(writer->Close(), IsOk());
    Reaper reaper;
    FERRET_ASSERT_OK_AND_ASSIGN(auto recovered,
                                VersionSet::Open(env, index, reaper));
    ASSERT_EQ(recovered.second.dropped_segments.size(), 1u);
    EXPECT_EQ(recovered.second.dropped_segments[0].file_number, 1u);
    EXPECT_FALSE(recovered.second.dropped_segments[0].reason.empty());
    EXPECT_EQ(Contents(*recovered.first->current()),
              (Docs{{"doc1.cc", "unique content 1\n"}}));
    AssertNoOrphans(env, index);
    recovered.first.reset();
    FERRET_ASSERT_OK_AND_ASSIGN(auto second,
                                VersionSet::Open(env, index, reaper));
    EXPECT_TRUE(second.second.dropped_segments.empty());
    EXPECT_TRUE(second.second.deleted_files.empty());
  }
}

TEST(VersionSetCrashTest, SectionChecksumIsNotVerifiedOnOpen) {
  TempDir dir;
  const std::string index = dir.Join("index");
  Prepare(*Env::Default(), index, 1);
  Env& env = *Env::Default();
  const std::string file = index + "/000001.fseg";
  FERRET_ASSERT_OK_AND_ASSIGN(std::string bytes, env.ReadFile(file));
  const size_t offset = bytes.find("unique content 0\n");
  ASSERT_NE(offset, std::string::npos);
  bytes[offset] ^= 1;
  FERRET_ASSERT_OK_AND_ASSIGN(auto writer, env.NewWritableFile(file));
  ASSERT_THAT(writer->Append(bytes), IsOk());
  ASSERT_THAT(writer->Close(), IsOk());

  Reaper reaper;
  FERRET_ASSERT_OK_AND_ASSIGN(auto reopened,
                              VersionSet::Open(env, index, reaper));
  EXPECT_TRUE(reopened.second.dropped_segments.empty());
  ASSERT_EQ(reopened.first->current()->segments.size(), 1u);
  EXPECT_EQ(reopened.first->current()->segments[0].segment->doc(0).content,
            "tnique content 0\n");
}

TEST(VersionSetCrashTest, NewSegmentRenamesAreSyncedBeforeManifestWrite) {
  const int64_t before =
      MutationBefore(ManifestMutationEnv::Target::kSegmentSyncDir, true);
  TempDir dir;
  const std::string index = dir.Join("index");
  Prepare(*Env::Default(), index, 1);
  FaultInjectionEnv fault(Env::Default());
  ManifestMutationEnv tracing(fault, index,
                              ManifestMutationEnv::Target::kSegmentSyncDir);
  Reaper reaper;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              VersionSet::Open(tracing, index, reaper));
  std::string path;
  ASSERT_THAT(AddDoc(tracing, *opened.first, "extra.cc", "extra\n", path),
              IsOk());
  const int64_t starting_count = fault.mutation_count();
  fault.FailAfter(before,
                  absl::UnavailableError("segment directory sync failed"));
  EXPECT_THAT(opened.first->Install({.add = {path}}),
              absl_testing::StatusIs(absl::StatusCode::kUnavailable));
  EXPECT_EQ(tracing.target_before() - starting_count, before);
  EXPECT_TRUE(tracing.target_failed());
  EXPECT_EQ(tracing.sync_dir_calls(), 1);
  EXPECT_FALSE(tracing.manifest_written_before_target());
  EXPECT_EQ(Contents(*opened.first->current()), Expected(1));
  opened.first.reset();
  fault.ClearFault();
  FERRET_ASSERT_OK_AND_ASSIGN(auto reopened,
                              VersionSet::Open(tracing, index, reaper));
  EXPECT_EQ(Contents(*reopened.first->current()), Expected(1));
}

TEST(VersionSetCrashTest, CorruptOrMissingManifestResetsNumberedIndex) {
  for (bool missing : {true, false}) {
    TempDir dir;
    const std::string index = dir.Join("index");
    Prepare(*Env::Default(), index, 1);
    Env& env = *Env::Default();
    const std::string path = index + "/MANIFEST";
    if (missing) {
      ASSERT_THAT(env.DeleteFile(path), IsOk());
    } else {
      FERRET_ASSERT_OK_AND_ASSIGN(auto writer, env.NewWritableFile(path));
      ASSERT_THAT(writer->Append("bad CRC"), IsOk());
      ASSERT_THAT(writer->Close(), IsOk());
    }
    Reaper reaper;
    FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                                VersionSet::Open(env, index, reaper));
    EXPECT_TRUE(opened.second.manifest_reset);
    EXPECT_TRUE(opened.first->current()->segments.empty());
    EXPECT_TRUE(opened.first->current()->views.empty());
    AssertNoOrphans(env, index);
  }
}

TEST(VersionSetCrashTest, FailureBeforeRenameCanBeRetried) {
  TempDir dir;
  const std::string index = dir.Join("index");
  Prepare(*Env::Default(), index, 1);
  FaultInjectionEnv fault(Env::Default());
  ManifestMutationEnv tracing(fault, index,
                              ManifestMutationEnv::Target::kRename);
  Reaper reaper;
  FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                              VersionSet::Open(tracing, index, reaper));
  const int64_t before = MutationBefore(ManifestMutationEnv::Target::kRename);
  const int64_t starting_count = fault.mutation_count();
  fault.FailAfter(before, absl::UnavailableError("rename failed"));
  EXPECT_THAT(opened.first->Install({.views = kLastViews}),
              absl_testing::StatusIs(absl::StatusCode::kUnavailable));
  EXPECT_EQ(tracing.target_before() - starting_count, before);
  EXPECT_TRUE(tracing.target_failed());
  EXPECT_EQ(opened.first->current()->views, kFirstViews);
  fault.ClearFault();
  ASSERT_THAT(opened.first->Install({.views = kLastViews}), IsOk());
  EXPECT_EQ(opened.first->current()->views, kLastViews);
}

TEST(VersionSetCrashTest, FailedDirectorySyncBlocksFurtherInstalls) {
  TempDir dir;
  const std::string index = dir.Join("index");
  Prepare(*Env::Default(), index, 1);
  FaultInjectionEnv fault(Env::Default());
  ManifestMutationEnv tracing(fault, index,
                              ManifestMutationEnv::Target::kSyncDir);
  const int64_t before = MutationBefore(ManifestMutationEnv::Target::kSyncDir);
  {
    Reaper reaper;
    FERRET_ASSERT_OK_AND_ASSIGN(auto opened,
                                VersionSet::Open(tracing, index, reaper));
    const int64_t starting_count = fault.mutation_count();
    fault.FailAfter(before, absl::UnavailableError("directory sync failed"));
    EXPECT_THAT(opened.first->Install({.views = kLastViews}),
                absl_testing::StatusIs(absl::StatusCode::kUnavailable));
    EXPECT_EQ(tracing.target_before() - starting_count, before);
    EXPECT_TRUE(tracing.target_failed());
    EXPECT_EQ(opened.first->current()->views, kFirstViews);
    EXPECT_THAT(opened.first->Install({}),
                absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
    EXPECT_THAT(opened.first->NewSegmentFile().status(),
                absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
  }
  ASSERT_THAT(fault.DropUnsyncedData(), IsOk());
  Reaper reaper;
  FERRET_ASSERT_OK_AND_ASSIGN(auto reopened,
                              VersionSet::Open(*Env::Default(), index, reaper));
  EXPECT_EQ(reopened.first->current()->views, kFirstViews);
}

}  // namespace
}  // namespace ferret

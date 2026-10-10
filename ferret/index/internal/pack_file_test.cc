// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/internal/pack_file.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/base/env.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"

namespace ferret::segment_internal {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

TEST(PackFileTest, AppendsConcurrentlyAndReadsBack) {
  ferret_testing::TempDir dir;
  const std::string path = dir.Join("p.pack");
  FERRET_ASSERT_OK_AND_ASSIGN(auto pack,
                              PackFile::Create(*Env::Default(), path));
  struct Entry {
    uint64_t offset;
    std::string data;
  };
  std::vector<std::vector<Entry>> entries(4);
  for (auto& list : entries) {
    list.reserve(200);
  }
  std::vector<std::thread> threads;
  threads.reserve(4);
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < 200; ++i) {
        std::string data(static_cast<size_t>(1 + (i * 37 + t) % 900),
                         static_cast<char>('a' + t));
        data += std::to_string(i);
        absl::StatusOr<uint64_t> offset = pack->Append(data);
        ASSERT_THAT(offset.status(), IsOk());
        entries[t].push_back({*offset, std::move(data)});
      }
    });
  }
  for (std::thread& thread : threads) thread.join();
  // Contents already appended can be read while others are still written.
  {
    std::string early;
    const Entry& entry = entries[0][0];
    ASSERT_THAT(pack->ReadAt(entry.offset, entry.data.size(), early), IsOk());
    EXPECT_EQ(early, entry.data);
  }
  ASSERT_THAT(pack->FinishWriting(), IsOk());
  EXPECT_THAT(pack->Append("late"),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  std::string out;
  for (const auto& list : entries) {
    for (const Entry& entry : list) {
      ASSERT_THAT(pack->ReadAt(entry.offset, entry.data.size(), out), IsOk());
      EXPECT_EQ(out, entry.data);
    }
  }
  ASSERT_THAT(pack->ReadAt(0, 0, out), IsOk());
  EXPECT_TRUE(out.empty());
  EXPECT_THAT(pack->ReadAt(1 << 30, 10, out),
              StatusIs(absl::StatusCode::kDataLoss));
  FERRET_ASSERT_OK_AND_ASSIGN(const bool exists,
                              Env::Default()->FileExists(path));
  EXPECT_TRUE(exists);
  pack.reset();
  FERRET_ASSERT_OK_AND_ASSIGN(const bool still,
                              Env::Default()->FileExists(path));
  EXPECT_FALSE(still);
}

TEST(PackFileTest, DeletedEvenIfNeverFinished) {
  ferret_testing::TempDir dir;
  const std::string path = dir.Join("q.pack");
  {
    FERRET_ASSERT_OK_AND_ASSIGN(auto pack,
                                PackFile::Create(*Env::Default(), path));
    ASSERT_THAT(pack->Append("x").status(), IsOk());
  }
  FERRET_ASSERT_OK_AND_ASSIGN(const bool exists,
                              Env::Default()->FileExists(path));
  EXPECT_FALSE(exists);
}

}  // namespace
}  // namespace ferret::segment_internal

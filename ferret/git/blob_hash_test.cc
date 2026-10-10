// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/blob_hash.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <random>
#include <string>

#include "absl/strings/ascii.h"
#include "gtest/gtest.h"

#include "ferret/git/oid.h"
#include "ferret/git/testing/git_repo_builder.h"
#include "ferret/testing/status_matchers.h"
#include "ferret/testing/temp_dir.h"

namespace ferret {
namespace {

using ::ferret_testing::GitRepoBuilder;
using ::ferret_testing::TempDir;

TEST(BlobHashTest, MatchesGitVectors) {
  EXPECT_EQ(HashBlob("").ToHex(), "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391");
  EXPECT_EQ(HashBlob("hello\n").ToHex(),
            "ce013625030ba8dba906f756967f9e9ca394464a");
}

TEST(BlobHashTest, MatchesGitHashObjectOnRandomContent) {
  TempDir temp;
  FERRET_ASSERT_OK_AND_ASSIGN(const GitRepoBuilder repo,
                              GitRepoBuilder::Init(temp, "repo"));
  constexpr uint64_t kSeed = 20260923;
  SCOPED_TRACE(testing::Message() << "seed " << kSeed);
  std::mt19937_64 rng(kSeed);
  // Sizes around SHA-1's 64-byte block and the header's digit-count changes,
  // then random sizes up to 256 KiB.
  constexpr size_t kEdgeSizes[] = {1,  2,  9,  10, 44, 45,  46,  54,  55,  56,
                                   57, 63, 64, 65, 99, 100, 119, 120, 127, 128};
  for (int i = 0; i < 40; ++i) {
    const size_t size = i < static_cast<int>(std::size(kEdgeSizes))
                            ? kEdgeSizes[i]
                            : static_cast<size_t>(rng() % (256 << 10));
    std::string content(size, '\0');
    for (char& c : content) c = static_cast<char>(rng());
    FERRET_ASSERT_OK_AND_ASSIGN(const std::string git_hex,
                                repo.Git({"hash-object", "--stdin"}, content));
    EXPECT_EQ(HashBlob(content).ToHex(), absl::StripAsciiWhitespace(git_hex))
        << "size " << size;
  }
}

}  // namespace
}  // namespace ferret

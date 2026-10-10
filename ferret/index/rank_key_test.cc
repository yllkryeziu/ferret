// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/rank_key.h"

#include <cstdint>
#include <string>

#include "gtest/gtest.h"

#include "ferret/git/oid.h"

namespace ferret {
namespace {

Oid MakeOid(uint8_t first_byte) {
  Oid::Bytes bytes = {};
  bytes[0] = first_byte;
  return Oid(bytes);
}

struct TierCase {
  std::string path;
  RankTier tier;
};

TEST(ClassifyTierTest, Table) {
  const TierCase k_cases[] = {
      {"main.cc", RankTier::kSource},
      {"src/net/socket.cc", RankTier::kSource},
      {"src/latest.py", RankTier::kSource},
      {"src/contest/judge.go", RankTier::kSource},
      {"src/testament.txt", RankTier::kSource},
      {"Makefile", RankTier::kSource},
      {".gitignore", RankTier::kSource},
      {"net/socket_test.cc", RankTier::kTest},
      {"net/socket_unittest.cc", RankTier::kTest},
      {"chrome/browser/ui_browsertest.cc", RankTier::kTest},
      {"src/FooTest.java", RankTier::kTest},
      {"web/app.test.js", RankTier::kTest},
      {"web/app.spec.ts", RankTier::kTest},
      {"pkg/test_parser.py", RankTier::kTest},
      {"tests/fixtures/input.txt", RankTier::kTest},
      {"src/testdata/golden.json", RankTier::kTest},
      {"README.md", RankTier::kDocumentation},
      {"docs/guide/intro.rst", RankTier::kDocumentation},
      {"LICENSE", RankTier::kDocumentation},
      {"Readme.txt", RankTier::kDocumentation},
      {"CHANGELOG.MD", RankTier::kDocumentation},
      {"CMakeLists.txt", RankTier::kSource},
      {"requirements.txt", RankTier::kSource},
      {"docs/conf.py", RankTier::kSource},
      {"tests/README.md", RankTier::kDocumentation},
      {"third_party/zlib/README", RankTier::kThirdParty},
      {"third_party/zlib/deflate.c", RankTier::kThirdParty},
      {"web/node_modules/react/index.js", RankTier::kThirdParty},
      {"vendor/github.com/x/y.go", RankTier::kThirdParty},
      // The worst matching tier wins.
      {"third_party/zlib/deflate_test.cc", RankTier::kThirdParty},
      {"proto/api.pb.cc", RankTier::kGenerated},
      {"proto/api_pb2.py", RankTier::kGenerated},
      {"static/app.min.js", RankTier::kGenerated},
      {"out/gen/version.h", RankTier::kGenerated},
      {"src/schema_generated.h", RankTier::kGenerated},
      {"package-lock.json", RankTier::kGenerated},
      {"third_party/foo/foo.pb.h", RankTier::kGenerated},
      // Only directory components count as directories.
      {"gen", RankTier::kSource},
      {"vendor", RankTier::kSource},
  };
  for (const TierCase& c : k_cases) {
    EXPECT_EQ(ClassifyTier(c.path, /*long_lines=*/false), c.tier) << c.path;
  }
}

TEST(ClassifyTierTest, LongLinesWins) {
  EXPECT_EQ(ClassifyTier("src/main.cc", /*long_lines=*/true),
            RankTier::kLongLines);
  EXPECT_EQ(ClassifyTier("third_party/x.min.js", /*long_lines=*/true),
            RankTier::kLongLines);
}

TEST(RankTierNameTest, Names) {
  EXPECT_EQ(RankTierName(RankTier::kSource), "source");
  EXPECT_EQ(RankTierName(RankTier::kTest), "test");
  EXPECT_EQ(RankTierName(RankTier::kDocumentation), "documentation");
  EXPECT_EQ(RankTierName(RankTier::kThirdParty), "third_party");
  EXPECT_EQ(RankTierName(RankTier::kGenerated), "generated");
  EXPECT_EQ(RankTierName(RankTier::kLongLines), "long_lines");
}

TEST(ComputePriorTest, Formula) {
  EXPECT_EQ(ComputePrior("a.cc", 0), 65535);
  EXPECT_EQ(ComputePrior("a.cc", 4095), 65535);
  EXPECT_EQ(ComputePrior("a.cc", 4096), 65535 - 2048);
  EXPECT_EQ(ComputePrior("x/a.cc", 100), 65535 - 4096);
  EXPECT_EQ(ComputePrior("x/y/a.cc", 1 << 20), 65535 - 2 * 4096 - 9 * 2048);
}

TEST(ComputePriorTest, BoundedBelow) {
  EXPECT_EQ(ComputePrior("a/b/c/d/e/f/g/h/i/j/k/l.cc", UINT64_MAX), 2047);
}

TEST(ComputePriorTest, ShallowerAndSmallerRankHigher) {
  EXPECT_GT(ComputePrior("a.cc", 1000), ComputePrior("x/a.cc", 1000));
  EXPECT_GT(ComputePrior("x/a.cc", 1000), ComputePrior("x/a.cc", 100000));
}

TEST(RankKeyTest, OrdersByTierThenPriorDescendingThenOid) {
  const RankKey source_low{RankTier::kSource, 100, MakeOid(9)};
  const RankKey source_high{RankTier::kSource, 200, MakeOid(9)};
  const RankKey test_high{RankTier::kTest, 60000, MakeOid(0)};
  const RankKey source_high_other{RankTier::kSource, 200, MakeOid(1)};
  EXPECT_LT(source_high, source_low);
  EXPECT_LT(source_low, test_high);
  EXPECT_LT(source_high_other, source_high);
  EXPECT_EQ(source_high, (RankKey{RankTier::kSource, 200, MakeOid(9)}));
}

TEST(RankKeyTest, ComputeRankKey) {
  const RankKey key =
      ComputeRankKey("net/socket_test.cc", 5000, false, MakeOid(3));
  EXPECT_EQ(key.tier, RankTier::kTest);
  EXPECT_EQ(key.prior, ComputePrior("net/socket_test.cc", 5000));
  EXPECT_EQ(key.oid, MakeOid(3));
}

}  // namespace
}  // namespace ferret

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/blob_registry.h"

#include <cstddef>
#include <thread>
#include <vector>

#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "ferret/git/blob_hash.h"
#include "ferret/git/oid.h"

namespace ferret {
namespace {

TEST(BlobRegistryTest, DenseAndRoundTrips) {
  BlobRegistry registry;
  std::vector<Oid> oids;
  for (int i = 0; i < 1000; ++i) {
    oids.push_back(HashBlob(absl::StrCat(i)));
    EXPECT_EQ(registry.Intern(oids.back()), static_cast<BlobId>(i));
    EXPECT_EQ(registry.Intern(oids.back()), static_cast<BlobId>(i));
  }
  EXPECT_EQ(registry.size(), oids.size());
  for (int i = 0; i < 1000; ++i) {
    EXPECT_EQ(registry.oid(static_cast<BlobId>(i)), oids[i]);
    EXPECT_EQ(registry.Find(oids[i]), static_cast<BlobId>(i));
  }
  EXPECT_FALSE(registry.Find(HashBlob("missing")));
}

TEST(BlobRegistryTest, ConcurrentInterning) {
  BlobRegistry registry;
  std::vector<Oid> oids;
  oids.reserve(200);
  for (int i = 0; i < 200; ++i) oids.push_back(HashBlob(absl::StrCat(i)));
  std::vector<std::thread> threads;
  threads.reserve(8);
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&] {
      for (int repeat = 0; repeat < 10; ++repeat) {
        for (const Oid& oid : oids) {
          const BlobId id = registry.Intern(oid);
          EXPECT_EQ(registry.oid(id), oid);
          EXPECT_EQ(registry.Find(oid), id);
        }
      }
    });
  }
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(registry.size(), oids.size());
  for (size_t i = 0; i < oids.size(); ++i) {
    EXPECT_EQ(
        registry.oid(static_cast<BlobId>(i)),
        registry.oid(registry.Intern(registry.oid(static_cast<BlobId>(i)))));
  }
}

}  // namespace
}  // namespace ferret

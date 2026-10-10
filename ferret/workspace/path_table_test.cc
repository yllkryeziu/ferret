// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/workspace/path_table.h"

#include <cstddef>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

namespace ferret {
namespace {

TEST(PathTableTest, ValidateDoesNotIntern) {
  PathTable table;
  EXPECT_TRUE(PathTable::Validate("directory/file.cc").ok());
  EXPECT_EQ(table.size(), 0u);
  EXPECT_EQ(PathTable::Validate("../outside").code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(table.size(), 0u);
  EXPECT_TRUE(table.Intern("directory/file.cc").ok());
  EXPECT_EQ(table.size(), 1u);
}

TEST(PathTableTest, ValidatesAndInterns) {
  PathTable table;
  for (absl::string_view bad : std::vector<absl::string_view>{
           "", "/a", "./a", "a/../b", "a//b", "a/", "a/.", "a\\b",
           absl::string_view("a\0b", 3)}) {
    EXPECT_FALSE(table.Intern(bad).ok()) << bad;
  }
  ASSERT_TRUE(table.Intern("a/b").ok());
  EXPECT_EQ(*table.Intern("a/b"), PathId{0});
  EXPECT_EQ(*table.Intern("a"), PathId{1});
  EXPECT_EQ(table.path(PathId{0}), "a/b");
  EXPECT_EQ(table.Find("a"), PathId{1});
  EXPECT_FALSE(table.Find("other"));
  EXPECT_EQ(table.size(), size_t{2});
}

TEST(PathTableTest, ConcurrentInterning) {
  PathTable table;
  std::vector<std::thread> threads;
  threads.reserve(8);
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < 200; ++i) {
        const std::string path = absl::StrCat("dir/", i);
        auto id = table.Intern(path);
        ASSERT_TRUE(id.ok());
        EXPECT_EQ(table.path(*id), path);
        EXPECT_EQ(table.Find(path), *id);
      }
    });
  }
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(table.size(), size_t{200});
}

}  // namespace
}  // namespace ferret

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/line_counter.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>

#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "ferret/text/internal/line_counter_kernels.h"
#include "ferret/text/testing/exact_buffer.h"

namespace ferret {
namespace {

using ::ferret_testing::ExactBuffer;

TEST(CountNewlinesTest, CountsNewlineBytesOnly) {
  EXPECT_EQ(CountNewlines(""), 0);
  EXPECT_EQ(CountNewlines("abc"), 0);
  EXPECT_EQ(CountNewlines("\n"), 1);
  EXPECT_EQ(CountNewlines("a\r\nb\n\nc"), 3);
  EXPECT_EQ(CountNewlines("\x0a\x8a\xca"), 1);  // Only 0x0A is a newline.
}

TEST(CountNewlinesTest, AllNewlinesDoNotOverflowByteLanes) {
  // 255 iterations of 64 bytes each is where the lanes are drained.
  for (const size_t size : {size_t{16320}, size_t{16384}, size_t{100000}}) {
    const ExactBuffer text(std::string(size, '\n'));
    EXPECT_EQ(CountNewlines(text.view()), static_cast<int64_t>(size));
  }
}

TEST(CountNewlinesTest, MatchesScalarReferenceOnEverySizeAndAlignment) {
  constexpr uint64_t kSeed = 7;
  SCOPED_TRACE(kSeed);
  std::mt19937_64 rng(kSeed);
  std::string pool(2048, '\0');
  for (char& c : pool) {
    // About one byte in four is a newline.
    c = rng() % 4 == 0 ? '\n' : static_cast<char>(rng() % 256);
  }
  for (size_t offset = 0; offset < 16; ++offset) {
    for (size_t size = 0; size + offset <= pool.size(); size += 1 + size / 8) {
      const ExactBuffer text(absl::string_view(pool).substr(offset, size));
      ASSERT_EQ(CountNewlines(text.view()),
                text_internal::CountNewlinesScalar(text.view()))
          << "offset " << offset << " size " << size;
    }
  }
}

TEST(LineBoundaryTest, FindsTheLineAroundAnOffset) {
  constexpr absl::string_view kText = "ab\ncd\n\nef";
  // Offsets inside "cd", on its newline, on the empty line, and in "ef".
  EXPECT_EQ(LineBegin(kText, 4), 3u);
  EXPECT_EQ(LineEnd(kText, 4), 5u);
  EXPECT_EQ(LineBegin(kText, 5), 3u);
  EXPECT_EQ(LineEnd(kText, 5), 5u);
  EXPECT_EQ(LineBegin(kText, 6), 6u);
  EXPECT_EQ(LineEnd(kText, 6), 6u);
  EXPECT_EQ(LineBegin(kText, 8), 7u);
  EXPECT_EQ(LineEnd(kText, 8), kText.size());
}

TEST(LineBoundaryTest, HandlesTextEdges) {
  EXPECT_EQ(LineBegin("", 0), 0u);
  EXPECT_EQ(LineEnd("", 0), 0u);
  EXPECT_EQ(LineBegin("abc", 0), 0u);
  EXPECT_EQ(LineBegin("abc", 3), 0u);
  EXPECT_EQ(LineEnd("abc", 3), 3u);
  EXPECT_EQ(LineBegin("abc\n", 4), 4u);
  EXPECT_EQ(LineEnd("abc\n", 4), 4u);
}

}  // namespace
}  // namespace ferret

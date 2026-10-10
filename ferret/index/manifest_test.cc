// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/manifest.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/crc/crc32c.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/testing/status_matchers.h"

namespace ferret {
namespace {

TEST(ManifestTest, RoundTrip) {
  Manifest manifest{
      .manifest_seq = 42,
      .next_file_number = 19,
      .segments = {{1, 65536, 2}, {18, 100000, 1}},
      .views = {{"/repo", "/repo/.git"}, {std::string("a\0b", 3), "c"}}};
  FERRET_ASSERT_OK_AND_ASSIGN(std::string bytes, EncodeManifest(manifest));
  FERRET_ASSERT_OK_AND_ASSIGN(Manifest decoded, DecodeManifest(bytes));
  EXPECT_EQ(decoded, manifest);
}

TEST(ManifestTest, GoldenEmptyEncoding) {
  FERRET_ASSERT_OK_AND_ASSIGN(std::string bytes, EncodeManifest(Manifest{}));
  EXPECT_EQ(bytes, std::string("FERRETMF\x01\x00\x00\x00"
                               "\x00\x00\x00\x00\x00\x00\x00\x00"
                               "\x01\x00\x00\x00\x00\x00\x00\x00"
                               "\x00\x00\x00\x00\x00\x00\x00\x00"
                               "\x99\xa6\x70\x03",
                               40));
}

TEST(ManifestTest, RejectsEveryTruncationAndBitFlips) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::string bytes, EncodeManifest(Manifest{.manifest_seq = 10,
                                                 .next_file_number = 3,
                                                 .segments = {{1, 100, 1}},
                                                 .views = {{"/one", "/git"}}}));
  for (size_t n = 0; n < bytes.size(); ++n) {
    EXPECT_THAT(DecodeManifest(bytes.substr(0, n)).status(),
                absl_testing::StatusIs(absl::StatusCode::kDataLoss))
        << n;
  }
  for (size_t n = 0; n < bytes.size(); n += 3) {
    std::string corrupted = bytes;
    corrupted[n] ^= 0x40;
    EXPECT_FALSE(DecodeManifest(corrupted).ok()) << n;
  }
  bytes.push_back('!');
  EXPECT_THAT(DecodeManifest(bytes).status(),
              absl_testing::StatusIs(absl::StatusCode::kDataLoss));
}

TEST(ManifestTest, RejectsInvalidNumbersAndNewerFormat) {
  EXPECT_THAT(EncodeManifest(Manifest{.next_file_number = 2,
                                      .segments = {{1, 1, 1}, {1, 1, 1}}})
                  .status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      EncodeManifest(Manifest{.next_file_number = 2, .segments = {{2, 1, 1}}})
          .status(),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  FERRET_ASSERT_OK_AND_ASSIGN(std::string bytes, EncodeManifest(Manifest{}));
  bytes[8] = 2;
  EXPECT_THAT(DecodeManifest(bytes).status(),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

// Recompute the checksum to exercise structure validation, not just CRC.
void ReplaceCrc(std::string& bytes) {
  const uint32_t crc = static_cast<uint32_t>(absl::ComputeCrc32c(
      absl::string_view(bytes).substr(0, bytes.size() - 4)));
  for (size_t i = 0; i < 4; ++i) {
    bytes[bytes.size() - 4 + i] = static_cast<char>(crc >> (8 * i));
  }
}

TEST(ManifestTest, ReportsTrailingBytesAndImpossibleValues) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      std::string bytes,
      EncodeManifest(Manifest{.next_file_number = 3,
                              .segments = {{1, 100, 1}, {2, 100, 1}}}));
  bytes.insert(bytes.size() - 4, 1, '!');
  ReplaceCrc(bytes);
  EXPECT_THAT(DecodeManifest(bytes).status(),
              absl_testing::StatusIs(absl::StatusCode::kDataLoss,
                                     ::testing::HasSubstr("trailing")));
  bytes.erase(bytes.size() - 5, 1);
  // The second segment begins at byte 52.
  bytes[52] = 1;
  ReplaceCrc(bytes);
  EXPECT_THAT(DecodeManifest(bytes).status(),
              absl_testing::StatusIs(absl::StatusCode::kDataLoss,
                                     ::testing::HasSubstr("duplicate")));
  bytes[52] = 3;
  ReplaceCrc(bytes);
  EXPECT_THAT(DecodeManifest(bytes).status(),
              absl_testing::StatusIs(absl::StatusCode::kDataLoss,
                                     ::testing::HasSubstr("file number")));
  EXPECT_THAT(
      EncodeManifest(Manifest{.next_file_number = 2, .segments = {{1, 0, 0}}})
          .status(),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace ferret

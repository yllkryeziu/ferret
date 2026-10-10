// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Round-trip and validation tests for daemon search payloads.
#include "ferret/rpc/search_codec.h"

#include <string>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "ferret/engine/engine.h"
#include "ferret/index/rank_key.h"
#include "ferret/rpc/json_codec.h"
#include "ferret/search/search_types.h"
#include "ferret/text/lexer.h"
#include "ferret/workspace/workspace.h"

namespace ferret {
namespace {

TEST(SearchCodecTest, PreservesInvalidUtf8AndAllSearchFields) {
  SearchRequest request;
  request.pattern = std::string("\xff\0x", 3);
  request.path_glob = "\xfe";
  request.regex = true;
  request.case_sensitive = true;
  request.context_lines = 7;
  request.max_lines_per_file = 8;
  request.max_lines = 9;
  request.max_files = 10;
  request.max_verify_bytes = 11;
  request.exact_counts = true;
  request.count_sample = true;
  const auto encoded = EncodeSearchRequest(request);
  auto decoded = DecodeSearchRequest(JsonView(encoded));
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(EncodeSearchRequest(*decoded), encoded);

  WorkspaceSearchResult result;
  FileResult file;
  file.path = "\xff.cc";
  file.lines.push_back({.number = 4,
                        .text = "\xfe\xff",
                        .is_match = true,
                        .match_begin = 0,
                        .match_end = 2,
                        .match_class = MatchClass::kComment});
  file.num_matching_lines = 17;
  file.tier = RankTier::kGenerated;
  result.response.files.push_back(file);
  result.response.stats.verified_bytes = 212;
  result.response.stats.inconsistent_files = 3;
  result.response.stats.compile_time = absl::Nanoseconds(102);
  result.response.verified_fraction = 0.5;
  result.stale = true;
  result.generation = 91;
  const auto wire = EncodeWorkspaceSearchResult(result);
  auto back = DecodeWorkspaceSearchResult(JsonView(wire));
  ASSERT_TRUE(back.ok()) << back.status();
  EXPECT_EQ(EncodeWorkspaceSearchResult(*back), wire);
  EXPECT_EQ(back->response.files[0].lines[0].text, "\xfe\xff");
  EXPECT_EQ(back->response.stats.inconsistent_files, 3);
}

TEST(SearchCodecTest, ConsistencyAndStatusRoundTrip) {
  SearchConsistency consistency = {.mode = SearchConsistencyMode::kNotify,
                                   .min_generation = 73,
                                   .timeout = absl::Milliseconds(8)};
  const auto wire = EncodeSearchConsistency(consistency);
  auto decoded = DecodeSearchConsistency(JsonView(wire));
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(EncodeSearchConsistency(*decoded), wire);
  WorktreeIndexStatus status;
  status.root = "\xff";
  status.file_count = 17;
  status.watcher_failed = true;
  const auto state = EncodeWorktreeIndexStatus(status);
  auto restored = DecodeWorktreeIndexStatus(JsonView(state));
  ASSERT_TRUE(restored.ok()) << restored.status();
  EXPECT_EQ(EncodeWorktreeIndexStatus(*restored), state);
}

TEST(SearchCodecTest, RejectsInvalidData) {
  auto value = EncodeSearchRequest(SearchRequest());
  value["pattern_hex"] = "fg";
  EXPECT_EQ(DecodeSearchRequest(JsonView(value)).status().code(),
            absl::StatusCode::kInvalidArgument);
  auto consistency = EncodeSearchConsistency(SearchConsistency());
  consistency["mode"] = 99;
  EXPECT_EQ(DecodeSearchConsistency(JsonView(consistency)).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ferret

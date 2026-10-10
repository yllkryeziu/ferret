// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/rank_key.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

#include "ferret/git/oid.h"

namespace ferret {
namespace {

bool IsAnyOf(absl::string_view s,
             std::initializer_list<absl::string_view> candidates) {
  return std::find(candidates.begin(), candidates.end(), s) != candidates.end();
}

bool EndsWithAnyOf(absl::string_view s,
                   std::initializer_list<absl::string_view> suffixes) {
  return std::any_of(
      suffixes.begin(), suffixes.end(),
      [s](absl::string_view suffix) { return absl::EndsWith(s, suffix); });
}

// The rules look at directory names and at the file name separately.
struct PathParts {
  // Every component but the last.
  std::vector<absl::string_view> directories;
  absl::string_view name;
  // The name without its last extension: "foo.test" for "foo.test.js".
  absl::string_view stem;
};

PathParts SplitPath(absl::string_view path) {
  PathParts parts;
  parts.directories = absl::StrSplit(path, '/');
  parts.name = parts.directories.back();
  parts.directories.pop_back();
  const size_t dot = parts.name.rfind('.');
  parts.stem = dot == absl::string_view::npos || dot == 0
                   ? parts.name
                   : parts.name.substr(0, dot);
  return parts;
}

bool AnyDirectoryIsOneOf(const PathParts& parts,
                         std::initializer_list<absl::string_view> names) {
  return std::any_of(parts.directories.begin(), parts.directories.end(),
                     [names](absl::string_view directory) {
                       return IsAnyOf(directory, names);
                     });
}

bool IsGenerated(const PathParts& parts) {
  return AnyDirectoryIsOneOf(parts, {"gen", "generated", "__generated__"}) ||
         EndsWithAnyOf(parts.name,
                       {".pb.h", ".pb.cc", ".pb.go", "_pb2.py", "_pb2_grpc.py",
                        ".min.js", ".min.css", ".g.dart"}) ||
         absl::StrContains(parts.name, ".generated.") ||
         absl::StrContains(parts.name, "_generated.") ||
         IsAnyOf(parts.name, {"package-lock.json", "yarn.lock",
                              "pnpm-lock.yaml", "Cargo.lock", "go.sum",
                              "poetry.lock", "Gemfile.lock", "composer.lock"});
}

bool IsThirdParty(const PathParts& parts) {
  return AnyDirectoryIsOneOf(
      parts, {"third_party", "thirdparty", "3rdparty", "vendor", "vendored",
              "node_modules", "external", "extern"});
}

// Prose, which agents searching for code rarely want before code. Plain .txt
// is not prose by itself: CMakeLists.txt and requirements.txt are build
// inputs.
bool IsDocumentation(const PathParts& parts) {
  const size_t dot = parts.name.rfind('.');
  const std::string extension =
      dot == absl::string_view::npos || dot == 0
          ? std::string()
          : absl::AsciiStrToLower(parts.name.substr(dot + 1));
  return IsAnyOf(extension, {"md", "markdown", "mdx", "rst", "adoc", "asciidoc",
                             "org", "rdoc", "textile"}) ||
         IsAnyOf(absl::AsciiStrToUpper(parts.stem),
                 {"README", "CHANGELOG", "CHANGES", "HISTORY", "NEWS",
                  "LICENSE", "LICENCE", "COPYING", "AUTHORS", "CONTRIBUTORS",
                  "CONTRIBUTING", "NOTICE", "PATENTS", "CODE_OF_CONDUCT"});
}

bool IsTest(const PathParts& parts) {
  return AnyDirectoryIsOneOf(
             parts, {"test", "tests", "testing", "testdata", "test_data",
                     "__tests__", "spec", "specs", "unittests"}) ||
         EndsWithAnyOf(parts.stem,
                       {"_test", "_tests", "_unittest", "_browsertest",
                        "_perftest", "Test", "Tests", ".test", ".spec"}) ||
         absl::StartsWith(parts.name, "test_");
}

}  // namespace

absl::string_view RankTierName(RankTier tier) {
  switch (tier) {
    case RankTier::kSource:
      return "source";
    case RankTier::kTest:
      return "test";
    case RankTier::kDocumentation:
      return "documentation";
    case RankTier::kThirdParty:
      return "third_party";
    case RankTier::kGenerated:
      return "generated";
    case RankTier::kLongLines:
      return "long_lines";
  }
  return "unknown";
}

RankTier ClassifyTier(absl::string_view path, bool long_lines) {
  if (long_lines) return RankTier::kLongLines;
  const PathParts parts = SplitPath(path);
  if (IsGenerated(parts)) return RankTier::kGenerated;
  if (IsThirdParty(parts)) return RankTier::kThirdParty;
  if (IsDocumentation(parts)) return RankTier::kDocumentation;
  if (IsTest(parts)) return RankTier::kTest;
  return RankTier::kSource;
}

uint16_t ComputePrior(absl::string_view path, uint64_t size) {
  const int64_t depth = std::count(path.begin(), path.end(), '/');
  const int64_t size_class =
      std::max<int64_t>(static_cast<int64_t>(std::bit_width(size)) - 12, 0);
  const int64_t prior = 65535 - 4096 * std::min<int64_t>(depth, 8) -
                        2048 * std::min<int64_t>(size_class, 15);
  return static_cast<uint16_t>(prior);
}

RankKey ComputeRankKey(absl::string_view hint_path, uint64_t size,
                       bool long_lines, const Oid& oid) {
  return RankKey{.tier = ClassifyTier(hint_path, long_lines),
                 .prior = ComputePrior(hint_path, size),
                 .oid = oid};
}

}  // namespace ferret

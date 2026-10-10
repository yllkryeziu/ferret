// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "bench/ferret/result_set.h"

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "openssl/sha.h"

#include "ferret/search/search_types.h"

namespace ferret_bench {

std::vector<LineRef> MatchedLines(const ferret::SearchResponse& response) {
  std::vector<LineRef> lines;
  for (const ferret::FileResult& file : response.files) {
    for (const ferret::ResultLine& line : file.lines) {
      if (line.is_match) lines.push_back({file.path, line.number});
    }
  }
  std::sort(lines.begin(), lines.end());
  lines.erase(std::unique(lines.begin(), lines.end()), lines.end());
  return lines;
}

std::string LineSetDigest(absl::Span<const LineRef> lines) {
  SHA256_CTX context;
  SHA256_Init(&context);
  std::string entry;
  for (const LineRef& line : lines) {
    entry.clear();
    absl::StrAppend(&entry, line.path, absl::string_view("\0", 1), line.line,
                    "\n");
    SHA256_Update(&context, entry.data(), entry.size());
  }
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest;
  SHA256_Final(digest.data(), &context);
  return absl::BytesToHexString(absl::string_view(
      reinterpret_cast<const char*>(digest.data()), digest.size()));
}

}  // namespace ferret_bench

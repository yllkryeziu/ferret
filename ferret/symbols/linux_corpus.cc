// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Scans a read-only Linux tree to measure C symbol extraction and failures.

#include <sys/stat.h>

#include <cinttypes>
#include <cstdio>
#include <string>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/log/initialize.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "ferret/base/env.h"
#include "ferret/symbols/symbol_extractor.h"
#include "ferret/symbols/symbol_kind.h"
#include "ferret/text/language.h"

namespace ferret {
namespace {

// Does not follow symlinks, including directory symlinks, outside the tree.
absl::Status CollectSources(absl::string_view root,
                            std::vector<std::string>& files) {
  std::vector<std::string> pending = {std::string(root)};
  while (!pending.empty()) {
    const std::string directory = std::move(pending.back());
    pending.pop_back();
    const auto children = Env::Default()->GetChildren(directory);
    if (!children.ok()) return children.status();
    for (const std::string& child : *children) {
      const std::string path = absl::StrCat(directory, "/", child);
      struct stat info;
      if (lstat(path.c_str(), &info) != 0) {
        return absl::ErrnoToStatus(errno, absl::StrCat("lstat(", path, ")"));
      }
      if (S_ISDIR(info.st_mode)) {
        pending.push_back(path);
      } else if (S_ISREG(info.st_mode) &&
                 (child.ends_with(".c") || child.ends_with(".h"))) {
        files.push_back(path);
      }
    }
  }
  return absl::OkStatus();
}

int Scan(absl::string_view root) {
  const absl::Time start = absl::Now();
  std::vector<std::string> files;
  const absl::Status listing = CollectSources(root, files);
  if (!listing.ok()) {
    std::fprintf(stderr, "%s\n", listing.ToString().c_str());
    return 1;
  }
  SymbolExtractor extractor;
  int64_t bytes = 0;
  int64_t errors = 0;
  absl::btree_map<absl::StatusCode, int64_t> error_codes;
  absl::btree_map<SymbolKind, int64_t> kinds;
  for (const std::string& path : files) {
    const auto content = Env::Default()->ReadFile(path);
    if (!content.ok()) {
      ++errors;
      ++error_codes[content.status().code()];
      continue;
    }
    bytes += content->size();
    const auto symbols = extractor.Extract(Language::kC, *content);
    if (!symbols.ok()) {
      ++errors;
      ++error_codes[symbols.status().code()];
      if (symbols.status().code() == absl::StatusCode::kResourceExhausted) {
        std::printf("%s: %s\n", path.c_str(),
                    std::string(symbols.status().message()).c_str());
      }
      continue;
    }
    for (const ExtractedSymbol& symbol : *symbols) ++kinds[symbol.kind];
  }
  const double seconds = absl::ToDoubleSeconds(absl::Now() - start);
  std::printf("files: %zu, bytes: %" PRId64
              ", time: %.3f s, MB/s: %.3f, errors: %" PRId64 "\n",
              files.size(), bytes, seconds, bytes / 1e6 / seconds, errors);
  for (const auto& [kind, count] : kinds) {
    std::printf("%s: %" PRId64 "\n", std::string(SymbolKindName(kind)).c_str(),
                count);
  }
  for (const auto& [code, count] : error_codes) {
    std::printf("error %s: %" PRId64 "\n",
                std::string(absl::StatusCodeToString(code)).c_str(), count);
  }
  return 0;
}

}  // namespace
}  // namespace ferret

int main(int argc, char** argv) {
  absl::InitializeLog();
  if (argc != 2) {
    std::fprintf(stderr, "usage: linux_corpus ROOT\n");
    return 1;
  }
  return ferret::Scan(argv[1]);
}

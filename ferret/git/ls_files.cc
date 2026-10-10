// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/ls_files.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "ferret/git/internal/git_fields.h"
#include "ferret/git/oid.h"

namespace ferret {
namespace {

// "<tag> <mode> <oid> <stage>": 1 + 1 + 6 + 1 + 40 + 1 + 1 bytes.
constexpr size_t kMetaSize = 51;
constexpr size_t kModeOffset = 2;
constexpr size_t kOidOffset = 9;
constexpr size_t kStageOffset = 50;

absl::Status MalformedRecord(int index, absl::string_view record,
                             absl::string_view problem) {
  return absl::InvalidArgumentError(
      absl::StrCat("git ls-files record ", index, ": ", problem, " in \"",
                   absl::CHexEscape(record.substr(0, 120)), "\""));
}

absl::StatusOr<LsFilesEntry> ParseRecord(int index, absl::string_view record) {
  const size_t tab = record.find('\t');
  if (tab != kMetaSize || record[1] != ' ' || record[kOidOffset - 1] != ' ' ||
      record[kStageOffset - 1] != ' ') {
    return MalformedRecord(index, record, "bad field layout");
  }
  LsFilesEntry entry;
  const char tag = record[0];
  entry.assume_unchanged = absl::ascii_islower(static_cast<unsigned char>(tag));
  const char stage = record[kStageOffset];
  if (stage < '0' || stage > '3') {
    return MalformedRecord(index, record, "bad stage");
  }
  entry.stage = stage - '0';
  switch (absl::ascii_toupper(static_cast<unsigned char>(tag))) {
    case 'H':
      if (entry.stage != 0) return MalformedRecord(index, record, "bad stage");
      break;
    case 'S':
      if (entry.stage != 0) return MalformedRecord(index, record, "bad stage");
      entry.skip_worktree = true;
      break;
    case 'M':
      if (entry.stage == 0) return MalformedRecord(index, record, "bad stage");
      break;
    default:
      return MalformedRecord(index, record, "unknown tag");
  }
  const std::optional<uint32_t> mode =
      git_internal::ParseGitMode(record.substr(kModeOffset, 6));
  if (!mode.has_value()) return MalformedRecord(index, record, "bad mode");
  entry.mode = *mode;
  absl::StatusOr<Oid> oid = Oid::FromHex(record.substr(kOidOffset, 40));
  if (!oid.ok()) return MalformedRecord(index, record, "bad object id");
  entry.oid = *oid;
  const absl::string_view path = record.substr(tab + 1);
  if (!git_internal::IsNormalizedRepoPath(path)) {
    return MalformedRecord(index, record, "bad path");
  }
  entry.path = std::string(path);
  return entry;
}

}  // namespace

absl::StatusOr<std::vector<LsFilesEntry>> ParseLsFiles(
    absl::string_view output) {
  if (!output.empty() && output.back() != '\0') {
    return absl::InvalidArgumentError(
        "git ls-files output: last record is not NUL-terminated");
  }
  std::vector<LsFilesEntry> entries;
  size_t start = 0;
  int index = 1;
  while (start < output.size()) {
    const size_t end = output.find('\0', start);
    ABSL_ASSIGN_OR_RETURN(
        LsFilesEntry entry,
        ParseRecord(index++, output.substr(start, end - start)));
    entries.push_back(std::move(entry));
    start = end + 1;
  }
  return entries;
}

}  // namespace ferret

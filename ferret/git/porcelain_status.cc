// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/porcelain_status.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"

#include "ferret/git/internal/git_fields.h"
#include "ferret/git/oid.h"

namespace ferret {
namespace {

// The largest record prefix: "u <XY> <sub> <m1> <m2> <m3> <mW> <h1> <h2>
// <h3>" is ten space-terminated fields before the path.
constexpr int kMaxFields = 10;

absl::Status MalformedRecord(int index, absl::string_view record,
                             absl::string_view problem) {
  return absl::InvalidArgumentError(
      absl::StrCat("git status record ", index, ": ", problem, " in \"",
                   absl::CHexEscape(record.substr(0, 120)), "\""));
}

// Splits off `count` space-terminated fields; the remainder (the path, which
// may itself contain spaces) goes to `rest`. Returns false if the record has
// fewer fields.
bool SplitFields(absl::string_view record, int count,
                 std::array<absl::string_view, kMaxFields>& fields,
                 absl::string_view& rest) {
  for (int i = 0; i < count; ++i) {
    const size_t space = record.find(' ');
    if (space == absl::string_view::npos) return false;
    fields[static_cast<size_t>(i)] = record.substr(0, space);
    record.remove_prefix(space + 1);
  }
  rest = record;
  return true;
}

bool IsStatusCode(char c) { return absl::StrContains(".MTADRCU", c); }

// Checks <XY> and <sub> and copies X and Y into `entry`.
bool ParseCodes(absl::string_view xy, absl::string_view sub,
                PorcelainStatusEntry& entry) {
  if (xy.size() != 2 || !IsStatusCode(xy[0]) || !IsStatusCode(xy[1])) {
    return false;
  }
  if (sub.size() != 4 || (sub[0] != 'N' && sub[0] != 'S')) return false;
  entry.index_status = xy[0];
  entry.worktree_status = xy[1];
  return true;
}

bool AllModes(absl::Span<const absl::string_view> fields) {
  for (const absl::string_view field : fields) {
    if (!git_internal::ParseGitMode(field).has_value()) return false;
  }
  return true;
}

bool AllObjectIds(absl::Span<const absl::string_view> fields) {
  for (const absl::string_view field : fields) {
    if (!Oid::FromHex(field).ok()) return false;
  }
  return true;
}

absl::StatusOr<PorcelainStatusEntry> ParseTrackedRecord(
    int index, absl::string_view record, PorcelainStatusKind kind) {
  // Changed: <XY> <sub> <mH> <mI> <mW> <hH> <hI>; unmerged adds a third mode
  // and object id for the conflict's three stages.
  const bool unmerged = kind == PorcelainStatusKind::kUnmerged;
  const int modes = unmerged ? 4 : 3;
  const int ids = unmerged ? 3 : 2;
  std::array<absl::string_view, kMaxFields> fields;
  absl::string_view path;
  if (!SplitFields(record.substr(2), 2 + modes + ids, fields, path)) {
    return MalformedRecord(index, record, "too few fields");
  }
  PorcelainStatusEntry entry;
  entry.kind = kind;
  const absl::Span<const absl::string_view> all(fields);
  if (!ParseCodes(fields[0], fields[1], entry)) {
    return MalformedRecord(index, record, "bad status code");
  }
  if (!AllModes(all.subspan(2, static_cast<size_t>(modes)))) {
    return MalformedRecord(index, record, "bad mode");
  }
  if (!AllObjectIds(all.subspan(2 + static_cast<size_t>(modes),
                                static_cast<size_t>(ids)))) {
    return MalformedRecord(index, record, "bad object id");
  }
  // mW is the last mode field.
  const std::optional<uint32_t> worktree_mode =
      git_internal::ParseGitMode(fields[1 + static_cast<size_t>(modes)]);
  if (!worktree_mode) return MalformedRecord(index, record, "bad mode");
  entry.worktree_mode = *worktree_mode;
  if (!git_internal::IsNormalizedRepoPath(path)) {
    return MalformedRecord(index, record, "bad path");
  }
  entry.path = std::string(path);
  return entry;
}

absl::StatusOr<PorcelainStatusEntry> ParseUntrackedRecord(
    int index, absl::string_view record, PorcelainStatusKind kind) {
  const absl::string_view path = record.substr(2);
  if (!git_internal::IsNormalizedRepoPath(absl::StripSuffix(path, "/"))) {
    return MalformedRecord(index, record, "bad path");
  }
  PorcelainStatusEntry entry;
  entry.kind = kind;
  entry.path = std::string(path);
  return entry;
}

// Returns the entry for `record`, or nullopt for a header record.
absl::StatusOr<std::optional<PorcelainStatusEntry>> ParseRecord(
    int index, absl::string_view record) {
  if (record.size() < 2 || record[1] != ' ') {
    return MalformedRecord(index, record, "bad record type");
  }
  switch (record[0]) {
    case '#':
      return std::nullopt;
    case '1':
      return ParseTrackedRecord(index, record, PorcelainStatusKind::kChanged);
    case 'u':
      return ParseTrackedRecord(index, record, PorcelainStatusKind::kUnmerged);
    case '?':
      return ParseUntrackedRecord(index, record,
                                  PorcelainStatusKind::kUntracked);
    case '!':
      return ParseUntrackedRecord(index, record, PorcelainStatusKind::kIgnored);
    case '2':
      return MalformedRecord(index, record,
                             "rename record despite --no-renames");
    default:
      return MalformedRecord(index, record, "bad record type");
  }
}

}  // namespace

absl::StatusOr<std::vector<PorcelainStatusEntry>> ParsePorcelainStatus(
    absl::string_view output) {
  if (!output.empty() && output.back() != '\0') {
    return absl::InvalidArgumentError(
        "git status output: last record is not NUL-terminated");
  }
  std::vector<PorcelainStatusEntry> entries;
  size_t start = 0;
  int index = 1;
  while (start < output.size()) {
    const size_t end = output.find('\0', start);
    ABSL_ASSIGN_OR_RETURN(
        std::optional<PorcelainStatusEntry> entry,
        ParseRecord(index++, output.substr(start, end - start)));
    if (entry.has_value()) entries.push_back(*std::move(entry));
    start = end + 1;
  }
  return entries;
}

}  // namespace ferret

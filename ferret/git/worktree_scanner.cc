// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/worktree_scanner.h"

#include <_stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/fcntl.h>
#include <sys/stat.h>
#include <sys/syslimits.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "ferret/base/env.h"
#include "ferret/base/subprocess.h"
#include "ferret/base/thread_pool.h"
#include "ferret/git/blob_hash.h"
#include "ferret/git/internal/path_pattern.h"
#include "ferret/git/ls_files.h"
#include "ferret/git/oid.h"
#include "ferret/git/porcelain_status.h"

namespace ferret {
namespace {

constexpr uint32_t kRegular = 0100644;
constexpr uint32_t kExecutable = 0100755;
constexpr uint32_t kSymlink = 0120000;
constexpr uint32_t kGitlink = 0160000;

absl::StatusOr<SubprocessResult> Git(absl::string_view root,
                                     std::vector<std::string> args,
                                     absl::string_view input = "") {
  std::vector<std::string> argv = {"git", "-C", std::string(root),
                                   "--no-optional-locks"};
  const std::string command = args[0];
  argv.insert(argv.end(), std::make_move_iterator(args.begin()),
              std::make_move_iterator(args.end()));
  ABSL_ASSIGN_OR_RETURN(
      SubprocessResult result,
      RunSubprocess(argv, {.env_overrides = {"LC_ALL=C"},
                           .stdin_data = std::string(input)}));
  if (command != "var" && (result.exit_code > 1 ||
                           (result.exit_code == 1 && command != "config"))) {
    return absl::FailedPreconditionError(
        absl::StrCat("git ", command, " in ", root, " exited with ",
                     result.exit_code, ": ", result.stderr_data));
  }
  return result;
}

std::string Lower(absl::string_view text) {
  std::string folded(text);
  for (char& c : folded) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return folded;
}

// Check the spelling of the existing .git entry without writing to the volume.
bool CaseInsensitive(absl::string_view root) {
  struct stat a, b;
  const std::string git = absl::StrCat(root, "/.git");
  const std::string alternate = absl::StrCat(root, "/.GIT");
  return stat(git.c_str(), &a) == 0 && stat(alternate.c_str(), &b) == 0 &&
         a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}

// F_GETPATH supplies the real directory-entry spelling on case-insensitive
// macOS volumes. Compare only the root-relative part of the canonical path.
bool SpellingMatches(absl::string_view canonical_root, absl::string_view path) {
#if defined(__APPLE__) || defined(__linux__)
  const std::string absolute = absl::StrCat(canonical_root, "/", path);
  const int fd = open(absolute.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) return false;
  char actual[PATH_MAX];
#ifdef __APPLE__
  const int length =
      fcntl(fd, F_GETPATH, actual) == 0 ? static_cast<int>(strlen(actual)) : -1;
#else
  const int length = static_cast<int>(readlink(
      absl::StrCat("/proc/self/fd/", fd).c_str(), actual, sizeof(actual)));
#endif
  close(fd);
  const std::string prefix = absl::StrCat(canonical_root, "/");
  const absl::string_view resolved(actual, length < 0 ? 0 : length);
  return length >= 0 && absl::StartsWith(resolved, prefix) &&
         resolved.substr(prefix.size()) == path;
#else
  return false;
#endif
}

struct StatResult {
  std::optional<int64_t> size;
  std::optional<absl::Time> mtime;
  bool regular = false;
  bool spelling_matches = true;
  absl::Status error;
};

StatResult StatFile(absl::string_view root, absl::string_view path) {
  const std::string absolute = absl::StrCat(root, "/", path);
  struct stat st;
  if (lstat(absolute.c_str(), &st) != 0) {
    if (errno == ENOENT || errno == ENOTDIR) return {};
    return {.error = absl::ErrnoToStatus(
                errno, absl::StrCat("lstat(", absolute, ")"))};
  }
#ifdef __APPLE__
  const timespec mtime = st.st_mtimespec;
#else
  const timespec mtime = st.st_mtim;
#endif
  return {.size = static_cast<int64_t>(st.st_size),
          .mtime = absl::FromUnixSeconds(mtime.tv_sec) +
                   absl::Nanoseconds(mtime.tv_nsec),
          .regular = S_ISREG(st.st_mode)};
}

// A bare attribute may invoke a macro. Assignments to unrelated attributes
// cannot invoke a macro; neither can an unset or unspecified attribute.
bool MayConvert(absl::string_view attrs) {
  for (absl::string_view token :
       absl::StrSplit(attrs, absl::ByAnyChar(" \t\r"), absl::SkipEmpty())) {
    if (token[0] == '-' || token[0] == '!') continue;
    if (absl::StrContains(token, "eol") || absl::StrContains(token, "text") ||
        absl::StrContains(token, "filter") ||
        absl::StrContains(token, "ident") ||
        absl::StrContains(token, "working-tree-encoding")) {
      return true;
    }
    if (!absl::StrContains(token, '=')) return true;
  }
  return false;
}

// Returns paths for check-attr. A relative attributes file only affects its
// directory subtree; unknown syntax conservatively selects the whole subtree.
absl::StatusOr<absl::flat_hash_set<std::string>> AttributeCandidates(
    absl::string_view root, const std::vector<ScannedWorktreeFile>& files,
    const std::vector<std::string>& attribute_files,
    absl::string_view info_path, absl::string_view global_path,
    absl::string_view system_path, bool unknown_source) {
  absl::flat_hash_set<std::string> candidates;
  if (unknown_source) {
    for (const auto& file : files) candidates.insert(file.path);
    return candidates;
  }
  std::vector<std::string> sources = attribute_files;
  if (!info_path.empty()) sources.emplace_back(info_path);
  if (!global_path.empty()) sources.emplace_back(global_path);
  if (!system_path.empty()) sources.emplace_back(system_path);
  for (const std::string& file : sources) {
    const bool external =
        file == info_path || file == global_path || file == system_path;
    const std::string absolute =
        external ? file : absl::StrCat(root, "/", file);
    ABSL_ASSIGN_OR_RETURN(bool exists, Env::Default()->FileExists(absolute));
    if (!exists) continue;
    ABSL_ASSIGN_OR_RETURN(std::string content,
                          Env::Default()->ReadFile(absolute));
    const size_t slash = file.find_last_of('/');
    const std::string directory =
        external || slash == std::string::npos ? "" : file.substr(0, slash);
    const std::string prefix =
        directory.empty() ? "" : absl::StrCat(directory, "/");
    const auto first = std::lower_bound(
        files.begin(), files.end(), prefix,
        [](const ScannedWorktreeFile& entry, absl::string_view value) {
          return entry.path < value;
        });
    // '0' immediately follows '/' in the bytewise path ordering.
    const std::string after_prefix = absl::StrCat(directory, "0");
    const auto last =
        directory.empty()
            ? files.end()
            : std::lower_bound(
                  first, files.end(), after_prefix,
                  [](const ScannedWorktreeFile& entry,
                     absl::string_view value) { return entry.path < value; });
    for (absl::string_view line : absl::StrSplit(content, '\n')) {
      if (line.empty() || line[0] == '#') continue;
      // A backslash can quote whitespace in a pattern; over-select on such
      // lines. Attribute macros may expand to conversion attributes.
      const bool uncertain =
          absl::StartsWith(line, "[attr]") || absl::StrContains(line, "\\");
      const size_t end = line.find_first_of(" \t\r");
      if (end == absl::string_view::npos) continue;
      if (!uncertain && !MayConvert(line.substr(end + 1))) continue;
      git_internal::PathPattern pattern(line.substr(0, end), directory);
      for (auto it = first; it != last; ++it) {
        if (uncertain || pattern.Matches(it->path, false, true)) {
          candidates.insert(it->path);
        }
      }
    }
  }
  return candidates;
}

// git var resolves Git's configured and platform-specific attributes paths.
// Older Git versions lack these variables; a missing system path is unknown.
absl::StatusOr<std::optional<std::string>> AttributeVar(
    absl::string_view root, absl::string_view name) {
  ABSL_ASSIGN_OR_RETURN(SubprocessResult result,
                        Git(root, {"var", std::string(name)}));
  if (result.exit_code != 0 || result.stdout_data.empty()) return std::nullopt;
  std::string path = result.stdout_data;
  if (path.back() == '\n') path.pop_back();
  if (path.empty() || path[0] != '/') return std::nullopt;
  return path;
}

absl::StatusOr<std::optional<std::string>> GlobalAttributesPath(
    absl::string_view root) {
  ABSL_ASSIGN_OR_RETURN(std::optional<std::string> path,
                        AttributeVar(root, "GIT_ATTR_GLOBAL"));
  if (path) return path;
  ABSL_ASSIGN_OR_RETURN(
      SubprocessResult config,
      Git(root, {"config", "--path", "--get", "core.attributesFile"}));
  if (config.exit_code == 0) {
    std::string configured = config.stdout_data;
    if (!configured.empty() && configured.back() == '\n') configured.pop_back();
    if (absl::StartsWith(configured, "/")) return configured;
    return std::nullopt;
  }
  const char* xdg = getenv("XDG_CONFIG_HOME");
  if (xdg != nullptr && xdg[0] != '\0') {
    if (xdg[0] != '/') return std::nullopt;
    return absl::StrCat(xdg, "/git/attributes");
  }
  const char* home = getenv("HOME");
  if (home == nullptr || home[0] != '/') return std::nullopt;
  return absl::StrCat(home, "/.config/git/attributes");
}

absl::StatusOr<absl::flat_hash_set<std::string>> ConversionPaths(
    absl::string_view root, const std::vector<ScannedWorktreeFile>& files,
    const std::vector<std::string>& attribute_files, ScanReport& report) {
  ABSL_ASSIGN_OR_RETURN(SubprocessResult config,
                        Git(root, {"config", "--null", "--get-regexp",
                                   "^core\\.(autocrlf|eol)$"}));
  bool global_conversion = false;
  for (absl::string_view field :
       absl::StrSplit(config.stdout_data, '\0', absl::SkipEmpty())) {
    const size_t split = field.find('\n');
    if (split == absl::string_view::npos) {
      global_conversion = true;
      break;
    }
    const std::string value = Lower(field.substr(split + 1));
    if (field.substr(0, split) == "core.autocrlf" && value != "false" &&
        value != "input") {
      global_conversion = true;
    }
    if (field.substr(0, split) == "core.eol" && value == "crlf") {
      global_conversion = true;
    }
  }
  absl::flat_hash_set<std::string> converted;
  if (global_conversion) {
    for (const auto& file : files) converted.insert(file.path);
    return converted;
  }
  ABSL_ASSIGN_OR_RETURN(
      SubprocessResult info,
      Git(root, {"rev-parse", "--git-path", "info/attributes"}));
  std::string info_path = info.stdout_data;
  if (!info_path.empty() && info_path.back() == '\n') info_path.pop_back();
  if (!absl::StartsWith(info_path, "/")) {
    info_path = absl::StrCat(root, "/", info_path);
  }
  ABSL_ASSIGN_OR_RETURN(std::optional<std::string> global_path,
                        GlobalAttributesPath(root));
  std::optional<std::string> system_path;
  if (getenv("GIT_ATTR_NOSYSTEM") == nullptr) {
    ABSL_ASSIGN_OR_RETURN(system_path, AttributeVar(root, "GIT_ATTR_SYSTEM"));
  }
  ABSL_ASSIGN_OR_RETURN(
      absl::flat_hash_set<std::string> candidates,
      AttributeCandidates(
          root, files, attribute_files, info_path, global_path.value_or(""),
          system_path.value_or(""),
          !global_path ||
              (getenv("GIT_ATTR_NOSYSTEM") == nullptr && !system_path)));
  report.attribute_candidates = static_cast<int64_t>(candidates.size());
  if (candidates.empty()) return converted;
  std::string input;
  for (const std::string& path : candidates) {
    input.append(path);
    input.push_back('\0');
  }
  ABSL_ASSIGN_OR_RETURN(SubprocessResult checked,
                        Git(root,
                            {"check-attr", "-z", "--stdin", "eol", "text",
                             "filter", "ident", "working-tree-encoding"},
                            input));
  std::vector<absl::string_view> fields =
      absl::StrSplit(checked.stdout_data, '\0');
  if (!fields.empty() && fields.back().empty()) fields.pop_back();
  if (fields.size() % 3 != 0) {
    return absl::InternalError("git check-attr: incomplete output");
  }
  for (size_t i = 0; i < fields.size(); i += 3) {
    const absl::string_view attr = fields[i + 1];
    const absl::string_view value = fields[i + 2];
    if (value == "unspecified" || value == "unset") continue;
    if ((attr == "eol" && value == "crlf") || attr == "filter" ||
        attr == "ident" || attr == "working-tree-encoding") {
      converted.insert(std::string(fields[i]));
    }
    // On macOS and Linux native checkout is LF. With autocrlf=false/input
    // and core.eol!=crlf, text alone does not introduce CRLF conversion.
  }
  return converted;
}

void CountRead(ScanReadReason reason, ScanReport& report) {
  switch (reason) {
    case ScanReadReason::kDirty:
      ++report.dirty;
      break;
    case ScanReadReason::kUntracked:
      ++report.untracked;
      break;
    case ScanReadReason::kUnmerged:
      ++report.unmerged;
      break;
    case ScanReadReason::kAssumeUnchanged:
      ++report.assume_unchanged;
      break;
    case ScanReadReason::kConversion:
      ++report.conversion;
      break;
  }
}

}  // namespace

absl::StatusOr<WorktreeScan> ScanWorktree(absl::string_view root,
                                          const ScanWorktreeOptions& options) {
#ifdef NDEBUG
  const double verify_fraction = options.verify_hint_fraction.value_or(0.0);
#else
  const double verify_fraction = options.verify_hint_fraction.value_or(0.01);
#endif
  if (!(verify_fraction >= 0 && verify_fraction <= 1) ||
      options.lstat_threads < 1) {
    return absl::InvalidArgumentError("invalid worktree scan options");
  }
  WorktreeScan scan;
  const absl::Time git_start = absl::Now();
  ABSL_ASSIGN_OR_RETURN(SubprocessResult format,
                        Git(root, {"rev-parse", "--show-object-format"}));
  if (format.stdout_data != "sha1\n") {
    return absl::FailedPreconditionError(
        "worktree uses a non-SHA-1 object format");
  }
  ABSL_ASSIGN_OR_RETURN(
      SubprocessResult status_output,
      Git(root, {"status", "--porcelain=v2", "-z", "-uall", "--no-renames"}));
  ABSL_ASSIGN_OR_RETURN(SubprocessResult index_output,
                        Git(root, {"ls-files", "-s", "-v", "-z"}));
  scan.report.git_commands = absl::Now() - git_start;
  ABSL_ASSIGN_OR_RETURN(std::vector<PorcelainStatusEntry> statuses,
                        ParsePorcelainStatus(status_output.stdout_data));
  ABSL_ASSIGN_OR_RETURN(std::vector<LsFilesEntry> index,
                        ParseLsFiles(index_output.stdout_data));
  absl::flat_hash_map<std::string, PorcelainStatusEntry> changes;
  for (PorcelainStatusEntry& entry : statuses) {
    changes[entry.path] = std::move(entry);
  }
  absl::flat_hash_map<std::string, LsFilesEntry> tracked;
  std::vector<std::string> attribute_files;
  for (LsFilesEntry& entry : index) {
    if (entry.mode == kGitlink) scan.nested_repositories.push_back(entry.path);
    if (entry.skip_worktree) {
      ++scan.report.skip_worktree;
      continue;
    }
    if (entry.mode == kSymlink) {
      ++scan.report.symlink;
      continue;
    }
    if (entry.mode == kGitlink) {
      ++scan.report.gitlink;
      continue;
    }
    if (entry.mode != kRegular && entry.mode != kExecutable) continue;
    if (absl::EndsWith(entry.path, ".gitattributes") &&
        (entry.path == ".gitattributes" ||
         absl::EndsWith(entry.path, "/.gitattributes"))) {
      attribute_files.push_back(entry.path);
    }
    tracked[entry.path] = std::move(entry);
  }
  const bool insensitive = CaseInsensitive(root);
  absl::flat_hash_map<std::string, int> folded_counts;
  if (insensitive) {
    for (const auto& [path, entry] : tracked) ++folded_counts[Lower(path)];
  }
  for (const auto& [path, entry] : tracked) {
    scan.files.push_back({.path = path});
  }
  for (const auto& [path, change] : changes) {
    if (change.kind != PorcelainStatusKind::kUntracked ||
        tracked.contains(path)) {
      continue;
    }
    // With -uall, git reports a directory only for a nested repository.
    if (!path.empty() && path.back() == '/') {
      scan.nested_repositories.push_back(path.substr(0, path.size() - 1));
      continue;
    }
    scan.files.push_back({.path = path});
    if (absl::EndsWith(path, "/.gitattributes") || path == ".gitattributes") {
      attribute_files.push_back(path);
    }
  }
  std::sort(scan.files.begin(), scan.files.end(),
            [](const auto& a, const auto& b) { return a.path < b.path; });
  std::sort(scan.nested_repositories.begin(), scan.nested_repositories.end());
  scan.nested_repositories.erase(std::unique(scan.nested_repositories.begin(),
                                             scan.nested_repositories.end()),
                                 scan.nested_repositories.end());
  const absl::Time attr_start = absl::Now();
  ABSL_ASSIGN_OR_RETURN(
      absl::flat_hash_set<std::string> converted,
      ConversionPaths(root, scan.files, attribute_files, scan.report));
  scan.report.attributes = absl::Now() - attr_start;
  const absl::Time stat_start = absl::Now();
  std::vector<StatResult> stats(scan.files.size());
  std::string canonical_root;
  if (insensitive) {
    char resolved[PATH_MAX];
    if (realpath(std::string(root).c_str(), resolved) == nullptr) {
      return absl::ErrnoToStatus(errno, "realpath(worktree root)");
    }
    canonical_root = resolved;
  }
  ThreadPool pool(options.lstat_threads, ThreadQos::kUtility);
  pool.ParallelFor(static_cast<int64_t>(scan.files.size()), 128,
                   options.lstat_threads, [&](int, int64_t begin, int64_t end) {
                     for (int64_t i = begin; i < end; ++i) {
                       const size_t n = static_cast<size_t>(i);
                       stats[n] = StatFile(root, scan.files[n].path);
                       if (insensitive && stats[n].regular &&
                           tracked.contains(scan.files[n].path)) {
                         stats[n].spelling_matches = SpellingMatches(
                             canonical_root, scan.files[n].path);
                       }
                     }
                   });
  scan.report.lstat = absl::Now() - stat_start;
  std::vector<bool> nonregular(scan.files.size());
  for (size_t i = 0; i < scan.files.size(); ++i) {
    ScannedWorktreeFile& file = scan.files[i];
    StatResult& stat = stats[i];
    ABSL_RETURN_IF_ERROR(stat.error);
    if (!stat.spelling_matches) {
      if (folded_counts[Lower(file.path)] > 1) {
        ++scan.report.case_collisions;
        nonregular[i] = true;
        continue;
      }
      stat.size.reset();
      stat.mtime.reset();
      stat.regular = false;
    }
    if (!stat.regular && stat.size.has_value()) {
      ++scan.report.not_regular;
      nonregular[i] = true;
      continue;
    }
    file.size = stat.size;
    file.mtime = stat.mtime;
    const auto tracked_it = tracked.find(file.path);
    ScanReadReason reason = ScanReadReason::kUntracked;
    if (tracked_it != tracked.end()) {
      const LsFilesEntry& entry = tracked_it->second;
      const auto changed = changes.find(file.path);
      if (entry.stage != 0 ||
          (changed != changes.end() &&
           changed->second.kind == PorcelainStatusKind::kUnmerged)) {
        reason = ScanReadReason::kUnmerged;
      } else if (entry.assume_unchanged) {
        reason = ScanReadReason::kAssumeUnchanged;
      } else if (!stat.regular || (changed != changes.end() &&
                                   changed->second.worktree_status != '.')) {
        reason = ScanReadReason::kDirty;
      } else if (converted.contains(file.path)) {
        reason = ScanReadReason::kConversion;
      } else {
        file.skip_hint = entry.oid;
        ++scan.report.hinted;
      }
    }
    if (!file.skip_hint) {
      file.read_reason = reason;
      CountRead(reason, scan.report);
    }
  }
  std::vector<ScannedWorktreeFile> regular_files;
  regular_files.reserve(scan.files.size());
  for (size_t i = 0; i < scan.files.size(); ++i) {
    if (!nonregular[i]) regular_files.push_back(std::move(scan.files[i]));
  }
  scan.files = std::move(regular_files);
  scan.report.files = static_cast<int64_t>(scan.files.size());
  for (const ScannedWorktreeFile& file : scan.files) {
    if (!file.skip_hint) continue;
    const Oid::Bytes& oid_bytes = file.skip_hint->bytes();
    const uint32_t sample = (static_cast<uint32_t>(oid_bytes[0]) << 24) |
                            (static_cast<uint32_t>(oid_bytes[1]) << 16) |
                            (static_cast<uint32_t>(oid_bytes[2]) << 8) |
                            oid_bytes[3];
    const double draw = static_cast<double>(sample) / 4294967296.0;
    if (draw >= verify_fraction) continue;
    absl::StatusOr<std::string> bytes =
        Env::Default()->ReadFile(absl::StrCat(root, "/", file.path));
    if (!bytes.ok() || HashBlob(*bytes) != *file.skip_hint) {
      return absl::InternalError(
          absl::StrCat("worktree skip hint mismatch: ", file.path));
    }
  }
  return scan;
}

}  // namespace ferret

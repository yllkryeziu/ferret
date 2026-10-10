// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Ripgrep's NUL-delimited path and newline-delimited match stream. The Git
// view and file eligibility are cached by filesystem identity and timestamps.
#include "ferret/engine/ripgrep_backend.h"

#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>

#include <cstdio>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <unistd.h>

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "re2/re2.h"

#include "ferret/base/env.h"
#include "ferret/base/subprocess.h"
#include "ferret/engine/engine.h"
#include "ferret/git/blob_hash.h"
#include "ferret/git/oid.h"
#include "ferret/git/path_glob.h"
#include "ferret/git/worktree_files.h"
#include "ferret/index/rank_key.h"
#include "ferret/regex/compile.h"
#include "ferret/search/formatter.h"
#include "ferret/search/kept_lines.h"
#include "ferret/search/ranking.h"
#include "ferret/search/search_types.h"
#include "ferret/text/eligibility.h"
#include "ferret/text/language.h"
#include "ferret/text/lexer.h"
#include "ferret/text/line_counter.h"
#include "ferret/text/trigram_extractor.h"
#include "ferret/workspace/workspace.h"

namespace ferret {
namespace {

absl::StatusOr<std::string> CanonicalRoot(absl::string_view root) {
  // NOLINTNEXTLINE(misc-include-cleaner)
  char* canonical = realpath(std::string(root).c_str(), nullptr);
  if (canonical == nullptr) return absl::NotFoundError(absl::StrCat(root));
  std::string result(canonical);
  free(canonical);
  return result;
}

struct FileStamp {
  uint64_t device = 0;
  uint64_t inode = 0;
  int64_t size = 0;
  int64_t mtime_sec = 0;
  int64_t mtime_nsec = 0;
  int64_t ctime_sec = 0;
  int64_t ctime_nsec = 0;
  uint32_t mode = 0;
  friend bool operator==(const FileStamp&, const FileStamp&) = default;
};

FileStamp Stamp(absl::string_view path) {
  struct stat info = {};
  if (lstat(std::string(path).c_str(), &info) != 0) return {};
#if defined(__APPLE__)
  const auto modified = info.st_mtimespec;
  const auto changed = info.st_ctimespec;
#else
  const auto modified = info.st_mtim;
  const auto changed = info.st_ctim;
#endif
  return {.device = static_cast<uint64_t>(info.st_dev),
          .inode = static_cast<uint64_t>(info.st_ino),
          .size = info.st_size,
          .mtime_sec = modified.tv_sec,
          .mtime_nsec = modified.tv_nsec,
          .ctime_sec = changed.tv_sec,
          .ctime_nsec = changed.tv_nsec,
          .mode = static_cast<uint32_t>(info.st_mode)};
}

struct CachedFile {
  FileStamp stamp;
  SkipReason skip_reason = SkipReason::kEmpty;
  bool long_lines = false;
  Oid oid;
  Language language = Language::kUnknown;
};

absl::StatusOr<std::string> GitValue(absl::string_view root,
                                     std::vector<std::string> args) {
  std::vector<std::string> argv = {"git", "-C", std::string(root)};
  argv.insert(argv.end(), args.begin(), args.end());
  ABSL_ASSIGN_OR_RETURN(SubprocessResult result, RunSubprocess(argv));
  if (result.exit_code != 0) return std::string();
  while (!result.stdout_data.empty() && result.stdout_data.back() == '\n') {
    result.stdout_data.pop_back();
  }
  return result.stdout_data;
}

}  // namespace

// A root's mutex serializes metadata refresh, not rg or result collection.
// A copy of the path set outlives refreshes in concurrent searches.
class RipgrepCache {
 public:
  explicit RipgrepCache(std::string root) : root_(std::move(root)) {}

  absl::StatusOr<std::shared_ptr<const absl::flat_hash_set<std::string>>> Paths(
      absl::string_view binary) {
    absl::MutexLock lock(mu_);
    bool fresh = paths_ != nullptr;
    for (const auto& [path, stamp] : inputs_) {
      if (!(Stamp(path) == stamp)) {
        fresh = false;
        break;
      }
    }
    if (fresh) return paths_;

    auto paths = std::make_shared<absl::flat_hash_set<std::string>>();
    auto git = ListWorktreeFiles(root_);
    if (git.ok()) {
      for (const std::string& path : *git) paths->insert(path);
    } else {
      // A plain directory has no Git index or ignore rules.
      SubprocessOptions options;
      options.cwd = root_;
      ABSL_ASSIGN_OR_RETURN(
          SubprocessResult files,
          RunSubprocess({std::string(binary), "--no-config", "--files", "-0",
                         "--hidden", "--no-ignore", "--glob=!.git", "."},
                        options));
      if (files.exit_code != 0) return absl::InternalError(files.stderr_data);
      absl::string_view listing = files.stdout_data;
      while (!listing.empty()) {
        const size_t end = listing.find('\0');
        if (end == absl::string_view::npos) break;
        absl::string_view path = listing.substr(0, end);
        if (path.starts_with("./")) path.remove_prefix(2);
        paths->insert(std::string(path));
        listing.remove_prefix(end + 1);
      }
    }
    std::set<std::string> inputs = {root_, absl::StrCat(root_, "/.git"),
                                    absl::StrCat(root_, "/.gitignore")};
    if (git.ok()) {
      for (const auto& args :
           {std::vector<std::string>{"rev-parse", "--path-format=absolute",
                                     "--git-path", "index"},
            {"rev-parse", "--path-format=absolute", "--git-path",
             "info/exclude"},
            {"rev-parse", "--path-format=absolute", "--git-path", "config"},
            {"config", "--path", "--get", "core.excludesFile"}}) {
        ABSL_ASSIGN_OR_RETURN(std::string path, GitValue(root_, args));
        if (!path.empty()) inputs.insert(std::move(path));
      }
      // Git's default global excludes path applies without core.excludesFile.
      if (const char* home = getenv("HOME")) {
        inputs.insert(absl::StrCat(home, "/.config/git/ignore"));
        inputs.insert(absl::StrCat(home, "/.gitconfig"));
      }
    }
    // Directory mtimes detect new/deleted files and newly created ignore
    // files. Stat every ignore path, including absent ones, on every search.
    for (const std::string& path : *paths) {
      size_t slash = 0;
      while ((slash = path.find('/', slash)) != std::string::npos) {
        const std::string dir = absl::StrCat(root_, "/", path.substr(0, slash));
        inputs.insert(dir);
        inputs.insert(absl::StrCat(dir, "/.gitignore"));
        ++slash;
      }
    }
    inputs_.clear();
    for (const std::string& path : inputs) inputs_.emplace(path, Stamp(path));
    for (auto it = files_.begin(); it != files_.end();) {
      if (!paths->contains(it->first)) {
        files_.erase(it++);
      } else {
        ++it;
      }
    }
    paths_ = paths;
    return paths_;
  }

  // The number of files the latest search left out because rg's bytes
  // disagreed with a later read of them.
  void SetSkippedPaths(int64_t count) {
    absl::MutexLock lock(mu_);
    skipped_paths_ = count;
  }

  int64_t SkippedPaths() const {
    absl::MutexLock lock(mu_);
    return skipped_paths_;
  }

  absl::StatusOr<CachedFile> File(absl::string_view path,
                                  const FileStamp& stamp) {
    absl::MutexLock lock(mu_);
    const std::string key(path);
    auto it = files_.find(key);
    if (it != files_.end() && it->second.stamp == stamp) return it->second;
    CachedFile result{.stamp = stamp};
    if (S_ISREG(stamp.mode)) {
      result.skip_reason = SkipReasonForSize(stamp.size, {});
      if (result.skip_reason == SkipReason::kNone) {
        ABSL_ASSIGN_OR_RETURN(
            std::string content,
            Env::Default()->ReadFile(absl::StrCat(root_, "/", path)));
        TrigramExtractor extractor;
        const Eligibility eligibility =
            CheckEligibility(content, {}, extractor);
        result.skip_reason = eligibility.skip_reason;
        result.long_lines = eligibility.long_lines;
        if (result.skip_reason == SkipReason::kNone) {
          result.oid = HashBlob(content);
          result.language = DetectLanguage(path, content);
        }
      }
    }
    files_[key] = result;
    return result;
  }

 private:
  std::string root_;
  mutable absl::Mutex mu_;
  std::shared_ptr<const absl::flat_hash_set<std::string>> paths_
      ABSL_GUARDED_BY(mu_);
  std::map<std::string, FileStamp> inputs_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, CachedFile> files_ ABSL_GUARDED_BY(mu_);
  int64_t skipped_paths_ ABSL_GUARDED_BY(mu_) = 0;
};

namespace {

struct RgFile {
  std::string path;
  std::string content;
  std::vector<KeptLine> kept;
  std::vector<ResultLine> raw_lines;
  CachedFile cached;
  int64_t matches = 0;
  bool eligible = false;
  std::unique_ptr<MatchClassifier> classifier;
};

// Plain rg emits path NUL line-number:byte-offset:line LF for each match.
// No JSON object survives; only kept lines (and, for context, one file) do.
class RgCollector {
 public:
  RgCollector(absl::string_view root, const SearchRequest& request,
              const CompiledPattern& pattern,
              const absl::flat_hash_set<std::string>& paths,
              const std::optional<PathGlob>& glob, RipgrepCache& cache,
              SearchResponse& response, bool profile)
      : root_(root),
        request_(request),
        pattern_(pattern),
        paths_(paths),
        glob_(glob),
        cache_(cache),
        response_(response),
        profile_(profile) {}

  absl::Status Consume(absl::string_view bytes) {
    const absl::Time start_time = profile_ ? absl::Now() : absl::InfinitePast();
    pending_.append(bytes);
    size_t start = 0;
    while (true) {
      const size_t zero = pending_.find('\0', start);
      if (zero == std::string::npos) break;
      const size_t end = pending_.find('\n', zero + 1);
      if (end == std::string::npos) break;
      ABSL_RETURN_IF_ERROR(
          Match(absl::string_view(pending_).substr(start, end - start)));
      start = end + 1;
    }
    pending_.erase(0, start);
    if (pending_.size() > (size_t{4} << 20)) {
      return absl::ResourceExhaustedError("rg line too large");
    }
    if (profile_) consume_time_ += absl::Now() - start_time;
    return absl::OkStatus();
  }

  double ConsumeSeconds() const { return absl::ToDoubleSeconds(consume_time_); }
  double EligibilitySeconds() const {
    return absl::ToDoubleSeconds(eligibility_time_);
  }
  double ClassificationSeconds() const {
    return absl::ToDoubleSeconds(classification_time_);
  }
  double RankingSeconds() const { return absl::ToDoubleSeconds(ranking_time_); }

  absl::Status Finish() {
    if (!pending_.empty()) return absl::DataLossError("truncated rg output");
    return EndFile();
  }

 private:
  absl::Status BeginFile(absl::string_view path) {
    ABSL_RETURN_IF_ERROR(EndFile());
    file_ = RgFile{};
    if (path.starts_with("./")) path.remove_prefix(2);
    file_.path = path;
    if (!paths_.contains(file_.path) ||
        (glob_ && !glob_->Matches(file_.path))) {
      return absl::OkStatus();
    }
    const FileStamp stamp = Stamp(absl::StrCat(root_, "/", path));
    if (!S_ISREG(stamp.mode)) return absl::OkStatus();
    const absl::Time eligibility_start =
        profile_ ? absl::Now() : absl::InfinitePast();
    const auto cached = cache_.File(path, stamp);
    if (profile_) eligibility_time_ += absl::Now() - eligibility_start;
    if (!cached.ok()) return absl::OkStatus();
    file_.cached = *cached;
    if (file_.cached.skip_reason != SkipReason::kNone) return absl::OkStatus();
    file_.eligible = true;
    if (!request_.text_only || request_.context_lines != 0) {
      const absl::Time classification_start =
          profile_ ? absl::Now() : absl::InfinitePast();
      const auto content =
          Env::Default()->ReadFile(absl::StrCat(root_, "/", path));
      if (!content.ok()) {
        file_.eligible = false;
        return absl::OkStatus();
      }
      file_.content = *content;
      if (!request_.text_only) {
        file_.classifier = std::make_unique<MatchClassifier>(
            file_.cached.language, file_.content, LexState::kCode);
      }
      if (profile_) classification_time_ += absl::Now() - classification_start;
    }
    return absl::OkStatus();
  }

  absl::Status Match(absl::string_view record) {
    const size_t zero = record.find('\0');
    if (zero == absl::string_view::npos) return absl::DataLossError("rg path");
    absl::string_view path = record.substr(0, zero);
    if (path.starts_with("./")) path.remove_prefix(2);
    if (file_.path != path) ABSL_RETURN_IF_ERROR(BeginFile(path));
    if (!file_.eligible) return absl::OkStatus();
    const size_t colon = record.find(':', zero + 1);
    const size_t second = record.find(':', colon + 1);
    if (colon == absl::string_view::npos || second == absl::string_view::npos) {
      return absl::DataLossError("rg match header");
    }
    int64_t number = 0;
    size_t offset = 0;
    const auto number_text = record.substr(zero + 1, colon - zero - 1);
    const auto offset_text = record.substr(colon + 1, second - colon - 1);
    if (std::from_chars(number_text.data(),
                        number_text.data() + number_text.size(), number)
                .ec != std::errc{} ||
        std::from_chars(offset_text.data(),
                        offset_text.data() + offset_text.size(), offset)
                .ec != std::errc{}) {
      return absl::DataLossError("rg match number or offset");
    }
    ++file_.matches;
    if (static_cast<int64_t>(file_.kept.size()) >=
            request_.max_lines_per_file ||
        kept_lines_ >= request_.max_lines) {
      return absl::OkStatus();
    }
    absl::string_view line = record.substr(second + 1);
    // CRLF's CR is part of the indexed line, not the line terminator.
    const bool raw = request_.text_only && request_.context_lines == 0;
    size_t match_begin = 0;
    size_t match_end = 0;
    // The raw formatter prints lines up to its default max_line_bytes (no
    // caller changes it) verbatim. It cuts longer lines around the match, so
    // only those need RE2 submatch recovery.
    if (!raw || line.size() > FormatOptions().max_line_bytes) {
      absl::string_view match;
      if (!pattern_.matcher().Match(line, 0, line.size(), RE2::UNANCHORED,
                                    &match, 1)) {
        return absl::DataLossError("rg line does not match compiled pattern");
      }
      match_begin = static_cast<size_t>(match.data() - line.data());
      match_end = match_begin + match.size();
    }
    if (raw) {
      file_.raw_lines.push_back({.number = number,
                                 .text = std::string(line),
                                 .is_match = true,
                                 .match_begin = match_begin,
                                 .match_end = match_end});
    } else {
      // rg's offset addresses bytes on disk, including any BOM. Its printed
      // line, rather than an independently inferred end, is authoritative.
      if (offset > file_.content.size() ||
          line.size() > file_.content.size() - offset ||
          file_.content.compare(offset, line.size(), line) != 0) {
        DropInconsistentFile();
        return absl::OkStatus();
      }
      const size_t end = offset + line.size();
      const absl::Time classify_start =
          profile_ ? absl::Now() : absl::InfinitePast();
      file_.kept.push_back(
          {.number = number,
           .begin = offset,
           .end = end,
           .match_begin = match_begin,
           .match_end = match_end,
           .match_class = file_.classifier
                              ? file_.classifier->Classify(offset + match_begin,
                                                           offset + match_end)
                              : MatchClass::kCodeIdentifier});
      if (profile_) classification_time_ += absl::Now() - classify_start;
    }
    ++kept_lines_;
    return absl::OkStatus();
  }

  // The file's bytes now differ from the ones rg searched, so no kept line or
  // count of it can be trusted. One such file leaves the results of the
  // others intact; the response's stats report it.
  void DropInconsistentFile() {
    kept_lines_ -= static_cast<int64_t>(file_.kept.size());
    file_ = RgFile{.path = std::move(file_.path)};
    ++response_.stats.inconsistent_files;
  }

  absl::Status EndFile() {
    if (!file_.eligible || file_.matches == 0) return absl::OkStatus();
    ++response_.num_matching_files;
    response_.num_matching_lines += file_.matches;
    if (file_.kept.empty() && file_.raw_lines.empty()) return absl::OkStatus();
    const absl::Time ranking_start =
        profile_ ? absl::Now() : absl::InfinitePast();
    const RankKey rank =
        ComputeRankKey(file_.path, file_.cached.stamp.size,
                       file_.cached.long_lines, file_.cached.oid);
    const size_t slash = file_.path.rfind('/');
    response_.files.push_back(FileResult{
        .path = file_.path,
        .oid = file_.cached.oid,
        .tier = rank.tier,
        .prior = rank.prior,
        .name_matches = RE2::PartialMatch(
            file_.path.substr(slash == std::string::npos ? 0 : slash + 1),
            pattern_.matcher()),
        .num_matching_lines = file_.matches,
        .lines = file_.raw_lines.empty()
                     ? WithContext(file_.content,
                                   CountNewlines(file_.content) +
                                       (!file_.content.empty() &&
                                        file_.content.back() != '\n'),
                                   file_.kept, request_.context_lines)
                     : std::move(file_.raw_lines)});
    if (profile_) ranking_time_ += absl::Now() - ranking_start;
    return absl::OkStatus();
  }

  absl::string_view root_;
  const SearchRequest& request_;
  const CompiledPattern& pattern_;
  const absl::flat_hash_set<std::string>& paths_;
  const std::optional<PathGlob>& glob_;
  RipgrepCache& cache_;
  SearchResponse& response_;
  bool profile_;
  absl::Duration consume_time_ = absl::ZeroDuration();
  absl::Duration eligibility_time_ = absl::ZeroDuration();
  absl::Duration classification_time_ = absl::ZeroDuration();
  absl::Duration ranking_time_ = absl::ZeroDuration();
  RgFile file_;
  std::string pending_;
  int64_t kept_lines_ = 0;
};

absl::StatusOr<std::string> ResolveRg(absl::string_view explicit_path) {
  if (!explicit_path.empty()) return std::string(explicit_path);
  const char* env = getenv("FERRET_RG");
  if (env != nullptr && *env != '\0') return std::string(env);
  struct stat self = {};
#if defined(__APPLE__)
  uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  std::string executable(size, '\0');
  if (_NSGetExecutablePath(executable.data(), &size) == 0) {
    stat(executable.c_str(), &self);
  }
#else
  stat("/proc/self/exe", &self);
#endif
  const char* raw_path = getenv("PATH");
  absl::string_view path = raw_path == nullptr ? "" : raw_path;
  while (true) {
    const size_t colon = path.find(':');
    const absl::string_view directory = path.substr(0, colon);
    std::string candidate =
        absl::StrCat(directory.empty() ? "." : directory, "/rg");
    struct stat info;
    if (stat(candidate.c_str(), &info) == 0 && S_ISREG(info.st_mode) &&
        // X_OK is declared in the platform unistd headers.
        // NOLINTNEXTLINE(misc-include-cleaner)
        access(candidate.c_str(), X_OK) == 0 &&
        !(self.st_ino == info.st_ino && self.st_dev == info.st_dev)) {
      return candidate;
    }
    if (colon == absl::string_view::npos) break;
    path.remove_prefix(colon + 1);
  }
  return absl::NotFoundError("rg not found on PATH");
}

}  // namespace

absl::Status RipgrepBackend::Register(absl::string_view root) {
  ABSL_ASSIGN_OR_RETURN(std::string canonical, CanonicalRoot(root));
  absl::MutexLock lock(mu_);
  if (caches_.contains(canonical)) return absl::OkStatus();
  caches_.emplace(canonical, std::make_shared<RipgrepCache>(canonical));
  roots_.push_back(std::move(canonical));
  return absl::OkStatus();
}

std::vector<std::string> RipgrepBackend::Roots() const {
  absl::MutexLock lock(mu_);
  return roots_;
}

absl::StatusOr<std::vector<WorktreeIndexStatus>> RipgrepBackend::IndexStatus()
    const {
  std::vector<WorktreeIndexStatus> status;
  absl::MutexLock lock(mu_);
  status.reserve(roots_.size());
  for (const std::string& root : roots_) {
    status.push_back(
        {.root = root, .skipped_paths = caches_.at(root)->SkippedPaths()});
  }
  return status;
}

absl::StatusOr<WorkspaceSearchResult> RipgrepBackend::Search(
    absl::string_view root, const SearchRequest& request,
    const SearchConsistency& /*consistency*/) const {
  std::shared_ptr<RipgrepCache> cache;
  {
    absl::MutexLock lock(mu_);
    const auto it = caches_.find(std::string(root));
    if (it == caches_.end()) {
      return absl::NotFoundError(absl::StrCat("unknown worktree: ", root));
    }
    cache = it->second;
  }
  if (request.context_lines < 0 || request.context_lines > 100 ||
      request.max_lines_per_file < 1 || request.max_lines < 0 ||
      request.max_files < 0 || request.max_verify_bytes < 1) {
    return absl::InvalidArgumentError("invalid search limits");
  }
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<CompiledPattern> pattern,
      CompilePattern({.pattern = request.pattern,
                      .literal = !request.regex,
                      .case_sensitive = request.case_sensitive}));
  std::optional<PathGlob> glob;
  if (!request.path_glob.empty()) {
    ABSL_ASSIGN_OR_RETURN(glob, PathGlob::Parse(request.path_glob));
  }
  ABSL_ASSIGN_OR_RETURN(const std::string binary, ResolveRg(rg_path_));
  const bool profile = getenv("FERRET_RGB_PROFILE") != nullptr;
  const absl::Time file_set_start =
      profile ? absl::Now() : absl::InfinitePast();
  ABSL_ASSIGN_OR_RETURN(const auto paths, cache->Paths(binary));
  const double file_set_seconds =
      profile ? absl::ToDoubleSeconds(absl::Now() - file_set_start) : 0;
  SearchResponse response;
  RgCollector collector(root, request, *pattern, *paths, glob, *cache, response,
                        profile);
  // --no-config removes user syntax/ignore overrides. --hidden traverses
  // tracked dotfiles. --no-ignore removes .ignore, .rgignore, and global rg
  // rules; the Git path set rejects ignored untracked files, but retains
  // tracked files ignored by Git. --glob omits Git metadata. --text lets the
  // eligibility check decide NUL policy. --encoding=none keeps disk offsets.
  // --null disambiguates even paths containing ':' or newlines. --no-heading
  // prefixes every line; --color never leaves text bytes untouched.
  std::vector<std::string> argv = {binary,
                                   "--no-config",
                                   "--null",
                                   "--line-number",
                                   "--byte-offset",
                                   "--no-heading",
                                   "--color",
                                   "never",
                                   "--hidden",
                                   "--no-ignore",
                                   "--glob=!.git",
                                   "--text",
                                   "--encoding=none",
                                   "--max-filesize=1048576"};
  if (!request.regex) argv.push_back("--fixed-strings");
  argv.push_back(request.case_sensitive ? "--case-sensitive" : "--ignore-case");
  argv.push_back("--");
  argv.push_back(request.pattern);
  argv.push_back(".");
  SubprocessOptions options;
  options.cwd = std::string(root);
  const absl::Time rg_start = profile ? absl::Now() : absl::InfinitePast();
  ABSL_ASSIGN_OR_RETURN(
      SubprocessResult result,
      StreamSubprocess(argv, options, [&](absl::string_view chunk) {
        return collector.Consume(chunk);
      }));
  if (result.exit_code != 0 && result.exit_code != 1) {
    if (result.exit_code == 2 &&
        absl::StrContains(result.stderr_data, "regex parse error")) {
      return absl::InvalidArgumentError(result.stderr_data);
    }
    return absl::InternalError(
        absl::StrCat("rg exited ", result.exit_code, ": ", result.stderr_data));
  }
  ABSL_RETURN_IF_ERROR(collector.Finish());
  cache->SetSkippedPaths(response.stats.inconsistent_files);
  const double rg_seconds =
      profile ? absl::ToDoubleSeconds(absl::Now() - rg_start) : 0;
  RankFiles(response.files);
  if (profile) {
    fprintf(stderr,
            "profile pattern=%s file_set=%.3f rg_io=%.3f parse_collect=%.3f "
            "eligibility=%.3f classification=%.3f ranking=%.3f\n",
            request.pattern.c_str(), file_set_seconds,
            rg_seconds - collector.ConsumeSeconds(),
            collector.ConsumeSeconds() - collector.EligibilitySeconds() -
                collector.ClassificationSeconds() - collector.RankingSeconds(),
            collector.EligibilitySeconds(), collector.ClassificationSeconds(),
            collector.RankingSeconds());
  }
  if (static_cast<int64_t>(response.files.size()) > request.max_files) {
    response.files.resize(static_cast<size_t>(request.max_files));
  }
  return WorkspaceSearchResult{.response = std::move(response)};
}

}  // namespace ferret

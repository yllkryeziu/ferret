// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Registration and rescans (the bulk lane) run in five phases, each timed in
// WorkspaceStats:
//  1. scan: ScanWorktree for a git root, ListDirectoryFiles otherwise;
//  2. classify: intern the paths, skip those whose scanned size makes them
//     ineligible, and resolve each git skip hint against the indexed and
//     known-ineligible Oids, so a second worktree at the same commit reads
//     nothing;
//  3. read: on the reader pool, stat every unhinted path and compare it with
//     the signature of its previous read, then read, check, and hash the
//     paths that remain, keeping only metadata;
//  4. write: deduplicate new Oids, keeping each one's bytewise-smallest path
//     (its first-seen path, which fixes its rank), sort them in rank order,
//     and reread them on the pool at most bulk_prefetch_bytes ahead of the
//     writer, which checks each Oid and streams the content into direct
//     segments of at most bulk_segment_bytes (or Index::Add(kBulk) below one
//     memtable's flush size);
//  5. publish: apply the updates newer than each path's last applied read,
//     delete the paths the scan omitted, and publish one view batch.
// A Notify batch runs phases 3 to 5 for its paths on the interactive thread,
// lstat'ing each path before opening it, keeping the bytes it read, and
// adding them through Index::Add(kInteractive).
//
// A failed index write leaves the paths it would have changed as they were,
// records them as failed, and holds back the worktree's generations. Most
// such failures are the sticky flush error of Index::Add, which clears only
// when Index::Flush retries it, so the bulk lane retries Flush with backoff
// and then rereads the failed paths.

#include "ferret/workspace/workspace.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <sys/fcntl.h>
#include <sys/stat.h>
#include <sys/syslimits.h>
#include <sys/unistd.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

#include "ferret/base/bitset.h"
#include "ferret/base/env.h"
#include "ferret/base/subprocess.h"
#include "ferret/base/thread_pool.h"
#include "ferret/git/blob_hash.h"
#include "ferret/git/ignore.h"
#include "ferret/git/oid.h"
#include "ferret/git/path_glob.h"
#include "ferret/git/worktree_files.h"
#include "ferret/git/worktree_scanner.h"
#include "ferret/index/blob_registry.h"
#include "ferret/index/doc_precomputer.h"
#include "ferret/index/index.h"
#include "ferret/index/manifest.h"
#include "ferret/index/prepared_document.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/read_state.h"
#include "ferret/index/segment_writer.h"
#include "ferret/index/version_set.h"
#include "ferret/search/search_types.h"
#include "ferret/search/searcher.h"
#include "ferret/symbols/symbol_extractor.h"
#include "ferret/text/eligibility.h"
#include "ferret/text/language.h"
#include "ferret/text/trigram_extractor.h"
#include "ferret/text/utf8.h"
#include "ferret/watch/internal/cookie_directory.h"
#include "ferret/watch/watch_trace.h"
#include "ferret/watch/watcher.h"
#include "ferret/workspace/path_table.h"
#include "ferret/workspace/view_state.h"

namespace ferret {
namespace {

// Notify paths per interactive batch, and per generation.
constexpr size_t kMaxNotifyBatchPaths = 256;
// A file whose fstat signature changes across this many reads is unstable.
constexpr int kReadAttempts = 3;
// Lookups per acquisition of a mutex that Notify and Search also take.
constexpr size_t kLockBatch = 1024;
// A file whose ctime is this close to the start of its read may change again
// without changing its signature, on a file system with timestamps of this
// granularity; its signature is not reused.
constexpr int64_t kRacyNanos = 100'000'000;
constexpr int64_t kRacyWholeSecondNanos = 2'000'000'000;

// Identifies one version of a file, as the fstat before and after a read
// must agree for the read to be stable.
struct FileSignature {
  int64_t device = 0;
  int64_t inode = 0;
  int64_t size = 0;
  int64_t mtime_nanos = 0;
  int64_t ctime_nanos = 0;

  friend bool operator==(const FileSignature&, const FileSignature&) = default;
};

int64_t Nanos(const timespec& time) {
  return int64_t{time.tv_sec} * 1'000'000'000 + time.tv_nsec;
}

FileSignature Signature(const struct stat& st) {
#ifdef __APPLE__
  const timespec& mtime = st.st_mtimespec;
  const timespec& ctime = st.st_ctimespec;
#else
  const timespec& mtime = st.st_mtim;
  const timespec& ctime = st.st_ctim;
#endif
  return {.device = static_cast<int64_t>(st.st_dev),
          .inode = static_cast<int64_t>(st.st_ino),
          .size = static_cast<int64_t>(st.st_size),
          .mtime_nanos = Nanos(mtime),
          .ctime_nanos = Nanos(ctime)};
}

// Whether a later write could leave the signature of a file read at
// `read_nanos` (wall time) unchanged. A file system with whole-second
// timestamps is assumed when both are whole seconds.
bool IsRacy(const FileSignature& signature, int64_t read_nanos) {
  constexpr int64_t kSecond = 1'000'000'000;
  const bool coarse = signature.mtime_nanos % kSecond == 0 &&
                      signature.ctime_nanos % kSecond == 0;
  return read_nanos - signature.ctime_nanos <
         (coarse ? kRacyWholeSecondNanos : kRacyNanos);
}

#if !defined(O_NOFOLLOW_ANY) || !defined(AT_SYMLINK_NOFOLLOW_ANY)
// Without kernel support, reject a path with a symlinked or missing ancestor.
bool HasSymlinkAncestor(absl::string_view root, absl::string_view path) {
  for (size_t slash = path.find('/'); slash != absl::string_view::npos;
       slash = path.find('/', slash + 1)) {
    struct stat st;
    const std::string parent = absl::StrCat(root, "/", path.substr(0, slash));
    if (lstat(parent.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return true;
  }
  return false;
}
#endif

// The root is canonical, so it contains no symlink; neither may the path.
// O_NONBLOCK keeps a FIFO from blocking the open.
int OpenNoFollow(absl::string_view root, absl::string_view path) {
  const std::string full = absl::StrCat(root, "/", path);
#ifdef O_NOFOLLOW_ANY
  return open(full.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC | O_NOFOLLOW_ANY);
#else
  if (HasSymlinkAncestor(root, path)) {
    errno = ELOOP;
    return -1;
  }
  return open(full.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC | O_NOFOLLOW);
#endif
}

// Sets errno on failure, to ELOOP for a symlinked ancestor.
bool StatNoFollow(absl::string_view root, absl::string_view path,
                  struct stat& st) {
  const std::string full = absl::StrCat(root, "/", path);
#ifdef AT_SYMLINK_NOFOLLOW_ANY
  return fstatat(AT_FDCWD, full.c_str(), &st, AT_SYMLINK_NOFOLLOW_ANY) == 0;
#else
  if (HasSymlinkAncestor(root, path)) {
    errno = ELOOP;
    return false;
  }
  return lstat(full.c_str(), &st) == 0;
#endif
}

// Errors that mean the path is not a readable file, not a failed read.
bool IsAbsentErrno(int error) {
  return error == ENOENT || error == ENOTDIR || error == ELOOP ||
         error == EACCES || error == EPERM || error == ENAMETOOLONG;
}

absl::string_view Dirname(absl::string_view path) {
  const size_t slash = path.rfind('/');
  return slash == absl::string_view::npos ? "" : path.substr(0, slash);
}

bool IsIgnoreFile(absl::string_view path) {
  const size_t slash = path.rfind('/');
  const absl::string_view name =
      slash == absl::string_view::npos ? path : path.substr(slash + 1);
  return name == ".gitignore" || name == ".ferretignore";
}

// A path with a ".git" component, which git never lists.
bool IsGitDirPath(absl::string_view path) {
  for (const absl::string_view component : absl::StrSplit(path, '/')) {
    if (component == ".git") return true;
  }
  return false;
}

// Whether `path` or a directory above it is in `dirs`.
bool UnderAny(const absl::flat_hash_set<std::string>& dirs,
              absl::string_view path) {
  if (dirs.empty()) return false;
  for (absl::string_view dir = path; !dir.empty(); dir = Dirname(dir)) {
    if (dirs.contains(dir)) return true;
  }
  return false;
}

// Whether a directory above `path`, below the root, holds a `.git` entry:
// git then treats it as a nested repository and lists nothing inside.
// `cache` remembers the directories already checked.
bool InsideNestedRepository(absl::string_view root, absl::string_view path,
                            absl::flat_hash_map<std::string, bool>& cache) {
  for (absl::string_view dir = Dirname(path); !dir.empty();
       dir = Dirname(dir)) {
    const auto [it, inserted] = cache.try_emplace(dir, false);
    if (inserted) {
      struct stat st;
      const std::string git = absl::StrCat(root, "/", dir, "/.git");
      it->second = lstat(git.c_str(), &st) == 0;
    }
    if (it->second) return true;
  }
  return false;
}

// `children` counts, for each directory with visible files below it, the
// visible files and such directories directly inside it; the root is not
// counted. These record that `path` became visible or stopped being visible,
// in O(1) unless directories above it become nonempty or empty, O(depth) at
// most.
void AddVisible(absl::flat_hash_map<std::string, int>& children,
                absl::string_view path) {
  for (absl::string_view dir = Dirname(path); !dir.empty();
       dir = Dirname(dir)) {
    if (children[dir]++ > 0) return;
  }
}

// Adds every directory above `path`, below the root, to `dirs`. A directory
// already present has its ancestors present too.
void AddAncestors(absl::flat_hash_set<std::string>& dirs,
                  absl::string_view path) {
  for (absl::string_view dir = Dirname(path);
       !dir.empty() && !dirs.contains(dir); dir = Dirname(dir)) {
    dirs.emplace(dir);
  }
}

void RemoveVisible(absl::flat_hash_map<std::string, int>& children,
                   absl::string_view path) {
  for (absl::string_view dir = Dirname(path); !dir.empty();
       dir = Dirname(dir)) {
    const auto it = children.find(dir);
    if (it == children.end() || --it->second > 0) return;
    children.erase(it);
  }
}

// Whether names on the volume holding `root` compare case-insensitively, as
// on default APFS.
bool CaseInsensitiveVolume(const std::string& root) {
#ifdef _PC_CASE_SENSITIVE
  return pathconf(root.c_str(), _PC_CASE_SENSITIVE) == 0;
#else
  return false;
#endif
}

// Whether the file open at `fd`, opened as `path` below a root of
// `root_size` bytes, has that spelling in its directory entries. On a
// case-insensitive volume open() also accepts other spellings. Only macOS
// reports the stored spelling (F_GETPATH); elsewhere, or if that fails, the
// spelling is accepted.
bool SpellingMatches(int fd, size_t root_size, absl::string_view path) {
#ifdef F_GETPATH
  char actual[PATH_MAX];
  if (fcntl(fd, F_GETPATH, actual) != 0) return true;
  const absl::string_view resolved(actual);
  return resolved.size() == root_size + 1 + path.size() &&
         absl::EndsWith(resolved, path);
#else
  return true;
#endif
}

void Accumulate(WorkspaceStats& total, const WorkspaceStats& delta) {
  total.interactive_files_read += delta.interactive_files_read;
  total.bulk_files_read += delta.bulk_files_read;
  total.bulk_files_reread += delta.bulk_files_reread;
  total.bulk_bytes_read += delta.bulk_bytes_read;
  total.bulk_bytes_reread += delta.bulk_bytes_reread;
  total.hint_reads += delta.hint_reads;
  total.hints_reused += delta.hints_reused;
  total.signatures_reused += delta.signatures_reused;
  total.interactive_bytes_ingested += delta.interactive_bytes_ingested;
  total.bulk_bytes_ingested += delta.bulk_bytes_ingested;
  total.unstable_files += delta.unstable_files;
  total.read_errors += delta.read_errors;
  total.changed_before_write += delta.changed_before_write;
  for (size_t i = 0; i < total.skipped.size(); ++i) {
    total.skipped[i] += delta.skipped[i];
  }
  total.ingest_failures += delta.ingest_failures;
  if (!delta.last_ingest_error.ok()) {
    total.last_ingest_error = delta.last_ingest_error;
  }
  total.ingest_retries += delta.ingest_retries;
  total.scan_time += delta.scan_time;
  total.classify_time += delta.classify_time;
  total.read_time += delta.read_time;
  total.segment_write_time += delta.segment_write_time;
  total.memtable_add_time += delta.memtable_add_time;
  total.publish_time += delta.publish_time;
  total.bulk_wait_time += delta.bulk_wait_time;
  total.bulk_add_time += delta.bulk_add_time;
  total.bulk_finish_time += delta.bulk_finish_time;
  total.bulk_reread_thread_time += delta.bulk_reread_thread_time;
  total.bulk_symbol_wait_thread_time += delta.bulk_symbol_wait_thread_time;
  total.bulk_symbol_parse_thread_time += delta.bulk_symbol_parse_thread_time;
  total.watcher_failures += delta.watcher_failures;
  total.sync_timeouts += delta.sync_timeouts;
  total.index_lock_timeouts += delta.index_lock_timeouts;
  if (!delta.last_watcher_error.ok()) {
    total.last_watcher_error = delta.last_watcher_error;
  }
}

// What decides ignore rules for Notify paths in a git worktree.
struct GitRootInfo {
  // The worktree's top level, as git reports it.
  std::string toplevel;
  // $GIT_DIR/info/exclude, which linked worktrees share.
  std::string info_exclude;
  // core.excludesFile, or git's default when unset.
  std::string global_excludes;
  bool ignore_case = false;
};

absl::StatusOr<SubprocessResult> RunGit(absl::string_view root,
                                        std::vector<std::string> args) {
  std::vector<std::string> argv = {"git", "-C", std::string(root)};
  argv.insert(argv.end(), std::make_move_iterator(args.begin()),
              std::make_move_iterator(args.end()));
  return RunSubprocess(argv,
                       {.env_overrides = {"GIT_OPTIONAL_LOCKS=0", "LC_ALL=C"}});
}

std::string FirstLine(absl::string_view text) {
  return std::string(text.substr(0, text.find('\n')));
}

// Git's default core.excludesFile, derived from the environment as git does,
// like ScanWorktree's global attributes file.
std::string DefaultGlobalExcludes() {
  const char* xdg = getenv("XDG_CONFIG_HOME");
  if (xdg != nullptr && xdg[0] != '\0') {
    return xdg[0] == '/' ? absl::StrCat(xdg, "/git/ignore") : "";
  }
  const char* home = getenv("HOME");
  if (home == nullptr || home[0] != '/') return "";
  return absl::StrCat(home, "/.config/git/ignore");
}

// Returns nullopt if `root` is not inside a git worktree.
absl::StatusOr<std::optional<GitRootInfo>> FindGitRoot(
    const std::string& root) {
  ABSL_ASSIGN_OR_RETURN(const SubprocessResult parsed,
                        RunGit(root, {"rev-parse", "--show-toplevel",
                                      "--git-path", "info/exclude"}));
  if (parsed.exit_code != 0) {
    if (absl::StrContains(parsed.stderr_data, "not a git repository")) {
      return std::nullopt;
    }
    return absl::FailedPreconditionError(
        absl::StrCat("git rev-parse in ", root, " exited with ",
                     parsed.exit_code, ": ", parsed.stderr_data));
  }
  const std::vector<std::string> lines =
      absl::StrSplit(parsed.stdout_data, '\n', absl::SkipEmpty());
  if (lines.size() != 2) {
    return absl::InternalError(
        absl::StrCat("git rev-parse in ", root, ": unexpected output"));
  }
  GitRootInfo info;
  info.toplevel = lines[0];
  info.info_exclude = absl::StartsWith(lines[1], "/")
                          ? lines[1]
                          : absl::StrCat(root, "/", lines[1]);
  ABSL_ASSIGN_OR_RETURN(
      const SubprocessResult excludes,
      RunGit(root, {"config", "--path", "--get", "core.excludesFile"}));
  info.global_excludes = excludes.exit_code == 0
                             ? FirstLine(excludes.stdout_data)
                             : DefaultGlobalExcludes();
  ABSL_ASSIGN_OR_RETURN(
      const SubprocessResult ignore_case,
      RunGit(root, {"config", "--bool", "--get", "core.ignoreCase"}));
  info.ignore_case = ignore_case.exit_code == 0 &&
                     FirstLine(ignore_case.stdout_data) == "true";
  return info;
}

absl::StatusOr<std::string> CanonicalDirectory(absl::string_view path) {
  const std::string requested(path);
  char resolved[PATH_MAX];
  // The macOS SDK declares realpath in a header that stdlib.h includes.
  // NOLINTNEXTLINE(misc-include-cleaner)
  if (realpath(requested.c_str(), resolved) == nullptr) {
    return absl::ErrnoToStatus(errno,
                               absl::StrCat("realpath(", requested, ")"));
  }
  struct stat st;
  if (stat(resolved, &st) != 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("stat(", resolved, ")"));
  }
  if (!S_ISDIR(st.st_mode)) {
    return absl::InvalidArgumentError(
        absl::StrCat("worktree root ", resolved, " is not a directory"));
  }
  return std::string(resolved);
}

absl::StatusOr<std::string> GitCommonDirectory(const std::string& root) {
  ABSL_ASSIGN_OR_RETURN(SubprocessResult result,
                        RunGit(root, {"rev-parse", "--path-format=absolute",
                                      "--git-common-dir"}));
  if (result.exit_code != 0) {
    return absl::FailedPreconditionError("git common directory unavailable");
  }
  return CanonicalDirectory(absl::StripAsciiWhitespace(result.stdout_data));
}

// The built-in excludes of ListDirectoryFiles.
constexpr std::array<absl::string_view, 4> kDirectoryDefaultExcludes = {
    ".git/", "node_modules/", "out/", "bazel-*/"};

// Mirrors the rules of ScanWorktree (git's excludes) or ListDirectoryFiles,
// so that Notify and Rescan agree on which untracked files exist.
IgnoreOptions IgnoreOptionsFor(const std::string& root,
                               const std::optional<GitRootInfo>& git,
                               absl::string_view index_prefix) {
  IgnoreOptions options;
  options.root = root;
  if (git.has_value()) {
    options.info_exclude_path = git->info_exclude;
    options.global_excludes_path = git->global_excludes;
    options.ignore_case = git->ignore_case;
  } else {
    options.ferretignore = true;
    for (const absl::string_view pattern : kDirectoryDefaultExcludes) {
      options.default_excludes.emplace_back(pattern);
    }
  }
  if (!index_prefix.empty()) {
    options.default_excludes.push_back(absl::StrCat("/", index_prefix));
  }
  return options;
}

// Runs read(i) for i = 0, 1, ... on a pool, at most `budget` bytes ahead of
// the consumer (but always at least one read), and hands the results to
// Next() in order. Not thread-safe; `read` must be safe to call concurrently.
// The destructor waits for every scheduled read.
template <typename Result>
class ReadAhead final {
 public:
  ReadAhead(ThreadPool& pool, std::vector<int64_t> sizes, int64_t budget,
            absl::AnyInvocable<Result(size_t) const> reader)
      : pool_(pool),
        sizes_(std::move(sizes)),
        budget_(budget),
        read_(std::move(reader)),
        slots_(sizes_.size()) {}
  ReadAhead(const ReadAhead&) = delete;
  ReadAhead& operator=(const ReadAhead&) = delete;
  ~ReadAhead() {
    for (size_t i = next_; i < scheduled_; ++i) {
      slots_[i]->done.WaitForNotification();
    }
  }

  // Blocks for the next result, which is valid until the next call.
  // Requires fewer calls than `sizes` has entries.
  Result& Next() {
    if (next_ > 0) {
      outstanding_ -= sizes_[next_ - 1];
      slots_[next_ - 1].reset();
    }
    while (
        scheduled_ < sizes_.size() &&
        (scheduled_ == next_ || outstanding_ + sizes_[scheduled_] <= budget_)) {
      outstanding_ += sizes_[scheduled_];
      slots_[scheduled_] = std::make_unique<Slot>();
      pool_.Schedule([this, slot = slots_[scheduled_].get(), i = scheduled_] {
        slot->result = read_(i);
        slot->done.Notify();
      });
      ++scheduled_;
    }
    Slot& slot = *slots_[next_++];
    slot.done.WaitForNotification();
    return slot.result;
  }

 private:
  struct Slot {
    absl::Notification done;
    Result result;
  };

  ThreadPool& pool_;
  const std::vector<int64_t> sizes_;
  const int64_t budget_;
  const absl::AnyInvocable<Result(size_t) const> read_;
  std::vector<std::unique_ptr<Slot>> slots_;
  size_t next_ = 0;
  size_t scheduled_ = 0;
  int64_t outstanding_ = 0;
};

// The per-document work of one planned direct segment, done by the readers:
// each concurrent reader holds a Slot, whose precomputer accumulates postings
// and code tokens for the files it handles under provisional chunk ids (from
// `next_chunk`, shared by the segment's slots) and provisional document ids
// (the file's position in the segment's plan).
class SegmentPrecompute final {
 public:
  struct Slot {
    Slot(const SegmentWriterOptions& options, std::atomic<uint32_t>& chunks)
        : precomputer(options, extractor, chunks) {}
    TrigramExtractor extractor;
    DocumentPrecomputer precomputer;
  };

  explicit SegmentPrecompute(SegmentWriterOptions options)
      : options_(std::move(options)) {}

  Slot* Acquire() ABSL_LOCKS_EXCLUDED(mu_) {
    {
      absl::MutexLock lock(mu_);
      if (!free_.empty()) {
        Slot* slot = free_.back();
        free_.pop_back();
        return slot;
      }
    }
    auto slot = std::make_unique<Slot>(options_, next_chunk_);
    Slot* raw = slot.get();
    absl::MutexLock lock(mu_);
    all_.push_back(std::move(slot));
    return raw;
  }
  void Release(Slot* slot) ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(mu_);
    free_.push_back(slot);
  }

  // Valid once every Acquire() has been matched by a Release().
  uint32_t num_provisional_chunks() const { return next_chunk_.load(); }
  // Only after every Acquire() has been matched by a Release(), when no other
  // thread touches the slots.
  std::vector<std::unique_ptr<Slot>>& slots() ABSL_NO_THREAD_SAFETY_ANALYSIS {
    return all_;
  }

 private:
  const SegmentWriterOptions options_;
  std::atomic<uint32_t> next_chunk_{0};
  absl::Mutex mu_;
  std::vector<std::unique_ptr<Slot>> all_ ABSL_GUARDED_BY(mu_);
  std::vector<Slot*> free_ ABSL_GUARDED_BY(mu_);
};

// Symbol extractors, which are costly to warm up, shared by the reads of one
// WriteBulk call.
class SymbolExtractorPool final {
 public:
  std::unique_ptr<SymbolExtractor> Acquire() ABSL_LOCKS_EXCLUDED(mu_) {
    {
      absl::MutexLock lock(mu_);
      if (!free_.empty()) {
        auto extractor = std::move(free_.back());
        free_.pop_back();
        return extractor;
      }
    }
    return std::make_unique<SymbolExtractor>();
  }
  void Release(std::unique_ptr<SymbolExtractor> extractor)
      ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(mu_);
    free_.push_back(std::move(extractor));
  }

 private:
  absl::Mutex mu_;
  std::vector<std::unique_ptr<SymbolExtractor>> free_ ABSL_GUARDED_BY(mu_);
};

}  // namespace

struct Workspace::PathState {
  // The sequence of the read or scan whose result is applied.
  uint64_t last_applied_seq = 0;
  // The visible content; nullopt for a skipped, missing, or deleted path.
  std::optional<BlobId> blob;
  // Why the path's content is not indexed; kNone otherwise.
  SkipReason skip = SkipReason::kNone;
};

struct Workspace::FileRead {
  enum class Outcome {
    kPresent,    // A stable read of a regular file.
    kMissing,    // Absent, unreadable, a symlink, or not a regular file.
    kDirectory,  // A directory.
    kTooLarge,   // Larger than max_file_bytes; not read.
    kUnstable,   // Changed during every attempt.
    kError,      // An I/O error.
  };
  Outcome outcome = Outcome::kMissing;
  // Taken when the read started, with the wall time then.
  uint64_t seq = 0;
  int64_t started_nanos = 0;
  // Set for kPresent and kTooLarge.
  FileSignature signature;
  // Set for kPresent.
  std::string bytes;
  // Set for kPresent by readers that hash the bytes.
  std::optional<Oid> oid;
  // Only set on successful bulk rereads for a direct segment.
  std::vector<ExtractedSymbol> symbols;
  bool symbol_failed = false;
  // Set by readers that did the document's chunking and trigram work: `pre`
  // names chunks of the segment's precomputer.
  bool precomputed = false;
  PrecomputedDocument pre;
  absl::Status precompute_status;

  // The signature a later scan may trust instead of reading the file.
  std::optional<FileSignature> ReusableSignature() const {
    if (IsRacy(signature, started_nanos)) return std::nullopt;
    return signature;
  }
};

// What one batch learned about one path.
struct Workspace::Update {
  enum class Kind {
    kHint,     // A git skip hint, not yet resolved.
    kStat,     // Unhinted: needs a stat compared with the previous read.
    kRead,     // Needs a read.
    kPresent,  // Visible content: `blob`, or the new content `oid`.
    kSkipped,  // Ineligible content, which no view shows.
    kDeleted,  // Missing, not a regular file, or excluded.
    kKeep,     // Keeps the previous entry: the read failed or was unstable,
               // or its content could not be added to the index.
  };
  PathId id{};
  // Owned by the scan or the Notify batch.
  absl::string_view path;
  Kind kind = Kind::kDeleted;
  uint64_t seq = 0;
  // kHint: the hinted Oid. kPresent without a `blob`: the Oid of the content
  // just read, which is not indexed yet. Unused otherwise.
  Oid oid;
  // kPresent: the indexed content, once resolved.
  std::optional<BlobId> blob;
  // kSkipped: why, and the Oid of the content if it was read, which marks
  // hints with that Oid ineligible too.
  SkipReason skip = SkipReason::kNone;
  std::optional<Oid> skipped_oid;
  // The signature to reuse at the next scan; nullopt for hinted content, a
  // racy read, or an unreadable path.
  std::optional<FileSignature> signature;
  // kStat: the path's applied state when the batch classified it.
  std::optional<FileSignature> previous_signature;
  std::optional<BlobId> previous_blob;
  SkipReason previous_skip = SkipReason::kNone;
  // New content from a read: its metadata and size in bytes.
  PreparedDocument prepared;
  int64_t size = 0;
  // Interactive reads keep the bytes of new content.
  std::string bytes;
  // kKeep: the content was read but could not be added to the index.
  bool ingest_failed = false;
  // An interactive read found a directory, or a path with visible files below
  // it that is not a directory any more: the worktree needs a rescan.
  bool rescan = false;
};

// What a scan learned about the whole worktree, beyond its updates.
struct Workspace::ScanSummary {
  // Taken before the scan started.
  uint64_t seq = 0;
  // Tracked paths; empty for a non-git root.
  absl::flat_hash_set<PathId> tracked;
  // Every directory with a tracked path below it.
  absl::flat_hash_set<std::string> tracked_dirs;
  // Gitlinks and untracked nested repositories.
  absl::flat_hash_set<std::string> nested;
};

struct Workspace::Worktree {
  Worktree(std::string canonical_root, bool is_git, bool insensitive,
           std::string index_subdir, IgnoreOptions options)
      : root(std::move(canonical_root)),
        git(is_git),
        case_insensitive(insensitive),
        index_prefix(std::move(index_subdir)),
        ignore_options(std::move(options)) {}

  const std::string root;
  const bool git;
  // The root's volume compares names case-insensitively.
  const bool case_insensitive;
  // The index directory relative to root, with a trailing '/', or empty.
  const std::string index_prefix;
  const IgnoreOptions ignore_options;
  // External ignore files are outside the watched stream (also for linked
  // worktrees). Compare their contents on every kSync search.
  absl::Mutex excludes_mu;
  std::optional<std::string> info_exclude ABSL_GUARDED_BY(excludes_mu);
  std::optional<std::string> global_excludes ABSL_GUARDED_BY(excludes_mu);
  // Relative git directory, if it is inside this root. Linked worktrees put
  // their git directory elsewhere, outside the watched stream.
  std::string git_dir;
  std::string cookie_dir;
  bool owns_cookie_dir = false;
  // A replacement and Unregister must not start or stop concurrently.
  absl::Mutex recovery_mu;
  // Sync and Stop are serialized; callbacks never take this lock.
  absl::Mutex watch_mu;
  std::unique_ptr<Watcher> watcher ABSL_GUARDED_BY(watch_mu);
  bool syncing ABSL_GUARDED_BY(watch_mu) = false;
  bool stopping_watch ABSL_GUARDED_BY(watch_mu) = false;
  bool NoSyncLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(watch_mu) {
    return !syncing;
  }
  // The delivery thread caches ignore rules independently of the interactive
  // lane. Rescans and ignore-file edits invalidate the corresponding rules.
  absl::Mutex filter_mu;
  std::optional<IgnoreMatcher> filter_ignore ABSL_GUARDED_BY(filter_mu);
  std::atomic<bool> filter_reload{true};
  // Capture events until the registration scan publishes its snapshot.
  absl::Mutex event_mu;
  bool registering ABSL_GUARDED_BY(event_mu) = true;
  std::vector<WatchEvent> pending_events ABSL_GUARDED_BY(event_mu);
  // True once a fallback loop has been scheduled.
  bool fallback_scheduled ABSL_GUARDED_BY(event_mu) = false;
  // Sticky failure is only a freshness hint; relaxed ordering suffices.
  std::atomic<bool> watch_failed{false};
  // A failure during the replacement scan invalidates that replacement.
  std::atomic<uint64_t> watch_failure_version{0};

  // Serializes view publications and guards the per-path state. Taken after
  // Workspace::mu_ and never with generation_mu.
  absl::Mutex mu;
  // ApplyBatch runs under mu; Snapshot is thread-safe.
  ViewWriter writer;
  absl::flat_hash_map<PathId, PathState> files ABSL_GUARDED_BY(mu);
  // Stat signatures of the reads that produced unhinted states.
  absl::flat_hash_map<PathId, FileSignature> signatures ABSL_GUARDED_BY(mu);
  // Tracked paths as of the last scan; ignore rules apply to the others.
  absl::flat_hash_set<PathId> tracked ABSL_GUARDED_BY(mu);
  // Directories with tracked paths below them, as of the last scan. A
  // subtree event for an ignored directory matters only when it is one.
  absl::flat_hash_set<std::string> tracked_dirs ABSL_GUARDED_BY(mu);
  // Gitlinks and untracked nested repositories as of the last scan.
  absl::flat_hash_set<std::string> nested ABSL_GUARDED_BY(mu);
  // Directories with visible files below them; see AddVisible.
  absl::flat_hash_map<std::string, int> visible_children ABSL_GUARDED_BY(mu);
  // Paths whose newest read found content that could not be added to the
  // index, by that read's sequence. A newer read of the path removes it.
  absl::flat_hash_map<PathId, uint64_t> failed ABSL_GUARDED_BY(mu);
  // Generations completed while `failed` was not empty.
  std::vector<uint64_t> held ABSL_GUARDED_BY(mu);
  // A path without state was absent from the scan that took this sequence,
  // so an older read of it is dropped.
  uint64_t omitted_before ABSL_GUARDED_BY(mu) = 0;

  // Taken after Workspace::mu_.
  absl::Mutex generation_mu;
  uint64_t assigned ABSL_GUARDED_BY(generation_mu) = 0;
  uint64_t applied ABSL_GUARDED_BY(generation_mu) = 0;
  // Finished generations above `applied`.
  absl::flat_hash_set<uint64_t> finished ABSL_GUARDED_BY(generation_mu);
  bool removed ABSL_GUARDED_BY(generation_mu) = false;

  // Used only on the interactive lane's thread.
  std::optional<IgnoreMatcher> ignore;
  // Set by rescans, which may follow ignore-file edits nobody notified.
  std::atomic<bool> reload_ignore{true};
};

struct Workspace::NotifyJob {
  std::shared_ptr<Worktree> tree;
  std::vector<std::string> paths;
  std::vector<PathId> ids;
  uint64_t generation = 0;
};

struct Workspace::BulkJob {
  std::shared_ptr<Worktree> tree;
  bool registration = false;
  // Rescan generations to complete. Appended under Workspace::mu_ while the
  // job is in queued_rescans_, and read only after it leaves.
  std::vector<uint64_t> generations;
  // Written before `done` is notified.
  absl::Status status;
  absl::Notification done;
};

Workspace::Workspace(std::unique_ptr<Index> index, std::string index_dir,
                     WorkspaceOptions options)
    : index_(std::move(index)),
      index_dir_(std::move(index_dir)),
      options_(std::move(options)),
      executor_(options_.search.executor == nullptr
                    ? std::make_unique<SearchExecutor>()
                    : nullptr),
      interactive_lane_(
          std::make_unique<ThreadPool>(1, ThreadQos::kUserInteractive)),
      bulk_lane_(std::make_unique<ThreadPool>(1, ThreadQos::kUtility)),
      readers_(std::make_unique<ThreadPool>(options_.reader_threads,
                                            options_.reader_qos)),
      watch_fallback_lane_(
          std::make_unique<ThreadPool>(4, ThreadQos::kUtility)) {
  if (options_.serial_bulk_write &&
      options_.index.memtable.segment.write_symbols) {
    for (int i = 0; i < options_.index.symbol_threads; ++i) {
      bulk_symbol_workers_.push_back(std::make_unique<BulkSymbolWorker>());
    }
    bulk_symbol_pool_ = std::make_unique<ThreadPool>(
        options_.index.symbol_threads, ThreadQos::kUtility);
  }
  watch_timer_ = std::thread([this] { RunWatchTimer(); });
  if (executor_ != nullptr) options_.search.executor = executor_.get();
  const std::shared_ptr<const ReadState> state = index_->GetReadState();
  absl::MutexLock lock(content_mu_);
  for (const VersionSegment& segment : state->version->segments) {
    for (const BlobId blob : *segment.blob_of_doc) {
      const uint32_t id = static_cast<uint32_t>(blob);
      if (id >= indexed_.size()) indexed_.Resize(std::max(id + 1, 2 * id));
      indexed_.Set(id);
    }
  }
}

absl::StatusOr<std::unique_ptr<Workspace>> Workspace::Open(
    Env& env, absl::string_view index_dir, WorkspaceOptions options) {
  if (options.reader_threads < 1 || options.bulk_segment_bytes < 1 ||
      options.bulk_prefetch_bytes < 0 ||
      options.ingest_retry_initial_delay <= absl::ZeroDuration() ||
      options.ingest_retry_max_delay < options.ingest_retry_initial_delay ||
      options.watch_failure_rescan_interval <= absl::ZeroDuration() ||
      options.index_lock_wait < absl::ZeroDuration() ||
      !options.watcher_factory || !*options.watcher_factory) {
    return absl::InvalidArgumentError(
        "workspace options: reader_threads, bulk_segment_bytes, and "
        "ingest_retry_initial_delay must be positive, bulk_prefetch_bytes not "
        "negative, and ingest_retry_max_delay at least the initial delay");
  }
  ABSL_ASSIGN_OR_RETURN(auto opened,
                        Index::Open(env, index_dir, options.index));
  // Paths below a root are compared with the canonical index directory.
  ABSL_ASSIGN_OR_RETURN(std::string canonical, CanonicalDirectory(index_dir));
  auto workspace = absl::WrapUnique(new Workspace(
      std::move(opened.first), std::move(canonical), std::move(options)));
  std::vector<ManifestView> views = workspace->index_->Views();
  // Keep collection suspended until all recovered registrations are resolved.
  workspace->PinGc();
  {
    absl::MutexLock lock(workspace->mu_);
    for (const ManifestView& view : views) {
      workspace->restoring_.insert(view.canonical_root);
      workspace->restore_order_.push_back(view.canonical_root);
    }
  }
  workspace->restore_thread_ =
      std::thread([ptr = workspace.get(), views = std::move(views)] {
        ptr->RestoreViews(views);
      });
  return workspace;
}

void Workspace::RestoreViews(const std::vector<ManifestView>& views) {
  struct Pending {
    ManifestView view;
    absl::Time next = absl::InfinitePast();
    absl::Duration delay;
  };
  std::vector<Pending> pending;
  pending.reserve(views.size());
  for (const ManifestView& view : views) {
    pending.push_back(
        {view, absl::InfinitePast(), options_.ingest_retry_initial_delay});
  }
  while (!pending.empty() && !cancelling_.load(std::memory_order_relaxed)) {
    absl::Time wake = absl::InfiniteFuture();
    for (auto it = pending.begin(); it != pending.end();) {
      const std::string& name = it->view.canonical_root;
      bool requested;
      {
        absl::MutexLock lock(mu_);
        requested = restore_requested_.contains(name);
      }
      if (!requested && absl::Now() < it->next) {
        wake = std::min(wake, it->next);
        ++it;
        continue;
      }
      {
        absl::MutexLock lock(mu_);
        ++restore_attempts_[name].started;
      }
      // Only a missing root or an affirmative repository mismatch may be
      // forgotten. Git subprocess and scan errors leave the view protected.
      const auto root = CanonicalDirectory(name);
      bool missing = absl::IsNotFound(root.status()) ||
                     absl::IsInvalidArgument(root.status());
      bool mismatch = root.ok() && *root != name;
      bool valid = root.ok() && *root == name;
      if (valid) {
        // One rev-parse is enough to check both the top level and the common
        // directory; registration itself reads ignore configuration later.
        const auto git =
            RunGit(*root, {"rev-parse", "--show-toplevel",
                           "--path-format=absolute", "--git-common-dir"});
        if (!git.ok()) {
          valid = false;
        } else if (git->exit_code != 0) {
          mismatch =
              !it->view.git_common_dir.empty() &&
              absl::StrContains(git->stderr_data, "not a git repository");
          valid = it->view.git_common_dir.empty() &&
                  absl::StrContains(git->stderr_data, "not a git repository");
        } else {
          const std::vector<std::string> lines =
              absl::StrSplit(git->stdout_data, '\n', absl::SkipEmpty());
          if (it->view.git_common_dir.empty()) {
            mismatch = true;
          } else if (lines.size() == 2) {
            const auto common = CanonicalDirectory(lines[1]);
            mismatch = common.ok() &&
                       (lines[0] != name || *common != it->view.git_common_dir);
            valid = common.ok();
          } else {
            valid = false;
          }
        }
      }
      absl::Status status = absl::UnavailableError("restore validation failed");
      if (missing || mismatch) {
        status = index_->ForgetView(name);
      } else if (valid && !cancelling_.load(std::memory_order_relaxed)) {
        status = options_.restore_scan_hook
                     ? (*options_.restore_scan_hook)(name)
                     : absl::OkStatus();
        if (status.ok()) {
          status = RegisterView(name, {.watch = true}, true).status();
        }
      }
      if (cancelling_.load(std::memory_order_relaxed)) break;
      if (!status.ok()) {
        {
          absl::MutexLock lock(mu_);
          RestoreAttempts& attempts = restore_attempts_[name];
          attempts.finished = attempts.started;
          attempts.status = status;
          if (restore_requested_.erase(name) != 0) {
            it->next = absl::Now();
          } else {
            it->next = absl::Now() + it->delay;
          }
        }
        it->delay = std::min(it->delay * 2, options_.ingest_retry_max_delay);
        wake = std::min(wake, it->next);
        ++it;
        continue;
      }
      {
        absl::MutexLock lock(mu_);
        if (!missing && !mismatch) restored_roots_.insert(name);
        restoring_.erase(name);
        restore_requested_.erase(name);
        restore_attempts_.erase(name);
      }
      it = pending.erase(it);
    }
    if (pending.empty() || cancelling_.load(std::memory_order_relaxed)) break;
    absl::MutexLock lock(mu_);
    const auto requested = [this]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return !restore_requested_.empty() || stopping_;
    };
    mu_.AwaitWithDeadline(absl::Condition(&requested), wake);
  }
  if (pending.empty() && !cancelling_.load(std::memory_order_relaxed)) {
    UnpinGc();
  }
}

std::vector<std::pair<std::string, WorktreeId>> Workspace::Roots() const {
  absl::MutexLock lock(mu_);
  std::vector<std::pair<std::string, WorktreeId>> roots;
  const auto append = [this, &roots](const std::string& name)
                          ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                            for (const auto& [id, tree] : trees_) {
                              if (tree->root == name) {
                                roots.emplace_back(name, id);
                                break;
                              }
                            }
                          };
  for (const std::string& name : registration_order_) append(name);
  for (const std::string& name : restore_order_) {
    if (restored_roots_.contains(name)) append(name);
  }
  return roots;
}

Workspace::~Workspace() {
  cancelling_.store(true, std::memory_order_relaxed);
  {
    absl::MutexLock lock(mu_);
    stopping_ = true;
    notify_jobs_.clear();
  }
  // Stop callbacks before joining lanes or releasing any view.
  std::vector<std::shared_ptr<Worktree>> trees;
  {
    absl::MutexLock lock(mu_);
    for (const auto& [id, tree] : trees_) trees.push_back(tree);
  }
  {
    absl::MutexLock lock(watch_timer_mu_);
    watch_timer_stopping_ = true;
    watch_tasks_.clear();
  }
  watch_timer_.join();
  watch_fallback_lane_.reset();
  for (const auto& tree : trees) StopWatcher(tree);
  // Each pool runs its queued tasks, which see stopping_ or cancelling_ and
  // return, before joining. Bulk work uses the readers, so they go last.
  if (restore_thread_.joinable()) restore_thread_.join();
  interactive_lane_.reset();
  bulk_lane_.reset();
  readers_.reset();
  bulk_symbol_pool_.reset();
  {
    absl::MutexLock lock(mu_);
    mu_.Await(absl::Condition(this, &Workspace::NoBulkCallsLocked));
  }
  // Views are registered again after a reopen, which reads nothing for the
  // documents that survive in segments. A failed flush only costs rereads.
  index_->Flush().IgnoreError();
}

absl::Status Workspace::EnterBulkCall() {
  absl::MutexLock lock(mu_);
  if (stopping_) return absl::CancelledError("workspace shutting down");
  ++bulk_calls_;
  return absl::OkStatus();
}

void Workspace::ExitBulkCall() {
  absl::MutexLock lock(mu_);
  --bulk_calls_;
}

bool Workspace::NoBulkCallsLocked() const { return bulk_calls_ == 0; }

absl::StatusOr<std::shared_ptr<Workspace::Worktree>> Workspace::FindTree(
    WorktreeId id) const {
  absl::MutexLock lock(mu_);
  const auto it = trees_.find(id);
  if (it == trees_.end()) {
    return absl::NotFoundError(
        absl::StrCat("worktree ", static_cast<uint32_t>(id), " not found"));
  }
  return it->second;
}

uint64_t Workspace::NextSeq() {
  return read_seq_.fetch_add(1, std::memory_order_relaxed) + 1;
}

uint64_t Workspace::AssignGeneration(Worktree& tree) {
  absl::MutexLock lock(tree.generation_mu);
  return ++tree.assigned;
}

void Workspace::Complete(Worktree& tree, uint64_t generation) {
  absl::MutexLock lock(tree.generation_mu);
  tree.finished.insert(generation);
  while (tree.finished.erase(tree.applied + 1) != 0) ++tree.applied;
  watch_trace::Mark(watch_trace::kCompleted);
}

void Workspace::CompleteWhenIndexed(Worktree& tree,
                                    std::vector<uint64_t> generations) {
  {
    absl::MutexLock lock(tree.mu);
    if (!tree.failed.empty()) {
      tree.held.insert(tree.held.end(), generations.begin(), generations.end());
      return;
    }
    generations.insert(generations.end(), tree.held.begin(), tree.held.end());
    tree.held.clear();
  }
  for (const uint64_t generation : generations) Complete(tree, generation);
}

uint64_t Workspace::WaitForApplied(Worktree& tree, uint64_t generation,
                                   absl::Duration timeout) {
  absl::MutexLock lock(tree.generation_mu);
  const auto reached = [&tree, generation]()
                           ABSL_EXCLUSIVE_LOCKS_REQUIRED(tree.generation_mu) {
                             return tree.applied >= generation || tree.removed;
                           };
  tree.generation_mu.AwaitWithTimeout(absl::Condition(&reached), timeout);
  return tree.applied;
}

absl::StatusOr<WorktreeId> Workspace::Register(absl::string_view root,
                                               RegisterOptions options) {
  return RegisterView(root, options, false);
}

absl::StatusOr<WorktreeId> Workspace::RegisterView(absl::string_view root,
                                                   RegisterOptions options,
                                                   bool restoring) {
  ABSL_RETURN_IF_ERROR(EnterBulkCall());
  absl::Cleanup exit_call = [this] { ExitBulkCall(); };
  ABSL_ASSIGN_OR_RETURN(const std::string canonical, CanonicalDirectory(root));
  {
    absl::MutexLock lock(mu_);
    if (!restoring) {
      if (restoring_.contains(canonical)) {
        // Wait for a restore attempt that starts after this request. If it
        // fails, return its error; the restore keeps retrying, so a view
        // that keeps failing never blocks the caller for good.
        const uint64_t target = restore_attempts_[canonical].started + 1;
        restore_requested_.insert(canonical);
        const auto answered = [this, &canonical,
                               target]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
          if (stopping_ || !restoring_.contains(canonical)) return true;
          const auto attempts = restore_attempts_.find(canonical);
          return attempts != restore_attempts_.end() &&
                 attempts->second.finished >= target;
        };
        mu_.Await(absl::Condition(&answered));
        if (!stopping_ && restoring_.contains(canonical)) {
          return restore_attempts_.find(canonical)->second.status;
        }
      }
      if (stopping_) return absl::CancelledError("workspace shutting down");
      // A restored registration remains idempotent for later client calls.
      if (restored_roots_.contains(canonical)) {
        for (const auto& [id, tree] : trees_) {
          if (tree->root == canonical) return id;
        }
      }
    }
    if (!roots_.insert(canonical).second) {
      return absl::AlreadyExistsError(
          absl::StrCat("worktree ", canonical, " is already registered"));
    }
  }
  absl::Cleanup release_root = [this, &canonical] {
    absl::MutexLock lock(mu_);
    roots_.erase(canonical);
  };
  ABSL_ASSIGN_OR_RETURN(const std::optional<GitRootInfo> git,
                        FindGitRoot(canonical));
  if (git.has_value()) {
    absl::StatusOr<std::string> toplevel = CanonicalDirectory(git->toplevel);
    if (!toplevel.ok() || *toplevel != canonical) {
      return absl::InvalidArgumentError(
          absl::StrCat(canonical, " is inside git worktree ", git->toplevel,
                       "; register its top level"));
    }
  }
  std::string index_prefix;
  if (absl::StartsWith(index_dir_, absl::StrCat(canonical, "/"))) {
    index_prefix = absl::StrCat(index_dir_.substr(canonical.size() + 1), "/");
  }
  auto job = std::make_shared<BulkJob>();
  job->registration = true;
  job->tree = std::make_shared<Worktree>(
      canonical, git.has_value(), CaseInsensitiveVolume(canonical),
      index_prefix, IgnoreOptionsFor(canonical, git, index_prefix));
  const std::shared_ptr<Worktree> tree = job->tree;
  absl::Cleanup stop_on_failure = [this, tree] {
    StopWatcher(tree);
    DropReferences(*tree);
  };
  if (options.watch) StartWatcher(tree);
  WorktreeId id;
  {
    absl::MutexLock lock(mu_);
    if (stopping_) return absl::CancelledError("workspace shutting down");
    id = static_cast<WorktreeId>(next_id_++);
    ScheduleBulkLocked(job);
  }
  job->done.WaitForNotification();
  if (!job->status.ok()) {
    StopWatcher(tree);
    return job->status;
  }
  if (tree->git) {
    absl::MutexLock excludes_lock(tree->excludes_mu);
    ABSL_ASSIGN_OR_RETURN(
        tree->info_exclude,
        ReadIgnoreFile(tree->ignore_options.info_exclude_path));
    if (!tree->ignore_options.global_excludes_path.empty()) {
      ABSL_ASSIGN_OR_RETURN(
          tree->global_excludes,
          ReadIgnoreFile(tree->ignore_options.global_excludes_path));
    }
  }
  if (!restoring) {
    std::string common;
    if (git.has_value()) {
      ABSL_ASSIGN_OR_RETURN(common, GitCommonDirectory(canonical));
    }
    ABSL_RETURN_IF_ERROR(index_->RecordView({canonical, common}));
  }
  {
    absl::MutexLock lock(mu_);
    if (stopping_) return absl::CancelledError("workspace shutting down");
    trees_.emplace(id, tree);
    if (!restoring) registration_order_.push_back(canonical);
  }
  // Keep buffering while draining so the snapshot's events cannot be
  // overtaken by callbacks delivered after the scan.
  for (;;) {
    std::vector<WatchEvent> pending;
    {
      absl::MutexLock lock(tree->event_mu);
      if (tree->pending_events.empty()) {
        tree->registering = false;
        if (tree->fallback_scheduled) {
          ScheduleWatchTask(
              absl::Now() + options_.watch_failure_rescan_interval,
              [this, tree] { RunWatchFallback(tree); });
        }
        break;
      }
      pending.swap(tree->pending_events);
    }
    ApplyWatchEvents(tree, pending);
  }
  std::move(stop_on_failure).Cancel();
  std::move(release_root).Cancel();
  return id;
}

void Workspace::StartWatcher(const std::shared_ptr<Worktree>& tree) {
  WatcherOptions watcher_options = options_.watcher;
  if (tree->git) {
    const auto git_path =
        RunGit(tree->root, {"rev-parse", "--absolute-git-dir"});
    if (!git_path.ok()) {
      WatchFailure(tree, git_path.status());
      return;
    }
    if (git_path->exit_code != 0) {
      WatchFailure(tree, absl::FailedPreconditionError(git_path->stderr_data));
      return;
    }
    const std::string path = FirstLine(git_path->stdout_data);
    if (absl::StartsWith(path, absl::StrCat(tree->root, "/"))) {
      tree->git_dir = path.substr(tree->root.size() + 1);
    }
  }
  // A real git directory is already excluded by both scanners. Otherwise
  // use a private directory, excluded explicitly below, for sync cookies.
  tree->cookie_dir = tree->git_dir.empty()
                         ? ".ferret-watch"
                         : absl::StrCat(tree->git_dir, "/ferret");
  watcher_options.cookie_dir = tree->cookie_dir;
  const std::string cookies = absl::StrCat(tree->root, "/", tree->cookie_dir);
  const absl::StatusOr<bool> made =
      watch_internal::MakeCookieDirectory(cookies);
  if (!made.ok()) {
    WatchFailure(tree, made.status());
    return;
  }
  tree->owns_cookie_dir = *made;
  auto watcher = (*options_.watcher_factory)(tree->root, watcher_options);
  if (!watcher.ok()) {
    WatchFailure(tree, watcher.status());
    return;
  }
  if ((*watcher)->root() != tree->root) {
    WatchFailure(tree,
                 absl::InvalidArgumentError("watcher root is not canonical"));
    return;
  }
  absl::Status status =
      (*watcher)->Start([this, tree](std::vector<WatchEvent> events) {
        OnWatchEvents(tree, std::move(events));
      });
  if (!status.ok()) {
    WatchFailure(tree, status);
    (*watcher)->Stop();
    return;
  }
  absl::MutexLock lock(tree->watch_mu);
  tree->watcher = std::move(*watcher);
}

bool Workspace::StoppingLocked() const { return stopping_; }

absl::Status Workspace::SyncWatcher(const std::shared_ptr<Worktree>& tree,
                                    absl::Time deadline) {
  watch_trace::Mark(watch_trace::kSyncStart);
  Watcher* watcher;
  {
    absl::MutexLock lock(tree->watch_mu);
    if (tree->watcher == nullptr) {
      return tree->watch_failed.load(std::memory_order_relaxed)
                 ? absl::FailedPreconditionError("watcher failed")
                 : absl::OkStatus();
    }
    if (!tree->watch_mu.AwaitWithDeadline(
            absl::Condition(tree.get(), &Worktree::NoSyncLocked), deadline)) {
      WorkspaceStats delta;
      ++delta.sync_timeouts;
      AddStats(delta);
      return absl::DeadlineExceededError("waiting for another watcher sync");
    }
    if (tree->stopping_watch) {
      return absl::FailedPreconditionError("watcher is stopping");
    }
    tree->syncing = true;
    watcher = tree->watcher.get();
  }
  absl::Status status =
      watcher->Sync(std::max(absl::ZeroDuration(), deadline - absl::Now()));
  watch_trace::Mark(watch_trace::kSyncReturned);
  {
    absl::MutexLock lock(tree->watch_mu);
    tree->syncing = false;
  }
  if (absl::IsDeadlineExceeded(status)) {
    WorkspaceStats delta;
    ++delta.sync_timeouts;
    AddStats(delta);
  } else if (!status.ok()) {
    WatchFailure(tree, status);
  }
  if (status.ok() && tree->watch_failed.load(std::memory_order_relaxed)) {
    return absl::FailedPreconditionError("watcher failed");
  }
  return status;
}

absl::Status Workspace::RefreshExternalExcludes(
    const std::shared_ptr<Worktree>& tree) {
  if (!tree->git) return absl::OkStatus();
  absl::MutexLock lock(tree->excludes_mu);
  ABSL_ASSIGN_OR_RETURN(const std::optional<std::string> info,
                        ReadIgnoreFile(tree->ignore_options.info_exclude_path));
  std::optional<std::string> global;
  if (!tree->ignore_options.global_excludes_path.empty()) {
    ABSL_ASSIGN_OR_RETURN(
        global, ReadIgnoreFile(tree->ignore_options.global_excludes_path));
  }
  if (info != tree->info_exclude || global != tree->global_excludes) {
    tree->info_exclude = info;
    tree->global_excludes = global;
    // Reconciliation reloads both the scan's rules and the Notify lane's
    // cached matcher. Assign the generation before kSync chooses its target.
    WatchRescan(tree);
  }
  return absl::OkStatus();
}

void Workspace::StopWatcher(const std::shared_ptr<Worktree>& tree) {
  std::unique_ptr<Watcher> watcher;
  {
    absl::MutexLock lock(tree->watch_mu);
    tree->stopping_watch = true;
    tree->watch_mu.Await(absl::Condition(tree.get(), &Worktree::NoSyncLocked));
    watcher = std::move(tree->watcher);
  }
  if (watcher != nullptr) watcher->Stop();
  if (tree->owns_cookie_dir) {
    watch_internal::RemoveCookieDirectory(
        absl::StrCat(tree->root, "/", tree->cookie_dir));
    tree->owns_cookie_dir = false;
  }
}

void Workspace::WatchFailure(const std::shared_ptr<Worktree>& tree,
                             const absl::Status& error) {
  WorkspaceStats delta;
  ++delta.watcher_failures;
  delta.last_watcher_error = error;
  AddStats(delta);
  absl::MutexLock lock(tree->event_mu);
  tree->watch_failed.store(true, std::memory_order_relaxed);
  tree->watch_failure_version.fetch_add(1, std::memory_order_relaxed);
  if (!tree->fallback_scheduled) {
    tree->fallback_scheduled = true;
    if (!tree->registering) {
      ScheduleWatchTask(absl::Now() + options_.watch_failure_rescan_interval,
                        [this, tree] { RunWatchFallback(tree); });
    }
  }
}

void Workspace::ScheduleWatchTask(absl::Time due,
                                  absl::AnyInvocable<void()> task) {
  absl::MutexLock lock(watch_timer_mu_);
  if (!watch_timer_stopping_) {
    watch_tasks_.push_back({due, std::move(task)});
    watch_timer_changed_ = true;
  }
}

bool Workspace::WatchTimerHasWorkLocked() const {
  return watch_timer_stopping_ || !watch_tasks_.empty();
}

bool Workspace::WatchTimerChangedLocked() const {
  return watch_timer_stopping_ || watch_timer_changed_;
}

void Workspace::RunWatchTimer() {
  for (;;) {
    absl::AnyInvocable<void()> task;
    {
      absl::MutexLock lock(watch_timer_mu_);
      while (!watch_timer_stopping_) {
        if (watch_tasks_.empty()) {
          watch_timer_mu_.Await(
              absl::Condition(this, &Workspace::WatchTimerHasWorkLocked));
          continue;
        }
        const auto earliest = std::min_element(
            watch_tasks_.begin(), watch_tasks_.end(),
            [](const TimedWatchTask& a, const TimedWatchTask& b) {
              return a.due < b.due;
            });
        if (absl::Now() < earliest->due) {
          watch_timer_changed_ = false;
          watch_timer_mu_.AwaitWithDeadline(
              absl::Condition(this, &Workspace::WatchTimerChangedLocked),
              earliest->due);
          continue;
        }
        task = std::move(earliest->run);
        watch_tasks_.erase(earliest);
        break;
      }
      if (watch_timer_stopping_) return;
    }
    watch_fallback_lane_->Schedule(std::move(task));
  }
}

void Workspace::RunWatchFallback(const std::shared_ptr<Worktree>& tree) {
  absl::MutexLock recovery_lock(tree->recovery_mu);
  {
    absl::MutexLock lock(tree->generation_mu);
    if (tree->removed) return;
  }
  if (cancelling_.load(std::memory_order_relaxed)) return;
  // Buffer callbacks across the replacement scan, just as in Register.
  {
    absl::MutexLock lock(tree->event_mu);
    tree->registering = true;
  }
  StopWatcher(tree);
  {
    absl::MutexLock lock(tree->watch_mu);
    tree->stopping_watch = false;
  }
  StartWatcher(tree);
  const bool started = [&] {
    absl::MutexLock lock(tree->watch_mu);
    return tree->watcher != nullptr;
  }();
  const uint64_t failure_version =
      tree->watch_failure_version.load(std::memory_order_relaxed);
  const uint64_t generation = AssignGeneration(*tree);
  const std::shared_ptr<BulkJob> job = EnqueueRescan(tree, generation);
  ScheduleWatchTask(absl::Now(), [this, tree, job, started, failure_version] {
    FinishWatchFallback(tree, job, started, failure_version);
  });
}

void Workspace::FinishWatchFallback(const std::shared_ptr<Worktree>& tree,
                                    const std::shared_ptr<BulkJob>& job,
                                    bool started, uint64_t failure_version) {
  if (cancelling_.load(std::memory_order_relaxed)) return;
  if (job != nullptr && !job->done.HasBeenNotified()) {
    ScheduleWatchTask(absl::Now() + absl::Milliseconds(10),
                      [this, tree, job, started, failure_version] {
                        FinishWatchFallback(tree, job, started,
                                            failure_version);
                      });
    return;
  }
  absl::MutexLock recovery_lock(tree->recovery_mu);
  {
    absl::MutexLock lock(tree->generation_mu);
    if (tree->removed) return;
  }
  for (;;) {
    std::vector<WatchEvent> pending;
    {
      absl::MutexLock lock(tree->event_mu);
      if (tree->pending_events.empty()) {
        tree->registering = false;
        break;
      }
      pending.swap(tree->pending_events);
    }
    ApplyWatchEvents(tree, pending);
  }
  {
    absl::MutexLock lock(tree->event_mu);
    if (started && job != nullptr && job->status.ok() &&
        tree->watch_failure_version.load(std::memory_order_relaxed) ==
            failure_version) {
      tree->watch_failed.store(false, std::memory_order_relaxed);
      tree->fallback_scheduled = false;
      return;
    }
  }
  ScheduleWatchTask(absl::Now() + options_.watch_failure_rescan_interval,
                    [this, tree] { RunWatchFallback(tree); });
}

void Workspace::DeferWatchRescan(const std::shared_ptr<Worktree>& tree,
                                 uint64_t generation, absl::Time deadline) {
  {
    absl::MutexLock lock(tree->generation_mu);
    if (tree->removed) return;
  }
  if (cancelling_.load(std::memory_order_relaxed)) return;
  const std::string lock_path =
      absl::StrCat(tree->root, "/", tree->git_dir, "/index.lock");
  if (access(lock_path.c_str(), F_OK) == 0 && absl::Now() < deadline) {
    ScheduleWatchTask(std::min(deadline, absl::Now() + absl::Milliseconds(10)),
                      [this, tree, generation, deadline] {
                        DeferWatchRescan(tree, generation, deadline);
                      });
    return;
  }
  if (access(lock_path.c_str(), F_OK) == 0) {
    WorkspaceStats delta;
    ++delta.index_lock_timeouts;
    AddStats(delta);
  }
  EnqueueRescan(tree, generation);
}

void Workspace::WatchRescan(const std::shared_ptr<Worktree>& tree) {
  const uint64_t generation = AssignGeneration(*tree);
  if (!tree->git_dir.empty() &&
      access(
          absl::StrCat(tree->root, "/", tree->git_dir, "/index.lock").c_str(),
          F_OK) == 0) {
    DeferWatchRescan(tree, generation, absl::Now() + options_.index_lock_wait);
  } else {
    EnqueueRescan(tree, generation);
  }
}

void Workspace::OnWatchEvents(const std::shared_ptr<Worktree>& tree,
                              std::vector<WatchEvent> events) {
  watch_trace::Mark(watch_trace::kCallbackStart);
  {
    absl::MutexLock lock(tree->event_mu);
    if (tree->registering) {
      tree->pending_events.insert(tree->pending_events.end(),
                                  std::make_move_iterator(events.begin()),
                                  std::make_move_iterator(events.end()));
      return;
    }
  }
  ApplyWatchEvents(tree, events);
}

void Workspace::ApplyWatchEvents(const std::shared_ptr<Worktree>& tree,
                                 const std::vector<WatchEvent>& events) {
  // Invalidate before matching so edits in this batch see the new rules.
  absl::MutexLock filter_lock(tree->filter_mu);
  if (tree->filter_reload.exchange(false, std::memory_order_relaxed) ||
      !tree->filter_ignore.has_value()) {
    tree->filter_ignore.emplace(tree->ignore_options);
  }
  IgnoreMatcher& ignore = *tree->filter_ignore;
  for (const WatchEvent& event : events) {
    const std::string& path = std::holds_alternative<SubtreeChanged>(event)
                                  ? std::get<SubtreeChanged>(event).rel_prefix
                                  : std::get<PathChanged>(event).rel_path;
    if (IsIgnoreFile(path)) ignore.Invalidate(Dirname(path));
  }
  std::vector<std::string> paths;
  // Ignored hints, checked against the tracked paths under one lock below
  // instead of copying them or locking per hint.
  std::vector<std::pair<std::string, PathId>> ignored_files;
  std::vector<std::string> ignored_subtrees;
  bool rescan = false;
  for (const WatchEvent& event : events) {
    const bool subtree = std::holds_alternative<SubtreeChanged>(event);
    const std::string& path = subtree
                                  ? std::get<SubtreeChanged>(event).rel_prefix
                                  : std::get<PathChanged>(event).rel_path;
    if (!path.empty() && !PathTable::Validate(path).ok()) {
      rescan = true;
      continue;
    }
    if (!tree->cookie_dir.empty() &&
        (path == tree->cookie_dir ||
         absl::StartsWith(path, absl::StrCat(tree->cookie_dir, "/")))) {
      continue;
    }
    if (!tree->git_dir.empty() &&
        absl::StartsWith(path, absl::StrCat(tree->git_dir, "/"))) {
      const std::string file = path.substr(tree->git_dir.size() + 1);
      if (file == "HEAD" || file == "index") rescan = true;
      continue;
    }
    if (IsGitDirPath(path)) continue;
    if (!path.empty() && !IsIgnoreFile(path)) {
      const absl::StatusOr<bool> ignored = ignore.IsIgnored(path, subtree);
      if (ignored.ok() && *ignored) {
        // Tracked files are searchable even when an ignore pattern matches.
        if (subtree) {
          ignored_subtrees.push_back(path);
        } else if (const std::optional<PathId> id = paths_.Find(path);
                   id.has_value()) {
          ignored_files.emplace_back(path, *id);
        }
        continue;
      }
    }
    if (subtree || path.empty()) {
      rescan = true;
    } else {
      paths.push_back(path);
    }
  }
  if (!ignored_files.empty() || !ignored_subtrees.empty()) {
    absl::MutexLock lock(tree->mu);
    for (auto& [path, id] : ignored_files) {
      if (tree->tracked.contains(id)) paths.push_back(std::move(path));
    }
    for (const std::string& dir : ignored_subtrees) {
      const std::optional<PathId> id = paths_.Find(dir);
      if (tree->tracked_dirs.contains(dir) ||
          (id.has_value() && tree->tracked.contains(*id))) {
        rescan = true;
      }
    }
  }
  if (!paths.empty()) {
    std::vector<PathId> ids;
    ids.reserve(paths.size());
    for (const std::string& path : paths) ids.push_back(*paths_.Intern(path));
    absl::MutexLock lock(mu_);
    if (!stopping_) QueueNotifyLocked(tree, paths, ids);
  }
  watch_trace::Mark(watch_trace::kQueued);
  if (rescan) WatchRescan(tree);
}

absl::Status Workspace::Unregister(WorktreeId id) {
  std::shared_ptr<Worktree> tree;
  {
    absl::MutexLock lock(mu_);
    const auto it = trees_.find(id);
    if (it == trees_.end()) {
      return absl::NotFoundError(
          absl::StrCat("worktree ", static_cast<uint32_t>(id), " not found"));
    }
    tree = std::move(it->second);
    trees_.erase(it);
  }
  {
    absl::MutexLock lock(tree->generation_mu);
    tree->removed = true;
  }
  {
    absl::MutexLock recovery_lock(tree->recovery_mu);
    StopWatcher(tree);
  }
  DropReferences(*tree);
  absl::Status status = index_->ForgetView(tree->root);
  {
    absl::MutexLock lock(mu_);
    roots_.erase(tree->root);
    restored_roots_.erase(tree->root);
    std::erase(registration_order_, tree->root);
  }
  return status;
}

absl::StatusOr<uint64_t> Workspace::Notify(
    WorktreeId id, absl::Span<const std::string> paths) {
  ABSL_ASSIGN_OR_RETURN(const std::shared_ptr<Worktree> tree, FindTree(id));
  std::vector<PathId> ids;
  ids.reserve(paths.size());
  for (const std::string& path : paths) {
    absl::StatusOr<PathId> interned = paths_.Intern(path);
    if (!interned.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("notify path \"", absl::CEscape(path),
                       "\": ", interned.status().message()));
    }
    ids.push_back(*interned);
  }
  absl::MutexLock lock(mu_);
  if (stopping_) return absl::CancelledError("workspace shutting down");
  return QueueNotifyLocked(tree, paths, ids);
}

uint64_t Workspace::QueueNotifyLocked(const std::shared_ptr<Worktree>& tree,
                                      absl::Span<const std::string> paths,
                                      absl::Span<const PathId> ids) {
  // Each batch completes its own generation; the last implies the others.
  uint64_t generation = 0;
  size_t begin = 0;
  do {
    const size_t end = std::min(begin + kMaxNotifyBatchPaths, paths.size());
    NotifyJob& job = notify_jobs_.emplace_back();
    job.tree = tree;
    job.paths.assign(paths.begin() + static_cast<ptrdiff_t>(begin),
                     paths.begin() + static_cast<ptrdiff_t>(end));
    job.ids.assign(ids.begin() + static_cast<ptrdiff_t>(begin),
                   ids.begin() + static_cast<ptrdiff_t>(end));
    job.generation = generation = AssignGeneration(*tree);
    begin = end;
  } while (begin < paths.size());
  if (!interactive_draining_) {
    interactive_draining_ = true;
    interactive_lane_->Schedule([this] { DrainInteractive(); });
  }
  return generation;
}

absl::StatusOr<uint64_t> Workspace::Rescan(WorktreeId id) {
  ABSL_RETURN_IF_ERROR(EnterBulkCall());
  absl::Cleanup exit_call = [this] { ExitBulkCall(); };
  ABSL_ASSIGN_OR_RETURN(const std::shared_ptr<Worktree> tree, FindTree(id));
  const uint64_t generation = AssignGeneration(*tree);
  const std::shared_ptr<BulkJob> job = EnqueueRescan(tree, generation);
  if (job == nullptr) return absl::CancelledError("workspace shutting down");
  job->done.WaitForNotification();
  if (!job->status.ok()) return job->status;
  return generation;
}

absl::StatusOr<uint64_t> Workspace::AwaitGeneration(WorktreeId id,
                                                    uint64_t generation,
                                                    absl::Duration timeout) {
  ABSL_ASSIGN_OR_RETURN(const std::shared_ptr<Worktree> tree, FindTree(id));
  return WaitForApplied(*tree, generation, timeout);
}

absl::StatusOr<WorkspaceSearchResult> Workspace::Search(
    WorktreeId id, const SearchRequest& request,
    const SearchConsistency& consistency) {
  if (consistency.timeout < absl::ZeroDuration()) {
    return absl::InvalidArgumentError("negative search timeout");
  }
  ABSL_ASSIGN_OR_RETURN(const std::shared_ptr<Worktree> tree, FindTree(id));
  WorkspaceSearchResult result;
  if (consistency.mode != SearchConsistencyMode::kNone) {
    const absl::Time deadline = absl::Now() + consistency.timeout;
    bool synced = true;
    uint64_t target = consistency.min_generation;
    if (consistency.mode == SearchConsistencyMode::kSync) {
      synced = SyncWatcher(tree, deadline).ok();
      if (synced) synced = RefreshExternalExcludes(tree).ok();
      watch_trace::Mark(watch_trace::kExcludesDone);
      absl::MutexLock lock(tree->generation_mu);
      target = std::max(target, tree->assigned);
    }
    result.generation = WaitForApplied(
        *tree, target, std::max(absl::ZeroDuration(), deadline - absl::Now()));
    watch_trace::Mark(watch_trace::kGenerationDone);
    result.stale = !synced || result.generation < target;
  } else {
    absl::MutexLock lock(tree->generation_mu);
    result.generation = tree->applied;
  }
  // Hold unregister until both snapshots have been captured. A caller that
  // found the tree just before its removal must not search an obsolete view
  // against a ReadState captured after its blobs have been collected.
  std::shared_ptr<const ViewState> view;
  std::shared_ptr<const ReadState> state;
  {
    absl::MutexLock lock(tree->generation_mu);
    if (tree->removed) return absl::NotFoundError("worktree unregistered");
    view = tree->writer.Snapshot();
    state = index_->GetReadState();
  }
  watch_trace::Mark(watch_trace::kSnapshotDone);
  ABSL_ASSIGN_OR_RETURN(
      result.response, ferret::Search(*state, SearchView{view, paths_}, request,
                                      options_.search));
  return result;
}

absl::StatusOr<FindSymbolResponse> Workspace::FindSymbol(
    WorktreeId id, const FindSymbolRequest& request,
    const SearchConsistency& consistency) {
  if (request.name.empty() || request.max_results < 0 ||
      consistency.timeout < absl::ZeroDuration()) {
    return absl::InvalidArgumentError("invalid symbol lookup request");
  }
  std::optional<PathGlob> glob;
  if (!request.path_glob.empty()) {
    ABSL_ASSIGN_OR_RETURN(glob, PathGlob::Parse(request.path_glob));
  }
  ABSL_ASSIGN_OR_RETURN(const std::shared_ptr<Worktree> tree, FindTree(id));
  FindSymbolResponse result;
  if (consistency.mode != SearchConsistencyMode::kNone) {
    const absl::Time deadline = absl::Now() + consistency.timeout;
    bool synced = true;
    uint64_t target = consistency.min_generation;
    if (consistency.mode == SearchConsistencyMode::kSync) {
      synced = SyncWatcher(tree, deadline).ok();
      if (synced) synced = RefreshExternalExcludes(tree).ok();
      absl::MutexLock lock(tree->generation_mu);
      target = std::max(target, tree->assigned);
    }
    result.generation = WaitForApplied(
        *tree, target, std::max(absl::ZeroDuration(), deadline - absl::Now()));
    result.stale = !synced || result.generation < target;
  } else {
    absl::MutexLock lock(tree->generation_mu);
    result.generation = tree->applied;
  }
  std::shared_ptr<const ViewState> view;
  std::shared_ptr<const ReadState> state;
  {
    absl::MutexLock lock(tree->generation_mu);
    if (tree->removed) return absl::NotFoundError("worktree unregistered");
    view = tree->writer.Snapshot();
    state = index_->GetReadState();
  }
  const SymbolLookup found =
      state->FindSymbol(request.name, request.case_insensitive);
  for (BlobId blob : found.pending_blobs) {
    for (PathId path_id : view->PathsFor(blob)) {
      const std::string path = paths_.path(path_id);
      if (!glob || glob->Matches(path)) ++result.pending_files;
    }
  }
  struct RankedLocation {
    SymbolLocation location;
    RankKey rank;
  };
  std::vector<RankedLocation> ranked;
  for (const IndexedSymbol& symbol : found.occurrences) {
    if (request.definitions_only && !symbol.is_definition) continue;
    if (!request.kinds.empty() &&
        std::find(request.kinds.begin(), request.kinds.end(), symbol.kind) ==
            request.kinds.end()) {
      continue;
    }
    size_t start = 0;
    for (uint32_t line = 1;
         line < symbol.line && start < symbol.doc.content.size(); ++line) {
      const size_t next = symbol.doc.content.find('\n', start);
      start = next == absl::string_view::npos ? symbol.doc.content.size()
                                              : next + 1;
    }
    const size_t end = symbol.doc.content.find('\n', start);
    const absl::string_view text = symbol.doc.content.substr(
        start, end == absl::string_view::npos ? end : end - start);
    std::string display;
    const size_t name_at = text.find(symbol.name);
    const size_t begin = name_at == absl::string_view::npos ? 0 : name_at;
    const size_t end_name =
        begin + (name_at == absl::string_view::npos ? 0 : symbol.name.size());
    AppendDisplayLine(text, begin, end_name, 256, display);
    for (PathId path_id : view->PathsFor(symbol.blob)) {
      const std::string path = paths_.path(path_id);
      if (glob && !glob->Matches(path)) continue;
      ranked.push_back(
          {{path, symbol.line, symbol.kind, symbol.is_definition, display},
           {symbol.doc.tier, symbol.doc.prior, symbol.doc.oid}});
    }
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const RankedLocation& a, const RankedLocation& b) {
              if (a.location.is_definition != b.location.is_definition) {
                return a.location.is_definition;
              }
              if (a.rank != b.rank) return a.rank < b.rank;
              if (a.location.path != b.location.path) {
                return a.location.path < b.location.path;
              }
              return a.location.line < b.location.line;
            });
  result.total_count = static_cast<int64_t>(ranked.size());
  for (const RankedLocation& item : ranked) {
    if (result.occurrences.size() >= static_cast<size_t>(request.max_results)) {
      break;
    }
    result.occurrences.push_back(item.location);
  }
  return result;
}

absl::StatusOr<WorkspaceWorktreeStatus> Workspace::WorktreeStatus(
    WorktreeId id) const {
  ABSL_ASSIGN_OR_RETURN(const std::shared_ptr<Worktree> tree, FindTree(id));
  WorkspaceWorktreeStatus status;
  {
    absl::MutexLock lock(tree->mu);
    absl::flat_hash_set<BlobId> blobs;
    for (const auto& [path, file] : tree->files) {
      if (file.blob.has_value()) {
        ++status.view_paths;
        blobs.insert(*file.blob);
      } else if (file.skip != SkipReason::kNone) {
        ++status.skipped_paths;
      }
    }
    status.view_blobs = static_cast<int64_t>(blobs.size());
    status.unstable_paths = static_cast<int64_t>(tree->failed.size());
  }
  {
    absl::MutexLock lock(tree->generation_mu);
    status.applied_generation = tree->applied;
    status.assigned_generation = tree->assigned;
  }
  {
    absl::MutexLock lock(mu_);
    for (const NotifyJob& job : notify_jobs_) {
      if (job.tree == tree) {
        status.queued_paths += static_cast<int64_t>(job.paths.size());
      }
    }
  }
  {
    absl::MutexLock lock(tree->watch_mu);
    status.watcher_running = tree->watcher != nullptr;
  }
  status.watcher_failed = tree->watch_failed.load(std::memory_order_relaxed);
  status.watcher_running = status.watcher_running && !status.watcher_failed;
  const std::shared_ptr<const ReadState> state = index_->GetReadState();
  status.segment_count = static_cast<int64_t>(state->version->segments.size());
  for (const VersionSegment& segment : state->version->segments) {
    status.index_bytes += static_cast<int64_t>(segment.metadata.file_size);
  }
  const IndexStats counters = index_->Stats();
  status.pending_symbols = counters.pending_symbols;
  status.failed_symbols = counters.failed_symbols;
  return status;
}

WorkspaceStats Workspace::Stats() const {
  WorkspaceStats result;
  {
    absl::MutexLock lock(stats_mu_);
    result = stats_;
  }
  result.index = index_->Stats();
  return result;
}

absl::Status Workspace::FlushIndex() { return index_->Flush(); }

void Workspace::WaitForCompactions() { index_->WaitForCompactions(); }

void Workspace::PinGc() {
  absl::MutexLock lock(gc_mu_);
  if (gc_pins_++ == 0) index_->SuspendGc();
}

void Workspace::UnpinGc() {
  absl::MutexLock lock(gc_mu_);
  CHECK_GT(gc_pins_, 0);
  if (--gc_pins_ == 0) index_->ResumeGc();
}

void Workspace::RefreshIndexed() {
  // A flush or a compaction publishes a new MANIFEST, and nothing else
  // changes which blobs the segments hold, so its sequence says whether the
  // set is stale. Index::Stats() would answer too, but it walks every
  // document of every segment on each call.
  const std::shared_ptr<const ReadState> state = index_->GetReadState();
  absl::MutexLock lock(content_mu_);
  if (indexed_manifest_seq_.has_value() &&
      *indexed_manifest_seq_ == state->version->manifest_seq) {
    return;
  }
  Bitset indexed;
  const auto mark = [&indexed](BlobId blob) {
    const uint32_t id = static_cast<uint32_t>(blob);
    if (id >= indexed.size()) indexed.Resize(std::max(id + 1, 2 * id));
    indexed.Set(id);
  };
  for (const VersionSegment& segment : state->version->segments) {
    for (BlobId blob : *segment.blob_of_doc) mark(blob);
  }
  for (const auto& mem : state->frozen) {
    for (uint32_t doc = 0; doc < mem->published_docs(); ++doc) {
      mark(mem->blob_of_doc(doc));
    }
  }
  for (uint32_t doc = 0; doc < state->mem_watermark; ++doc) {
    mark(state->mem->blob_of_doc(doc));
  }
  indexed_ = std::move(indexed);
  indexed_manifest_seq_ = state->version->manifest_seq;
}

void Workspace::DropReferences(Worktree& tree) {
  absl::MutexLock lock(tree.mu);
  absl::flat_hash_set<BlobId> blobs;
  for (const auto& [path, state] : tree.files) {
    if (state.blob.has_value()) blobs.insert(*state.blob);
  }
  std::vector<Index::ReferenceDelta> deltas;
  deltas.reserve(blobs.size());
  for (BlobId blob : blobs) deltas.push_back({blob, -1});
  // Removal from trees_ precedes this lock; an in-flight publisher finishes
  // before these decrements. A search with an old ReadState retains its docs.
  CHECK_OK(index_->AdjustReferences(deltas));
}

void Workspace::AddStats(const WorkspaceStats& delta) {
  absl::MutexLock lock(stats_mu_);
  Accumulate(stats_, delta);
}

std::shared_ptr<Workspace::BulkJob> Workspace::EnqueueRescan(
    const std::shared_ptr<Worktree>& tree, uint64_t generation) {
  absl::MutexLock lock(mu_);
  if (stopping_) return nullptr;
  // A queued rescan has not started scanning, so it covers this request.
  std::shared_ptr<BulkJob>& queued = queued_rescans_[tree.get()];
  if (queued == nullptr) {
    queued = std::make_shared<BulkJob>();
    queued->tree = tree;
    ScheduleBulkLocked(queued);
  }
  queued->generations.push_back(generation);
  return queued;
}

void Workspace::ScheduleBulkLocked(std::shared_ptr<BulkJob> job) {
  ++bulk_queued_;
  bulk_lane_->Schedule([this, job = std::move(job)] { RunBulkJob(*job); });
}

void Workspace::RunBulkJob(BulkJob& job) {
  std::vector<uint64_t> generations;
  {
    absl::MutexLock lock(mu_);
    --bulk_queued_;
    const auto it = queued_rescans_.find(job.tree.get());
    if (it != queued_rescans_.end() && it->second.get() == &job) {
      queued_rescans_.erase(it);
    }
    generations = job.generations;
  }
  bool removed;
  {
    absl::MutexLock lock(job.tree->generation_mu);
    removed = job.tree->removed;
  }
  WorkspaceStats delta;
  absl::Status status;
  if (cancelling_.load(std::memory_order_relaxed)) {
    status = absl::CancelledError("workspace shutting down");
  } else if (!removed) {
    if (!job.tree->git_dir.empty()) {
      const std::string lock_path =
          absl::StrCat(job.tree->root, "/", job.tree->git_dir, "/index.lock");
      const absl::Time deadline = absl::Now() + options_.index_lock_wait;
      while (access(lock_path.c_str(), F_OK) == 0 && absl::Now() < deadline &&
             !cancelling_.load(std::memory_order_relaxed)) {
        absl::MutexLock lock(mu_);
        mu_.AwaitWithDeadline(absl::Condition(this, &Workspace::StoppingLocked),
                              absl::Now() + absl::Milliseconds(10));
      }
      if (access(lock_path.c_str(), F_OK) == 0 &&
          !cancelling_.load(std::memory_order_relaxed)) {
        ++delta.index_lock_timeouts;
      }
    }
    if (!cancelling_.load(std::memory_order_relaxed)) {
      status = Reconcile(*job.tree, delta);
    } else {
      status = absl::CancelledError("workspace shutting down");
    }
  }
  if (!job.registration) {
    job.tree->reload_ignore.store(true, std::memory_order_relaxed);
    job.tree->filter_reload.store(true, std::memory_order_relaxed);
  }
  if (delta.ingest_failures != 0) RequestIngestRecovery();
  AddStats(delta);
  CompleteWhenIndexed(*job.tree, std::move(generations));
  job.status = std::move(status);
  job.done.Notify();
}

void Workspace::RequestIngestRecovery() {
  absl::MutexLock lock(mu_);
  recovery_requested_ = true;
  if (recovery_scheduled_ || stopping_) return;
  recovery_scheduled_ = true;
  // A failure soon after a successful retry, such as a segment write that
  // fails again, keeps backing off.
  const absl::Time now = absl::Now();
  recovery_delay_ =
      now - recovery_ended_ > options_.ingest_retry_max_delay
          ? options_.ingest_retry_initial_delay
          : std::min(2 * recovery_delay_, options_.ingest_retry_max_delay);
  recovery_due_ = now + recovery_delay_;
  bulk_lane_->Schedule([this] { RunIngestRecovery(); });
}

bool Workspace::RecoveryWaitOverLocked() const {
  return stopping_ || bulk_queued_ > 0;
}

void Workspace::RunIngestRecovery() {
  for (;;) {
    {
      absl::MutexLock lock(mu_);
      mu_.AwaitWithDeadline(
          absl::Condition(this, &Workspace::RecoveryWaitOverLocked),
          recovery_due_);
      if (stopping_) {
        recovery_scheduled_ = false;
        return;
      }
      if (absl::Now() < recovery_due_) {
        // Let the queued registrations and rescans run, and wait behind them.
        bulk_lane_->Schedule([this] { RunIngestRecovery(); });
        return;
      }
      recovery_requested_ = false;
    }
    // The sticky flush error clears only when Flush retries the flush.
    const absl::Status flushed = index_->Flush();
    WorkspaceStats delta;
    ++delta.ingest_retries;
    AddStats(delta);
    if (flushed.ok()) ResubmitFailed();
    absl::MutexLock lock(mu_);
    if (flushed.ok() && !recovery_requested_) {
      recovery_scheduled_ = false;
      recovery_ended_ = absl::Now();
      return;
    }
    // The retry failed, or another write failed after it began.
    recovery_delay_ =
        std::min(2 * recovery_delay_, options_.ingest_retry_max_delay);
    recovery_due_ = absl::Now() + recovery_delay_;
  }
}

void Workspace::ResubmitFailed() {
  std::vector<std::shared_ptr<Worktree>> trees;
  {
    absl::MutexLock lock(mu_);
    trees.reserve(trees_.size());
    for (const auto& [id, tree] : trees_) trees.push_back(tree);
  }
  for (const std::shared_ptr<Worktree>& tree : trees) {
    std::vector<PathId> ids;
    {
      absl::MutexLock lock(tree->mu);
      ids.reserve(tree->failed.size());
      for (const auto& [id, seq] : tree->failed) ids.push_back(id);
    }
    if (ids.empty()) continue;
    // The bulk lane rereads many paths faster, writing segments if needed.
    if (ids.size() > kMaxNotifyBatchPaths) {
      EnqueueRescan(tree, AssignGeneration(*tree));
      continue;
    }
    std::vector<std::string> paths;
    paths.reserve(ids.size());
    for (const PathId id : ids) paths.push_back(paths_.path(id));
    absl::MutexLock lock(mu_);
    if (stopping_) return;
    QueueNotifyLocked(tree, paths, ids);
  }
}

void Workspace::DrainInteractive() {
  for (;;) {
    std::vector<NotifyJob> batch;
    {
      absl::MutexLock lock(mu_);
      if (stopping_ || notify_jobs_.empty()) {
        interactive_draining_ = false;
        return;
      }
      // Coalesce queued jobs of one worktree into one view publication.
      const Worktree* tree = notify_jobs_.front().tree.get();
      size_t taken = 0;
      size_t paths = 0;
      while (taken < notify_jobs_.size() &&
             notify_jobs_[taken].tree.get() == tree &&
             (taken == 0 || paths + notify_jobs_[taken].paths.size() <=
                                kMaxNotifyBatchPaths)) {
        paths += notify_jobs_[taken].paths.size();
        batch.push_back(std::move(notify_jobs_[taken++]));
      }
      notify_jobs_.erase(notify_jobs_.begin(),
                         notify_jobs_.begin() + static_cast<ptrdiff_t>(taken));
    }
    watch_trace::Mark(watch_trace::kDequeued);
    ApplyNotifyBatch(batch.front().tree, batch);
  }
}

Workspace::FileRead Workspace::ReadFile(const Worktree& tree,
                                        absl::string_view path, bool bulk) {
  FileRead result;
  result.seq = NextSeq();
  result.started_nanos = absl::ToUnixNanos(absl::Now());
  // A Notify path may name anything, and opening a device or FIFO can have
  // side effects, so only a regular file is opened. Bulk paths come from a
  // scan that has already checked this.
  if (!bulk) {
    struct stat st;
    if (!StatNoFollow(tree.root, path, st)) {
      result.outcome = IsAbsentErrno(errno) ? FileRead::Outcome::kMissing
                                            : FileRead::Outcome::kError;
      return result;
    }
    if (!S_ISREG(st.st_mode)) {
      result.outcome = S_ISDIR(st.st_mode) ? FileRead::Outcome::kDirectory
                                           : FileRead::Outcome::kMissing;
      return result;
    }
  }
  for (int attempt = 0; attempt < kReadAttempts; ++attempt) {
    const int fd = OpenNoFollow(tree.root, path);
    if (fd < 0) {
      result.outcome = IsAbsentErrno(errno) ? FileRead::Outcome::kMissing
                                            : FileRead::Outcome::kError;
      return result;
    }
    absl::Cleanup close_fd = [fd] { close(fd); };
    if (options_.read_hook != nullptr) (*options_.read_hook)(path, bulk);
    struct stat before;
    if (fstat(fd, &before) != 0) {
      result.outcome = FileRead::Outcome::kError;
      return result;
    }
    if (!S_ISREG(before.st_mode)) {
      result.outcome = S_ISDIR(before.st_mode) ? FileRead::Outcome::kDirectory
                                               : FileRead::Outcome::kMissing;
      return result;
    }
    // Another spelling of an existing name is not a path of the worktree, as
    // for scans. A file with several links may report another link's path, so
    // only single-link files are checked.
    if (!bulk && tree.case_insensitive && before.st_nlink == 1 &&
        !SpellingMatches(fd, tree.root.size(), path)) {
      result.outcome = FileRead::Outcome::kMissing;
      return result;
    }
    result.signature = Signature(before);
    if (before.st_size > options_.eligibility.max_file_bytes) {
      result.outcome = FileRead::Outcome::kTooLarge;
      return result;
    }
    result.bytes.resize(static_cast<size_t>(before.st_size));
    size_t filled = 0;
    bool failed = false;
    while (filled < result.bytes.size()) {
      const ssize_t count =
          read(fd, result.bytes.data() + filled, result.bytes.size() - filled);
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) {
        failed = count < 0;
        break;
      }
      filled += static_cast<size_t>(count);
    }
    if (options_.after_read_hook != nullptr) {
      (*options_.after_read_hook)(path, bulk);
    }
    struct stat after;
    if (failed || fstat(fd, &after) != 0) {
      result.bytes.clear();
      result.outcome = FileRead::Outcome::kError;
      return result;
    }
    if (filled == result.bytes.size() && Signature(after) == result.signature) {
      result.outcome = FileRead::Outcome::kPresent;
      return result;
    }
  }
  result.bytes = std::string();
  result.outcome = FileRead::Outcome::kUnstable;
  return result;
}

void Workspace::ApplyRead(Update& update, FileRead& read,
                          TrigramExtractor& extractor, bool retain,
                          WorkspaceStats& delta) const {
  update.seq = read.seq;
  switch (read.outcome) {
    case FileRead::Outcome::kMissing:
      update.kind = Update::Kind::kDeleted;
      return;
    case FileRead::Outcome::kDirectory:
      update.kind = Update::Kind::kDeleted;
      update.rescan = true;
      return;
    case FileRead::Outcome::kTooLarge:
      update.kind = Update::Kind::kSkipped;
      update.skip = SkipReason::kTooLarge;
      update.signature = read.signature;
      return;
    case FileRead::Outcome::kUnstable:
      update.kind = Update::Kind::kKeep;
      ++delta.unstable_files;
      return;
    case FileRead::Outcome::kError:
      update.kind = Update::Kind::kKeep;
      ++delta.read_errors;
      return;
    case FileRead::Outcome::kPresent:
      break;
  }
  update.signature = read.ReusableSignature();
  update.prepared =
      PrepareDocument(update.path, read.bytes, options_.eligibility, extractor);
  if (update.prepared.skip_reason != SkipReason::kNone) {
    update.kind = Update::Kind::kSkipped;
    update.skip = update.prepared.skip_reason;
    // Eligibility depends only on content, so other paths with this Oid,
    // including hints in other worktrees, need not be read.
    update.skipped_oid = HashBlob(read.bytes);
    return;
  }
  update.kind = Update::Kind::kPresent;
  update.oid = update.prepared.rank_key.oid;
  update.blob.reset();
  update.size = static_cast<int64_t>(read.bytes.size());
  if (retain) update.bytes = std::move(read.bytes);
}

std::optional<BlobId> Workspace::FindIndexed(const Oid& oid) const {
  const std::optional<BlobId> blob = index_->blobs().Find(oid);
  if (!blob.has_value()) return std::nullopt;
  absl::MutexLock lock(content_mu_);
  return IndexedLocked(*blob) ? blob : std::nullopt;
}

bool Workspace::IndexedLocked(BlobId blob) const {
  const uint32_t id = static_cast<uint32_t>(blob);
  return id < indexed_.size() && indexed_.Test(id);
}

void Workspace::MarkIndexed(const absl::flat_hash_map<Oid, BlobId>& added) {
  absl::MutexLock lock(content_mu_);
  for (const auto& [oid, blob] : added) {
    const uint32_t id = static_cast<uint32_t>(blob);
    if (id >= indexed_.size()) indexed_.Resize(std::max(id + 1, 2 * id));
    indexed_.Set(id);
  }
}

void Workspace::RecordSkips(absl::Span<const Update> updates,
                            WorkspaceStats& delta) {
  std::vector<std::pair<Oid, SkipReason>> ineligible;
  for (const Update& update : updates) {
    if (update.kind != Update::Kind::kSkipped) continue;
    ++delta.skipped[static_cast<size_t>(update.skip)];
    if (update.skipped_oid.has_value()) {
      ineligible.emplace_back(*update.skipped_oid, update.skip);
    }
  }
  if (ineligible.empty()) return;
  absl::MutexLock lock(content_mu_);
  ineligible_.insert(ineligible.begin(), ineligible.end());
}

absl::Status Workspace::Reconcile(Worktree& tree, WorkspaceStats& delta) {
  PinGc();
  absl::Cleanup unpin = [this] { UnpinGc(); };
  RefreshIndexed();
  // Hint and signature results carry the scan's sequence: they describe the
  // tree no earlier than the start of the scan.
  ScanSummary summary;
  summary.seq = NextSeq();
  const uint64_t scan_seq = summary.seq;
  absl::Time phase = absl::Now();
  std::vector<ScannedWorktreeFile> scanned;
  if (tree.git) {
    ABSL_ASSIGN_OR_RETURN(WorktreeScan scan, ScanWorktree(tree.root));
    scanned = std::move(scan.files);
    summary.nested.insert(
        std::make_move_iterator(scan.nested_repositories.begin()),
        std::make_move_iterator(scan.nested_repositories.end()));
  } else {
    DirectoryFilesOptions options;
    if (!tree.index_prefix.empty()) {
      options.index_directory =
          tree.index_prefix.substr(0, tree.index_prefix.size() - 1);
    }
    ABSL_ASSIGN_OR_RETURN(std::vector<std::string> listed,
                          ListDirectoryFiles(tree.root, options));
    scanned.resize(listed.size());
    for (size_t i = 0; i < listed.size(); ++i) {
      scanned[i].path = std::move(listed[i]);
    }
  }
  if (!tree.cookie_dir.empty()) {
    std::erase_if(scanned, [&tree](const ScannedWorktreeFile& file) {
      return file.path == tree.cookie_dir ||
             absl::StartsWith(file.path, absl::StrCat(tree.cookie_dir, "/"));
    });
  }
  delta.scan_time += absl::Now() - phase;
  if (cancelling_.load(std::memory_order_relaxed)) {
    return absl::CancelledError("workspace shutting down");
  }

  phase = absl::Now();
  std::vector<Update> updates;
  updates.reserve(scanned.size());
  absl::flat_hash_set<PathId>& tracked = summary.tracked;
  std::vector<size_t> hinted;
  std::vector<size_t> unhinted;
  for (const ScannedWorktreeFile& file : scanned) {
    if (!tree.index_prefix.empty() &&
        absl::StartsWith(file.path, tree.index_prefix)) {
      continue;
    }
    // PathTable rejects a few names git allows, such as ones with a
    // backslash; such files are not indexed.
    const absl::StatusOr<PathId> id = paths_.Intern(file.path);
    if (!id.ok()) continue;
    Update& update = updates.emplace_back();
    update.id = *id;
    update.path = file.path;
    update.seq = scan_seq;
    if (tree.git && file.read_reason != ScanReadReason::kUntracked) {
      tracked.insert(*id);
      AddAncestors(summary.tracked_dirs, file.path);
    }
    const SkipReason by_size =
        file.size.has_value()
            ? SkipReasonForSize(*file.size, options_.eligibility)
            : SkipReason::kNone;
    if (tree.git && !file.size.has_value()) {
      // Missing, or spelled unlike its directory entry.
      update.kind = Update::Kind::kDeleted;
    } else if (by_size != SkipReason::kNone) {
      update.kind = Update::Kind::kSkipped;
      update.skip = by_size;
    } else if (file.skip_hint.has_value()) {
      update.kind = Update::Kind::kHint;
      update.oid = *file.skip_hint;
      update.blob = index_->blobs().Find(update.oid);
      hinted.push_back(updates.size() - 1);
    } else {
      update.kind = Update::Kind::kStat;
      unhinted.push_back(updates.size() - 1);
    }
  }
  for (size_t begin = 0; begin < hinted.size(); begin += kLockBatch) {
    const size_t end = std::min(begin + kLockBatch, hinted.size());
    absl::MutexLock lock(content_mu_);
    for (size_t i = begin; i < end; ++i) {
      Update& update = updates[hinted[i]];
      if (update.blob.has_value() && IndexedLocked(*update.blob)) {
        update.kind = Update::Kind::kPresent;
        ++delta.hints_reused;
        continue;
      }
      update.blob.reset();
      const auto ineligible = ineligible_.find(update.oid);
      if (ineligible != ineligible_.end()) {
        update.kind = Update::Kind::kSkipped;
        update.skip = ineligible->second;
        ++delta.hints_reused;
        continue;
      }
      update.kind = Update::Kind::kRead;
      ++delta.hint_reads;
    }
  }
  for (size_t begin = 0; begin < unhinted.size(); begin += kLockBatch) {
    const size_t end = std::min(begin + kLockBatch, unhinted.size());
    absl::MutexLock lock(tree.mu);
    for (size_t i = begin; i < end; ++i) {
      Update& update = updates[unhinted[i]];
      const auto state = tree.files.find(update.id);
      if (state == tree.files.end()) continue;
      update.previous_blob = state->second.blob;
      update.previous_skip = state->second.skip;
      const auto signature = tree.signatures.find(update.id);
      if (signature != tree.signatures.end()) {
        update.previous_signature = signature->second;
      }
    }
  }
  delta.classify_time += absl::Now() - phase;

  phase = absl::Now();
  ReadForBulk(tree, updates, delta);
  delta.read_time += absl::Now() - phase;
  if (cancelling_.load(std::memory_order_relaxed)) {
    return absl::CancelledError("workspace shutting down");
  }
  RecordSkips(updates, delta);

  // A failed write keeps the affected paths' previous entries; publish the
  // rest before reporting it.
  const absl::Status written =
      AddNewContent(tree, updates, /*bulk=*/true, delta);
  if (absl::IsCancelled(written)) return written;

  phase = absl::Now();
  Publish(tree, updates, &summary);
  delta.publish_time += absl::Now() - phase;
  return written;
}

void Workspace::ReadForBulk(const Worktree& tree, std::vector<Update>& updates,
                            WorkspaceStats& delta) {
  std::vector<size_t> work;
  for (size_t i = 0; i < updates.size(); ++i) {
    if (updates[i].kind == Update::Kind::kStat ||
        updates[i].kind == Update::Kind::kRead) {
      work.push_back(i);
    }
  }
  const int workers = options_.reader_threads;
  std::vector<WorkspaceStats> worker_delta(static_cast<size_t>(workers));
  std::vector<std::unique_ptr<TrigramExtractor>> extractors(
      static_cast<size_t>(workers));
  readers_->ParallelFor(
      static_cast<int64_t>(work.size()), /*grain=*/16, workers,
      [&](int worker, int64_t begin, int64_t end) {
        const auto slot = static_cast<size_t>(worker);
        if (extractors[slot] == nullptr) {
          extractors[slot] = std::make_unique<TrigramExtractor>();
        }
        WorkspaceStats& local = worker_delta[slot];
        for (int64_t i = begin; i < end; ++i) {
          if (cancelling_.load(std::memory_order_relaxed)) return;
          Update& update = updates[work[static_cast<size_t>(i)]];
          if (update.kind == Update::Kind::kStat) {
            struct stat st;
            if (!StatNoFollow(tree.root, update.path, st) ||
                !S_ISREG(st.st_mode)) {
              update.kind = Update::Kind::kDeleted;
              continue;
            }
            const FileSignature signature = Signature(st);
            if (update.previous_signature == signature &&
                (update.previous_blob.has_value() ||
                 update.previous_skip != SkipReason::kNone)) {
              ++local.signatures_reused;
              update.signature = signature;
              if (update.previous_blob.has_value()) {
                update.kind = Update::Kind::kPresent;
                update.blob = update.previous_blob;
              } else {
                update.kind = Update::Kind::kSkipped;
                update.skip = update.previous_skip;
              }
              continue;
            }
            // A signature with this size keeps the reason valid.
            const SkipReason by_size = SkipReasonForSize(
                static_cast<int64_t>(st.st_size), options_.eligibility);
            if (by_size != SkipReason::kNone) {
              update.kind = Update::Kind::kSkipped;
              update.skip = by_size;
              update.signature = signature;
              continue;
            }
            update.kind = Update::Kind::kRead;
          }
          FileRead read = ReadFile(tree, update.path, /*bulk=*/true);
          ++local.bulk_files_read;
          local.bulk_bytes_read += read.bytes.size();
          ApplyRead(update, read, *extractors[slot], /*retain=*/false, local);
        }
      });
  for (const WorkspaceStats& local : worker_delta) Accumulate(delta, local);
}

absl::Status Workspace::AddNewContent(const Worktree& tree,
                                      std::vector<Update>& updates, bool bulk,
                                      WorkspaceStats& delta) {
  // Resolve content that other batches already indexed.
  std::vector<size_t> fresh;
  for (size_t i = 0; i < updates.size(); ++i) {
    Update& update = updates[i];
    if (update.kind != Update::Kind::kPresent || update.blob.has_value()) {
      continue;
    }
    update.blob = index_->blobs().Find(update.oid);
    fresh.push_back(i);
  }
  std::vector<size_t> unresolved;
  for (size_t begin = 0; begin < fresh.size(); begin += kLockBatch) {
    const size_t end = std::min(begin + kLockBatch, fresh.size());
    absl::MutexLock lock(content_mu_);
    for (size_t i = begin; i < end; ++i) {
      Update& update = updates[fresh[i]];
      if (update.blob.has_value() && IndexedLocked(*update.blob)) continue;
      update.blob.reset();
      unresolved.push_back(fresh[i]);
    }
  }
  if (unresolved.empty()) return absl::OkStatus();

  // Each new Oid is written once, from its bytewise-smallest path in this
  // batch; that first-seen path fixes its rank key.
  absl::flat_hash_map<Oid, size_t> first;
  for (const size_t i : unresolved) {
    const auto [it, inserted] = first.try_emplace(updates[i].oid, i);
    if (!inserted && updates[i].path < updates[it->second].path) {
      it->second = i;
    }
  }
  std::vector<size_t> order;
  order.reserve(first.size());
  int64_t new_bytes = 0;
  for (const auto& [oid, i] : first) {
    order.push_back(i);
    new_bytes += updates[i].size;
  }
  std::sort(order.begin(), order.end(), [&updates](size_t a, size_t b) {
    return std::tie(updates[a].prepared.rank_key, updates[a].path) <
           std::tie(updates[b].prepared.rank_key, updates[b].path);
  });

  absl::flat_hash_map<Oid, BlobId> added;
  absl::Status status;
  if (!bulk) {
    for (const size_t i : order) {
      Update& update = updates[i];
      status = index_->Add(update.prepared.Bind(update.path, update.bytes),
                           IngestLane::kInteractive);
      if (!status.ok()) break;
      added.emplace(update.oid, index_->blobs().Intern(update.oid));
      delta.interactive_bytes_ingested += static_cast<uint64_t>(update.size);
    }
  } else {
    const bool segments =
        new_bytes >= static_cast<int64_t>(options_.index.memtable.flush_bytes);
    std::vector<Oid> changed;
    status = WriteBulk(tree, updates, order, segments, added, changed, delta);
    // Another path may still hold the content a changed file had.
    if (status.ok() && !changed.empty()) {
      status =
          AddFromOtherPaths(tree, updates, unresolved, changed, added, delta);
    }
  }
  MarkIndexed(added);
  for (const size_t i : unresolved) {
    Update& update = updates[i];
    if (update.kind != Update::Kind::kPresent || update.blob.has_value()) {
      continue;
    }
    const auto it = added.find(update.oid);
    if (it != added.end()) {
      update.blob = it->second;
    } else {
      update.kind = Update::Kind::kKeep;
      update.ingest_failed = !status.ok() && !absl::IsCancelled(status);
    }
  }
  if (!status.ok() && !absl::IsCancelled(status)) {
    ++delta.ingest_failures;
    delta.last_ingest_error = status;
  }
  return status;
}

absl::Status Workspace::WriteBulk(const Worktree& tree,
                                  std::vector<Update>& updates,
                                  absl::Span<const size_t> order, bool segments,
                                  absl::flat_hash_map<Oid, BlobId>& added,
                                  std::vector<Oid>& changed,
                                  WorkspaceStats& delta) {
  std::vector<int64_t> sizes;
  sizes.reserve(order.size());
  for (const size_t i : order) sizes.push_back(updates[i].size);

  // Direct segments are planned before the reads, from the sizes of the first
  // reads: a segment ends before the file that would pass bulk_segment_bytes.
  // plan[s] is the first rank position of segment s.
  const bool precompute = segments && !options_.serial_bulk_write;
  const SegmentWriterOptions& writer_options = options_.index.memtable.segment;
  std::vector<size_t> plan;
  std::vector<size_t> segment_of;
  std::vector<std::unique_ptr<SegmentPrecompute>> contexts;
  SymbolExtractorPool symbol_pool;
  if (precompute) {
    int64_t segment_bytes = 0;
    segment_of.reserve(order.size());
    for (size_t k = 0; k < order.size(); ++k) {
      if (k == 0 || segment_bytes + sizes[k] > options_.bulk_segment_bytes) {
        plan.push_back(k);
        contexts.push_back(std::make_unique<SegmentPrecompute>(writer_options));
        segment_bytes = 0;
      }
      segment_bytes += sizes[k];
      segment_of.push_back(plan.size() - 1);
    }
    plan.push_back(order.size());
  }

  std::atomic<int64_t> reread_ns{0};
  std::atomic<int64_t> symbol_wait_ns{0};
  std::atomic<int64_t> symbol_parse_ns{0};
  ReadAhead<FileRead> rereads(
      *readers_, std::move(sizes), options_.bulk_prefetch_bytes,
      [this, &tree, &updates, order, segments, precompute, &plan, &segment_of,
       &contexts, &symbol_pool, &reread_ns, &symbol_wait_ns, &symbol_parse_ns,
       &writer_options](size_t k) -> FileRead {
        const int64_t t0 = absl::GetCurrentTimeNanos();
        const Update& update = updates[order[k]];
        FileRead read = ReadFile(tree, update.path, /*bulk=*/true);
        if (read.outcome == FileRead::Outcome::kPresent) {
          read.oid = HashBlob(read.bytes);
        }
        reread_ns += absl::GetCurrentTimeNanos() - t0;
        if (!segments || read.oid != update.oid) return read;
        const Language language = update.prepared.language;
        if (precompute) {
          // The Oid matches the one the rank was computed for, so these are
          // the bytes the document will store.
          const size_t s = segment_of[k];
          SegmentPrecompute& context = *contexts[s];
          SegmentPrecompute::Slot* slot = context.Acquire();
          absl::StatusOr<PrecomputedDocument> pre =
              slot->precomputer.Precompute(read.bytes, language,
                                           static_cast<uint32_t>(k - plan[s]));
          context.Release(slot);
          if (pre.ok()) {
            read.pre = std::move(*pre);
            read.precomputed = true;
          } else {
            read.precompute_status = pre.status();
          }
          if (writer_options.write_symbols && HasSymbolGrammar(language)) {
            const int64_t p0 = absl::GetCurrentTimeNanos();
            std::unique_ptr<SymbolExtractor> extractor = symbol_pool.Acquire();
            absl::StatusOr<std::vector<ExtractedSymbol>> extracted =
                extractor->Extract(language, read.bytes);
            symbol_pool.Release(std::move(extractor));
            symbol_parse_ns += absl::GetCurrentTimeNanos() - p0;
            if (extracted.ok()) {
              read.symbols = std::move(*extracted);
            } else {
              read.symbol_failed = true;
            }
          }
        } else if (bulk_symbol_pool_ && HasSymbolGrammar(language)) {
          const size_t slot =
              next_bulk_symbol_worker_.fetch_add(1, std::memory_order_relaxed) %
              bulk_symbol_workers_.size();
          BulkSymbolWorker& worker = *bulk_symbol_workers_[slot];
          absl::Notification finished;
          const int64_t w0 = absl::GetCurrentTimeNanos();
          bulk_symbol_pool_->Schedule([&] {
            absl::MutexLock lock(worker.mu);
            const int64_t p0 = absl::GetCurrentTimeNanos();
            absl::StatusOr<std::vector<ExtractedSymbol>> extracted =
                worker.extractor.Extract(language, read.bytes);
            symbol_parse_ns += absl::GetCurrentTimeNanos() - p0;
            if (extracted.ok()) {
              read.symbols = std::move(*extracted);
            } else {
              read.symbol_failed = true;
            }
            finished.Notify();
          });
          finished.WaitForNotification();
          symbol_wait_ns += absl::GetCurrentTimeNanos() - w0;
        }
        return read;
      });
  TrigramExtractor extractor;
  size_t next = 0;
  // Takes the next reread. `write` receives it if the Oid still matches;
  // otherwise the file changed and its new content goes to the memtable,
  // where rank order does not matter.
  const auto take =
      [&](absl::FunctionRef<absl::Status(Update&, FileRead&)> write)
      -> absl::Status {
    Update& update = updates[order[next++]];
    const absl::Time wait_start = absl::Now();
    FileRead& read = rereads.Next();
    delta.bulk_wait_time += absl::Now() - wait_start;
    ++delta.bulk_files_reread;
    delta.bulk_bytes_reread += read.bytes.size();
    if (read.oid == update.oid) {
      update.seq = read.seq;
      update.signature = read.ReusableSignature();
      return write(update, read);
    }
    ++delta.changed_before_write;
    changed.push_back(update.oid);
    return AddChanged(update, read, extractor, added, delta);
  };
  const auto cancelled = [this]() -> bool {
    return cancelling_.load(std::memory_order_relaxed);
  };
  absl::Cleanup record_thread_times = [&] {
    delta.bulk_reread_thread_time += absl::Nanoseconds(reread_ns.load());
    delta.bulk_symbol_wait_thread_time +=
        absl::Nanoseconds(symbol_wait_ns.load());
    delta.bulk_symbol_parse_thread_time +=
        absl::Nanoseconds(symbol_parse_ns.load());
  };

  if (!segments) {
    const absl::Time start = absl::Now();
    absl::Status status;
    while (status.ok() && next < order.size()) {
      if (cancelled()) {
        status = absl::CancelledError("workspace shutting down");
        break;
      }
      status = take([&](Update& update, FileRead& read) -> absl::Status {
        const absl::string_view bytes = read.bytes;
        ABSL_RETURN_IF_ERROR(index_->Add(
            update.prepared.Bind(update.path, bytes), IngestLane::kBulk));
        added.emplace(update.oid, index_->blobs().Intern(update.oid));
        delta.bulk_bytes_ingested += bytes.size();
        return absl::OkStatus();
      });
    }
    delta.memtable_add_time += absl::Now() - start;
    return status;
  }

  // Consecutive rank ranges, each at most bulk_segment_bytes.
  size_t segment_index = 0;
  while (next < order.size()) {
    const absl::Time start = absl::Now();
    std::vector<size_t> written;
    int64_t segment_bytes = 0;
    absl::Duration callback_time;
    const auto write_documents = [&](SegmentWriter& writer) -> absl::Status {
      const absl::Cleanup time_callback = [&] {
        callback_time = absl::Now() - start;
      };
      // Where each provisional chunk and document of a precomputed segment
      // ended up; chunks and documents that were not committed keep 0, and
      // no posting or token names them.
      std::vector<ChunkId> chunk_remap;
      std::vector<DocId> doc_remap;
      uint32_t next_chunk = 0;
      size_t first = 0;
      size_t end = order.size();
      if (precompute) {
        first = plan[segment_index];
        end = plan[segment_index + 1];
        chunk_remap.resize(contexts[segment_index]->num_provisional_chunks());
        doc_remap.assign(end - first, 0);
      }
      while (next < end && (precompute || segment_bytes == 0 ||
                            segment_bytes + updates[order[next]].size <=
                                options_.bulk_segment_bytes)) {
        if (cancelled()) {
          return absl::CancelledError("workspace shutting down");
        }
        const size_t k = next;
        const size_t index = order[k];
        ABSL_RETURN_IF_ERROR(
            take([&](Update& update, FileRead& read) -> absl::Status {
              const absl::string_view bytes = read.bytes;
              const absl::Time add_start = absl::Now();
              if (precompute) {
                ABSL_RETURN_IF_ERROR(read.precompute_status);
                const PrecomputedDocument& pre = read.pre;
                if (chunk_remap.size() < pre.first_chunk + pre.chunks.size()) {
                  chunk_remap.resize(pre.first_chunk + pre.chunks.size());
                }
                for (size_t c = 0; c < pre.chunks.size(); ++c) {
                  chunk_remap[pre.first_chunk + c] =
                      next_chunk + static_cast<uint32_t>(c);
                }
                doc_remap[k - first] = writer.num_docs();
                ABSL_RETURN_IF_ERROR(writer.AddPrecomputed(
                    update.prepared.Bind(update.path, bytes), pre.num_lines,
                    pre.chunks));
                next_chunk += static_cast<uint32_t>(pre.chunks.size());
                writer.AddSymbolsFor(doc_remap[k - first], read.symbols);
                if (read.symbol_failed) {
                  writer.MarkSymbolsFailedFor(doc_remap[k - first]);
                }
              } else {
                ABSL_RETURN_IF_ERROR(
                    writer.Add(update.prepared.Bind(update.path, bytes),
                               /*extract_symbols=*/false));
                writer.AddSymbols(read.symbols);
                if (read.symbol_failed) writer.MarkSymbolsFailed();
              }
              delta.bulk_add_time += absl::Now() - add_start;
              segment_bytes += static_cast<int64_t>(bytes.size());
              written.push_back(index);
              return absl::OkStatus();
            }));
      }
      if (precompute && writer.num_docs() != 0) {
        // Every read of the segment has been taken, so its slots are idle.
        SegmentPrecompute& context = *contexts[segment_index];
        chunk_remap.resize(std::max<size_t>(chunk_remap.size(),
                                            context.num_provisional_chunks()));
        std::vector<segment_internal::PostingAccumulator::Run> runs;
        std::vector<segment_internal::CodeTokenMap> token_maps;
        for (const auto& slot : context.slots()) {
          for (auto& run : slot->precomputer.TakeRuns()) {
            runs.push_back(std::move(run));
          }
          token_maps.push_back(slot->precomputer.TakeCodeTokens());
        }
        writer.SetPostings(std::move(runs), std::move(chunk_remap));
        if (writer_options.write_code_tokens) {
          writer.SetCodeTokenMaps(std::move(token_maps), std::move(doc_remap));
        }
      }
      return absl::OkStatus();
    };
    const absl::StatusOr<SegmentStats> segment =
        precompute ? index_->AddSegment(write_documents, readers_.get(),
                                        options_.reader_threads)
                   : index_->AddSegment(write_documents);
    if (precompute) contexts[segment_index].reset();
    ++segment_index;
    delta.segment_write_time += absl::Now() - start;
    delta.bulk_finish_time += absl::Now() - start - callback_time;
    if (!segment.ok()) return segment.status();
    // AddSegment returns after publishing the segment's ReadState.
    for (const size_t i : written) {
      added.emplace(updates[i].oid, index_->blobs().Intern(updates[i].oid));
      delta.bulk_bytes_ingested += static_cast<uint64_t>(updates[i].size);
    }
  }
  return absl::OkStatus();
}

absl::Status Workspace::AddChanged(Update& update, FileRead& read,
                                   TrigramExtractor& extractor,
                                   absl::flat_hash_map<Oid, BlobId>& added,
                                   WorkspaceStats& delta) {
  ApplyRead(update, read, extractor, /*retain=*/false, delta);
  if (update.kind == Update::Kind::kSkipped) {
    RecordSkips(absl::MakeConstSpan(&update, 1), delta);
  }
  if (update.kind != Update::Kind::kPresent) return absl::OkStatus();
  if (const auto it = added.find(update.oid); it != added.end()) {
    update.blob = it->second;
    return absl::OkStatus();
  }
  update.blob = FindIndexed(update.oid);
  if (update.blob.has_value()) return absl::OkStatus();
  ABSL_RETURN_IF_ERROR(index_->Add(
      update.prepared.Bind(update.path, read.bytes), IngestLane::kBulk));
  update.blob = index_->blobs().Intern(update.oid);
  added.emplace(update.oid, *update.blob);
  delta.bulk_bytes_ingested += read.bytes.size();
  return absl::OkStatus();
}

absl::Status Workspace::AddFromOtherPaths(
    const Worktree& tree, std::vector<Update>& updates,
    absl::Span<const size_t> unresolved, absl::Span<const Oid> changed,
    absl::flat_hash_map<Oid, BlobId>& added, WorkspaceStats& delta) {
  const absl::flat_hash_set<Oid> lost(changed.begin(), changed.end());
  std::vector<size_t> candidates;
  for (const size_t i : unresolved) {
    const Update& update = updates[i];
    if (update.kind == Update::Kind::kPresent && !update.blob.has_value() &&
        lost.contains(update.oid)) {
      candidates.push_back(i);
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [&updates](size_t a, size_t b) {
              return updates[a].path < updates[b].path;
            });
  TrigramExtractor extractor;
  for (const size_t i : candidates) {
    Update& update = updates[i];
    if (added.contains(update.oid)) continue;
    FileRead read = ReadFile(tree, update.path, /*bulk=*/true);
    ++delta.bulk_files_reread;
    delta.bulk_bytes_reread += read.bytes.size();
    const Oid expected = update.oid;
    ABSL_RETURN_IF_ERROR(AddChanged(update, read, extractor, added, delta));
    if (update.kind != Update::Kind::kPresent || update.oid != expected) {
      ++delta.changed_before_write;
    }
  }
  return absl::OkStatus();
}

void Workspace::ApplyNotifyBatch(const std::shared_ptr<Worktree>& tree,
                                 absl::Span<const NotifyJob> jobs) {
  Worktree& worktree = *tree;
  PinGc();
  watch_trace::Mark(watch_trace::kPinned);
  absl::Cleanup unpin = [this] { UnpinGc(); };
  RefreshIndexed();
  watch_trace::Mark(watch_trace::kIndexRefreshed);
  WorkspaceStats delta;
  if (worktree.reload_ignore.exchange(false, std::memory_order_relaxed) ||
      !worktree.ignore.has_value()) {
    worktree.ignore.emplace(worktree.ignore_options);
  }
  IgnoreMatcher& ignore = *worktree.ignore;
  watch_trace::Mark(watch_trace::kIgnoreReady);
  std::vector<Update> updates;
  absl::flat_hash_map<PathId, size_t> position;
  for (const NotifyJob& job : jobs) {
    for (size_t k = 0; k < job.ids.size(); ++k) {
      if (!position.try_emplace(job.ids[k], updates.size()).second) continue;
      Update& update = updates.emplace_back();
      update.id = job.ids[k];
      update.path = job.paths[k];
    }
  }
  // Rules edited in this batch apply to the paths it reads.
  for (const Update& update : updates) {
    if (IsIgnoreFile(update.path)) {
      ignore.Invalidate(Dirname(update.path));
    }
  }
  std::vector<bool> tracked(updates.size());
  std::vector<bool> nested(updates.size());
  if (worktree.git) {
    absl::MutexLock lock(worktree.mu);
    for (size_t i = 0; i < updates.size(); ++i) {
      tracked[i] = worktree.tracked.contains(updates[i].id);
      nested[i] = !tracked[i] && UnderAny(worktree.nested, updates[i].path);
    }
  }
  watch_trace::Mark(watch_trace::kBatchSetup);
  // Directories of this batch known to hold a `.git` entry, or not to.
  absl::flat_hash_map<std::string, bool> repositories;
  // Paths that are not directories, whose old subtrees may still be visible.
  std::vector<bool> not_directory(updates.size());
  for (size_t i = 0; i < updates.size(); ++i) {
    Update& update = updates[i];
    if ((!worktree.cookie_dir.empty() &&
         (update.path == worktree.cookie_dir ||
          absl::StartsWith(update.path,
                           absl::StrCat(worktree.cookie_dir, "/")))) ||
        IsGitDirPath(update.path) ||
        (!worktree.index_prefix.empty() &&
         absl::StartsWith(update.path, worktree.index_prefix))) {
      update.kind = Update::Kind::kDeleted;
      update.seq = NextSeq();
      continue;
    }
    if (!tracked[i]) {
      update.seq = NextSeq();
      // Git lists nothing inside a submodule or nested repository, including
      // one created since the last scan (known by its `.git` entry), so
      // Notify does not either.
      if (nested[i]) {
        update.kind = Update::Kind::kDeleted;
        continue;
      }
      const absl::StatusOr<bool> ignored =
          ignore.IsIgnored(update.path, /*is_dir=*/false);
      if (!ignored.ok() || *ignored) {
        update.kind =
            ignored.ok() ? Update::Kind::kDeleted : Update::Kind::kKeep;
        if (!ignored.ok()) ++delta.read_errors;
        continue;
      }
      if (worktree.git &&
          InsideNestedRepository(worktree.root, update.path, repositories)) {
        update.kind = Update::Kind::kDeleted;
        continue;
      }
    }
    FileRead read = ReadFile(worktree, update.path, /*bulk=*/false);
    watch_trace::Mark(watch_trace::kFileRead);
    if (read.outcome != FileRead::Outcome::kMissing &&
        read.outcome != FileRead::Outcome::kDirectory &&
        read.outcome != FileRead::Outcome::kTooLarge) {
      ++delta.interactive_files_read;
    }
    not_directory[i] = read.outcome != FileRead::Outcome::kDirectory &&
                       read.outcome != FileRead::Outcome::kError;
    ApplyRead(update, read, interactive_extractor_, /*retain=*/true, delta);
  }
  // A directory that vanished or became a file leaves its files visible. The
  // rescan that completes the generation deletes them; reconciling just the
  // subtree is future work.
  {
    absl::MutexLock lock(worktree.mu);
    for (size_t i = 0; i < updates.size(); ++i) {
      if (not_directory[i] &&
          worktree.visible_children.contains(updates[i].path)) {
        updates[i].rescan = true;
      }
    }
  }
  RecordSkips(updates, delta);
  watch_trace::Mark(watch_trace::kRead);
  // A failed Add is counted in `delta`, keeps the paths' entries, and holds
  // back the generations until a retry adds the content.
  AddNewContent(worktree, updates, /*bulk=*/false, delta).IgnoreError();
  watch_trace::Mark(watch_trace::kAdded);
  Publish(worktree, updates, nullptr);
  watch_trace::Mark(watch_trace::kPublished);
  std::vector<uint64_t> completed;
  for (const NotifyJob& job : jobs) {
    bool rescan = false;
    for (const PathId id : job.ids) {
      rescan = rescan || updates[position.at(id)].rescan;
    }
    // Such a generation completes with the rescan of its worktree.
    if (!rescan || EnqueueRescan(tree, job.generation) == nullptr) {
      completed.push_back(job.generation);
    }
  }
  CompleteWhenIndexed(worktree, std::move(completed));
  if (delta.ingest_failures != 0) RequestIngestRecovery();
  AddStats(delta);
  std::move(unpin).Invoke();
  watch_trace::Mark(watch_trace::kUnpinned);
}

void Workspace::Publish(Worktree& tree, std::vector<Update>& updates,
                        ScanSummary* absl_nullable scan) {
  {
    absl::MutexLock lock(tree.generation_mu);
    if (tree.removed) return;
  }
  std::vector<ViewEdit> edits;
  absl::MutexLock lock(tree.mu);
  {
    absl::MutexLock generation_lock(tree.generation_mu);
    if (tree.removed) return;
  }
  if (tree.files.empty()) tree.files.reserve(updates.size());
  for (const Update& update : updates) {
    // Any newer read settles a failed path, whether or not it applies.
    if (update.ingest_failed) {
      uint64_t& failed = tree.failed[update.id];
      failed = std::max(failed, update.seq);
    } else if (const auto failed = tree.failed.find(update.id);
               failed != tree.failed.end() && update.seq > failed->second) {
      tree.failed.erase(failed);
    }
    const auto [it, inserted] = tree.files.try_emplace(update.id);
    PathState& state = it->second;
    const uint64_t newest =
        inserted ? tree.omitted_before : state.last_applied_seq;
    if (update.seq <= newest || update.kind == Update::Kind::kKeep) {
      if (inserted) tree.files.erase(it);
      continue;
    }
    state.last_applied_seq = update.seq;
    std::optional<BlobId> blob;
    state.skip = SkipReason::kNone;
    if (update.kind == Update::Kind::kPresent) {
      blob = update.blob;
    } else if (update.kind == Update::Kind::kSkipped) {
      state.skip = update.skip;
    }
    if (blob != state.blob) {
      edits.push_back({update.id, blob});
      if (!state.blob.has_value()) {
        AddVisible(tree.visible_children, update.path);
      } else if (!blob.has_value()) {
        RemoveVisible(tree.visible_children, update.path);
      }
    }
    state.blob = blob;
    if (update.signature.has_value()) {
      tree.signatures.insert_or_assign(update.id, *update.signature);
    } else {
      tree.signatures.erase(update.id);
    }
  }
  if (scan != nullptr) {
    // Paths absent from the scan, and not read since it started, are gone.
    std::vector<bool> in_scan(paths_.size());
    for (const Update& update : updates) {
      in_scan[static_cast<size_t>(update.id)] = true;
    }
    const auto omitted = [&in_scan, scan](PathId path, uint64_t seq) {
      const auto id = static_cast<size_t>(path);
      return (id >= in_scan.size() || !in_scan[id]) && seq < scan->seq;
    };
    for (auto it = tree.files.begin(); it != tree.files.end();) {
      if (!omitted(it->first, it->second.last_applied_seq)) {
        ++it;
        continue;
      }
      if (it->second.blob.has_value()) {
        edits.push_back({it->first, std::nullopt});
        RemoveVisible(tree.visible_children, paths_.path(it->first));
      }
      tree.signatures.erase(it->first);
      tree.files.erase(it++);
    }
    absl::erase_if(tree.failed, [&omitted](const auto& entry) {
      return omitted(entry.first, entry.second);
    });
    tree.omitted_before = scan->seq;
    tree.tracked = std::move(scan->tracked);
    tree.tracked_dirs = std::move(scan->tracked_dirs);
    tree.nested = std::move(scan->nested);
  }
  if (!edits.empty()) {
    tree.writer.ApplyBatch(edits, [this](absl::Span<const BlobId> entered,
                                         absl::Span<const BlobId> left,
                                         absl::FunctionRef<void()> publish) {
      std::vector<Index::ReferenceDelta> deltas;
      deltas.reserve(entered.size() + left.size());
      for (BlobId blob : entered) deltas.push_back({blob, 1});
      for (BlobId blob : left) deltas.push_back({blob, -1});
      // One state-lock acquisition per view batch. Underflow means our
      // per-view multiplicity and index counts diverged: fail loudly.
      CHECK_OK(index_->AdjustReferences(deltas, publish));
    });
  }
}

}  // namespace ferret

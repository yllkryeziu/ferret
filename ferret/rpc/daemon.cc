// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// One socket acceptor per repository, protected by a nonblocking flock.
#include "ferret/rpc/daemon.h"

#include <errno.h>
#include <fcntl.h>
#if defined(__APPLE__)
#include <mach-o/loader.h>
#elif defined(__linux__)
#include <elf.h>
#endif
#include <poll.h>
#include <sys/file.h>  // NOLINT(misc-include-cleaner)
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "nlohmann/json.hpp"
#include "openssl/sha.h"

#include "ferret/base/env.h"
#include "ferret/engine/engine.h"
#include "ferret/rpc/framing.h"
#include "ferret/rpc/json_codec.h"
#include "ferret/rpc/json_rpc.h"
#include "ferret/rpc/search_codec.h"
#include "ferret/rpc/symbol_codec.h"
#include "ferret/search/search_types.h"
#include "ferret/workspace/workspace.h"

namespace ferret {
namespace {

constexpr int kProtocolVersion = 1;
constexpr size_t kMaxConnections = 64;

struct Registrations {
  absl::Mutex mu;
  struct Root {
    absl::Mutex mu;
    bool registered ABSL_GUARDED_BY(mu) = false;
  };
  absl::flat_hash_map<std::string, std::shared_ptr<Root>> roots;
};

absl::Status CheckRoot(absl::string_view root, absl::string_view key) {
  ABSL_ASSIGN_OR_RETURN(std::string candidate, RepositoryKeyForRoot(root));
  if (candidate != key) {
    return absl::InvalidArgumentError("root belongs to another repository");
  }
  return absl::OkStatus();
}

// Only a completed registration may be used by search or notify. The key in
// an unregistered request distinguishes a foreign repository without running
// git; it never authorizes a request for a root absent from the map.
absl::Status CheckRegisteredRoot(Registrations& registrations,
                                 absl::string_view root, const JsonView& params,
                                 absl::string_view key) {
  std::shared_ptr<Registrations::Root> entry;
  {
    absl::MutexLock lock(registrations.mu);
    auto it = registrations.roots.find(root);
    if (it != registrations.roots.end()) entry = it->second;
  }
  if (entry != nullptr) {
    absl::MutexLock lock(entry->mu);
    if (entry->registered) return absl::OkStatus();
  }
  ABSL_ASSIGN_OR_RETURN(std::optional<std::string> candidate,
                        params.GetOptionalString("key"));
  if (candidate.has_value() && *candidate != key) {
    return absl::InvalidArgumentError("root belongs to another repository");
  }
  return absl::NotFoundError("root is not registered");
}

void Wake(int fd) {
  const char byte = 0;
  (void)write(fd, &byte, 1);
}

absl::Status SystemError(absl::string_view call) {
  return absl::ErrnoToStatus(errno, std::string(call));
}

// Owns descriptors on every early-return path.
struct OwnedFd {
  int fd = -1;
  explicit OwnedFd(int value = -1) : fd(value) {}
  OwnedFd(const OwnedFd&) = delete;
  OwnedFd& operator=(const OwnedFd&) = delete;
  ~OwnedFd() {
    if (fd >= 0) close(fd);
  }
};

std::string IndexPath(absl::string_view index_dir, absl::string_view key) {
  if (!index_dir.empty()) return absl::StrCat(index_dir, "/", key);
  const char* home = getenv("HOME");
  const char* xdg = getenv("XDG_CACHE_HOME");
#if defined(__APPLE__)
  constexpr bool kMac = true;
#else
  constexpr bool kMac = false;
#endif
  return absl::StrCat(CacheDirectory(home == nullptr ? "" : home,
                                     xdg == nullptr ? "" : xdg, kMac),
                      "/ferret/", key);
}

nlohmann::json EncodeStatuses(
    const std::vector<WorktreeIndexStatus>& statuses) {
  nlohmann::json result = nlohmann::json::array();
  for (const WorktreeIndexStatus& status : statuses) {
    result.push_back(EncodeWorktreeIndexStatus(status));
  }
  return result;
}

absl::StatusOr<nlohmann::json> Dispatch(
    Engine& engine, const JsonRpcRequest& request, absl::string_view build_id,
    bool& greeted, std::atomic<bool>& drain, Registrations& registrations,
    absl::string_view key, bool measure_latency, int wake_fd,
    absl::AnyInvocable<void(absl::string_view)>* registration_hook) {
  const JsonView params(request.params);
  if (request.method == "hello") {
    if (greeted) return absl::FailedPreconditionError("hello already sent");
    ABSL_ASSIGN_OR_RETURN(int64_t protocol,
                          params.GetInt64("protocol_version"));
    ABSL_ASSIGN_OR_RETURN(std::string client_build,
                          params.GetString("build_id"));
    greeted = true;
    return nlohmann::json{{"protocol_version", kProtocolVersion},
                          {"build_id", std::string(build_id)},
                          {"compatible", protocol == kProtocolVersion &&
                                             client_build == build_id}};
  }
  if (!greeted) return absl::FailedPreconditionError("hello must be first");
  if (request.method == "drain") {
    drain.store(true);
    Wake(wake_fd);
    return nlohmann::json::object();
  }
  if (request.method == "register") {
    ABSL_ASSIGN_OR_RETURN(std::string root, params.GetString("root"));
    char* resolved =
        realpath(root.c_str(), nullptr);  // NOLINT(misc-include-cleaner)
    if (resolved == nullptr) return SystemError("realpath root");
    std::string canonical(resolved);
    free(resolved);
    std::shared_ptr<Registrations::Root> entry;
    {
      absl::MutexLock lock(registrations.mu);
      entry = registrations.roots[canonical];
      if (!entry) {
        entry = registrations.roots[canonical] =
            std::make_shared<Registrations::Root>();
      }
    }
    absl::MutexLock lock(entry->mu);
    if (entry->registered) return nlohmann::json::object();
    ABSL_RETURN_IF_ERROR(CheckRoot(canonical, key));
    if (registration_hook != nullptr) (*registration_hook)(canonical);
    absl::Status status = engine.Register(canonical);
    if (!status.ok() && !absl::IsAlreadyExists(status)) return status;
    entry->registered = true;
    return nlohmann::json::object();
  }
  if (request.method == "search") {
    ABSL_ASSIGN_OR_RETURN(std::string root, params.GetString("root"));
    ABSL_RETURN_IF_ERROR(CheckRegisteredRoot(registrations, root, params, key));
    ABSL_ASSIGN_OR_RETURN(JsonView wire_request, params.GetObject("request"));
    ABSL_ASSIGN_OR_RETURN(SearchRequest search,
                          DecodeSearchRequest(wire_request));
    ABSL_ASSIGN_OR_RETURN(JsonView wire_consistency,
                          params.GetObject("consistency"));
    ABSL_ASSIGN_OR_RETURN(SearchConsistency consistency,
                          DecodeSearchConsistency(wire_consistency));
    const absl::Time started = absl::Now();
    absl::StatusOr<WorkspaceSearchResult> result =
        engine.Search(root, search, consistency);
    if (measure_latency) {
      std::fprintf(stderr, "daemon search elapsed_us=%lld\n",
                   static_cast<long long>(
                       absl::ToInt64Microseconds(absl::Now() - started)));
    }
    if (!result.ok()) return result.status();
    return EncodeWorkspaceSearchResult(*result);
  }
  if (request.method == "find_symbol") {
    ABSL_ASSIGN_OR_RETURN(std::string root, params.GetString("root"));
    ABSL_RETURN_IF_ERROR(CheckRegisteredRoot(registrations, root, params, key));
    ABSL_ASSIGN_OR_RETURN(JsonView wire_request, params.GetObject("request"));
    ABSL_ASSIGN_OR_RETURN(FindSymbolRequest lookup,
                          DecodeFindSymbolRequest(wire_request));
    ABSL_ASSIGN_OR_RETURN(JsonView wire_consistency,
                          params.GetObject("consistency"));
    ABSL_ASSIGN_OR_RETURN(SearchConsistency consistency,
                          DecodeSearchConsistency(wire_consistency));
    ABSL_ASSIGN_OR_RETURN(FindSymbolResponse result,
                          engine.FindSymbol(root, lookup, consistency));
    return EncodeFindSymbolResponse(result);
  }
  if (request.method == "index_status") {
    ABSL_ASSIGN_OR_RETURN(std::vector<WorktreeIndexStatus> statuses,
                          engine.IndexStatus());
    return EncodeStatuses(statuses);
  }
  if (request.method == "notify") {
    ABSL_ASSIGN_OR_RETURN(std::string root, params.GetString("root"));
    ABSL_RETURN_IF_ERROR(CheckRegisteredRoot(registrations, root, params, key));
    ABSL_ASSIGN_OR_RETURN(JsonView paths, params.GetArray("paths"));
    std::vector<std::string> names;
    for (size_t i = 0; i < paths.size(); ++i) {
      ABSL_ASSIGN_OR_RETURN(std::string name, paths.Element(i).AsString());
      names.push_back(std::move(name));
    }
    ABSL_ASSIGN_OR_RETURN(uint64_t generation, engine.Notify(root, names));
    return nlohmann::json{{"generation", generation}};
  }
  return absl::UnimplementedError("unknown method");
}

void ServeConnection(
    int fd, Engine& engine, absl::string_view build_id,
    std::atomic<bool>& drain, Registrations& registrations,
    absl::string_view key, bool measure_latency, int wake_fd,
    absl::AnyInvocable<void(absl::string_view)>* registration_hook) {
  if (!CheckDaemonPeer(fd).ok()) return;
  RpcFrameReader reader(fd);
  RpcFrameWriter writer(fd);
  bool greeted = false;
  while (!drain.load()) {
    absl::StatusOr<std::string> frame = reader.Read();
    if (!frame.ok()) {
      if (!absl::IsOutOfRange(frame.status())) {
        (void)writer.Write(SerializeJsonRpcMessage(JsonRpcResponse{
            .error = JsonRpcError{.code = -32700, .message = "parse error"}}));
      }
      return;
    }
    absl::StatusOr<nlohmann::json> document = ParseJson(*frame);
    if (!document.ok()) {
      (void)writer.Write(SerializeJsonRpcMessage(JsonRpcResponse{
          .error = JsonRpcError{.code = -32700, .message = "parse error"}}));
      return;
    }
    absl::StatusOr<JsonRpcMessage> message = DecodeJsonRpcMessage(*document);
    if (!message.ok() || !std::holds_alternative<JsonRpcRequest>(*message)) {
      (void)writer.Write(SerializeJsonRpcMessage(
          JsonRpcResponse{.error = JsonRpcError{
                              .code = -32600, .message = "invalid request"}}));
      return;
    }
    const JsonRpcRequest& request = std::get<JsonRpcRequest>(*message);
    absl::StatusOr<nlohmann::json> result =
        Dispatch(engine, request, build_id, greeted, drain, registrations, key,
                 measure_latency, wake_fd, registration_hook);
    JsonRpcResponse response{.id = request.id};
    if (result.ok()) {
      response.result = std::move(*result);
    } else {
      response.error =
          JsonRpcError{.code = -32000,
                       .message = std::string(result.status().message()),
                       .data = static_cast<int>(result.status().code())};
    }
    if (!writer.Write(SerializeJsonRpcMessage(response)).ok()) return;
  }
}

}  // namespace

absl::Status CheckSocketDirectory(absl::string_view path) {
  std::string name(path);
  if (mkdir(name.c_str(), 0700) != 0 && errno != EEXIST) {
    return SystemError("mkdir socket directory");
  }
  struct stat st;
  if (lstat(name.c_str(), &st) != 0) {
    return SystemError("lstat socket directory");
  }
  if (!S_ISDIR(st.st_mode) || st.st_uid != geteuid() ||
      (st.st_mode & 0777) != 0700) {
    return absl::PermissionDeniedError(
        "socket directory must be owned by uid and mode 0700");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> DaemonSocketPath(absl::string_view root) {
  ABSL_ASSIGN_OR_RETURN(std::string key, RepositoryKeyForRoot(root));
  return DaemonSocketPathForKey(key);
}

absl::StatusOr<std::string> DaemonSocketPathForKey(absl::string_view key) {
  const char* tmp = getenv("TMPDIR");
  const std::string dir =
      absl::StrCat(tmp == nullptr ? "/tmp" : tmp, "/ferret-", geteuid());
  ABSL_RETURN_IF_ERROR(CheckSocketDirectory(dir));
  return absl::StrCat(dir, "/ferret-", key, ".sock");
}

absl::Status CheckDaemonPeer(int fd) {
#if defined(__APPLE__) || defined(__FreeBSD__)
  uid_t uid;
  gid_t gid;
  if (getpeereid(fd, &uid, &gid) != 0) return SystemError("getpeereid");
#else
  struct ucred peer;
  socklen_t size = sizeof(peer);
  if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &peer, &size) != 0) {
    return SystemError("SO_PEERCRED");
  }
  const uid_t uid = peer.uid;
#endif
  if (uid != geteuid()) return absl::PermissionDeniedError("wrong peer uid");
  return absl::OkStatus();
}

// Apple SDK declarations are not recognized by include-cleaner.
// NOLINTBEGIN(misc-include-cleaner)
// A linker-issued identity avoids reading the entire executable on each client
// invocation. Do not use file timestamps: replacing a binary can preserve them.
#if defined(__APPLE__)
std::optional<std::string> LinkedBuildId(int fd) {
  mach_header_64 header = {};
  if (pread(fd, &header, sizeof(header), 0) != sizeof(header) ||
      header.magic != MH_MAGIC_64) {
    return std::nullopt;
  }
  uint64_t offset = sizeof(header);
  const uint64_t end = offset + header.sizeofcmds;
  for (uint32_t i = 0; i < header.ncmds; ++i) {
    load_command command = {};
    if (offset + sizeof(command) > end ||
        pread(fd, &command, sizeof(command), static_cast<off_t>(offset)) !=
            sizeof(command) ||
        command.cmdsize < sizeof(command) || command.cmdsize > end - offset) {
      return std::nullopt;
    }
    if (command.cmd == LC_UUID && command.cmdsize >= sizeof(uuid_command)) {
      uuid_command uuid = {};
      if (pread(fd, &uuid, sizeof(uuid), static_cast<off_t>(offset)) !=
          sizeof(uuid)) {
        return std::nullopt;
      }
      return absl::StrCat(
          "macho:",
          absl::BytesToHexString(absl::string_view(
              reinterpret_cast<const char*>(uuid.uuid), sizeof(uuid.uuid))));
    }
    offset += command.cmdsize;
  }
  return std::nullopt;
}
#elif defined(__linux__)
std::optional<std::string> LinkedBuildId(int fd) {
  Elf64_Ehdr header = {};
  if (pread(fd, &header, sizeof(header), 0) != sizeof(header) ||
      memcmp(header.e_ident, ELFMAG, SELFMAG) != 0 ||
      header.e_ident[EI_CLASS] != ELFCLASS64 ||
      header.e_ident[EI_DATA] != ELFDATA2LSB ||
      header.e_phentsize != sizeof(Elf64_Phdr)) {
    return std::nullopt;
  }
  for (uint16_t i = 0; i < header.e_phnum; ++i) {
    Elf64_Phdr program = {};
    const uint64_t position = header.e_phoff + uint64_t{i} * sizeof(program);
    if (position > INT64_MAX ||
        pread(fd, &program, sizeof(program), static_cast<off_t>(position)) !=
            sizeof(program)) {
      return std::nullopt;
    }
    if (program.p_type != PT_NOTE || program.p_filesz > 65536 ||
        program.p_offset > INT64_MAX)
      continue;
    std::vector<unsigned char> notes(program.p_filesz);
    if (pread(fd, notes.data(), notes.size(),
              static_cast<off_t>(program.p_offset)) != notes.size()) {
      return std::nullopt;
    }
    size_t offset = 0;
    while (notes.size() - offset >= sizeof(Elf64_Nhdr)) {
      Elf64_Nhdr note = {};
      memcpy(&note, notes.data() + offset, sizeof(note));
      offset += sizeof(note);
      const uint64_t name_size = (uint64_t{note.n_namesz} + 3) & ~uint64_t{3};
      const uint64_t desc_size = (uint64_t{note.n_descsz} + 3) & ~uint64_t{3};
      if (name_size + desc_size > notes.size() - offset) break;
      if (note.n_type == NT_GNU_BUILD_ID && note.n_namesz == 4 &&
          memcmp(notes.data() + offset, "GNU", 4) == 0) {
        return absl::StrCat("elf:", absl::BytesToHexString(absl::string_view(
                                        reinterpret_cast<const char*>(
                                            notes.data() + offset + name_size),
                                        note.n_descsz)));
      }
      offset += name_size + desc_size;
    }
  }
  return std::nullopt;
}
#else
std::optional<std::string> LinkedBuildId(int fd) { return std::nullopt; }
#endif

absl::StatusOr<std::string> DaemonBuildId(absl::string_view executable) {
  OwnedFd fd(open(std::string(executable).c_str(),
                  O_RDONLY | O_CLOEXEC));  // NOLINT(misc-include-cleaner)
  if (fd.fd < 0) return SystemError("open executable");
  if (std::optional<std::string> id = LinkedBuildId(fd.fd); id.has_value()) {
    return *id;
  }
  SHA256_CTX context;
  SHA256_Init(&context);
  std::array<unsigned char, 65536> buffer;
  for (;;) {
    const ssize_t count = read(fd.fd, buffer.data(), buffer.size());
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) return SystemError("read executable");
    if (count == 0) break;
    SHA256_Update(&context, buffer.data(), static_cast<size_t>(count));
  }
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest;
  SHA256_Final(digest.data(), &context);
  return absl::BytesToHexString(absl::string_view(
      reinterpret_cast<const char*>(digest.data()), digest.size()));
}

absl::Status RunDaemon(
    absl::string_view root, absl::string_view index_dir,
    absl::string_view build_id, absl::Duration idle_timeout,
    bool measure_latency,
    absl::AnyInvocable<void(absl::string_view)>* registration_hook) {
  ABSL_ASSIGN_OR_RETURN(std::string key, RepositoryKeyForRoot(root));
  const std::string index = IndexPath(index_dir, key);
  ABSL_RETURN_IF_ERROR(Env::Default()->CreateDir(index));
  OwnedFd lock(open(absl::StrCat(index, "/DAEMON_LOCK").c_str(),
                    O_RDWR | O_CREAT | O_CLOEXEC,
                    0600));  // NOLINT(misc-include-cleaner)
  if (lock.fd < 0) return SystemError("open daemon lock");
  if (flock(lock.fd, LOCK_EX | LOCK_NB) != 0) {  // NOLINT(misc-include-cleaner)
    return absl::AlreadyExistsError("daemon already running");
  }
  OwnedFd log(open(absl::StrCat(index, "/daemon.log").c_str(),
                   O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC,
                   0600));  // NOLINT(misc-include-cleaner)
  if (log.fd < 0) {
    return SystemError("open daemon log");
  }
  if (dup2(log.fd, STDERR_FILENO) < 0) {
    return SystemError("redirect daemon log");
  }
  ABSL_ASSIGN_OR_RETURN(std::string path, DaemonSocketPathForKey(key));
  if (path.size() >= sizeof(sockaddr_un::sun_path)) {
    return absl::InvalidArgumentError("socket path too long");
  }
  if (unlink(path.c_str()) != 0 && errno != ENOENT) {
    return SystemError("unlink stale socket");
  }
  OwnedFd listener(
      socket(AF_UNIX, SOCK_STREAM, 0));  // NOLINT(misc-include-cleaner)
  if (listener.fd < 0) return SystemError("socket");
  if (fcntl(listener.fd, F_SETFD, FD_CLOEXEC) != 0) return SystemError("fcntl");
  sockaddr_un address = {};
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
  const mode_t old_mask = umask(077);
  const int bound =
      bind(listener.fd, reinterpret_cast<sockaddr*>(&address), sizeof(address));
  umask(old_mask);
  if (bound != 0) {
    return SystemError("bind");
  }
  if (listen(listener.fd, 64) != 0) return SystemError("listen");
  // The wake pipe interrupts poll on drain and on a connection finishing.
  int pipe_fds[2];
  if (pipe(pipe_fds) != 0) return SystemError("pipe");
  OwnedFd wake_read(pipe_fds[0]);
  OwnedFd wake_write(pipe_fds[1]);
  for (int fd : pipe_fds) {
    if (fcntl(fd, F_SETFD, FD_CLOEXEC) != 0 ||
        fcntl(fd, F_SETFL, O_NONBLOCK) != 0) {
      return SystemError("fcntl pipe");
    }
  }
  Engine engine(*Env::Default(), std::string(index_dir));
  ABSL_RETURN_IF_ERROR(engine.OpenRepository(root));
  std::atomic<bool> drain(false);
  Registrations registrations;
  struct ConnectionThread {
    std::thread thread;
    int fd;
    std::shared_ptr<std::atomic<bool>> done;
  };
  std::vector<ConnectionThread> threads;
  absl::Time last_active = absl::Now();
  std::fprintf(stderr, "daemon start pid=%ld\n", static_cast<long>(getpid()));
  for (;;) {
    for (auto it = threads.begin(); it != threads.end();) {
      if (it->done->load()) {
        it->thread.join();
        close(it->fd);
        it = threads.erase(it);
      } else {
        ++it;
      }
    }
    if (drain.load()) {
      std::fprintf(stderr, "daemon drain\n");
      break;
    }
    if (!threads.empty()) last_active = absl::Now();
    const absl::Duration remaining = idle_timeout - (absl::Now() - last_active);
    if (threads.empty() && remaining <= absl::ZeroDuration()) {
      std::fprintf(stderr, "daemon idle exit\n");
      break;
    }
    const int timeout = threads.empty()
                            ? static_cast<int>(std::min<int64_t>(
                                  absl::ToInt64Milliseconds(remaining) + 1,
                                  std::numeric_limits<int>::max()))
                            : -1;
    pollfd events[2] = {{.fd = listener.fd, .events = POLLIN, .revents = 0},
                        {.fd = wake_read.fd, .events = POLLIN, .revents = 0}};
    const int ready = poll(events, 2, timeout);
    if (ready < 0 && errno == EINTR) continue;
    if (ready < 0) {
      std::fprintf(stderr, "daemon poll error: %d\n", errno);
      break;
    }
    if (events[1].revents & POLLIN) {
      char buffer[128];
      while (read(wake_read.fd, buffer, sizeof(buffer)) > 0) {
      }
    }
    if (drain.load()) continue;
    if (events[0].revents & POLLIN) {
      const int fd = accept(listener.fd, nullptr, nullptr);
      if (fd < 0) continue;
      if (fcntl(fd, F_SETFD, FD_CLOEXEC) != 0) {
        close(fd);
        continue;
      }
      if (threads.size() >= kMaxConnections) {
        RpcFrameWriter writer(fd);
        (void)writer.Write(SerializeJsonRpcMessage(JsonRpcResponse{
            .error = JsonRpcError{.code = -32000,
                                  .message = "connection limit reached"}}));
        close(fd);
        continue;
      }
      auto done = std::make_shared<std::atomic<bool>>(false);
      threads.push_back({std::thread([&, fd, done] {
                           ServeConnection(fd, engine, build_id, drain,
                                           registrations, key, measure_latency,
                                           wake_write.fd, registration_hook);
                           done->store(true);
                           Wake(wake_write.fd);
                         }),
                         fd, done});
    }
  }
  // Shutdown unblocks idle readers; active searches complete before joining.
  for (auto& connection : threads) shutdown(connection.fd, SHUT_RD);
  for (auto& connection : threads) {
    connection.thread.join();
    close(connection.fd);
  }
  unlink(path.c_str());
  return absl::OkStatus();
}
// NOLINTEND(misc-include-cleaner)

}  // namespace ferret

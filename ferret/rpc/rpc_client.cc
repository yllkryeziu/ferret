// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Reconnecting synchronous RPC client. A connection lock covers write and
// response read, so replies cannot cross between concurrent callers.
#include "ferret/rpc/rpc_client.h"

#include <errno.h>
#include <fcntl.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "nlohmann/json.hpp"

#include "ferret/engine/engine.h"
#include "ferret/rpc/daemon.h"
#include "ferret/rpc/framing.h"
#include "ferret/rpc/json_codec.h"
#include "ferret/rpc/json_rpc.h"
#include "ferret/rpc/search_codec.h"
#include "ferret/rpc/symbol_codec.h"
#include "ferret/search/search_types.h"
#include "ferret/workspace/workspace.h"

extern char** environ;

namespace ferret {
namespace {

// Apple SDK declarations are not recognized by include-cleaner.
// NOLINTBEGIN(misc-include-cleaner)
absl::StatusOr<std::string> CanonicalRoot(absl::string_view root) {
  char* path = realpath(std::string(root).c_str(),
                        nullptr);  // NOLINT(misc-include-cleaner)
  if (path == nullptr) return absl::ErrnoToStatus(errno, "realpath root");
  std::string canonical(path);
  free(path);
  return canonical;
}

absl::StatusOr<int> Connect(absl::string_view path) {
  const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return absl::ErrnoToStatus(errno, "socket");
  if (fcntl(fd, F_SETFD, FD_CLOEXEC) != 0) {  // NOLINT(misc-include-cleaner)
    absl::Status error = absl::ErrnoToStatus(errno, "fcntl");
    close(fd);
    return error;
  }
  sockaddr_un address = {};
  address.sun_family = AF_UNIX;
  if (path.size() >= sizeof(address.sun_path)) {
    close(fd);
    return absl::InvalidArgumentError("socket path too long");
  }
  std::memcpy(address.sun_path, path.data(), path.size());
  if (connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) !=
      0) {
    absl::Status error = errno == ENOTSOCK
                             ? absl::UnavailableError("stale daemon socket")
                             : absl::ErrnoToStatus(errno, "connect daemon");
    close(fd);
    return error;
  }
  absl::Status peer = CheckDaemonPeer(fd);
  if (!peer.ok()) {
    close(fd);
    return peer;
  }
  return fd;
}

absl::StatusOr<pid_t> Spawn(absl::string_view executable,
                            absl::string_view root,
                            absl::string_view index_dir) {
  posix_spawn_file_actions_t actions;
  int error = posix_spawn_file_actions_init(&actions);
  if (error != 0) return absl::ErrnoToStatus(error, "spawn actions");
  for (int fd = 0; fd < 3 && error == 0; ++fd) {
    error = posix_spawn_file_actions_addopen(
        &actions, fd, "/dev/null", fd == 0 ? O_RDONLY : O_WRONLY,
        0);  // NOLINT(misc-include-cleaner)
  }
  posix_spawnattr_t attr;
  if (error == 0) error = posix_spawnattr_init(&attr);
  if (error != 0) {
    posix_spawn_file_actions_destroy(&actions);
    return absl::ErrnoToStatus(error, "spawn init");
  }
  int16_t flags = POSIX_SPAWN_SETSID;  // NOLINT(misc-include-cleaner)
#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
  flags |= POSIX_SPAWN_CLOEXEC_DEFAULT;
#endif
  error = posix_spawnattr_setflags(&attr, flags);
  std::string binary(executable);
  std::string canonical(root);
  const std::string index_flag = absl::StrCat("--index_dir=", index_dir);
  char* argv[] = {binary.data(), const_cast<char*>("serve"), canonical.data(),
                  const_cast<char*>(index_flag.c_str()), nullptr};
  pid_t pid;
  if (error == 0) {
    error = posix_spawn(&pid, binary.c_str(), &actions, &attr, argv, environ);
  }
  posix_spawnattr_destroy(&attr);
  posix_spawn_file_actions_destroy(&actions);
  if (error != 0) return absl::ErrnoToStatus(error, "posix_spawn serve");
  return pid;
}
// NOLINTEND(misc-include-cleaner)

absl::StatusOr<nlohmann::json> Exchange(RpcFrameReader& reader,
                                        RpcFrameWriter& writer,
                                        absl::string_view method,
                                        const nlohmann::json& params) {
  const JsonRpcRequest request{
      .id = int64_t{1}, .method = std::string(method), .params = params};
  ABSL_RETURN_IF_ERROR(writer.Write(SerializeJsonRpcMessage(request)));
  ABSL_ASSIGN_OR_RETURN(std::string frame, reader.Read());
  ABSL_ASSIGN_OR_RETURN(nlohmann::json document, ParseJson(frame));
  ABSL_ASSIGN_OR_RETURN(JsonRpcMessage message, DecodeJsonRpcMessage(document));
  if (!std::holds_alternative<JsonRpcResponse>(message)) {
    return absl::DataLossError("daemon did not send a response");
  }
  const JsonRpcResponse& response = std::get<JsonRpcResponse>(message);
  if (!response.id || *response.id != request.id) {
    return absl::DataLossError("daemon answered the wrong request");
  }
  if (response.error) {
    absl::StatusCode code = absl::StatusCode::kUnknown;
    if (response.error->data.is_number_integer()) {
      const int value = response.error->data.get<int>();
      if (value > 0 && value <= 16) code = static_cast<absl::StatusCode>(value);
    }
    return absl::Status(code, response.error->message);
  }
  return response.result;
}

}  // namespace

struct RpcClient::Connection {
  std::string root;
  std::string socket;
  std::vector<std::string> roots;
  mutable absl::Mutex mu;
  int fd ABSL_GUARDED_BY(mu) = -1;
  std::unique_ptr<RpcFrameReader> reader ABSL_GUARDED_BY(mu);
  std::unique_ptr<RpcFrameWriter> writer ABSL_GUARDED_BY(mu);
  void Close() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) {
    reader.reset();
    writer.reset();
    if (fd >= 0) close(fd);
    fd = -1;
  }
  ~Connection() {
    if (fd >= 0) close(fd);
  }
};

RpcClient::RpcClient(std::string executable, std::string index_dir,
                     RpcClientOptions options)
    : options_(options),
      executable_(std::move(executable)),
      index_dir_(std::move(index_dir)),
      build_id_(DaemonBuildId(executable_)) {}
RpcClient::~RpcClient() {
  absl::MutexLock lock(children_mu_);
  for (pid_t pid : children_) (void)waitpid(pid, nullptr, WNOHANG);
}

void RpcClient::ReapChildren() const {
  absl::MutexLock lock(children_mu_);
  for (auto it = children_.begin(); it != children_.end();) {
    if (waitpid(*it, nullptr, WNOHANG) != 0) {
      it = children_.erase(it);
    } else {
      ++it;
    }
  }
}

absl::Status RpcClient::SpawnDaemon(const Connection& connection) const {
  ABSL_ASSIGN_OR_RETURN(pid_t pid,
                        Spawn(executable_, connection.root, index_dir_));
  absl::MutexLock lock(children_mu_);
  children_.push_back(pid);
  return absl::OkStatus();
}

absl::StatusOr<nlohmann::json> RpcClient::Call(
    Connection& connection, absl::string_view method,
    const nlohmann::json& params) const {
  if (options_.spawn_daemon) ReapChildren();
  absl::MutexLock lock(connection.mu);
  if (!build_id_.ok()) return build_id_.status();
  if (connection.fd < 0) {
    if (!options_.spawn_daemon) {
      ABSL_ASSIGN_OR_RETURN(int fd, Connect(connection.socket));
      auto reader = std::make_unique<RpcFrameReader>(fd);
      auto writer = std::make_unique<RpcFrameWriter>(fd);
      absl::StatusOr<nlohmann::json> hello =
          Exchange(*reader, *writer, "hello",
                   {{"protocol_version", 1}, {"build_id", *build_id_}});
      if (!hello.ok()) {
        close(fd);
        return hello.status();
      }
      absl::StatusOr<bool> compatible = JsonView(*hello).GetBool("compatible");
      if (!compatible.ok() || !*compatible) {
        close(fd);
        return absl::UnavailableError("daemon build is incompatible");
      }
      connection.fd = fd;
      connection.reader = std::move(reader);
      connection.writer = std::move(writer);
      for (const std::string& root : connection.roots) {
        absl::Status status = Exchange(*connection.reader, *connection.writer,
                                       "register", {{"root", root}})
                                  .status();
        if (!status.ok()) {
          connection.Close();
          return status;
        }
      }
    } else {
      // At most 20 seconds for startup, including a five-second drain.
      // Retry at 10, 20, 40, ... 200 ms. Spawn once, then wait: a daemon runs
      // git before it takes its lock, so it can take seconds to listen on a
      // loaded machine, and spawning each round started hundreds of daemons.
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(20);
      int delay_us = 10000;
      bool spawned = false;
      while (std::chrono::steady_clock::now() < deadline) {
        absl::StatusOr<int> fd = Connect(connection.socket);
        if (!fd.ok()) {
          if (!absl::IsNotFound(fd.status()) &&
              !absl::IsUnavailable(fd.status())) {
            return fd.status();
          }
          if (!spawned) {
            ABSL_RETURN_IF_ERROR(SpawnDaemon(connection));
            spawned = true;
          }
        } else {
          auto reader = std::make_unique<RpcFrameReader>(*fd);
          auto writer = std::make_unique<RpcFrameWriter>(*fd);
          absl::StatusOr<nlohmann::json> hello =
              Exchange(*reader, *writer, "hello",
                       {{"protocol_version", 1}, {"build_id", *build_id_}});
          if (hello.ok()) {
            absl::StatusOr<bool> compatible =
                JsonView(*hello).GetBool("compatible");
            if (!compatible.ok()) {
              close(*fd);
              return compatible.status();
            }
            if (!*compatible) {
              (void)Exchange(*reader, *writer, "drain",
                             nlohmann::json::object());
              close(*fd);
              const auto drain_deadline =
                  std::min(deadline, std::chrono::steady_clock::now() +
                                         std::chrono::seconds(5));
              // Do not connect to a still-draining daemon and drain it again.
              while (std::chrono::steady_clock::now() < drain_deadline) {
                absl::StatusOr<int> probe = Connect(connection.socket);
                if (!probe.ok()) break;
                close(*probe);
                usleep(20000);
              }
              if (std::chrono::steady_clock::now() >= drain_deadline) {
                return absl::UnavailableError(
                    "daemon did not drain in 5 seconds");
              }
              ABSL_RETURN_IF_ERROR(SpawnDaemon(connection));
              spawned = true;
            } else {
              connection.fd = *fd;
              connection.reader = std::move(reader);
              connection.writer = std::move(writer);
              // Re-register roots after replacing a broken connection.
              bool failed = false;
              for (const std::string& root : connection.roots) {
                if (!Exchange(*connection.reader, *connection.writer,
                              "register", {{"root", root}})
                         .ok()) {
                  failed = true;
                  break;
                }
              }
              if (!failed) break;
              connection.Close();
            }
          } else {
            close(*fd);
          }
        }
        usleep(delay_us);
        delay_us = std::min(delay_us * 2, 200000);
      }
      if (connection.fd < 0) {
        return absl::UnavailableError("daemon did not start in 20 seconds");
      }
    }
  }
  absl::StatusOr<nlohmann::json> result =
      Exchange(*connection.reader, *connection.writer, method, params);
  if (!result.ok() && (absl::IsOutOfRange(result.status()) ||
                       absl::IsDataLoss(result.status()) ||
                       absl::IsUnavailable(result.status()) ||
                       absl::IsFailedPrecondition(result.status()))) {
    connection.Close();
  }
  return result;
}

absl::Status RpcClient::Register(absl::string_view root) {
  ABSL_ASSIGN_OR_RETURN(std::string canonical, CanonicalRoot(root));
  {
    absl::MutexLock lock(mu_);
    if (registered_.contains(canonical)) return absl::OkStatus();
  }
  ABSL_ASSIGN_OR_RETURN(std::string key, RepositoryKeyForRoot(canonical));
  ABSL_ASSIGN_OR_RETURN(std::string socket, DaemonSocketPathForKey(key));
  std::shared_ptr<Connection> connection;
  {
    absl::MutexLock lock(mu_);
    if (registered_.contains(canonical)) return absl::OkStatus();
    auto it = connections_.find(key);
    if (it == connections_.end()) {
      connection = std::make_shared<Connection>();
      connection->root = canonical;
      connection->socket = std::move(socket);
      connections_.emplace(key, connection);
    } else {
      connection = it->second;
    }
  }
  bool available = options_.spawn_daemon;
  if (!available) {
    absl::StatusOr<int> probe = Connect(connection->socket);
    if (probe.ok()) {
      close(*probe);
      available = true;
    }
  }
  if (available) {
    ABSL_RETURN_IF_ERROR(
        Call(*connection, "register", {{"root", canonical}}).status());
  }
  {
    absl::MutexLock lock(mu_);
    // Two registrations may have raced through the idempotent server call.
    if (registered_.contains(canonical)) return absl::OkStatus();
    registered_.emplace(canonical, RegisteredRoot{canonical, key, connection});
    roots_.push_back(canonical);
    absl::MutexLock connection_lock(connection->mu);
    connection->roots.push_back(canonical);
  }
  return absl::OkStatus();
}

std::vector<std::string> RpcClient::Roots() const {
  absl::MutexLock lock(mu_);
  return roots_;
}

absl::StatusOr<RpcClient::RegisteredRoot> RpcClient::Find(
    absl::string_view root) const {
  {
    absl::MutexLock lock(mu_);
    auto it = registered_.find(root);
    if (it != registered_.end()) return it->second;
    auto alias = aliases_.find(root);
    if (alias != aliases_.end()) return registered_.at(alias->second);
  }
  // Preserve aliases for callers that did not pass the registered spelling.
  ABSL_ASSIGN_OR_RETURN(std::string canonical, CanonicalRoot(root));
  absl::MutexLock lock(mu_);
  auto it = registered_.find(canonical);
  if (it != registered_.end()) {
    aliases_.emplace(root, canonical);
    return it->second;
  }
  return absl::NotFoundError("root is not registered");
}

absl::StatusOr<WorkspaceSearchResult> RpcClient::Search(
    absl::string_view root, const SearchRequest& request,
    const SearchConsistency& consistency) const {
  ABSL_ASSIGN_OR_RETURN(RegisteredRoot registration, Find(root));
  ABSL_ASSIGN_OR_RETURN(
      nlohmann::json result,
      Call(*registration.connection, "search",
           {{"root", registration.root},
            {"key", registration.key},
            {"request", EncodeSearchRequest(request)},
            {"consistency", EncodeSearchConsistency(consistency)}}));
  return DecodeWorkspaceSearchResult(JsonView(result));
}

absl::StatusOr<FindSymbolResponse> RpcClient::FindSymbol(
    absl::string_view root, const FindSymbolRequest& request,
    const SearchConsistency& consistency) const {
  ABSL_ASSIGN_OR_RETURN(RegisteredRoot registration, Find(root));
  ABSL_ASSIGN_OR_RETURN(
      nlohmann::json result,
      Call(*registration.connection, "find_symbol",
           {{"root", registration.root},
            {"key", registration.key},
            {"request", EncodeFindSymbolRequest(request)},
            {"consistency", EncodeSearchConsistency(consistency)}}));
  return DecodeFindSymbolResponse(JsonView(result));
}

absl::StatusOr<std::vector<WorktreeIndexStatus>> RpcClient::IndexStatus()
    const {
  std::vector<std::shared_ptr<Connection>> connections;
  {
    absl::MutexLock lock(mu_);
    for (const auto& [key, connection] : connections_) {
      connections.push_back(connection);
    }
  }
  std::vector<WorktreeIndexStatus> statuses;
  for (const auto& connection : connections) {
    ABSL_ASSIGN_OR_RETURN(
        nlohmann::json response,
        Call(*connection, "index_status", nlohmann::json::object()));
    JsonView array(response);
    ABSL_ASSIGN_OR_RETURN(JsonView values, array.AsArray());
    for (size_t i = 0; i < values.size(); ++i) {
      ABSL_ASSIGN_OR_RETURN(WorktreeIndexStatus status,
                            DecodeWorktreeIndexStatus(values.Element(i)));
      statuses.push_back(std::move(status));
    }
  }
  return statuses;
}

absl::StatusOr<uint64_t> RpcClient::Notify(
    absl::string_view root, const std::vector<std::string>& paths) const {
  ABSL_ASSIGN_OR_RETURN(RegisteredRoot registration, Find(root));
  ABSL_ASSIGN_OR_RETURN(nlohmann::json result,
                        Call(*registration.connection, "notify",
                             {{"root", registration.root},
                              {"key", registration.key},
                              {"paths", paths}}));
  return JsonView(result).GetUint64("generation");
}

absl::Status RpcClient::Drain() const {
  std::vector<std::shared_ptr<Connection>> connections;
  {
    absl::MutexLock lock(mu_);
    for (const auto& [key, connection] : connections_) {
      connections.push_back(connection);
    }
  }
  for (const auto& connection : connections) {
    ABSL_RETURN_IF_ERROR(
        Call(*connection, "drain", nlohmann::json::object()).status());
    absl::MutexLock lock(connection->mu);
    if (connection->fd >= 0) {
      connection->Close();
    }
  }
  return absl::OkStatus();
}

}  // namespace ferret

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// A SearchBackend over repository-scoped daemon sockets. RPC lives here,
// not in engine, so engine stays independent of JSON and transport details.
#ifndef FERRET_RPC_RPC_CLIENT_H_
#define FERRET_RPC_RPC_CLIENT_H_

#include <sys/types.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "nlohmann/json.hpp"

#include "ferret/engine/engine.h"

namespace ferret {

struct RpcClientOptions {
  bool spawn_daemon = true;
};

// Thread-safe. Calls on one connection serialize under its mutex; calls to
// different repositories run independently. Destroy only after calls return.
class RpcClient final : public SearchBackend {
 public:
  // `executable` is the path to the ferret binary to spawn, not argv[0] of
  // an unrelated test runner. It must remain available during this lifetime.
  explicit RpcClient(std::string executable, std::string index_dir = "",
                     RpcClientOptions options = {});
  ~RpcClient() override;
  RpcClient(const RpcClient&) = delete;
  RpcClient& operator=(const RpcClient&) = delete;

  absl::Status Register(absl::string_view root);
  std::vector<std::string> Roots() const override;
  absl::StatusOr<WorkspaceSearchResult> Search(
      absl::string_view root, const SearchRequest& request,
      const SearchConsistency& consistency) const override;
  bool SupportsFindSymbol() const override { return true; }
  absl::StatusOr<FindSymbolResponse> FindSymbol(
      absl::string_view root, const FindSymbolRequest& request,
      const SearchConsistency& consistency) const override;
  absl::StatusOr<std::vector<WorktreeIndexStatus>> IndexStatus() const override;
  absl::StatusOr<uint64_t> Notify(absl::string_view root,
                                  const std::vector<std::string>& paths) const;
  // Requests a clean exit from all owned daemons. Primarily for shutdown
  // tests; other clients may still be connected and will be disconnected.
  absl::Status Drain() const;

 private:
  struct Connection;
  struct RegisteredRoot {
    std::string root;
    std::string key;
    std::shared_ptr<Connection> connection;
  };
  absl::StatusOr<RegisteredRoot> Find(absl::string_view root) const;
  absl::StatusOr<nlohmann::json> Call(Connection& connection,
                                      absl::string_view method,
                                      const nlohmann::json& params) const;
  void ReapChildren() const;
  absl::Status SpawnDaemon(const Connection& connection) const;
  mutable absl::Mutex children_mu_;
  mutable std::vector<pid_t> children_ ABSL_GUARDED_BY(children_mu_);
  const RpcClientOptions options_;
  std::string executable_;
  std::string index_dir_;
  absl::StatusOr<std::string> build_id_;
  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::shared_ptr<Connection>> connections_
      ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, RegisteredRoot> registered_
      ABSL_GUARDED_BY(mu_);
  mutable absl::flat_hash_map<std::string, std::string> aliases_
      ABSL_GUARDED_BY(mu_);
  std::vector<std::string> roots_ ABSL_GUARDED_BY(mu_);
};

}  // namespace ferret
#endif  // FERRET_RPC_RPC_CLIENT_H_

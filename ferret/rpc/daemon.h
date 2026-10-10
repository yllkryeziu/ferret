// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Unix socket daemon transport. The lock, socket and index share a repository
// key, while the engine owns the index lock itself.
#ifndef FERRET_RPC_DAEMON_H_
#define FERRET_RPC_DAEMON_H_

#include <string>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"

namespace ferret {

// Creates and checks a 0700 directory owned by the current uid. Rejects
// symlinks, group/world permission bits, and an unexpected owner.
absl::Status CheckSocketDirectory(absl::string_view path);
// Returns the socket path for a repository root, creating its secure parent.
absl::StatusOr<std::string> DaemonSocketPath(absl::string_view root);
// Returns the socket path for a previously resolved repository key.
absl::StatusOr<std::string> DaemonSocketPathForKey(absl::string_view key);
// Rejects a connection unless its peer has the daemon's uid.
absl::Status CheckDaemonPeer(int fd);
// Uses the linked build ID when available, else hashes the executable bytes.
absl::StatusOr<std::string> DaemonBuildId(absl::string_view executable);
// Serves a repository until drain or the idle timeout. AlreadyExists means
// another daemon holds the lock, without touching its socket.
absl::Status RunDaemon(
    absl::string_view root, absl::string_view index_dir,
    absl::string_view build_id, absl::Duration idle_timeout,
    // The optional hook is used by tests to pause a cold registration.
    bool measure_latency = false,
    absl::AnyInvocable<void(absl::string_view)>* registration_hook = nullptr);

}  // namespace ferret
#endif  // FERRET_RPC_DAEMON_H_

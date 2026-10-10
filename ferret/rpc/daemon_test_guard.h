// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Test-only ownership of daemons spawned by an RPC client or subprocess.
#ifndef FERRET_RPC_DAEMON_TEST_GUARD_H_
#define FERRET_RPC_DAEMON_TEST_GUARD_H_

#include <sys/types.h>

#include <string>
#include <vector>

#include "absl/strings/string_view.h"

namespace ferret {

// A fixture calls Watch before starting a daemon and Stop in TearDown. Stop
// drains even if the test body returned early, kills unresponsive processes,
// and reports a daemon or socket that outlived the test.
class DaemonTestGuard {
 public:
  void Watch(absl::string_view root);
  void WatchPid(pid_t pid);
  void Stop(absl::string_view executable, absl::string_view index_dir);

 private:
  std::vector<std::string> roots_;
  std::vector<pid_t> pids_;
};

}  // namespace ferret
#endif  // FERRET_RPC_DAEMON_TEST_GUARD_H_

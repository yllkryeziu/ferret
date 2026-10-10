// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/watch/watcher.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"

#ifdef __APPLE__
#include "ferret/watch/fsevents_watcher.h"
#endif

namespace ferret {

absl::StatusOr<std::unique_ptr<Watcher>> NewWatcher(
    const std::string& root, const WatcherOptions& options) {
#ifdef __APPLE__
  auto watcher = FsEventsWatcher::Open(root, options);
#else
  auto watcher = PollingWatcher::Open(root, options);
#endif
  if (!watcher.ok()) return watcher.status();
  return std::unique_ptr<Watcher>(std::move(*watcher));
}

}  // namespace ferret

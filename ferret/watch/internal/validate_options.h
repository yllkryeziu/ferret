// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#ifndef FERRET_WATCH_INTERNAL_VALIDATE_OPTIONS_H_
#define FERRET_WATCH_INTERNAL_VALIDATE_OPTIONS_H_

#include "absl/status/status.h"
#include "absl/time/time.h"

#include "ferret/watch/watcher.h"

namespace ferret::watch_internal {

inline absl::Status ValidateOptions(const WatcherOptions& options) {
  if (options.quiet_window <= absl::ZeroDuration() ||
      options.max_delay <= absl::ZeroDuration()) {
    return absl::InvalidArgumentError(
        "watcher debounce durations must be positive");
  }
  return absl::OkStatus();
}

}  // namespace ferret::watch_internal

#endif  // FERRET_WATCH_INTERNAL_VALIDATE_OPTIONS_H_

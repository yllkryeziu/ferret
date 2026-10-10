// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#ifndef FERRET_WATCH_INTERNAL_COOKIE_DIRECTORY_H_
#define FERRET_WATCH_INTERNAL_COOKIE_DIRECTORY_H_

#include <string>

#include "absl/status/statusor.h"

namespace ferret::watch_internal {

// Creates the sync cookie directory `path` if it is missing, and a
// `.gitignore` of `*` inside it if that is missing, so `git status` never
// lists the directory and `git clean -d` keeps it. Returns whether this call
// created the directory.
absl::StatusOr<bool> MakeCookieDirectory(const std::string& path);

// Removes the `.gitignore` and then `path`, if it is empty; errors are
// ignored.
void RemoveCookieDirectory(const std::string& path);

}  // namespace ferret::watch_internal

#endif  // FERRET_WATCH_INTERNAL_COOKIE_DIRECTORY_H_

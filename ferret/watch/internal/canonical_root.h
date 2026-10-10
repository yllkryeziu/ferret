// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#ifndef FERRET_WATCH_INTERNAL_CANONICAL_ROOT_H_
#define FERRET_WATCH_INTERNAL_CANONICAL_ROOT_H_

#include <string>

#include "absl/status/statusor.h"

namespace ferret::watch_internal {

absl::StatusOr<std::string> CanonicalRoot(const std::string& root);

}  // namespace ferret::watch_internal

#endif  // FERRET_WATCH_INTERNAL_CANONICAL_ROOT_H_

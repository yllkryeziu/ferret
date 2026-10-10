// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/blob_registry.h"

#include <cstddef>
#include <cstdint>
#include <optional>

#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"

#include "ferret/git/oid.h"

namespace ferret {

BlobId BlobRegistry::Intern(const Oid& oid) {
  absl::MutexLock lock(mu_);
  auto [it, inserted] = ids_.try_emplace(oid, BlobId{});
  if (inserted) {
    CHECK_LT(oids_.size(), uint64_t{1} << 32);
    it->second = static_cast<BlobId>(oids_.size());
    oids_.push_back(oid);
  }
  return it->second;
}

std::optional<BlobId> BlobRegistry::Find(const Oid& oid) const {
  absl::MutexLock lock(mu_);
  auto it = ids_.find(oid);
  if (it == ids_.end()) return std::nullopt;
  return it->second;
}

Oid BlobRegistry::oid(BlobId id) const {
  absl::MutexLock lock(mu_);
  DCHECK_LT(static_cast<size_t>(id), oids_.size());
  return oids_[static_cast<size_t>(id)];
}

size_t BlobRegistry::size() const {
  absl::MutexLock lock(mu_);
  return oids_.size();
}

}  // namespace ferret

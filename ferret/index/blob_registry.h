// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Process-local identifiers for indexed content.

#ifndef FERRET_INDEX_BLOB_REGISTRY_H_
#define FERRET_INDEX_BLOB_REGISTRY_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"

#include "ferret/git/oid.h"

namespace ferret {

// A dense, process-local content identifier; never written to disk.
// Thread-compatible.
enum class BlobId : uint32_t {};

// Interns Oids in first-seen order. All operations are thread-safe.
class BlobRegistry final {
 public:
  BlobRegistry() = default;
  BlobRegistry(const BlobRegistry&) = delete;
  BlobRegistry& operator=(const BlobRegistry&) = delete;
  BlobRegistry(BlobRegistry&&) = delete;
  BlobRegistry& operator=(BlobRegistry&&) = delete;

  // Returns the existing id or assigns the next dense id.
  BlobId Intern(const Oid& oid) ABSL_LOCKS_EXCLUDED(mu_);
  // Returns nullopt if the Oid has never been interned.
  std::optional<BlobId> Find(const Oid& oid) const ABSL_LOCKS_EXCLUDED(mu_);
  // Requires an id previously returned by Intern(). Returns a copy.
  Oid oid(BlobId id) const ABSL_LOCKS_EXCLUDED(mu_);
  // Returns the number of interned Oids.
  size_t size() const ABSL_LOCKS_EXCLUDED(mu_);

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<Oid, BlobId> ids_ ABSL_GUARDED_BY(mu_);
  std::vector<Oid> oids_ ABSL_GUARDED_BY(mu_);
};

}  // namespace ferret

#endif  // FERRET_INDEX_BLOB_REGISTRY_H_

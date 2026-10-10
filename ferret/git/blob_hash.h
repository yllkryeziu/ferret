// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Computes the Oid of a blob: git's object id for the same bytes.
//
// Ferret hashes exactly the bytes it indexes, so every ingested file goes
// through HashBlob(). The header "blob <len>\0" is fed to SHA-1 separately
// from the content, so hashing never copies the content.

#ifndef FERRET_GIT_BLOB_HASH_H_
#define FERRET_GIT_BLOB_HASH_H_

#include "absl/strings/string_view.h"

#include "ferret/git/oid.h"

namespace ferret {

// Returns SHA-1("blob " + decimal(content.size()) + '\0' + content), which is
// what `git hash-object --no-filters` prints for `content`. O(content.size()).
// Thread-safe.
Oid HashBlob(absl::string_view content);

}  // namespace ferret

#endif  // FERRET_GIT_BLOB_HASH_H_

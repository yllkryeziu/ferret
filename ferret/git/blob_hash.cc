// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/blob_hash.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "openssl/sha.h"

#include "ferret/git/oid.h"

namespace ferret {

Oid HashBlob(absl::string_view content) {
  static_assert(SHA_DIGEST_LENGTH == Oid::kSize);
  // AlphaNum formats into an internal buffer, so the header costs no
  // allocation.
  const absl::AlphaNum length(content.size());
  SHA_CTX ctx;
  SHA1_Init(&ctx);
  SHA1_Update(&ctx, "blob ", 5);
  SHA1_Update(&ctx, length.data(), length.size());
  SHA1_Update(&ctx, "", 1);  // The terminating NUL of "".
  SHA1_Update(&ctx, content.data(), content.size());
  Oid::Bytes digest;
  SHA1_Final(digest.data(), &ctx);
  return Oid(digest);
}

}  // namespace ferret

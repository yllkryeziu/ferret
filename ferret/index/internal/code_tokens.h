// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The code-token section (kCodeTokens): which tokens are kept and how the
// section is built. The segment writer and the bulk build's partition builder
// share it, so both write the same bytes.

#ifndef FERRET_INDEX_INTERNAL_CODE_TOKENS_H_
#define FERRET_INDEX_INTERNAL_CODE_TOKENS_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "ferret/base/thread_pool.h"
#include "ferret/postings/ids.h"
#include "ferret/text/language.h"
#include "ferret/text/lexer.h"

namespace ferret::segment_internal {

// Folded code-identifier tokens to the ascending ids of the documents that
// contain one.
using CodeTokenMap = absl::flat_hash_map<std::string, std::vector<DocId>>;

// Lexes `text` from `entry` and calls `fn(key)` for every code token the
// section stores, folded. Returns the lex state after `text`, as
// ForEachCodeIdentifier does. `scratch` is reused between calls.
LexState ForEachCodeTokenKey(Language language, absl::string_view text,
                             LexState entry, std::string& scratch,
                             absl::FunctionRef<void(absl::string_view)> fn);

// Adds `doc` to the list of every token of `text`, once per token. Documents
// must be added in increasing order.
LexState AddCodeTokens(Language language, absl::string_view text,
                       LexState entry, DocId doc, std::string& scratch,
                       CodeTokenMap& tokens);

// Builds the kCodeTokens section from the union of `maps`. A token's list is
// the union of its lists, ascending; when `remap` is not empty each id is
// mapped through it first and the list sorted again. The ids in different
// maps must be disjoint. `pool` may be null. Returns
// ResourceExhausted when the section is full.
absl::StatusOr<std::string> BuildCodeTokensSection(
    absl::Span<const CodeTokenMap> maps, absl::Span<const DocId> remap,
    ThreadPool* pool, int threads);

}  // namespace ferret::segment_internal

#endif  // FERRET_INDEX_INTERNAL_CODE_TOKENS_H_

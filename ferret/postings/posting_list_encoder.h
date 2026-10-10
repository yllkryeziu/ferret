// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Writes posting lists in the format that PostingListView reads
// (posting_list.h).
//
// A segment writer stores a list of n ids as follows:
//   * If IsInlinedPostingList(n), the list has no postings bytes: its ids go
//     in the dictionary entry and are read back with
//     PostingListView::FromInlined.
//   * Otherwise AppendEncodedPostingList writes it, and
//     PostingListView::FromEncoded reads it back.
// For aligned NEON loads, the writer should start every list of at least
// kPostingBlockSize ids at a 16-byte-aligned offset of a 16-byte-aligned
// section (kPostings starts on a 16 KiB boundary): its block payloads then
// fall on 16-byte boundaries. This affects speed only; readers accept a list
// at any offset. Shorter lists have no alignment preference and can be packed
// back to back. Readers never read past the end of a list, so the section
// needs no padding after the last one.

#ifndef FERRET_POSTINGS_POSTING_LIST_ENCODER_H_
#define FERRET_POSTINGS_POSTING_LIST_ENCODER_H_

#include <string>

#include "absl/types/span.h"

#include "ferret/postings/ids.h"

namespace ferret {

// Appends the encoding of `ids` to `out`. The encoding is position
// independent: it depends only on `ids`, and its offsets are relative to its
// own first byte. Requires `ids` to be strictly increasing (DCHECKed) and
// kMaxInlinedPostings < ids.size() <= kMaxPostingListSize (CHECKed).
// O(ids.size()).
void AppendEncodedPostingList(absl::Span<const ChunkId> ids, std::string& out);

}  // namespace ferret

#endif  // FERRET_POSTINGS_POSTING_LIST_ENCODER_H_

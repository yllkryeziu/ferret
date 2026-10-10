// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The segment-local identifiers that posting lists carry.
//
// A DocId is the ordinal of a blob inside one segment, and a ChunkId the
// ordinal of a chunk (a line-aligned slice of one blob); both are assigned in
// rank order (rank_key.h), and a blob's chunks are contiguous. They are plain
// uint32_t aliases rather than strong types because the codecs and SIMD
// kernels do arithmetic on them. Every uint32_t value is a valid id: the
// codecs reserve no sentinel. Neither type may appear in the headers of
// ferret/search, engine, rpc or mcp, whose results name files by path and Oid.

#ifndef FERRET_POSTINGS_IDS_H_
#define FERRET_POSTINGS_IDS_H_

#include <cstddef>
#include <cstdint>

namespace ferret {

using DocId = uint32_t;
using ChunkId = uint32_t;

// The inclusive range of chunk ids [first, last], the unit of windowed query
// evaluation. Every function that takes one requires first <= last, so a
// range is never empty; it may span all 2^32 ids.
struct ChunkRange {
  ChunkId first = 0;
  ChunkId last = 0;

  // The number of ids in the range, from 1 to 2^32.
  size_t size() const { return size_t{last} - first + 1; }

  bool Contains(ChunkId id) const { return first <= id && id <= last; }

  friend bool operator==(const ChunkRange&, const ChunkRange&) = default;
};

}  // namespace ferret

#endif  // FERRET_POSTINGS_IDS_H_

// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// A query's immutable view of all published documents.

#ifndef FERRET_INDEX_READ_STATE_H_
#define FERRET_INDEX_READ_STATE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ferret/index/memtable.h"
#include "ferret/index/reaper.h"
#include "ferret/index/version_set.h"
#include "ferret/symbols/symbol_kind.h"

namespace ferret {

// The content is retained by the ReadState that produced this occurrence.
struct IndexedSymbol {
  BlobId blob;
  std::string name;
  SegmentDoc doc;
  uint32_t line;
  SymbolKind kind;
  bool is_definition;
};

struct SymbolLookup {
  std::vector<IndexedSymbol> occurrences;
  // Oids not yet extracted, including documents in legacy segments.
  std::vector<BlobId> pending_blobs;
};

// Frozen memtables are complete; only the active memtable needs a watermark.
// Holding a state retains its segment mappings and memtables. Thread-safe.
struct ReadState {
  // Declared before version so segment mappings release before the reaper.
  // Snapshots may outlive the Index that created them (but not its Env).
  std::shared_ptr<Reaper> reaper;
  // Retains the process-local Oid mapping for all captured BlobIds.
  std::shared_ptr<BlobRegistry> blobs;
  std::shared_ptr<const Version> version;
  std::vector<std::shared_ptr<const Memtable>> frozen;
  std::shared_ptr<const Memtable> mem;
  uint32_t mem_watermark = 0;

  // Queries every source by folded name and keeps the best RankKey for each
  // Oid, breaking rank ties by source order. Captured memtable watermarks
  // exclude future documents.
  SymbolLookup FindSymbol(absl::string_view name, bool case_insensitive) const;
};

}  // namespace ferret

#endif  // FERRET_INDEX_READ_STATE_H_

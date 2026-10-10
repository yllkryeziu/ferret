// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/read_state.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"

#include "ferret/git/oid.h"
#include "ferret/index/memtable.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/segment.h"
#include "ferret/index/version_set.h"
#include "ferret/postings/ids.h"
#include "ferret/symbols/symbol_extractor.h"

namespace ferret {

SymbolLookup ReadState::FindSymbol(absl::string_view name,
                                   bool case_insensitive) const {
  SymbolLookup result;
  absl::flat_hash_map<Oid, RankKey> best;
  const auto consider = [&](const SegmentDoc& doc) {
    const RankKey key{doc.tier, doc.prior, doc.oid};
    auto [it, inserted] = best.try_emplace(doc.oid, key);
    if (!inserted && key < it->second) it->second = key;
  };
  for (const VersionSegment& entry : version->segments) {
    for (DocId d = 0; d < entry.segment->num_docs(); ++d) {
      consider(entry.segment->doc(d));
    }
  }
  for (const auto& table : frozen) {
    for (DocId d = 0; d < table->published_docs(); ++d) consider(table->doc(d));
  }
  if (mem) {
    for (DocId d = 0; d < mem_watermark; ++d) consider(mem->doc(d));
  }
  absl::flat_hash_set<Oid> seen;
  const auto is_winner = [&](const SegmentDoc& doc) {
    return RankKey{doc.tier, doc.prior, doc.oid} == best.at(doc.oid) &&
           seen.insert(doc.oid).second;
  };
  const auto visit_mem = [&](const Memtable& table, uint32_t watermark) {
    absl::flat_hash_set<DocId> selected;
    for (DocId d = 0; d < watermark; ++d) {
      if (!is_winner(table.doc(d))) {
        continue;
      }
      selected.insert(d);
      if (table.SymbolsPending(d)) {
        result.pending_blobs.push_back(table.blob_of_doc(d));
      }
    }
    table.FindSymbols(name, case_insensitive, watermark,
                      [&](DocId id, const ExtractedSymbol& symbol) {
                        if (!selected.contains(id)) return;
                        result.occurrences.push_back(
                            {table.blob_of_doc(id), symbol.name, table.doc(id),
                             symbol.line, symbol.kind, symbol.is_definition});
                      });
  };
  const auto visit_segment = [&](const VersionSegment& entry) {
    absl::flat_hash_set<DocId> selected;
    for (DocId d = 0; d < entry.segment->num_docs(); ++d) {
      if (!is_winner(entry.segment->doc(d))) continue;
      selected.insert(d);
      if (!entry.segment->has_symbols()) {
        result.pending_blobs.push_back((*entry.blob_of_doc)[d]);
      }
    }
    entry.segment->FindSymbols(
        name, case_insensitive, [&](SegmentSymbol symbol) {
          if (!selected.contains(symbol.doc)) return;
          result.occurrences.push_back(
              {(*entry.blob_of_doc)[symbol.doc], std::string(symbol.name),
               entry.segment->doc(symbol.doc), symbol.line, symbol.kind,
               symbol.is_definition});
        });
  };
  // RankKey selects the best copy; equal-key ties use source order.
  for (const VersionSegment& entry : version->segments) visit_segment(entry);
  for (const auto& table : frozen) visit_mem(*table, table->published_docs());
  if (mem) visit_mem(*mem, mem_watermark);
  return result;
}

}  // namespace ferret

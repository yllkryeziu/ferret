// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// K-way merges ordered documents and dictionaries using heaps. Old-to-new id
// maps let posting cursors emit directly in new order.

#include "ferret/index/compactor.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "ferret/base/env.h"
#include "ferret/git/oid.h"
#include "ferret/index/rank_key.h"
#include "ferret/index/segment.h"
#include "ferret/index/segment_writer.h"
#include "ferret/postings/ids.h"
#include "ferret/postings/posting_list.h"
#include "ferret/symbols/symbol_extractor.h"
#include "ferret/text/language.h"
#include "ferret/text/trigram.h"

namespace ferret {
namespace {

constexpr uint32_t kDropped = std::numeric_limits<uint32_t>::max();

RankKey Key(const SegmentDoc& doc) {
  return RankKey{.tier = doc.tier, .prior = doc.prior, .oid = doc.oid};
}

struct DocumentPosition {
  RankKey key;
  size_t input;
  DocId doc;
  bool operator>(const DocumentPosition& other) const {
    if (key != other.key) return key > other.key;
    return input > other.input;
  }
};

struct TrigramPosition {
  Trigram trigram;
  size_t input;
  bool operator>(const TrigramPosition& other) const {
    if (trigram != other.trigram) return trigram > other.trigram;
    return input > other.input;
  }
};

struct PostingPosition {
  ChunkId id;
  size_t input;
  bool operator>(const PostingPosition& other) const { return id > other.id; }
};

// Skips postings whose document was dropped, preserving input list order.
void EnqueueNext(
    size_t input, PostingCursor& cursor, absl::Span<const ChunkId> remap,
    std::priority_queue<PostingPosition, std::vector<PostingPosition>,
                        std::greater<PostingPosition>>& heap) {
  while (!cursor.done() && remap[cursor.value()] == kDropped) cursor.Next();
  if (!cursor.done()) heap.push({remap[cursor.value()], input});
}

// Streams one nonempty remapped posting list per call. Dictionary positions
// advance even when all postings for a trigram were dropped.
class MergedPostingLists {
 public:
  MergedPostingLists(absl::Span<const Segment* const> inputs,
                     const std::vector<std::vector<ChunkId>>& remaps)
      : inputs_(inputs), remaps_(remaps), positions_(inputs.size(), 0) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i]->num_trigrams() != 0) {
        dictionary_.push({inputs[i]->trigram_at(0), i});
      }
    }
  }

  absl::StatusOr<bool> Next(Trigram& trigram, std::vector<ChunkId>& ids) {
    while (!dictionary_.empty()) {
      trigram = dictionary_.top().trigram;
      std::vector<PostingCursor> cursors;
      std::vector<size_t> sources;
      while (!dictionary_.empty() && dictionary_.top().trigram == trigram) {
        const size_t input = dictionary_.top().input;
        dictionary_.pop();
        ABSL_ASSIGN_OR_RETURN(PostingListView list,
                              inputs_[input]->FindPostingList(trigram));
        sources.push_back(input);
        cursors.emplace_back(list);
        if (++positions_[input] < inputs_[input]->num_trigrams()) {
          dictionary_.push(
              {inputs_[input]->trigram_at(positions_[input]), input});
        }
      }
      std::priority_queue<PostingPosition, std::vector<PostingPosition>,
                          std::greater<PostingPosition>>
          postings;
      for (size_t i = 0; i < cursors.size(); ++i) {
        EnqueueNext(i, cursors[i], remaps_[sources[i]], postings);
      }
      ids.clear();
      while (!postings.empty()) {
        const PostingPosition position = postings.top();
        postings.pop();
        if (ids.empty() || ids.back() != position.id) {
          ids.push_back(position.id);
        }
        cursors[position.input].Next();
        EnqueueNext(position.input, cursors[position.input],
                    remaps_[sources[position.input]], postings);
      }
      if (!ids.empty()) return true;
    }
    return false;
  }

 private:
  absl::Span<const Segment* const> inputs_;
  const std::vector<std::vector<ChunkId>>& remaps_;
  std::vector<uint32_t> positions_;
  std::priority_queue<TrigramPosition, std::vector<TrigramPosition>,
                      std::greater<TrigramPosition>>
      dictionary_;
};

}  // namespace

absl::StatusOr<SegmentStats> MergeSegments(
    Env& env, absl::Span<const Segment* const> inputs,
    absl::FunctionRef<bool(size_t, DocId)> keep, absl::string_view output_path,
    const SegmentWriterOptions& options, const std::atomic<bool>* cancelled) {
  if (!inputs.empty() && inputs.front() == nullptr) {
    return absl::InvalidArgumentError("null input segment");
  }
  for (const Segment* input : inputs) {
    if (input == nullptr ||
        input->chunk_bytes() != inputs.front()->chunk_bytes()) {
      return absl::InvalidArgumentError("input chunk_bytes mismatch");
    }
  }
  if (!inputs.empty() && inputs.front()->chunk_bytes() != options.chunk_bytes) {
    return absl::InvalidArgumentError("output chunk_bytes mismatch");
  }
  // The winner table records only references to input documents. This also
  // lets a later input with a better key displace an earlier copy.
  absl::flat_hash_map<Oid, DocumentPosition> winners;
  std::vector<std::vector<bool>> live(inputs.size());
  std::vector<std::vector<DocId>> doc_remaps(inputs.size());
  std::vector<std::vector<ChunkId>> remaps(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    live[i].resize(inputs[i]->num_docs());
    doc_remaps[i].assign(inputs[i]->num_docs(), kDropped);
    remaps[i].assign(inputs[i]->num_chunks(), kDropped);
    for (DocId d = 0; d < inputs[i]->num_docs(); ++d) {
      if (cancelled && cancelled->load(std::memory_order_relaxed)) {
        return absl::CancelledError("compaction stopped");
      }
      live[i][d] = keep(i, d);
      if (!live[i][d]) continue;
      const SegmentDoc doc = inputs[i]->doc(d);
      DocumentPosition candidate{Key(doc), i, d};
      auto [it, inserted] = winners.try_emplace(doc.oid, candidate);
      if (!inserted &&
          (candidate.key < it->second.key ||
           (candidate.key == it->second.key && i < it->second.input))) {
        it->second = candidate;
      }
    }
  }
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<SegmentWriter> writer,
                        SegmentWriter::Create(env, output_path, options));
  std::priority_queue<DocumentPosition, std::vector<DocumentPosition>,
                      std::greater<DocumentPosition>>
      documents;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i]->num_docs() != 0) {
      documents.push({Key(inputs[i]->doc(0)), i, 0});
    }
  }
  std::vector<std::vector<std::vector<ExtractedSymbol>>> stored(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!options.write_symbols || !inputs[i]->has_symbols()) continue;
    stored[i].resize(inputs[i]->num_docs());
    inputs[i]->ForEachSymbol([&](SegmentSymbol symbol) {
      stored[i][symbol.doc].push_back({.name = std::string(symbol.name),
                                       .kind = symbol.kind,
                                       .is_definition = symbol.is_definition,
                                       .line = symbol.line});
    });
  }
  SymbolExtractor symbol_extractor;
  // Every exit after writer creation must close and remove partial output.
  auto abort = [&](absl::Status status) -> absl::StatusOr<SegmentStats> {
    writer.reset();
    env.DeleteFile(output_path).IgnoreError();
    return status;
  };
  ChunkId next_chunk = 0;
  std::vector<ChunkId> output_first_chunks;
  while (!documents.empty()) {
    if (cancelled && cancelled->load(std::memory_order_relaxed)) {
      return abort(absl::CancelledError("compaction stopped"));
    }
    DocumentPosition position = documents.top();
    documents.pop();
    const SegmentDoc doc = inputs[position.input]->doc(position.doc);
    if (live[position.input][position.doc] &&
        winners.at(doc.oid).input == position.input &&
        winners.at(doc.oid).doc == position.doc) {
      doc_remaps[position.input][position.doc] =
          static_cast<DocId>(output_first_chunks.size());
      output_first_chunks.push_back(next_chunk);
      next_chunk += doc.num_chunks;
      std::vector<SegmentChunk> chunks;
      chunks.reserve(doc.num_chunks);
      for (ChunkId c = doc.first_chunk; c < doc.first_chunk + doc.num_chunks;
           ++c) {
        chunks.push_back(inputs[position.input]->chunk(c));
      }
      absl::Status status = writer->AddChunked(doc, chunks);
      if (!status.ok()) return abort(status);
      if (options.write_symbols && inputs[position.input]->has_symbols()) {
        writer->AddSymbols(stored[position.input][position.doc]);
      } else if (options.write_symbols && HasSymbolGrammar(doc.language)) {
        absl::StatusOr<std::vector<ExtractedSymbol>> symbols =
            symbol_extractor.Extract(doc.language, doc.content);
        if (symbols.ok()) {
          writer->AddSymbols(*symbols);
        } else {
          writer->MarkSymbolsFailed();
        }
      }
    }
    if (++position.doc < inputs[position.input]->num_docs()) {
      position.key = Key(inputs[position.input]->doc(position.doc));
      documents.push(position);
    }
  }
  // Both maps are monotonic in each input: documents and their chunks were
  // already ordered by RankKey. Dropped ids retain the sentinel.
  for (size_t i = 0; i < inputs.size(); ++i) {
    for (DocId d = 0; d < inputs[i]->num_docs(); ++d) {
      if (doc_remaps[i][d] == kDropped) continue;
      const SegmentDoc doc = inputs[i]->doc(d);
      const ChunkId first = output_first_chunks[doc_remaps[i][d]];
      for (ChunkId offset = 0; offset < doc.num_chunks; ++offset) {
        remaps[i][doc.first_chunk + offset] = first + offset;
      }
    }
  }
  // Copying keeps the pseudo-trigrams of the inputs that have them; a copy
  // of any input without them leaves some chunks unindexed.
  const bool short_line_trigrams = std::all_of(
      inputs.begin(), inputs.end(),
      [](const Segment* input) { return input->has_short_line_trigrams(); });
  MergedPostingLists lists(inputs, remaps);
  absl::StatusOr<SegmentStats> result = writer->FinishMerged(
      [&](Trigram& trigram, std::vector<ChunkId>& ids) -> absl::StatusOr<bool> {
        if (cancelled && cancelled->load(std::memory_order_relaxed)) {
          return absl::CancelledError("compaction stopped");
        }
        return lists.Next(trigram, ids);
      },
      short_line_trigrams);
  if (!result.ok()) return abort(result.status());
  return result;
}

}  // namespace ferret

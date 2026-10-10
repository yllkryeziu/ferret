// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/index/internal/code_tokens.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "ferret/base/thread_pool.h"
#include "ferret/index/internal/ordered_parallel.h"
#include "ferret/index/internal/segment_format.h"
#include "ferret/postings/ids.h"
#include "ferret/text/fold.h"
#include "ferret/text/language.h"
#include "ferret/text/lexer.h"

namespace ferret::segment_internal {
namespace {

namespace sf = segment_format;

using Entry = CodeTokenMap::value_type;

// Keys are ASCII, so the first two bytes order them into 128 * 128 shards
// that follow the key order: a key with one byte counts as a second byte 0.
constexpr size_t kNumShards = size_t{128} * 128;

size_t ShardOf(absl::string_view key) {
  return (static_cast<size_t>(static_cast<uint8_t>(key[0])) << 7) |
         (key.size() > 1 ? static_cast<uint8_t>(key[1]) : 0);
}

// The section pieces of a range of shards.
struct Piece {
  std::string pool;
  std::string postings;
  // Offsets relative to this piece.
  std::vector<sf::CodeTokenEntry> entries;
};

void AppendVarint(uint32_t value, std::string& out) {
  while (value >= 0x80) {
    out.push_back(static_cast<char>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<char>(value));
}

}  // namespace

LexState ForEachCodeTokenKey(Language language, absl::string_view text,
                             LexState entry, std::string& scratch,
                             absl::FunctionRef<void(absl::string_view)> fn) {
  return ForEachCodeIdentifier(
      language, text, entry, [&](size_t begin, size_t end) {
        const absl::string_view run = text.substr(begin, end - begin);
        // A query is an ASCII identifier and cannot start with a digit, so
        // it cannot equal a token with a digit first, a '$', or any byte
        // left non-ASCII after folding (only U+212A and U+017F fold to
        // ASCII). Long tokens are dropped and the reader knows the limit.
        if (run.front() >= '0' && run.front() <= '9') return;
        // Most runs are plain lowercase ASCII, which folds to itself.
        bool plain = run.size() <= sf::kMaxCodeTokenBytes;
        for (const char c : run) {
          if ((c >= 'A' && c <= 'Z') || static_cast<uint8_t>(c) >= 0x80) {
            plain = false;
            break;
          }
        }
        absl::string_view key = run;
        if (!plain) {
          scratch.clear();
          AppendFolded(run, scratch);
          key = scratch;
          if (key.size() > sf::kMaxCodeTokenBytes) return;
        }
        for (const char c : key) {
          if (c == '$' || static_cast<uint8_t>(c) >= 0x80) return;
        }
        fn(key);
      });
}

LexState AddCodeTokens(Language language, absl::string_view text,
                       LexState entry, DocId doc, std::string& scratch,
                       CodeTokenMap& tokens) {
  return ForEachCodeTokenKey(
      language, text, entry, scratch, [&](absl::string_view key) {
        std::vector<DocId>& docs = tokens.try_emplace(key).first->second;
        if (docs.empty() || docs.back() != doc) docs.push_back(doc);
      });
}

absl::StatusOr<std::string> BuildCodeTokensSection(
    absl::Span<const CodeTokenMap> maps, absl::Span<const DocId> remap,
    ThreadPool* pool, int threads) {
  // Each map's entries in key order.
  std::vector<std::vector<const Entry*>> sorted(maps.size());
  const auto sort_map = [&](size_t m) {
    sorted[m].reserve(maps[m].size());
    for (const Entry& entry : maps[m]) sorted[m].push_back(&entry);
    std::sort(
        sorted[m].begin(), sorted[m].end(),
        [](const Entry* a, const Entry* b) { return a->first < b->first; });
  };
  if (pool != nullptr && threads > 1 && maps.size() > 1) {
    pool->ParallelFor(static_cast<int64_t>(maps.size()), 1, threads,
                      [&](int, int64_t begin, int64_t end) {
                        for (int64_t m = begin; m < end; ++m) {
                          sort_map(static_cast<size_t>(m));
                        }
                      });
  } else {
    for (size_t m = 0; m < maps.size(); ++m) sort_map(m);
  }

  struct Scratch {
    std::vector<DocId> docs;
    std::vector<DocId> mapped;
  };
  std::vector<Scratch> scratch(static_cast<size_t>(std::max(threads, 1)));

  std::string pool_bytes;
  std::string posting_bytes;
  std::vector<sf::CodeTokenEntry> entries;

  const auto produce = [&](int worker, int64_t shard) -> absl::StatusOr<Piece> {
    Piece piece;
    // The range of each map's entries in this shard.
    struct Cursor {
      size_t next;
      size_t end;
    };
    std::vector<Cursor> cursors(maps.size());
    bool any = false;
    for (size_t m = 0; m < maps.size(); ++m) {
      const std::vector<const Entry*>& list = sorted[m];
      const auto lower = [&](size_t s) {
        return static_cast<size_t>(
            std::partition_point(
                list.begin(), list.end(),
                [&](const Entry* e) { return ShardOf(e->first) < s; }) -
            list.begin());
      };
      cursors[m] = {lower(static_cast<size_t>(shard)),
                    lower(static_cast<size_t>(shard) + 1)};
      any = any || cursors[m].next < cursors[m].end;
    }
    if (!any) return piece;
    std::vector<DocId>& merged = scratch[static_cast<size_t>(worker)].docs;
    std::vector<DocId>& mapped = scratch[static_cast<size_t>(worker)].mapped;
    while (true) {
      const Entry* smallest = nullptr;
      for (size_t m = 0; m < maps.size(); ++m) {
        if (cursors[m].next == cursors[m].end) continue;
        const Entry* e = sorted[m][cursors[m].next];
        if (smallest == nullptr || e->first < smallest->first) smallest = e;
      }
      if (smallest == nullptr) break;
      const std::vector<DocId>* docs = nullptr;
      size_t found = 0;
      for (size_t m = 0; m < maps.size(); ++m) {
        if (cursors[m].next == cursors[m].end) continue;
        const Entry* e = sorted[m][cursors[m].next];
        if (e->first != smallest->first) continue;
        ++cursors[m].next;
        ++found;
        if (found == 1) {
          docs = &e->second;
        } else {
          if (found == 2) {
            merged.assign(docs->begin(), docs->end());
            docs = &merged;
          }
          merged.insert(merged.end(), e->second.begin(), e->second.end());
        }
      }
      if (found > 1 && remap.empty()) std::sort(merged.begin(), merged.end());
      piece.pool.append(smallest->first);
      if (!remap.empty()) {
        // The ids are provisional: map them, then restore ascending order.
        mapped.clear();
        for (const DocId raw : *docs) mapped.push_back(remap[raw]);
        std::sort(mapped.begin(), mapped.end());
        docs = &mapped;
      }
      DocId previous = 0;
      for (const DocId doc : *docs) {
        AppendVarint(doc - previous, piece.postings);
        previous = doc;
      }
      if (piece.pool.size() > std::numeric_limits<uint32_t>::max() / 2 ||
          piece.postings.size() > std::numeric_limits<uint32_t>::max() / 2) {
        return absl::ResourceExhaustedError("code token section is full");
      }
      piece.entries.push_back(
          {.key_end = static_cast<uint32_t>(piece.pool.size()),
           .postings_end = static_cast<uint32_t>(piece.postings.size())});
    }
    return piece;
  };
  const auto commit = [&](int64_t, Piece&& piece) -> absl::Status {
    for (sf::CodeTokenEntry entry : piece.entries) {
      entry.key_end += static_cast<uint32_t>(pool_bytes.size());
      entry.postings_end += static_cast<uint32_t>(posting_bytes.size());
      entries.push_back(entry);
    }
    pool_bytes.append(piece.pool);
    posting_bytes.append(piece.postings);
    if (pool_bytes.size() > std::numeric_limits<uint32_t>::max() / 2 ||
        posting_bytes.size() > std::numeric_limits<uint32_t>::max() / 2) {
      return absl::ResourceExhaustedError("code token section is full");
    }
    return absl::OkStatus();
  };
  absl::Status status =
      ParallelOrdered<Piece>(pool, threads, static_cast<int64_t>(kNumShards),
                             threads + 2, produce, commit);
  if (!status.ok()) return status;

  std::string section;
  section.reserve(sizeof(sf::CodeTokensHeader) +
                  entries.size() * sizeof(sf::CodeTokenEntry) +
                  pool_bytes.size() + posting_bytes.size());
  sf::AppendRecord(
      sf::CodeTokensHeader{
          .num_tokens = static_cast<uint32_t>(entries.size()),
          .max_token_bytes = sf::kMaxCodeTokenBytes,
          .pool_bytes = static_cast<uint32_t>(pool_bytes.size()),
          .postings_bytes = static_cast<uint32_t>(posting_bytes.size())},
      section);
  for (const sf::CodeTokenEntry& entry : entries) {
    sf::AppendRecord(entry, section);
  }
  section.append(pool_bytes);
  section.append(posting_bytes);
  return section;
}

}  // namespace ferret::segment_internal

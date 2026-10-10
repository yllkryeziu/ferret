// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Per-worktree visibility and result expansion.

#ifndef FERRET_WORKSPACE_VIEW_STATE_H_
#define FERRET_WORKSPACE_VIEW_STATE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

#include "ferret/index/blob_registry.h"
#include "ferret/text/literal_finder.h"
#include "ferret/workspace/path_table.h"

namespace ferret {

// An immutable view over BlobIds. Each directory entry points to a shared
// chunk of 256 persistent path trees and a live bit per tree (set when its
// root is non-null).
// Old trees share untouched nodes, even when a blob has very many paths.
// Thread-safe.
class ViewState final {
 public:
  // Number of BlobIds per copy-on-write directory chunk.
  static constexpr size_t kChunkSize = 256;

  ViewState() = default;
  ViewState(const ViewState&) = default;
  ViewState& operator=(const ViewState&) = default;
  ViewState(ViewState&&) = default;
  ViewState& operator=(ViewState&&) = default;

  // Ids outside this snapshot's directory, including newly interned blobs,
  // are not visible. O(1).
  bool Contains(BlobId blob) const;
  // Lists paths in PathId order. The result owns its bytes; traversal costs
  // O(paths of blob), without locking or changing this snapshot.
  std::vector<PathId> PathsFor(BlobId blob) const;
  // The number of paths of `blob`, as PathsFor(blob).size() without building
  // the list. O(paths of blob).
  size_t PathCount(BlobId blob) const;
  // Visits each view path whose basename contains the literal. An empty
  // literal visits every path. Names are built once per shared chunk; a
  // modified chunk builds its own names on first use. Callbacks run without
  // holding a lock, and may call PathTable::path().
  void ForEachBasenameContaining(
      absl::string_view literal, LiteralFinder::Mode mode,
      const PathTable& paths,
      absl::FunctionRef<void(BlobId, PathId)> callback) const;
  // Like ForEachBasenameContaining, and also passes the basename, which is
  // valid for the call only.
  // Only the directory chunks [first_chunk, last_chunk) are visited; the
  // ranges of a partition of [0, num_chunks()) report exactly what one call
  // over all of them does. Concurrent calls are safe.
  void ForEachNameContaining(
      absl::string_view literal, LiteralFinder::Mode mode,
      const PathTable& paths,
      absl::FunctionRef<void(BlobId, PathId, absl::string_view)> callback,
      size_t first_chunk = 0,
      size_t last_chunk = std::numeric_limits<size_t>::max()) const;
  // The number of directory chunks, each of kChunkSize BlobIds.
  size_t num_chunks() const { return chunks_.size(); }
  // Logical bytes in this snapshot's directory, chunks and path trees;
  // shared allocations are counted in each snapshot that references them.
  size_t memory_bytes() const;

 private:
  friend class ViewWriter;
  struct PathNode;
  struct Chunk {
    Chunk() = default;
    // Copy only the immutable trees and the live bits; names belong to the old
    // snapshot chunk.
    Chunk(const Chunk& other) : live(other.live), roots(other.roots) {}
    const std::shared_ptr<const PathNode>& operator[](size_t i) const {
      return roots[i];
    }
    // Replaces slot `i`'s tree (null: the blob is not live) and its live bit.
    void Set(size_t i, std::shared_ptr<const PathNode> root) {
      const uint64_t bit = uint64_t{1} << (i % 64);
      live[i / 64] = root ? live[i / 64] | bit : live[i / 64] & ~bit;
      roots[i] = std::move(root);
    }
    bool Live(size_t i) const { return (live[i / 64] >> (i % 64) & 1) != 0; }

    struct Names {
      std::string bytes;
      std::vector<size_t> ends;
      std::vector<BlobId> blobs;
      std::vector<PathId> paths;
    };
    // Bit i is set when roots[i] is non-null, so that Contains reads one
    // cache line instead of the slot (which is 16 bytes of a 4 KiB array).
    std::array<uint64_t, kChunkSize / 64> live = {};
    std::array<std::shared_ptr<const PathNode>, kChunkSize> roots;
    mutable absl::once_flag names_once;
    mutable std::unique_ptr<const Names> names;
    const Names& Basenames(size_t chunk_id, const PathTable& paths) const;
  };
  const std::shared_ptr<const PathNode>& Entry(BlobId blob) const;
  static size_t CountPaths(const PathNode* node);
  static void CollectPaths(const PathNode* node, std::vector<PathId>& out);
  static size_t TreeBytes(const PathNode* node);
  std::vector<std::shared_ptr<const Chunk>> chunks_;
};

// One atomic path edit: nullopt deletes the path. Thread-compatible.
struct ViewEdit {
  // The interned path to update.
  PathId path;
  // The replacement content, or nullopt to delete the path.
  std::optional<BlobId> blob;
};

// The single writer of one worktree's mutable file and multiplicity maps.
// ApplyBatch requires external single-writer serialization. Snapshot() is
// safe concurrently and never exposes a partial batch. Thread-compatible.
class ViewWriter final {
 public:
  ViewWriter();
  ViewWriter(const ViewWriter&) = delete;
  ViewWriter& operator=(const ViewWriter&) = delete;
  ViewWriter(ViewWriter&&) = delete;
  ViewWriter& operator=(ViewWriter&&) = delete;

  // Applies edits in order, then publishes one immutable state. Copies the
  // directory and touched chunks, never the full files map or path lists.
  std::shared_ptr<const ViewState> ApplyBatch(absl::Span<const ViewEdit> edits);
  // Reports distinct blobs entering and leaving the view once per batch. The
  // callback must publish after adding, and before removing, references.
  using Reconcile =
      absl::FunctionRef<void(absl::Span<const BlobId>, absl::Span<const BlobId>,
                             absl::FunctionRef<void()>)>;
  std::shared_ptr<const ViewState> ApplyBatch(absl::Span<const ViewEdit> edits,
                                              Reconcile reconcile);
  // Takes a short lock to retain the latest published snapshot.
  std::shared_ptr<const ViewState> Snapshot() const ABSL_LOCKS_EXCLUDED(mu_);

 private:
  absl::flat_hash_map<PathId, BlobId> files_;
  absl::flat_hash_map<BlobId, uint32_t> multiplicity_;
  using Tree = std::shared_ptr<const ViewState::PathNode>;
  // Immutable trees share untouched nodes across batches and snapshots.
  absl::flat_hash_map<BlobId, Tree> paths_;
  static Tree Insert(const Tree& tree, PathId path);
  static Tree Erase(const Tree& tree, PathId path);
  static Tree Merge(Tree left, Tree right);
  mutable absl::Mutex mu_;
  std::shared_ptr<const ViewState> current_ ABSL_GUARDED_BY(mu_);
};

inline bool ViewState::Contains(BlobId blob) const {
  const size_t id = static_cast<size_t>(blob);
  // A directory slot between two used ranges holds no chunk.
  if (id / kChunkSize >= chunks_.size() || !chunks_[id / kChunkSize]) {
    return false;
  }
  return chunks_[id / kChunkSize]->Live(id % kChunkSize);
}

}  // namespace ferret

#endif  // FERRET_WORKSPACE_VIEW_STATE_H_

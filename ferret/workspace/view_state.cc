// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/workspace/view_state.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

#include "ferret/index/blob_registry.h"
#include "ferret/text/literal_finder.h"
#include "ferret/workspace/path_table.h"

namespace ferret {

// A persistent treap node. The priority is a deterministic permutation of
// PathId, so tree shape does not depend on the batch or edit order.
struct ViewState::PathNode {
  PathId path;
  uint32_t priority;
  std::shared_ptr<const PathNode> left;
  std::shared_ptr<const PathNode> right;
};

namespace {

uint32_t Priority(PathId path) {
  uint32_t x = static_cast<uint32_t>(path);
  x ^= x >> 16;
  x *= 0x7feb352d;
  x ^= x >> 15;
  x *= 0x846ca68b;
  return x ^ (x >> 16);
}

}  // namespace

const std::shared_ptr<const ViewState::PathNode>& ViewState::Entry(
    BlobId blob) const {
  const size_t id = static_cast<size_t>(blob);
  // The empty entry is deliberately leaked: static storage must be trivially
  // destructible, and a reference is returned for both present and absent ids.
  static const auto* empty = new std::shared_ptr<const PathNode>();
  // A directory slot between two used ranges holds no chunk.
  if (id / kChunkSize >= chunks_.size() || !chunks_[id / kChunkSize]) {
    return *empty;
  }
  return (*chunks_[id / kChunkSize])[id % kChunkSize];
}

void ViewState::CollectPaths(const PathNode* node, std::vector<PathId>& out) {
  if (!node) return;
  CollectPaths(node->left.get(), out);
  out.push_back(node->path);
  CollectPaths(node->right.get(), out);
}

std::vector<PathId> ViewState::PathsFor(BlobId blob) const {
  std::vector<PathId> result;
  CollectPaths(Entry(blob).get(), result);
  return result;
}

size_t ViewState::PathCount(BlobId blob) const {
  return CountPaths(Entry(blob).get());
}

size_t ViewState::CountPaths(const PathNode* node) {
  // A treap is as deep as log2 of its size, so the recursion is shallow.
  return node == nullptr
             ? 0
             : 1 + CountPaths(node->left.get()) + CountPaths(node->right.get());
}

const ViewState::Chunk::Names& ViewState::Chunk::Basenames(
    size_t chunk_id, const PathTable& paths) const {
  absl::call_once(names_once, [&] {
    auto built = std::make_unique<Names>();
    for (size_t i = 0; i < kChunkSize; ++i) {
      std::vector<PathId> ids;
      CollectPaths(roots[i].get(), ids);
      for (PathId id : ids) {
        const std::string path = paths.path(id);
        const size_t slash = path.rfind('/');
        built->bytes.append(path, slash == std::string::npos ? 0 : slash + 1);
        built->ends.push_back(built->bytes.size());
        built->blobs.push_back(static_cast<BlobId>(chunk_id * kChunkSize + i));
        built->paths.push_back(id);
      }
    }
    names = std::move(built);
  });
  return *names;
}

void ViewState::ForEachBasenameContaining(
    absl::string_view literal, LiteralFinder::Mode mode, const PathTable& paths,
    absl::FunctionRef<void(BlobId, PathId)> callback) const {
  ForEachNameContaining(literal, mode, paths,
                        [&](BlobId blob, PathId path, absl::string_view) {
                          callback(blob, path);
                        });
}

void ViewState::ForEachNameContaining(
    absl::string_view literal, LiteralFinder::Mode mode, const PathTable& paths,
    absl::FunctionRef<void(BlobId, PathId, absl::string_view)> callback,
    size_t first_chunk, size_t last_chunk) const {
  const LiteralFinder finder(literal, mode);
  last_chunk = std::min(last_chunk, chunks_.size());
  for (size_t chunk_id = first_chunk; chunk_id < last_chunk; ++chunk_id) {
    if (!chunks_[chunk_id]) continue;
    const Chunk::Names& names = chunks_[chunk_id]->Basenames(chunk_id, paths);
    if (literal.empty()) {
      for (size_t i = 0; i < names.ends.size(); ++i) {
        const size_t begin = i == 0 ? 0 : names.ends[i - 1];
        callback(names.blobs[i], names.paths[i],
                 absl::string_view(names.bytes)
                     .substr(begin, names.ends[i] - begin));
      }
      continue;
    }
    size_t from = 0;
    while (true) {
      const size_t hit = finder.Find(names.bytes, from);
      if (hit == absl::string_view::npos) break;
      const auto end =
          std::upper_bound(names.ends.begin(), names.ends.end(), hit);
      if (hit + literal.size() <= *end) {
        const size_t i = end - names.ends.begin();
        const size_t begin = i == 0 ? 0 : names.ends[i - 1];
        callback(names.blobs[i], names.paths[i],
                 absl::string_view(names.bytes).substr(begin, *end - begin));
      }
      // Discard hits crossing a basename boundary, including empty names.
      from = *end;
    }
  }
}

size_t ViewState::TreeBytes(const PathNode* node) {
  if (!node) return 0;
  return sizeof(PathNode) + TreeBytes(node->left.get()) +
         TreeBytes(node->right.get());
}

size_t ViewState::memory_bytes() const {
  size_t bytes = sizeof(*this) + chunks_.capacity() * sizeof(chunks_[0]);
  for (const auto& chunk : chunks_) {
    bytes += sizeof(Chunk);
    for (const auto& root : chunk->roots) bytes += TreeBytes(root.get());
  }
  return bytes;
}

ViewWriter::Tree ViewWriter::Merge(Tree left, Tree right) {
  if (!left) return right;
  if (!right) return left;
  if (left->priority < right->priority) {
    return std::make_shared<ViewState::PathNode>(
        left->path, left->priority, left->left, Merge(left->right, right));
  }
  return std::make_shared<ViewState::PathNode>(
      right->path, right->priority, Merge(left, right->left), right->right);
}

ViewWriter::Tree ViewWriter::Insert(const Tree& tree, PathId path) {
  if (!tree) {
    return std::make_shared<ViewState::PathNode>(path, Priority(path), nullptr,
                                                 nullptr);
  }
  if (path < tree->path) {
    Tree left = Insert(tree->left, path);
    if (left->priority < tree->priority) {
      Tree right = std::make_shared<ViewState::PathNode>(
          tree->path, tree->priority, left->right, tree->right);
      return std::make_shared<ViewState::PathNode>(
          left->path, left->priority, left->left, std::move(right));
    }
    return std::make_shared<ViewState::PathNode>(tree->path, tree->priority,
                                                 std::move(left), tree->right);
  }
  Tree right = Insert(tree->right, path);
  if (right->priority < tree->priority) {
    Tree left = std::make_shared<ViewState::PathNode>(
        tree->path, tree->priority, tree->left, right->left);
    return std::make_shared<ViewState::PathNode>(right->path, right->priority,
                                                 std::move(left), right->right);
  }
  return std::make_shared<ViewState::PathNode>(tree->path, tree->priority,
                                               tree->left, std::move(right));
}

ViewWriter::Tree ViewWriter::Erase(const Tree& tree, PathId path) {
  if (tree->path == path) return Merge(tree->left, tree->right);
  if (path < tree->path) {
    return std::make_shared<ViewState::PathNode>(
        tree->path, tree->priority, Erase(tree->left, path), tree->right);
  }
  return std::make_shared<ViewState::PathNode>(
      tree->path, tree->priority, tree->left, Erase(tree->right, path));
}

ViewWriter::ViewWriter() : current_(std::make_shared<ViewState>()) {}

std::shared_ptr<const ViewState> ViewWriter::Snapshot() const {
  absl::MutexLock lock(mu_);
  return current_;
}

std::shared_ptr<const ViewState> ViewWriter::ApplyBatch(
    absl::Span<const ViewEdit> edits) {
  return ApplyBatch(edits,
                    [](absl::Span<const BlobId>, absl::Span<const BlobId>,
                       absl::FunctionRef<void()> publish) { publish(); });
}

std::shared_ptr<const ViewState> ViewWriter::ApplyBatch(
    absl::Span<const ViewEdit> edits, Reconcile reconcile) {
  const std::shared_ptr<const ViewState> previous = Snapshot();
  absl::flat_hash_set<BlobId> touched;
  for (const ViewEdit& edit : edits) {
    auto old = files_.find(edit.path);
    if (old != files_.end()) {
      if (edit.blob == old->second) continue;
      const BlobId blob = old->second;
      touched.insert(blob);
      paths_[blob] = Erase(paths_[blob], edit.path);
      if (--multiplicity_[blob] == 0) {
        paths_.erase(blob);
        multiplicity_.erase(blob);
      }
      files_.erase(old);
    }
    if (edit.blob) {
      files_[edit.path] = *edit.blob;
      paths_[*edit.blob] = Insert(paths_[*edit.blob], edit.path);
      ++multiplicity_[*edit.blob];
      touched.insert(*edit.blob);
    }
  }
  auto next = std::make_shared<ViewState>(*previous);
  absl::flat_hash_map<size_t, std::shared_ptr<ViewState::Chunk>> modified;
  for (BlobId blob : touched) {
    const size_t id = static_cast<size_t>(blob);
    const size_t chunk_id = id / ViewState::kChunkSize;
    if (chunk_id >= next->chunks_.size()) next->chunks_.resize(chunk_id + 1);
    auto [it, inserted] = modified.try_emplace(chunk_id);
    if (inserted) {
      it->second =
          next->chunks_[chunk_id]
              ? std::make_shared<ViewState::Chunk>(*next->chunks_[chunk_id])
              : std::make_shared<ViewState::Chunk>();
      next->chunks_[chunk_id] = it->second;
    }
    const auto paths = paths_.find(blob);
    it->second->Set(id % ViewState::kChunkSize,
                    paths == paths_.end() ? nullptr : paths->second);
  }
  std::vector<BlobId> entered;
  std::vector<BlobId> left;
  for (BlobId blob : touched) {
    const bool was_present = previous->Contains(blob);
    const bool is_present = next->Contains(blob);
    if (is_present && !was_present) entered.push_back(blob);
    if (was_present && !is_present) left.push_back(blob);
  }
  reconcile(entered, left, [this, &next] {
    absl::MutexLock lock(mu_);
    current_ = next;
  });
  return next;
}

}  // namespace ferret

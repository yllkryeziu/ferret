// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Merges immutable segments without re-extracting text. Documents and
// dictionaries are k-way merged; memory holds id maps, output records, and
// only the posting list currently being encoded.

#ifndef FERRET_INDEX_COMPACTOR_H_
#define FERRET_INDEX_COMPACTOR_H_

#include <atomic>
#include <cstddef>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "ferret/base/env.h"
#include "ferret/index/segment.h"
#include "ferret/index/segment_writer.h"
#include "ferret/postings/ids.h"

namespace ferret {

// Writes one segment of kept, Oid-deduplicated documents in RankKey order.
// The lowest RankKey copy wins, with input order breaking equal-key ties.
// Inputs must remain open until return and have identical chunk_bytes; keep
// is called once per document. Input pointers must be nonnull. InvalidArgument
// is returned for mismatched chunk sizes. Thread-compatible with the inputs;
// the callback and output path must not be used concurrently. O(input docs +
// chunks + dictionary entries + decoded postings log(inputs)).
absl::StatusOr<SegmentStats> MergeSegments(
    Env& env, absl::Span<const Segment* const> inputs,
    absl::FunctionRef<bool(size_t, DocId)> keep, absl::string_view output_path,
    const SegmentWriterOptions& options = {},
    const std::atomic<bool>* cancelled = nullptr);

}  // namespace ferret

#endif  // FERRET_INDEX_COMPACTOR_H_

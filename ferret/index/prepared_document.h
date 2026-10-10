// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Shared file eligibility and metadata for both ingest lanes and bulk build.

#ifndef FERRET_INDEX_PREPARED_DOCUMENT_H_
#define FERRET_INDEX_PREPARED_DOCUMENT_H_

#include "absl/strings/string_view.h"

#include "ferret/index/rank_key.h"
#include "ferret/index/segment_writer.h"
#include "ferret/text/eligibility.h"
#include "ferret/text/language.h"
#include "ferret/text/trigram_extractor.h"

namespace ferret {

// Metadata for one file. Does not own the path or content used to index it.
struct PreparedDocument {
  // kNone means eligible; other values explain why the file was skipped.
  SkipReason skip_reason = SkipReason::kNone;
  // Defined only for eligible files.
  RankKey rank_key;
  Language language = Language::kUnknown;
  bool long_lines = false;

  // Binds the original path and bytes to eligible metadata. The caller must
  // keep both alive through SegmentWriter::Add or Index::Add.
  SegmentDocument Bind(absl::string_view path, absl::string_view bytes) const;
};

// Computes the same metadata for all ingest routes. Reuse the extractor only
// within one thread; concurrent calls require separate extractors.
PreparedDocument PrepareDocument(absl::string_view path,
                                 absl::string_view bytes,
                                 const EligibilityOptions& options,
                                 TrigramExtractor& extractor);

}  // namespace ferret

#endif  // FERRET_INDEX_PREPARED_DOCUMENT_H_

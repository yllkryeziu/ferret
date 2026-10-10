// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
#include <ostream>
#include <set>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/escaping.h"

#include "ferret/cli/command_util.h"
#include "ferret/cli/commands.h"
#include "ferret/index/segment.h"
#include "ferret/postings/ids.h"
#include "ferret/postings/list_ops.h"
#include "ferret/postings/posting_list.h"
#include "ferret/query/evaluator.h"
#include "ferret/query/trigram_query.h"
#include "ferret/regex/compile.h"
#include "ferret/text/trigram.h"

namespace ferret {
namespace {

void CollectTrigrams(const TrigramQuery& query, std::set<Trigram>& out) {
  out.insert(query.trigrams().begin(), query.trigrams().end());
  for (const TrigramQuery& sub : query.subs()) CollectTrigrams(sub, out);
}

const char* YesNo(bool value) { return value ? "yes" : "no"; }

// Shows each trigram's posting list size, and the candidates the query
// selects from the whole segment.
absl::Status ExplainAgainstIndex(const CompiledPattern& pattern,
                                 const Segment& segment, std::ostream& out) {
  std::set<Trigram> trigrams;
  CollectTrigrams(pattern.query(), trigrams);
  out << "index         " << segment.num_docs() << " docs, "
      << segment.num_chunks() << " chunks\n";
  for (const Trigram trigram : trigrams) {
    ABSL_ASSIGN_OR_RETURN(const PostingListView list,
                          segment.FindPostingList(trigram));
    out << "  " << TrigramToString(trigram) << "  " << list.size()
        << " chunks\n";
  }
  ABSL_ASSIGN_OR_RETURN(TrigramQueryEvaluator<PostingListView> evaluator,
                        segment.NewEvaluator(pattern.query()));
  ChunkIdBuffer candidates;
  if (segment.num_chunks() > 0 && !evaluator.matches_nothing()) {
    evaluator.Evaluate({.first = 0, .last = segment.num_chunks() - 1},
                       candidates);
  }
  std::set<DocId> docs;
  int64_t bytes = 0;
  for (const ChunkId id : candidates.ids()) {
    const SegmentChunk chunk = segment.chunk(id);
    docs.insert(chunk.doc);
    bytes += std::ssize(chunk.content);
  }
  out << "candidates    " << candidates.size() << " chunks in " << docs.size()
      << " files (" << HumanBytes(bytes) << ")\n";
  return absl::OkStatus();
}

}  // namespace

absl::Status RunPlanCommand(const PlanCommandOptions& options,
                            std::ostream& out) {
  ABSL_ASSIGN_OR_RETURN(const std::unique_ptr<CompiledPattern> pattern,
                        CompilePattern(options.pattern));
  const std::string& literal = pattern->required_literal();
  out << "pattern       \"" << absl::CEscape(options.pattern.pattern) << "\" ("
      << (options.pattern.literal ? "literal" : "regex") << ", "
      << (options.pattern.case_sensitive ? "case-sensitive"
                                         : "case-insensitive")
      << ")\n"
      << "query         " << pattern->query().ToString() << "\n"
      << "literal       "
      << (literal.empty() ? "none" : "\"" + absl::CEscape(literal) + "\"")
      << "\n"
      << "complete      " << YesNo(pattern->plan_complete()) << "\n"
      << "text anchor   " << YesNo(pattern->has_text_anchor()) << "\n";
  if (!options.explain) return absl::OkStatus();
  ABSL_ASSIGN_OR_RETURN(const std::string index_path,
                        ResolveIndexPath(options.root, options.index_path));
  ABSL_ASSIGN_OR_RETURN(const std::unique_ptr<Segment> segment,
                        OpenIndex(index_path));
  return ExplainAgainstIndex(*pattern, *segment, out);
}

}  // namespace ferret

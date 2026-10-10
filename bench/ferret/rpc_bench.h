// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The serving path of G1 ("request decoded -> response encoded") against the
// in-process search of RunSearchBench: the same queries sent to a
// SearchBackend, which is the Engine in this process or an RpcClient talking
// to a `ferret serve` daemon. Only the gap between the two matters, so the
// bench reports client-observed latencies of the same request G1 defines
// (K = 20 files, 2 context lines, estimated totals).

#ifndef BENCH_FERRET_RPC_BENCH_H_
#define BENCH_FERRET_RPC_BENCH_H_

#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "bench/ferret/query_set.h"
#include "bench/ferret/search_bench.h"

#include "ferret/engine/engine.h"

namespace ferret_bench {

// Searches `root` of `backend` with every query, `config.warmup_passes` and
// then `config.timed_runs` shuffled passes, one client, consistency kNone.
// Records carry engine `engine_name` and the client-observed latencies; the
// result digests are those of RunSearchBench, so the two can be compared.
absl::StatusOr<std::vector<SearchRecord>> RunBackendBench(
    const ferret::SearchBackend& backend, absl::string_view root,
    absl::string_view engine_name, absl::Span<const QuerySpec> queries,
    const SearchBenchConfig& config);

}  // namespace ferret_bench

#endif  // BENCH_FERRET_RPC_BENCH_H_

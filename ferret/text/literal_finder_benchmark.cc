// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Throughput of LiteralFinder against libc memmem and std::string_view::find
// when counting every occurrence of a needle in 16 MiB of source-like text.
// Target: at least 15 GB/s per performance core.

#include "ferret/text/literal_finder.h"

#include <_string.h>
#include <string.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/base/no_destructor.h"
#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"

#include "ferret/text/testing/synthetic_source.h"

namespace ferret {
namespace {

using Mode = LiteralFinder::Mode;

absl::string_view Haystack() {
  static const absl::NoDestructor<std::string> k_text(
      ferret_testing::SyntheticSource(int64_t{16} << 20, /*seed=*/1));
  return *k_text;
}

void BM_LiteralFinder(benchmark::State& state, absl::string_view needle,
                      Mode mode) {
  const absl::string_view haystack = Haystack();
  const LiteralFinder finder(needle, mode);
  for (auto _ : state) {
    int64_t count = 0;
    for (size_t pos = finder.Find(haystack); pos != absl::string_view::npos;
         pos = finder.Find(haystack, pos + 1)) {
      ++count;
    }
    benchmark::DoNotOptimize(count);
  }
  state.SetBytesProcessed(state.iterations() *
                          static_cast<int64_t>(haystack.size()));
}

void BM_Memmem(benchmark::State& state, absl::string_view needle) {
  const absl::string_view haystack = Haystack();
  const char* const end = haystack.data() + haystack.size();
  for (auto _ : state) {
    int64_t count = 0;
    for (const char* p = haystack.data();;) {
      const void* hit =
          memmem(p, static_cast<size_t>(end - p), needle.data(), needle.size());
      if (hit == nullptr) break;
      ++count;
      p = static_cast<const char*>(hit) + 1;
    }
    benchmark::DoNotOptimize(count);
  }
  state.SetBytesProcessed(state.iterations() *
                          static_cast<int64_t>(haystack.size()));
}

void BM_StringViewFind(benchmark::State& state, absl::string_view needle) {
  const absl::string_view haystack = Haystack();
  for (auto _ : state) {
    int64_t count = 0;
    for (size_t pos = haystack.find(needle); pos != absl::string_view::npos;
         pos = haystack.find(needle, pos + 1)) {
      ++count;
    }
    benchmark::DoNotOptimize(count);
  }
  state.SetBytesProcessed(state.iterations() *
                          static_cast<int64_t>(haystack.size()));
}

// A rare identifier, common identifiers, a short token, and punctuation.
BENCHMARK_CAPTURE(BM_LiteralFinder, OnConnectionError, "OnConnectionError",
                  Mode::kExact);
BENCHMARK_CAPTURE(BM_LiteralFinder, OnConnectionErrorCaseInsensitive,
                  "OnConnectionError", Mode::kAsciiCaseInsensitive);
BENCHMARK_CAPTURE(BM_Memmem, OnConnectionError, "OnConnectionError");
BENCHMARK_CAPTURE(BM_StringViewFind, OnConnectionError, "OnConnectionError");
BENCHMARK_CAPTURE(BM_LiteralFinder, WebContents, "WebContents", Mode::kExact);
BENCHMARK_CAPTURE(BM_LiteralFinder, WebContentsCaseInsensitive, "WebContents",
                  Mode::kAsciiCaseInsensitive);
BENCHMARK_CAPTURE(BM_Memmem, WebContents, "WebContents");
BENCHMARK_CAPTURE(BM_StringViewFind, WebContents, "WebContents");
BENCHMARK_CAPTURE(BM_LiteralFinder, DCHECK, "DCHECK", Mode::kExact);
BENCHMARK_CAPTURE(BM_LiteralFinder, DCHECKCaseInsensitive, "DCHECK",
                  Mode::kAsciiCaseInsensitive);
BENCHMARK_CAPTURE(BM_Memmem, DCHECK, "DCHECK");
BENCHMARK_CAPTURE(BM_StringViewFind, DCHECK, "DCHECK");
BENCHMARK_CAPTURE(BM_LiteralFinder, Int, "int", Mode::kExact);
BENCHMARK_CAPTURE(BM_LiteralFinder, IntCaseInsensitive, "int",
                  Mode::kAsciiCaseInsensitive);
BENCHMARK_CAPTURE(BM_Memmem, Int, "int");
BENCHMARK_CAPTURE(BM_StringViewFind, Int, "int");
BENCHMARK_CAPTURE(BM_LiteralFinder, StdString, "std::string", Mode::kExact);
BENCHMARK_CAPTURE(BM_LiteralFinder, StdStringCaseInsensitive, "std::string",
                  Mode::kAsciiCaseInsensitive);
BENCHMARK_CAPTURE(BM_Memmem, StdString, "std::string");
BENCHMARK_CAPTURE(BM_StringViewFind, StdString, "std::string");

}  // namespace
}  // namespace ferret

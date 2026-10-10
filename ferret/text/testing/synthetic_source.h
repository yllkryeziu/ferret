// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Deterministic C++-like text for benchmarks.
//
// Text kernels are sensitive to the byte distribution of their input: the
// literal finder's filter keys on rare bytes, and trigram deduplication on
// repetition. Random bytes would flatter both. Microbenchmarks cannot read the
// corpora (they must run anywhere), so they use this generator, which strings
// together identifiers, keywords, punctuation, indentation, and comments drawn
// from Chromium and Linux style code.

#ifndef FERRET_TEXT_TESTING_SYNTHETIC_SOURCE_H_
#define FERRET_TEXT_TESTING_SYNTHETIC_SOURCE_H_

#include <cstdint>
#include <string>

namespace ferret_testing {

// Returns `size` bytes of source-like text in '\n'-terminated lines of about 20
// to 110 bytes; the last line may be cut short. The same (size, seed) yields
// the same text on every platform. Thread-safe.
std::string SyntheticSource(int64_t size, uint64_t seed);

}  // namespace ferret_testing

#endif  // FERRET_TEXT_TESTING_SYNTHETIC_SOURCE_H_

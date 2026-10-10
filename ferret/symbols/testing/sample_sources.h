// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Realistic source files for symbol extraction tests and benchmarks.
//
// Each sample is a complete file in the style of the corpora Ferret targets:
// Linux-style C, Chromium-style C++, and a typical Python library module. They
// mix the constructs the queries must classify (prototypes and definitions,
// out-of-line methods, export macros, K&R C, decorators, nested classes, and
// locals that must not be reported) the way real files do. Microbenchmarks
// cannot read the corpora, so these samples stand in for them.

#ifndef FERRET_SYMBOLS_TESTING_SAMPLE_SOURCES_H_
#define FERRET_SYMBOLS_TESTING_SAMPLE_SOURCES_H_

#include "absl/strings/string_view.h"

#include "ferret/text/language.h"

namespace ferret_testing {

// Returns the sample file for `language`, which must be C, C++, or Python.
// The view refers to static storage. Thread-safe.
absl::string_view SampleSource(ferret::Language language);

}  // namespace ferret_testing

#endif  // FERRET_SYMBOLS_TESTING_SAMPLE_SOURCES_H_

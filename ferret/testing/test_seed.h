// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The seed of every randomized test.
//
// Randomized tests draw from std::mt19937_64 seeded with a fixed default, so
// every run is deterministic. Setting FERRET_TEST_SEED replays or explores
// another seed without editing the test:
//
//   bazel test --test_env=FERRET_TEST_SEED=12345 //ferret/base:bitset_test

#ifndef FERRET_TESTING_TEST_SEED_H_
#define FERRET_TESTING_TEST_SEED_H_

#include <cstdint>

namespace ferret_testing {

// Returns the value of the FERRET_TEST_SEED environment variable, or
// `default_seed` if it is unset or empty. CHECK-fails if it is not a decimal
// uint64_t, because silently falling back would hide a typo. Record the
// result with SCOPED_TRACE so that a failure names its seed. Thread-safe as
// long as nothing modifies the environment concurrently.
uint64_t TestSeed(uint64_t default_seed);

}  // namespace ferret_testing

#endif  // FERRET_TESTING_TEST_SEED_H_

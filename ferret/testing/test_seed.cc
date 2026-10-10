// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/testing/test_seed.h"

#include <cstdint>
#include <cstdlib>

#include "absl/log/check.h"
#include "absl/strings/numbers.h"

namespace ferret_testing {

uint64_t TestSeed(uint64_t default_seed) {
  const char* const value = std::getenv("FERRET_TEST_SEED");
  if (value == nullptr || value[0] == '\0') return default_seed;
  uint64_t seed = 0;
  CHECK(absl::SimpleAtoi(value, &seed))
      << "FERRET_TEST_SEED is not an unsigned integer: \"" << value << "\"";
  return seed;
}

}  // namespace ferret_testing

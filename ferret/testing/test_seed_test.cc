// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/testing/test_seed.h"

#include <_stdlib.h>
#include <stdlib.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include "gtest/gtest.h"

namespace ferret {
namespace {

using ::ferret_testing::TestSeed;

constexpr char kVariable[] = "FERRET_TEST_SEED";

// Restores FERRET_TEST_SEED after each test, so that the suite also passes
// when it is run with the variable set.
class TestSeedTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (const char* const value = getenv(kVariable); value != nullptr) {
      saved_ = value;
    }
  }
  void TearDown() override {
    if (saved_.has_value()) {
      setenv(kVariable, saved_->c_str(), /*overwrite=*/1);
    } else {
      unsetenv(kVariable);
    }
  }

 private:
  std::optional<std::string> saved_;
};

using TestSeedDeathTest = TestSeedTest;

TEST_F(TestSeedTest, UnsetOrEmptyGivesTheDefault) {
  unsetenv(kVariable);
  EXPECT_EQ(TestSeed(42), 42u);
  setenv(kVariable, "", /*overwrite=*/1);
  EXPECT_EQ(TestSeed(42), 42u);
}

TEST_F(TestSeedTest, EnvironmentOverridesTheDefault) {
  setenv(kVariable, "0", /*overwrite=*/1);
  EXPECT_EQ(TestSeed(42), 0u);
  setenv(kVariable, "18446744073709551615", /*overwrite=*/1);
  EXPECT_EQ(TestSeed(42), std::numeric_limits<uint64_t>::max());
}

TEST_F(TestSeedDeathTest, MalformedSeedFailsLoudly) {
  for (const char* const bad : {"12x", "-1", "18446744073709551616"}) {
    setenv(kVariable, bad, /*overwrite=*/1);
    EXPECT_DEATH(TestSeed(42), "FERRET_TEST_SEED is not an unsigned integer")
        << bad;
  }
}

}  // namespace
}  // namespace ferret

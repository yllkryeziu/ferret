// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The one status macro Abseil does not ship. Everything else comes from
// absl/status/status_matchers.h (absl_testing::IsOk, IsOkAndHolds, StatusIs,
// ABSL_ASSERT_OK).

#ifndef FERRET_TESTING_STATUS_MATCHERS_H_
#define FERRET_TESTING_STATUS_MATCHERS_H_

#include <utility>

#include "gtest/gtest.h"

// Evaluates `rexpr` (an absl::StatusOr<T>), fails the test with the status if
// it is not OK, and otherwise moves the value into `lhs`, which may be a
// declaration:
//
//   FERRET_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Segment> segment,
//                               Segment::Open(path));
#define FERRET_ASSERT_OK_AND_ASSIGN(lhs, rexpr) \
  FERRET_ASSERT_OK_AND_ASSIGN_IMPL_(            \
      FERRET_STATUS_MACROS_CONCAT_(ferret_status_or_, __LINE__), lhs, rexpr)

// `statusor` names a variable and `lhs` may be a declaration, so neither can
// be parenthesized.
// NOLINTBEGIN(bugprone-macro-parentheses)
#define FERRET_ASSERT_OK_AND_ASSIGN_IMPL_(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                                      \
  ASSERT_TRUE(statusor.ok()) << statusor.status();              \
  lhs = std::move(statusor).value()
// NOLINTEND(bugprone-macro-parentheses)

#define FERRET_STATUS_MACROS_CONCAT_(x, y) \
  FERRET_STATUS_MACROS_CONCAT_IMPL_(x, y)
#define FERRET_STATUS_MACROS_CONCAT_IMPL_(x, y) x##y

#endif  // FERRET_TESTING_STATUS_MATCHERS_H_

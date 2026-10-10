// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/oid.h"

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ferret {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

constexpr absl::string_view kHelloBlob =
    "ce013625030ba8dba906f756967f9e9ca394464a";

TEST(OidTest, DefaultIsNull) {
  EXPECT_TRUE(Oid().is_null());
  EXPECT_EQ(Oid().ToHex(), std::string(40, '0'));
}

TEST(OidTest, HexRoundTrips) {
  absl::StatusOr<Oid> oid = Oid::FromHex(kHelloBlob);
  ASSERT_TRUE(oid.ok());
  EXPECT_EQ(oid->ToHex(), kHelloBlob);
  EXPECT_FALSE(oid->is_null());
}

TEST(OidTest, FromHexAcceptsUppercase) {
  EXPECT_THAT(Oid::FromHex("CE013625030BA8DBA906F756967F9E9CA394464A"),
              IsOkAndHolds(*Oid::FromHex(kHelloBlob)));
}

TEST(OidTest, FromHexRejectsBadInput) {
  EXPECT_THAT(Oid::FromHex("abc"), StatusIs(absl::StatusCode::kInvalidArgument,
                                            HasSubstr("expected 40")));
  EXPECT_THAT(Oid::FromHex(std::string(39, 'a') + "g"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("invalid hex digit")));
}

TEST(OidTest, RawRoundTrips) {
  const Oid oid = *Oid::FromHex(kHelloBlob);
  EXPECT_EQ(Oid::FromRaw(oid.raw()), oid);
}

TEST(OidTest, OrderingIsLexicographicOnBytes) {
  const Oid a = *Oid::FromHex(std::string(40, '0'));
  const Oid b = *Oid::FromHex(std::string(39, '0') + "1");
  EXPECT_LT(a, b);
  EXPECT_NE(a, b);
}

TEST(OidTest, WorksAsHashKeyAndInStrCat) {
  absl::flat_hash_set<Oid> set = {*Oid::FromHex(kHelloBlob), Oid()};
  EXPECT_TRUE(set.contains(Oid()));
  EXPECT_EQ(absl::StrCat(*Oid::FromHex(kHelloBlob)), kHelloBlob);
}

}  // namespace
}  // namespace ferret

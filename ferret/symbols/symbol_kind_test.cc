// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Tests the stable names of every persisted symbol kind.

#include "ferret/symbols/symbol_kind.h"

#include <utility>

#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

namespace ferret {
namespace {

TEST(SymbolKindTest, EveryEnumeratorHasAStableRoundTrippingName) {
  constexpr std::pair<SymbolKind, absl::string_view> kKinds[] = {
      {SymbolKind::kFunction, "function"},
      {SymbolKind::kMethod, "method"},
      {SymbolKind::kClass, "class"},
      {SymbolKind::kStruct, "struct"},
      {SymbolKind::kUnion, "union"},
      {SymbolKind::kEnum, "enum"},
      {SymbolKind::kEnumerator, "enumerator"},
      {SymbolKind::kTypeAlias, "type_alias"},
      {SymbolKind::kMacro, "macro"},
      {SymbolKind::kVariable, "variable"},
  };
  for (const auto& [kind, name] : kKinds) {
    EXPECT_EQ(SymbolKindName(kind), name);
    EXPECT_EQ(ParseSymbolKind(name), kind);
  }
  EXPECT_EQ(SymbolKindName(static_cast<SymbolKind>(255)), "unknown");
  EXPECT_FALSE(ParseSymbolKind("unknown").has_value());
  EXPECT_FALSE(ParseSymbolKind("Function").has_value());
}

}  // namespace
}  // namespace ferret

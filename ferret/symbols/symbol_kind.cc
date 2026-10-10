// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/symbols/symbol_kind.h"

#include <optional>

#include "absl/strings/string_view.h"

namespace ferret {
namespace {

constexpr SymbolKind kAllKinds[] = {
    SymbolKind::kFunction,   SymbolKind::kMethod,    SymbolKind::kClass,
    SymbolKind::kStruct,     SymbolKind::kUnion,     SymbolKind::kEnum,
    SymbolKind::kEnumerator, SymbolKind::kTypeAlias, SymbolKind::kMacro,
    SymbolKind::kVariable,
};

}  // namespace

absl::string_view SymbolKindName(SymbolKind kind) {
  switch (kind) {
    case SymbolKind::kFunction:
      return "function";
    case SymbolKind::kMethod:
      return "method";
    case SymbolKind::kClass:
      return "class";
    case SymbolKind::kStruct:
      return "struct";
    case SymbolKind::kUnion:
      return "union";
    case SymbolKind::kEnum:
      return "enum";
    case SymbolKind::kEnumerator:
      return "enumerator";
    case SymbolKind::kTypeAlias:
      return "type_alias";
    case SymbolKind::kMacro:
      return "macro";
    case SymbolKind::kVariable:
      return "variable";
  }
  return "unknown";
}

std::optional<SymbolKind> ParseSymbolKind(absl::string_view name) {
  for (const SymbolKind kind : kAllKinds) {
    if (SymbolKindName(kind) == name) return kind;
  }
  return std::nullopt;
}

}  // namespace ferret

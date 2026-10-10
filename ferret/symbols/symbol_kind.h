// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// SymbolKind: what a symbol extracted by tree-sitter names.
//
// The set is the entities an agent navigates to by name: callables, types,
// macros, enumerators, and namespace-scope variables. Namespaces are left out
// because every file reopens its namespaces, so a namespace "definition"
// locates nothing. Data members are left out because they are only reachable
// through their class, which is itself a symbol.
//
// Whether an occurrence defines or only declares its symbol is orthogonal to
// the kind; it is carried by ExtractedSymbol::is_definition
// (symbol_extractor.h).

#ifndef FERRET_SYMBOLS_SYMBOL_KIND_H_
#define FERRET_SYMBOLS_SYMBOL_KIND_H_

#include <cstdint>
#include <optional>

#include "absl/strings/string_view.h"

namespace ferret {

// Persisted in .fseg: append only, never renumber.
enum class SymbolKind : uint8_t {
  kFunction = 0,  // A function that is not a class member.
  kMethod = 1,    // A member function, including constructors and operators.
  kClass = 2,     // A C++ or Python class.
  kStruct = 3,
  kUnion = 4,
  kEnum = 5,
  kEnumerator = 6,
  kTypeAlias = 7,  // typedef, `using X = …`, Python `type X = …`.
  kMacro = 8,      // Object-like and function-like #define.
  kVariable = 9,   // A variable at namespace or module scope.
};

// Returns a stable lowercase name such as "type_alias", used in query capture
// names and tool output; "unknown" for a value outside the enum.
absl::string_view SymbolKindName(SymbolKind kind);

// The inverse of SymbolKindName: nullopt unless `name` names a kind.
std::optional<SymbolKind> ParseSymbolKind(absl::string_view name);

}  // namespace ferret

#endif  // FERRET_SYMBOLS_SYMBOL_KIND_H_

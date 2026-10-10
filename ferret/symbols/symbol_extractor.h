// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Extracts the symbols a C, C++, or Python file defines and declares, for the
// index's kSymbols section and the router.
//
// Extraction parses the file with tree-sitter and runs one of Ferret's own
// queries (queries/{c,cpp,python}.scm, embedded at build time) over the tree.
// The queries decide what is a symbol, its kind, and whether an occurrence
// defines or only declares it. Upstream tags.scm cannot be used for this: it
// tags every C++ function declarator, prototypes included, as a definition.
//
// Query contract. Each pattern captures either a symbol or a scope:
//
//   @definition.<kind>, @declaration.<kind>
//       The construct that introduces a symbol. <kind> is a SymbolKindName.
//   @name
//       The symbol's name, or a C/C++ declarator that contains it. The
//       extractor unwraps function, pointer, reference, array, parenthesized,
//       init, qualified, and template declarators down to the name.
//   @scope.local
//       A function body. Nothing named inside one is reported.
//   @scope.class
//       A class body. A function named directly in it is a method, and a
//       variable named directly in it (a data member) is not reported.
//   @scope.namespace
//       A region of a class body whose names belong to the enclosing
//       namespace: a friend declaration.
//
// A name's scope is the innermost scope capture that contains it, except that
// a name inside any @scope.local is local. Three predicates, evaluated on the
// unwrapped @name declarator, express what the query language cannot:
//
//   (#declares-function? @name)      The derivation nearest the name is a
//                                    function: `f(void)`, `*f(void)`.
//   (#not-declares-function? @name)  It is not, so an object is declared:
//                                    `x`, `*p`, `a[4]`, `(*fp)(void)`.
//   (#is-qualified? @name)           The name is qualified: `Foo::Bar`.
//
// When several patterns capture the same name, the first one listed wins, so
// specific patterns precede general ones. The queries are compiled and
// checked against this contract on first use; a violation is a defect in the
// build, not in the input, and CHECK-fails.
//
// Tree-sitter-cpp cannot tell that `class CONTENT_EXPORT WebContents` names
// an attribute macro and parses the whole class as a function, losing every
// member. C++ input therefore has such macros blanked out first
// (internal/export_macros.h), which keeps every offset and line unchanged.

#ifndef FERRET_SYMBOLS_SYMBOL_EXTRACTOR_H_
#define FERRET_SYMBOLS_SYMBOL_EXTRACTOR_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tree_sitter/api.h"

#include "ferret/symbols/symbol_kind.h"
#include "ferret/text/language.h"

namespace ferret {

struct SymbolExtractorOptions {
  // Larger inputs are rejected with ResourceExhausted before parsing. The
  // indexer never offers more than EligibilityOptions::max_file_bytes (1 MiB);
  // this bounds direct callers. Tree-sitter offsets are 32-bit, so the
  // effective limit is at most 4 GiB - 1.
  int64_t max_input_bytes = int64_t{4} << 20;
  // Work budget for parsing and querying one input, in tree-sitter progress
  // checks, one per 100 parser or query operations; exceeding it returns
  // ResourceExhausted. Unlike a time limit, a count of work gives the same
  // answer under any load, so two builds of the same files write the same
  // symbols. The costliest file of Linux v7.3-rc1 takes 19,563 checks, and
  // of Chromium 24,633, apart from one generated fontconfig header of 88,894
  // (2.1 s). At the slowest rate measured, 23 µs per check on an M-series
  // core, the default takes 1.2 s.
  int64_t max_progress_checks = 50'000;
};

// One occurrence of a symbol in a file.
struct ExtractedSymbol {
  // The unqualified name as written, with whitespace inside operator names
  // normalized: "Bar" for `Foo::Bar`, "~Foo", "operator==", "operator new",
  // "operator const char*".
  std::string name;
  SymbolKind kind = SymbolKind::kFunction;
  // True where the occurrence fully introduces the entity: a function with a
  // body, a class with a body, a variable that is not `extern`, a macro, an
  // alias, or an enumerator. False for prototypes, in-class member function
  // declarations, and `extern` variables.
  bool is_definition = false;
  // 1-based line of the name. For a decorated Python function or a C++
  // template this is the line of the name, not of the decorator or
  // `template <…>`.
  uint32_t line = 0;

  friend bool operator==(const ExtractedSymbol&,
                         const ExtractedSymbol&) = default;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ExtractedSymbol& symbol) {
    sink.Append(
        absl::StrCat(symbol.line, ": ",
                     symbol.is_definition ? "definition " : "declaration ",
                     SymbolKindName(symbol.kind), " ", symbol.name));
  }
};

// Owns a tree-sitter parser and, per language, the compiled query, which is
// built on first use. Reuse one extractor per thread: construction is cheap,
// but compiling a query is not (about 23 ms for C++ on an M-series core,
// against about 0.3 ms to extract from 1 KiB). Not thread-safe.
class SymbolExtractor final {
 public:
  explicit SymbolExtractor(SymbolExtractorOptions options = {});
  ~SymbolExtractor();

  SymbolExtractor(const SymbolExtractor&) = delete;
  SymbolExtractor& operator=(const SymbolExtractor&) = delete;
  SymbolExtractor(SymbolExtractor&&) = delete;
  SymbolExtractor& operator=(SymbolExtractor&&) = delete;

  // Returns the symbols of `content`, a whole file in `language`, in the order
  // their names appear. Syntax errors are not errors: tree-sitter recovers,
  // and the symbols of the parts it recognizes are returned. The result
  // depends only on (language, content) and the options.
  //
  // Returns InvalidArgument if !HasSymbolGrammar(language), ResourceExhausted
  // if `content` is larger than max_input_bytes or parsing and querying it
  // take more than max_progress_checks.
  absl::StatusOr<std::vector<ExtractedSymbol>> Extract(
      Language language, absl::string_view content);

 private:
  // A compiled query and the rule of each of its patterns; defined in the .cc.
  struct LanguageQuery;

  struct ParserDeleter {
    void operator()(TSParser* parser) const { ts_parser_delete(parser); }
  };
  struct QueryCursorDeleter {
    void operator()(TSQueryCursor* cursor) const {
      ts_query_cursor_delete(cursor);
    }
  };

  // Returns the query for `language`, compiling it on first use.
  // Requires HasSymbolGrammar(language).
  const LanguageQuery& QueryFor(Language language);

  const SymbolExtractorOptions options_;
  const std::unique_ptr<TSParser, ParserDeleter> parser_;
  const std::unique_ptr<TSQueryCursor, QueryCursorDeleter> cursor_;
  std::unique_ptr<const LanguageQuery> c_query_;
  std::unique_ptr<const LanguageQuery> cpp_query_;
  std::unique_ptr<const LanguageQuery> python_query_;
  // C++ input with export macros blanked, reused across calls.
  std::string blanked_;
};

}  // namespace ferret

#endif  // FERRET_SYMBOLS_SYMBOL_EXTRACTOR_H_

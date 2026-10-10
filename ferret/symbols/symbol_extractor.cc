// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/symbols/symbol_extractor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "tree_sitter/api.h"

#include "ferret/symbols/internal/embedded_queries.h"
#include "ferret/symbols/internal/export_macros.h"
#include "ferret/symbols/symbol_kind.h"
#include "ferret/text/language.h"

// Entry points of the grammar libraries, which ship no header.
extern "C" {
const TSLanguage* tree_sitter_c(void);
const TSLanguage* tree_sitter_cpp(void);
const TSLanguage* tree_sitter_python(void);
}

namespace ferret {
namespace {

// The capture roles and predicates of the query contract (symbol_extractor.h).

struct NameRole {};

struct SymbolRole {
  SymbolKind kind;
  bool is_definition;
};

enum class ScopeRole : uint8_t { kLocal, kClass, kNamespace };

using CaptureRole = std::variant<NameRole, SymbolRole, ScopeRole>;

enum class NamePredicate : uint8_t {
  kDeclaresFunction,
  kNotDeclaresFunction,
  kIsQualified,
};

struct SymbolRule {
  SymbolRole role;
  absl::InlinedVector<NamePredicate, 1> predicates;
};

// A pattern's rule: the symbol it introduces, or the scope it marks.
using PatternRule = std::variant<SymbolRule, ScopeRole>;

struct QueryDeleter {
  void operator()(TSQuery* query) const { ts_query_delete(query); }
};

struct TreeDeleter {
  void operator()(TSTree* tree) const { ts_tree_delete(tree); }
};

// Payload of the tree-sitter progress callbacks, which cancel parsing or
// querying once the budget is spent. Tree-sitter calls them after a fixed
// number of operations, so whether an input fits depends only on the input.
struct Budget {
  int64_t checks_left;
  bool exhausted = false;

  bool Spend() {
    if (!exhausted && --checks_left < 0) exhausted = true;
    return exhausted;
  }
};

bool ParseProgress(TSParseState* state) {
  return static_cast<Budget*>(state->payload)->Spend();
}

bool QueryProgress(TSQueryCursorState* state) {
  return static_cast<Budget*>(state->payload)->Spend();
}

const char* ReadString(void* payload, uint32_t byte_index, TSPoint /*position*/,
                       uint32_t* bytes_read) {
  const absl::string_view text =
      *static_cast<const absl::string_view*>(payload);
  if (byte_index >= text.size()) {
    *bytes_read = 0;
    return "";
  }
  // Extract() bounds the input to 32 bits.
  *bytes_read = static_cast<uint32_t>(text.size() - byte_index);
  return text.data() + byte_index;
}

struct Grammar {
  const TSLanguage* language;
  absl::string_view query_file;
  absl::string_view query_source;
};

Grammar GrammarFor(Language language) {
  switch (language) {
    case Language::kC:
      return {tree_sitter_c(), "c.scm", symbols_internal::kCQuery};
    case Language::kCpp:
      return {tree_sitter_cpp(), "cpp.scm", symbols_internal::kCppQuery};
    case Language::kPython:
      return {tree_sitter_python(), "python.scm",
              symbols_internal::kPythonQuery};
    default:
      LOG(FATAL) << "no symbol grammar for " << LanguageName(language);
  }
}

std::optional<CaptureRole> ParseCaptureName(absl::string_view name) {
  if (name == "name") return NameRole{};
  if (name == "scope.local") return ScopeRole::kLocal;
  if (name == "scope.class") return ScopeRole::kClass;
  if (name == "scope.namespace") return ScopeRole::kNamespace;
  bool is_definition;
  if (absl::ConsumePrefix(&name, "definition.")) {
    is_definition = true;
  } else if (absl::ConsumePrefix(&name, "declaration.")) {
    is_definition = false;
  } else {
    return std::nullopt;
  }
  const std::optional<SymbolKind> kind = ParseSymbolKind(name);
  if (!kind.has_value()) return std::nullopt;
  return SymbolRole{*kind, is_definition};
}

// A tree node's type as a string_view.
absl::string_view Type(TSNode node) { return ts_node_type(node); }

TSNode Field(TSNode node, absl::string_view name) {
  return ts_node_child_by_field_name(node, name.data(),
                                     static_cast<uint32_t>(name.size()));
}

TSNode LastNamedChild(TSNode node) {
  const uint32_t count = ts_node_named_child_count(node);
  return count == 0 ? TSNode{} : ts_node_named_child(node, count - 1);
}

// The name a declarator declares, and what the declarator says about it.
struct DeclaredName {
  uint32_t start = 0;  // Byte range of the name.
  uint32_t end = 0;
  uint32_t row = 0;  // 0-based.
  bool declares_function = false;
  bool is_qualified = false;
};

// A conversion operator's node spans `operator const char*() const`; its name
// ends where the parameter list begins.
uint32_t ConversionOperatorNameEnd(TSNode node) {
  for (TSNode declarator = Field(node, "declarator");
       !ts_node_is_null(declarator);
       declarator = Field(declarator, "declarator")) {
    if (Type(declarator) != "abstract_function_declarator") continue;
    const TSNode parameters = Field(declarator, "parameters");
    if (!ts_node_is_null(parameters)) return ts_node_start_byte(parameters);
  }
  return ts_node_end_byte(node);
}

// Unwraps a @name capture down to the name it declares, or returns nullopt if
// the node has an unexpected shape (as in error recovery) or the name is
// empty (a node tree-sitter inserted as missing). Iterative, so the cost is
// the depth of the declarator.
std::optional<DeclaredName> UnwrapName(TSNode node) {
  DeclaredName name;
  while (!ts_node_is_null(node)) {
    const absl::string_view type = Type(node);
    if (type == "function_declarator") {
      name.declares_function = true;
      node = Field(node, "declarator");
    } else if (type == "pointer_declarator" || type == "array_declarator") {
      name.declares_function = false;
      node = Field(node, "declarator");
    } else if (type == "reference_declarator") {
      name.declares_function = false;
      node = LastNamedChild(node);
    } else if (type == "parenthesized_declarator") {
      // The last child, after an optional calling convention: `(__cdecl *f)`.
      node = LastNamedChild(node);
    } else if (type == "attributed_declarator") {
      node = ts_node_named_child(node, 0);
    } else if (type == "init_declarator") {
      node = Field(node, "declarator");
    } else if (type == "qualified_identifier") {
      name.is_qualified = true;
      node = Field(node, "name");
    } else if (type == "template_function" || type == "template_method" ||
               type == "template_type") {
      node = Field(node, "name");
    } else if (type == "operator_cast" || type == "identifier" ||
               type == "field_identifier" || type == "type_identifier" ||
               type == "primitive_type" || type == "destructor_name" ||
               type == "operator_name") {
      name.start = ts_node_start_byte(node);
      name.end = type == "operator_cast" ? ConversionOperatorNameEnd(node)
                                         : ts_node_end_byte(node);
      name.row = ts_node_start_point(node).row;
      // A conversion operator is a function; its declarator is built in.
      if (type == "operator_cast") name.declares_function = true;
      if (name.end <= name.start) return std::nullopt;
      return name;
    } else {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

bool Satisfies(absl::Span<const NamePredicate> predicates,
               const DeclaredName& name) {
  for (const NamePredicate predicate : predicates) {
    switch (predicate) {
      case NamePredicate::kDeclaresFunction:
        if (!name.declares_function) return false;
        break;
      case NamePredicate::kNotDeclaresFunction:
        if (name.declares_function) return false;
        break;
      case NamePredicate::kIsQualified:
        if (!name.is_qualified) return false;
        break;
    }
  }
  return true;
}

bool IsIdentifierByte(char c) {
  return absl::ascii_isalnum(static_cast<unsigned char>(c)) || c == '_';
}

// Drops whitespace inside a name unless it separates two identifier
// characters, where it becomes one space: `operator ==` is "operator==", and
// `operator  new [ ]` is "operator new[]".
std::string NormalizeName(absl::string_view text) {
  std::string name;
  name.reserve(text.size());
  bool pending_space = false;
  for (const char c : text) {
    if (absl::ascii_isspace(static_cast<unsigned char>(c))) {
      pending_space = true;
      continue;
    }
    if (pending_space && !name.empty() && IsIdentifierByte(name.back()) &&
        IsIdentifierByte(c)) {
      name.push_back(' ');
    }
    pending_space = false;
    name.push_back(c);
  }
  return name;
}

// A symbol pattern's match, before scopes are applied.
struct Candidate {
  DeclaredName name;
  uint32_t pattern;
  SymbolRole role;
};

struct Scope {
  uint32_t start;
  uint32_t end;
  ScopeRole role;
};

// Applies the scope rules and the first-pattern-wins rule to `candidates` and
// returns the resulting symbols in name order. Both inputs are sorted in
// place. Scopes nest because they are tree nodes, so a sweep in position
// order with a stack of open scopes finds each name's innermost scope.
std::vector<ExtractedSymbol> ResolveSymbols(absl::string_view content,
                                            std::vector<Candidate>& candidates,
                                            std::vector<Scope>& scopes) {
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              if (a.name.start != b.name.start) {
                return a.name.start < b.name.start;
              }
              return a.pattern < b.pattern;
            });
  // Outer scopes first among those starting at the same byte.
  std::sort(scopes.begin(), scopes.end(), [](const Scope& a, const Scope& b) {
    if (a.start != b.start) return a.start < b.start;
    return a.end > b.end;
  });

  std::vector<ExtractedSymbol> symbols;
  std::vector<const Scope*> open;
  int open_locals = 0;
  const auto close_innermost = [&open, &open_locals] {
    if (open.back()->role == ScopeRole::kLocal) --open_locals;
    open.pop_back();
  };
  size_t next_scope = 0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    const Candidate& candidate = candidates[i];
    const uint32_t position = candidate.name.start;
    if (i > 0 && candidates[i - 1].name.start == position) continue;
    for (; next_scope < scopes.size() && scopes[next_scope].start <= position;
         ++next_scope) {
      const Scope& scope = scopes[next_scope];
      while (!open.empty() && open.back()->end <= scope.start) {
        close_innermost();
      }
      open.push_back(&scope);
      if (scope.role == ScopeRole::kLocal) ++open_locals;
    }
    while (!open.empty() && open.back()->end <= position) close_innermost();
    if (open_locals > 0) continue;

    SymbolKind kind = candidate.role.kind;
    if (!open.empty() && open.back()->role == ScopeRole::kClass) {
      if (kind == SymbolKind::kVariable) continue;
      if (kind == SymbolKind::kFunction) kind = SymbolKind::kMethod;
    }
    symbols.push_back({
        .name = NormalizeName(
            content.substr(position, candidate.name.end - position)),
        .kind = kind,
        .is_definition = candidate.role.is_definition,
        .line = candidate.name.row + 1,
    });
  }
  return symbols;
}

}  // namespace

struct SymbolExtractor::LanguageQuery {
  // Compiles the embedded query for `language` and checks it against the
  // query contract, CHECK-failing on a violation.
  explicit LanguageQuery(Language language);

  const TSLanguage* ts_language;
  std::unique_ptr<TSQuery, QueryDeleter> query;
  uint32_t name_capture = 0;
  std::vector<PatternRule> rules;  // Indexed by pattern.
};

SymbolExtractor::LanguageQuery::LanguageQuery(Language language) {
  const Grammar grammar = GrammarFor(language);
  const absl::string_view file = grammar.query_file;
  ts_language = grammar.language;

  uint32_t error_offset = 0;
  TSQueryError error = TSQueryErrorNone;
  query.reset(ts_query_new(grammar.language, grammar.query_source.data(),
                           static_cast<uint32_t>(grammar.query_source.size()),
                           &error_offset, &error));
  CHECK(query != nullptr) << file << ": tree-sitter query error " << error
                          << " at byte " << error_offset << ": "
                          << grammar.query_source.substr(error_offset, 40);

  std::vector<CaptureRole> captures;
  std::optional<uint32_t> name_id;
  for (uint32_t id = 0; id < ts_query_capture_count(query.get()); ++id) {
    uint32_t length = 0;
    const absl::string_view name(
        ts_query_capture_name_for_id(query.get(), id, &length), length);
    const std::optional<CaptureRole> role = ParseCaptureName(name);
    CHECK(role.has_value()) << file << ": unknown capture @" << name;
    if (std::holds_alternative<NameRole>(*role)) name_id = id;
    captures.push_back(*role);
  }
  CHECK(name_id.has_value()) << file << ": no @name capture";
  name_capture = *name_id;

  for (uint32_t pattern = 0; pattern < ts_query_pattern_count(query.get());
       ++pattern) {
    const uint32_t offset =
        ts_query_start_byte_for_pattern(query.get(), pattern);
    std::optional<SymbolRole> symbol;
    std::optional<ScopeRole> scope;
    bool has_name = false;
    for (uint32_t id = 0; id < captures.size(); ++id) {
      const TSQuantifier quantifier =
          ts_query_capture_quantifier_for_id(query.get(), pattern, id);
      if (quantifier == TSQuantifierZero) continue;
      CHECK_EQ(quantifier, TSQuantifierOne)
          << file << ": pattern at byte " << offset
          << " must capture each node exactly once";
      if (const auto* symbol_role = std::get_if<SymbolRole>(&captures[id])) {
        CHECK(!symbol.has_value()) << file << ": pattern at byte " << offset
                                   << " introduces two symbols";
        symbol = *symbol_role;
      } else if (const auto* scope_role =
                     std::get_if<ScopeRole>(&captures[id])) {
        CHECK(!scope.has_value())
            << file << ": pattern at byte " << offset << " marks two scopes";
        scope = *scope_role;
      } else {
        has_name = true;
      }
    }

    uint32_t step_count = 0;
    const TSQueryPredicateStep* steps =
        ts_query_predicates_for_pattern(query.get(), pattern, &step_count);
    absl::InlinedVector<NamePredicate, 1> predicates;
    // Each predicate is three steps: `#operator?`, `@name`, and the end.
    for (uint32_t i = 0; i < step_count; i += 3) {
      CHECK(i + 2 < step_count &&
            steps[i].type == TSQueryPredicateStepTypeString &&
            steps[i + 1].type == TSQueryPredicateStepTypeCapture &&
            steps[i + 1].value_id == name_capture &&
            steps[i + 2].type == TSQueryPredicateStepTypeDone)
          << file << ": pattern at byte " << offset
          << " has a predicate that is not (#operator? @name)";
      uint32_t length = 0;
      const absl::string_view op(
          ts_query_string_value_for_id(query.get(), steps[i].value_id, &length),
          length);
      if (op == "declares-function?") {
        predicates.push_back(NamePredicate::kDeclaresFunction);
      } else if (op == "not-declares-function?") {
        predicates.push_back(NamePredicate::kNotDeclaresFunction);
      } else if (op == "is-qualified?") {
        predicates.push_back(NamePredicate::kIsQualified);
      } else {
        LOG(FATAL) << file << ": pattern at byte " << offset
                   << " has unknown predicate #" << op;
      }
    }

    if (scope.has_value()) {
      CHECK(!symbol.has_value() && !has_name && predicates.empty())
          << file << ": scope pattern at byte " << offset
          << " must capture only the scope";
      rules.push_back(*scope);
    } else {
      CHECK(symbol.has_value() && has_name)
          << file << ": pattern at byte " << offset
          << " must capture a symbol and its @name";
      rules.push_back(SymbolRule{*symbol, std::move(predicates)});
    }
  }
}

SymbolExtractor::SymbolExtractor(SymbolExtractorOptions options)
    : options_(options),
      parser_(ts_parser_new()),
      cursor_(ts_query_cursor_new()) {}

SymbolExtractor::~SymbolExtractor() = default;

const SymbolExtractor::LanguageQuery& SymbolExtractor::QueryFor(
    Language language) {
  std::unique_ptr<const LanguageQuery>* slot = nullptr;
  switch (language) {
    case Language::kC:
      slot = &c_query_;
      break;
    case Language::kCpp:
      slot = &cpp_query_;
      break;
    case Language::kPython:
      slot = &python_query_;
      break;
    default:
      LOG(FATAL) << "no symbol grammar for " << LanguageName(language);
  }
  if (*slot == nullptr) *slot = std::make_unique<const LanguageQuery>(language);
  return **slot;
}

absl::StatusOr<std::vector<ExtractedSymbol>> SymbolExtractor::Extract(
    Language language, absl::string_view content) {
  if (!HasSymbolGrammar(language)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "no symbol grammar for language ", LanguageName(language)));
  }
  const int64_t max_bytes = std::min<int64_t>(
      options_.max_input_bytes, std::numeric_limits<uint32_t>::max());
  if (static_cast<int64_t>(content.size()) > max_bytes) {
    return absl::ResourceExhaustedError(absl::StrCat(
        LanguageName(language), " input of ", content.size(),
        " bytes exceeds the symbol extraction limit of ", max_bytes, " bytes"));
  }
  const LanguageQuery& query = QueryFor(language);
  Budget budget{.checks_left = options_.max_progress_checks};
  const auto over_budget = [&](absl::string_view phase) {
    return absl::ResourceExhaustedError(absl::StrCat(
        phase, " ", content.size(), " bytes of ", LanguageName(language),
        " exceeded the symbol extraction budget of ",
        options_.max_progress_checks, " progress checks"));
  };

  const absl::string_view source =
      language == Language::kCpp
          ? symbols_internal::BlankExportMacros(content, blanked_)
          : content;

  if (ts_parser_language(parser_.get()) != query.ts_language) {
    CHECK(ts_parser_set_language(parser_.get(), query.ts_language))
        << LanguageName(language) << " grammar ABI version "
        << ts_language_abi_version(query.ts_language)
        << " is not supported by the tree-sitter runtime";
  }
  absl::string_view input_text = source;
  const TSInput input = {
      .payload = &input_text,
      .read = ReadString,
      .encoding = TSInputEncodingUTF8,
      .decode = nullptr,
  };
  const TSParseOptions parse_options = {.payload = &budget,
                                        .progress_callback = ParseProgress};
  const std::unique_ptr<TSTree, TreeDeleter> tree(ts_parser_parse_with_options(
      parser_.get(), /*old_tree=*/nullptr, input, parse_options));
  if (tree == nullptr) {
    // Only cancellation fails a parse. Without a reset, the next parse would
    // resume this one.
    ts_parser_reset(parser_.get());
    return over_budget("parsing");
  }

  const TSQueryCursorOptions cursor_options = {
      .payload = &budget, .progress_callback = QueryProgress};
  ts_query_cursor_exec_with_options(cursor_.get(), query.query.get(),
                                    ts_tree_root_node(tree.get()),
                                    &cursor_options);
  std::vector<Candidate> candidates;
  std::vector<Scope> scopes;
  TSQueryMatch match;
  while (ts_query_cursor_next_match(cursor_.get(), &match)) {
    const PatternRule& rule = query.rules[match.pattern_index];
    if (const auto* scope = std::get_if<ScopeRole>(&rule)) {
      const TSNode node = match.captures[0].node;
      scopes.push_back(
          {ts_node_start_byte(node), ts_node_end_byte(node), *scope});
      continue;
    }
    const auto& symbol = std::get<SymbolRule>(rule);
    // An invalid declaration can contain a synthetic variable declarator
    // assembled from annotation tokens preceding a real function definition.
    if (symbol.role.kind == SymbolKind::kVariable) {
      bool broken = false;
      for (uint16_t i = 0; i < match.capture_count; ++i) {
        if (match.captures[i].index != query.name_capture &&
            ts_node_has_error(match.captures[i].node)) {
          broken = true;
        }
      }
      if (broken) continue;
    }
    for (uint16_t i = 0; i < match.capture_count; ++i) {
      if (match.captures[i].index != query.name_capture) continue;
      const std::optional<DeclaredName> name =
          UnwrapName(match.captures[i].node);
      if (name.has_value() && Satisfies(symbol.predicates, *name)) {
        candidates.push_back({*name, match.pattern_index, symbol.role});
      }
      break;
    }
  }
  if (budget.exhausted) return over_budget("querying");
  return ResolveSymbols(content, candidates, scopes);
}

}  // namespace ferret

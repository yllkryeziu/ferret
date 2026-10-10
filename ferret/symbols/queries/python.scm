; Copyright 2026 The Ferret Authors.
; SPDX-License-Identifier: Apache-2.0
;
; Symbols of Python source, run by SymbolExtractor over a
; tree-sitter-python tree. The capture and predicate vocabulary is defined in
; symbol_extractor.h. When several patterns capture the same name, the first
; one listed wins.
;
; Functions nested in functions are local (@scope.local): closures such as a
; decorator's `wrapper` are implementation details that would flood name
; lookups. Nested classes and their methods are reported. Decorators do not
; change what is reported; the line is that of the `def` or `class` name.

(function_definition body: (_) @scope.local)
(class_definition body: (_) @scope.class)

; Functions, and methods when directly in a class body (`async def` included).

(function_definition
  name: (identifier) @name) @definition.function

(class_definition
  name: (identifier) @name) @definition.class

; Module-level names. Only an assignment to a single name binds a symbol; an
; annotation without a value (`x: int`) only declares one. Assignments inside
; `if`, `try`, and other blocks are not reported.

(module
  (expression_statement
    (assignment
      left: (identifier) @name
      right: (_))) @definition.variable)

(module
  (expression_statement
    (assignment
      left: (identifier) @name
      !right)) @declaration.variable)

(type_alias_statement
  left: (type (identifier) @name)) @definition.type_alias

(type_alias_statement
  left: (type (generic_type (identifier) @name))) @definition.type_alias

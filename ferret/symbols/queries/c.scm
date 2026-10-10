; Copyright 2026 The Ferret Authors.
; SPDX-License-Identifier: Apache-2.0
;
; Symbols of C source, run by SymbolExtractor over a
; tree-sitter-c tree. The capture and predicate vocabulary is defined in
; symbol_extractor.h. When several patterns capture the same name, the first
; one listed wins.

(function_definition body: (_) @scope.local)

; K&R parameter declarations sit between the declarator and the body.
(function_definition (declaration) @scope.local)

; Functions.

(function_definition
  declarator: (_) @name) @definition.function

(declaration
  declarator: (_) @name
  (#declares-function? @name)) @declaration.function

; Variables at file scope. An `extern` declaration declares; any other one,
; with or without an initializer, defines (a tentative definition in C). An
; initialized `extern` is also a definition in C, but compilers warn about it
; and code avoids it, so it is not told apart.

(declaration
  (storage_class_specifier "extern")
  declarator: (_) @name
  (#not-declares-function? @name)) @declaration.variable

(declaration
  declarator: (_) @name
  (#not-declares-function? @name)) @definition.variable

; Types. Only a specifier with a body defines a type; `struct stat *st` and
; `struct list;` merely name one.

(struct_specifier
  name: (_) @name
  body: (_)) @definition.struct

(union_specifier
  name: (_) @name
  body: (_)) @definition.union

(enum_specifier
  name: (_) @name
  body: (_)) @definition.enum

(enumerator
  name: (_) @name) @definition.enumerator

(type_definition
  declarator: (_) @name) @definition.type_alias

; Macros.

(preproc_def
  name: (_) @name) @definition.macro

(preproc_function_def
  name: (_) @name) @definition.macro

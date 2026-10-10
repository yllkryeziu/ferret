; Copyright 2026 The Ferret Authors.
; SPDX-License-Identifier: Apache-2.0
;
; Symbols of C++ source, run by SymbolExtractor over a
; tree-sitter-cpp tree. The capture and predicate vocabulary is defined in
; symbol_extractor.h. When several patterns capture the same name, the first
; one listed wins.
;
; Unlike upstream tags.scm, which tags every function declarator as a
; definition, a prototype or an in-class member declaration is a declaration
; here; only a function with a body (or `= default`, `= delete`) defines.

(function_definition body: (_) @scope.local)
(lambda_expression body: (_) @scope.local)
(field_declaration_list) @scope.class
(friend_declaration) @scope.namespace

; Functions and methods. Inside a class body (@scope.class) every function is
; a method: in-class definitions, and constructors and destructors, which
; parse as plain declarations. Out of line, a method is defined under its
; qualified name: `void Foo::Bar() {}`. A free function defined under a
; namespace-qualified name (`void ns::Free() {}`) is indistinguishable in the
; syntax tree and is reported as a method too.

(function_definition
  declarator: (_) @name
  (#is-qualified? @name)) @definition.method

(function_definition
  declarator: (_) @name) @definition.function

(declaration
  declarator: (_) @name
  (#declares-function? @name)) @declaration.function

(field_declaration
  declarator: (_) @name
  (#declares-function? @name)) @declaration.method

; Variables at namespace scope. `extern` and a single-declaration
; `extern "C"` declare; any other declaration defines. An initialized
; `extern` is also a definition, but compilers warn about it and code avoids
; it, so it is not told apart.

(declaration
  (storage_class_specifier "extern")
  declarator: (_) @name
  (#not-declares-function? @name)) @declaration.variable

(linkage_specification
  body: (declaration
    declarator: (_) @name
    (#not-declares-function? @name)) @declaration.variable)

(declaration
  declarator: (_) @name
  (#not-declares-function? @name)) @definition.variable

; Types. Only a specifier with a body defines a type; `class Foo;` and
; `struct stat* st` merely name one. Specializations (`class Vec<bool> {}`)
; and out-of-line nested classes (`class Foo::Impl {}`) define the
; unqualified template or class name.

(class_specifier
  name: (_) @name
  body: (_)) @definition.class

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

(alias_declaration
  name: (_) @name) @definition.type_alias

; Macros.

(preproc_def
  name: (_) @name) @definition.macro

(preproc_function_def
  name: (_) @name) @definition.macro

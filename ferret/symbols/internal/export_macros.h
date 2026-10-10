// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Blanks attribute macros between a C++ class-key and the class name.
//
// Chromium declares exported classes as
// `class CONTENT_EXPORT WebContents : public PageNavigator {`. Without a
// preprocessor, tree-sitter-cpp takes CONTENT_EXPORT for the class name and
// parses the rest as a function definition, so the class and all its members
// are lost. The same shape appears in V8 (V8_EXPORT), libc++
// (_LIBCPP_TEMPLATE_VIS), Abseil (ABSL_MUST_USE_RESULT), and Qt
// (Q_CORE_EXPORT).
//
// A macro is recognized by its uppercase-and-underscore shape (or as
// `__attribute__` or `__declspec`), optionally followed by a short
// parenthesized argument list. It must follow `class`, `struct`, or `union`
// and precede another identifier. An uppercase class name is kept
// (`class FOO {`, `class FOO final`), as are template parameters and ordinary
// identifiers in comments and string literals. An elaborated declaration
// such as `struct FOO_BAR field;` remains ambiguous without a preprocessor.

#ifndef FERRET_SYMBOLS_INTERNAL_EXPORT_MACROS_H_
#define FERRET_SYMBOLS_INTERNAL_EXPORT_MACROS_H_

#include <string>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"

namespace ferret::symbols_internal {

// Returns `content` with every such macro, and its argument list, replaced by
// spaces. Newlines are kept, so every byte offset and line number is
// unchanged. When there is nothing to blank, returns `content` itself without
// copying; otherwise the copy is made in `buffer`. The result is valid while
// both `content` and `buffer` are alive and unmodified. O(|content|).
absl::string_view BlankExportMacros(
    absl::string_view content ABSL_ATTRIBUTE_LIFETIME_BOUND,
    std::string& buffer ABSL_ATTRIBUTE_LIFETIME_BOUND);

}  // namespace ferret::symbols_internal

#endif  // FERRET_SYMBOLS_INTERNAL_EXPORT_MACROS_H_

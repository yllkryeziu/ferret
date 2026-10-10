// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Source-language detection from a path and the first bytes of content.
//
// Language drives symbol extraction (tree-sitter grammar choice), the match
// classification lexer, and ranking priors. Detection is by extension, with a
// content heuristic for `.h`, which is C in Linux (27k files) but C++ in
// Chromium (62k files).

#ifndef FERRET_TEXT_LANGUAGE_H_
#define FERRET_TEXT_LANGUAGE_H_

#include <cstdint>

#include "absl/strings/string_view.h"

namespace ferret {

// Persisted in .fseg: append only, never renumber.
enum class Language : uint8_t {
  kUnknown = 0,
  kC = 1,
  kCpp = 2,
  kObjectiveCpp = 3,
  kPython = 4,
  kGo = 5,
  kJava = 6,
  kJavaScript = 7,
  kTypeScript = 8,
  kRust = 9,
  kShell = 10,
  kMarkup = 11,  // HTML, XML, Markdown: comments and strings still matter.
};

// Returns a stable lowercase name such as "cpp"; "unknown" for kUnknown.
absl::string_view LanguageName(Language language);

// Detects the language of the file at repo-relative `path`. `content_prefix`
// is the start of the file (a few KiB suffice) and is consulted only for
// ambiguous extensions such as `.h`.
Language DetectLanguage(absl::string_view path,
                        absl::string_view content_prefix);

// True for languages with a tree-sitter grammar in this build (C, C++,
// Python). Objective-C++ is deliberately excluded: tree-sitter-cpp cannot
// parse it.
bool HasSymbolGrammar(Language language);

}  // namespace ferret

#endif  // FERRET_TEXT_LANGUAGE_H_

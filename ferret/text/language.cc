// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/language.h"

#include <cstddef>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"

namespace ferret {
namespace {

struct ExtensionEntry {
  absl::string_view extension;  // Lowercase, without the dot.
  Language language;
};

// Sorted only for readability; lookup is linear over a small table.
constexpr ExtensionEntry kExtensions[] = {
    {"bash", Language::kShell},     {"c", Language::kC},
    {"cc", Language::kCpp},         {"cjs", Language::kJavaScript},
    {"cpp", Language::kCpp},        {"cts", Language::kTypeScript},
    {"cxx", Language::kCpp},        {"go", Language::kGo},
    {"h++", Language::kCpp},        {"hh", Language::kCpp},
    {"hpp", Language::kCpp},        {"htm", Language::kMarkup},
    {"html", Language::kMarkup},    {"hxx", Language::kCpp},
    {"inc", Language::kCpp},        {"inl", Language::kCpp},
    {"ipp", Language::kCpp},        {"java", Language::kJava},
    {"js", Language::kJavaScript},  {"jsx", Language::kJavaScript},
    {"m", Language::kObjectiveCpp}, {"md", Language::kMarkup},
    {"mjs", Language::kJavaScript}, {"mm", Language::kObjectiveCpp},
    {"mts", Language::kTypeScript}, {"py", Language::kPython},
    {"pyi", Language::kPython},     {"rs", Language::kRust},
    {"sh", Language::kShell},       {"ts", Language::kTypeScript},
    {"tsx", Language::kTypeScript}, {"xml", Language::kMarkup},
};

absl::string_view Extension(absl::string_view path) {
  const size_t slash = path.rfind('/');
  const absl::string_view base =
      slash == absl::string_view::npos ? path : path.substr(slash + 1);
  const size_t dot = base.rfind('.');
  if (dot == absl::string_view::npos || dot == 0) return {};
  return base.substr(dot + 1);
}

// A `.h` file is C++ if its prefix contains a construct C cannot have.
bool LooksLikeCpp(absl::string_view prefix) {
  constexpr absl::string_view kCppMarkers[] = {
      "namespace ", "class ",   "template <",        "template<",
      "public:",    "private:", "#include <string>", "std::",
      "constexpr ", "nullptr",
  };
  for (absl::string_view marker : kCppMarkers) {
    if (absl::StrContains(prefix, marker)) return true;
  }
  return false;
}

}  // namespace

absl::string_view LanguageName(Language language) {
  switch (language) {
    case Language::kUnknown:
      return "unknown";
    case Language::kC:
      return "c";
    case Language::kCpp:
      return "cpp";
    case Language::kObjectiveCpp:
      return "objective-cpp";
    case Language::kPython:
      return "python";
    case Language::kGo:
      return "go";
    case Language::kJava:
      return "java";
    case Language::kJavaScript:
      return "javascript";
    case Language::kTypeScript:
      return "typescript";
    case Language::kRust:
      return "rust";
    case Language::kShell:
      return "shell";
    case Language::kMarkup:
      return "markup";
  }
  return "unknown";
}

Language DetectLanguage(absl::string_view path,
                        absl::string_view content_prefix) {
  const absl::string_view extension = Extension(path);
  if (extension.empty()) return Language::kUnknown;
  if (absl::EqualsIgnoreCase(extension, "h")) {
    return LooksLikeCpp(content_prefix) ? Language::kCpp : Language::kC;
  }
  for (const ExtensionEntry& entry : kExtensions) {
    if (absl::EqualsIgnoreCase(extension, entry.extension)) {
      return entry.language;
    }
  }
  return Language::kUnknown;
}

bool HasSymbolGrammar(Language language) {
  return language == Language::kC || language == Language::kCpp ||
         language == Language::kPython;
}

}  // namespace ferret

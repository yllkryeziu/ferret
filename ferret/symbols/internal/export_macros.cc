// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/symbols/internal/export_macros.h"

#include <algorithm>
#include <cstddef>
#include <string>

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"

namespace ferret::symbols_internal {
namespace {

// Macro arguments are short (`COMPONENT_EXPORT(NETWORK_CPP_BASE)`). Bounding
// the search for the closing parenthesis keeps the scan linear on input with
// many unclosed ones.
constexpr size_t kMaxArgumentBytes = 128;

bool IsWordByte(char c) {
  return absl::ascii_isalnum(static_cast<unsigned char>(c)) || c == '_';
}

// Returns the end of the run of word bytes at `pos`, which is `pos` itself if
// there is none.
size_t WordEnd(absl::string_view text, size_t pos) {
  while (pos < text.size() && IsWordByte(text[pos])) ++pos;
  return pos;
}

size_t SkipSpace(absl::string_view text, size_t pos) {
  while (pos < text.size() &&
         absl::ascii_isspace(static_cast<unsigned char>(text[pos]))) {
    ++pos;
  }
  return pos;
}

// Returns the position just past the parenthesis that closes the one at
// `open`, or npos if it does not close within kMaxArgumentBytes.
size_t SkipArguments(absl::string_view text, size_t open) {
  const size_t limit = std::min(text.size(), open + kMaxArgumentBytes);
  int depth = 0;
  for (size_t pos = open; pos < limit; ++pos) {
    if (text[pos] == '(') {
      ++depth;
    } else if (text[pos] == ')' && --depth == 0) {
      return pos + 1;
    }
  }
  return absl::string_view::npos;
}

// Skips literals and comments so their text cannot be mistaken for C++ code.
size_t SkipNonCode(absl::string_view text, size_t pos) {
  if (text[pos] == '/' && pos + 1 < text.size()) {
    if (text[pos + 1] == '/') {
      const size_t end = text.find('\n', pos + 2);
      return end == absl::string_view::npos ? text.size() : end;
    }
    if (text[pos + 1] == '*') {
      const size_t end = text.find("*/", pos + 2);
      return end == absl::string_view::npos ? text.size() : end + 2;
    }
  }
  if (text[pos] == 'R' && pos + 1 < text.size() && text[pos + 1] == '"') {
    const size_t open = text.find('(', pos + 2);
    if (open != absl::string_view::npos && open - pos <= 18) {
      std::string close = ")";
      close.append(text.substr(pos + 2, open - pos - 2));
      close.push_back('"');
      const size_t end = text.find(close, open + 1);
      return end == absl::string_view::npos ? text.size() : end + close.size();
    }
  }
  if (text[pos] == '"' || text[pos] == '\'') {
    const char quote = text[pos++];
    while (pos < text.size()) {
      if (text[pos] == '\\') {
        pos = std::min(pos + 2, text.size());
      } else if (text[pos++] == quote) {
        break;
      }
    }
    return pos;
  }
  return pos;
}

bool IsMacroName(absl::string_view word) {
  if (word == "__attribute__" || word == "__declspec") return true;
  if (word.empty() ||
      absl::ascii_isdigit(static_cast<unsigned char>(word[0]))) {
    return false;
  }
  bool has_letter = false;
  for (const char c : word) {
    const auto byte = static_cast<unsigned char>(c);
    if (absl::ascii_isupper(byte)) {
      has_letter = true;
    } else if (!absl::ascii_isdigit(byte) && c != '_') {
      return false;
    }
  }
  return has_letter && absl::StrContains(word, '_');
}

// True if `word` can be the class name that follows a macro.
bool IsClassName(absl::string_view word) {
  return !word.empty() &&
         !absl::ascii_isdigit(static_cast<unsigned char>(word[0])) &&
         word != "final";
}

}  // namespace

absl::string_view BlankExportMacros(absl::string_view content,
                                    std::string& buffer) {
  bool copied = false;
  size_t pos = 0;
  while (pos < content.size()) {
    const size_t skipped = SkipNonCode(content, pos);
    if (skipped != pos) {
      pos = skipped;
      continue;
    }
    if (!IsWordByte(content[pos])) {
      ++pos;
      continue;
    }
    const size_t word_end = WordEnd(content, pos);
    const absl::string_view word = content.substr(pos, word_end - pos);
    pos = word_end;
    if (word != "class" && word != "struct" && word != "union") continue;
    // There may be several macros: `class ABSL_MUST_USE_RESULT FOO_EXPORT X`.
    for (;;) {
      const size_t macro_begin = SkipSpace(content, pos);
      size_t macro_end = WordEnd(content, macro_begin);
      if (!IsMacroName(content.substr(macro_begin, macro_end - macro_begin))) {
        break;
      }
      size_t next = SkipSpace(content, macro_end);
      if (next < content.size() && content[next] == '(') {
        macro_end = SkipArguments(content, next);
        if (macro_end == absl::string_view::npos) break;
        next = SkipSpace(content, macro_end);
      }
      const size_t class_end = WordEnd(content, next);
      if (!IsClassName(content.substr(next, class_end - next))) break;
      // An elaborated type in a variable declaration is not an exported class.
      const size_t after_name = SkipSpace(content, class_end);
      const bool next_is_macro =
          IsMacroName(content.substr(next, class_end - next));
      if (!next_is_macro &&
          (after_name == content.size() ||
           (content[after_name] != '{' && content[after_name] != ':' &&
            content[after_name] != ';' &&
            content.substr(after_name, 5) != "final"))) {
        break;
      }
      if (!copied) {
        buffer.assign(content.data(), content.size());
        copied = true;
      }
      for (size_t i = macro_begin; i < macro_end; ++i) {
        if (buffer[i] != '\n') buffer[i] = ' ';
      }
      pos = next;
    }
  }
  return copied ? absl::string_view(buffer) : content;
}

}  // namespace ferret::symbols_internal

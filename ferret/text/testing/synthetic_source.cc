// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/testing/synthetic_source.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <random>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace ferret_testing {
namespace {

constexpr absl::string_view kWords[] = {
    "if",          "for",         "return",      "const",    "auto",
    "int",         "void",        "static",      "struct",   "nullptr",
    "std::string", "std::vector", "size_t",      "uint32_t", "bool",
    "true",        "false",       "else",        "while",    "namespace",
    "base",        "content",     "WebContents", "GetValue", "Profile",
    "callback",    "delegate",    "request",     "response", "buffer",
    "length",      "offset",      "index",       "count",    "result",
    "status",      "kMaxSize",    "DCHECK",      "CHECK_EQ", "LOG(ERROR)",
    "->",          "::",          "(",           ")",        "{",
    "}",           ";",           ",",           "=",        "==",
    "!=",          "<",           ">",           "&&",       "||",
    "+",           "-",           "*",           "&",        "0",
    "1",           "42",          "\"text\"",    "'c'",      "[i]",
};

// A bounded draw that does not depend on the standard library's
// distributions, which differ between implementations.
size_t Draw(std::mt19937_64& rng, size_t bound) {
  return static_cast<size_t>(rng() % bound);
}

}  // namespace

std::string SyntheticSource(int64_t size, uint64_t seed) {
  std::mt19937_64 rng(seed);
  const size_t target = static_cast<size_t>(size);
  std::string text;
  text.reserve(target + 128);
  while (text.size() < target) {
    const size_t line_start = text.size();
    text.append(2 * Draw(rng, 4), ' ');
    const size_t length = 20 + Draw(rng, 81);
    if (Draw(rng, 8) == 0) {
      text.append("// ");
    }
    while (text.size() - line_start < length) {
      absl::StrAppend(&text, kWords[Draw(rng, std::size(kWords))], " ");
    }
    text.back() = '\n';
  }
  text.resize(target);
  return text;
}

}  // namespace ferret_testing

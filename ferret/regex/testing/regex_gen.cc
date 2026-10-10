// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/regex/testing/regex_gen.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <random>
#include <string>
#include <utility>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "re2/regexp.h"
#include "util/utf.h"

namespace ferret_testing {
namespace {

// A draw in [0, n). Hand-written because standard distributions differ across
// library implementations; the modulo bias is irrelevant here.
size_t Uniform(std::mt19937_64& rng, size_t n) {
  return static_cast<size_t>(rng() % n);
}

bool Chance(std::mt19937_64& rng, int percent) {
  return Uniform(rng, 100) < static_cast<size_t>(percent);
}

template <typename T>
const T& Pick(std::mt19937_64& rng, absl::Span<const T> items) {
  return items[Uniform(rng, items.size())];
}

// Runes that stress folding and UTF-8: every fold orbit that reaches an ASCII
// letter (k/K/U+212A, s/S/U+017F), orbits F does not fold (é/É, σ/ς/Σ, the
// Turkish dotted and dotless i), and one sequence of each UTF-8 length.
constexpr re2::Rune kRunes[] = {
    'a',    'b',    'c',    'k',    'K',    's',     'S',    'x',
    'y',    'z',    'A',    'B',    '0',    '9',     '_',    '-',
    ' ',    0x00E9, 0x00C9, 0x00DF, 0x0130, 0x0131,  0x03C3, 0x03C2,
    0x03A3, 0x212A, 0x017F, 0x65E5, 0xFFFD, 0x1F600,
};

std::string EncodeRune(re2::Rune rune) {
  char buffer[re2::UTFmax];
  const int length = re2::runetochar(buffer, &rune);
  return std::string(buffer, static_cast<size_t>(length));
}

re2::Rune RandomRune(std::mt19937_64& rng) {
  return Pick<re2::Rune>(rng, kRunes);
}

// A rune as pattern syntax: plain, or as a \x{...} escape.
std::string RuneSyntax(re2::Rune rune, std::mt19937_64& rng) {
  if (rune < 0x80 && (absl::ascii_isalnum(static_cast<unsigned char>(rune)) ||
                      rune == '_' || rune == ' ')) {
    return std::string(1, static_cast<char>(rune));
  }
  if (rune == '-') return "\\-";
  if (Chance(rng, 50)) return absl::StrFormat("\\x{%X}", rune);
  return EncodeRune(rune);
}

std::string RandomClass(std::mt19937_64& rng) {
  std::string out = "[";
  if (Chance(rng, 20)) out += '^';
  const size_t items = 1 + Uniform(rng, 3);
  for (size_t i = 0; i < items; ++i) {
    re2::Rune lo = RandomRune(rng);
    if (Chance(rng, 25)) {
      re2::Rune hi = RandomRune(rng);
      if (hi < lo) std::swap(lo, hi);
      absl::StrAppendFormat(&out, "\\x{%X}-\\x{%X}", lo, hi);
    } else {
      out += RuneSyntax(lo, rng);
    }
  }
  if (Chance(rng, 5)) out += "[:alpha:]";
  out += ']';
  return out;
}

std::string RandomExpr(std::mt19937_64& rng, int depth);

std::string RandomAtom(std::mt19937_64& rng, int depth) {
  const size_t roll = Uniform(rng, 100);
  if (roll < 40) return RuneSyntax(RandomRune(rng), rng);
  if (roll < 55) return RandomClass(rng);
  if (roll < 60) {
    static constexpr absl::string_view kPerl[] = {"\\d", "\\w", "\\s",
                                                  "\\D", "\\W", "\\S"};
    return std::string(Pick<absl::string_view>(rng, kPerl));
  }
  if (roll < 65) return ".";
  if (roll < 75 && depth > 0) {
    return absl::StrCat(Chance(rng, 50) ? "(?:" : "(",
                        RandomExpr(rng, depth - 1), ")");
  }
  if (roll < 80 && depth > 0) {
    return absl::StrCat(
        Chance(rng, 50) ? "(?i:" : "(?-i:", RandomExpr(rng, depth - 1), ")");
  }
  if (roll < 86) {
    static constexpr absl::string_view kAnchors[] = {"^",   "$",   "\\b",
                                                     "\\B", "\\A", "\\z"};
    return std::string(Pick<absl::string_view>(rng, kAnchors));
  }
  if (roll < 89) {
    std::string quoted = "\\Q";
    const size_t n = 1 + Uniform(rng, 3);
    for (size_t i = 0; i < n; ++i) quoted += Chance(rng, 30) ? "." : "k";
    return quoted + "\\E";
  }
  if (roll < 92) {
    // Braces that do not form a repetition are literals in RE2.
    static constexpr absl::string_view kBraces[] = {"{", "}", "x{", "{1",
                                                    "{,2}"};
    return std::string(Pick<absl::string_view>(rng, kBraces));
  }
  if (roll < 94) {
    static constexpr absl::string_view kUnicode[] = {"\\pL", "\\p{Greek}",
                                                     "\\pN", "\\PL"};
    return std::string(Pick<absl::string_view>(rng, kUnicode));
  }
  if (roll < 95) {
    // Atoms that NeverNL reduces to NoMatch, inside something that can still
    // match: an optional newline and an alternation with a newline branch.
    static constexpr absl::string_view kNoMatch[] = {"\\n?", "(?:\\n|x)",
                                                     "[\\n]*"};
    return std::string(Pick<absl::string_view>(rng, kNoMatch));
  }
  return EncodeRune(RandomRune(rng)) + EncodeRune(RandomRune(rng));
}

std::string RandomQuantifier(std::mt19937_64& rng) {
  static constexpr absl::string_view kQuantifiers[] = {
      "*",     "+",     "?",    "{0}",  "{1}", "{2}", "{3}", "{4}",
      "{0,1}", "{1,3}", "{2,}", "{0,}", "*?",  "+?",  "??"};
  return std::string(Pick<absl::string_view>(rng, kQuantifiers));
}

std::string RandomPiece(std::mt19937_64& rng, int depth) {
  std::string piece = RandomAtom(rng, depth);
  if (Chance(rng, 30)) piece += RandomQuantifier(rng);
  return piece;
}

std::string RandomExpr(std::mt19937_64& rng, int depth) {
  std::string out;
  const size_t branches = Chance(rng, 75) ? 1 : 2 + Uniform(rng, 2);
  for (size_t b = 0; b < branches; ++b) {
    if (b > 0) out += '|';
    const size_t pieces = Chance(rng, 5) ? 0 : 1 + Uniform(rng, 4);
    for (size_t i = 0; i < pieces; ++i) out += RandomPiece(rng, depth);
  }
  return out;
}

// A sample from RE2's parse tree, or false if the chosen path cannot match.
bool AppendSample(re2::Regexp* re, std::mt19937_64& rng, std::string& out) {
  const bool fold = (re->parse_flags() & re2::Regexp::FoldCase) != 0;
  const auto append_rune = [&](re2::Rune rune) {
    // RE2 keeps FoldCase only on ASCII letters whose orbit is {X, x}.
    if (fold && rune < 0x80 && Chance(rng, 50)) {
      rune = static_cast<unsigned char>(
          absl::ascii_isupper(static_cast<unsigned char>(rune))
              ? absl::ascii_tolower(static_cast<unsigned char>(rune))
              : absl::ascii_toupper(static_cast<unsigned char>(rune)));
    }
    out += EncodeRune(rune);
  };
  switch (re->op()) {
    case re2::kRegexpNoMatch:
    case re2::kRegexpHaveMatch:
      return false;
    case re2::kRegexpEmptyMatch:
    case re2::kRegexpBeginLine:
    case re2::kRegexpEndLine:
    case re2::kRegexpBeginText:
    case re2::kRegexpEndText:
    case re2::kRegexpWordBoundary:
    case re2::kRegexpNoWordBoundary:
      return true;
    case re2::kRegexpLiteral:
      append_rune(re->rune());
      return true;
    case re2::kRegexpLiteralString:
      for (int i = 0; i < re->nrunes(); ++i) append_rune(re->runes()[i]);
      return true;
    case re2::kRegexpCharClass: {
      re2::CharClass* cc = re->cc();
      if (cc->empty()) return false;
      size_t ranges = 0;
      for (const re2::RuneRange range : *cc) {
        static_cast<void>(range);
        ++ranges;
      }
      size_t chosen = Uniform(rng, ranges);
      for (const re2::RuneRange range : *cc) {
        if (chosen-- != 0) continue;
        const re2::Rune span = range.hi - range.lo;
        re2::Rune rune = range.lo;
        if (Chance(rng, 30)) {
          rune = range.hi;
        } else if (span > 0) {
          rune = range.lo + static_cast<re2::Rune>(
                                Uniform(rng, static_cast<size_t>(span) + 1));
        }
        out += EncodeRune(rune);
        break;
      }
      return true;
    }
    case re2::kRegexpAnyChar:
    case re2::kRegexpAnyByte:
      out += EncodeRune(RandomRune(rng));
      return true;
    case re2::kRegexpCapture:
      return AppendSample(re->sub()[0], rng, out);
    case re2::kRegexpStar:
    case re2::kRegexpPlus:
    case re2::kRegexpQuest: {
      const size_t min = re->op() == re2::kRegexpPlus ? 1 : 0;
      const size_t max = re->op() == re2::kRegexpQuest ? 1 : 3;
      const size_t count = min + Uniform(rng, max - min + 1);
      for (size_t i = 0; i < count; ++i) {
        if (!AppendSample(re->sub()[0], rng, out)) return false;
      }
      return true;
    }
    case re2::kRegexpRepeat: {
      const int min = re->min();
      const int max = re->max() < 0 ? min + 2 : std::min(re->max(), min + 2);
      const int count = min + static_cast<int>(Uniform(
                                  rng, static_cast<size_t>(max - min) + 1));
      for (int i = 0; i < count; ++i) {
        if (!AppendSample(re->sub()[0], rng, out)) return false;
      }
      return true;
    }
    case re2::kRegexpConcat:
      for (int i = 0; i < re->nsub(); ++i) {
        if (!AppendSample(re->sub()[i], rng, out)) return false;
      }
      return true;
    case re2::kRegexpAlternate:
      return AppendSample(
          re->sub()[Uniform(rng, static_cast<size_t>(re->nsub()))], rng, out);
  }
  return false;
}

}  // namespace

std::string RandomPattern(std::mt19937_64& rng) {
  std::string pattern = RandomExpr(rng, /*depth=*/3);
  if (Chance(rng, 15)) pattern = "(?i)" + pattern;
  return pattern;
}

std::optional<std::string> SampleMatch(re2::Regexp* re, std::mt19937_64& rng) {
  std::string out;
  if (!AppendSample(re, rng, out)) return std::nullopt;
  return out;
}

std::string RandomText(std::mt19937_64& rng) {
  std::string text;
  const size_t length = Uniform(rng, 12);
  for (size_t i = 0; i < length; ++i) {
    const size_t roll = Uniform(rng, 100);
    if (roll < 80) {
      text += EncodeRune(RandomRune(rng));
    } else if (roll < 90) {
      text += '\n';
    } else {
      // Invalid UTF-8: a stray continuation byte, a truncated Kelvin sign,
      // and an encoded surrogate.
      static constexpr absl::string_view kInvalid[] = {
          "\x80", "\xFF", "\xE2\x84", "\xED\xA0\x80"};
      text += Pick<absl::string_view>(rng, kInvalid);
    }
  }
  return text;
}

std::string Mutate(std::string text, std::mt19937_64& rng) {
  const size_t edits = 1 + Uniform(rng, 3);
  for (size_t e = 0; e < edits; ++e) {
    const size_t pos = Uniform(rng, text.size() + 1);
    switch (Uniform(rng, 4)) {
      case 0:
        text.insert(pos, EncodeRune(RandomRune(rng)));
        break;
      case 1:
        if (pos < text.size()) text.erase(pos, 1);
        break;
      case 2:
        if (pos < text.size()) text[pos] = static_cast<char>(rng());
        break;
      default:
        if (pos < text.size() &&
            absl::ascii_isalpha(static_cast<unsigned char>(text[pos]))) {
          text[pos] = static_cast<char>(text[pos] ^ 0x20);
        }
        break;
    }
  }
  return text;
}

std::string Embed(const std::string& core, std::mt19937_64& rng) {
  std::string text = RandomText(rng);
  if (Chance(rng, 30)) text += '\n';
  text += core;
  if (Chance(rng, 30)) text += '\n';
  text += RandomText(rng);
  return text;
}

}  // namespace ferret_testing

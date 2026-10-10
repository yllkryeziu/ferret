// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/git/internal/wildmatch.h"

#include <cstddef>
#include <cstdint>
#include <optional>

#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"

namespace ferret::git_internal {
namespace {

// kAbortAll means no later start position of any enclosing star can match;
// kAbortToStarStar means only an enclosing '**' may keep trying. Both are
// failures to the caller of Wildmatch().
enum class MatchResult { kMatch, kNoMatch, kAbortAll, kAbortToStarStar };

// Git's own ASCII-only character classes (ctype.c), which do not depend on
// the locale.
constexpr bool IsUpper(uint8_t c) { return c >= 'A' && c <= 'Z'; }
constexpr bool IsLower(uint8_t c) { return c >= 'a' && c <= 'z'; }
constexpr bool IsDigit(uint8_t c) { return c >= '0' && c <= '9'; }
constexpr bool IsAlpha(uint8_t c) { return IsUpper(c) || IsLower(c); }
constexpr bool IsAlnum(uint8_t c) { return IsAlpha(c) || IsDigit(c); }
constexpr bool IsXdigit(uint8_t c) {
  return IsDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}
constexpr bool IsBlank(uint8_t c) { return c == ' ' || c == '\t'; }
constexpr bool IsSpace(uint8_t c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}
constexpr bool IsCntrl(uint8_t c) { return c < 0x20 || c == 0x7F; }
constexpr bool IsPrint(uint8_t c) { return c >= 0x20 && c <= 0x7E; }
constexpr bool IsGraph(uint8_t c) { return c > 0x20 && c <= 0x7E; }
constexpr bool IsPunct(uint8_t c) { return IsGraph(c) && !IsAlnum(c); }
constexpr bool IsGlobSpecial(uint8_t c) {
  return c == '*' || c == '?' || c == '[' || c == '\\';
}
constexpr uint8_t ToLower(uint8_t c) {
  return IsUpper(c) ? static_cast<uint8_t>(c + ('a' - 'A')) : c;
}
constexpr uint8_t ToUpper(uint8_t c) {
  return IsLower(c) ? static_cast<uint8_t>(c - ('a' - 'A')) : c;
}

// Returns whether `c` belongs to the [:name:] class, or nullopt if git does
// not know the class name.
std::optional<bool> InNamedClass(absl::string_view name, uint8_t c,
                                 bool casefold) {
  if (name == "alnum") return IsAlnum(c);
  if (name == "alpha") return IsAlpha(c);
  if (name == "blank") return IsBlank(c);
  if (name == "cntrl") return IsCntrl(c);
  if (name == "digit") return IsDigit(c);
  if (name == "graph") return IsGraph(c);
  if (name == "lower") return IsLower(c);
  if (name == "print") return IsPrint(c);
  if (name == "punct") return IsPunct(c);
  if (name == "space") return IsSpace(c);
  if (name == "upper") return IsUpper(c) || (casefold && IsLower(c));
  if (name == "xdigit") return IsXdigit(c);
  return std::nullopt;
}

// Mirrors dowild() in git's wildmatch.c. Positions past the end of either
// string read as NUL, exactly like the C strings git walks. Recursive calls
// receive suffixes of `pattern`, so "the start of the pattern" in the '**'
// rule means the start of the suffix, as in git.
MatchResult DoWild(absl::string_view pattern, absl::string_view text,
                   const WildmatchOptions& options) {
  const auto pat = [pattern](size_t i) -> uint8_t {
    return i < pattern.size() ? static_cast<uint8_t>(pattern[i]) : 0;
  };
  const auto txt = [text](size_t i) -> uint8_t {
    return i < text.size() ? static_cast<uint8_t>(text[i]) : 0;
  };
  size_t p = 0;
  size_t t = 0;
  for (; pat(p) != 0; ++t, ++p) {
    uint8_t p_ch = pat(p);
    uint8_t t_ch = txt(t);
    if (t_ch == 0 && p_ch != '*') return MatchResult::kAbortAll;
    if (options.casefold) {
      t_ch = ToLower(t_ch);
      p_ch = ToLower(p_ch);
    }
    switch (p_ch) {
      case '\\':
        // The escaped character is compared as is; a trailing backslash
        // compares NUL against a non-NUL text byte and fails.
        p_ch = pat(++p);
        [[fallthrough]];
      default:
        if (t_ch != p_ch) return MatchResult::kNoMatch;
        continue;
      case '?':
        if (options.pathname && t_ch == '/') return MatchResult::kNoMatch;
        continue;
      case '*': {
        bool match_slash;
        if (pat(++p) == '*') {
          const size_t second_star = p;
          while (pat(++p) == '*') {
          }
          if (!options.pathname) {
            match_slash = true;
          } else if ((second_star < 2 || pat(second_star - 2) == '/') &&
                     (pat(p) == 0 || pat(p) == '/' ||
                      (pat(p) == '\\' && pat(p + 1) == '/'))) {
            // "**/" may match no directory at all: try the rest of the
            // pattern right here before letting "**" consume text.
            if (pat(p) == '/' && DoWild(pattern.substr(p + 1), text.substr(t),
                                        options) == MatchResult::kMatch) {
              return MatchResult::kMatch;
            }
            match_slash = true;
          } else {
            match_slash = false;
          }
        } else {
          match_slash = !options.pathname;
        }
        if (pat(p) == 0) {
          // A trailing "**" matches everything; a trailing '*' only if no
          // directory separator remains.
          if (!match_slash && absl::StrContains(text.substr(t), '/')) {
            return MatchResult::kAbortToStarStar;
          }
          return MatchResult::kMatch;
        }
        if (!match_slash && pat(p) == '/') {
          // "*/" matches the rest of this path component; the loop
          // increment consumes the slash in both strings.
          const size_t slash = text.find('/', t);
          if (slash == absl::string_view::npos) return MatchResult::kAbortAll;
          t = slash;
          break;
        }
        while (t_ch != 0) {
          if (!IsGlobSpecial(pat(p))) {
            // A literal follows the star: skip ahead to its next occurrence,
            // since everything before it must belong to the star.
            p_ch = options.casefold ? ToLower(pat(p)) : pat(p);
            while ((t_ch = txt(t)) != 0 && (match_slash || t_ch != '/')) {
              if (options.casefold) t_ch = ToLower(t_ch);
              if (t_ch == p_ch) break;
              ++t;
            }
            if (t_ch != p_ch) {
              return match_slash ? MatchResult::kAbortAll
                                 : MatchResult::kAbortToStarStar;
            }
          }
          const MatchResult matched =
              DoWild(pattern.substr(p), text.substr(t), options);
          if (matched != MatchResult::kNoMatch) {
            if (!match_slash || matched != MatchResult::kAbortToStarStar) {
              return matched;
            }
          } else if (!match_slash && t_ch == '/') {
            return MatchResult::kAbortToStarStar;
          }
          t_ch = txt(++t);
        }
        return MatchResult::kAbortAll;
      }
      case '[': {
        p_ch = pat(++p);
        if (p_ch == '^') p_ch = '!';
        const bool negated = p_ch == '!';
        if (negated) p_ch = pat(++p);
        uint8_t prev_ch = 0;
        bool matched = false;
        do {
          if (p_ch == 0) return MatchResult::kAbortAll;
          if (p_ch == '\\') {
            p_ch = pat(++p);
            if (p_ch == 0) return MatchResult::kAbortAll;
            if (t_ch == p_ch) matched = true;
          } else if (p_ch == '-' && prev_ch != 0 && pat(p + 1) != 0 &&
                     pat(p + 1) != ']') {
            p_ch = pat(++p);
            if (p_ch == '\\') {
              p_ch = pat(++p);
              if (p_ch == 0) return MatchResult::kAbortAll;
            }
            if (t_ch <= p_ch && t_ch >= prev_ch) {
              matched = true;
            } else if (options.casefold && IsLower(t_ch)) {
              const uint8_t upper = ToUpper(t_ch);
              if (upper <= p_ch && upper >= prev_ch) matched = true;
            }
            p_ch = 0;  // A range end cannot start another range.
          } else if (p_ch == '[' && pat(p + 1) == ':') {
            p += 2;
            const size_t name_start = p;
            while ((p_ch = pat(p)) != 0 && p_ch != ']') ++p;
            if (p_ch == 0) return MatchResult::kAbortAll;
            if (p == name_start || pat(p - 1) != ':') {
              // No closing ":]": the '[' is an ordinary member.
              p = name_start - 2;
              p_ch = '[';
              if (t_ch == p_ch) matched = true;
              continue;
            }
            const std::optional<bool> in_class =
                InNamedClass(pattern.substr(name_start, p - 1 - name_start),
                             t_ch, options.casefold);
            if (!in_class.has_value()) return MatchResult::kAbortAll;
            if (*in_class) matched = true;
            p_ch = 0;  // A class cannot start a range.
          } else if (t_ch == p_ch) {
            matched = true;
          }
        } while (prev_ch = p_ch, (p_ch = pat(++p)) != ']');
        if (matched == negated || (options.pathname && t_ch == '/')) {
          return MatchResult::kNoMatch;
        }
        continue;
      }
    }
  }
  return t < text.size() ? MatchResult::kNoMatch : MatchResult::kMatch;
}

}  // namespace

bool Wildmatch(absl::string_view pattern, absl::string_view text,
               WildmatchOptions options) {
  DCHECK(!absl::StrContains(pattern, '\0'));
  DCHECK(!absl::StrContains(text, '\0'));
  return DoWild(pattern, text, options) == MatchResult::kMatch;
}

}  // namespace ferret::git_internal

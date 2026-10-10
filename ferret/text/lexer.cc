// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Each language is a Syntax: a table of the lexical features it has. One
// Scanner interprets any Syntax, splitting text into tokens that are code,
// comment, or literal. A comment or literal is scanned to its terminator in
// one step; a construct still open at the end of the text becomes the exit
// LexState, and an entry LexState is resumed by scanning to its terminator
// first. Code is consumed a byte at a time, since a code byte only matters
// if it opens a construct.

#include "ferret/text/lexer.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"

#include "ferret/text/language.h"
#include "ferret/text/line_counter.h"
#include "ferret/text/utf8.h"

namespace ferret {
namespace {

enum class HashComment : uint8_t {
  kNone,
  kAnywhere,     // Python.
  kAtWordStart,  // Shell: `#` in `$#` or `a#b` is not a comment.
};

enum class SingleQuote : uint8_t {
  kCode,            // `'` is not special.
  kEscaped,         // A literal with backslash escapes, like "...".
  kVerbatim,        // Shell: no escapes inside '...'.
  kCharOrLifetime,  // Rust: 'a' is a character, 'a alone a lifetime.
};

enum class Backtick : uint8_t {
  kCode,       // Shell command substitution is code.
  kRawString,  // Go: no escapes, may span lines.
  kTemplate,   // JavaScript: escapes and ${...} substitutions.
};

// The lexical features of a language family. Everything defaults to off, so
// a default Syntax lexes all text as code.
struct Syntax {
  // `//` line comments and `/* */` block comments.
  bool slash_comments = false;
  // `/* /* */ */` nests (Rust).
  bool nested_block_comments = false;
  // A backslash before the newline continues a `//` comment (C and C++ line
  // splicing).
  bool spliced_line_comments = false;
  HashComment hash_comments = HashComment::kNone;
  // `"` opens a literal with backslash escapes.
  bool double_quotes = false;
  SingleQuote single_quote = SingleQuote::kCode;
  // Quoted literals may span lines. Otherwise an unescaped newline ends one,
  // which also recovers from a quote that was never closed.
  bool multiline_quotes = false;
  // `'` right after a number is a digit separator, as in 1'000 (C23, C++14).
  bool digit_separators = false;
  bool triple_double_quotes = false;  // """...""": Python, Java text blocks.
  bool triple_single_quotes = false;  // '''...''': Python.
  Backtick backtick = Backtick::kCode;
  bool cpp_raw_strings = false;   // R"delim(...)delim".
  bool rust_raw_strings = false;  // r#"..."#.
  // JavaScript `/.../` where an operand is expected.
  bool regex_literals = false;
  bool here_documents = false;  // Shell <<WORD.
  // A backslash in code escapes the next byte, so `\'` opens nothing (shell).
  bool escapes_in_code = false;
  bool dollar_in_identifiers = false;  // Java, JavaScript.
};

constexpr Syntax kPlainSyntax = {};

constexpr Syntax kCSyntax = {
    .slash_comments = true,
    .spliced_line_comments = true,
    .double_quotes = true,
    .single_quote = SingleQuote::kEscaped,
    .digit_separators = true,
};

constexpr Syntax kCppSyntax = {
    .slash_comments = true,
    .spliced_line_comments = true,
    .double_quotes = true,
    .single_quote = SingleQuote::kEscaped,
    .digit_separators = true,
    .cpp_raw_strings = true,
};

constexpr Syntax kGoSyntax = {
    .slash_comments = true,
    .double_quotes = true,
    .single_quote = SingleQuote::kEscaped,
    .backtick = Backtick::kRawString,
};

constexpr Syntax kJavaSyntax = {
    .slash_comments = true,
    .double_quotes = true,
    .single_quote = SingleQuote::kEscaped,
    .triple_double_quotes = true,
    .dollar_in_identifiers = true,
};

constexpr Syntax kJavaScriptSyntax = {
    .slash_comments = true,
    .double_quotes = true,
    .single_quote = SingleQuote::kEscaped,
    .backtick = Backtick::kTemplate,
    .regex_literals = true,
    .dollar_in_identifiers = true,
};

constexpr Syntax kPythonSyntax = {
    .hash_comments = HashComment::kAnywhere,
    .double_quotes = true,
    .single_quote = SingleQuote::kEscaped,
    .triple_double_quotes = true,
    .triple_single_quotes = true,
};

constexpr Syntax kRustSyntax = {
    .slash_comments = true,
    .nested_block_comments = true,
    .double_quotes = true,
    .single_quote = SingleQuote::kCharOrLifetime,
    .multiline_quotes = true,
    .rust_raw_strings = true,
};

constexpr Syntax kShellSyntax = {
    .hash_comments = HashComment::kAtWordStart,
    .double_quotes = true,
    .single_quote = SingleQuote::kVerbatim,
    .multiline_quotes = true,
    .here_documents = true,
    .escapes_in_code = true,
};

const Syntax& SyntaxOf(Language language) {
  switch (language) {
    case Language::kC:
      return kCSyntax;
    case Language::kCpp:
    case Language::kObjectiveCpp:
      return kCppSyntax;
    case Language::kGo:
      return kGoSyntax;
    case Language::kJava:
      return kJavaSyntax;
    case Language::kJavaScript:
    case Language::kTypeScript:
      return kJavaScriptSyntax;
    case Language::kPython:
      return kPythonSyntax;
    case Language::kRust:
      return kRustSyntax;
    case Language::kShell:
      return kShellSyntax;
    case Language::kUnknown:
    case Language::kMarkup:
      return kPlainSyntax;
  }
  return kPlainSyntax;
}

// The longest delimiter a C++ raw string may have.
constexpr size_t kMaxRawDelimiter = 16;

// Raw strings with more '#' than this resume as if they had this many.
constexpr int kMaxRustRawHashes = 3;
constexpr std::array<LexState, kMaxRustRawHashes + 1> kRustRawStates = {
    LexState::kRustRawString0, LexState::kRustRawString1,
    LexState::kRustRawString2, LexState::kRustRawString3};

bool IsAsciiAlpha(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
bool IsAsciiDigit(char c) { return c >= '0' && c <= '9'; }
bool IsSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' ||
         c == '\f';
}

// Bytes of identifiers and numbers. Every non-ASCII byte counts, so that a
// match never splits a UTF-8 identifier.
bool IsIdentifierByte(const Syntax& syntax, char c) {
  return IsAsciiAlpha(c) || IsAsciiDigit(c) || c == '_' ||
         static_cast<uint8_t>(c) >= 0x80 ||
         (c == '$' && syntax.dollar_in_identifiers);
}

// A character of a C++ raw string delimiter: printable ASCII other than
// space, parentheses, backslash, and (for sanity) the quote.
bool IsRawDelimiterByte(char c) {
  return c > ' ' && c < 0x7F && c != '(' && c != ')' && c != '\\' && c != '"';
}

// A byte that ends a shell word, so that a `#` after it starts a comment.
bool IsShellWordBreak(char c) {
  return IsSpace(c) || absl::StrContains(absl::string_view(";&|()<>"), c);
}

// Where a JavaScript `/` starts a regular expression rather than dividing:
// at the start of the text, or after a byte that ends no operand. After `)`,
// `]`, an identifier, or a literal it divides, which misreads `if (x) /re/`
// and `return /re/`, both rare.
bool AllowsRegexAfter(char c) {
  return c == '\0' ||
         absl::StrContains(absl::string_view("(,=:[!&|?{};}+-*%<>~^"), c);
}

// A line that plausibly ends a here-document whose word is unknown: one word
// of identifier characters, after optional tabs.
bool LooksLikeHereDocumentEnd(absl::string_view line) {
  while (!line.empty() && line.front() == '\t') line.remove_prefix(1);
  if (line.empty() || !(IsAsciiAlpha(line.front()) || line.front() == '_')) {
    return false;
  }
  return std::all_of(line.begin(), line.end(), [](char c) {
    return IsAsciiAlpha(c) || IsAsciiDigit(c) || c == '_';
  });
}

// Maps a state from a newer index, which this build cannot resume, to kCode.
LexState KnownOrCode(LexState state) {
  return static_cast<uint8_t>(state) <=
                 static_cast<uint8_t>(LexState::kRustRawString3)
             ? state
             : LexState::kCode;
}

enum class Region : uint8_t { kCode, kComment, kString };

Region RegionOf(LexState state) {
  switch (state) {
    case LexState::kCode:
    case LexState::kTemplateSubstitution:
      return Region::kCode;
    case LexState::kBlockComment:
    case LexState::kLineComment:
      return Region::kComment;
    case LexState::kDoubleQuotedString:
    case LexState::kSingleQuotedString:
    case LexState::kTripleDoubleQuotedString:
    case LexState::kTripleSingleQuotedString:
    case LexState::kBacktickString:
    case LexState::kCppRawString:
    case LexState::kHereDocument:
    case LexState::kRustRawString0:
    case LexState::kRustRawString1:
    case LexState::kRustRawString2:
    case LexState::kRustRawString3:
      return Region::kString;
  }
  return Region::kCode;
}

// A maximal run of one region. Tokens tile the text: each starts where the
// previous one ended.
struct Token {
  Region region;
  size_t end;
};

// Splits a text into tokens. Not thread-safe.
class Scanner {
 public:
  Scanner(const Syntax& syntax, absl::string_view text, LexState entry)
      : syntax_(syntax), text_(text), pending_(KnownOrCode(entry)) {}
  Scanner(const Scanner&) = delete;
  Scanner& operator=(const Scanner&) = delete;

  bool done() const { return pos_ == text_.size() && !stashed_.has_value(); }

  // Returns the next token. Requires !done().
  Token Next();

  // The state at the end of the text. Requires done().
  LexState exit_state() const {
    DCHECK(done());
    if (pending_ != LexState::kCode) return pending_;
    return template_braces_.empty() ? LexState::kCode
                                    : LexState::kTemplateSubstitution;
  }

 private:
  // The here-document announced on the current line, if any.
  struct HereDocument {
    absl::string_view word;
    bool strip_tabs;  // <<-WORD: the terminator may be indented by tabs.
  };

  // Returns the byte at `i`, or '\0' past the end, for lookahead.
  char At(size_t i) const { return i < text_.size() ? text_[i] : '\0'; }

  // The identifier or number bytes that end just before `pos`.
  absl::string_view WordBefore(size_t pos) const;

  Token Resume(LexState state);
  // Scans the construct that starts at pos_, if one does.
  std::optional<Token> ScanConstruct();
  // Consumes one element of code: a byte, or an escape or here-document
  // marker. May set pending_ to a construct that starts right after it.
  void AdvanceCode();

  std::optional<Token> ScanDoubleQuote();
  std::optional<Token> ScanSingleQuote();
  std::optional<absl::string_view> CppRawDelimiter() const;
  int RustRawHashes() const;
  bool IsHashComment() const;
  bool MarkHereDocument();

  // Each Scan function scans a construct whose body starts at `body` (past
  // its opening delimiter, or at pos_ when resuming) and returns its token.
  Token ScanLineComment(size_t body);
  Token ScanBlockComment(size_t body);
  Token ScanQuoted(size_t body, char quote);
  Token ScanTripleQuoted(size_t body, char quote);
  Token ScanRawBacktick(size_t body);
  Token ScanTemplateText(size_t body);
  Token ScanRegex(size_t body);
  Token ScanCppRawString(size_t body,
                         std::optional<absl::string_view> delimiter);
  Token ScanRustRawString(size_t body, int hashes);
  Token ScanHereDocumentBody(size_t body);

  // Ends the current token at `end`.
  Token Finish(Region region, size_t end) {
    pos_ = end;
    // A literal is an operand, after which `/` divides.
    if (region == Region::kString) previous_ = '"';
    return {region, end};
  }
  // Ends the current token at the end of the text, with `state` open.
  Token FinishOpen(Region region, LexState state) {
    pending_ = state;
    pos_ = text_.size();
    return {region, pos_};
  }

  const Syntax& syntax_;
  const absl::string_view text_;
  size_t pos_ = 0;
  // A construct that starts at pos_: the entry state, a here-document body,
  // template text after `}`, or the construct left open at the end.
  LexState pending_;
  // A construct token found after a code token, returned by the next call.
  std::optional<Token> stashed_;
  // The last non-space code byte, or '\0' at the start of the text.
  char previous_ = '\0';
  std::optional<HereDocument> here_document_;
  // One entry per enclosing template substitution: the depth of `{` nesting
  // inside it.
  absl::InlinedVector<int, 4> template_braces_;
};

absl::string_view Scanner::WordBefore(size_t pos) const {
  size_t begin = pos;
  while (begin > 0 && IsIdentifierByte(syntax_, text_[begin - 1])) --begin;
  return text_.substr(begin, pos - begin);
}

namespace {

// How the scanner treats a byte of code. kPlain bytes change nothing but the
// last non-space byte; kSpace bytes change nothing; kSpecial bytes may start
// a construct (ScanConstruct) or change the scanner's state (AdvanceCode).
// The table is a superset over all syntaxes, so a byte that is special in
// one language is special in every language.
enum ByteKind : uint8_t { kPlain = 0, kSpace = 1, kSpecial = 2 };

constexpr std::array<uint8_t, 256> MakeByteKinds() {
  std::array<uint8_t, 256> kinds = {};
  for (const char c : {' ', '\t', '\r', '\v', '\f'}) {
    kinds[static_cast<unsigned char>(c)] = kSpace;
  }
  for (const char c : {'/', '#', '"', '\'', '`', '\\', '<', '\n', '{', '}'}) {
    kinds[static_cast<unsigned char>(c)] = kSpecial;
  }
  return kinds;
}

constexpr std::array<uint8_t, 256> kByteKinds = MakeByteKinds();

}  // namespace

Token Scanner::Next() {
  DCHECK(!done());
  if (stashed_.has_value()) {
    const Token token = *stashed_;
    stashed_.reset();
    return token;
  }
  if (pending_ != LexState::kCode) {
    return Resume(std::exchange(pending_, LexState::kCode));
  }
  const size_t start = pos_;
  while (pos_ < text_.size()) {
    // Skip bytes that cannot start a construct or change the state, which
    // is nearly all of code. Inside a template substitution braces matter
    // and are special in the table, so no case is excluded.
    {
      size_t i = pos_;
      char last = '\0';
      bool any_plain = false;
      const size_t size = text_.size();
      while (i < size) {
        const uint8_t kind = kByteKinds[static_cast<uint8_t>(text_[i])];
        if (kind == kSpecial) break;
        if (kind == kPlain) {
          last = text_[i];
          any_plain = true;
        }
        ++i;
      }
      if (any_plain) previous_ = last;
      pos_ = i;
      if (pos_ == size) break;
    }
    const size_t here = pos_;
    if (std::optional<Token> construct = ScanConstruct()) {
      if (here == start) return *construct;
      stashed_ = construct;
      return {Region::kCode, here};
    }
    AdvanceCode();
    if (pending_ != LexState::kCode) break;
  }
  return {Region::kCode, pos_};
}

Token Scanner::Resume(LexState state) {
  switch (state) {
    case LexState::kCode:
      break;
    case LexState::kBlockComment:
      return ScanBlockComment(pos_);
    case LexState::kLineComment:
      return ScanLineComment(pos_);
    case LexState::kDoubleQuotedString:
      return ScanQuoted(pos_, '"');
    case LexState::kSingleQuotedString:
      return ScanQuoted(pos_, '\'');
    case LexState::kTripleDoubleQuotedString:
      return ScanTripleQuoted(pos_, '"');
    case LexState::kTripleSingleQuotedString:
      return ScanTripleQuoted(pos_, '\'');
    case LexState::kBacktickString:
      return syntax_.backtick == Backtick::kTemplate ? ScanTemplateText(pos_)
                                                     : ScanRawBacktick(pos_);
    case LexState::kTemplateSubstitution:
      template_braces_.push_back(0);
      break;
    case LexState::kCppRawString:
      return ScanCppRawString(pos_, std::nullopt);
    case LexState::kHereDocument:
      return ScanHereDocumentBody(pos_);
    case LexState::kRustRawString0:
      return ScanRustRawString(pos_, 0);
    case LexState::kRustRawString1:
      return ScanRustRawString(pos_, 1);
    case LexState::kRustRawString2:
      return ScanRustRawString(pos_, 2);
    case LexState::kRustRawString3:
      return ScanRustRawString(pos_, 3);
  }
  return pos_ < text_.size() ? Next() : Token{Region::kCode, pos_};
}

std::optional<Token> Scanner::ScanConstruct() {
  const char next = At(pos_ + 1);
  switch (text_[pos_]) {
    case '/':
      if (syntax_.slash_comments && next == '/') {
        return ScanLineComment(pos_ + 2);
      }
      if (syntax_.slash_comments && next == '*') {
        return ScanBlockComment(pos_ + 2);
      }
      if (syntax_.regex_literals && AllowsRegexAfter(previous_)) {
        return ScanRegex(pos_ + 1);
      }
      return std::nullopt;
    case '#':
      if (IsHashComment()) return ScanLineComment(pos_ + 1);
      return std::nullopt;
    case '"':
      return ScanDoubleQuote();
    case '\'':
      return ScanSingleQuote();
    case '`':
      switch (syntax_.backtick) {
        case Backtick::kCode:
          return std::nullopt;
        case Backtick::kRawString:
          return ScanRawBacktick(pos_ + 1);
        case Backtick::kTemplate:
          return ScanTemplateText(pos_ + 1);
      }
      return std::nullopt;
    default:
      return std::nullopt;
  }
}

void Scanner::AdvanceCode() {
  const char c = text_[pos_];
  if (c == '\\' && syntax_.escapes_in_code) {
    previous_ = At(pos_ + 1);
    pos_ = std::min(pos_ + 2, text_.size());
    return;
  }
  if (c == '<' && syntax_.here_documents && MarkHereDocument()) return;
  ++pos_;
  if (c == '\n' && here_document_.has_value()) {
    pending_ = LexState::kHereDocument;
    return;
  }
  if (!template_braces_.empty()) {
    if (c == '{') {
      ++template_braces_.back();
    } else if (c == '}') {
      if (template_braces_.back() == 0) {
        // The substitution ends; template text resumes.
        template_braces_.pop_back();
        pending_ = LexState::kBacktickString;
        return;
      }
      --template_braces_.back();
    }
  }
  if (!IsSpace(c)) previous_ = c;
}

std::optional<Token> Scanner::ScanDoubleQuote() {
  if (!syntax_.double_quotes) return std::nullopt;
  if (syntax_.cpp_raw_strings) {
    if (const std::optional<absl::string_view> delimiter = CppRawDelimiter()) {
      return ScanCppRawString(pos_ + delimiter->size() + 2, delimiter);
    }
  }
  if (syntax_.rust_raw_strings) {
    if (const int hashes = RustRawHashes(); hashes >= 0) {
      return ScanRustRawString(pos_ + 1, hashes);
    }
  }
  if (syntax_.triple_double_quotes && text_.substr(pos_, 3) == "\"\"\"") {
    return ScanTripleQuoted(pos_ + 3, '"');
  }
  return ScanQuoted(pos_ + 1, '"');
}

std::optional<Token> Scanner::ScanSingleQuote() {
  switch (syntax_.single_quote) {
    case SingleQuote::kCode:
      return std::nullopt;
    case SingleQuote::kVerbatim:
      return ScanQuoted(pos_ + 1, '\'');
    case SingleQuote::kCharOrLifetime: {
      const char first = At(pos_ + 1);
      if (first == '\\') return ScanQuoted(pos_ + 1, '\'');
      if (first == '\0' || first == '\'' || first == '\n') return std::nullopt;
      const size_t close = pos_ + 1 + DecodeUtf8(text_.substr(pos_ + 1)).length;
      if (At(close) != '\'') return std::nullopt;  // A lifetime or label.
      return Finish(Region::kString, close + 1);
    }
    case SingleQuote::kEscaped:
      break;
  }
  if (syntax_.digit_separators) {
    const absl::string_view word = WordBefore(pos_);
    if (!word.empty() && IsAsciiDigit(word.front())) return std::nullopt;
  }
  if (syntax_.triple_single_quotes && text_.substr(pos_, 3) == "'''") {
    return ScanTripleQuoted(pos_ + 3, '\'');
  }
  return ScanQuoted(pos_ + 1, '\'');
}

// If the `"` at pos_ opens a C++ raw string, returns its delimiter.
std::optional<absl::string_view> Scanner::CppRawDelimiter() const {
  const absl::string_view prefix = WordBefore(pos_);
  if (prefix != "R" && prefix != "LR" && prefix != "uR" && prefix != "UR" &&
      prefix != "u8R") {
    return std::nullopt;
  }
  const size_t begin = pos_ + 1;
  for (size_t i = begin; i < text_.size() && i - begin <= kMaxRawDelimiter;
       ++i) {
    if (text_[i] == '(') return text_.substr(begin, i - begin);
    if (!IsRawDelimiterByte(text_[i])) break;
  }
  return std::nullopt;
}

// If the `"` at pos_ opens a Rust raw string (r"", br#""#, cr##""##, ...),
// returns its number of '#'; otherwise -1.
int Scanner::RustRawHashes() const {
  size_t r = pos_;
  while (r > 0 && text_[r - 1] == '#') --r;
  const absl::string_view prefix = WordBefore(r);
  if (prefix != "r" && prefix != "br" && prefix != "cr") return -1;
  return static_cast<int>(pos_ - r);
}

bool Scanner::IsHashComment() const {
  switch (syntax_.hash_comments) {
    case HashComment::kNone:
      return false;
    case HashComment::kAnywhere:
      return true;
    case HashComment::kAtWordStart:
      return pos_ == 0 || IsShellWordBreak(text_[pos_ - 1]);
  }
  return false;
}

// Parses a here-document marker at pos_: <<WORD, <<-WORD, <<'WORD',
// <<"WORD", or <<\WORD, but not the here-string <<<. On success, records it
// (its body starts on the next line) and advances past it.
bool Scanner::MarkHereDocument() {
  if (At(pos_ + 1) != '<' || At(pos_ + 2) == '<' ||
      (pos_ > 0 && text_[pos_ - 1] == '<')) {
    return false;
  }
  size_t i = pos_ + 2;
  const bool strip_tabs = At(i) == '-';
  if (strip_tabs) ++i;
  while (At(i) == ' ' || At(i) == '\t') ++i;
  if (At(i) == '\\') ++i;
  size_t word_begin;
  size_t word_end;
  size_t marker_end;
  if (const char quote = At(i); quote == '\'' || quote == '"') {
    word_begin = i + 1;
    word_end = text_.find(quote, word_begin);
    if (word_end == absl::string_view::npos ||
        absl::StrContains(text_.substr(word_begin, word_end - word_begin),
                          '\n')) {
      return false;
    }
    marker_end = word_end + 1;
  } else {
    // An unquoted word; `1<<2` is a shift, not a here-document.
    if (!IsAsciiAlpha(At(i)) && At(i) != '_') return false;
    word_begin = i;
    while (i < text_.size() && !IsShellWordBreak(text_[i]) &&
           text_[i] != '\'' && text_[i] != '"') {
      ++i;
    }
    word_end = marker_end = i;
  }
  if (word_end == word_begin) return false;
  here_document_ =
      HereDocument{.word = text_.substr(word_begin, word_end - word_begin),
                   .strip_tabs = strip_tabs};
  previous_ = text_[marker_end - 1];
  pos_ = marker_end;
  return true;
}

Token Scanner::ScanLineComment(size_t body) {
  for (size_t i = body;;) {
    const size_t newline = text_.find('\n', i);
    if (newline == absl::string_view::npos) {
      return FinishOpen(Region::kComment, LexState::kLineComment);
    }
    const bool spliced = syntax_.spliced_line_comments &&
                         ((newline >= 1 && text_[newline - 1] == '\\') ||
                          (newline >= 2 && text_[newline - 1] == '\r' &&
                           text_[newline - 2] == '\\'));
    if (!spliced) return Finish(Region::kComment, newline);
    i = newline + 1;
  }
}

Token Scanner::ScanBlockComment(size_t body) {
  int depth = 1;
  for (size_t i = body; i + 1 < text_.size(); ++i) {
    if (text_[i] == '*' && text_[i + 1] == '/') {
      if (--depth == 0) return Finish(Region::kComment, i + 2);
      ++i;
    } else if (syntax_.nested_block_comments && text_[i] == '/' &&
               text_[i + 1] == '*') {
      ++depth;
      ++i;
    }
  }
  return FinishOpen(Region::kComment, LexState::kBlockComment);
}

Token Scanner::ScanQuoted(size_t body, char quote) {
  const bool escapes =
      quote == '"' || syntax_.single_quote != SingleQuote::kVerbatim;
  for (size_t i = body; i < text_.size(); ++i) {
    const char c = text_[i];
    if (c == '\\' && escapes) {
      // Skip the escaped byte, or a whole "\r\n" line continuation.
      i += At(i + 1) == '\r' && At(i + 2) == '\n' ? 2 : 1;
    } else if (c == quote) {
      return Finish(Region::kString, i + 1);
    } else if (c == '\n' && !syntax_.multiline_quotes) {
      return Finish(Region::kString, i);
    }
  }
  return FinishOpen(Region::kString, quote == '"'
                                         ? LexState::kDoubleQuotedString
                                         : LexState::kSingleQuotedString);
}

Token Scanner::ScanTripleQuoted(size_t body, char quote) {
  const char triple[] = {quote, quote, quote};
  for (size_t i = body; i < text_.size(); ++i) {
    if (text_[i] == '\\') {
      ++i;
    } else if (text_.substr(i, 3) == absl::string_view(triple, 3)) {
      return Finish(Region::kString, i + 3);
    }
  }
  return FinishOpen(Region::kString, quote == '"'
                                         ? LexState::kTripleDoubleQuotedString
                                         : LexState::kTripleSingleQuotedString);
}

Token Scanner::ScanRawBacktick(size_t body) {
  const size_t close = text_.find('`', body);
  if (close == absl::string_view::npos) {
    return FinishOpen(Region::kString, LexState::kBacktickString);
  }
  return Finish(Region::kString, close + 1);
}

Token Scanner::ScanTemplateText(size_t body) {
  for (size_t i = body; i < text_.size(); ++i) {
    const char c = text_[i];
    if (c == '\\') {
      ++i;
    } else if (c == '`') {
      return Finish(Region::kString, i + 1);
    } else if (c == '$' && At(i + 1) == '{') {
      template_braces_.push_back(0);
      const Token token = Finish(Region::kString, i + 2);
      previous_ = '{';  // An expression starts.
      return token;
    }
  }
  return FinishOpen(Region::kString, LexState::kBacktickString);
}

Token Scanner::ScanRegex(size_t body) {
  bool in_class = false;
  for (size_t i = body; i < text_.size(); ++i) {
    const char c = text_[i];
    if (c == '\n') return Finish(Region::kString, i);
    if (c == '\\') {
      if (At(i + 1) != '\n') ++i;
    } else if (in_class) {
      in_class = c != ']';
    } else if (c == '[') {
      in_class = true;
    } else if (c == '/') {
      return Finish(Region::kString, i + 1);
    }
  }
  // A regular expression cannot span lines, so nothing stays open.
  return Finish(Region::kString, text_.size());
}

Token Scanner::ScanCppRawString(size_t body,
                                std::optional<absl::string_view> delimiter) {
  for (size_t close = text_.find(')', body); close != absl::string_view::npos;
       close = text_.find(')', close + 1)) {
    size_t quote = close + 1;
    if (delimiter.has_value()) {
      if (!absl::StartsWith(text_.substr(quote), *delimiter)) continue;
      quote += delimiter->size();
    } else {
      while (quote - close <= kMaxRawDelimiter &&
             IsRawDelimiterByte(At(quote))) {
        ++quote;
      }
    }
    if (At(quote) == '"') return Finish(Region::kString, quote + 1);
  }
  return FinishOpen(Region::kString, LexState::kCppRawString);
}

Token Scanner::ScanRustRawString(size_t body, int hashes) {
  const size_t count = static_cast<size_t>(hashes);
  for (size_t quote = text_.find('"', body); quote != absl::string_view::npos;
       quote = text_.find('"', quote + 1)) {
    size_t end = quote + 1;
    while (end - quote - 1 < count && At(end) == '#') ++end;
    if (end - quote - 1 == count) return Finish(Region::kString, end);
  }
  return FinishOpen(
      Region::kString,
      kRustRawStates[static_cast<size_t>(std::min(hashes, kMaxRustRawHashes))]);
}

// The body ends before its terminator line, which is lexed as code.
Token Scanner::ScanHereDocumentBody(size_t body) {
  const std::optional<HereDocument> marker =
      std::exchange(here_document_, std::nullopt);
  for (size_t line = body; line < text_.size();) {
    const size_t end = LineEnd(text_, line);
    absl::string_view content = text_.substr(line, end - line);
    absl::ConsumeSuffix(&content, "\r");
    bool terminator;
    if (marker.has_value()) {
      if (marker->strip_tabs) {
        while (absl::ConsumePrefix(&content, "\t")) {
        }
      }
      terminator = content == marker->word;
    } else {
      terminator = LooksLikeHereDocumentEnd(content);
    }
    if (terminator) return Finish(Region::kString, line);
    line = end + 1;
  }
  return FinishOpen(Region::kString, LexState::kHereDocument);
}

}  // namespace

LexState AdvanceLexState(Language language, absl::string_view text,
                         LexState entry) {
  Scanner scanner(SyntaxOf(language), text, entry);
  while (!scanner.done()) scanner.Next();
  return scanner.exit_state();
}

LexState ForEachCodeIdentifier(
    Language language, absl::string_view text, LexState entry,
    absl::FunctionRef<void(size_t begin, size_t end)> fn) {
  const Syntax& syntax = SyntaxOf(language);
  Scanner scanner(syntax, text, entry);
  size_t token_begin = 0;
  while (!scanner.done()) {
    const Token token = scanner.Next();
    if (token.region == Region::kCode) {
      size_t pos = token_begin;
      while (pos < token.end) {
        if (!IsIdentifierByte(syntax, text[pos])) {
          ++pos;
          continue;
        }
        size_t end = pos + 1;
        while (end < text.size() && IsIdentifierByte(syntax, text[end])) ++end;
        // A run that began in an earlier token belongs to that token's
        // region, and one that began before this token was already reported.
        if (pos == 0 || !IsIdentifierByte(syntax, text[pos - 1])) fn(pos, end);
        pos = end;
      }
    }
    token_begin = token.end;
  }
  return scanner.exit_state();
}

struct MatchClassifier::Impl {
  Impl(Language language, absl::string_view text, LexState entry)
      : syntax(SyntaxOf(language)), text(text), scanner(syntax, text, entry) {}

  const Syntax& syntax;
  absl::string_view text;
  Scanner scanner;
  // The cached token covers bytes before token_end; a match at its end
  // belongs to the next token, or to the scanner's exit state.
  size_t token_end = 0;
  size_t last_match_begin = 0;
  Region region = Region::kCode;
};

MatchClassifier::MatchClassifier(Language language, absl::string_view text,
                                 LexState entry)
    : impl_(std::make_unique<Impl>(language, text, entry)) {}

MatchClassifier::~MatchClassifier() = default;

MatchClass MatchClassifier::Classify(size_t match_begin, size_t match_end) {
  DCHECK_LE(impl_->last_match_begin, match_begin);
  DCHECK_LE(match_begin, match_end);
  DCHECK_LE(match_end, impl_->text.size());
  impl_->last_match_begin = match_begin;
  while (impl_->token_end <= match_begin && !impl_->scanner.done()) {
    const Token token = impl_->scanner.Next();
    impl_->token_end = token.end;
    impl_->region = token.region;
  }
  const Region region = impl_->scanner.done() && impl_->token_end <= match_begin
                            ? RegionOf(impl_->scanner.exit_state())
                            : impl_->region;
  switch (region) {
    case Region::kComment:
      return MatchClass::kComment;
    case Region::kString:
      return MatchClass::kString;
    case Region::kCode:
      break;
  }
  auto splits = [&](size_t boundary) {
    return boundary > 0 && boundary < impl_->text.size() &&
           IsIdentifierByte(impl_->syntax, impl_->text[boundary - 1]) &&
           IsIdentifierByte(impl_->syntax, impl_->text[boundary]);
  };
  return splits(match_begin) || splits(match_end) ? MatchClass::kPartialToken
                                                  : MatchClass::kCodeIdentifier;
}

MatchClass ClassifyMatch(Language language, absl::string_view text,
                         LexState entry, size_t match_begin, size_t match_end) {
  MatchClassifier classifier(language, text, entry);
  return classifier.Classify(match_begin, match_end);
}

}  // namespace ferret

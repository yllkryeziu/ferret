// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// A lightweight lexer that classifies search matches.
//
// The router needs to know whether a match of an identifier-like query is a
// code reference, part of a longer identifier, or inside a comment or a
// literal. That takes far less than a parser, and tree-sitter covers only
// three languages, so this lexer recognizes just comments, string, character,
// and regular-expression literals, here-documents, and identifier boundaries
// for C, C++, Objective-C, Go, Java, JavaScript, TypeScript, Python, Rust, and
// shell. Other languages are lexed as plain code.
//
// Lexing is per chunk (chunker.h). The index stores the LexState at the start
// of each chunk, computed at index time by AdvanceLexState over the previous
// chunk, so classifying a match lexes only from its chunk's start. At a line
// boundary the state is exact, and lexing chunk by chunk gives the same result
// as lexing the whole file, except where a construct's terminator needs more
// than one byte of state to recognize. A chunk that starts inside
//  * a C++ raw string R"delim(...)delim" resumes without the delimiter: the
//    string ends at the first ')' followed by up to 16 delimiter characters
//    and '"';
//  * a shell here-document resumes without its word: the body ends at the
//    first line that is a single word;
//  * a Rust raw string with more than three '#' resumes as one with three;
//  * a nested Rust block comment resumes at depth one;
//  * a JavaScript template literal nested in another's substitution, or a
//    comment or literal inside a substitution, resumes without the enclosing
//    template.
// In JavaScript, a chunk that starts with '/' after a line ending in an
// operand also reads it as a regular expression rather than a division. Each
// case needs a construct to span a chunk boundary (8 KiB apart by default).
//
// Like any lexer without a grammar, this one is a heuristic for malformed or
// context-sensitive input: JSX text, shell `$'...'` quoting, and Python
// f-string nesting are lexed approximately.

#ifndef FERRET_TEXT_LEXER_H_
#define FERRET_TEXT_LEXER_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"

#include "ferret/text/language.h"

namespace ferret {

// The construct open at a chunk boundary. Every value fits in four bits, so
// the index can pack it with other per-chunk flags.
// Persisted in .fseg: append only, never renumber.
enum class LexState : uint8_t {
  kCode = 0,
  kBlockComment = 1,  // /* ... */
  // A C or C++ // comment continued past its line by a backslash.
  kLineComment = 2,
  // "..." spanning lines: in Rust and shell, or continued by a backslash.
  kDoubleQuotedString = 3,
  // '...' spanning lines: in shell, or continued by a backslash.
  kSingleQuotedString = 4,
  kTripleDoubleQuotedString = 5,  // Python """...""", Java text blocks.
  kTripleSingleQuotedString = 6,  // Python '''...'''.
  kBacktickString = 7,  // Go raw strings, JavaScript template literal text.
  kTemplateSubstitution = 8,  // Code in a JavaScript template's ${...}.
  kCppRawString = 9,          // R"delim(...)delim".
  kHereDocument = 10,         // The body of a shell <<WORD.
  kRustRawString0 = 11,       // r"...".
  kRustRawString1 = 12,       // r#"..."#.
  kRustRawString2 = 13,       // r##"..."##.
  kRustRawString3 = 14,       // r###"..."###, or with more '#'.
};

// What a match is, judged from the byte where it starts.
enum class MatchClass {
  // In code, and neither end falls inside an identifier or number: "foo" in
  // "foo(x)" or "a.foo", but not in "foobar" or "foo_1".
  kCodeIdentifier,
  // In code, but starting or ending inside a longer identifier or number.
  kPartialToken,
  kComment,
  // In a string, character, or regular-expression literal, or a
  // here-document.
  kString,
};

// Returns the state at the end of `text` when lexing starts in `entry`. When
// `text` ends at a line boundary, as every chunk but a file's last does, this
// is the entry state of the text that follows; the index computes chunk k+1's
// entry as AdvanceLexState(language, chunk k, entry of chunk k), starting from
// kCode. Any `entry` value is accepted, including values from a newer index.
// O(|text|). Thread-safe.
LexState AdvanceLexState(Language language, absl::string_view text,
                         LexState entry);

// Calls fn(begin, end) for every match that MatchClassifier would label
// kCodeIdentifier when the match is made of identifier bytes only, in
// increasing order: each maximal run text[begin, end) of identifier bytes (in
// the sense of Classify's neighbor test) that starts in code. A match of an
// identifier-only query is kCodeIdentifier exactly when it equals such a run,
// so the runs are the code tokens of `text`. Lexing starts in `entry` as
// MatchClassifier does. Returns the state at the end of `text`, as
// AdvanceLexState does, so an indexer needs one pass. O(|text|). Thread-safe.
LexState ForEachCodeIdentifier(
    Language language, absl::string_view text, LexState entry,
    absl::FunctionRef<void(size_t begin, size_t end)> fn);

// Classifies successive matches in one text without restarting the scan.
// `text` must outlive the classifier. Calls require nondecreasing match_begin
// and match_begin <= match_end <= text.size(). The total scan cost for any
// number of calls is O(|text|), plus O(1) per call. Thread-compatible.
class MatchClassifier final {
 public:
  MatchClassifier(Language language, absl::string_view text, LexState entry);
  ~MatchClassifier();

  MatchClassifier(const MatchClassifier&) = delete;
  MatchClassifier& operator=(const MatchClassifier&) = delete;
  MatchClassifier(MatchClassifier&&) = delete;
  MatchClassifier& operator=(MatchClassifier&&) = delete;

  MatchClass Classify(size_t match_begin, size_t match_end);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Classifies the match text[match_begin, match_end), where `text` starts at a
// line boundary in state `entry`, such as a chunk and its stored entry state.
// Requires match_begin <= match_end <= text.size(). O(match_begin plus the
// length of the literal or comment containing it). Thread-safe.
MatchClass ClassifyMatch(Language language, absl::string_view text,
                         LexState entry, size_t match_begin, size_t match_end);

}  // namespace ferret

#endif  // FERRET_TEXT_LEXER_H_

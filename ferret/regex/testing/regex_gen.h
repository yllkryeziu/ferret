// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Random patterns and texts for the planner's soundness property test.
//
// The generator aims at the places where a trigram planner goes wrong: case
// fold orbits (k/K/U+212A, s/S/U+017F, é/É, σ/ς/Σ), UTF-8 sequences of every
// length, repetition edge cases ({0}, {n,}, nested quantifiers), anchors and
// word boundaries, \Q...\E, and literal braces. Texts are mostly strings the
// pattern should match, produced by walking RE2's own parse tree, so that the
// oracle "RE2 matches => the plan accepts" is exercised often rather than
// vacuously.

#ifndef FERRET_REGEX_TESTING_REGEX_GEN_H_
#define FERRET_REGEX_TESTING_REGEX_GEN_H_

#include <optional>
#include <random>
#include <string>

#include "re2/regexp.h"

namespace ferret_testing {

// Returns a random pattern in RE2 syntax. Most are valid; callers skip the
// ones CompilePattern() rejects.
std::string RandomPattern(std::mt19937_64& rng);

// Returns a string the pattern would match if every anchor and word-boundary
// assertion held, built by walking `re`. Returns nullopt if `re` contains a
// NoMatch node on the chosen path.
std::optional<std::string> SampleMatch(re2::Regexp* re, std::mt19937_64& rng);

// Returns `text` with a few random edits: byte and rune insertions,
// deletions, substitutions, and ASCII case flips.
std::string Mutate(std::string text, std::mt19937_64& rng);

// Returns a random string over an adversarial alphabet that includes fold
// orbit members, newlines, and invalid UTF-8.
std::string RandomText(std::mt19937_64& rng);

// Wraps `core` in random context, sometimes separated from it by a newline.
std::string Embed(const std::string& core, std::mt19937_64& rng);

}  // namespace ferret_testing

#endif  // FERRET_REGEX_TESTING_REGEX_GEN_H_

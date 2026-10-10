# Why the trigram planner is sound

The planner turns a regular expression into a `TrigramQuery`, a monotone
AND/OR formula over trigrams. The index returns only chunks whose trigram set
satisfies the query, so a single unsound rule silently drops real matches.
This note states the property the planner guarantees and sketches why each
rule preserves it. The code is `ferret/regex/analyze.cc`. The property is
tested by `//ferret/regex:analyze_soundness_test`.

## Setting

* **F** is the case-folding byte transducer applied at index time:
  * A–Z → a–z
  * U+212A (KELVIN SIGN) → `k`
  * U+017F (LATIN SMALL LETTER LONG S) → `s`
  * every other byte is unchanged.
* **T(x)** is the set of 3-byte windows of `x` that contain no `\n`. The
  index stores `T(F(chunk))` for each chunk.
* **L(n)** is the set of strings the RE2 node `n` matches. Patterns compile
  with NeverNL, so no string in any `L(n)` contains `\n`.
* A query **accepts** a set of trigrams `S` when the formula is true with
  each trigram `t` read as `t ∈ S`.

**Soundness.** If a line of a chunk contains a match `m` of the pattern, then
the plan accepts `T(F(m))`. Queries are monotone, and `F(m)` is a substring
of `F(chunk)` (see "Folding" below), so `T(F(m)) ⊆ T(F(chunk))`. The plan
therefore also accepts the chunk's trigram set.

## Folding commutes with matching

F rewrites only whole, valid UTF-8 sequences, and each rewrite depends on
nothing outside the sequence. A match `m` of a UTF-8 pattern is a sequence
of whole runes, because `\C` is rejected. So if `m` occurs in `t`, the bytes
of `m` in `t` start and end on rune boundaries, and `F(t)` contains `F(m)` at
the corresponding position.

This is why U+212A and U+017F must be folded. `(?i)k` matches U+212A, and if
F left U+212A alone, `F("blacK")` would lack the trigram `ack` that the
plan for `(?i)black` requires. `fold_exceptions_test` scans every rune and
confirms that these two are the only non-ASCII runes RE2's `(?i)` maps to an
ASCII letter. Other fold orbits, such as é/É, are not folded. RE2 parses
`(?i)é` as the class {é, É}, and the planner enumerates both F-images.

## The invariant

The analyzer computes an `Info` for every node, bottom-up. For every
`s ∈ L(n)`:

1. `s = ""` implies `can_empty`;
2. if `exact` is set, `F(s) ∈ exact`;
3. otherwise some `p ∈ prefix` is a prefix of `F(s)`, and some
   `q ∈ suffix` is a suffix of `F(s)`;
4. `match` accepts `T(F(s))`;
5. `can_empty` implies that `"" ∈ exact`, or that `""` is in both `prefix`
   and `suffix`.

The final plan is `match` after a last `Simplify(force)` and `AddExact`.
Clause 4 at the root is exactly soundness.

## The rules preserve it

**Leaves.**
* A literal `r` matches only `U(r)`, the UTF-8 encoding of `r`, so
  `exact = {F(U(r))}` satisfies clause 2.
* A class matches `U(r)` for each of its runes `r`, and `exact` lists every
  `F(U(r))`.
* A class too large to enumerate becomes AnyChar:
  `prefix = suffix = {""}`, which holds trivially.
* Assertions (`^`, `$`, `\b`, …) match only `""`, so they get
  `exact = {""}`, `can_empty = true`, and `match = All`.

**Concatenation `xy`.**
* Every `s ∈ L(xy)` is `uv` with `u ∈ L(x)` and `v ∈ L(y)`. By the folding
  argument, `F(s) = F(u)F(v)`, and `T(F(u)) ∪ T(F(v)) ⊆ T(F(s))`. So
  `x.match ∧ y.match` holds (clause 4).
* **Both exact.** `F(s) ∈ x.exact × y.exact`.
* **Only x exact.** Take `v`'s prefix `p` (clause 3 for `y`). Then `F(u)p` is
  a prefix of `F(s)`, and `F(u)p ∈ x.exact × y.prefix`. If `y` can be empty,
  clause 5 puts `""` in `y.prefix`, which covers `v = ""`.
* **Neither exact.** The new prefix set is `x.prefix`, plus `y.prefix` when
  `x` can be empty. When `u = ""`, clause 5 gives `"" ∈ x.prefix`. Suffixes
  are symmetric.
* **The boundary.** With neither side exact, some `q ∈ x.suffix` ends `F(u)`
  and some `p ∈ y.prefix` starts `F(v)`. So `qp` is a substring of `F(s)`,
  and the trigrams of some string in `x.suffix × y.prefix` all occur in it.
  That product is ANDed in only when every string in it is at least 3 bytes
  long. Otherwise one of the strings guarantees no trigram, and the OR
  could not be required.
* `can_empty = x.can_empty ∧ y.can_empty`.

**Alternation `x|y`.** Every `s` lies in `L(x)` or `L(y)`, so
`x.match ∨ y.match` holds. If only one side has `exact`, that side first
records its exact strings' trigrams (`AddExact`, an OR over the strings).
Then its exact set is unioned into both `prefix` and `suffix`. Each exact
string is a prefix and a suffix of itself.

**Repetition.**
* `x*` and `x{0,n}` can match `""`, which requires no trigram, so they
  become AnyMatch.
* `x{0}` matches only `""`, so it becomes EmptyString. RE2 keeps `{0}` in its
  parse tree, so the analyzer sees it.
* `x+` and `x{m,}` with `m ≥ 1` start with a match of `x` and end with one.
  So `x`'s exact set, if any, moves into `prefix` and `suffix`, and `x.match`
  holds for the first copy.
* `x{m}` with `m ≤ 3` is planned as `m` concatenated copies of `x`. Larger
  counts are planned as `x+`, which is weaker and therefore still sound.

**Simplify.** Simplification keeps the sets small. Every step either records
information in `match` before discarding it, or only weakens what it keeps:

* **Flushing an exact set** (too many strings, or all at least 4 bytes, or at
  least 3 at the top level):
  1. `AddExact` records `∨_{e ∈ exact} ∧ T(e)`. This holds because
     `F(s) ∈ exact`.
  2. Each `e` moves to `prefix` and `suffix` as itself, if it is shorter
     than 3 bytes, or as its first and last 2 bytes. Those are a prefix and
     a suffix of `e`, so clause 3 holds.
* **`SimplifySet`:**
  1. It first ANDs in the OR of the trigrams of every string in the set,
     which is valid because each string is a substring of `F(s)`.
  2. It then truncates each string to `n − 1` bytes for `n = 3, 2, …` until
     the set is small enough. A prefix of a prefix is still a prefix.
  3. Finally it drops each string that extends another string in the set.
     The shorter string already covers every `s` the longer one did.
* **Canonicalizing a query** (factoring, absorption) preserves its meaning.
  `Implies` answers true only when implication is provable.
  `query_algebra_test` checks both exhaustively over a 6-trigram universe.
* **`WeakenExpensiveOrs`** replaces a wide OR with All. The query only gets
  weaker, and queries are monotone, so this is sound.

## Newlines

NeverNL makes RE2 treat every `\n` in the pattern as "no match", so no string
in any `L(n)` contains `\n`. No planned string contains `\n` either, and the
index can omit such trigrams. As a second line of defense,
`AndTrigrams` treats any trigram with `\n` as All, and a DCHECK fires if one
ever appears.

A pattern that matches nothing under NeverNL but something without it could
only match across a line break. `CompilePattern` rejects such patterns
instead of silently returning nothing. `\s` and `[^x]` are unaffected: under
NeverNL they simply exclude `\n`.

## The required literal

The verifier may prefilter lines with an ASCII case-insensitive substring
search for `required_literal`. A string `r` is required when `r` is a
substring of `F(s)` for every `s ∈ L(pattern)`:

* A singleton exact set `{e}` means `F(s) = e`, so `e` is required.
* In a concatenation, a string required by either side is required.
* `x+` keeps `x`'s required string.
* Alternations and optional parts drop it.

`F` agrees with ASCII lowercasing except on U+212A and U+017F. So when the
pattern can match either rune, the literal is restricted to runs that avoid
`k` and `s`, the longest run of at least 3 bytes. In that case the text may
spell the letter with the non-ASCII rune, and an ASCII search would miss it.

## What the test checks

`analyze_soundness_test` generates patterns that exercise every rule above:
fold orbits, UTF-8 lengths, `{0}`, nested quantifiers, anchors, `\Q…\E`,
literal braces, and newline-only atoms. Texts come from three sources:
samples that walk RE2's own parse tree, mutations of those samples, and
adversarial random strings.

For every text that RE2 matches, the test checks two things:
* the plan accepts `T(F(match))`;
* the lowercased match contains the required literal.

Guards keep the test from passing vacuously:
* the oracle must fire on at least 30% of texts;
* every parse-tree op must be visited;
* four planners with seeded bugs must each be caught within 2×10⁵ cases.

A shrinker reduces any counterexample to a line ready to paste into the
regression table.

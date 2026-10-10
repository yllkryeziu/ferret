# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Generates Q_mix, the labeled synthetic query set.

Q_mix is sampled from the corpus itself, so that every class has a known
relation to the text: identifiers and literals are guaranteed to occur,
zero-result queries are verified to be absent, and regexes are instantiated
from templates with real identifiers. The same seed, file list, and corpus
produce byte-identical output; the output is then frozen and committed as
bench/queries/<corpus>_mix_v<N>.jsonl.

Classes and shares (of --count):
  identifier  40%  Names harvested from definitions, split evenly across
                   three document-frequency terciles of equal width on a log
                   scale (df_tercile; 1 = rarest).
  literal     20%  2-4 consecutive tokens with their separators, cut from a
                   random real line, e.g. `return -EINVAL`.
  regex       20%  Templates (REGEX_TEMPLATES) instantiated with harvested
                   identifiers, used round-robin.
  short       10%  Queries of 1-2 bytes from real lines: half short
                   identifier tokens (`fd`), half punctuation (`->`, `{`).
  zero        10%  Plausible identifiers built from parts of real ones,
                   verified absent from every listed file.

A quarter of the identifier, literal, and regex queries are case-insensitive
(--case-insensitive-share); all other queries are case-sensitive.

Definition heuristics (DEFINITION_PATTERNS) are per-extension regexes over
whole files, not parsers. They only need to yield names that someone would
search for; a missed or spurious definition changes which names are sampled,
not the correctness of any query. Document frequency is the number of listed
files containing the name as a whole token; the substring frequency the
engines see can only be higher.

Regex templates never contain a construct that can match a newline (`\\s`, a
negated class, `\\W`), because Zoekt's regex dialect lets those cross lines
while Ferret's line-oriented semantics do not.

Output: one JSON object per line (the schema bench/zoekt/queries.go reads):
  {"id": "linux-0001", "class": "identifier", "kind": "literal",
   "pattern": "sched_entity", "case_sensitive": true, "tercile": 2,
   "df": 31, "template": null}
plus a `<out>.meta.json` sidecar recording the inputs and the generator.

Usage (from the repository root):
  python3 -m bench.queries.gen_mix --root ~/ferret-data/corpora/linux \\
      --files linux.files --corpus linux --out bench/queries/linux_mix_v1.jsonl
"""

import argparse
import collections
import hashlib
import itertools
import json
import pathlib
import platform
import random
import re
import sys
from collections.abc import Callable, Iterable, Sequence

SCHEMA_VERSION = 1

# Class shares, in percent, in output-allocation order.
CLASS_SHARES = (
    ("identifier", 40),
    ("literal", 20),
    ("regex", 20),
    ("short", 10),
    ("zero", 10),
)

MIN_IDENTIFIER_BYTES = 4
MAX_IDENTIFIER_BYTES = 64
MIN_LITERAL_BYTES = 8
MAX_LITERAL_BYTES = 48
MIN_LINE_BYTES = 12
MAX_LINE_BYTES = 160
LINE_RESERVOIR_SIZE = 20_000
MIN_ZERO_BYTES = 8
MAX_ZERO_BYTES = 40
ZERO_CANDIDATES_PER_QUERY = 2

# Maximal runs of word bytes. A pattern made only of word bytes can occur in
# a file only inside one such run, which is why identifiers are restricted
# to word bytes.
TOKEN_RE = re.compile(rb"[A-Za-z0-9_]+")
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
PUNCTUATION_RUN_RE = re.compile(rb"[!-/:-@\[-^`{-~]+")
# Identifier parts: acronyms (`HTTP` in HTTPServer), capitalized or lowercase
# words, and digit runs.
PART_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")

_M = re.MULTILINE
_C_FAMILY = (
    # A function definition or prototype that starts in column 0.
    re.compile(rb"^(?:[A-Za-z_][\w \t*&:<>,]*?[ \t*&])?([A-Za-z_]\w*)[ \t]*\(", _M),
    # A macro.
    re.compile(rb"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)", _M),
    # A struct, union, enum, or class whose body or base list opens on the line.
    re.compile(
        rb"\b(?:struct|union|enum|class)[ \t]+([A-Za-z_]\w*)[ \t]*(?:\{|:[^:]|$)", _M
    ),
    # A one-line typedef.
    re.compile(rb"^typedef\b[^;\n]*?\b([A-Za-z_]\w*)[ \t]*;", _M),
)
_PYTHON = (
    re.compile(rb"^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+([A-Za-z_]\w*)", _M),
)
_GO = (
    re.compile(rb"^func[ \t]+(?:\([^)\n]*\)[ \t]*)?([A-Za-z_]\w*)", _M),
    re.compile(rb"^(?:type|const|var)[ \t]+([A-Za-z_]\w*)", _M),
)
_RUST = (re.compile(rb"\b(?:fn|struct|enum|trait|type|mod)[ \t]+([A-Za-z_]\w*)", _M),)
_JS = (
    re.compile(rb"\b(?:function\*?|class)[ \t]+([A-Za-z_]\w*)", _M),
    re.compile(
        rb"^[ \t]*(?:export[ \t]+)?(?:const|let|var)[ \t]+([A-Za-z_]\w*)[ \t]*=", _M
    ),
)
_JAVA = (
    re.compile(rb"\b(?:class|interface|enum|record)[ \t]+([A-Za-z_]\w*)", _M),
    re.compile(
        rb"^[ \t]+"
        rb"(?:(?:public|protected|private|static|final|abstract|synchronized)[ \t]+)+"
        rb"[\w<>\[\],.? ]+?[ \t]([A-Za-z_]\w*)[ \t]*\(",
        _M,
    ),
)
_SHELL = (re.compile(rb"^(?:function[ \t]+)?([A-Za-z_]\w*)[ \t]*\(\)[ \t]*\{", _M),)

DEFINITION_PATTERNS: dict[str, tuple[re.Pattern[bytes], ...]] = {
    **dict.fromkeys(
        (".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx", ".inl", ".m", ".mm"),
        _C_FAMILY,
    ),
    ".py": _PYTHON,
    ".go": _GO,
    ".rs": _RUST,
    **dict.fromkeys((".js", ".jsx", ".mjs", ".ts", ".tsx"), _JS),
    **dict.fromkeys((".java", ".kt"), _JAVA),
    ".sh": _SHELL,
}

# Words the heuristics can capture (`if (` in column 0, `class Foo` in a
# comment) that nobody searches for as an identifier.
KEYWORDS = frozenset(
    b"auto break case catch char class const continue default define defined "
    b"delete double else elif endif enum extern false float for func function "
    b"goto if ifdef ifndef import include inline int long namespace new null "
    b"nullptr private protected public register return short signed sizeof "
    b"static struct switch template this throw true try typedef typename union "
    b"unsigned using var virtual void volatile while with".split()
)


class GenerationError(Exception):
    """The corpus cannot supply the requested query mix."""


def allocate(total: int, weights: Sequence[int]) -> list[int]:
    """Splits total into integers proportional to weights.

    Uses the largest-remainder method; equal remainders favor the earlier
    weight. The parts always sum to total.
    """
    weight_sum = sum(weights)
    parts = [total * w // weight_sum for w in weights]
    remainders = [total * w % weight_sum for w in weights]
    order = sorted(range(len(weights)), key=lambda i: (-remainders[i], i))
    for i in order[: total - sum(parts)]:
        parts[i] += 1
    return parts


def split_identifier(identifier: str) -> list[str]:
    """Splits snake_case and camelCase identifiers into their parts."""
    return [p for word in identifier.split("_") for p in PART_RE.findall(word)]


def harvest_definitions(path: str, data: bytes) -> set[bytes]:
    """Returns the names that DEFINITION_PATTERNS finds in a file."""
    patterns = DEFINITION_PATTERNS.get(pathlib.PurePosixPath(path).suffix, ())
    names = set()
    for pattern in patterns:
        for match in pattern.finditer(data):
            name = match.group(1)
            if (
                MIN_IDENTIFIER_BYTES <= len(name) <= MAX_IDENTIFIER_BYTES
                and name not in KEYWORDS
            ):
                names.add(name)
    return names


def candidate_line(line: bytes) -> bytes | None:
    """Returns line stripped if it can supply literal and short queries."""
    line = line.strip()
    if not MIN_LINE_BYTES <= len(line) <= MAX_LINE_BYTES or not line.isascii():
        return None
    if sum(1 for _ in TOKEN_RE.finditer(line)) < 2:
        return None
    return line


class Reservoir:
    """A uniform random sample of fixed size over a stream (Algorithm R)."""

    def __init__(self, size: int, rng: random.Random):
        self._size = size
        self._rng = rng
        self._seen = 0
        self.items: list[bytes] = []

    def offer(self, item: bytes) -> None:
        """Considers item; each item offered so far is kept with equal
        probability."""
        self._seen += 1
        if len(self.items) < self._size:
            self.items.append(item)
            return
        slot = self._rng.randrange(self._seen)
        if slot < self._size:
            self.items[slot] = item


def literal_from_line(line: bytes, rng: random.Random) -> str | None:
    """Cuts 2-4 consecutive tokens, with their separators, out of line."""
    tokens = list(TOKEN_RE.finditer(line))
    width = rng.randint(2, min(4, len(tokens)))
    first = rng.randrange(len(tokens) - width + 1)
    literal = line[tokens[first].start() : tokens[first + width - 1].end()]
    if not MIN_LITERAL_BYTES <= len(literal) <= MAX_LITERAL_BYTES or b"\t" in literal:
        return None
    return literal.decode("ascii")


def short_pools(lines: Iterable[bytes]) -> tuple[list[str], list[str]]:
    """Returns the sorted distinct 1-2 byte identifier tokens and punctuation
    strings that occur in lines."""
    identifiers, punctuation = set(), set()
    for line in lines:
        for token in TOKEN_RE.findall(line):
            if len(token) <= 2 and not token[:1].isdigit():
                identifiers.add(token.decode("ascii"))
        for run in PUNCTUATION_RUN_RE.findall(line):
            for i in range(len(run)):
                punctuation.add(run[i : i + 1].decode("ascii"))
                if i + 2 <= len(run):
                    punctuation.add(run[i : i + 2].decode("ascii"))
    return sorted(identifiers), sorted(punctuation)


def identifier_parts(pool: Iterable[str]) -> list[str]:
    """Returns the sorted distinct parts, of 3 bytes or more, of the names."""
    return sorted({p for name in pool for p in split_identifier(name) if len(p) >= 3})


def zero_candidate(parts: Sequence[str], rng: random.Random) -> str | None:
    """Joins three real identifier parts into a plausible new identifier, in
    snake_case or camelCase."""
    parts = [rng.choice(parts) for _ in range(3)]
    if rng.random() < 0.5:
        candidate = "_".join(p.lower() for p in parts)
    else:
        candidate = parts[0].lower() + "".join(p.capitalize() for p in parts[1:])
    if not MIN_ZERO_BYTES <= len(candidate) <= MAX_ZERO_BYTES:
        return None
    return candidate


# Regex templates. Each takes the identifier pool, the literal-line sample,
# and the generator, and returns a pattern or None when it cannot apply.
def _word(pool, lines, rng):
    return rf"\b{rng.choice(pool)}\b"


def _call(pool, lines, rng):
    return rf"{rng.choice(pool)}[ \t]*\("


def _prefix_suffix(pool, lines, rng):
    parts = split_identifier(rng.choice(pool))
    if len(parts) < 2 or len(parts[0]) < 3 or len(parts[-1]) < 3:
        return None
    return rf"{parts[0]}\w*{parts[-1]}"


def _alternation(pool, lines, rng):
    first, second = rng.sample(pool, 2)
    return rf"(?:{first}|{second})"


def _co_occurrence(pool, lines, rng):
    # Two names from one real line, in order, so the pattern matches it.
    names = []
    for token in TOKEN_RE.findall(rng.choice(lines)):
        name = token.decode("ascii")
        if (
            IDENTIFIER_RE.fullmatch(name)
            and len(name) >= MIN_IDENTIFIER_BYTES
            and token not in KEYWORDS
            and name not in names
        ):
            names.append(name)
    if len(names) < 2:
        return None
    first, second = sorted(rng.sample(range(len(names)), 2))
    return rf"{names[first]}.*{names[second]}"


def _line_start(pool, lines, rng):
    return rf"^[ \t]*{rng.choice(pool)}\b"


REGEX_TEMPLATES: tuple[tuple[str, Callable], ...] = (
    ("word", _word),
    ("call", _call),
    ("prefix_suffix", _prefix_suffix),
    ("alternation", _alternation),
    ("co_occurrence", _co_occurrence),
    ("line_start", _line_start),
)


def read_file_list(path: pathlib.Path) -> list[str]:
    """Reads a benchmark file list (one repo-relative path per line)."""
    text = path.read_text(encoding="utf-8")
    paths = text.removesuffix("\n").split("\n") if text else []
    if not paths or "" in paths:
        raise GenerationError(f"{path}: empty file list or blank line")
    return paths


def scan_corpus(
    root: pathlib.Path, paths: Sequence[str], rng: random.Random
) -> tuple[list[str], list[bytes]]:
    """First pass: returns the sorted harvested definitions and a uniform
    sample of candidate lines."""
    definitions: set[bytes] = set()
    reservoir = Reservoir(LINE_RESERVOIR_SIZE, rng)
    for path in paths:
        data = (root / path).read_bytes()
        definitions |= harvest_definitions(path, data)
        for line in data.splitlines():
            if (candidate := candidate_line(line)) is not None:
                reservoir.offer(candidate)
    if not definitions:
        raise GenerationError("no definitions found in the listed files")
    if not reservoir.items:
        raise GenerationError("no line qualifies for literal, regex, or short queries")
    return sorted(d.decode("ascii") for d in definitions), reservoir.items


def count_and_verify(
    root: pathlib.Path,
    paths: Sequence[str],
    names: Iterable[str],
    candidates: Iterable[str],
) -> tuple[collections.Counter[str], set[str]]:
    """Second pass: returns the document frequency of each name (as a whole
    token) and the candidates that occur in no file."""
    wanted = {n.encode("ascii") for n in names}
    absent = {c.encode("ascii") for c in candidates}
    df: collections.Counter[bytes] = collections.Counter()
    for path in paths:
        data = (root / path).read_bytes()
        df.update(set(TOKEN_RE.findall(data)) & wanted)
        absent = {c for c in absent if c not in data}
    return (
        collections.Counter({k.decode("ascii"): v for k, v in df.items()}),
        {c.decode("ascii") for c in absent},
    )


def query(cls, kind, pattern, *, tercile=None, df=None, template=None):
    """Returns a query record without its id and case sensitivity."""
    return {
        "class": cls,
        "kind": kind,
        "pattern": pattern,
        "tercile": tercile,
        "df": df,
        "template": template,
    }


class QueryMix:
    """Query records under construction, with distinct patterns."""

    def __init__(self):
        self.records: list[dict] = []
        self._patterns: set[str] = set()

    def add(self, record: dict) -> None:
        """Adds record; its pattern must be new."""
        if not self.offer(record):
            raise GenerationError(f"duplicate pattern {record['pattern']!r}")

    def offer(self, record: dict | None) -> bool:
        """Adds record unless it is None or its pattern is already present."""
        if record is None or record["pattern"] in self._patterns:
            return False
        self._patterns.add(record["pattern"])
        self.records.append(record)
        return True

    def fill(self, cls: str, n: int, make: Callable[[], dict | None]) -> None:
        """Adds n records from make, which may return None or a duplicate."""
        added = 0
        for _ in range(100 * n):
            if added == n:
                return
            added += self.offer(make())
        if added < n:
            raise GenerationError(f"could only generate {added} of {n} {cls} queries")


def df_tercile(df: int, max_df: int) -> int:
    """Returns the log-scale document-frequency tercile of df: 1 below
    max_df^(1/3), 2 below max_df^(2/3), else 3. Integer arithmetic keeps the
    edges exact."""
    cube = df**3
    return 1 + (cube >= max_df) + (cube >= max_df**2)


def identifier_queries(
    names: Sequence[str], df: collections.Counter[str], n: int, rng: random.Random
) -> tuple[list[dict], list[list[int]]]:
    """Samples n names split evenly across the three df_tercile bands;
    returns the records and each band's [min, max] document frequency.

    Rank terciles would not stratify: most definitions occur in one file
    (73% in linux/kernel and mm), so the lower two rank terciles would both
    be df = 1 and common names, the costly ones, would be rare in the mix.
    """
    max_df = max(df[name] for name in names)
    bands: list[list[str]] = [[], [], []]
    for name in names:
        bands[df_tercile(df[name], max_df) - 1].append(name)
    records, ranges = [], []
    shares = allocate(n, [1, 1, 1])
    for tercile, (members, k) in enumerate(zip(bands, shares, strict=True), start=1):
        if len(members) < k:
            raise GenerationError(
                f"df tercile {tercile} has {len(members)} names, need {k}"
            )
        frequencies = [df[name] for name in members]
        ranges.append([min(frequencies), max(frequencies)])
        for name in rng.sample(members, k):
            records.append(
                query("identifier", "literal", name, tercile=tercile, df=df[name])
            )
    return records, ranges


def generate(
    root: pathlib.Path,
    paths: Sequence[str],
    *,
    corpus: str,
    count: int,
    seed: int,
    case_insensitive_share: float,
) -> tuple[list[dict], dict]:
    """Generates the query mix; returns the records and generation metadata.

    Reads every listed file twice: once to harvest definitions and sample
    lines, and once to count document frequencies and verify zero-result
    candidates.
    """
    rng = random.Random(seed)
    shares = allocate(count, [share for _, share in CLASS_SHARES])
    counts = {cls: n for (cls, _), n in zip(CLASS_SHARES, shares, strict=True)}
    ordered_paths = sorted(paths)
    names, lines = scan_corpus(root, ordered_paths, rng)

    parts = identifier_parts(names)
    candidates: list[str] = []
    for _ in range(100 * counts["zero"]):
        if len(candidates) == ZERO_CANDIDATES_PER_QUERY * counts["zero"]:
            break
        candidate = zero_candidate(parts, rng)
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)
    df, absent = count_and_verify(root, ordered_paths, names, candidates)

    mix = QueryMix()
    identifiers, tercile_ranges = identifier_queries(
        names, df, counts["identifier"], rng
    )
    for record in identifiers:
        mix.add(record)

    def make_literal():
        literal = literal_from_line(rng.choice(lines), rng)
        return None if literal is None else query("literal", "literal", literal)

    mix.fill("literal", counts["literal"], make_literal)

    templates = itertools.cycle(REGEX_TEMPLATES)

    def make_regex():
        name, template = next(templates)
        pattern = template(names, lines, rng)
        return (
            None if pattern is None else query("regex", "regex", pattern, template=name)
        )

    mix.fill("regex", counts["regex"], make_regex)

    short_identifiers, punctuation = short_pools(lines)
    short_shares = allocate(counts["short"], [1, 1])
    for source, k in zip((short_identifiers, punctuation), short_shares, strict=True):
        if len(source) < k:
            raise GenerationError(
                f"only {len(source)} distinct short strings, need {k}"
            )
        for pattern in rng.sample(source, k):
            mix.add(query("short", "literal", pattern))

    verified = [c for c in candidates if c in absent]
    if len(verified) < counts["zero"]:
        raise GenerationError(
            f"only {len(verified)} verified zero-result queries, need {counts['zero']}"
        )
    for pattern in verified[: counts["zero"]]:
        mix.add(query("zero", "literal", pattern))

    records = mix.records
    for record in records:
        record["case_sensitive"] = True
    for cls in ("identifier", "literal", "regex"):
        members = [r for r in records if r["class"] == cls]
        for record in rng.sample(members, round(case_insensitive_share * len(members))):
            record["case_sensitive"] = False

    rng.shuffle(records)
    width = max(4, len(str(len(records))))
    fields = ("class", "kind", "pattern", "case_sensitive", "tercile", "df", "template")
    ordered = [
        {"id": f"{corpus}-{i:0{width}d}", **{k: r[k] for k in fields}}
        for i, r in enumerate(records, start=1)
    ]
    meta = {
        "schema_version": SCHEMA_VERSION,
        "generator": "bench/queries/gen_mix.py",
        "python": platform.python_version(),
        "corpus": corpus,
        "seed": seed,
        "count": count,
        "class_counts": counts,
        "case_insensitive_share": case_insensitive_share,
        "files": len(paths),
        "definitions": len(names),
        "tercile_df_ranges": tercile_ranges,
    }
    return ordered, meta


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the generator on the command line argv; returns the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=pathlib.Path, required=True, help="corpus root")
    parser.add_argument(
        "--files", type=pathlib.Path, required=True, help="benchmark file list"
    )
    parser.add_argument(
        "--corpus", required=True, help="corpus name, the prefix of query ids"
    )
    parser.add_argument(
        "--out", type=pathlib.Path, required=True, help="output JSONL path"
    )
    parser.add_argument("--count", type=int, default=1000, help="number of queries")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--case-insensitive-share",
        type=float,
        default=0.25,
        help="share of identifier, literal, and regex queries that ignore case",
    )
    args = parser.parse_args(argv)
    if args.count < len(CLASS_SHARES):
        parser.error(f"--count must be at least {len(CLASS_SHARES)}")
    if not 0 <= args.case_insensitive_share <= 1:
        parser.error("--case-insensitive-share must be in [0, 1]")
    try:
        paths = read_file_list(args.files)
        records, meta = generate(
            args.root.expanduser(),
            paths,
            corpus=args.corpus,
            count=args.count,
            seed=args.seed,
            case_insensitive_share=args.case_insensitive_share,
        )
    except (GenerationError, OSError) as e:
        print(f"gen_mix: error: {e}", file=sys.stderr)
        return 1
    meta["files_sha256"] = hashlib.sha256(args.files.read_bytes()).hexdigest()
    with args.out.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record) + "\n")
    meta_path = args.out.with_name(args.out.name.removesuffix(".jsonl") + ".meta.json")
    meta_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

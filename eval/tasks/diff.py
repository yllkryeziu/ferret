# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""A parser for the unified diffs git produces.

It reads the file headers and the hunk ranges of `git diff` output, of the
patches benchmark datasets ship, and of `git format-patch` mails (whose
leading message text it skips). Hunk bodies are skipped by count, so a
removed line that happens to start with "--- " cannot be mistaken for a file
header.
"""

import dataclasses
import re

_HUNK_HEADER = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
# The escapes git's quote_c_style() emits besides three-digit octal.
_C_ESCAPES = {
    "a": 7,
    "b": 8,
    "t": 9,
    "n": 10,
    "v": 11,
    "f": 12,
    "r": 13,
    '"': 34,
    "\\": 92,
}
_C_ESCAPE = re.compile(r'\\([0-7]{3}|[abtnvfr"\\])')


@dataclasses.dataclass(frozen=True, slots=True)
class Hunk:
    """The line ranges of one hunk; counts of 0 mean a pure insertion or deletion.

    For a pure insertion (`old_count == 0`) git sets `old_start` to the line
    after which the new lines go, which is 0 at the top of the file.
    """

    old_start: int
    old_count: int
    new_start: int
    new_count: int

    def changed_old_lines(self) -> range:
        """Returns the preimage lines this hunk removes or replaces."""
        return range(self.old_start, self.old_start + self.old_count)


@dataclasses.dataclass(frozen=True, slots=True)
class FileDiff:
    """One file's part of a diff; a path is None when the file is added or deleted."""

    old_path: str | None
    new_path: str | None
    hunks: tuple[Hunk, ...]


def parse_unified_diff(text: str) -> list[FileDiff]:
    """Parses every file section of a unified diff.

    Binary and mode-only changes have no "---"/"+++" headers and are skipped.

    Raises:
        ValueError: if a hunk header is malformed or its body is truncated.
    """
    lines = text.splitlines()
    files: list[FileDiff] = []
    index = 0
    while index < len(lines):
        if not (lines[index].startswith("--- ") and index + 1 < len(lines)):
            index += 1
            continue
        if not lines[index + 1].startswith("+++ "):
            index += 1
            continue
        old_path = _header_path(lines[index][4:], "a/")
        new_path = _header_path(lines[index + 1][4:], "b/")
        index += 2
        hunks = []
        while index < len(lines) and lines[index].startswith("@@"):
            hunk = _parse_hunk_header(lines[index])
            hunks.append(hunk)
            index = _skip_hunk_body(lines, index + 1, hunk)
        files.append(FileDiff(old_path, new_path, tuple(hunks)))
    return files


def _parse_hunk_header(line: str) -> Hunk:
    match = _HUNK_HEADER.match(line)
    if match is None:
        raise ValueError(f"malformed hunk header {line!r}")
    old_start, old_count, new_start, new_count = match.groups()
    return Hunk(
        old_start=int(old_start),
        old_count=1 if old_count is None else int(old_count),
        new_start=int(new_start),
        new_count=1 if new_count is None else int(new_count),
    )


def _skip_hunk_body(lines: list[str], index: int, hunk: Hunk) -> int:
    """Returns the index of the first line after the body of `hunk`."""
    old_left, new_left = hunk.old_count, hunk.new_count
    while old_left > 0 or new_left > 0:
        if index >= len(lines):
            raise ValueError(f"truncated hunk {hunk}")
        line = lines[index]
        if line.startswith("-"):
            old_left -= 1
        elif line.startswith("+"):
            new_left -= 1
        elif not line.startswith("\\"):  # "\ No newline at end of file"
            old_left -= 1
            new_left -= 1
        index += 1
    while index < len(lines) and lines[index].startswith("\\"):
        index += 1
    return index


def _header_path(field: str, prefix: str) -> str | None:
    """Extracts the path from a "---" or "+++" header field."""
    field = field.split("\t", 1)[0]  # Plain diff(1) appends a timestamp.
    if field == "/dev/null":
        return None
    if field.startswith('"'):
        field = _unquote_c(field)
    return field.removeprefix(prefix)


def _unquote_c(quoted: str) -> str:
    """Decodes a path git C-quoted because it holds unusual bytes."""
    if len(quoted) < 2 or not quoted.endswith('"'):
        raise ValueError(f"malformed quoted path {quoted!r}")
    raw = bytearray()
    position = 1
    for match in _C_ESCAPE.finditer(quoted, 1, len(quoted) - 1):
        raw += quoted[position : match.start()].encode("utf-8", "surrogateescape")
        escape = match.group(1)
        raw.append(int(escape, 8) if len(escape) == 3 else _C_ESCAPES[escape])
        position = match.end()
    raw += quoted[position:-1].encode("utf-8", "surrogateescape")
    return raw.decode("utf-8", "surrogateescape")

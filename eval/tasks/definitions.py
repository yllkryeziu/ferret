# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Finds the top-level C-family definitions a diff changes.

Git's hunk header names only the definition above a hunk's first line, which
is the wrong one when a hunk starts on a signature or spans two functions,
and it names global statements such as `EXPORT_SYMBOL(foo);` as if they were
functions. This module instead scans the whole preimage once and assigns
every line to the top-level definition that encloses it, then maps the
changed lines of each -U0 hunk onto those spans.

The scan relies on a convention that both Linux and Google C++ style follow:
a top-level definition starts in column 0 and its closing brace is in column
0. A definition line is a column-0 line that begins with an identifier, is
not a label, a preprocessor line, or a `namespace`/`using`/`extern "C"` line,
and does not end in ";". Its name is the (qualified) identifier before the
first "(" for functions, before "[" or "=" for data such as tables, and after
`struct`/`class`/`union`/`enum` for types; lines such as `static int` or
`template <typename T>` that precede the named line in the same definition
extend its span upward. A column-0 "}" or ";"-terminated line ends the
current definition. Lines inside block comments and preprocessor
continuations are ignored.

The result is a heuristic with known misses: definitions indented inside a
class body are attributed to the class, Objective-C methods get no name, and
macros that define functions, such as SYSCALL_DEFINE3(open, ...), are named
after the macro. The protocol therefore scores functions as a secondary
endpoint only.
"""

import dataclasses
import re
from collections.abc import Iterable

from tasks.diff import Hunk

_LABEL = re.compile(r"[A-Za-z_]\w*:(?!:)\s*($|//|/\*)")
_NON_DEFINITION_PREFIX = re.compile(r'(namespace|using|extern\s+"C"(\+\+)?)\b')
_TEMPLATE_ARGUMENTS = re.compile(r"<[^<>]*>")
_QUALIFIED_NAME = re.compile(r"((?:[A-Za-z_]\w*\s*::\s*)*(?:~\s*)?[A-Za-z_]\w*)\s*$")
_OPERATOR_NAME = re.compile(
    r"((?:[A-Za-z_]\w*\s*::\s*)*operator\s*(?:\(\)|[^\s(]+))\s*\("
)
_TYPE_NAME = re.compile(r"\b(?:struct|class|union|enum)\b(?:\s+[A-Za-z_]\w*)*")
_TRAILING_COMMENT = re.compile(r"\s*(//.*|/\*.*)$")


@dataclasses.dataclass(frozen=True, slots=True)
class DefinitionSpan:
    """A named top-level definition covering preimage lines [first, last]."""

    name: str
    first: int
    last: int


def definition_spans(source: str) -> list[DefinitionSpan]:
    """Returns the top-level definitions of a C-family file, in order.

    Line numbers are 1-based. Unnamed definitions, such as an anonymous
    struct, get no span. Runs in time linear in the size of `source`.
    """
    spans: list[DefinitionSpan] = []
    name: str | None = None  # The open definition, if any.
    first = 0
    prefix_start: int | None = None  # First line of an unnamed lead-in.
    in_comment = False
    in_continuation = False
    # Split on "\n" only, as git numbers lines: form feeds are not breaks, and
    # a final "\n" ends the last line rather than starting another.
    lines = [line.removesuffix("\r") for line in source.split("\n")]
    if lines and not lines[-1]:
        lines.pop()

    def close(last: int) -> None:
        nonlocal name
        if name is not None:
            spans.append(DefinitionSpan(name, first, last))
        name = None

    for number, line in enumerate(lines, start=1):
        if in_continuation:
            in_continuation = line.endswith("\\")
            continue
        if in_comment:
            in_comment = "*/" not in line
            continue
        if line.startswith("#"):
            in_continuation = line.endswith("\\")
            continue
        code = _TRAILING_COMMENT.sub("", line)
        if line.startswith("/*"):
            in_comment = "*/" not in line
            continue
        if line.startswith("}"):
            close(number)
            prefix_start = None
            continue
        if not line[:1].isalpha() and not line.startswith("_"):
            continue  # Indented or blank: stays in the open definition.
        if (
            code.endswith(";")
            or _LABEL.match(line)
            or _NON_DEFINITION_PREFIX.match(line)
        ):
            if not _LABEL.match(line):
                close(number - 1)
                prefix_start = None
            continue
        defined = _defined_name(code)
        if defined is None:
            if prefix_start is None:
                close(number - 1)
                prefix_start = number
            continue
        close((prefix_start or number) - 1)
        name, first = defined, prefix_start or number
        prefix_start = None
    close(len(lines))
    return spans


def changed_definitions(source: str, hunks: Iterable[Hunk]) -> set[str]:
    """Returns the names of the definitions in `source` that `hunks` change.

    `source` is the preimage and `hunks` come from a -U0 diff against it. A
    replaced or removed line belongs to the definition that encloses it. An
    inserted line belongs to a definition only if the lines on both sides of
    the insertion point do, since text added between two definitions (a new
    function, say) changes neither.
    """
    owner: dict[int, str] = {}
    for span in definition_spans(source):
        for line in range(span.first, span.last + 1):
            owner[line] = span.name
    names = set()
    for hunk in hunks:
        if hunk.old_count > 0:
            names.update(
                owner[line] for line in hunk.changed_old_lines() if line in owner
            )
            continue
        before = owner.get(hunk.old_start)
        if before is not None and before == owner.get(hunk.old_start + 1):
            names.add(before)
    return names


def _defined_name(code: str) -> str | None:
    """Returns the name a column-0 definition line defines, if it names one."""
    code = _strip_template_arguments(code)
    operator = _OPERATOR_NAME.search(code)
    if operator is not None:
        return re.sub(r"\s+", "", operator.group(1))
    delimiters = [position for position in map(code.find, "([=") if position >= 0]
    if delimiters:
        match = _QUALIFIED_NAME.search(code[: min(delimiters)])
        return re.sub(r"\s+", "", match.group(1)) if match else None
    head = re.split(r"\{|(?<!:):(?!:)", code, maxsplit=1)[0]  # Before the bases.
    match = _TYPE_NAME.search(head)
    if match is None:
        return None
    words = [word for word in match.group(0).split() if word != "final"]
    return words[-1] if len(words) > 1 else None


def _strip_template_arguments(code: str) -> str:
    """Removes <...> groups so that `Foo<T>::Bar` reads as `Foo::Bar`."""
    while True:
        stripped = _TEMPLATE_ARGUMENTS.sub("", code)
        if stripped == code:
            return code
        code = stripped

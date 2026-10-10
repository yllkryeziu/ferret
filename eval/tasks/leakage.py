# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Rejects rewritten prompts that name their own answer.

A commit-derived prompt leaks when it names a gold function or a gold path
component (a directory, the file name, or its stem). Function names are
matched as whole identifiers, case-sensitively, including each segment of a
qualified name: `Foo::Bar` leaks through "Foo" or "Bar", but the English word
"bar" does not.

Path components are matched as whole tokens, case-insensitively, but only
when they are specific. Taken literally, "names a gold path component" would
reject nearly every Chromium report for saying "Chrome" or "browser" and
every Linux report for saying "kernel", even though those words narrow the
search to hundreds of thousands of files. A component therefore leaks only if
it identifies at most `MAX_COMPONENT_FILES` tracked files in the snapshot:
"video_detect" (1 file) leaks, "acpi" (hundreds) does not. Everything a
component-level rule lets through is still subject to the lexical-solvability
cap (`tasks.lexical`).
"""

import posixpath
import re
from collections import Counter
from collections.abc import Iterable

from tasks.schema import split_function_id

# A path component that identifies at most this many tracked files counts as
# naming the gold location. Fixed before the test split runs.
MAX_COMPONENT_FILES = 100

_NAME_SEGMENT = re.compile(r"::|\.")


def path_components(path: str) -> set[str]:
    """Returns the lowercased directories, file name, and stem of `path`."""
    parts = path.lower().split("/")
    stem, _ = posixpath.splitext(parts[-1])
    return {*parts, stem} - {""}


class PathComponentIndex:
    """How many tracked files each path component identifies. Immutable."""

    def __init__(self, paths: Iterable[str]) -> None:
        """Counts, for every component, the files whose path contains it."""
        self._files = Counter(
            component for path in paths for component in path_components(path)
        )

    def file_count(self, component: str) -> int:
        """Returns the number of indexed files with `component` in their path."""
        return self._files[component.lower()]


def find_leaks(
    prompt: str,
    gold_paths: Iterable[str],
    gold_functions: Iterable[str],
    components: PathComponentIndex,
    *,
    max_component_files: int = MAX_COMPONENT_FILES,
) -> list[str]:
    """Returns the gold names that `prompt` mentions, sorted; empty if none.

    Args:
        prompt: The candidate problem statement.
        gold_paths: The task's gold paths.
        gold_functions: The task's "path::name" gold function ids.
        components: Component counts over the snapshot's tracked files.
        max_component_files: Components identifying more files than this are
            too generic to leak.
    """
    leaks = set()
    # Multiword product names are allowed; only exact gold path components
    # (including driver abbreviations such as efa) and functions leak.
    for path in gold_paths:
        for component in path_components(path):
            if components.file_count(component) > max_component_files:
                continue
            if _mentions(prompt, component, re.IGNORECASE):
                leaks.add(component)
    for function_id in gold_functions:
        _, name = split_function_id(function_id)
        for segment in {name, *_NAME_SEGMENT.split(name)} - {""}:
            if _mentions(prompt, segment, 0):
                leaks.add(segment)
    return sorted(leaks)


def _mentions(text: str, word: str, flags: int) -> bool:
    """Returns whether `word` occurs in `text` not adjacent to identifier chars."""
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(word)}(?![A-Za-z0-9_])"
    return re.search(pattern, text, flags) is not None

# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""The lexical-solvability check and the per-bucket cap on solvable tasks.

A task is lexically solvable when one grep for the identifiers in its prompt
already localizes it, which is exactly the case in which every search arm
would saturate. The oracle models that grep: take every identifier of four
or more characters in the prompt (maximal runs of [A-Za-z0-9_] not starting
with a digit), match them as whole words case-insensitively against every
text file of the snapshot, and rank files by how many distinct identifiers
they contain. The task is solvable if every gold file contains at least one
identifier and ranks in the top `TOP_K` even when all ties break against it,
i.e. at most `TOP_K` files score at least as high. Case-insensitive matching
mirrors the search tool's default and makes the oracle stronger, which is the
conservative direction for a cap.

Text files follow Ferret's eligibility defaults: regular files of at most
1 MiB without a NUL byte in their first 8 KiB.

The oracle tokenizes each file once and keeps postings only for the
vocabulary it was built with (the union of the prompts to be checked), so
checking a task costs one bincount over its identifiers' postings.
"""

import dataclasses
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from fractions import Fraction
from pathlib import Path

import numpy as np

from tasks.schema import Bucket, Task
from tasks.split import hash_order

TOP_K = 5
MAX_SOLVABLE_FRACTION = Fraction(3, 10)
CAP_SALT = "ferret-eval-lexical-cap-v1"

MAX_FILE_BYTES = 1 << 20
_BINARY_PROBE_BYTES = 8 << 10
_MIN_IDENTIFIER = 4
_WORD = re.compile(rb"[a-z0-9_]+")


def prompt_identifiers(prompt: str) -> frozenset[str]:
    """Returns the lowercased identifiers of `prompt` that the oracle greps."""
    return frozenset(
        word.decode()
        for word in _WORD.findall(prompt.lower().encode("ascii", "replace"))
        if len(word) >= _MIN_IDENTIFIER and not word[:1].isdigit()
    )


def iter_text_files(root: Path) -> Iterator[tuple[str, bytes]]:
    """Yields (relative POSIX path, content) for each eligible file under `root`.

    Skips the `.git` directory and symbolic links. Paths come out sorted, so
    the oracle's file numbering is deterministic.
    """
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = sorted(name for name in subdirectories if name != ".git")
        for name in sorted(files):
            path = Path(directory, name)
            if path.is_symlink() or path.stat().st_size > MAX_FILE_BYTES:
                continue
            content = path.read_bytes()
            if b"\0" not in content[:_BINARY_PROBE_BYTES]:
                yield path.relative_to(root).as_posix(), content


class LexicalOracle:
    """A single-grep localizer over one snapshot. Immutable after construction."""

    def __init__(
        self, files: Iterable[tuple[str, bytes]], vocabulary: Iterable[str]
    ) -> None:
        """Indexes `files` for the identifiers in `vocabulary`.

        Args:
            files: (path, content) pairs, such as `iter_text_files(root)`.
            vocabulary: Every identifier any later query may contain, as
                returned by `prompt_identifiers`.
        """
        wanted = {word.encode() for word in vocabulary}
        postings: dict[bytes, list[int]] = {word: [] for word in wanted}
        self._paths: list[str] = []
        for number, (path, content) in enumerate(files):
            self._paths.append(path)
            for word in wanted.intersection(_WORD.findall(content.lower())):
                postings[word].append(number)
        self._postings = {
            word.decode(): np.asarray(files_with_word, dtype=np.int64)
            for word, files_with_word in postings.items()
        }
        self._number = {path: number for number, path in enumerate(self._paths)}

    def is_solvable(
        self, prompt: str, gold_paths: Iterable[str], *, k: int = TOP_K
    ) -> bool:
        """Returns whether the grep ranks every gold path in the top `k`.

        Raises:
            KeyError: if the prompt has an identifier outside the vocabulary.
        """
        words = prompt_identifiers(prompt)
        if not words:
            return False
        scores = np.bincount(
            np.concatenate([self._postings[word] for word in words]),
            minlength=len(self._paths),
        )
        for path in gold_paths:
            number = self._number.get(path)
            if number is None or scores[number] == 0:
                return False
            if np.count_nonzero(scores >= scores[number]) > k:
                return False
        return True


def assess(tasks: Iterable[Task], oracle: LexicalOracle) -> list[Task]:
    """Returns `tasks` with `lexically_solvable` set by `oracle`."""
    return [
        dataclasses.replace(
            task, lexically_solvable=oracle.is_solvable(task.prompt, task.gold_paths)
        )
        for task in tasks
    ]


def apply_cap(
    tasks: Iterable[Task],
    *,
    max_fraction: Fraction = MAX_SOLVABLE_FRACTION,
    salt: str = CAP_SALT,
) -> tuple[list[Task], list[Task]]:
    """Drops lexically solvable tasks until each bucket is within the cap.

    Per bucket with u unsolvable tasks, keeps the floor(f·u / (1 - f))
    solvable tasks that come first in salted-hash order, the largest number
    that keeps the solvable share at or below f. Order is otherwise preserved.

    Returns:
        (kept, dropped).

    Raises:
        ValueError: if a task has not been assessed.
    """
    tasks = list(tasks)
    unsolvable: Counter[Bucket] = Counter()
    solvable: defaultdict[Bucket, list[Task]] = defaultdict(list)
    for task in tasks:
        if task.lexically_solvable is None:
            raise ValueError(f"task {task.id}: lexical solvability not assessed")
        if task.lexically_solvable:
            solvable[task.bucket].append(task)
        else:
            unsolvable[task.bucket] += 1
    kept_ids = set()
    for bucket, candidates in solvable.items():
        limit = math.floor(max_fraction * unsolvable[bucket] / (1 - max_fraction))
        ordered = hash_order(candidates, key=lambda task: task.id, salt=salt)
        kept_ids.update(task.id for task in ordered[:limit])
    kept = [
        task for task in tasks if not task.lexically_solvable or task.id in kept_ids
    ]
    dropped = [
        task for task in tasks if task.lexically_solvable and task.id not in kept_ids
    ]
    return kept, dropped

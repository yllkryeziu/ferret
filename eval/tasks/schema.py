# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""The task record shared by every task source, and its JSONL encoding.

A `Task` is immutable and validated on construction, so any instance that
exists is well formed: its snapshot is a full SHA-1 commit id, its gold paths
are normalized repository-relative POSIX paths in sorted order, and every gold
function names one of its gold paths. Task files are JSON Lines, one task per
line, and reading one rejects unknown or missing fields rather than guessing.
"""

import dataclasses
import enum
import json
import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any


class Bucket(enum.StrEnum):
    """Task strata."""

    S = "S"  # LocBench: small Python repositories.
    L = "L"  # Linux: commit-derived tasks plus LinuxFLBench.
    XL = "XL"  # Chromium: commit-derived tasks.
    E = "E"  # Enumeration and definition tasks (optional).
    P = "P"  # SWE-bench Verified contamination probe, arm A0 only.


class Split(enum.StrEnum):
    """Dev tasks may be used for tuning; test tasks stay frozen until the run."""

    DEV = "dev"
    TEST = "test"


class Source(enum.StrEnum):
    """Where a task came from."""

    LOCBENCH = "locbench"
    LINUXFLBENCH = "linuxflbench"
    SWEBENCH_VERIFIED = "swebench_verified"
    COMMIT = "commit"


class PromptOrigin(enum.StrEnum):
    """How the prompt text was obtained."""

    ISSUE = "issue"  # A natural issue or bug report, verbatim.
    REWRITE = "rewrite"  # A commit message rewritten into a symptom report.


# Separates the path from the qualified name in a gold function id. A path
# never contains it; a C++ name may, so ids split at the first occurrence.
FUNCTION_SEPARATOR = "::"

_COMMIT_ID = re.compile(r"[0-9a-f]{40}")
_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/+-]*")


def is_normalized_path(path: str) -> bool:
    """Returns whether `path` is a normalized repository-relative POSIX path.

    Normalized means non-empty, relative, and free of empty, "." and ".."
    components, so that equal files always compare equal as strings.
    """
    if not path or path.startswith("/") or FUNCTION_SEPARATOR in path:
        return False
    return all(part not in ("", ".", "..") for part in path.split("/"))


def split_function_id(function_id: str) -> tuple[str, str]:
    """Splits "path::qualified_name" into its path and name.

    Raises:
        ValueError: if either part is empty or the separator is missing.
    """
    path, separator, name = function_id.partition(FUNCTION_SEPARATOR)
    if not separator or not path or not name:
        raise ValueError(f"function id {function_id!r}: expected 'path::name'")
    return path, name


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Provenance:
    """Where a task came from, and when its problem became public.

    Attributes:
        source: The task source.
        source_id: The source's own identifier: a benchmark instance id, a
            Bugzilla id, or the full id of the fix commit.
        created_at: When the issue was filed or the fix was authored; compared
            with a model's training cutoff to separate clean tasks from
            possibly contaminated ones. Must be timezone-aware.
        prompt_origin: How the prompt text was obtained.
        rewriter: Identifies the rewriting model and prompt version; required
            exactly when `prompt_origin` is REWRITE.
    """

    source: Source
    source_id: str
    created_at: datetime
    prompt_origin: PromptOrigin
    rewriter: str = ""

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("provenance: empty source_id")
        if self.created_at.tzinfo is None:
            raise ValueError(
                f"provenance {self.source_id}: created_at {self.created_at} has no "
                "timezone"
            )
        if (self.prompt_origin is PromptOrigin.REWRITE) != bool(self.rewriter):
            raise ValueError(
                f"provenance {self.source_id}: rewriter must be set exactly for "
                "rewritten prompts"
            )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Task:
    """One localization task: a problem statement and where it is fixed.

    Attributes:
        id: Unique, stable id such as "linux/3f2a9c1d0b7e"; the dev/test split
            is a function of it (`tasks.split`).
        bucket: The stratum the task belongs to.
        repo: Repository name, such as "linux" or "django/django".
        snapshot: Full id of the commit whose tree the agent searches.
        prompt: The problem statement shown to the agent.
        gold_paths: Files the fix changes that exist in the snapshot; sorted,
            unique, normalized, and non-empty.
        gold_functions: "path::qualified_name" ids of the definitions the fix
            changes; sorted and unique, each path one of `gold_paths`. Empty
            when the source provides no function-level gold.
        provenance: Where the task came from.
        lexically_solvable: Whether a single grep of the prompt's identifiers
            ranks every gold path in the top k (`tasks.lexical`); None until
            assessed against the snapshot, and for buckets it does not apply to.
        split: DEV or TEST.
    """

    id: str
    bucket: Bucket
    repo: str
    snapshot: str
    prompt: str
    gold_paths: tuple[str, ...]
    gold_functions: tuple[str, ...]
    provenance: Provenance
    lexically_solvable: bool | None
    split: Split

    def __post_init__(self) -> None:
        if not _TASK_ID.fullmatch(self.id):
            raise ValueError(f"task id {self.id!r}: not of the form [A-Za-z0-9._/+-]+")
        if not self.repo:
            raise ValueError(f"task {self.id}: empty repo")
        if not _COMMIT_ID.fullmatch(self.snapshot):
            raise ValueError(
                f"task {self.id}: snapshot {self.snapshot!r} is not a full lowercase "
                "SHA-1 commit id"
            )
        if not self.prompt.strip():
            raise ValueError(f"task {self.id}: empty prompt")
        if not self.gold_paths:
            raise ValueError(f"task {self.id}: no gold paths")
        _check_sorted_unique(self.id, "gold_paths", self.gold_paths)
        for path in self.gold_paths:
            if not is_normalized_path(path):
                raise ValueError(
                    f"task {self.id}: gold path {path!r} is not normalized"
                )
        _check_sorted_unique(self.id, "gold_functions", self.gold_functions)
        for function_id in self.gold_functions:
            path, name = split_function_id(function_id)
            if path not in self.gold_paths:
                raise ValueError(
                    f"task {self.id}: gold function {function_id!r} is not in a gold "
                    "path"
                )
            if "\n" in name:
                raise ValueError(
                    f"task {self.id}: gold function {function_id!r} spans lines"
                )

    def to_json(self) -> dict[str, Any]:
        """Returns the JSON object for this task; `from_json` inverts it."""
        return {
            "id": self.id,
            "bucket": self.bucket.value,
            "repo": self.repo,
            "snapshot": self.snapshot,
            "prompt": self.prompt,
            "gold_paths": list(self.gold_paths),
            "gold_functions": list(self.gold_functions),
            "provenance": {
                "source": self.provenance.source.value,
                "source_id": self.provenance.source_id,
                "created_at": self.provenance.created_at.isoformat(),
                "prompt_origin": self.provenance.prompt_origin.value,
                "rewriter": self.provenance.rewriter,
            },
            "lexically_solvable": self.lexically_solvable,
            "split": self.split.value,
        }

    @classmethod
    def from_json(cls, obj: Any) -> "Task":
        """Parses a task from its JSON object.

        Raises:
            ValueError: on a missing, unknown, or mistyped field, or on any
                violated invariant.
        """
        fields = _Fields(obj, "task")
        provenance = _Fields(fields.take("provenance", dict), "provenance")
        solvable = fields.take("lexically_solvable", (bool, type(None)))
        task = cls(
            id=fields.take("id", str),
            bucket=Bucket(fields.take("bucket", str)),
            repo=fields.take("repo", str),
            snapshot=fields.take("snapshot", str),
            prompt=fields.take("prompt", str),
            gold_paths=tuple(fields.take_strings("gold_paths")),
            gold_functions=tuple(fields.take_strings("gold_functions")),
            provenance=Provenance(
                source=Source(provenance.take("source", str)),
                source_id=provenance.take("source_id", str),
                created_at=datetime.fromisoformat(provenance.take("created_at", str)),
                prompt_origin=PromptOrigin(provenance.take("prompt_origin", str)),
                rewriter=provenance.take("rewriter", str),
            ),
            lexically_solvable=solvable,
            split=Split(fields.take("split", str)),
        )
        fields.check_consumed()
        provenance.check_consumed()
        return task


def write_tasks(path: Path, tasks: Iterable[Task]) -> None:
    """Writes `tasks` to `path` as JSON Lines, in the given order.

    Raises:
        ValueError: if two tasks share an id.
    """
    tasks = list(tasks)
    check_unique_ids(tasks)
    with path.open("w", encoding="utf-8") as out:
        for task in tasks:
            out.write(json.dumps(task.to_json(), ensure_ascii=False) + "\n")


def read_tasks(path: Path) -> list[Task]:
    """Reads a JSON Lines task file written by `write_tasks`.

    Raises:
        ValueError: naming the line of the first malformed task, or if two
            tasks share an id.
    """
    tasks = []
    with path.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            try:
                tasks.append(Task.from_json(json.loads(line)))
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    check_unique_ids(tasks)
    return tasks


def check_unique_ids(tasks: Iterable[Task]) -> None:
    """Raises ValueError if two of `tasks` share an id."""
    seen: set[str] = set()
    for task in tasks:
        if task.id in seen:
            raise ValueError(f"duplicate task id {task.id}")
        seen.add(task.id)


def _check_sorted_unique(task_id: str, name: str, values: tuple[str, ...]) -> None:
    if list(values) != sorted(set(values)):
        raise ValueError(f"task {task_id}: {name} must be sorted and unique")


class _Fields:
    """Checked access to the fields of one JSON object."""

    def __init__(self, obj: Any, what: str) -> None:
        if not isinstance(obj, Mapping):
            raise ValueError(f"{what}: expected a JSON object")
        self._obj = obj
        self._what = what
        self._taken: set[str] = set()

    def take(self, key: str, kind: type | tuple[type, ...]) -> Any:
        if key not in self._obj:
            raise ValueError(f"{self._what}: missing field {key!r}")
        value = self._obj[key]
        if not isinstance(value, kind):
            raise ValueError(
                f"{self._what}: field {key!r} has type {type(value).__name__}"
            )
        self._taken.add(key)
        return value

    def take_strings(self, key: str) -> list[str]:
        values = self.take(key, list)
        if not all(isinstance(value, str) for value in values):
            raise ValueError(f"{self._what}: field {key!r} must hold strings")
        return values

    def check_consumed(self) -> None:
        unknown = sorted(set(self._obj) - self._taken)
        if unknown:
            raise ValueError(f"{self._what}: unknown fields {unknown}")

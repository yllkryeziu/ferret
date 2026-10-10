# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Loaders for the public benchmarks. For now, LocBench.

Each loader turns the benchmark's own JSON Lines rows into tasks and keeps
only rows whose fix changes 1 to `MAX_GOLD_PATHS` gold files (`tasks.paths`).
Sampling to the protocol's sizes is a separate, seeded step:
`hash_order(tasks, key=lambda task: task.id, salt=SAMPLE_SALT)[:n]`.

LocBench (bucket S): https://huggingface.co/datasets/czlll/Loc-Bench_V1,
revision c44cf3b74e07ca642cec841b471a9939907c12a7, 560 rows, split "test".
Export it with the `datasets` package of the eval environment:

    uv run python -c "import datasets; datasets.load_dataset(
        'czlll/Loc-Bench_V1', split='test',
        revision='c44cf3b74e07ca642cec841b471a9939907c12a7'
    ).to_json('locbench.jsonl')"
"""

import json
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tasks.diff import parse_unified_diff
from tasks.paths import MAX_GOLD_PATHS, is_gold_candidate
from tasks.schema import (
    FUNCTION_SEPARATOR,
    Bucket,
    PromptOrigin,
    Provenance,
    Source,
    Task,
)
from tasks.split import assign_split

SAMPLE_SALT = "ferret-eval-sample-v1"
LOCBENCH_TASKS = 100


def load_locbench(path: Path) -> list[Task]:
    """Loads LocBench rows as bucket-S tasks.

    The prompt is the issue text without hints. Gold functions come from the
    row's `edit_functions` ("path:Qualified.name"); `added_functions` are
    ignored because they do not exist at the base commit.
    """
    tasks = []
    for row in _read_rows(path):
        gold_paths = _gold_paths_from_patch(_string(row, "patch"))
        if not 1 <= len(gold_paths) <= MAX_GOLD_PATHS:
            continue
        gold_functions = set()
        for entry in _strings(row, "edit_functions"):
            file, _, name = entry.partition(":")
            if file in gold_paths and name:
                gold_functions.add(f"{file}{FUNCTION_SEPARATOR}{name}")
        task_id = f"locbench/{_string(row, 'instance_id')}"
        tasks.append(
            Task(
                id=task_id,
                bucket=Bucket.S,
                repo=_string(row, "repo"),
                snapshot=_string(row, "base_commit"),
                prompt=_string(row, "problem_statement"),
                gold_paths=gold_paths,
                gold_functions=tuple(sorted(gold_functions)),
                provenance=Provenance(
                    source=Source.LOCBENCH,
                    source_id=_string(row, "instance_id"),
                    created_at=datetime.fromtimestamp(
                        _int(row, "created_at") / 1000, UTC
                    ),
                    prompt_origin=PromptOrigin.ISSUE,
                ),
                lexically_solvable=None,
                split=assign_split(task_id),
            )
        )
    return tasks


def _gold_paths_from_patch(patch: str) -> tuple[str, ...]:
    """Returns the gold candidates a patch changes that exist before it."""
    return tuple(
        sorted(
            {
                diff.old_path
                for diff in parse_unified_diff(patch)
                if diff.old_path is not None and is_gold_candidate(diff.old_path)
            }
        )
    )


def _read_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield row


def _row_name(row: Mapping[str, Any]) -> str:
    return str(row.get("instance_id", row.get("id")))


def _string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"row {_row_name(row)}: {key!r} is not a string")
    return value


def _int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"row {_row_name(row)}: {key!r} is not an integer")
    return value


def _strings(row: Mapping[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"row {_row_name(row)}: {key!r} is not a list of strings")
    return value


# SWE-bench Verified, test split (500 rows), pinned to the dataset commit.
SWEBENCH_VERIFIED_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
SWEBENCH_PROBE_TASKS = 50


def load_swebench_verified(path: Path) -> list[Task]:
    """Loads eligible Verified issues, without hints or prompt rewriting."""
    tasks = []
    for row in _read_rows(path):
        gold_paths = _gold_paths_from_patch(_string(row, "patch"))
        if not 1 <= len(gold_paths) <= MAX_GOLD_PATHS:
            continue
        task_id = f"swebench_verified/{_string(row, 'instance_id')}"
        tasks.append(
            Task(
                id=task_id,
                bucket=Bucket.P,
                repo=_string(row, "repo"),
                snapshot=_string(row, "base_commit"),
                prompt=_string(row, "problem_statement"),
                gold_paths=gold_paths,
                gold_functions=(),
                provenance=Provenance(
                    source=Source.SWEBENCH_VERIFIED,
                    source_id=_string(row, "instance_id"),
                    created_at=datetime.fromisoformat(
                        _string(row, "created_at").replace("Z", "+00:00")
                    ),
                    prompt_origin=PromptOrigin.ISSUE,
                ),
                lexically_solvable=None,
                split=assign_split(task_id),
            )
        )
    return tasks

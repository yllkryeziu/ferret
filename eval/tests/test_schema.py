# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tasks.schema: task validation and the JSON Lines format."""

import dataclasses
import json
from datetime import UTC, datetime

import pytest

from tasks.schema import (
    Bucket,
    PromptOrigin,
    Provenance,
    Source,
    Split,
    Task,
    is_normalized_path,
    read_tasks,
    split_function_id,
    write_tasks,
)

_SNAPSHOT = "0123456789abcdef0123456789abcdef01234567"


def _task(**changes) -> Task:
    fields = {
        "id": "locbench/django__django-1",
        "bucket": Bucket.S,
        "repo": "django/django",
        "snapshot": _SNAPSHOT,
        "prompt": "QuerySet.only() crashes after select_related().",
        "gold_paths": ("django/db/models/query.py", "django/db/models/sql/query.py"),
        "gold_functions": ("django/db/models/query.py::QuerySet.only",),
        "provenance": Provenance(
            source=Source.LOCBENCH,
            source_id="django__django-1",
            created_at=datetime(2025, 3, 1, 12, tzinfo=UTC),
            prompt_origin=PromptOrigin.ISSUE,
        ),
        "lexically_solvable": None,
        "split": Split.DEV,
    }
    return Task(**(fields | changes))


def test_json_round_trip():
    task = _task(lexically_solvable=True)
    encoded = json.loads(json.dumps(task.to_json()))
    assert Task.from_json(encoded) == task


def test_file_round_trip(tmp_path):
    tasks = [_task(), _task(id="locbench/other", prompt="Ünïcode prompt ✓")]
    path = tmp_path / "tasks.jsonl"
    write_tasks(path, tasks)
    assert read_tasks(path) == tasks
    assert "✓" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"id": "-leading-dash"}, "task id"),
        ({"id": "has space"}, "task id"),
        ({"repo": ""}, "empty repo"),
        ({"snapshot": _SNAPSHOT[:12]}, "snapshot"),
        ({"snapshot": _SNAPSHOT.upper()}, "snapshot"),
        ({"prompt": " \n"}, "empty prompt"),
        ({"gold_paths": (), "gold_functions": ()}, "no gold paths"),
        ({"gold_paths": ("b.py", "a.py"), "gold_functions": ()}, "sorted"),
        ({"gold_paths": ("a.py", "a.py"), "gold_functions": ()}, "sorted"),
        ({"gold_paths": ("src/../a.py",), "gold_functions": ()}, "not normalized"),
        ({"gold_functions": ("other.py::f",)}, "not in a gold path"),
        ({"gold_functions": ("django/db/models/query.py::f\ng",)}, "spans lines"),
        ({"gold_functions": ("django/db/models/query.py",)}, "path::name"),
    ],
)
def test_rejects_invalid_tasks(changes, message):
    with pytest.raises(ValueError, match=message):
        _task(**changes)


def test_rejects_invalid_provenance():
    with pytest.raises(ValueError, match="no timezone"):
        Provenance(
            source=Source.COMMIT,
            source_id="abc",
            created_at=datetime(2025, 1, 1),
            prompt_origin=PromptOrigin.ISSUE,
        )
    with pytest.raises(ValueError, match="rewriter"):
        Provenance(
            source=Source.COMMIT,
            source_id="abc",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            prompt_origin=PromptOrigin.REWRITE,
        )
    with pytest.raises(ValueError, match="rewriter"):
        Provenance(
            source=Source.LOCBENCH,
            source_id="abc",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            prompt_origin=PromptOrigin.ISSUE,
            rewriter="model-v1",
        )


@pytest.mark.parametrize(
    "path", ["a.py", "src/a.py", ".github/workflows/ci.py", "a..b/c.py"]
)
def test_normalized_paths(path):
    assert is_normalized_path(path)


@pytest.mark.parametrize(
    "path", ["", "/abs.py", "a//b.py", "./a.py", "a/./b.py", "a/../b.py", "a/", "x::y"]
)
def test_unnormalized_paths(path):
    assert not is_normalized_path(path)


def test_function_ids_split_at_the_first_separator():
    assert split_function_id("net/url.cc::net::GURL::Resolve") == (
        "net/url.cc",
        "net::GURL::Resolve",
    )


def test_from_json_rejects_malformed_objects():
    good = _task().to_json()
    with pytest.raises(ValueError, match="expected a JSON object"):
        Task.from_json([good])
    with pytest.raises(ValueError, match="missing field 'repo'"):
        Task.from_json({key: value for key, value in good.items() if key != "repo"})
    with pytest.raises(ValueError, match=r"unknown fields \['extra'\]"):
        Task.from_json(good | {"extra": 1})
    with pytest.raises(ValueError, match=r"provenance: unknown fields \['x'\]"):
        Task.from_json(good | {"provenance": good["provenance"] | {"x": 1}})
    with pytest.raises(ValueError, match="field 'gold_paths' has type str"):
        Task.from_json(good | {"gold_paths": "a.py"})
    with pytest.raises(ValueError, match="must hold strings"):
        Task.from_json(good | {"gold_functions": [1]})
    with pytest.raises(ValueError, match="'lexically_solvable' has type int"):
        Task.from_json(good | {"lexically_solvable": 1})
    with pytest.raises(ValueError, match="'Q' is not a valid Bucket"):
        Task.from_json(good | {"bucket": "Q"})


def test_read_tasks_names_the_bad_line(tmp_path):
    path = tmp_path / "tasks.jsonl"
    bad = _task().to_json() | {"snapshot": "HEAD"}
    path.write_text(json.dumps(_task().to_json()) + "\n" + json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match=r"tasks\.jsonl:2: task .*snapshot"):
        read_tasks(path)


def test_duplicate_ids_are_rejected(tmp_path):
    task = _task()
    path = tmp_path / "tasks.jsonl"
    with pytest.raises(ValueError, match="duplicate task id"):
        write_tasks(path, [task, dataclasses.replace(task, prompt="Another.")])
    line = json.dumps(task.to_json()) + "\n"
    path.write_text(line + line)
    with pytest.raises(ValueError, match="duplicate task id"):
        read_tasks(path)

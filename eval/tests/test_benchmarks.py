# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tasks.benchmarks: turning LocBench rows into tasks."""

import json
from datetime import UTC, datetime

import pytest

from tasks.benchmarks import load_locbench
from tasks.schema import Bucket, PromptOrigin, Source
from tasks.split import assign_split

_BASE = "0123456789abcdef0123456789abcdef01234567"


def _patch(*paths: str) -> str:
    return "".join(
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
        "@@ -1 +1 @@\n-old\n+new\n"
        for path in paths
    )


def _row(instance_id: str, patch: str, **changes) -> dict:
    row = {
        "instance_id": instance_id,
        "repo": "psf/requests",
        "base_commit": _BASE,
        "problem_statement": "Session.get ignores the timeout.",
        "patch": patch,
        "edit_functions": [],
        "added_functions": [],
        "created_at": 1_740_000_000_000,
    }
    return row | changes


def _write(tmp_path, rows):
    path = tmp_path / "locbench.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_loads_a_row_as_a_bucket_s_task(tmp_path):
    row = _row(
        "psf__requests-1",
        _patch("requests/sessions.py", "tests/test_sessions.py", "HISTORY.md"),
        edit_functions=[
            "requests/sessions.py:Session.request",
            "requests/sessions.py:Session.get",
            "tests/test_sessions.py:test_timeout",
            "requests/models.py:Request.prepare",
        ],
        added_functions=["requests/sessions.py:Session.new_helper"],
    )
    [task] = load_locbench(_write(tmp_path, [row]))
    assert task.id == "locbench/psf__requests-1"
    assert task.bucket is Bucket.S
    assert task.repo == "psf/requests"
    assert task.snapshot == _BASE
    assert task.prompt == "Session.get ignores the timeout."
    # Tests and docs are not gold, and functions outside gold paths are dropped.
    assert task.gold_paths == ("requests/sessions.py",)
    assert task.gold_functions == (
        "requests/sessions.py::Session.get",
        "requests/sessions.py::Session.request",
    )
    assert task.provenance.source is Source.LOCBENCH
    assert task.provenance.source_id == "psf__requests-1"
    assert task.provenance.created_at == datetime.fromtimestamp(1_740_000_000, UTC)
    assert task.provenance.prompt_origin is PromptOrigin.ISSUE
    assert task.lexically_solvable is None
    assert task.split is assign_split(task.id)


def test_keeps_only_fixes_with_one_to_three_gold_files(tmp_path):
    rows = [
        _row("only-tests", _patch("tests/test_a.py")),
        _row("new-file-only", _patch("a.py").replace("--- a/a.py", "--- /dev/null")),
        _row("three", _patch("a.py", "b.py", "c.py")),
        _row("four", _patch("a.py", "b.py", "c.py", "d.py")),
    ]
    tasks = load_locbench(_write(tmp_path, rows))
    assert [task.id for task in tasks] == ["locbench/three"]
    assert tasks[0].gold_paths == ("a.py", "b.py", "c.py")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"repo": None}, "'repo' is not a string"),
        ({"created_at": "2025-01-01"}, "'created_at' is not an integer"),
        ({"created_at": True}, "'created_at' is not an integer"),
        ({"edit_functions": "a.py:f"}, "'edit_functions' is not a list"),
    ],
)
def test_rejects_mistyped_rows(tmp_path, changes, message):
    row = _row("bad", _patch("a.py"), **changes)
    with pytest.raises(ValueError, match=f"row bad: {message}"):
        load_locbench(_write(tmp_path, [row]))


def test_rejects_rows_that_are_not_objects(tmp_path):
    path = tmp_path / "locbench.jsonl"
    path.write_text("[1, 2]\n")
    with pytest.raises(ValueError, match=r"locbench\.jsonl:1: expected a JSON object"):
        load_locbench(path)


def test_verified_probe_keeps_only_eligible_gold_and_verbatim_issue(tmp_path):
    from tasks.benchmarks import load_swebench_verified

    rows = [
        {
            "instance_id": "psf__requests-2",
            "repo": "psf/requests",
            "base_commit": _BASE,
            "problem_statement": "The request fails; don't rewrite this.",
            "patch": _patch("requests/sessions.py", "tests/test_sessions.py"),
            "created_at": "2024-01-01T00:00:00Z",
        },
        {
            "instance_id": "only-tests",
            "repo": "psf/requests",
            "base_commit": _BASE,
            "problem_statement": "Broken test",
            "patch": _patch("tests/test_sessions.py"),
            "created_at": "2024-01-01T00:00:00Z",
        },
    ]
    [task] = load_swebench_verified(_write(tmp_path, rows))
    assert task.bucket is Bucket.P
    assert task.gold_paths == ("requests/sessions.py",)
    assert task.prompt == rows[0]["problem_statement"]
    assert task.provenance.source is Source.SWEBENCH_VERIFIED
    assert task.provenance.prompt_origin is PromptOrigin.ISSUE

# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for harness.episode: whole episodes with a scripted model."""

from pathlib import Path

import pytest

from harness.arms import ARMS
from harness.episode import EpisodeCaps, Task, run_episode
from harness.prompts import Prompts
from harness.sandbox import RepoSandbox
from models.scripted import ScriptedCall, ScriptedModel
from tests.test_ferret_env import FakeSearch

TASK = Task(task_id="toy/1", prompt="helper returns the wrong value", snapshot="s")


def _run(toy_repo: Path, arm_name: str, turns, **caps):
    arm = ARMS[arm_name]
    return run_episode(
        task=TASK,
        arm=arm,
        model=ScriptedModel(turns, tools=arm.tools),
        sandbox=RepoSandbox(toy_repo),
        search=FakeSearch() if arm.shape is not None else None,
        caps=EpisodeCaps(**caps),
        prompts=Prompts.load(),
    )


def test_runs_to_a_submission(toy_repo):
    record, messages = _run(
        toy_repo,
        "A3",
        [
            [ScriptedCall("search", {"pattern": "helper"})],
            [ScriptedCall("read", {"path": "src/util.py"})],
            [ScriptedCall("submit", {"files": ["src/util.py"]})],
        ],
    )
    assert record["exit_status"] == "Submitted"
    assert record["submission"] == {"files": ["src/util.py"], "functions": []}
    assert record["arm"] == "A3"
    assert record["shape"] == "context"
    assert record["search"] == {"kind": "fake"}
    assert record["totals"]["calls_by_tool"] == {"read": 1, "search": 1}
    assert record["forced_submit"] is None
    # The task and the top-level listing reach the model.
    first_user = next(m for m in messages if m.get("role") == "user")
    assert "helper returns the wrong value" in first_user["content"]
    assert "src/" in first_user["content"]


def test_forces_a_submit_when_the_budget_runs_out(toy_repo):
    record, _ = _run(
        toy_repo,
        "A1",
        [
            [ScriptedCall("search", {"pattern": "a"})],
            [ScriptedCall("submit", {"files": ["src/app.py"]})],
        ],
        max_tool_calls=1,
    )
    assert record["forced_submit"] == "tool_calls"
    assert record["exit_status"] == "Submitted"


def test_ends_without_a_submission_after_the_forced_turn(toy_repo):
    record, _ = _run(
        toy_repo,
        "A2",
        [
            [ScriptedCall("search", {"pattern": "a"})],
            [ScriptedCall("search", {"pattern": "b"})],
        ],
        max_tool_calls=1,
    )
    assert record["exit_status"] == "LimitsExceeded"
    assert record["submission"] is None
    assert record["totals"]["calls_by_status"] == {"ok": 1, "over_budget": 1}


def test_counts_format_errors(toy_repo):
    record, _ = _run(
        toy_repo,
        "A1",
        [
            [ScriptedCall("bash", {"command": "grep -r x"})],
            [ScriptedCall("submit", {"files": ["src/app.py"]})],
        ],
    )
    assert record["totals"]["format_errors"] == 1
    assert record["totals"]["invalid_tool_calls"] == 1
    assert record["exit_status"] == "Submitted"


def test_zero_tool_arm_offers_only_submit(toy_repo):
    record, _ = _run(
        toy_repo, "A0", [[ScriptedCall("submit", {"files": ["src/app.py"]})]]
    )
    assert record["tools"] == ["submit"]
    assert record["search"] is None
    assert record["exit_status"] == "Submitted"


def test_requires_a_backend_exactly_for_search_arms(toy_repo):
    with pytest.raises(ValueError):
        run_episode(
            task=TASK,
            arm=ARMS["A1"],
            model=ScriptedModel([], tools=ARMS["A1"].tools),
            sandbox=RepoSandbox(toy_repo),
            search=None,
            caps=EpisodeCaps(),
            prompts=Prompts.load(),
        )

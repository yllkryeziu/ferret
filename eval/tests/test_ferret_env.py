# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for harness.ferret_env: dispatch, statuses, budget, and submit."""

import json
from pathlib import Path

import pytest
from minisweagent.exceptions import Submitted

from harness import tools
from harness.ferret_env import FerretEnvironment
from harness.sandbox import RepoSandbox
from harness.search import SearchResult, SearchTimeoutError


class FakeSearch:
    """Answers every search with its pattern; "slow" times out."""

    def __init__(self):
        self.calls = []

    def search(self, arguments):
        self.calls.append(dict(arguments))
        if arguments["pattern"] == "slow":
            raise SearchTimeoutError("search timed out after 1 s")
        if arguments["pattern"] == "bad":
            return SearchResult("error: bad pattern", is_error=True)
        return SearchResult(f"results for {arguments['pattern']}", is_error=False)

    def describe(self):
        return {"kind": "fake"}


@pytest.fixture
def env(toy_repo: Path) -> FerretEnvironment:
    return FerretEnvironment(
        sandbox=RepoSandbox(toy_repo),
        tools=tools.ALL_TOOLS,
        search=FakeSearch(),
        max_tool_calls=3,
    )


def _call(env, tool, **arguments):
    observation = env.execute({"tool": tool, "arguments": arguments})
    return observation["output"], observation["extra"]["status"]


def test_dispatches_each_tool(env):
    assert _call(env, "search", pattern="helper") == ("results for helper", "ok")
    output, status = _call(env, "read", path="src/util.py")
    assert status == "ok"
    assert "def helper" in output
    assert _call(env, "ls", path="lib") == ("a.c\nb.c", "ok")


@pytest.mark.parametrize(
    ("tool", "arguments", "status"),
    [
        ("search", {"pattern": "slow"}, "timeout"),
        ("search", {"pattern": "bad"}, "error"),
        ("read", {"path": "missing.py"}, "error"),
        ("read", {"path": "../outside/secret.txt"}, "rejected"),
        ("search", {"pattern": 1}, "rejected"),
        ("bash", {"command": "ls"}, "rejected"),
    ],
)
def test_reports_statuses(env, tool, arguments, status):
    output, actual = _call(env, tool, **arguments)
    assert actual == status
    assert output.startswith("error:")


def test_every_call_spends_the_budget_until_only_submit_remains(env):
    _call(env, "search", pattern="a")
    _call(env, "bash")
    _call(env, "read", path="missing.py")
    assert env.calls_remaining == 0
    assert _call(env, "search", pattern="b") == (
        tools.OVER_BUDGET_ERROR,
        "over_budget",
    )
    with pytest.raises(Submitted):
        _call(env, "submit", files=["a.py"])


def test_submit_carries_the_canonical_answer(env):
    with pytest.raises(Submitted) as submitted:
        _call(env, "submit", files=["src/app.py"], functions=["src/app.py::main"])
    message = submitted.value.messages[0]
    assert message["extra"]["exit_status"] == "Submitted"
    assert json.loads(message["extra"]["submission"]) == {
        "files": ["src/app.py"],
        "functions": ["src/app.py::main"],
    }


def test_invalid_submit_is_rejected_and_spends_the_budget(env):
    assert _call(env, "submit", files=[])[1] == "rejected"
    assert env.calls_remaining == 2


def test_observations_carry_metrics(env):
    observation = env.execute({"tool": "search", "arguments": {"pattern": "x"}})
    assert observation["returncode"] == 0
    extra = observation["extra"]
    assert extra["tool"] == "search"
    assert extra["output_bytes"] == len("results for x")
    assert extra["tool_latency_s"] >= 0


def test_requires_search_exactly_when_offered(toy_repo):
    sandbox = RepoSandbox(toy_repo)
    with pytest.raises(ValueError):
        FerretEnvironment(
            sandbox=sandbox, tools=tools.ALL_TOOLS, search=None, max_tool_calls=1
        )
    with pytest.raises(ValueError):
        FerretEnvironment(
            sandbox=sandbox,
            tools=(tools.SUBMIT,),
            search=FakeSearch(),
            max_tool_calls=1,
        )

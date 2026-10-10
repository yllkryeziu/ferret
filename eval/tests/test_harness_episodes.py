# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for analysis.harness_episodes and analysis.pilot."""

import csv

import pytest

from analysis import pilot
from analysis.harness_episodes import (
    FORMAT_ERROR_CALL,
    episode_from_trajectory,
    read_run,
)
from analysis.protocol import Arm
from analysis.trajectory import Submission, ToolCall
from harness.run import ExperimentConfig, load_tasks, run_experiment
from models.scripted import ScriptedModelFactory

_RECORD = {
    "episode_id": "t__A2__s0",
    "task_id": "locbench/t",
    "arm": "A2",
    "shape": "line",
    "sample": 0,
    "submission": {"files": ["a.py"], "functions": ["a.py::f"]},
    "forced_submit": None,
}


def _responses_usage(input_tokens, cached_tokens, output_tokens):
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens": output_tokens,
    }


def test_converts_responses_api_messages():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"object": "response", "output": [], "usage": _responses_usage(100, 0, 10)},
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": "a.py:1:def f():",
            "extra": {"tool": "search", "status": "ok"},
        },
        {
            "type": "function_call_output",
            "call_id": "c2",
            "output": "error: unknown argument",
            "extra": {"tool": "read", "status": "rejected"},
        },
        {
            "role": "user",
            "content": "error: no tool call",
            "extra": {
                "interrupt_type": "FormatError",
                "response": {"usage": _responses_usage(150, 100, 5)},
            },
        },
        {"object": "response", "output": [], "usage": _responses_usage(200, 150, 20)},
        {"role": "exit", "content": "{}", "extra": {"exit_status": "Submitted"}},
    ]
    episode = episode_from_trajectory(_RECORD, messages)
    assert (episode.task_id, episode.arm, episode.sample) == ("locbench/t", Arm.A2, 0)
    assert [
        (turn.input_tokens, turn.cached_input_tokens, turn.output_tokens)
        for turn in episode.turns
    ] == [(100, 0, 10), (150, 100, 5), (200, 150, 20)]
    assert episode.turns[0].tool_calls == (
        ToolCall(name="search", valid=True, observation="a.py:1:def f():"),
        ToolCall(name="read", valid=False, observation="error: unknown argument"),
    )
    assert episode.turns[1].tool_calls == (
        ToolCall(
            name=FORMAT_ERROR_CALL, valid=False, observation="error: no tool call"
        ),
    )
    assert episode.turns[2].tool_calls == ()
    assert episode.submission == Submission(files=("a.py",), functions=("a.py::f",))
    assert not episode.cap_hit


def test_converts_chat_messages_and_caps():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "extra": {
                "response": {
                    "usage": {
                        "prompt_tokens": 10,
                        "prompt_tokens_details": {"cached_tokens": 4},
                        "completion_tokens": 1,
                    }
                }
            },
        },
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "src/"},
                {"type": "text", "text": "x"},
            ],
            "extra": {"tool": "ls", "status": "ok"},
        },
    ]
    record = _RECORD | {"submission": None, "forced_submit": "tool_calls"}
    episode = episode_from_trajectory(record, messages)
    assert (episode.turns[0].input_tokens, episode.turns[0].cached_input_tokens) == (
        10,
        4,
    )
    assert episode.turns[0].tool_calls[0].observation == "src/x"
    assert episode.submission is None
    assert episode.cap_hit
    a0 = record | {"arm": "A0", "shape": None}
    assert not episode_from_trajectory(a0, messages).cap_hit


def test_rejects_observations_before_any_model_call():
    message = {"role": "tool", "content": "x", "extra": {"tool": "ls"}}
    with pytest.raises(ValueError, match="observation before any model call"):
        episode_from_trajectory(_RECORD, [message])


def test_scores_a_scripted_run_end_to_end(run_inputs, tmp_path, capsys):
    data, ferret, tasks_path, script = run_inputs
    run = tmp_path / "run"
    report = run_experiment(
        ExperimentConfig(
            arms=("A0", "A2"), samples=1, seed=0, data_dir=data, ferret=ferret
        ),
        load_tasks(tasks_path),
        ScriptedModelFactory.from_file(script),
        run,
        workers=2,
    )
    assert report.failed == 0
    episodes = read_run(run)
    assert len(episodes) == 3  # The probe task runs A0 only.

    assert (
        pilot.main(
            [
                f"--tasks={tasks_path}",
                f"--run={run}",
                "--input-price=1",
                "--output-price=2",
                "--cached-input-price=0.1",
            ]
        )
        == 0
    )
    summary = capsys.readouterr().out
    assert "success" in summary
    assert "dollars" in summary
    with (run / "episodes.csv").open() as rows:
        by_arm = {(row["task_id"], row["arm"]): row for row in csv.DictReader(rows)}
    # The script submits the gold file in A2 and the wrong one in A0.
    assert by_arm[("locbench/toy-1", "A2")]["success"] == "True"
    assert by_arm[("locbench/toy-1", "A2")]["exposure_turn"] == "1.0"
    assert by_arm[("locbench/toy-1", "A0")]["success"] == "False"
    assert pilot.main([f"--tasks={tasks_path}", f"--run={run}", "--input-price=1"]) == 2

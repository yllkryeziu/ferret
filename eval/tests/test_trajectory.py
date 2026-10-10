# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for harness.trajectory: metrics, summaries, and the file format."""

import json

import pytest

from harness import trajectory


def test_normalizes_both_usage_flavors():
    chat = {
        "prompt_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 8},
        "completion_tokens": 5,
        "completion_tokens_details": {"reasoning_tokens": 2},
    }
    responses = {
        "input_tokens": 10,
        "input_tokens_details": {"cached_tokens": 8},
        "output_tokens": 5,
        "output_tokens_details": {"reasoning_tokens": 2},
    }
    expected = {
        "input_tokens": 10,
        "cached_input_tokens": 8,
        "output_tokens": 5,
        "reasoning_tokens": 2,
    }
    assert trajectory.normalize_usage(chat) == expected
    assert trajectory.normalize_usage(responses) == expected
    assert trajectory.normalize_usage(None) == {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
    }


def _turn(input_tokens, output_tokens, *, format_error=False):
    extra = {
        "llm_latency_s": 1.0,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": 0,
        },
        "served_model": "m-1",
    }
    if format_error:
        return {
            "role": "user",
            "content": "error",
            "extra": {**extra, "interrupt_type": "FormatError"},
        }
    return {"role": "assistant", "content": "", "extra": extra}


def _observation(tool, status):
    return {
        "role": "tool",
        "content": "out",
        "extra": {
            "tool": tool,
            "status": status,
            "tool_latency_s": 0.5,
            "output_bytes": 3,
        },
    }


def test_summarizes_an_episode():
    submission = json.dumps({"files": ["a.py"], "functions": []})
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
        _turn(100, 10),
        _observation("search", "ok"),
        _observation("read", "error"),
        _turn(200, 20, format_error=True),
        _turn(300, 30),
        _observation("search", "rejected"),
        {
            "role": "user",
            "content": "submit now",
            "extra": {"forced_submit": "tool_calls"},
        },
        _turn(400, 40),
        {
            "role": "exit",
            "content": submission,
            "extra": {"exit_status": "Submitted", "submission": submission},
        },
    ]
    summary = trajectory.summarize(messages)
    assert summary["exit_status"] == "Submitted"
    assert summary["submission"] == {"files": ["a.py"], "functions": []}
    assert summary["forced_submit"] == "tool_calls"
    assert summary["served_models"] == ["m-1"]
    totals = summary["totals"]
    assert totals["model_calls"] == 4
    assert totals["format_errors"] == 1
    assert totals["tool_calls"] == 3
    assert totals["invalid_tool_calls"] == 2
    assert totals["calls_by_tool"] == {"read": 1, "search": 2}
    assert totals["calls_by_status"] == {"error": 1, "ok": 1, "rejected": 1}
    assert totals["input_tokens"] == 1000
    assert totals["output_tokens"] == 100
    assert totals["tool_latency_s"] == pytest.approx(1.5)
    assert [len(step["tool_calls"]) for step in summary["steps"]] == [2, 0, 1, 0]


def test_writes_and_reads_episodes(tmp_path):
    path = trajectory.episode_path(tmp_path, "t1-A1")
    record = {"format_version": trajectory.FORMAT_VERSION, "task_id": "t1"}
    messages = [{"role": "user", "content": "hi"}]
    trajectory.write_episode(path, record, messages)
    assert path == tmp_path / "episodes" / "t1-A1.jsonl"
    assert trajectory.read_episode(path) == ({"record": "episode", **record}, messages)
    assert not list(path.parent.glob(".*.tmp"))


def test_read_rejects_other_versions(tmp_path):
    path = tmp_path / "e.jsonl"
    trajectory.write_episode(path, {"format_version": 999}, [])
    with pytest.raises(ValueError, match="format version 999"):
        trajectory.read_episode(path)

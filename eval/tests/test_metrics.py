# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for analysis.metrics: per-episode success, rank, and exposure."""

import dataclasses
import math
from datetime import UTC, datetime

import pytest

from analysis.metrics import (
    file_reciprocal_rank,
    file_success,
    first_exposure,
    function_success,
    ranked_files,
    score_episode,
)
from analysis.protocol import TOKEN_HORIZON, TURN_HORIZON, Arm, TokenPrices
from analysis.trajectory import Episode, Submission, ToolCall, Turn
from tasks.schema import Bucket, PromptOrigin, Provenance, Source, Split, Task

_TASK = Task(
    id="locbench/t",
    bucket=Bucket.S,
    repo="o/r",
    snapshot="0" * 40,
    prompt="Broken.",
    gold_paths=("pkg/core.py", "pkg/util.py"),
    gold_functions=("pkg/core.py::Engine.run",),
    provenance=Provenance(
        source=Source.LOCBENCH,
        source_id="t",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        prompt_origin=PromptOrigin.ISSUE,
    ),
    lexically_solvable=None,
    split=Split.DEV,
)


def _turn(tokens: int, *calls: tuple[str, bool, str]) -> Turn:
    return Turn(
        input_tokens=tokens,
        output_tokens=tokens // 10,
        tool_calls=tuple(
            ToolCall(name=name, valid=valid, observation=observation)
            for name, valid, observation in calls
        ),
    )


def test_ranked_files_normalizes_and_deduplicates():
    assert ranked_files(["./a//b.py", "a/b.py", " c.py ", ".", "d.py"]) == [
        "a/b.py",
        "c.py",
        "d.py",
    ]


def test_file_success_needs_every_gold_file_in_the_top_five():
    gold = ["pkg/core.py", "pkg/util.py"]
    assert file_success(["pkg/util.py", "x.py", "./pkg/core.py"], gold)
    assert not file_success(["pkg/core.py"], gold)
    # Duplicates do not use up ranks.
    assert file_success(["a", "a", "b", "b", "c", "pkg/core.py", "pkg/util.py"], gold)
    assert not file_success(["a", "b", "c", "d", "pkg/core.py", "pkg/util.py"], gold)


def test_file_reciprocal_rank():
    assert file_reciprocal_rank(["x", "x", "pkg/util.py"], ["pkg/util.py"]) == 0.5
    assert file_reciprocal_rank(["x"], ["pkg/util.py"]) == 0.0


def test_function_success_accepts_the_last_name_segment():
    gold = ["pkg/core.py::Engine.run"]
    assert function_success(["pkg/core.py::Engine.run"], gold)
    assert function_success(["./pkg/core.py::run"], gold)
    assert not function_success(["pkg/core.py::Engine"], gold)
    assert not function_success(["pkg/util.py::run", "malformed"], gold)


def test_first_exposure_matches_whole_paths_only():
    episode = Episode(
        task_id="t",
        arm=Arm.A1,
        sample=0,
        turns=(
            _turn(100, ("search", True, "xpkg/core.py:1: pkg/core.pyc pkg/core.py_x")),
            _turn(200, ("read", True, "== ./pkg/core.py lines 1-20 ==")),
        ),
        submission=None,
        cap_hit=False,
    )
    assert first_exposure(episode, ["pkg/core.py"]) == (2, 110 + 220)
    assert first_exposure(episode, ["pkg/other.py"]) is None


def test_scores_an_episode():
    episode = Episode(
        task_id="locbench/t",
        arm=Arm.A3,
        sample=1,
        turns=(
            _turn(1000, ("search", True, "pkg/util.py:3:def f"), ("ls", False, "")),
            _turn(2000, ("", False, "error: no tool call")),
            _turn(3000),
        ),
        submission=Submission(
            files=("pkg/util.py", "pkg/core.py"), functions=("pkg/core.py::run",)
        ),
        cap_hit=True,
    )
    metrics = score_episode(episode, _TASK)
    assert metrics.success
    assert metrics.file_reciprocal_rank == 1.0
    assert metrics.function_success == 1.0
    assert (metrics.input_tokens, metrics.output_tokens) == (6000, 600)
    assert (metrics.turns, metrics.tool_calls, metrics.invalid_tool_calls) == (3, 3, 2)
    assert (metrics.search_calls, metrics.read_calls) == (1, 0)
    assert metrics.cap_hit
    assert (metrics.exposure_turn, metrics.exposure_tokens) == (1, 1100)
    assert metrics.restricted_exposure_turns == 1.0


def test_an_episode_without_submission_or_exposure_fails_at_the_horizon():
    episode = Episode(
        task_id="locbench/t",
        arm=Arm.A0,
        sample=0,
        turns=(_turn(500),),
        submission=None,
        cap_hit=False,
    )
    metrics = score_episode(episode, _TASK)
    assert not metrics.submitted
    assert not metrics.success
    assert math.isnan(metrics.exposure_turn)
    assert metrics.restricted_exposure_turns == TURN_HORIZON
    assert metrics.restricted_exposure_tokens == TOKEN_HORIZON


@pytest.mark.parametrize("functions", [(), ("pkg/core.py::Engine.run",)])
def test_function_success_is_undefined_without_function_gold(functions):
    task = dataclasses.replace(_TASK, gold_functions=())
    episode = Episode(
        task_id=task.id,
        arm=Arm.A1,
        sample=0,
        turns=(),
        submission=Submission(files=("pkg/core.py",), functions=functions),
        cap_hit=False,
    )
    assert math.isnan(score_episode(episode, task).function_success)


def test_token_prices_discount_cached_input():
    prices = TokenPrices(2.0, 8.0, cached_input_per_million=0.5)
    assert prices.cost(1_000_000, 100_000, 600_000) == pytest.approx(
        0.4 * 2.0 + 0.6 * 0.5 + 0.1 * 8.0
    )
    assert TokenPrices(2.0, 8.0).cost(1_000_000, 0, 600_000) == pytest.approx(2.0)


def test_turns_reject_impossible_cache_counts():
    with pytest.raises(ValueError, match="more cached than input"):
        Turn(input_tokens=5, cached_input_tokens=6, output_tokens=0, tool_calls=())
    with pytest.raises(ValueError, match="negative"):
        Turn(input_tokens=5, cached_input_tokens=-1, output_tokens=0, tool_calls=())

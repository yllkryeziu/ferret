# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Reduces harness trajectories to the episodes the analysis consumes.

This module is the only bridge between the two formats: the harness records
everything a run did (`harness.trajectory`), and the analysis reads only the
episode contract of `analysis.trajectory`. Each model call becomes a turn
with its tokens as the API's own usage report gives them, which every
trajectory keeps beside the harness's summary of it. Each tool observation
becomes a call on the turn that issued it, invalid when the environment
rejected it before running it. A response the model's parser could not turn
into tool calls becomes one invalid call without a tool name, as it is the
same failure one step earlier. An episode hit a cap when a tool arm's
episode was forced to submit.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from analysis.protocol import Arm
from analysis.trajectory import Episode, Submission, ToolCall, Turn
from harness.trajectory import (
    is_model_turn,
    is_observation,
    normalize_usage,
    read_episode,
)

# The name of the invalid call that stands for an unparseable response.
FORMAT_ERROR_CALL = ""


def episode_from_trajectory(
    record: Mapping[str, Any], messages: Sequence[Mapping[str, Any]]
) -> Episode:
    """Returns the analysis episode of one harness trajectory.

    Raises:
        ValueError: if an observation precedes every model call.
    """
    turns: list[tuple[Mapping[str, Any], list[ToolCall]]] = []
    for message in messages:
        extra = message.get("extra") or {}
        if is_model_turn(message):
            calls = []
            if extra.get("interrupt_type") == "FormatError":
                calls.append(
                    ToolCall(
                        name=FORMAT_ERROR_CALL,
                        valid=False,
                        observation=_text(message.get("content")),
                    )
                )
            turns.append((normalize_usage(_api_usage(message)), calls))
        elif is_observation(message):
            if not turns:
                raise ValueError(
                    f"episode {record.get('episode_id')}: observation before any "
                    "model call"
                )
            turns[-1][1].append(
                ToolCall(
                    name=str(extra.get("tool")),
                    valid=extra.get("status") != "rejected",
                    observation=_text(
                        message.get("content", message.get("output", ""))
                    ),
                )
            )
    submission = record.get("submission")
    return Episode(
        task_id=record["task_id"],
        arm=Arm(record["arm"]),
        sample=record["sample"],
        turns=tuple(
            Turn(
                input_tokens=usage["input_tokens"],
                cached_input_tokens=usage["cached_input_tokens"],
                output_tokens=usage["output_tokens"],
                tool_calls=tuple(calls),
            )
            for usage, calls in turns
        ),
        submission=None
        if submission is None
        else Submission(
            files=tuple(submission["files"]), functions=tuple(submission["functions"])
        ),
        # Arm A0 starts at its tool budget of zero, so its forced-submit turn
        # is the protocol rather than a cap.
        cap_hit=record.get("forced_submit") is not None
        and record.get("shape") is not None,
    )


def read_run(run_dir: Path) -> list[Episode]:
    """Returns the episodes of every trajectory in a run directory."""
    return [
        episode_from_trajectory(*read_episode(path))
        for path in sorted((run_dir / "episodes").glob("*.jsonl"))
    ]


def _api_usage(message: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Returns the usage report of the API response behind a model turn.

    A Responses API turn is the response itself; a Chat Completions turn and
    a FormatError keep the response under `extra.response`. Offline models
    call no API, and their turns have no report.
    """
    if message.get("object") == "response":
        return message.get("usage")
    response = (message.get("extra") or {}).get("response")
    return response.get("usage") if isinstance(response, Mapping) else None


def _text(content: Any) -> str:
    """Returns message content as text; list content is joined text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, Iterable):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, Mapping)
        )
    return ""

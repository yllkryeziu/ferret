# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""The trajectory format: one JSON Lines file per episode.

Line 1 is the episode record: identity, provenance, outcome, and per-turn
metrics. Every later line is one message of
mini-swe-agent's message list, verbatim, so an episode can be re-analyzed or
replayed (`models.replay`) without running it again.

Per-turn metrics travel in the `extra` of each model message. Harness models
write them with `record_turn_metrics`; `summarize` reads them back together
with the `extra` that `FerretEnvironment` attaches to each observation. LLM
latency is the duration of the API call that produced the turn, excluding
retries; tool latency is measured inside the environment. Keeping the two
apart is what lets the analysis replace live tool time with replayed tool
time.
"""

import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1

_ZERO_USAGE = {
    "input_tokens": 0,
    "cached_input_tokens": 0,
    "output_tokens": 0,
    "reasoning_tokens": 0,
}


def record_turn_metrics(
    message: dict[str, Any], response: Mapping[str, Any] | None, latency_s: float
) -> None:
    """Adds the per-turn metrics to `message["extra"]`.

    Args:
        message: A model message or FormatError message with an `extra` dict.
        response: The JSON dump of the API response, Chat Completions or
            Responses API, or None for a model that calls no API.
        latency_s: Duration of the API call.
    """
    response = response or {}
    message["extra"].update(
        {
            "llm_latency_s": latency_s,
            "usage": normalize_usage(response.get("usage")),
            "system_fingerprint": response.get("system_fingerprint"),
            "served_model": response.get("model"),
        }
    )


def normalize_usage(usage: Mapping[str, Any] | None) -> dict[str, int]:
    """Maps Chat Completions or Responses API usage to input, cached input,
    output, and reasoning token counts. Cached input tokens, served from the
    provider's prompt cache and billed at a discount, are part of input
    tokens; reasoning tokens are part of output tokens."""
    if not usage:
        return dict(_ZERO_USAGE)
    input_details = (
        usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    )
    details = (
        usage.get("output_tokens_details")
        or usage.get("completion_tokens_details")
        or {}
    )
    return {
        "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens") or 0,
        "cached_input_tokens": input_details.get("cached_tokens") or 0,
        "output_tokens": usage.get("output_tokens")
        or usage.get("completion_tokens")
        or 0,
        "reasoning_tokens": details.get("reasoning_tokens") or 0,
    }


def is_model_turn(message: Mapping[str, Any]) -> bool:
    """Whether `message` came from a model call: a Chat Completions
    assistant message, a Responses API response, or a FormatError that
    replaced an unparseable response."""
    return (
        message.get("role") == "assistant"
        or message.get("object") == "response"
        or _extra(message).get("interrupt_type") == "FormatError"
    )


def is_observation(message: Mapping[str, Any]) -> bool:
    """Whether `message` carries the output of one tool call."""
    return (
        message.get("role") == "tool" or message.get("type") == "function_call_output"
    )


def summarize(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Computes an episode's outcome and metrics from its messages.

    Returns:
        A JSON object with `exit_status`; `submission`, the submitted
        {"files", "functions"} or None; `forced_submit`, the reason for the
        forced-submit turn or None; the distinct `system_fingerprints` and
        `served_models`; `steps`, one entry per model call with its tool
        calls; and `totals`. Tool calls issued in the same turn as a valid
        submit have no observation and are not counted.
    """
    steps: list[dict[str, Any]] = []
    forced_submit = None
    exit_extra: Mapping[str, Any] = {}
    for message in messages:
        extra = _extra(message)
        if is_model_turn(message):
            usage = extra.get("usage") or _ZERO_USAGE
            steps.append(
                {
                    "llm_latency_s": extra.get("llm_latency_s", 0.0),
                    **{key: usage.get(key, 0) for key in _ZERO_USAGE},
                    "system_fingerprint": extra.get("system_fingerprint"),
                    "served_model": extra.get("served_model"),
                    "format_error": extra.get("interrupt_type") == "FormatError",
                    "tool_calls": [],
                }
            )
        elif is_observation(message) and steps:
            steps[-1]["tool_calls"].append(
                {
                    "tool": extra.get("tool"),
                    "status": extra.get("status"),
                    "latency_s": extra.get("tool_latency_s", 0.0),
                    "output_bytes": extra.get("output_bytes", 0),
                }
            )
        elif "forced_submit" in extra:
            forced_submit = extra["forced_submit"]
        elif message.get("role") == "exit":
            exit_extra = extra
    calls = [call for step in steps for call in step["tool_calls"]]
    statuses = Counter(call["status"] for call in calls)
    format_errors = sum(step["format_error"] for step in steps)
    submission = exit_extra.get("submission") or None
    return {
        "exit_status": exit_extra.get("exit_status"),
        "submission": json.loads(submission) if submission else None,
        "forced_submit": forced_submit,
        "system_fingerprints": _distinct(step["system_fingerprint"] for step in steps),
        "served_models": _distinct(step["served_model"] for step in steps),
        "totals": {
            "model_calls": len(steps),
            "format_errors": format_errors,
            "tool_calls": len(calls),
            "invalid_tool_calls": format_errors + statuses["rejected"],
            "calls_by_tool": dict(
                sorted(Counter(call["tool"] for call in calls).items())
            ),
            "calls_by_status": dict(sorted(statuses.items())),
            "llm_latency_s": sum(step["llm_latency_s"] for step in steps),
            "tool_latency_s": sum(call["latency_s"] for call in calls),
            **{key: sum(step[key] for step in steps) for key in _ZERO_USAGE},
        },
        "steps": steps,
    }


def episode_path(out_dir: Path, episode_id: str) -> Path:
    """Returns where the trajectory of `episode_id` lives under `out_dir`."""
    return out_dir / "episodes" / f"{episode_id}.jsonl"


def write_episode(
    path: Path, record: Mapping[str, Any], messages: Sequence[Mapping[str, Any]]
) -> None:
    """Writes a trajectory atomically: a reader, or a resumed run, sees
    either the complete file or no file.

    Raises:
        OSError: on a filesystem error; TypeError or ValueError if the
            record or a message is not JSON-serializable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as out:
        out.write(json.dumps({"record": "episode", **record}, allow_nan=False) + "\n")
        for message in messages:
            out.write(json.dumps(message, allow_nan=False) + "\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)


def read_episode(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reads a trajectory written by `write_episode`.

    Returns:
        The episode record and the messages.

    Raises:
        ValueError: if the file is not a trajectory of FORMAT_VERSION.
    """
    with path.open(encoding="utf-8") as lines:
        record = json.loads(next(lines, "null"))
        if not isinstance(record, dict) or record.get("record") != "episode":
            raise ValueError(f"{path}: first line is not an episode record")
        if record.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"{path}: format version {record.get('format_version')!r}, expected "
                f"{FORMAT_VERSION}"
            )
        return record, [json.loads(line) for line in lines]


def _extra(message: Mapping[str, Any]) -> Mapping[str, Any]:
    extra = message.get("extra")
    return extra if isinstance(extra, Mapping) else {}


def _distinct(values: Any) -> list[Any]:
    return sorted({value for value in values if value is not None})

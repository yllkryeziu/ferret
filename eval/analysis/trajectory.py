# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""The episode record the analysis consumes, and its JSONL encoding.

The harness writes one episode per (task, arm, sample): every model turn with
its billed tokens and the tool calls it made, and the final submission. This
is the contract between harness and analysis; the analysis never reads
harness-specific trajectory formats. One JSON object per line:

    {"task_id": "linux/3f2a9c1d0b7e", "arm": "A3", "sample": 0,
     "cap_hit": false,
     "turns": [{"input_tokens": 5120, "cached_input_tokens": 4096,
                "output_tokens": 310,
                "tool_calls": [{"name": "search", "valid": true,
                                "observation": "drivers/acpi/battery.c:..."}]}],
     "submission": {"files": ["drivers/acpi/battery.c"],
                    "functions": ["drivers/acpi/battery.c::acpi_battery_update"]}}

`submission` is null when the agent never submitted. A tool call is invalid
when the environment rejected it before running it (unknown tool, arguments
that are not JSON or violate the schema); a valid call that returns an error,
such as a pattern RE2 rejects, is valid. Observations are the exact text the
model saw, and a `read` observation names the file it shows.
"""

import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from analysis.protocol import Arm


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ToolCall:
    """One tool call and what the environment returned."""

    name: str
    valid: bool
    observation: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Turn:
    """One model call, with the tokens the API billed for it.

    `cached_input_tokens` is the part of `input_tokens` served from the
    provider's prompt cache, which is billed at a discount.
    """

    input_tokens: int
    cached_input_tokens: int = 0
    output_tokens: int
    tool_calls: tuple[ToolCall, ...]

    def __post_init__(self) -> None:
        if min(self.input_tokens, self.cached_input_tokens, self.output_tokens) < 0:
            raise ValueError(f"negative token count in {self}")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError(f"more cached than input tokens in {self}")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Submission:
    """The agent's ranked answer."""

    files: tuple[str, ...]
    functions: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Episode:
    """One agent run on one task in one arm."""

    task_id: str
    arm: Arm
    sample: int
    turns: tuple[Turn, ...]
    submission: Submission | None
    cap_hit: bool

    def __post_init__(self) -> None:
        if self.sample < 0:
            raise ValueError(f"episode {self.key}: negative sample")

    @property
    def key(self) -> tuple[str, Arm, int]:
        """(task_id, arm, sample), unique within an experiment."""
        return (self.task_id, self.arm, self.sample)

    def to_json(self) -> dict[str, Any]:
        """Returns the JSON object for this episode; `from_json` inverts it."""
        return {
            "task_id": self.task_id,
            "arm": self.arm.value,
            "sample": self.sample,
            "cap_hit": self.cap_hit,
            "turns": [dataclasses.asdict(turn) for turn in self.turns],
            "submission": None
            if self.submission is None
            else {
                "files": list(self.submission.files),
                "functions": list(self.submission.functions),
            },
        }

    @classmethod
    def from_json(cls, obj: Any) -> "Episode":
        """Parses an episode from its JSON object.

        Raises:
            ValueError: on a missing or mistyped field.
        """
        try:
            submission = obj["submission"]
            return cls(
                task_id=_typed(obj["task_id"], str),
                arm=Arm(obj["arm"]),
                sample=_typed(obj["sample"], int),
                cap_hit=_typed(obj["cap_hit"], bool),
                turns=tuple(
                    Turn(
                        input_tokens=_typed(turn["input_tokens"], int),
                        cached_input_tokens=_typed(
                            turn.get("cached_input_tokens", 0), int
                        ),
                        output_tokens=_typed(turn["output_tokens"], int),
                        tool_calls=tuple(
                            ToolCall(
                                name=_typed(call["name"], str),
                                valid=_typed(call["valid"], bool),
                                observation=_typed(call["observation"], str),
                            )
                            for call in turn["tool_calls"]
                        ),
                    )
                    for turn in obj["turns"]
                ),
                submission=None
                if submission is None
                else Submission(
                    files=tuple(_typed(path, str) for path in submission["files"]),
                    functions=tuple(_typed(f, str) for f in submission["functions"]),
                ),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"malformed episode: {error!r}") from error


def write_episodes(path: Path, episodes: Iterable[Episode]) -> None:
    """Writes `episodes` to `path` as JSON Lines."""
    with path.open("w", encoding="utf-8") as out:
        for episode in episodes:
            out.write(json.dumps(episode.to_json(), ensure_ascii=False) + "\n")


def read_episodes(path: Path) -> list[Episode]:
    """Reads a JSON Lines episode file.

    Raises:
        ValueError: naming the line of the first malformed episode, or if two
            episodes share a (task, arm, sample) key.
    """
    episodes = []
    seen = set()
    with path.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            try:
                episode = Episode.from_json(json.loads(line))
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            if episode.key in seen:
                raise ValueError(f"{path}:{number}: duplicate episode {episode.key}")
            seen.add(episode.key)
            episodes.append(episode)
    return episodes


def _typed[T](value: Any, kind: type[T]) -> T:
    # bool is an int subclass; a JSON true must not pass as a count.
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise TypeError(f"expected {kind.__name__}, got {value!r}")
    return value

# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""A model that makes a fixed sequence of tool calls.

Used by the tests and for offline smoke runs of the whole pipeline. Each
turn is encoded as Chat Completions tool calls and parsed by the same code as
a live model's, so an unknown tool or malformed arguments in a script
produce the same FormatError a live model would get.

A script file is a JSON object mapping an arm name, or "default", to a list
of turns; a turn is a list of calls {"tool": name, "arguments": object}.
"""

import dataclasses
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

from minisweagent.exceptions import FormatError

from harness.arms import Arm
from harness.tools import ToolSpec
from harness.trajectory import record_turn_metrics
from models.offline import OfflineModel
from models.toolcalls import ApiFlavor, parse_chat_tool_calls


@dataclasses.dataclass(frozen=True, slots=True)
class ScriptedCall:
    """One tool call of a script.

    Attributes:
        tool: The tool name, which need not exist in the arm.
        arguments: The arguments; a string is sent verbatim as the JSON
            argument text, which lets tests send malformed JSON.
    """

    tool: str
    arguments: Mapping[str, Any] | str = dataclasses.field(default_factory=dict)


class ScriptedModel(OfflineModel):
    """Emits one scripted turn per query. Not thread-safe."""

    def __init__(
        self, turns: Sequence[Sequence[ScriptedCall]], *, tools: Sequence[ToolSpec]
    ) -> None:
        """Creates the model.

        Args:
            turns: The turns, in order. An empty turn is a response without
                tool calls.
            tools: The arm's tools.
        """
        super().__init__(api_flavor=ApiFlavor.CHAT, config={"model_name": "scripted"})
        self._turns = [tuple(turn) for turn in turns]
        self._tool_names = tuple(tool.name for tool in tools)
        self._next_turn = 0

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Returns the next scripted turn as an assistant message.

        Raises:
            FormatError: if the turn has no call, or an unknown or malformed
                one.
            RuntimeError: if the script is exhausted.
        """
        del messages, kwargs
        if self._next_turn >= len(self._turns):
            raise RuntimeError(f"script has no turn {self._next_turn}")
        index = self._next_turn
        self._next_turn += 1
        tool_calls = [
            {
                "id": f"call_{index}_{position}",
                "type": "function",
                "function": {
                    "name": call.tool,
                    "arguments": (
                        call.arguments
                        if isinstance(call.arguments, str)
                        else json.dumps(call.arguments)
                    ),
                },
            }
            for position, call in enumerate(self._turns[index])
        ]
        message: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
            "extra": {"timestamp": time.time()},
        }
        try:
            message["extra"]["actions"] = parse_chat_tool_calls(
                tool_calls, self._tool_names
            )
        except FormatError as error:
            record_turn_metrics(error.messages[0], None, 0.0)
            raise
        record_turn_metrics(message, None, 0.0)
        return message


@dataclasses.dataclass(frozen=True, slots=True)
class ScriptedModelFactory:
    """Builds a ScriptedModel for each episode from per-arm scripts.

    Attributes:
        scripts: Turns by arm name; "default" applies to other arms.
        source_sha256: SHA-256 of the script file, for provenance.
    """

    scripts: Mapping[str, Sequence[Sequence[ScriptedCall]]]
    source_sha256: str

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Loads a script file (see the module docstring).

        Raises:
            ValueError: if the file is not a well-formed script.
        """
        data = path.read_bytes()
        scripts = json.loads(data)
        if not isinstance(scripts, dict) or "default" not in scripts:
            raise ValueError(f"{path}: expected an object with a 'default' script")
        try:
            parsed = {
                arm: [
                    [
                        ScriptedCall(call["tool"], call.get("arguments", {}))
                        for call in turn
                    ]
                    for turn in turns
                ]
                for arm, turns in scripts.items()
            }
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"{path}: malformed script: {error}") from error
        return cls(scripts=parsed, source_sha256=hashlib.sha256(data).hexdigest())

    def build(
        self, arm: Arm, episode_id: str, *, tools: Sequence[ToolSpec] | None = None
    ) -> ScriptedModel:
        """Returns a fresh model running the script for `arm`."""
        del episode_id
        turns = self.scripts.get(arm.name, self.scripts["default"])
        return ScriptedModel(turns, tools=tools if tools is not None else arm.tools)

    def provenance(self) -> dict[str, Any]:
        """Describes the model for the trajectory."""
        return {"kind": "scripted", "script_sha256": self.source_sha256}

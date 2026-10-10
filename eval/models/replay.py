# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Replays recorded model turns, retaining their usage and offered-tool hash."""

import copy
import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from minisweagent.exceptions import FormatError

from harness.arms import Arm
from harness.tools import ToolSpec, tool_set_sha256
from harness.trajectory import (
    episode_path,
    is_model_turn,
    read_episode,
    record_turn_metrics,
)
from models.offline import OfflineModel
from models.toolcalls import ApiFlavor, parse_chat_tool_calls


class ReplayModel(OfflineModel):
    """Consumes recorded assistant messages without contacting a provider."""

    def __init__(
        self,
        turns: Sequence[dict[str, Any]],
        tools: Sequence[ToolSpec],
        *,
        api_flavor: ApiFlavor = ApiFlavor.CHAT,
    ) -> None:
        super().__init__(api_flavor=api_flavor, config={"model_name": "replay"})
        self._turns = iter(turns)
        self._tool_names = [tool.name for tool in tools]

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Returns a deep copy of the next turn, or fails on an exhausted replay."""
        del messages, kwargs
        try:
            message = copy.deepcopy(next(self._turns))
        except StopIteration:
            raise RuntimeError("replay has no more recorded turns") from None
        extra = message.setdefault("extra", {})
        if extra.get("interrupt_type") == "FormatError":
            raise FormatError(message)
        if "actions" not in extra and self.api_flavor is ApiFlavor.CHAT:
            extra["actions"] = parse_chat_tool_calls(
                message.get("tool_calls"), self._tool_names
            )
        if "usage" not in extra:
            record_turn_metrics(message, None, 0.0)
        return message


@dataclasses.dataclass(frozen=True, slots=True)
class ReplayModelFactory:
    """Replays one recorded episode per run episode with a pinned tool schema."""

    source_dir: Path

    def build(
        self, arm: Arm, episode_id: str, *, tools: Sequence[ToolSpec] | None = None
    ) -> ReplayModel:
        """Loads recorded turns only when the recording offered identical tools."""
        offered = arm.tools if tools is None else tools
        path = episode_path(self.source_dir, episode_id)
        record, messages = read_episode(path)
        if record["tool_schema_sha256"] != tool_set_sha256(offered):
            raise ValueError(f"{path}: recorded with a different tool schema")
        return ReplayModel(
            [message for message in messages if is_model_turn(message)],
            offered,
            api_flavor=ApiFlavor(record["api_flavor"]),
        )

    def provenance(self) -> dict[str, Any]:
        """Identifies the recording, not a live model provider."""
        return {"kind": "replay", "source_dir": str(self.source_dir)}

# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Injectable, in-process model backend for development without Azure."""

import dataclasses
import time
from collections.abc import Callable, Sequence
from typing import Any

from minisweagent.exceptions import FormatError

from harness.arms import Arm
from harness.tools import ToolSpec
from harness.trajectory import record_turn_metrics
from models.litellm_models import ToolCallingChatModel
from models.offline import OfflineModel
from models.toolcalls import OBSERVATION_TEMPLATE, ApiFlavor, parse_chat_tool_calls


class LocalModel(OfflineModel):
    """Calls a supplied local inference function; never accesses a network."""

    def __init__(
        self,
        inference: Callable[[list[dict[str, Any]]], dict[str, Any]],
        tools: Sequence[ToolSpec],
    ) -> None:
        super().__init__(api_flavor=ApiFlavor.CHAT, config={"model_name": "local"})
        self._inference = inference
        self._tool_names = [tool.name for tool in tools]

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Runs inference and applies the same tool parser as scripted calls."""
        del kwargs
        start = time.monotonic()
        message = self._inference(messages)
        message.setdefault("extra", {})
        try:
            message["extra"]["actions"] = parse_chat_tool_calls(
                message.get("tool_calls"), self._tool_names
            )
        except FormatError as error:
            record_turn_metrics(error.messages[0], None, time.monotonic() - start)
            raise
        record_turn_metrics(message, None, time.monotonic() - start)
        return message


DEFAULT_BASE_URL = "http://localhost:11434/v1"

# The OpenAI client refuses to send a request without a key; local servers
# ignore it. Not a secret.
_PLACEHOLDER_API_KEY = "local"


@dataclasses.dataclass(frozen=True, slots=True)
class LocalModelFactory:
    """Builds a model served by a local OpenAI-compatible endpoint.

    Attributes:
        model: The model name the endpoint serves, such as "gpt-oss:20b".
        base_url: The endpoint's OpenAI-compatible base URL.
    """

    model: str
    base_url: str = DEFAULT_BASE_URL

    def build(
        self, arm: Arm, episode_id: str, *, tools: Sequence[ToolSpec] | None = None
    ) -> ToolCallingChatModel:
        """Returns a fresh model offering `arm`'s tools."""
        del episode_id
        return ToolCallingChatModel(
            tools=tools if tools is not None else arm.tools,
            model_name=f"openai/{self.model}",
            model_kwargs={
                "api_base": self.base_url,
                "api_key": _PLACEHOLDER_API_KEY,
                **self._sampling(),
            },
            cost_tracking="ignore_errors",
            observation_template=OBSERVATION_TEMPLATE,
        )

    def provenance(self) -> dict[str, Any]:
        """Describes the model for the trajectory."""
        return {
            "kind": "local",
            "model_id": f"openai/{self.model}",
            "api": ApiFlavor.CHAT.value,
            "base_url": self.base_url,
            "sampling": self._sampling(),
        }

    @staticmethod
    def _sampling() -> dict[str, float]:
        return {"temperature": 1.0, "top_p": 1.0}

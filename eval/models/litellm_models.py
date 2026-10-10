# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""mini-swe-agent's litellm models, declaring the harness tools.

In mini-swe-agent 2.4.6, `LitellmModel` and `LitellmResponseModel` inline
their single `bash` tool in `_query` and check for it by name in
`_parse_actions`. These subclasses override exactly those two methods to
offer the arm's tools instead, and wrap `query` only to attach the per-turn
metrics. Retries, cost tracking, message preparation (including the
round-trip of Responses API reasoning items), and observation formatting are
inherited unchanged.
"""

import time
from collections.abc import Sequence
from typing import Any

import litellm
from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.litellm_response_model import LitellmResponseModel

from harness.tools import ToolSpec
from harness.trajectory import record_turn_metrics
from models.toolcalls import (
    ApiFlavor,
    chat_tools,
    parse_chat_tool_calls,
    parse_responses_output,
    responses_tools,
)

# litellm prints a provider-list banner whenever cost tracking cannot place a
# served model name such as "gpt-6-luna"; the analysis computes cost from
# token counts instead.
litellm.suppress_debug_info = True


class ToolCallingChatModel(LitellmModel):
    """A Chat Completions model offering the harness tools.

    Not thread-safe; create one per episode.
    """

    api_flavor = ApiFlavor.CHAT

    def __init__(self, *, tools: Sequence[ToolSpec], **kwargs: Any) -> None:
        """Creates the model; `kwargs` are LitellmModelConfig fields."""
        super().__init__(**kwargs)
        self._wire_tools = chat_tools(tools)
        self._tool_names = tuple(tool.name for tool in tools)
        self._latency_s = 0.0

    def _query(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        start = time.monotonic()
        response = litellm.completion(
            model=self.config.model_name,
            messages=messages,
            tools=self._wire_tools,
            **(self.config.model_kwargs | kwargs),
        )
        self._latency_s = time.monotonic() - start
        return response

    def _parse_actions(self, response: Any) -> list[dict[str, Any]]:
        return parse_chat_tool_calls(
            response.choices[0].message.tool_calls, self._tool_names
        )

    def query(self, messages: list[dict[str, str]], **kwargs: Any) -> dict:
        """Queries the model; the message's `extra` carries turn metrics."""
        try:
            message = super().query(messages, **kwargs)
        except FormatError as error:
            extra = error.messages[0]["extra"]
            record_turn_metrics(
                error.messages[0], extra.get("response"), self._latency_s
            )
            raise
        record_turn_metrics(message, message["extra"]["response"], self._latency_s)
        return message


class ToolCallingResponsesModel(LitellmResponseModel):
    """A Responses API model offering the harness tools.

    The whole response, reasoning items included, is kept as the assistant
    message and sent back on the next turn, so a reasoning model keeps its
    chain of thought across tool calls. Not thread-safe; create one per
    episode.
    """

    api_flavor = ApiFlavor.RESPONSES

    def __init__(self, *, tools: Sequence[ToolSpec], **kwargs: Any) -> None:
        """Creates the model; `kwargs` are LitellmResponseModelConfig fields."""
        super().__init__(**kwargs)
        self._wire_tools = responses_tools(tools)
        self._tool_names = tuple(tool.name for tool in tools)
        self._latency_s = 0.0

    def _query(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        start = time.monotonic()
        response = litellm.responses(
            model=self.config.model_name,
            input=messages,
            tools=self._wire_tools,
            **(self.config.model_kwargs | kwargs),
        )
        self._latency_s = time.monotonic() - start
        return response

    def _parse_actions(self, response: Any) -> list[dict[str, Any]]:
        output = (
            response.get("output") if isinstance(response, dict) else response.output
        )
        return parse_responses_output(output, self._tool_names)

    def query(self, messages: list[dict[str, str]], **kwargs: Any) -> dict:
        """Queries the model; the message's `extra` carries turn metrics."""
        try:
            message = super().query(messages, **kwargs)
        except FormatError as error:
            extra = error.messages[0]["extra"]
            record_turn_metrics(
                error.messages[0], extra.get("response"), self._latency_s
            )
            raise
        # The message is the response itself, with usage at the top level.
        record_turn_metrics(message, message, self._latency_s)
        return message

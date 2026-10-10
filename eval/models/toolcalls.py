# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tool calls on the wire: encoding the harness tools, parsing the calls.

A harness model speaks one of two API flavors, Chat Completions or the
Responses API, but always reports calls as the same actions:
{"tool": name, "arguments": dict, "tool_call_id": id}. Parsing is
deliberately shallow. It checks only that the tool exists in the arm and
that the arguments are a JSON object, and raises a mini-swe-agent FormatError
with a fixed message otherwise; the arguments themselves are validated by
`FerretEnvironment`, which answers with an observation the model can act on.
"""

import enum
import json
from collections.abc import Iterable, Sequence
from typing import Any

from minisweagent.exceptions import FormatError

from harness.tools import NO_TOOL_CALL_ERROR, ToolSpec, unknown_tool_error

# Observations are shown verbatim: no return-code or XML wrapper, which would
# add identical tokens to every arm and carry no information.
OBSERVATION_TEMPLATE = "{{ output.output }}"


class ApiFlavor(enum.StrEnum):
    """The message format a model speaks, and its observations use."""

    CHAT = "chat"
    RESPONSES = "responses"


def chat_tools(tools: Iterable[ToolSpec]) -> list[dict[str, Any]]:
    """Encodes tools for Chat Completions (`{"type", "function": {...}}`)."""
    return [{"type": "function", "function": tool.to_json()} for tool in tools]


def responses_tools(tools: Iterable[ToolSpec]) -> list[dict[str, Any]]:
    """Encodes tools for the Responses API, which uses a flat object."""
    return [{"type": "function", **tool.to_json()} for tool in tools]


def parse_chat_tool_calls(
    tool_calls: Sequence[Any] | None, tool_names: Sequence[str]
) -> list[dict[str, Any]]:
    """Parses the `tool_calls` of a Chat Completions message.

    Args:
        tool_calls: litellm tool-call objects or equivalent dicts.
        tool_names: The arm's tools, in presentation order.

    Raises:
        FormatError: if there is no call, or a call names an unknown tool or
            has arguments that are not a JSON object.
    """
    if not tool_calls:
        raise format_error(NO_TOOL_CALL_ERROR)
    actions = []
    for call in tool_calls:
        function = _get(call, "function")
        actions.append(
            _action(
                _get(function, "name"),
                _get(function, "arguments"),
                _get(call, "id"),
                tool_names,
            )
        )
    return actions


def parse_responses_output(
    output: Sequence[Any] | None, tool_names: Sequence[str]
) -> list[dict[str, Any]]:
    """Parses the `function_call` items of a Responses API output.

    Raises:
        FormatError: as for `parse_chat_tool_calls`.
    """
    calls = [item for item in output or () if _get(item, "type") == "function_call"]
    if not calls:
        raise format_error(NO_TOOL_CALL_ERROR)
    return [
        _action(
            _get(call, "name"),
            _get(call, "arguments"),
            _get(call, "call_id"),
            tool_names,
        )
        for call in calls
    ]


def format_error(text: str) -> FormatError:
    """Returns the FormatError whose message tells the model `text`.

    The message is a plain user message, which both API flavors accept.
    """
    return FormatError(
        {"role": "user", "content": text, "extra": {"interrupt_type": "FormatError"}}
    )


def _action(
    name: object, arguments: object, call_id: object, tool_names: Sequence[str]
) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            # Some servers send "" for a call without arguments.
            arguments = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            arguments = None
    if name not in tool_names or not isinstance(arguments, dict):
        raise format_error(unknown_tool_error(tool_names))
    return {"tool": name, "arguments": arguments, "tool_call_id": call_id}


def _get(value: Any, key: str) -> Any:
    """Reads `key` from a dict or an attribute of a pydantic object."""
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)

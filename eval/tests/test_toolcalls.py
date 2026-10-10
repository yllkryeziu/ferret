# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for models.toolcalls: encoding tools and parsing calls."""

import types

import pytest
from minisweagent.exceptions import FormatError

from harness import tools
from models import toolcalls

NAMES = ("search", "submit")


def test_encodes_tools_for_both_flavors():
    chat = toolcalls.chat_tools([tools.SEARCH])
    assert chat == [{"type": "function", "function": tools.SEARCH.to_json()}]
    responses = toolcalls.responses_tools([tools.SEARCH])
    assert responses == [{"type": "function", **tools.SEARCH.to_json()}]


def test_parses_chat_tool_calls_from_dicts_and_objects():
    calls = [
        {"id": "a", "function": {"name": "search", "arguments": '{"pattern": "x"}'}},
        types.SimpleNamespace(
            id="b", function=types.SimpleNamespace(name="submit", arguments="")
        ),
    ]
    assert toolcalls.parse_chat_tool_calls(calls, NAMES) == [
        {"tool": "search", "arguments": {"pattern": "x"}, "tool_call_id": "a"},
        {"tool": "submit", "arguments": {}, "tool_call_id": "b"},
    ]


def test_parses_responses_function_calls_only():
    output = [
        {"type": "reasoning", "summary": []},
        {"type": "function_call", "name": "search", "arguments": "{}", "call_id": "c"},
    ]
    assert toolcalls.parse_responses_output(output, NAMES) == [
        {"tool": "search", "arguments": {}, "tool_call_id": "c"}
    ]


@pytest.mark.parametrize(
    "calls",
    [
        None,
        [],
        [{"id": "a", "function": {"name": "bash", "arguments": "{}"}}],
        [{"id": "a", "function": {"name": "search", "arguments": "[1]"}}],
        [{"id": "a", "function": {"name": "search", "arguments": "{bad json"}}],
    ],
)
def test_rejects_missing_unknown_and_malformed_calls(calls):
    with pytest.raises(FormatError) as error:
        toolcalls.parse_chat_tool_calls(calls, NAMES)
    message = error.value.messages[0]
    assert message["role"] == "user"
    assert message["content"].startswith("error:")
    assert message["extra"]["interrupt_type"] == "FormatError"


def test_no_call_message_asks_for_a_tool_call():
    with pytest.raises(FormatError) as error:
        toolcalls.parse_responses_output([{"type": "message"}], NAMES)
    assert error.value.messages[0]["content"] == tools.NO_TOOL_CALL_ERROR

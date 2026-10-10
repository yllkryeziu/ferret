# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for harness.tools: the tool specs and argument validation."""

import dataclasses

import pytest

from harness import tools
from harness.tools import ToolArgumentError, validate_arguments


def test_tools_are_presented_in_a_fixed_order():
    assert [tool.name for tool in tools.ALL_TOOLS] == [
        "search",
        "read",
        "ls",
        "find",
        "submit",
    ]


def test_tool_set_hash_is_stable_and_sensitive():
    first = tools.tool_set_sha256(tools.ALL_TOOLS)
    assert first == tools.tool_set_sha256(tools.ALL_TOOLS)
    changed = dataclasses.replace(tools.SEARCH, description="Search.")
    assert tools.tool_set_sha256((changed, *tools.ALL_TOOLS[1:])) != first


def test_to_json_returns_a_copy():
    spec = tools.SEARCH.to_json()
    spec["parameters"]["properties"].clear()
    assert tools.SEARCH.parameters["properties"]


def test_fills_defaults():
    assert validate_arguments(tools.SEARCH, {"pattern": "foo"}) == {
        "pattern": "foo",
        "regex": False,
        "case_sensitive": False,
    }
    assert validate_arguments(tools.READ, {"path": "a.py"}) == {
        "path": "a.py",
        "start_line": 1,
    }
    assert validate_arguments(tools.SUBMIT, {"files": ["a.py"]}) == {
        "files": ["a.py"],
        "functions": [],
    }


@pytest.mark.parametrize(
    ("spec", "arguments", "message"),
    [
        (tools.SEARCH, [], "arguments must be a JSON object"),
        (tools.SEARCH, {}, "missing argument 'pattern'"),
        (tools.SEARCH, {"pattern": "x", "limit": 3}, "unknown argument 'limit'"),
        (tools.SEARCH, {"pattern": 3}, "must be a string"),
        (tools.SEARCH, {"pattern": "x", "regex": "yes"}, "must be true or false"),
        (tools.READ, {"path": "a", "start_line": True}, "must be an integer"),
        (tools.READ, {"path": "a", "start_line": 0}, "must be at least 1"),
        (tools.SEARCH, {"pattern": "x" * 4097}, "longer than 4096"),
        (tools.SEARCH, {"pattern": "a\0b"}, "NUL"),
        (tools.SEARCH, {"pattern": "\ud800"}, "not valid Unicode"),
        (tools.SUBMIT, {"files": []}, "must have 1 to 5 items"),
        (tools.SUBMIT, {"files": ["a"] * 6}, "must have 1 to 5 items"),
        (tools.SUBMIT, {"files": [1]}, "must be an array of strings"),
    ],
)
def test_rejects_invalid_arguments(spec, arguments, message):
    with pytest.raises(ToolArgumentError, match=message) as error:
        validate_arguments(spec, arguments)
    assert error.value.message.startswith(f"error: {spec.name}")


def test_canonical_json_is_order_independent():
    assert tools.canonical_json({"b": 1, "a": [2]}) == '{"a":[2],"b":1}'

# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""The tools the agent is offered, and the fixed messages it gets back.

The experiment's claim is that arms differ only in what `search` returns, so
the tool names, descriptions, and JSON schemas are protocol constants: every
arm that offers a tool offers the byte-identical spec, and
`tool_set_sha256` is logged in every trajectory. Arguments are validated
against the same schemas the model is shown, so what the model is told and
what the harness enforces cannot drift apart.

The limits below appear inside the tool descriptions; changing one changes the
tool-set hash, which is intended.
"""

import copy
import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from jsonschema import Draft202012Validator

# Upper bound on every string argument. It keeps a runaway model from sending
# megabytes to the search server or the filesystem layer.
MAX_ARGUMENT_CHARS = 4096

# A `read` returns at most this many lines and this many bytes of UTF-8, the
# same in every arm.
READ_PAGE_LINES = 200
READ_PAGE_BYTES = 16 * 1024

# `ls` and `find` return at most this many entries.
LIST_MAX_ENTRIES = 200

SUBMIT_MAX_FILES = 5
SUBMIT_MAX_FUNCTIONS = 10


@dataclasses.dataclass(frozen=True, slots=True)
class ToolSpec:
    """A provider-neutral tool definition: name, description, JSON schema.

    Instances are module constants; treat `parameters` as immutable and use
    `to_json` to get a copy for an API request.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        """Returns a deep copy of this spec as a JSON object."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": copy.deepcopy(dict(self.parameters)),
        }


def _string(description: str) -> dict[str, Any]:
    return {
        "type": "string",
        "maxLength": MAX_ARGUMENT_CHARS,
        "description": description,
    }


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


SEARCH = ToolSpec(
    name="search",
    description="Search file contents in the repository. Returns matching lines.",
    parameters=_object(
        {
            "pattern": _string(
                "Text to search for; a regular expression in RE2 syntax if regex is "
                "true."
            ),
            "regex": {
                "type": "boolean",
                "default": False,
                "description": "Treat pattern as a regular expression.",
            },
            "path_glob": _string(
                "Gitignore-style path glob: a matched directory includes "
                "descendants; dir/* includes files below matching directories."
            ),
            "case_sensitive": {
                "type": "boolean",
                "default": False,
                "description": "Match letter case exactly.",
            },
        },
        required=["pattern"],
    ),
)

READ = ToolSpec(
    name="read",
    description=(
        f"Read a file. Returns up to {READ_PAGE_LINES} numbered lines, at most "
        f"{READ_PAGE_BYTES // 1024} KiB, starting at start_line."
    ),
    parameters=_object(
        {
            "path": _string("File path relative to the repository root."),
            "start_line": {
                "type": "integer",
                "minimum": 1,
                "default": 1,
                "description": "Number of the first line to return.",
            },
        },
        required=["path"],
    ),
)

LS = ToolSpec(
    name="ls",
    description=(
        f"List a directory. Returns up to {LIST_MAX_ENTRIES} entries in sorted "
        "order; directory names end with '/'."
    ),
    parameters=_object(
        {
            "path": {
                **_string("Directory path relative to the repository root."),
                "default": ".",
            },
        },
        required=[],
    ),
)

FIND = ToolSpec(
    name="find",
    description=(
        f"Find files by path. Returns up to {LIST_MAX_ENTRIES} matching file paths "
        "in sorted order."
    ),
    parameters=_object(
        {
            "glob": _string(
                "Glob over paths relative to the repository root. '*' and '?' match "
                "within a component, but a matched directory includes all descendants "
                "(dir/* includes files below matched directories). '**' crosses "
                "directories. A glob "
                "without '/' matches file names at any depth."
            ),
        },
        required=["glob"],
    ),
)

SUBMIT = ToolSpec(
    name="submit",
    description="Submit your answer and end the task.",
    parameters=_object(
        {
            "files": {
                "type": "array",
                "items": {"type": "string", "maxLength": MAX_ARGUMENT_CHARS},
                "minItems": 1,
                "maxItems": SUBMIT_MAX_FILES,
                "description": (
                    "Paths of the files that need to change, most likely first."
                ),
            },
            "functions": {
                "type": "array",
                "items": {"type": "string", "maxLength": MAX_ARGUMENT_CHARS},
                "maxItems": SUBMIT_MAX_FUNCTIONS,
                "default": [],
                "description": (
                    "Functions that need to change, as 'path::name', most likely first."
                ),
            },
        },
        required=["files"],
    ),
)

# Every tool, in the order it is presented to the model.
ALL_TOOLS: tuple[ToolSpec, ...] = (SEARCH, READ, LS, FIND, SUBMIT)

# Parse-level failures, raised as mini-swe-agent FormatErrors by the models.
NO_TOOL_CALL_ERROR = "error: no tool call; respond with a call to one of the tools"

# Returned for any non-submit call once the tool-call budget is spent.
OVER_BUDGET_ERROR = "error: no tool calls remain; call submit with your best answer"


def unknown_tool_error(tool_names: Iterable[str]) -> str:
    """Returns the fixed message for an unknown tool or non-object arguments."""
    return "error: unknown tool or malformed arguments; available tools: " + ", ".join(
        tool_names
    )


def canonical_json(value: Any) -> str:
    """Serializes `value` with sorted keys and no whitespace, for hashing."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def tool_set_sha256(tools: Sequence[ToolSpec]) -> str:
    """Returns the SHA-256 of the canonical JSON of `tools`, in order."""
    encoded = canonical_json([tool.to_json() for tool in tools]).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ToolArgumentError(ValueError):
    """Arguments that violate a tool's schema; `message` is shown to the model."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def validate_arguments(
    spec: ToolSpec, arguments: object, *, fill_defaults: bool = True
) -> dict[str, Any]:
    """Checks `arguments` against `spec` and fills in schema defaults.

    Beyond the schema, every string must be free of NUL characters and of
    unpaired surrogates, which no path or pattern legitimately contains.

    Args:
        spec: The tool being called.
        arguments: The decoded JSON arguments of the call.

    Returns:
        A new dict with every argument the schema declares a default for.

    Raises:
        ToolArgumentError: naming the first offending argument, checked in a
            deterministic order.
    """
    tool = spec.name
    if not isinstance(arguments, dict):
        raise ToolArgumentError(f"error: {tool}: arguments must be a JSON object")
    if tool == "search" and spec is not SEARCH:
        return _validate_search(spec, arguments)
    properties: Mapping[str, Mapping[str, Any]] = spec.parameters["properties"]
    for name in sorted(arguments):
        if name not in properties:
            raise ToolArgumentError(f"error: {tool}: unknown argument {name[:64]!r}")
    for name in spec.parameters["required"]:
        if name not in arguments:
            raise ToolArgumentError(f"error: {tool}: missing argument {name!r}")
    result: dict[str, Any] = {}
    for name, schema in properties.items():
        if name in arguments:
            result[name] = _check_value(
                f"{tool}: argument {name!r}", schema, arguments[name]
            )
        elif fill_defaults and "default" in schema:
            result[name] = copy.deepcopy(schema["default"])
    return result


def _validate_search(spec: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    """Uses the exact search JSON schema offered to the model, without defaults."""
    properties = spec.parameters.get("properties", {})
    if spec.parameters.get("additionalProperties") is False:
        for name in sorted(arguments):
            if name not in properties:
                raise ToolArgumentError(
                    f"error: search: unknown argument {name[:64]!r}"
                )
    errors = Draft202012Validator(spec.parameters).iter_errors(arguments)
    error = next(
        iter(sorted(errors, key=lambda e: (list(map(str, e.path)), e.message))), None
    )
    if error is not None:
        location = "".join(f"[{part!r}]" for part in error.path)
        raise ToolArgumentError(f"error: search{location}: {error.message}")
    return dict(arguments)


def _check_value(what: str, schema: Mapping[str, Any], value: object) -> Any:
    kind = schema["type"]
    if kind == "string":
        if not isinstance(value, str):
            raise ToolArgumentError(f"error: {what} must be a string")
        _check_string(what, schema, value)
    elif kind == "boolean":
        if not isinstance(value, bool):
            raise ToolArgumentError(f"error: {what} must be true or false")
    elif kind == "integer":
        # bool is a subclass of int; JSON true is not an integer.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolArgumentError(f"error: {what} must be an integer")
        if value < schema.get("minimum", 0):
            raise ToolArgumentError(
                f"error: {what} must be at least {schema['minimum']}"
            )
    elif kind == "array":
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ToolArgumentError(f"error: {what} must be an array of strings")
        low, high = schema.get("minItems", 0), schema.get("maxItems", len(value))
        if not low <= len(value) <= high:
            raise ToolArgumentError(f"error: {what} must have {low} to {high} items")
        for item in value:
            _check_string(what, schema["items"], item)
    else:
        raise AssertionError(f"unsupported schema type {kind!r}")
    return value


def _check_string(what: str, schema: Mapping[str, Any], value: str) -> None:
    if len(value) > schema.get("maxLength", MAX_ARGUMENT_CHARS):
        raise ToolArgumentError(
            f"error: {what} is longer than {schema['maxLength']} characters"
        )
    if "\0" in value:
        raise ToolArgumentError(f"error: {what} contains a NUL character")
    if any("\ud800" <= char <= "\udfff" for char in value):
        raise ToolArgumentError(f"error: {what} is not valid Unicode")

# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""The mini-swe-agent environment: five tools, no shell.

`FerretEnvironment.execute` accepts exactly the tools of the current arm
(`harness.tools`): `search` is forwarded to a `SearchBackend`, `read`, `ls`,
and `find` are served by a `RepoSandbox`, and `submit` ends the episode.
There is no shell, no network, and no path into `.git`, which prevents
leakage of the fix without Docker.

Why five native tools rather than mini-swe-agent's default `bash` tool: a
tool named `bash` invites pipes, redirections, and real grep, all of which a
strict parser must reject, and the rate of such rejections depends on the
output the model has just seen, so it would differ by arm and confound the
comparison. Native tools also carry regex patterns as JSON strings, with no
shell-quoting layer to mangle backslashes and quotes. In mini-swe-agent 2.4.6
the tool schema is not a separate hook: it is inlined in the models' `_query`
and checked by name in `_parse_actions`, so `models.litellm_models` overrides
exactly those two methods and leaves the rest of the model and the agent
loop unchanged.

Every call except a valid `submit` spends one unit of the tool-call budget,
invalid calls included, so no model can loop forever. Once the budget is
spent only `submit` is accepted. Each observation records the tool, a
status, the tool's own latency, and the output size, for the trajectory.
"""

import dataclasses
import enum
import time
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from minisweagent.exceptions import Submitted

from harness.sandbox import RepoSandbox, SandboxError
from harness.search import SearchBackend, SearchTimeoutError
from harness.tools import (
    OVER_BUDGET_ERROR,
    ToolArgumentError,
    ToolSpec,
    canonical_json,
    unknown_tool_error,
    validate_arguments,
)


class ToolStatus(enum.StrEnum):
    """The outcome of one tool call, recorded in its observation."""

    OK = "ok"
    # The tool ran and reported a failure: a missing file, a bad regex.
    ERROR = "error"
    TIMEOUT = "timeout"
    # The harness refused the call: a schema violation or a sandbox escape.
    # Counted as an invalid tool call.
    REJECTED = "rejected"
    # The call arrived after the budget was spent and was not executed.
    OVER_BUDGET = "over_budget"


@dataclasses.dataclass(frozen=True, slots=True)
class FerretEnvironmentConfig:
    """What `FerretEnvironment.serialize` reports about the environment."""

    root: str
    tools: tuple[str, ...]
    max_tool_calls: int


class FerretEnvironment:
    """Executes tool calls for one episode.

    Not thread-safe; create one per episode. Implements mini-swe-agent's
    `Environment` protocol.
    """

    def __init__(
        self,
        *,
        sandbox: RepoSandbox,
        tools: Sequence[ToolSpec],
        search: SearchBackend | None,
        max_tool_calls: int,
    ) -> None:
        """Creates the environment.

        Args:
            sandbox: The snapshot `read`, `ls`, and `find` operate on.
            tools: The arm's tools; calls to any other tool are rejected.
            search: The search backend; required iff `tools` includes
                `search`.
            max_tool_calls: The tool-call budget.
        """
        names = tuple(tool.name for tool in tools)
        if ("search" in names) != (search is not None):
            raise ValueError(
                "a search backend is required exactly when search is offered"
            )
        self.config = FerretEnvironmentConfig(
            root=sandbox.root,
            tools=names,
            max_tool_calls=max_tool_calls,
        )
        self._sandbox = sandbox
        self._tools = {tool.name: tool for tool in tools}
        self._search = search
        self._calls_remaining = max_tool_calls

    @property
    def calls_remaining(self) -> int:
        """How many more non-submit calls will be executed."""
        return self._calls_remaining

    def end_exploration(self) -> None:
        """Spends the rest of the budget, so only `submit` is accepted."""
        self._calls_remaining = 0

    def execute(self, action: dict[str, Any], cwd: str = "") -> dict[str, Any]:
        """Runs one tool call and returns its observation.

        Args:
            action: {"tool": name, "arguments": decoded JSON arguments, ...},
                as produced by the harness models.
            cwd: Ignored; part of the mini-swe-agent protocol.

        Raises:
            Submitted: for a valid `submit`; `extra.submission` holds the
                canonical JSON of {"files", "functions"}.
        """
        del cwd
        tool = action.get("tool")
        start = time.monotonic()
        text, status = self._run(tool, action.get("arguments"))
        return {
            "output": text,
            "returncode": 0 if status is ToolStatus.OK else 1,
            "exception_info": "",
            "extra": {
                "tool": tool if isinstance(tool, str) else None,
                "status": status.value,
                "tool_latency_s": time.monotonic() - start,
                "output_bytes": len(text.encode("utf-8")),
            },
        }

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        """Returns `kwargs` only; the process environment must never reach a
        prompt, unlike in mini-swe-agent's LocalEnvironment."""
        return dict(kwargs)

    def serialize(self) -> dict[str, Any]:
        """Describes the environment for mini-swe-agent's trajectory format."""
        return {
            "info": {
                "config": {
                    "environment": dataclasses.asdict(self.config),
                    "environment_type": f"{__name__}.{type(self).__name__}",
                }
            }
        }

    def _run(self, tool: object, arguments: object) -> tuple[str, ToolStatus]:
        spec = self._tools.get(tool) if isinstance(tool, str) else None
        if spec is not None and spec.name == "submit":
            try:
                submission = validate_arguments(spec, arguments)
            except ToolArgumentError as error:
                self._calls_remaining = max(self._calls_remaining - 1, 0)
                return error.message, ToolStatus.REJECTED
            _submit(submission)
        if self._calls_remaining <= 0:
            return OVER_BUDGET_ERROR, ToolStatus.OVER_BUDGET
        self._calls_remaining -= 1
        try:
            if spec is None:
                raise ToolArgumentError(unknown_tool_error(self._tools))
            return self._dispatch(
                spec.name,
                validate_arguments(
                    spec, arguments, fill_defaults=spec.name != "search"
                ),
            )
        except ToolArgumentError as error:
            return error.message, ToolStatus.REJECTED
        except SandboxError as error:
            status = ToolStatus.REJECTED if error.rejected else ToolStatus.ERROR
            return error.message, status

    def _dispatch(
        self, tool: str, arguments: Mapping[str, Any]
    ) -> tuple[str, ToolStatus]:
        match tool:
            case "search":
                return self._run_search(arguments)
            case "read":
                page = self._sandbox.read_page(
                    arguments["path"], arguments["start_line"]
                )
                return page, ToolStatus.OK
            case "ls":
                return self._sandbox.list_dir(arguments["path"]), ToolStatus.OK
            case "find":
                return self._sandbox.find(arguments["glob"]), ToolStatus.OK
        raise AssertionError(f"no handler for tool {tool!r}")

    def _run_search(self, arguments: Mapping[str, Any]) -> tuple[str, ToolStatus]:
        assert self._search is not None
        try:
            result = self._search.search(arguments)
        except SearchTimeoutError as error:
            return f"error: {error}", ToolStatus.TIMEOUT
        return result.text, ToolStatus.ERROR if result.is_error else ToolStatus.OK


def _submit(submission: Mapping[str, Any]) -> NoReturn:
    payload = canonical_json(
        {
            "files": submission["files"],
            "functions": submission["functions"],
        }
    )
    raise Submitted(
        {
            "role": "exit",
            "content": payload,
            "extra": {"exit_status": "Submitted", "submission": payload},
        }
    )

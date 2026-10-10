# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""The backend of the `search` tool: the ferret command-line tool.

Each call runs `ferret search` over the task's prebuilt index with the arm's
output shape, exactly as an agent with shell access would run it. Exit
status 0 (matches) and 1 (no match) both carry the formatted results; 2 is
an error, such as an invalid regular expression, whose message the model
sees. Some shapes print nothing when nothing matches, as grep does; the
model then sees `NO_MATCHES` rather than an empty observation, much as a
shell would show it the exit status. The child's environment is reduced to
PATH, HOME, and TMPDIR, so model credentials never reach it.
"""

import dataclasses
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from harness.arms import Shape

_CHILD_ENVIRONMENT_VARIABLES = ("HOME", "PATH", "TMPDIR")

NO_MATCHES = "no matches"


@dataclasses.dataclass(frozen=True, slots=True)
class SearchResult:
    """What one search returned.

    Attributes:
        text: The model-facing output: the formatted results, or an error
            message starting with "error: ".
        is_error: Whether the search failed, for example on a bad pattern.
    """

    text: str
    is_error: bool


class SearchTimeoutError(Exception):
    """The search did not finish within its deadline."""


class SearchBackend(Protocol):
    """Runs searches with validated `search` tool arguments."""

    def search(self, arguments: Mapping[str, Any]) -> SearchResult:
        """Runs one search.

        Raises:
            SearchTimeoutError: if the search exceeds its deadline.
        """
        ...

    def describe(self) -> dict[str, Any]:
        """Returns the backend's provenance for the trajectory."""
        ...


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FerretCliSearch:
    """Searches with `ferret search`. Thread-safe.

    Attributes:
        binary: The ferret executable.
        index: The prebuilt index of the task's snapshot.
        shape: The output shape of the arm.
        timeout_s: Deadline of one search.
    """

    binary: Path
    index: Path
    shape: Shape
    timeout_s: float = 60.0

    def argv(self, arguments: Mapping[str, Any]) -> list[str]:
        """Returns the command line for validated `search` arguments."""
        argv = [
            str(self.binary),
            "search",
            f"--index={self.index}",
            f"--shape={self.shape}",
        ]
        if arguments.get("regex"):
            argv.append("--regex")
        if arguments.get("case_sensitive"):
            argv.append("--case_sensitive")
        if arguments.get("path_glob"):
            argv.append(f"--path={arguments['path_glob']}")
        # "--" keeps a pattern that starts with '-' from reading as a flag.
        return [*argv, "--", arguments["pattern"]]

    def search(self, arguments: Mapping[str, Any]) -> SearchResult:
        """Runs one search; see the module docstring.

        Raises:
            SearchTimeoutError: if ferret exceeds `timeout_s`.
            RuntimeError: if ferret dies or exits with an unknown status.
        """
        try:
            completed = subprocess.run(
                self.argv(arguments),
                capture_output=True,
                timeout=self.timeout_s,
                env=_child_environment(),
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise SearchTimeoutError(
                f"search timed out after {self.timeout_s:g} s"
            ) from None
        stdout = completed.stdout.decode("utf-8", "replace")
        if completed.returncode in (0, 1):
            return SearchResult(stdout or NO_MATCHES, is_error=False)
        stderr = completed.stderr.decode("utf-8", "replace").strip()
        if completed.returncode == 2:
            message = stderr.removeprefix("ferret: ").splitlines()[0] if stderr else ""
            return SearchResult(f"error: {message}", is_error=True)
        raise RuntimeError(
            f"ferret search exited with status {completed.returncode}: {stderr}"
        )

    def describe(self) -> dict[str, Any]:
        """Returns the command template, which contains no secrets."""
        return {
            "kind": "ferret_cli",
            "binary": str(self.binary),
            "index": str(self.index),
            "shape": str(self.shape),
            "timeout_s": self.timeout_s,
        }


def _child_environment() -> dict[str, str]:
    return {
        name: os.environ[name]
        for name in _CHILD_ENVIRONMENT_VARIABLES
        if name in os.environ
    }

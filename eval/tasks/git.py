# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Runs git with an argv and a configuration-independent environment.

User and system configuration can change git's output format (color,
`diff.noprefix`, `core.quotepath`, pagers, `log.showSignature`), so every
command runs with them disabled; only the repository's own configuration
applies. Arguments are passed as an argv, never through a shell.
"""

import os
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

_ISOLATION = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_ATTR_NOSYSTEM": "1",
    # Every pathspec this package passes is a literal path, never a glob.
    "GIT_LITERAL_PATHSPECS": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "LC_ALL": "C",
}


class GitError(RuntimeError):
    """A git command exited with an unexpected status."""


class Git:
    """Runs git commands in one repository. Thread-compatible."""

    def __init__(self, repo: Path) -> None:
        self._repo = repo

    @property
    def repo(self) -> Path:
        """The repository's worktree or git directory."""
        return self._repo

    def run(
        self,
        *args: str,
        env: Mapping[str, str] | None = None,
        stdin: bytes | None = None,
    ) -> bytes:
        """Runs `git <args>` and returns its stdout.

        Args:
            args: The git arguments.
            env: Extra environment variables, such as commit identities.
            stdin: Bytes to feed to the command.

        Raises:
            GitError: if git exits with a non-zero status.
        """
        result = self._spawn(args, env, stdin)
        if result.returncode != 0:
            raise GitError(self._describe(args, result))
        return result.stdout

    def text(self, *args: str) -> str:
        """Runs `git <args>` and returns stdout decoded as UTF-8.

        Undecodable bytes (legacy commit messages, Latin-1 sources) become
        U+FFFD rather than failing.
        """
        return self.run(*args).decode("utf-8", "replace")

    def test(self, *args: str) -> bool:
        """Runs a predicate command such as `merge-base --is-ancestor`.

        Returns:
            True on exit status 0 and False on 1.

        Raises:
            GitError: on any other status.
        """
        result = self._spawn(args, None, None)
        if result.returncode not in (0, 1):
            raise GitError(self._describe(args, result))
        return result.returncode == 0

    def resolve(self, revision: str) -> str | None:
        """Returns the full id of the commit `revision` names, or None if none."""
        result = self._spawn(
            ("rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}"), None, None
        )
        return result.stdout.decode().strip() if result.returncode == 0 else None

    def popen(self, *args: str) -> subprocess.Popen[bytes]:
        """Starts `git <args>` with stdout as a pipe, for streaming output."""
        return subprocess.Popen(
            ["git", "-C", str(self._repo), *args],
            stdout=subprocess.PIPE,
            env=os.environ | _ISOLATION,
        )

    def _spawn(
        self,
        args: tuple[str, ...],
        env: Mapping[str, str] | None,
        stdin: bytes | None,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["git", "-C", str(self._repo), *args],
            input=stdin,
            capture_output=True,
            env=os.environ | _ISOLATION | dict(env or {}),
            check=False,
        )

    def _describe(
        self, args: tuple[str, ...], result: subprocess.CompletedProcess[bytes]
    ) -> str:
        stderr = result.stderr.decode("utf-8", "replace").strip()
        command = " ".join(args)
        return (
            f"git {command} in {self._repo} exited with {result.returncode}: {stderr}"
        )


def tree_blobs(
    git: Git, commit: str, paths: Iterable[str] = ()
) -> dict[str, tuple[str, str]]:
    """Maps each non-submodule path in `commit` to its (mode, blob id).

    Lists only `paths` when any are given; paths absent from the tree are
    absent from the result.
    """
    blobs = {}
    for record in git.text("ls-tree", "-r", "-z", commit, "--", *paths).split("\0"):
        if not record:
            continue
        header, _, path = record.partition("\t")
        mode, kind, object_id = header.split(" ")
        if kind == "blob":
            blobs[path] = (mode, object_id)
    return blobs

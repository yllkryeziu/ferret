# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Throwaway git repositories with controlled history, for pipeline tests.

Commits carry explicit author and committer dates, so tests can place fixes
inside or outside a snapshot's window, and a fixed identity, so the same
inputs always produce the same commit ids.
"""

from collections.abc import Mapping
from pathlib import Path

from tasks.git import Git


class ToyRepo:
    """A git repository built commit by commit. Not thread-safe."""

    def __init__(self, root: Path) -> None:
        """Creates an empty repository at `root`, which must not exist."""
        root.mkdir(parents=True)
        self._root = root
        self._git = Git(root)
        self._git.run("init", "--quiet", "--initial-branch=main")

    @property
    def git(self) -> Git:
        """A runner for this repository."""
        return self._git

    def commit(
        self,
        message: str,
        files: Mapping[str, str | None],
        *,
        date: str,
        author_date: str | None = None,
    ) -> str:
        """Writes `files` (None deletes), commits them, and returns the commit id.

        Args:
            message: The full commit message.
            files: Repository-relative path to new content, or None to delete.
            date: Committer date, ISO 8601.
            author_date: Author date; defaults to `date`.
        """
        for path, content in files.items():
            target = self._root / path
            if content is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
        self._git.run("add", "--all")
        self._git.run(
            "commit", "--quiet", "--allow-empty", "--no-verify", "-m", message,
            env={
                "GIT_AUTHOR_NAME": "Toy Author",
                "GIT_AUTHOR_EMAIL": "author@toy.invalid",
                "GIT_AUTHOR_DATE": author_date or date,
                "GIT_COMMITTER_NAME": "Toy Committer",
                "GIT_COMMITTER_EMAIL": "committer@toy.invalid",
                "GIT_COMMITTER_DATE": date,
            },
        )  # fmt: skip
        return self._git.text("rev-parse", "HEAD").strip()

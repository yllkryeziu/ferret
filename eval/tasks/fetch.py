# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""A local cache of the commits that tasks are snapshotted from.

Each repository gets one bare cache repository, and each commit is fetched
by id at depth 1: its whole tree in one pack and none of its history, so
even a large repository costs one download per snapshot rather than a full
clone. Fetching an unadvertised commit needs protocol v2 or a server that
allows it; GitHub does.
"""

from pathlib import Path

from tasks.git import Git, GitError
from tasks.schema import is_normalized_path


class CommitCache:
    """Bare repositories under one root, one per repository name.

    Thread-compatible; separate processes must not fetch into the same
    repository at once.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def fetch(self, name: str, url: str, commit: str) -> Git:
        """Returns the cache repository of `name`, which then holds `commit`.

        Fetches `commit` from `url` unless the cache already has it, and
        keeps it under refs/commits/ so that garbage collection never prunes
        it.

        Args:
            name: The repository's name, such as "django/django"; a normalized
                relative path.
            url: Where to fetch from.
            commit: A full commit id.

        Raises:
            ValueError: if `name` is not a normalized relative path.
            GitError: if the fetch fails or does not produce `commit`.
        """
        if not is_normalized_path(name):
            raise ValueError(f"repository name {name!r} is not a normalized path")
        path = self._root / f"{name}.git"
        git = Git(path)
        if not path.exists():
            path.mkdir(parents=True)
            git.run("init", "--quiet", "--bare")
        if git.resolve(commit) != commit:
            git.run(
                "fetch", "--quiet", "--depth=1", "--no-tags", url,
                f"{commit}:refs/commits/{commit}",
            )  # fmt: skip
            if git.resolve(commit) != commit:
                raise GitError(f"fetching {commit} from {url} did not produce it")
        return git

# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""History-free snapshots: the only repositories agents ever see.

A normal checkout carries the future: its object store holds the very fix
commit a task was derived from. A snapshot instead holds only the objects of
the snapshot commit's tree, committed as the only commit of a fresh
repository, so neither `.git` nor the worktree reveals anything after the
snapshot.

Ferret indexes by git blob id, so a snapshot must reproduce every blob
exactly. Copying the tree's objects, rather than exporting files and adding
them back, makes that true by construction: no attribute-driven conversion
(`text`, `eol`, `filter`, `export-ignore`, `export-subst`) runs on the way,
and the snapshot's tree id equals the source's. The worktree is checked out
with those conversions disabled, and `export_snapshot` verifies that every
file then holds its blob's bytes.

On macOS, export Linux onto a case-sensitive volume: it tracks paths that
differ only in case, which a case-insensitive filesystem cannot hold, and
verification rejects the resulting worktree.
"""

from collections.abc import Iterable
from pathlib import Path

from tasks.git import Git, tree_blobs

# Fixed identity and date make the snapshot commit id a function of the tree.
_SNAPSHOT_IDENTITY = {
    "GIT_AUTHOR_NAME": "Ferret Eval",
    "GIT_AUTHOR_EMAIL": "eval@ferret.invalid",
    "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
    "GIT_COMMITTER_NAME": "Ferret Eval",
    "GIT_COMMITTER_EMAIL": "eval@ferret.invalid",
    "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
}
# Highest-precedence attributes in the snapshot repository: check every file
# out byte for byte, whatever the tree's own .gitattributes say.
_VERBATIM_ATTRIBUTES = "* -text -eol -filter -ident -working-tree-encoding\n"
# Differing paths named in a SnapshotMismatchError.
_EXAMPLES = 5


class SnapshotMismatchError(RuntimeError):
    """The snapshot's worktree does not hold the source commit's blobs."""


def export_snapshot(source: Git, commit: str, destination: Path) -> str:
    """Materializes `commit` of `source` as a one-commit repository.

    Args:
        source: The repository holding `commit` and all objects of its tree.
        commit: The snapshot commit.
        destination: A directory to create; it must not exist.

    Returns:
        The commit id of the snapshot repository's only commit.

    Raises:
        FileExistsError: if `destination` exists.
        GitError: if a git command fails.
        SnapshotMismatchError: if a checked-out file differs from its blob.
    """
    tree = source.text("rev-parse", "--verify", f"{commit}^{{tree}}").strip()
    destination.mkdir(parents=True)
    snapshot = Git(destination)
    snapshot.run("init", "--quiet", "--initial-branch=snapshot")
    git_dir = destination / ".git"
    (git_dir / "info").mkdir(exist_ok=True)
    (git_dir / "info" / "attributes").write_text(_VERBATIM_ATTRIBUTES)
    # With --revs, pack-objects packs the tree and everything it reaches,
    # which excludes submodule commits, into the snapshot's object store.
    source.run(
        "pack-objects", "--revs", "--quiet", str(git_dir / "objects/pack/pack"),
        stdin=f"{tree}\n".encode(),
    )  # fmt: skip
    snapshot.run("read-tree", tree)
    snapshot.run("checkout-index", "--all", "--force")
    snapshot_commit = (
        snapshot.run(
            "commit-tree", tree, "-m", f"snapshot {commit}", env=_SNAPSHOT_IDENTITY
        )
        .decode()
        .strip()
    )
    snapshot.run("update-ref", "refs/heads/snapshot", snapshot_commit)
    _verify_worktree(snapshot, commit)
    return snapshot_commit


def missing_gold_paths(source: Git, commit: str, paths: Iterable[str]) -> list[str]:
    """Returns the `paths` that are not files or symlinks in `commit`."""
    paths = set(paths)
    return sorted(paths - tree_blobs(source, commit, paths).keys())


def _verify_worktree(snapshot: Git, commit: str) -> None:
    # Rehashes files whose stat data changed since checkout; files that still
    # differ from their index entries are reported by diff-files.
    snapshot.test("update-index", "-q", "--really-refresh")
    differing = [
        path
        for path in snapshot.text("diff-files", "--name-only", "-z").split("\0")
        if path
    ]
    if differing:
        raise SnapshotMismatchError(
            f"snapshot of {commit}: files differ from their blobs "
            f"({len(differing)} total), e.g. {differing[:_EXAMPLES]}; on macOS, export "
            "onto a case-sensitive volume"
        )

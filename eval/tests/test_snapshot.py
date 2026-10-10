# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tasks.snapshot and tasks.fetch: history-free, blob-exact snapshots."""

import os

import pytest

from tasks.fetch import CommitCache
from tasks.git import Git, GitError, tree_blobs
from tasks.snapshot import SnapshotMismatchError, export_snapshot, missing_gold_paths
from tasks.testing.toy_repo import ToyRepo


@pytest.fixture
def history(tmp_path):
    """A repository whose second commit fixes a bug in the first."""
    repo = ToyRepo(tmp_path / "upstream")
    repo.commit(
        "Add the parser",
        {
            "src/parse.c": "int parse(const char *p) { return p[1]; }\n",
            "src/crlf.txt": "windows\r\nline\r\n",
            ".gitattributes": "*.txt text eol=crlf\n",
            "README": "toy\n",
        },
        date="2025-01-01T00:00:00Z",
    )
    (repo.git.repo / "run.sh").write_text("#!/bin/sh\n")
    (repo.git.repo / "run.sh").chmod(0o755)
    (repo.git.repo / "link").symlink_to("src/parse.c")
    base = repo.commit("Add the runner", {}, date="2025-01-02T00:00:00Z")
    fix = repo.commit(
        "Fix an off-by-one in parse()",
        {"src/parse.c": "int parse(const char *p) { return p[0]; }\n"},
        date="2025-02-01T00:00:00Z",
    )
    return repo, base, fix


def test_snapshot_reproduces_the_tree_without_history(history, tmp_path):
    repo, base, fix = history
    destination = tmp_path / "snapshot"
    snapshot_commit = export_snapshot(repo.git, base, destination)

    snapshot = Git(destination)
    assert snapshot.text("rev-parse", "HEAD").strip() == snapshot_commit
    assert snapshot.text("rev-list", "--all").split() == [snapshot_commit]
    assert tree_blobs(snapshot, "HEAD") == tree_blobs(repo.git, base)
    # Neither the fix commit nor the fixed blob is reachable or even present.
    fixed_blob = tree_blobs(repo.git, fix, ["src/parse.c"])["src/parse.c"][1]
    for object_id in (fix, fixed_blob):
        assert not snapshot.test("cat-file", "-e", object_id)
    assert (destination / "src/parse.c").read_text().endswith("p[1]; }\n")
    # The blob was normalized to LF, and the checkout does not convert it back.
    assert (destination / "src/crlf.txt").read_bytes() == b"windows\nline\n"
    assert os.access(destination / "run.sh", os.X_OK)
    assert os.readlink(destination / "link") == "src/parse.c"


def test_snapshot_commit_is_a_function_of_the_tree(history, tmp_path):
    repo, base, _ = history
    first = export_snapshot(repo.git, base, tmp_path / "one")
    second = export_snapshot(repo.git, base, tmp_path / "two")
    assert first == second


def test_destination_must_not_exist(history, tmp_path):
    repo, base, _ = history
    (tmp_path / "taken").mkdir()
    with pytest.raises(FileExistsError):
        export_snapshot(repo.git, base, tmp_path / "taken")


def test_attributes_never_alter_blobs(tmp_path):
    repo = ToyRepo(tmp_path / "upstream")
    commit = repo.commit(
        "Attributes that change archives and checkouts",
        {
            ".gitattributes": (
                "ci.yml export-ignore\nversion.txt export-subst ident\n*.bat eol=crlf\n"
            ),
            "ci.yml": "jobs: {}\n",
            "version.txt": "$Format:%H$ $Id$\n",
            "run.bat": "echo hi\n",
        },
        date="2025-01-01T00:00:00Z",
    )
    destination = tmp_path / "snapshot"
    export_snapshot(repo.git, commit, destination)
    assert tree_blobs(Git(destination), "HEAD") == tree_blobs(repo.git, commit)
    assert (destination / "ci.yml").exists()
    assert (destination / "version.txt").read_text() == "$Format:%H$ $Id$\n"
    assert (destination / "run.bat").read_bytes() == b"echo hi\n"


_IDENTITY = {
    "GIT_AUTHOR_NAME": "A",
    "GIT_AUTHOR_EMAIL": "a@a.invalid",
    "GIT_COMMITTER_NAME": "A",
    "GIT_COMMITTER_EMAIL": "a@a.invalid",
}


def _commit_tree(git, files):
    """Commits `files` through the index alone, so no worktree must hold them."""
    for path, content in files.items():
        blob = git.run("hash-object", "-w", "--stdin", stdin=content).decode().strip()
        git.run("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
    tree = git.text("write-tree").strip()
    return git.run("commit-tree", tree, "-m", "case", env=_IDENTITY).decode().strip()


def test_rejects_paths_the_filesystem_cannot_hold(tmp_path):
    (tmp_path / "probe").touch()
    if not (tmp_path / "PROBE").exists():
        pytest.skip("needs a case-insensitive filesystem")
    repo = ToyRepo(tmp_path / "upstream")
    commit = _commit_tree(repo.git, {"Kconfig": b"upper\n", "kconfig": b"lower\n"})
    with pytest.raises(
        SnapshotMismatchError, match=r"files differ from their blobs \(1 total\)"
    ):
        export_snapshot(repo.git, commit, tmp_path / "snapshot")


def test_missing_gold_paths(history):
    repo, base, _ = history
    assert missing_gold_paths(repo.git, base, ["src/parse.c", "link", "src"]) == ["src"]
    assert missing_gold_paths(repo.git, base, ["gone.c"]) == ["gone.c"]


def test_commit_cache_fetches_each_commit_once(history, tmp_path):
    repo, base, fix = history
    cache = CommitCache(tmp_path / "cache")
    url = f"file://{repo.git.repo}"
    cached = cache.fetch("toy/upstream", url, base)
    assert cached.repo == tmp_path / "cache/toy/upstream.git"
    assert tree_blobs(cached, base) == tree_blobs(repo.git, base)
    # Depth 1: the base commit's parent is not fetched.
    assert cached.text("rev-list", base).split() == [base]
    assert cached.resolve(fix) is None

    # Fetching a cached commit again needs no remote at all.
    assert cache.fetch("toy/upstream", "file:///nonexistent", base).resolve(base)
    cache.fetch("toy/upstream", url, fix)
    assert cached.text("for-each-ref", "--format=%(refname)").split() == sorted(
        [f"refs/commits/{base}", f"refs/commits/{fix}"]
    )


def test_commit_cache_errors(history, tmp_path):
    repo, _, _ = history
    cache = CommitCache(tmp_path / "cache")
    with pytest.raises(ValueError, match="not a normalized path"):
        cache.fetch("../escape", f"file://{repo.git.repo}", "0" * 40)
    with pytest.raises(GitError):
        cache.fetch("toy/upstream", f"file://{repo.git.repo}", "0" * 40)


def test_snapshot_export_scales_to_fifty_thousand_files(tmp_path, monkeypatch):
    """An export uses a fixed number of git calls, not one call per blob."""
    source = Git(tmp_path / "source")
    source.repo.mkdir()
    source.run("init", "--quiet")
    for directory in range(50):
        group = source.repo / f"group{directory:02d}"
        group.mkdir()
        for number in range(1000):
            (group / f"file{number:04d}.txt").write_bytes(b"example\n")
    source.run("add", "--all")
    commit = (
        source.run(
            "commit-tree",
            source.text("write-tree").strip(),
            "-m",
            "base",
            env=_IDENTITY
            | {
                "GIT_AUTHOR_DATE": "2026-09-23T00:00:00Z",
                "GIT_COMMITTER_DATE": "2026-09-23T00:00:00Z",
            },
        )
        .decode()
        .strip()
    )
    calls = []
    original = Git.run

    def count_calls(self, *args, **kwargs):
        calls.append(args[0])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Git, "run", count_calls)
    destination = tmp_path / "export"
    export_snapshot(source, commit, destination)
    assert len(calls) < 20
    assert len(list(destination.glob("group*/*.txt"))) == 50_000
    assert (destination / "group49/file0999.txt").read_bytes() == b"example\n"

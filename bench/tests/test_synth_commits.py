# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the synthetic commits: determinism, diff sizes, churn shape."""

import json
import os
import random
import subprocess
from pathlib import Path

from bench import synth_commits as sc

FIXED = {
    "GIT_AUTHOR_DATE": "2025-01-01T00:00:00+0000",
    "GIT_COMMITTER_DATE": "2025-01-01T00:00:00+0000",
    "PATH": os.environ["PATH"],
    "HOME": "/nonexistent",
}


def run(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=FIXED,
    ).stdout.strip()


def make_repo(root: Path, name: str = "repo", files: int = 40) -> tuple[Path, Path]:
    repo = root / name
    repo.mkdir()
    run(repo, "init", "--quiet", "--initial-branch=main")
    run(repo, "config", "user.name", "t")
    run(repo, "config", "user.email", "t@localhost")
    names = []
    for i in range(files):
        sub = repo / f"d{i % 4}"
        sub.mkdir(exist_ok=True)
        text = "".join(f"int value_{i}_{j} = {j * 7 + i};\n" for j in range(30))
        (sub / f"f{i}.c").write_text(text)
        names.append(f"d{i % 4}/f{i}.c")
    run(repo, "add", "-A")
    run(repo, "commit", "--quiet", "-m", "base")
    listing = root / f"{name}.files"
    listing.write_text("\n".join(names) + "\n")
    return repo, listing


def changed(repo: Path, a: str, b: str) -> list[str]:
    return sorted(run(repo, "diff", "--name-only", a, b).splitlines())


def test_edit_always_changes_and_is_seeded():
    text = "int a = 1;\nint b = 2;\nint c = 3;\nint d = 4;\n"
    for seed in range(50):
        one = sc.edit_file("x.c", text, random.Random(seed), "tag")
        two = sc.edit_file("x.c", text, random.Random(seed), "tag")
        assert one == two
        assert one != text
        assert sc.PLACEHOLDER not in one


def test_edit_without_comment_syntax_keeps_original_lines():
    text = "alpha\nbeta\ngamma\ndelta\n"
    for seed in range(50):
        edited = sc.edit_file("data.json", text, random.Random(seed), "tag")
        assert edited != text
        assert sc.PLACEHOLDER not in edited


def test_chain_is_deterministic_and_diffs_are_exact(tmp_path):
    results = []
    for name in ("a", "b"):
        repo, listing = make_repo(tmp_path, name)
        result = sc.make_chain(repo, sc.read_list(listing), 7, steps=3, per_step=4)
        results.append((repo, result))
    (repo_a, a), (_, b) = results
    assert [c["commit"] for c in a["commits"]] == [c["commit"] for c in b["commits"]]
    commits = [c["commit"] for c in a["commits"]]
    assert len(commits) == 4
    # Worktree n differs from worktree 1 by (n - 1) slices, from n - 1 by one.
    for n in range(1, 4):
        assert len(changed(repo_a, commits[0], commits[n])) == 4 * n
        assert len(changed(repo_a, commits[n - 1], commits[n])) == 4
    # The slices are disjoint.
    first = set(changed(repo_a, commits[0], commits[1]))
    second = set(changed(repo_a, commits[1], commits[2]))
    assert not first & second
    # The worktree is back at the base.
    assert run(repo_a, "rev-parse", "HEAD") == commits[0]
    assert run(repo_a, "status", "--porcelain") == ""
    assert run(repo_a, "rev-parse", "m3/k3") == commits[2]


def test_other_seed_changes_the_commits(tmp_path):
    repo_a, list_a = make_repo(tmp_path, "a")
    repo_b, list_b = make_repo(tmp_path, "b")
    a = sc.make_chain(repo_a, sc.read_list(list_a), 1, steps=1, per_step=5)
    b = sc.make_chain(repo_b, sc.read_list(list_b), 2, steps=1, per_step=5)
    assert a["commits"][1]["commit"] != b["commits"][1]["commit"]


def test_chain_skips_files_that_cannot_be_edited(tmp_path):
    repo, listing = make_repo(tmp_path, files=12)
    (repo / "d0" / "f0.c").write_bytes(b"\0binary\0")
    (repo / "d1" / "f1.c").write_bytes(b"")
    run(repo, "commit", "--quiet", "-am", "binary")
    files = sc.read_list(listing)
    picked = sc.select_editable(repo, files, 1, 10)
    assert {p for p, _ in picked}.isdisjoint({"d0/f0.c", "d1/f1.c"})


def test_not_enough_files_is_an_error(tmp_path):
    repo, listing = make_repo(tmp_path, files=5)
    try:
        sc.make_chain(repo, sc.read_list(listing), 1, steps=3, per_step=3)
    except SystemExit as error:
        assert "needed" in str(error)
    else:
        raise AssertionError("expected SystemExit")


def test_churn_commit_shape(tmp_path):
    repo, listing = make_repo(tmp_path, files=40)
    result = sc.make_churn(repo, sc.read_list(listing), 3, changed=20)
    assert (result["edited"], result["deleted"], result["added"]) == (16, 2, 3)
    status = run(repo, "diff", "--name-status", result["base"], result["commit"])
    codes = [line.split("\t")[0] for line in status.splitlines()]
    assert codes.count("M") == 16
    assert codes.count("D") == 2
    assert codes.count("A") == 3  # two copies and the marker
    marker = run(repo, "show", f"{result['commit']}:{result['marker_path']}")
    assert marker == result["marker"]
    assert run(repo, "rev-parse", "HEAD") == result["base"]
    again, listing2 = make_repo(tmp_path, "again", files=40)
    other = sc.make_churn(again, sc.read_list(listing2), 3, changed=20)
    assert other["commit"] == result["commit"]


def test_cli_writes_json(tmp_path):
    repo, listing = make_repo(tmp_path)
    out = tmp_path / "chain.json"
    code = sc.main(
        ["chain", str(repo), "--files", str(listing), "--steps", "2",
         "--diff-fraction", "0.1", "--out", str(out)]
    )  # fmt: skip
    assert code == 0
    data = json.loads(out.read_text())
    assert data["files_per_step"] == 4
    assert [c["k"] for c in data["commits"]] == [1, 2, 3]

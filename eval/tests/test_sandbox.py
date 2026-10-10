# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for harness.sandbox: confinement, paging, listing, and finding."""

import re
from pathlib import Path

import pytest

from harness.sandbox import RepoSandbox, SandboxError
from harness.tools import LIST_MAX_ENTRIES, READ_PAGE_BYTES, READ_PAGE_LINES


@pytest.fixture
def sandbox(toy_repo: Path) -> RepoSandbox:
    return RepoSandbox(toy_repo)


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../outside/secret.txt",
        "src/../../outside/secret.txt",
        ".git/config",
        ".GIT/config",
        "vendor/.git/HEAD",
        "link_git/config",
        "link_out",
        "link_out_dir/secret.txt",
    ],
)
def test_refuses_escapes_and_git_metadata(sandbox, path):
    with pytest.raises(SandboxError) as error:
        sandbox.read_page(path, 1)
    assert error.value.rejected
    assert "SECRET" not in error.value.message
    assert sandbox.root not in error.value.message


def test_follows_symlinks_inside_the_root(sandbox):
    page = sandbox.read_page("link_ok", 1)
    assert "def main():" in page
    assert page.endswith("[src/app.py: lines 1-5 of 5]")
    assert "def helper():" in sandbox.read_page("link_dir/util.py", 1)


def test_read_numbers_lines(sandbox):
    assert sandbox.read_page("src/util.py", 1) == (
        "     1\tdef helper():\n     2\t    return 42\n[src/util.py: lines 1-2 of 2]"
    )
    assert sandbox.read_page("./src//util.py", 1).endswith(
        "[src/util.py: lines 1-2 of 2]"
    )


def test_read_pages_through_long_files(sandbox):
    first = sandbox.read_page("big.txt", 1)
    assert first.endswith(
        f"[big.txt: lines 1-{READ_PAGE_LINES} of 450; next start_line=201]"
    )
    last = sandbox.read_page("big.txt", 401)
    assert last.startswith("   401\tline 401\n")
    assert last.endswith("[big.txt: lines 401-450 of 450]")


def test_read_truncates_a_line_wider_than_a_page(sandbox, toy_repo):
    page = sandbox.read_page("wide.txt", 1)
    assert "[line truncated]" in page
    assert len(page.encode()) <= READ_PAGE_BYTES
    # The footer's path counts against the page, however long it is.
    deep = "/".join(["directory_name_of_some_length"] * 20) + "/wide.txt"
    (toy_repo / deep).parent.mkdir(parents=True)
    (toy_repo / deep).write_text("y" * 40_000 + "\n")
    page = sandbox.read_page(deep, 1)
    assert page.endswith(f"[{deep}: lines 1-1 of 1]")
    assert len(page.encode()) <= READ_PAGE_BYTES


@pytest.mark.parametrize(
    ("path", "start_line", "message"),
    [
        ("missing.py", 1, "no such file or directory: missing.py"),
        ("link_dangling", 1, "no such file or directory"),
        ("src", 1, "is a directory: src"),
        ("fifo", 1, "not a regular file: fifo"),
        ("blob.bin", 1, "binary file: blob.bin"),
        ("src/util.py", 3, "start_line 3 is past the end of src/util.py (2 lines)"),
    ],
)
def test_read_failures_are_ordinary_errors(sandbox, path, start_line, message):
    with pytest.raises(SandboxError, match=re.escape(message)) as error:
        sandbox.read_page(path, start_line)
    assert not error.value.rejected


def test_read_empty_file(sandbox, toy_repo):
    (toy_repo / "empty.txt").write_text("")
    assert sandbox.read_page("empty.txt", 1) == "[empty.txt: empty file]"


def test_list_dir_shows_only_openable_entries(sandbox):
    assert sandbox.list_dir(".").splitlines() == [
        "README.md",
        "big.txt",
        "blob.bin",
        "lib/",
        "link_dir/",
        "link_ok",
        "src/",
        "vendor/",
        "wide.txt",
    ]
    assert sandbox.list_dir("vendor").splitlines() == ["lib.c"]
    with pytest.raises(SandboxError, match=re.escape("not a directory: README.md")):
        sandbox.list_dir("README.md")


def test_find_matches_names_or_whole_paths(sandbox):
    assert sandbox.find("*.c").splitlines() == ["lib/a.c", "lib/b.c", "vendor/lib.c"]
    assert sandbox.find("src/*.py").splitlines() == ["src/app.py", "src/util.py"]
    assert sandbox.find("**/util.py").splitlines() == ["src/util.py"]
    assert sandbox.find("*.rs") == "[no matching files]"


def test_find_caps_its_output(sandbox, toy_repo):
    many = toy_repo / "many"
    many.mkdir()
    for i in range(LIST_MAX_ENTRIES + 5):
        (many / f"f{i:04d}.txt").write_text("x\n")
    lines = sandbox.find("many/*.txt").splitlines()
    assert len(lines) == LIST_MAX_ENTRIES + 1
    assert (
        lines[-1] == f"[showing {LIST_MAX_ENTRIES} of {LIST_MAX_ENTRIES + 5} matches]"
    )


def test_rejects_a_root_that_is_not_a_directory(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        RepoSandbox(tmp_path / "missing")


def test_find_gitignore_directory_match_includes_descendants(tmp_path):
    from harness.sandbox import RepoSandbox

    root = tmp_path / "repo"
    (root / "Documentation/networking/devices").mkdir(parents=True)
    (root / "Documentation/networking/direct.txt").write_text("direct")
    (root / "Documentation/networking/devices/efa.txt").write_text("nested")
    sandbox = RepoSandbox(root)
    assert sandbox.find("Documentation/networking/*").splitlines() == [
        "Documentation/networking/devices/efa.txt",
        "Documentation/networking/direct.txt",
    ]


def test_find_prefilter_and_memo_keep_full_match_results(tmp_path):
    import random

    from harness.sandbox import _capped, _display, _matches_glob

    rng = random.Random(7)
    names = ["a", "ab", "b.c", "Bc", "x.py", "net", "dev", "a.b.c"]
    root = tmp_path / "repo"
    for _ in range(200):
        parts = [rng.choice(names) for _ in range(rng.randint(1, 4))]
        try:
            root.joinpath(*parts[:-1]).mkdir(parents=True, exist_ok=True)
            root.joinpath(*parts).write_text("x\n")
        except (FileExistsError, IsADirectoryError, NotADirectoryError):
            continue
    sandbox = RepoSandbox(root)
    files = sandbox._file_list()
    atoms = ["*", "**", "?", "a", "b", ".c", "c", "net", "[ab]", "x", ".", "./"]
    for _ in range(400):
        glob = "".join(rng.choice(atoms) for _ in range(rng.randint(1, 4)))
        if rng.random() < 0.5:
            glob = "/".join([glob, rng.choice(atoms)])
        whole = "/" in glob
        # The matcher before the prefilter and memo, applied to every file.
        old = [_display(path) for path in files if _matches_glob(path, glob, whole)]
        expected = _capped(old, "matches") if old else "[no matching files]"
        assert sandbox.find(glob) == expected, glob

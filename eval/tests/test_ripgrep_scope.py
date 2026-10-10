# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""ripgrep, given `ripgrep_globs`, searches exactly the files Ferret's
`path_glob` admits.

Differential test on a generated tree with the real `ripgrep` and
`ferret_bench files --path_glob` (which applies `ferret::PathGlob`). Needs
`rg` and `ferret_bench` (FERRET_BENCH_BINARY, or a Bazel output); skipped
otherwise.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from harness.replay import ripgrep_argv, ripgrep_globs
from tests.test_replay_engines import _FERRET_BENCH

_FILES = [
    "a.c", "b.h", "README", "build", ".hidden", ".github/workflows/ci.yml",
    "src/a.c", "src/b.c", "src/deep/x.h/y.c", "src/deep/z.c", "src/build/out.o",
    "src/testdata/u.txt", "testdata/t.txt", "docs/testdata", "docs/guide.md",
    "docs/sub/guide.md", "docs/sub/deeper/note.md", "fs/btrfs/ctree.c",
    "fs/btrfs/sub/inode.c", "fs/ext4/inode.c", "foo.c/bar.txt", "foo.c/sub/baz.c",
    "tools/gen.c", "tools/.dot/x.c", "BUILD", "lib/BUILD", "x.h/top.c",
]  # fmt: skip

# Every rule of ferret/git/path_glob.h.
_GLOBS = [
    "*.c", "*.h", "BUILD", "build", "testdata", "/testdata", "testdata/",
    "/docs", "docs/", "docs", "docs/*", "docs/**", "docs/*/", "fs/btrfs/*",
    "fs/btrfs", "fs/*/inode.c", "src/**/*.c", "src/**", "**/*", "**", "?.c",
    "*.?", "/*.c", "x.h/", "*.h/", "src/deep/x.h", "**/deep", "/src/deep/",
    "*.md", ".hidden", ".github", "deep/**/y.c", "**/testdata", "foo.c",
    "!*.c", "!testdata", "!/docs", "!docs/*", "!fs/btrfs/", "!**/testdata/**",
    "!*", "!x.h/", "!src/**", "!/*.c", "!build",
]  # fmt: skip


@pytest.fixture(scope="module")
def tree(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("scope")
    for name in _FILES:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(f"needle in {name}\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    return root


def _tools():
    rg = shutil.which("rg")
    if rg is None:
        pytest.skip("ripgrep is not installed")
    if not _FERRET_BENCH.exists():
        pytest.skip(f"{_FERRET_BENCH} is missing; build it with bazel first")
    return rg, _FERRET_BENCH


def _ferret_files(bench: Path, tree: Path, glob: str, out: Path) -> set[str]:
    subprocess.run(
        [str(bench), "files", f"--root={tree}", f"--out={out}", f"--path_glob={glob}"],
        check=True,
        capture_output=True,
    )
    return set(out.read_text().split("\n")) - {""}


def _rg_files(rg: str, tree: Path, glob: str) -> set[str]:
    argv = ripgrep_argv(Path(rg), {"pattern": "needle", "path_glob": glob})
    # The same flags, listing files instead of searching.
    argv = [a for a in argv[:-2] if a not in ("--line-number", "--with-filename")]
    completed = subprocess.run(
        [*argv, "--files"], cwd=tree, capture_output=True, text=True, check=False
    )
    assert completed.returncode in (0, 1), completed.stderr
    return set(completed.stdout.split("\n")) - {""}


@pytest.mark.parametrize("glob", _GLOBS)
def test_ripgrep_with_translated_globs_admits_ferrets_files(tree, tmp_path, glob):
    rg, bench = _tools()
    expected = _ferret_files(bench, tree, glob, tmp_path / "list.txt")
    assert _rg_files(rg, tree, glob) == expected, ripgrep_globs(glob)


def test_the_globs_cover_both_empty_and_nonempty_scopes(tree, tmp_path):
    _, bench = _tools()
    sizes = {len(_ferret_files(bench, tree, g, tmp_path / "l.txt")) for g in _GLOBS}
    assert 0 in sizes or min(sizes) < 3
    assert max(sizes) >= len(_FILES) - 3


def test_translation_forms():
    assert ripgrep_globs("testdata") == ["testdata", "**/testdata/**"]
    assert ripgrep_globs("testdata/") == ["**/testdata/**"]
    assert ripgrep_globs("fs/btrfs/*") == ["fs/btrfs/*", "fs/btrfs/*/**"]
    assert ripgrep_globs("/docs") == ["/docs", "/docs/**"]
    assert ripgrep_globs("!*.h") == ["!*.h", "!**/*.h/**"]

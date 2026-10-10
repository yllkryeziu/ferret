# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for harness.search: the ferret CLI search backend."""

import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from harness.arms import Shape
from harness.search import (
    NO_MATCHES,
    FerretCliSearch,
    SearchResult,
    SearchTimeoutError,
)

# Behaves like `ferret search`, keyed on the pattern (the last argument).
_FAKE_FERRET = """\
import os, sys, time
pattern = sys.argv[-1]
if pattern == "none":
    print("0 matching lines in 0 files"); sys.exit(1)
if pattern == "silent":
    sys.exit(1)
if pattern == "bad(":
    sys.stderr.write("ferret: missing ): bad(\\n"); sys.exit(2)
if pattern == "slow":
    time.sleep(5)
if pattern == "crash":
    sys.exit(3)
if pattern == "env":
    print("AZURE_API_KEY" in os.environ); sys.exit(0)
print(" ".join(sys.argv[1:]))
"""


@pytest.fixture
def fake_ferret(tmp_path: Path) -> Path:
    path = tmp_path / "ferret"
    path.write_text(f"#!{sys.executable}\n{_FAKE_FERRET}")
    path.chmod(0o755)
    return path


def _search(binary: Path, **kwargs) -> FerretCliSearch:
    return FerretCliSearch(
        binary=binary, index=Path("/idx.fseg"), shape=Shape.CONTEXT, **kwargs
    )


def test_builds_the_command_line(fake_ferret):
    search = _search(fake_ferret)
    assert search.argv(
        {"pattern": "-x", "regex": True, "case_sensitive": True, "path_glob": "*.py"}
    ) == [
        str(fake_ferret),
        "search",
        "--index=/idx.fseg",
        "--shape=context",
        "--regex",
        "--case_sensitive",
        "--path=*.py",
        "--",
        "-x",
    ]
    assert search.argv({"pattern": "x", "regex": False, "case_sensitive": False})[
        -3:
    ] == ["--shape=context", "--", "x"]


def test_maps_exit_statuses(fake_ferret):
    search = _search(fake_ferret)
    assert search.search({"pattern": "hello"}) == SearchResult(
        "search --index=/idx.fseg --shape=context -- hello\n", is_error=False
    )
    assert search.search({"pattern": "none"}) == SearchResult(
        "0 matching lines in 0 files\n", is_error=False
    )
    assert search.search({"pattern": "silent"}) == SearchResult(
        NO_MATCHES, is_error=False
    )
    assert search.search({"pattern": "bad("}) == SearchResult(
        "error: missing ): bad(", is_error=True
    )
    with pytest.raises(RuntimeError, match="status 3"):
        search.search({"pattern": "crash"})


def test_times_out(fake_ferret):
    with pytest.raises(SearchTimeoutError, match=re.escape("timed out after 0.2 s")):
        _search(fake_ferret, timeout_s=0.2).search({"pattern": "slow"})


def test_keeps_credentials_from_the_child(fake_ferret, monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "secret")
    assert _search(fake_ferret).search({"pattern": "env"}).text == "False\n"


def test_describe_has_no_secrets(fake_ferret):
    assert _search(fake_ferret).describe() == {
        "kind": "ferret_cli",
        "binary": str(fake_ferret),
        "index": "/idx.fseg",
        "shape": "context",
        "timeout_s": 60.0,
    }


def _ferret_binary() -> Path | None:
    configured = os.environ.get("FERRET_BINARY")
    if configured:
        return Path(configured)
    built = Path(__file__).resolve().parents[2] / "bazel-bin/ferret/cli/ferret"
    return built if built.exists() else None


@pytest.mark.skipif(_ferret_binary() is None, reason="ferret is not built")
def test_searches_a_real_index(tmp_path):
    binary = _ferret_binary()
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src/app.py").write_text("def connect():\n    return 1\n")
    git = shutil.which("git")
    subprocess.run([git, "init", "-q", str(repo)], check=True)
    subprocess.run([git, "-C", str(repo), "add", "-A"], check=True)
    index = tmp_path / "index.fseg"
    subprocess.run(
        [str(binary), "index", f"--index={index}", str(repo)],
        check=True,
        capture_output=True,
    )
    result = FerretCliSearch(binary=binary, index=index, shape=Shape.LINE).search(
        {"pattern": "connect"}
    )
    assert result == SearchResult("src/app.py:1:def connect():\n", is_error=False)
    bad = FerretCliSearch(binary=binary, index=index, shape=Shape.LINE).search(
        {"pattern": "(", "regex": True}
    )
    assert bad.is_error
    assert bad.text.startswith("error: ")
    assert textwrap.shorten(bad.text, 200) == bad.text

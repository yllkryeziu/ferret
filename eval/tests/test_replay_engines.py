# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""The replay's three engines agree on a generated git repository.

Uses the real `ferret`, `ferret_bench`, and `zoekt-bench` binaries. Build the
first two with `bazel build -c opt //ferret/cli:ferret
//bench/ferret:ferret_bench`, or point FERRET_BINARY and FERRET_BENCH_BINARY at
them; `zoekt-bench` is built here with Go. The test is skipped when a binary,
Go, or ripgrep is missing.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from harness.arms import Shape
from harness.replay import SearchCall, ZoektConfig, replay
from tasks.prepare import index_path, snapshot_dir

_ROOT = Path(__file__).resolve().parents[2]


def _bazel_binary(variable: str, target_path: str) -> Path:
    """Returns the binary named by `variable`, else the one Bazel built."""
    if variable in os.environ:
        return Path(os.environ[variable])
    for output in ("bazel-bin", "bazel-out/darwin_arm64-opt/bin"):
        if (_ROOT / output / target_path).exists():
            return _ROOT / output / target_path
    return _ROOT / "bazel-bin" / target_path


_FERRET = _bazel_binary("FERRET_BINARY", "ferret/cli/ferret")
_FERRET_BENCH = _bazel_binary("FERRET_BENCH_BINARY", "bench/ferret/ferret_bench")
_SNAPSHOT = "c" * 40
_SUBDIRECTORY_GLOB = "Documentation/networking/*"
_BULK_LINES = 10_500
# Go's regexp/syntax rejects an expression nested this deeply; RE2 does not.
_GO_REJECTED = "(" * 1001 + "needle" + ")" * 1001

_FILES = {
    "Documentation/networking/index.rst": "Needle in the index\nnothing\n",
    "Documentation/networking/dsa/dsa.rst": "a needle in a subdirectory\n",
    "Documentation/admin/guide.rst": "needle outside the scope\n",
    "src/main.c": "int main(void) {\n  return needle(1) + Needle(2);\n}\n",
    "src/util.h": "int needle(int x);\nint other(void);\n",
    "tools/gen.c": "int needle_count = 3;\nint needle_total(int x);\n",
    # More matching lines than Ferret's default cap of 10,000.
    "data/bulk.txt": "zebra\n" * _BULK_LINES,
}

# (pattern, arguments); a mix of scoped and unscoped literal and regex calls.
_CALLS = [
    {"pattern": "needle"},
    {"pattern": "Needle", "case_sensitive": True},
    {"pattern": r"needle\(\d\)", "regex": True},
    {"pattern": r"^int \w+", "regex": True},
    {"pattern": "needle", "path_glob": _SUBDIRECTORY_GLOB},
    {"pattern": "needle", "path_glob": "*.c"},
    {"pattern": r"needle_\w+", "regex": True, "path_glob": "tools/**"},
    {"pattern": "needle", "path_glob": "!*.rst"},
    {"pattern": "no such text", "path_glob": "src/*"},
    {"pattern": "zebra"},
    {"pattern": "zebra", "path_glob": "data/*"},
    {"pattern": _GO_REJECTED, "regex": True, "path_glob": "src/*"},
    {"pattern": "needle", "path_glob": "src/*"},
]


@pytest.fixture(scope="module")
def zoekt_bench(tmp_path_factory) -> Path:
    if shutil.which("go") is None:
        pytest.skip("Go is not installed")
    binary = tmp_path_factory.mktemp("zoekt") / "zoekt-bench"
    subprocess.run(
        ["go", "build", "-o", str(binary), "."],
        cwd=_ROOT / "bench/zoekt",
        check=True,
        capture_output=True,
    )
    return binary


def test_three_engines_return_the_same_match_sets(tmp_path, zoekt_bench):
    rg = shutil.which("rg")
    if rg is None:
        pytest.skip("ripgrep is not installed")
    for binary in (_FERRET, _FERRET_BENCH):
        if not binary.exists():
            pytest.skip(f"{binary} is missing; build it with bazel first")
    data = tmp_path / "data"
    snapshot = snapshot_dir(data, _SNAPSHOT)
    for path, content in _FILES.items():
        (snapshot / path).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / path).write_text(content)
    subprocess.run(["git", "init", "-q", str(snapshot)], check=True)
    subprocess.run(["git", "-C", str(snapshot), "add", "."], check=True)
    index = index_path(data, _SNAPSHOT)
    index.parent.mkdir(parents=True)
    subprocess.run(
        [str(_FERRET), "index", f"--root={snapshot}", f"--index={index}"],
        check=True,
        capture_output=True,
    )
    calls = [
        SearchCall("e", "A1", _SNAPSHOT, Shape.RAW, arguments, 0.0)
        for arguments in _CALLS
    ]
    zoekt = ZoektConfig(zoekt_bench, _FERRET_BENCH, tmp_path / "zoekt-work")

    rows = replay(calls, data, _FERRET, Path(rg), repeats=1, zoekt=zoekt)

    assert len(rows) == len(_CALLS)
    by_pattern = {(row["pattern"], row["path_glob"]): row for row in rows}
    rejected = by_pattern[(_GO_REJECTED, "src/*")]
    # The query Go rejects is a status of its own; the others still ran.
    assert rejected["zoekt_status"].startswith("error: "), rejected["zoekt_status"]
    assert rejected["zoekt_ms"] is None
    assert rejected["ferret_matches"] == 2
    assert len(rows) == len(_CALLS)
    rows = [row for row in rows if row is not rejected]
    for row in rows:
        context = f"{row['pattern']!r} scoped to {row['path_glob']!r}"
        assert row["zoekt_status"] == "ok", context
        assert not row["zoekt_truncated"], context
        assert row["zoekt_only_vs_ferret"] == row["ferret_only_vs_zoekt"] == 0, context
        assert row["zoekt_matches"] == row["ferret_matches"], context
        assert row["only_ferret"] == row["only_rg"] == 0, context
        assert row["zoekt_only_vs_rg"] == row["rg_only_vs_zoekt"] == 0, context
    # Neither engine stops at Ferret's default cap of 10,000 lines.
    for key in (("zebra", ""), ("zebra", "data/*")):
        row = by_pattern[key]
        assert row["ferret_matches"] == row["rg_matches"] == _BULK_LINES
        assert row["zoekt_matches"] == _BULK_LINES
        assert row["only_ferret"] == row["only_rg"] == 0
    matches = [row["zoekt_matches"] for row in rows]
    # Scoping matters: a scope that admits fewer files finds fewer matches
    # than the unscoped call, and the empty scope finds none.
    assert matches[:9] == [7, 2, 1, 5, 2, 3, 2, 4, 0]

    # ripgrep's own --glob=dir/* would not admit files in subdirectories of
    # dir, which Ferret's path_glob does; with the translated globs
    # (`ripgrep_globs`) all three engines agree on it, too.
    (subdirectory,) = [row for row in rows if row["path_glob"] == _SUBDIRECTORY_GLOB]
    assert subdirectory["only_ferret"] == subdirectory["only_rg"] == 0
    assert subdirectory["zoekt_only_vs_rg"] == subdirectory["rg_only_vs_zoekt"] == 0
    assert subdirectory["ferret_matches"] == 2

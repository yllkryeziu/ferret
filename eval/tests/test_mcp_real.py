# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""MCP and CLI search equivalence on the same generated repository."""

import subprocess
from pathlib import Path

import pytest

from harness.mcp_client import McpClient
from harness.mcp_servers import ManagedServers
from harness.search import FerretCliSearch


@pytest.mark.parametrize("shape", ["raw", "line", "context"])
def test_real_mcp_matches_cli(tmp_path, shape):
    binary = Path(__file__).resolve().parents[2] / "bazel-bin/ferret/cli/ferret"
    if not binary.exists():
        pytest.skip("build //ferret/cli:ferret first")
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src/a.txt").write_text("alpha needle\nbeta\n")
    (root / "src/b.txt").write_text("needle\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "src"], check=True)
    index = tmp_path / "cli.fseg"
    subprocess.run(
        [str(binary), "index", f"--root={root}", f"--index={index}"],
        check=True,
        capture_output=True,
    )
    args = {"pattern": "needle", "path_glob": "src/*"}
    with (
        ManagedServers(binary, [root], tmp_path / "daemon"),
        McpClient(binary, root, tmp_path / "daemon", shape, fixed_shape=False) as mcp,
    ):
        actual = mcp.search(args)
        expected = FerretCliSearch(binary=binary, index=index, shape=shape).search(args)
        assert actual == expected


def test_real_server_exposes_shape_in_schema_and_reaps_children(tmp_path):
    binary = Path(__file__).resolve().parents[2] / "bazel-bin/ferret/cli/ferret"
    if not binary.exists():
        pytest.skip("build //ferret/cli:ferret first")
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "file.txt").write_text("needle\n")
    with ManagedServers(binary, [root], tmp_path / "daemon") as servers:
        children = [child.pid for child in servers._children]
        with (
            McpClient(
                binary, root, tmp_path / "daemon", "raw", fixed_shape=False
            ) as raw,
            McpClient(
                binary, root, tmp_path / "daemon", "line", fixed_shape=False
            ) as line,
        ):
            assert raw.schema_sha256 != line.schema_sha256
    import os

    for pid in children:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_real_fixed_schema_preflight_and_run(tmp_path):
    from harness.episode import Task
    from harness.run import ExperimentConfig, run_experiment
    from models.scripted import ScriptedCall, ScriptedModelFactory

    binary = Path(__file__).resolve().parents[2] / "bazel-bin/ferret/cli/ferret"
    if not binary.exists():
        pytest.skip("build //ferret/cli:ferret first")
    flags = subprocess.run(
        [str(binary), "mcp", "--helpfull"], capture_output=True, text=True
    )
    if not all(
        flag in flags.stdout + flags.stderr
        for flag in ("--fixed_shape", "--spawn_daemon")
    ):
        pytest.skip("real binary lacks --fixed_shape or --spawn_daemon")
    data = tmp_path / "data"
    root = data / "snapshots/s"
    root.mkdir(parents=True)
    (root / "source.txt").write_text("needle\n")
    factory = ScriptedModelFactory(
        scripts={"default": [[ScriptedCall("submit", {"files": ["source.txt"]})]]},
        source_sha256="fake",
    )
    config = ExperimentConfig(
        arms=("A1", "A2"),
        samples=1,
        seed=1,
        data_dir=data,
        ferret=binary,
        search_backend="mcp",
    )
    out = tmp_path / "run"
    report = run_experiment(
        config,
        [Task(task_id="t", prompt="needle", snapshot="s")],
        factory,
        out,
        workers=2,
    )
    assert report.failed == 0
    assert len(list((out / "episodes").glob("*.jsonl"))) == 2
    with (
        ManagedServers(binary, [root], tmp_path / "schema-index"),
        McpClient(binary, root, tmp_path / "schema-index", "raw") as raw,
        McpClient(binary, root, tmp_path / "schema-index", "line") as line,
    ):
        assert "shape" not in raw.tool.parameters["properties"]
        assert raw.schema_sha256 == line.schema_sha256


def test_real_warm_up_waits_for_the_index_and_symbols(tmp_path):
    binary = Path(__file__).resolve().parents[2] / "bazel-bin/ferret/cli/ferret"
    if not binary.exists():
        pytest.skip("build //ferret/cli:ferret first")
    root = tmp_path / "snapshot"
    (root / "src").mkdir(parents=True)
    for number in range(40):
        (root / f"src/f{number}.c").write_text(
            f"int handler_{number}(int x) {{ return x + {number}; }}\n"
        )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "src"], check=True)
    with ManagedServers(binary, [root], tmp_path / "daemon") as servers:
        assert "--idle_timeout=inf" in servers._children[0].args
        servers.warm_up(timeout_s=120)
        with McpClient(binary, root, tmp_path / "daemon", "context") as client:
            (status,) = client.index_status()
    assert status["file_count"] == 40
    assert status["queued_paths"] == 0
    assert status["pending_symbols"] == 0

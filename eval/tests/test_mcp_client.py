# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""MCP lifecycle, both protocol eras, tool filtering, errors and crashes."""

import os
import stat
import sys
from pathlib import Path

import pytest

from harness.mcp_client import (
    McpClient,
    McpError,
    canary_arguments,
    canary_patterns,
    check_search_canary,
)


def _fake(tmp_path: Path, mode: str) -> tuple[Path, Path]:
    root = tmp_path / "root"
    root.mkdir()
    (root / "mode").write_text(mode)
    binary = tmp_path / "ferret"
    binary.write_text(
        f"#!{sys.executable}\n"
        + (Path(__file__).parent / "fakes/fake_mcp_server.py").read_text()
    )
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return binary, root


@pytest.mark.parametrize("mode", ["legacy", "modern"])
def test_mcp_protocol_eras_and_errors(tmp_path, mode):
    binary, root = _fake(tmp_path, mode)
    for shape in ("raw", "context"):
        with McpClient(binary, root, tmp_path, shape) as client:
            pid = client._process.pid
            assert client.tool.name == "search"
            assert client.tool.description == "fixed"
            assert client.search({"pattern": "needle"}).text == f"{shape}:needle"
            assert client.search({"pattern": "error"}).text == "error: bad regex"
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_mcp_server_crash_is_not_silenced_and_next_connection_works(tmp_path):
    binary, root = _fake(tmp_path, "modern")
    with McpClient(binary, root, tmp_path, "line") as client:
        pid = client._process.pid
        with pytest.raises(McpError, match="exited during tools/call"):
            client.search({"pattern": "crash"})
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    with McpClient(binary, root, tmp_path, "line") as client:
        assert client.search({"pattern": "retry"}).text == "line:retry"


def test_schema_hash_detects_server_drift(tmp_path):
    binary, root = _fake(tmp_path, "different")
    with (
        McpClient(binary, root, tmp_path, "raw") as raw,
        McpClient(binary, root, tmp_path, "line") as line,
    ):
        assert raw.schema_sha256 != line.schema_sha256


def test_fixed_shape_forwards_every_schema_argument_without_overriding_shape(tmp_path):
    from harness.ferret_env import FerretEnvironment, ToolStatus
    from harness.sandbox import RepoSandbox

    binary, root = _fake(tmp_path, "modern")
    with (
        McpClient(binary, root, tmp_path, "raw") as raw,
        McpClient(binary, root, tmp_path, "context") as context,
    ):
        assert raw.schema_sha256 == context.schema_sha256
        assert "shape" not in raw.tool.parameters["properties"]
        env = FerretEnvironment(
            sandbox=RepoSandbox(root),
            tools=(raw.tool,),
            search=raw,
            max_tool_calls=5,
        )
        ok = env.execute(
            {
                "tool": "search",
                "arguments": {
                    "pattern": "echo_args",
                    "regex": False,
                    "case_sensitive": False,
                    "path_glob": "src/*",
                },
            }
        )
        assert ok["extra"]["status"] == ToolStatus.OK.value
        assert ok["output"] == (
            '{"case_sensitive": false, "path_glob": "src/*", '
            '"pattern": "echo_args", "regex": false}'
        )
        rejected = env.execute(
            {"tool": "search", "arguments": {"pattern": "needle", "shape": "line"}}
        )
        assert rejected["extra"]["status"] == ToolStatus.REJECTED.value
        assert "unknown argument 'shape'" in rejected["output"]


def test_daemon_loss_is_fatal_without_respawn(tmp_path):
    binary, root = _fake(tmp_path, "modern")
    with McpClient(binary, root, tmp_path, "line") as client:
        with pytest.raises(McpError, match="failure: daemon unavailable"):
            client.search({"pattern": "daemon_crash"})
        assert (root / "daemon_dead").exists()
        with pytest.raises(McpError, match="failure: daemon unavailable"):
            client.search({"pattern": "retry"})


def test_harness_records_daemon_crash_without_respawn(tmp_path, monkeypatch):
    from harness import run
    from harness.episode import Task
    from harness.run import ExperimentConfig, run_experiment
    from models.scripted import ScriptedCall, ScriptedModelFactory

    binary, root = _fake(tmp_path, "modern")
    data = tmp_path / "data"
    snapshot = data / "snapshots" / "s"
    snapshot.mkdir(parents=True)
    (snapshot / "mode").write_text("modern")

    class ExistingDaemon:
        def __init__(self, *args):
            pass

        def close(self):
            pass

        def warm_up(self):
            pass

        def assert_alive(self, root):
            if (root / "daemon_dead").exists():
                raise RuntimeError("MCP daemon exited")

    monkeypatch.setattr(run, "ManagedServers", ExistingDaemon)
    factory = ScriptedModelFactory(
        scripts={"default": [[ScriptedCall("search", {"pattern": "daemon_crash"})]]},
        source_sha256="script",
    )
    report = run_experiment(
        ExperimentConfig(
            arms=("A1", "A2", "A3", "A4"),
            samples=1,
            seed=1,
            data_dir=data,
            ferret=binary,
            search_backend="mcp",
        ),
        [Task(task_id="t", prompt="needle", snapshot="s")],
        factory,
        tmp_path / "run",
        workers=1,
    )
    assert report.failed == 4
    failures = list((tmp_path / "run/episodes").glob("*.failure.json"))
    assert len(failures) == 4
    assert all("MCP daemon exited" in path.read_text() for path in failures)
    assert (snapshot / "daemon_dead").exists()
    assert not list((tmp_path / "run/episodes").glob("*.jsonl"))
    assert not (root / "daemon_dead").exists()


def test_search_validates_server_constraints_not_only_known_fields():
    from harness.tools import ToolArgumentError, ToolSpec, validate_arguments

    schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "enum": ["allowed"]},
            "options": {
                "type": "object",
                "properties": {"count": {"type": "integer", "minimum": 2}},
                "required": ["count"],
                "additionalProperties": False,
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    spec = ToolSpec("search", "server", schema)
    args = {"pattern": "allowed", "options": {"count": 2}}
    assert validate_arguments(spec, args) == args
    with pytest.raises(ToolArgumentError, match="error: search"):
        validate_arguments(spec, {"pattern": "not allowed"})
    with pytest.raises(ToolArgumentError, match="minimum"):
        validate_arguments(spec, {"pattern": "allowed", "options": {"count": 1}})


def test_preflight_requires_each_serving_flag_before_output(tmp_path):
    from harness.episode import Task
    from harness.run import ExperimentConfig, run_experiment
    from models.scripted import ScriptedModelFactory

    binary, _ = _fake(tmp_path, "modern")
    original = binary.read_text()
    for missing in ("--fixed_shape", "--spawn_daemon", "--router"):
        binary.write_text(
            original.replace(
                'print("--fixed_shape --spawn_daemon --router")',
                'print("'
                + " ".join(
                    flag
                    for flag in ("--fixed_shape", "--spawn_daemon", "--router")
                    if flag != missing
                )
                + '")',
            )
        )
        config = ExperimentConfig(
            arms=("A4",),
            samples=1,
            seed=0,
            data_dir=tmp_path / "missing",
            ferret=binary,
            search_backend="mcp",
        )
        target = tmp_path / missing.lstrip("-")
        with pytest.raises(ValueError, match=missing):
            run_experiment(
                config,
                [Task(task_id="one", prompt="x", snapshot="s")],
                ScriptedModelFactory(scripts={}, source_sha256="script"),
                target,
                workers=1,
            )
        assert not target.exists()


def test_parked_protocol_timeout_and_server_requests(tmp_path):
    binary, root = _fake(tmp_path, "modern")
    with McpClient(binary, root, tmp_path, "raw", timeout_s=0.2) as client:
        assert client.search({"pattern": "__server_request__"}).text == (
            "raw:__server_request__"
        )
        with pytest.raises(McpError, match="timed out"):
            client.search({"pattern": "__hang__"})


def test_fake_router_changes_only_structural_result_not_schema(tmp_path):
    binary, root = _fake(tmp_path, "modern")
    with (
        McpClient(binary, root, tmp_path, "context") as context,
        McpClient(binary, root, tmp_path, "context", router=True) as route,
    ):
        assert route.schema_sha256 == context.schema_sha256
        assert context.search({"pattern": "symbol:foo"}).text == "context:symbol:foo"
        assert route.search({"pattern": "symbol:foo"}).text == "route:symbol:foo"


def test_fixed_shape_rejects_server_schema_with_model_shape(tmp_path):
    binary, root = _fake(tmp_path, "has_shape")
    with pytest.raises(McpError, match="model-supplied shape"):
        McpClient(binary, root, tmp_path, "raw")


def test_mcp_pagination_finds_search_on_later_page(tmp_path):
    binary, root = _fake(tmp_path, "paginated")
    with McpClient(binary, root, tmp_path, "raw") as client:
        assert client.tool.name == "search"
        assert client.search({"pattern": "x"}).text == "raw:x"


def _status(**overrides):
    status = {
        "queued_paths": 0,
        "applied_generation": 3,
        "assigned_generation": 3,
        "pending_symbols": 0,
    }
    return {**status, **overrides}


def test_is_indexed_needs_every_root_applied_and_extracted():
    from harness.mcp_client import is_indexed

    assert is_indexed([_status(), _status()])
    assert not is_indexed([])
    assert not is_indexed([_status(), _status(queued_paths=1)])
    assert not is_indexed([_status(applied_generation=2)])
    assert not is_indexed([_status(pending_symbols=7)])


def test_wait_until_indexed_polls_until_ready_or_times_out():
    from harness.mcp_client import wait_until_indexed

    class Client:
        def __init__(self, statuses):
            self.statuses = list(statuses)
            self.calls = 0

        def index_status(self):
            self.calls += 1
            return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    ready = Client(
        [[_status(pending_symbols=5)], [_status(queued_paths=2)], [_status()]]
    )
    assert wait_until_indexed(ready, 60, 2, lambda: now[0], sleep) == [_status()]
    assert ready.calls == 3

    never = Client([[_status(pending_symbols=1)]])
    now[0] = 0.0
    with pytest.raises(McpError, match="index not ready after 10 s"):
        wait_until_indexed(never, 10, 2, lambda: now[0], sleep)
    assert never.calls == 6


def test_canary_arguments_fill_every_property_like_the_model():
    schema = {
        "properties": {
            "pattern": {"type": "string"},
            "regex": {"type": "boolean", "default": False},
            "path_glob": {"type": "string"},
            "max_files": {"type": "integer", "default": 10, "minimum": 0},
            "worktree": {"type": "string"},
            "flag": {"type": "boolean"},
            "count": {"type": "integer", "minimum": 0},
        }
    }
    blank, typical = canary_arguments(schema, "needle")
    assert blank == {
        "pattern": "needle",
        "regex": False,
        "path_glob": "",
        "max_files": 10,
        "worktree": "",
        "flag": False,
        "count": 1,
    }
    assert typical == {**blank, "path_glob": "**", "max_files": 20, "worktree": "."}


def test_canary_patterns_take_words_from_top_level_text_files(tmp_path):
    (tmp_path / "a_binary").write_bytes(b"\xff\xfe binaryword")
    (tmp_path / "b.txt").write_text("x = identifier_one + short\n")
    (tmp_path / "c.txt").write_text("identifier_one again\n")
    (tmp_path / "d").mkdir()
    (tmp_path / "d/deep.txt").write_text("deepword_here\n")
    (tmp_path / "e.txt").write_text("secondword\n")
    assert canary_patterns(tmp_path) == ["identifier_one", "secondword"]


def test_canary_passes_a_working_server(tmp_path):
    binary, root = _fake(tmp_path, "modern")
    with McpClient(binary, root, tmp_path, "line") as client:
        check_search_canary(client, root)


def test_canary_and_search_fail_on_the_worktree_trap(tmp_path):
    binary, root = _fake(tmp_path, "worktree_trap")
    with McpClient(binary, root, tmp_path, "line") as client:
        with pytest.raises(McpError, match="worktree must be a registered root"):
            check_search_canary(client, root)
        with pytest.raises(McpError, match="infrastructure failure"):
            client.search({"pattern": "needle", "worktree": "."})
        assert client.search({"pattern": "needle"}).text == "line:needle"


def test_canary_fails_when_searches_find_nothing(tmp_path):
    binary, root = _fake(tmp_path, "no_matches")
    with (
        McpClient(binary, root, tmp_path, "line") as client,
        pytest.raises(McpError, match="found nothing"),
    ):
        check_search_canary(client, root)


def test_worktree_trap_stops_the_run_before_any_model_call(tmp_path, monkeypatch):
    from harness import run
    from harness.episode import Task
    from harness.run import ExperimentConfig, run_experiment
    from models.scripted import ScriptedCall, ScriptedModelFactory

    binary, _ = _fake(tmp_path, "worktree_trap")
    data = tmp_path / "data"
    snapshot = data / "snapshots" / "s"
    snapshot.mkdir(parents=True)
    (snapshot / "mode").write_text("worktree_trap")

    class ExistingDaemon:
        def __init__(self, *args):
            pass

        def close(self):
            pass

        def warm_up(self):
            pass

        def assert_alive(self, root):
            pass

    monkeypatch.setattr(run, "ManagedServers", ExistingDaemon)

    class NoModels(ScriptedModelFactory):
        def build(self, *args, **kwargs):
            raise AssertionError("a model started")

    factory = NoModels(
        scripts={"default": [[ScriptedCall("search", {"pattern": "needle"})]]},
        source_sha256="script",
    )
    with pytest.raises(ValueError, match="A1: canary search"):
        run_experiment(
            ExperimentConfig(
                arms=("A0", "A1"),
                samples=1,
                seed=1,
                data_dir=data,
                ferret=binary,
                search_backend="mcp",
            ),
            [Task(task_id="t", prompt="needle", snapshot="s")],
            factory,
            tmp_path / "run",
            workers=1,
        )
    assert not list((tmp_path / "run/episodes").glob("*"))


def test_health_problem_thresholds():
    from harness.run import _health_problem

    assert _health_problem(4, 4, 0, 0) is None
    assert _health_problem(5, 5, 0, 0) == "5 of 5 finished episodes failed"
    assert _health_problem(100, 10, 0, 0) is None
    assert _health_problem(100, 11, 0, 0) is not None
    assert _health_problem(10, 0, 49, 49) is None
    assert _health_problem(10, 0, 100, 25) is None
    assert _health_problem(10, 0, 100, 26) == "26 of 100 searches returned errors"


def test_search_errors_stop_the_run(tmp_path, monkeypatch):
    from harness import run
    from harness.episode import Task
    from harness.run import ExperimentConfig, run_experiment
    from models.scripted import ScriptedCall, ScriptedModelFactory

    binary, _ = _fake(tmp_path, "modern")
    data = tmp_path / "data"
    snapshot = data / "snapshots" / "s"
    snapshot.mkdir(parents=True)
    (snapshot / "mode").write_text("modern")

    class ExistingDaemon:
        def __init__(self, *args):
            pass

        def close(self):
            pass

        def warm_up(self):
            pass

        def assert_alive(self, root):
            pass

    monkeypatch.setattr(run, "ManagedServers", ExistingDaemon)
    factory = ScriptedModelFactory(
        scripts={
            "default": [
                [ScriptedCall("search", {"pattern": "error"})] * 30,
                [ScriptedCall("submit", {"files": ["mode"]})],
            ]
        },
        source_sha256="script",
    )
    with pytest.raises(ValueError, match="60 of 60 searches returned errors"):
        run_experiment(
            ExperimentConfig(
                arms=("A1", "A2", "A3", "A4"),
                samples=1,
                seed=1,
                data_dir=data,
                ferret=binary,
                search_backend="mcp",
            ),
            [Task(task_id="t", prompt="needle", snapshot="s")],
            factory,
            tmp_path / "run",
            workers=1,
        )
    assert len(list((tmp_path / "run/episodes").glob("*.jsonl"))) < 4

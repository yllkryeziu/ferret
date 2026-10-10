# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for harness.run: planning, running, resuming, and pinning a run."""

import dataclasses
import json
from pathlib import Path

import pytest

from harness.run import (
    EpisodeKey,
    ExperimentConfig,
    RunDirectoryError,
    load_tasks,
    main,
    plan_episodes,
    run_experiment,
)
from harness.trajectory import episode_path, read_episode
from models.scripted import ScriptedModelFactory
from tasks.prepare import index_path


def _config(data: Path, ferret: Path | None, **changes) -> ExperimentConfig:
    fields = {
        "arms": ("A0", "A1", "A3"),
        "samples": 2,
        "seed": 7,
        "data_dir": data,
        "ferret": ferret,
    }
    return ExperimentConfig(**(fields | changes))


def test_episode_ids_are_file_name_safe():
    assert EpisodeKey("abc-1.2", "A1", 0).episode_id == "abc-1.2__A1__s0"
    unsafe = EpisodeKey("locbench/x y", "A3", 2).episode_id
    assert unsafe.startswith("locbench_x_y-")
    assert unsafe.endswith("__A3__s2")
    assert unsafe != EpisodeKey("locbench_x y", "A3", 2).episode_id


def test_plan_groups_each_task_and_runs_probes_in_a0_only(run_inputs):
    _, _, tasks_path, _ = run_inputs
    tasks = load_tasks(tasks_path)
    plan = plan_episodes(tasks, ["A0", "A1", "A3"], samples=2, seed=7)
    by_task = [key.task_id for key in plan]
    assert sorted(by_task) == sorted(["locbench/toy-1"] * 6 + ["probe/toy-2"] * 2)
    # Each task's episodes are contiguous.
    assert by_task == sorted(by_task, key=by_task.index)
    assert {key.arm for key in plan if key.task_id == "probe/toy-2"} == {"A0"}
    assert plan == plan_episodes(list(reversed(tasks)), ["A0", "A1", "A3"], 2, 7)
    assert plan != plan_episodes(tasks, ["A0", "A1", "A3"], 2, 8)

    with pytest.raises(ValueError, match="permits only arm A0"):
        plan_episodes(tasks, ("A1",), samples=1, seed=7)


def test_runs_resumes_and_pins_the_configuration(run_inputs, tmp_path):
    data, ferret, tasks_path, script = run_inputs
    tasks = load_tasks(tasks_path)
    factory = ScriptedModelFactory.from_file(script)
    out = tmp_path / "run"

    report = run_experiment(_config(data, ferret), tasks, factory, out, workers=3)
    assert (report.completed, report.skipped, report.failed) == (8, 0, 0)
    record, messages = read_episode(
        episode_path(out, EpisodeKey("locbench/toy-1", "A3", 1).episode_id)
    )
    assert record["arm"] == "A3"
    assert record["sample"] == 1
    assert record["exit_status"] == "Submitted"
    assert record["submission"] == {"files": ["src/util.py"], "functions": []}
    assert record["search"]["shape"] == "context"
    assert record["model"] == factory.provenance()
    assert (
        record["run"]["config_sha256"]
        == json.loads((out / "run.json").read_text())["config_sha256"]
    )
    assert any("def helper():" in json.dumps(message) for message in messages)
    a0, _ = read_episode(
        episode_path(out, EpisodeKey("probe/toy-2", "A0", 0).episode_id)
    )
    assert a0["search"] is None
    assert a0["submission"]["files"] == ["src/app.py"]

    report = run_experiment(_config(data, ferret), tasks, factory, out, workers=3)
    assert (report.completed, report.skipped, report.failed) == (0, 8, 0)
    with pytest.raises(RunDirectoryError, match="another configuration"):
        run_experiment(_config(data, ferret, seed=8), tasks, factory, out, workers=1)


def test_rejects_missing_inputs(run_inputs, tmp_path):
    data, ferret, tasks_path, script = run_inputs
    tasks = load_tasks(tasks_path)
    factory = ScriptedModelFactory.from_file(script)
    index_path(data, tasks[0].snapshot).unlink()
    with pytest.raises(ValueError, match="missing index"):
        run_experiment(_config(data, ferret), tasks, factory, tmp_path / "a", workers=1)
    with pytest.raises(ValueError, match="need --ferret"):
        run_experiment(
            _config(data, None, arms=("A1",)),
            tasks[:1],
            factory,
            tmp_path / "b",
            workers=1,
        )
    # Arm A0 needs neither an index nor ferret.
    report = run_experiment(
        _config(data, None, arms=("A0",)), tasks, factory, tmp_path / "c", workers=1
    )
    assert report.failed == 0


def test_main(run_inputs, tmp_path):
    data, ferret, tasks_path, script = run_inputs
    from tasks.schema import read_tasks, write_tasks

    source_only = tmp_path / "source-only.jsonl"
    write_tasks(source_only, read_tasks(tasks_path)[:1])
    argv = [
        f"--tasks={source_only}",
        f"--out={tmp_path / 'run'}",
        f"--data={data}",
        f"--ferret={ferret}",
        "--arms=A1,A2",
        "--samples=1",
        "--model=scripted",
        f"--script={script}",
    ]
    assert main(argv) == 0
    assert len(list((tmp_path / "run" / "episodes").glob("*.jsonl"))) == 2
    assert main(argv[:-1]) == 2  # --model=scripted without --script.


def test_failed_episode_records_reason_and_does_not_write_success(run_inputs, tmp_path):
    data, ferret, tasks_file, script = run_inputs
    factory = ScriptedModelFactory.from_file(script)
    factory.scripts["default"] = []
    run = tmp_path / "failed"
    report = run_experiment(
        _config(data, ferret, arms=("A1",), samples=1),
        load_tasks(tasks_file)[:1],
        factory,
        run,
        workers=1,
    )
    assert report.failed == 1
    failures = list((run / "episodes").glob("*.failure.json"))
    assert len(failures) == 1
    assert "script has no turn" in json.loads(failures[0].read_text())["reason"]
    assert not list((run / "episodes").glob("*.jsonl"))


def test_a4_gated_before_server_or_model_start(run_inputs, tmp_path):
    data, ferret, tasks_file, script = run_inputs
    with pytest.raises(ValueError, match="--router"):
        run_experiment(
            dataclasses.replace(
                _config(data, ferret, arms=("A4",)), search_backend="mcp"
            ),
            load_tasks(tasks_file),
            ScriptedModelFactory.from_file(script),
            tmp_path / "run",
            workers=1,
        )


def test_probe_episode_refuses_search_even_without_plan():
    from harness.arms import ARMS
    from harness.episode import EpisodeCaps, Task, run_episode
    from harness.prompts import Prompts

    with pytest.raises(ValueError, match="permits only arm A0"):
        run_episode(
            task=Task(
                task_id="probe/one",
                prompt="An issue",
                snapshot="0" * 40,
                bucket="P",
                split="dev",
            ),
            arm=ARMS["A1"],
            model=None,
            sandbox=None,
            search=None,
            caps=EpisodeCaps(),
            prompts=Prompts.load(),
        )

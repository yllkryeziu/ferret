# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end analysis over scripted trajectories, without an API call."""

from analysis.analyze import main
from harness.run import ExperimentConfig, load_tasks, run_experiment
from models.scripted import ScriptedModelFactory


def test_analyze_scripted_run(run_inputs, tmp_path):
    data, ferret, tasks_file, script = run_inputs
    run = tmp_path / "run"
    config = ExperimentConfig(
        arms=("A0", "A1", "A2", "A3"), samples=1, seed=1, data_dir=data, ferret=ferret
    )
    report = run_experiment(
        config,
        load_tasks(tasks_file),
        ScriptedModelFactory.from_file(script),
        run,
        workers=2,
    )
    assert report.failed == 0
    assert main(["--tasks", str(tasks_file), "--run", str(run), "--split", "dev"]) == 0
    assert (run / "episodes.csv").is_file()
    assert (run / "arm_summary.csv").is_file()
    assert (run / "contrasts.csv").is_file()
    assert (run / "costs.csv").is_file()


def test_analyze_rejects_unknown_task_id(tmp_path):
    import pytest

    from analysis.analyze import analyze
    from analysis.trajectory import Episode

    with pytest.raises(ValueError, match="unknown tasks"):
        analyze(
            [],
            [
                Episode(
                    task_id="unknown",
                    arm="A1",
                    sample=0,
                    turns=(),
                    submission=None,
                    cap_hit=False,
                )
            ],
            tmp_path,
        )
    assert not (tmp_path / "episodes.csv").exists()


def test_mounted_pilot_v2_matches_published_means(tmp_path):
    import math
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import pytest

    from analysis.analyze import analyze
    from analysis.harness_episodes import read_run
    from tasks.schema import Split, read_tasks

    manifest = Path("/Volumes/ferret-cs/eval-linux/pilot-v2.jsonl")
    run = Path.home() / "ferret-data/eval/runs/pilot-v2"
    if not manifest.exists() or not (run / "episodes").exists():
        pytest.skip("pilot-v2 manifest and trajectories not mounted")
    analyze(read_tasks(manifest), read_run(run), tmp_path, split=Split.DEV)
    frame = pd.read_csv(tmp_path / "episodes.csv")
    assert frame.groupby("arm").success.sum().to_dict() == {
        "A0": 6,
        "A1": 12,
        "A2": 10,
        "A3": 10,
    }
    logs = frame.pivot(index="task_id", columns="arm", values="log_total_tokens")
    ratio = logs.A3 - logs.A1
    assert math.exp(ratio.mean()) == pytest.approx(0.84, abs=0.01)
    assert ratio.std(ddof=1) == pytest.approx(1.00, abs=0.01)
    rng = np.random.default_rng(0)
    draws = ratio.to_numpy()[rng.integers(20, size=(10000, 20))].mean(axis=1)
    assert np.exp(np.quantile(draws, [0.025, 0.975])) == pytest.approx(
        [0.54, 1.26], abs=0.02
    )


def test_probe_only_analysis_has_no_ladder_figure(tmp_path, run_inputs):
    from analysis.analyze import analyze
    from analysis.harness_episodes import read_run
    from tasks.schema import Split, read_tasks

    data, ferret, tasks_file, script = run_inputs
    run = tmp_path / "probe"
    report = run_experiment(
        ExperimentConfig(arms=("A0",), samples=1, seed=0, data_dir=data, ferret=ferret),
        [task for task in load_tasks(tasks_file) if task.bucket == "P"],
        ScriptedModelFactory.from_file(script),
        run,
        workers=1,
    )
    assert report.failed == 0
    analyze(
        read_tasks(tasks_file), read_run(run), tmp_path / "analysis", split=Split.DEV
    )
    assert (tmp_path / "analysis/episodes.csv").exists()
    assert not (tmp_path / "analysis/cost_per_solved.png").exists()

# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""The replay's build step, systems.json, held-out fit, summary and blocks."""

import csv
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.analyze import main as analyze_main
from analysis.analyze import read_measurements
from analysis.breakeven import (
    BreakEvenModel,
    PowerLaw,
    RepoMeasurement,
    break_even_queries,
    break_even_table,
    holdout_table,
    plot_break_even,
)
from analysis.replay_summary import replay_summary
from analysis.report import (
    render_break_even,
    render_break_even_holdout,
    render_findings,
    render_replay,
)
from harness.run import ExperimentConfig, load_tasks, run_experiment
from harness.systems import (
    build_indexes,
    count_regular_files,
    measure,
    measurement_name,
)
from harness.systems import main as systems_main
from models.scripted import ScriptedModelFactory
from tasks.schema import Bucket, read_tasks

from .conftest import RUN_SNAPSHOT


def _synthetic(name, bucket, files, scale=1.0):
    """A repository on the power laws B = 1e-3 N, t_rg = 2e-6 N, t_idx = 3e-3."""
    return RepoMeasurement(
        name=name,
        bucket=bucket,
        files=files,
        build_seconds=1e-3 * files * scale,
        rg_seconds=2e-6 * files + 0.01,
        index_seconds=0.003,
    )


def _repos(linux_scale=1.0, linux_rg_scale=1.0):
    repos = [_synthetic(f"s{n}", "S", n) for n in (200, 800, 3_000, 9_000)]
    repos.append(_synthetic("chromium", "XL", 400_000))
    linux = _synthetic("linux", "L", 90_000, scale=linux_scale)
    return [
        *repos,
        dataclasses.replace(linux, rg_seconds=linux.rg_seconds * linux_rg_scale),
    ]


def test_power_law_is_recovered_from_synthetic_measurements():
    law = PowerLaw.fit(
        [10, 100, 1000, 10_000], [0.05 * n**0.7 for n in (10, 100, 1000, 10_000)]
    )
    assert law.exponent == pytest.approx(0.7)
    assert law.coefficient == pytest.approx(0.05)


def test_holdout_predicts_linux_from_the_other_snapshots():
    table = holdout_table(_repos())
    assert set(table["quantity"]) == {
        "build_seconds",
        "rg_seconds",
        "index_seconds",
        "break_even_queries",
    }
    # B is an exact power law; t_rg is a·N + c, so its power-law fit is only
    # approximate, but the prediction stays well inside a factor of two.
    q = table[table["quantity"] == "break_even_queries"].iloc[0]
    assert q["fitted_on"] == 5
    assert q["worktrees"] == 1
    assert q["factor"] < 2.0
    assert bool(q["validated"]) is True
    build = table[table["quantity"] == "build_seconds"].iloc[0]
    assert build["predicted"] == pytest.approx(build["measured"], rel=1e-6)
    assert build["factor"] == pytest.approx(1.0)


def test_a_perturbed_holdout_fails_the_factor_two_test():
    # Linux's build takes 5 times what the other snapshots predict.
    table = holdout_table(_repos(linux_scale=5.0))
    q = table[table["quantity"] == "break_even_queries"].iloc[0]
    assert q["factor"] > 4.0
    assert bool(q["validated"]) is False
    # And a Linux where the index never wins is infinitely far.
    never = holdout_table(_repos(linux_rg_scale=1e-9))
    q = never[never["quantity"] == "break_even_queries"].iloc[0]
    assert math.isinf(q["factor"])
    assert bool(q["validated"]) is False


def test_holdout_excludes_the_held_out_bucket_from_the_fit():
    repos = _repos()
    with_outlier = [
        dataclasses.replace(m, build_seconds=m.build_seconds * 50)
        if m.bucket == "L"
        else m
        for m in repos
    ]
    # Changing Linux's own numbers must not change what is predicted for it.
    a = holdout_table(repos).set_index("quantity")["predicted"]
    b = holdout_table(with_outlier).set_index("quantity")["predicted"]
    assert a.equals(b)
    assert holdout_table([m for m in repos if m.bucket != "L"]).empty


def test_break_even_table_carries_buckets_and_inputs():
    table = break_even_table(_repos(), (1, 10))
    assert {"bucket", "build_seconds", "rg_seconds", "index_seconds"} <= set(table)
    assert set(table["bucket"]) == {"S", "XL", "L"}
    one, ten = (table[table["worktrees"] == w].set_index("repo") for w in (1, 10))
    assert one.loc["linux", "break_even_queries"] == pytest.approx(
        10 * ten.loc["linux", "break_even_queries"]
    )


def test_old_measurement_files_still_load(tmp_path):
    path = tmp_path / "systems.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "linux",
                    "files": 9,
                    "build_seconds": 1.0,
                    "rg_seconds": 0.5,
                    "index_seconds": 0.1,
                }
            ]
        )
    )
    assert read_measurements(path)[0].bucket == ""


def test_the_figure_labels_one_point_per_bucket(tmp_path, monkeypatch):
    labels = []
    from matplotlib.axes import Axes

    original = Axes.annotate

    def spy(self, text, *args, **kwargs):
        labels.append(text)
        return original(self, text, *args, **kwargs)

    monkeypatch.setattr(Axes, "annotate", spy)
    repos = _repos()
    plot_break_even(
        BreakEvenModel.fit(repos),
        repos,
        [3, 5, 8],
        tmp_path / "fig.png",
        worktrees=(1, 10),
    )
    assert (tmp_path / "fig.png").stat().st_size > 0
    repo_labels = [x for x in labels if x != "searches per task (median, IQR)"]
    assert sorted(repo_labels) == [
        "bucket S: 4 repos",
        "chromium (bucket XL)",
        "linux (bucket L)",
    ]


def _write_snapshot(root: Path, files: dict[str, str]) -> None:
    for name, content in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(content)


def test_counts_regular_files_without_git(tmp_path):
    _write_snapshot(
        tmp_path,
        {"a.c": "x", "src/b.c": "x", ".git/HEAD": "x", ".git/objects/o": "x"},
    )
    (tmp_path / "link").symlink_to(tmp_path / "a.c")
    # A nested directory named .git (a submodule's) is not the repository's.
    _write_snapshot(tmp_path, {"vendor/.git/config": "x"})
    assert count_regular_files(tmp_path) == 3


_FAKE_FERRET = """\
import pathlib, sys, time
index = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--index="))
log = pathlib.Path(index).parent / "calls.log"
with log.open("a") as handle:
    handle.write("exists\\n" if pathlib.Path(index).exists() else "fresh\\n")
time.sleep(0.05)
pathlib.Path(index).write_text("index")
"""


def test_build_indexes_builds_twice_from_scratch_and_records_the_binary(tmp_path):
    source = tmp_path / "source"
    _write_snapshot(source / ("a" * 40), {"x.c": "1", "d/y.c": "2", ".git/HEAD": "3"})
    _write_snapshot(source / ("b" * 40), {"z.c": "1"})
    ferret = tmp_path / "ferret"
    ferret.write_text(f"#!{sys.executable}\n{_FAKE_FERRET}")
    ferret.chmod(0o755)
    data = tmp_path / "data"

    builds = build_indexes(ferret, source, ["a" * 40, "b" * 40], data)

    assert (data / "snapshots" / ("a" * 40)).is_symlink()
    assert (data / "indexes" / ("a" * 40 + ".fseg")).read_text() == "index"
    # Each build starts without the previous one's file.
    log = (data / "indexes" / "calls.log").read_text().split()
    assert log == ["fresh"] * 4
    entry = builds["snapshots"]["a" * 40]
    assert entry["files"] == 2
    assert entry["build_seconds"] >= 0.05
    assert entry["build_cold_seconds"] >= 0.05
    assert entry["index_bytes"] == 5
    import hashlib

    assert builds["ferret"]["sha256"] == hashlib.sha256(ferret.read_bytes()).hexdigest()
    with pytest.raises(FileNotFoundError):
        build_indexes(ferret, source, ["c" * 40], data)


def _task_file(tmp_path, spec):
    """Writes tasks for {(bucket, repo, snapshot)}; returns the path."""
    from tasks import schema

    from .conftest import _run_task

    tasks = []
    for i, (bucket, repo, snapshot) in enumerate(spec):
        task = _run_task(f"{repo.replace('/', '-')}/t{i}", Bucket(bucket))
        tasks.append(dataclasses.replace(task, repo=repo, snapshot=snapshot))
    path = tmp_path / "tasks.jsonl"
    schema.write_tasks(path, tasks)
    return path


def _rows():
    def row(snapshot, rg_status, rg_ms, raw_ms, **extra):
        return {
            "snapshot": snapshot,
            "episode_id": f"e-{snapshot[0]}",
            "arm": "A1",
            "rg_status": str(rg_status),
            "rg_ms": str(rg_ms),
            "ferret_raw_ms": str(raw_ms),
        } | extra

    return [
        row("a" * 40, 0, 30, 10),
        row("a" * 40, 1, 50, 20),
        row("a" * 40, 2, 9_999, 1),
        row("a" * 40, 0, 70, 30),
        row("b" * 40, 0, 400, 20),
    ]


def test_systems_json_has_one_measurement_per_replayed_snapshot(tmp_path):
    tasks = read_tasks(
        _task_file(
            tmp_path,
            [("S", "pydata/xarray", "a" * 40), ("L", "linux", "b" * 40)],
        )
    )
    builds = {
        "snapshots": {
            "a" * 40: {"files": 120, "build_seconds": 1.5},
            "b" * 40: {"files": 90_000, "build_seconds": 80.0},
            "c" * 40: {"files": 5, "build_seconds": 0.1},  # not replayed
        }
    }
    measurements = measure(_rows(), builds, tasks)
    assert [m.name for m in measurements] == ["pydata/xarray@aaaaaaa", "linux"]
    small, linux = measurements
    # Medians over the calls ripgrep answered (status 0 or 1), in seconds.
    assert small.rg_seconds == pytest.approx(0.050)
    assert small.index_seconds == pytest.approx(0.020)
    assert (small.files, small.build_seconds, small.bucket) == (120, 1.5, "S")
    assert (linux.files, linux.bucket) == (90_000, "L")
    assert measurement_name(tasks[1]) == "linux"
    with pytest.raises(ValueError, match="without a build"):
        measure(
            _rows(), {"snapshots": {"a" * 40: builds["snapshots"]["a" * 40]}}, tasks
        )


def test_systems_cli_reads_the_csv_and_writes_json(tmp_path):
    tasks = _task_file(
        tmp_path, [("S", "pydata/xarray", "a" * 40), ("L", "linux", "b" * 40)]
    )
    replay = tmp_path / "replay.csv"
    rows = _rows()
    with replay.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    builds = tmp_path / "builds.json"
    builds.write_text(
        json.dumps(
            {
                "snapshots": {
                    "a" * 40: {"files": 120, "build_seconds": 1.5},
                    "b" * 40: {"files": 90_000, "build_seconds": 80.0},
                }
            }
        )
    )
    out = tmp_path / "systems.json"
    argv = ["measure", f"--replay={replay}", f"--builds={builds}", f"--tasks={tasks}"]
    assert systems_main([*argv, f"--out={out}"]) == 0
    loaded = read_measurements(out)
    assert [m.bucket for m in loaded] == ["S", "L"]
    assert loaded[0].files == 120


def _replay_frame():
    base = {
        "arm": "A1",
        "ferret_ms": 10.0,
        "ferret_raw_ms": 8.0,
        "rg_status": 0,
        "only_ferret": 0,
        "only_rg": 0,
        "zoekt_status": "ok",
        "zoekt_truncated": False,
        "zoekt_only_vs_ferret": 0,
        "ferret_only_vs_zoekt": 0,
        "zoekt_build_s": 12.0,
    }
    rows = [
        base
        | {"snapshot": "a" * 40, "episode_id": "ea", "rg_ms": 20.0, "zoekt_ms": 1.0},
        base
        | {
            "snapshot": "a" * 40,
            "episode_id": "ea",
            "rg_ms": 30.0,
            "zoekt_ms": 2.0,
            "only_rg": 3,
        },
        base
        | {
            "snapshot": "a" * 40,
            "episode_id": "ea",
            "rg_status": 2,
            "rg_ms": 40.0,
            "zoekt_status": "error: query 3: invalid regexp",
            "zoekt_ms": np.nan,
            "zoekt_only_vs_ferret": np.nan,
            "ferret_only_vs_zoekt": np.nan,
        },
        base
        | {"snapshot": "b" * 40, "episode_id": "eb", "rg_ms": 1_010.0, "zoekt_ms": 5.0},
        base
        | {
            "snapshot": "b" * 40,
            "episode_id": "eb",
            "rg_ms": 1_010.0,
            "zoekt_ms": 5.0,
            "zoekt_truncated": True,
            "zoekt_build_s": 30.0,
        },
    ]
    return pd.DataFrame(rows)


def test_replay_summary_counts_comparable_calls_per_bucket(tmp_path):
    tasks = read_tasks(
        _task_file(
            tmp_path, [("L", "linux", "b" * 40), ("S", "pydata/xarray", "a" * 40)]
        )
    )
    provenance = {
        "snapshots": {
            "a" * 40: {"calls_in_run": 9},
            "b" * 40: {"calls_in_run": 700},
        }
    }
    summary = replay_summary(
        _replay_frame(), tasks, {"ea": 10.0, "eb": 100.0}, provenance
    ).set_index("bucket")
    assert list(summary.index) == ["S", "L"]
    s, big = summary.loc["S"], summary.loc["L"]
    assert (s["calls_replayed"], s["calls_in_run"]) == (3, 9)
    assert (big["calls_replayed"], big["calls_in_run"]) == (2, 700)
    # ripgrep: 2 comparable calls (one errored), one of them identical.
    assert (s["rg_comparable"], s["rg_identical"], s["rg_errors"]) == (2, 1, 1)
    # Zoekt: one error is neither comparable nor dropped.
    assert (s["zoekt_comparable"], s["zoekt_identical"], s["zoekt_errors"]) == (
        2,
        2,
        0 + 1,
    )
    assert (big["zoekt_comparable"], big["zoekt_truncated"]) == (1, 1)
    assert s["ferret_p50_ms"] == 10.0
    assert s["rg_p50_ms"] == pytest.approx(30.0)
    assert s["zoekt_p50_ms"] == pytest.approx(1.5)
    # Two comparable ripgrep calls, 10 and 20 ms slower, on a 10 s episode.
    assert s["ratio_rg_over_ferret"] == pytest.approx(10.03 / 10)
    assert s["zoekt_build_seconds"] == 12.0
    assert big["zoekt_build_seconds"] == 12.0


def test_render_blocks_from_files(tmp_path):
    tasks = read_tasks(
        _task_file(
            tmp_path, [("L", "linux", "b" * 40), ("S", "pydata/xarray", "a" * 40)]
        )
    )
    replay_summary(_replay_frame(), tasks, {"ea": 10.0, "eb": 100.0}).to_csv(
        tmp_path / "replay_summary.csv", index=False
    )
    repos = _repos()
    break_even_table(repos, (1, 10)).to_csv(tmp_path / "breakeven.csv", index=False)
    holdout_table(repos).to_csv(tmp_path / "breakeven_holdout.csv", index=False)
    pd.DataFrame(
        {
            "bucket": ["S", "S", "L", "XL", "XL", "P"],
            "arm": ["A1", "A3", "A2", "A4", "A0", "A0"],
            "search_calls": [2, 4, 7, 10, 0, 0],
        }
    ).to_csv(tmp_path / "episodes.csv", index=False)

    replay = render_replay(tmp_path)
    assert replay.splitlines()[2].startswith("| S | 3 of nan")  # no provenance: nan
    assert "1 of 2" in replay

    table = render_break_even(tmp_path)
    lines = table.splitlines()
    assert "Median N" in lines[0] and "Never pay" in lines[0]
    s_row = next(line for line in lines if line.startswith("| S "))
    assert "| 4 |" in s_row
    # Median searches per ladder episode, A0 left out.
    assert s_row.rstrip().endswith("| 3 |")
    assert (
        next(line for line in lines if line.startswith("| XL "))
        .rstrip()
        .endswith("| 10 |")
    )

    holdout = render_break_even_holdout(tmp_path)
    assert "validated" in holdout and "Q* (W = 1)" in holdout
    holdout_table(_repos(linux_scale=5.0)).to_csv(
        tmp_path / "breakeven_holdout.csv", index=False
    )
    assert "not validated" in render_break_even_holdout(tmp_path)
    holdout_table([m for m in repos if m.bucket != "L"]).to_csv(
        tmp_path / "breakeven_holdout.csv", index=False
    )
    assert "No held-out snapshot" in render_break_even_holdout(tmp_path)

    document = (
        "<!-- BEGIN GENERATED: replay -->\nSTALE\n<!-- END GENERATED: replay -->\n"
        "<!-- BEGIN GENERATED: break-even -->\nSTALE\n"
        "<!-- END GENERATED: break-even -->\n"
        "<!-- BEGIN GENERATED: break-even-holdout -->\nSTALE\n"
        "<!-- END GENERATED: break-even-holdout -->\n"
    )
    rendered = render_findings(document, tmp_path)
    assert "STALE" not in rendered and rendered.count("| S ") >= 2


def test_never_paying_snapshots_read_never(tmp_path):
    repos = [
        dataclasses.replace(_synthetic("s1", "S", 100), rg_seconds=0.001),
        _synthetic("s2", "S", 5_000),
    ]
    break_even_table(repos, (1, 10)).to_csv(tmp_path / "breakeven.csv", index=False)
    pd.DataFrame({"bucket": ["S"], "arm": ["A1"], "search_calls": [3]}).to_csv(
        tmp_path / "episodes.csv", index=False
    )
    assert math.isinf(float(break_even_queries(1.0, 0.001, 0.003)))
    row = render_break_even(tmp_path).splitlines()[2]
    assert "never" in row
    assert row.split("|")[-3].strip() == "1"  # one snapshot never pays


def test_analyze_cli_writes_replay_outputs(run_inputs, tmp_path):
    data, ferret, tasks_file, script = run_inputs
    run = tmp_path / "run"
    config = ExperimentConfig(
        arms=("A0", "A1", "A2", "A3"), samples=1, seed=1, data_dir=data, ferret=ferret
    )
    run_experiment(
        config,
        load_tasks(tasks_file),
        ScriptedModelFactory.from_file(script),
        run,
        workers=2,
    )
    episode = next(
        json.loads(path.read_text().splitlines()[0])["episode_id"]
        for path in sorted((run / "episodes").glob("*A1*.jsonl"))
    )
    replay = tmp_path / "replay.csv"
    frame = _replay_frame()
    frame = frame.assign(snapshot=RUN_SNAPSHOT, episode_id=episode)
    frame.to_csv(replay, index=False)
    replay.with_name("replay.provenance.json").write_text(
        json.dumps({"snapshots": {RUN_SNAPSHOT: {"calls_in_run": 11}}})
    )
    systems = tmp_path / "systems.json"
    systems.write_text(
        json.dumps(
            [
                dataclasses.asdict(_synthetic("toy@1", "S", 100)),
                dataclasses.asdict(_synthetic("toy@2", "S", 900)),
            ]
        )
    )
    out = tmp_path / "results"
    argv = [
        f"--tasks={tasks_file}",
        f"--run={run}",
        "--split=dev",
        f"--out={out}",
        f"--systems={systems}",
        f"--replay={replay}",
    ]
    assert analyze_main(argv) == 0
    summary = pd.read_csv(out / "replay_summary.csv")
    # Both toy tasks share the snapshot, so the last one names its bucket.
    assert len(summary) == 1
    assert summary["calls_in_run"].iloc[0] == 11
    assert pd.read_csv(out / "breakeven.csv")["bucket"].eq("S").all()
    # No Linux in this run, so there is nothing held out.
    assert pd.read_csv(out / "breakeven_holdout.csv").empty
    assert (out / "break_even.png").is_file()
    manifest = json.loads((out / "manifest.json").read_text())
    assert any(path.endswith("replay.csv") for path in manifest["inputs"])

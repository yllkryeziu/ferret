# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Runs the pre-registered analysis and writes its result tables and figures.

    uv run python -m analysis.analyze --tasks tasks.jsonl \\
        --run RUN_DIR --out results/ [--systems systems.json] \\
        [--replay replay.csv]

reads one split of the tasks (test by default) and every episode on them, and
writes into --out:

* episodes.csv: one row of `analysis.metrics` per episode.
* arm_summary.csv: per bucket and arm (and "all" pooled over S, L, XL, E):
  success with a bootstrap CI, the secondary metrics, and exposure.
* contrasts.csv: every ladder contrast for every metric, pooled and per
  bucket, for all tasks and for the two sensitivity subsets ("clean":
  created after the model's training cutoff; "search_needed": not solved by
  A0 in at least 2 of 3 samples). Holm adjustment runs within each (metric,
  subset) family over the three pooled contrasts; per-bucket rows are
  exploratory and unadjusted.
* glmm.csv: the GLMM's fixed effects and the task-intercept SD.
* contamination.csv: arm A0 per bucket, including the SWE-bench probe.
* costs.csv and cost_per_solved.png: dollars per episode and per solved task.
* breakeven.csv, breakeven_holdout.csv and break_even.png, when --systems
  gives benchmark numbers (`harness.systems`).
* replay_summary.csv, when --replay gives the replay's CSV (`harness.replay`).
* manifest.json: input hashes and parameters, so every table is traceable.

`analysis.report` renders these files into the tables of a Markdown report.
"""

import argparse
import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from analysis.bootstrap import arm_mean, bootstrap_mean, paired_differences
from analysis.breakeven import (
    BreakEvenModel,
    RepoMeasurement,
    break_even_table,
    cost_table,
    holdout_table,
    plot_break_even,
    plot_cost_per_solved,
)
from analysis.glmm import fit_success_glmm
from analysis.harness_episodes import read_run
from analysis.metrics import metrics_frame
from analysis.multiplicity import holm_adjust
from analysis.protocol import CONTRASTS, Arm, TokenPrices, contrast_name
from analysis.replay_summary import replay_summary
from analysis.trajectory import Episode
from harness.replay import episode_wall_times
from tasks.schema import Bucket, Split, Task, read_tasks

A0_SOLVED_MIN_SAMPLES = 2
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
GPT_OSS_120B_PRICES = TokenPrices(1.25, 10.0, 0.125)
MODEL_TRAINING_CUTOFF = datetime(2026, 8, 30, tzinfo=UTC)

# Contrast families, primary first. Each is Holm-adjusted on its own.
CONTRAST_METRICS = (
    "success",
    "restricted_exposure_turns",
    "restricted_exposure_tokens",
    "log_total_tokens",
    "file_reciprocal_rank",
    "function_success",
)
POOLED = "all"
# The contamination probe runs only A0, so it never enters pooled estimates.
_POOLED_BUCKETS = frozenset(bucket.value for bucket in Bucket) - {Bucket.P.value}
BREAK_EVEN_WORKTREES = (1, 10)


def analyze(
    tasks: Sequence[Task],
    episodes: Sequence[Episode],
    out: Path,
    *,
    split: Split = Split.TEST,
    systems: Sequence[RepoMeasurement] = (),
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> None:
    """Writes every result file for the episodes on `split` tasks into `out`.

    Raises:
        ValueError: if an episode names an unknown task.
    """
    by_id = {task.id: task for task in tasks}
    unknown = sorted({e.task_id for e in episodes} - by_id.keys())
    if unknown:
        raise ValueError(f"episodes name unknown tasks, e.g. {unknown[:3]}")
    chosen = [e for e in episodes if by_id[e.task_id].split is split]
    frame = metrics_frame(chosen, by_id)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "episodes.csv", index=False)
    pooled = frame[frame["bucket"].isin(_POOLED_BUCKETS)]
    arm_summary(frame, n_resamples=n_resamples, seed=seed).to_csv(
        out / "arm_summary.csv", index=False
    )
    subsets = {
        POOLED: pooled,
        "clean": pooled[pooled["task_id"].isin(_clean_tasks(by_id.values()))],
        "search_needed": pooled[~pooled["task_id"].isin(a0_solved_tasks(frame))],
    }
    contrasts(subsets, n_resamples=n_resamples, seed=seed).to_csv(
        out / "contrasts.csv", index=False
    )
    if {"A1", "A2", "A3", "A4"} <= set(pooled.arm):
        fit = fit_success_glmm(pooled)
        glmm = pd.concat(
            [
                fit.fixed_effects,
                pd.DataFrame([{"term": "task intercept SD", "mean": fit.task_sd}]),
            ]
        )
        glmm.to_csv(out / "glmm.csv", index=False)
    contamination(frame, n_resamples=n_resamples, seed=seed).to_csv(
        out / "contamination.csv", index=False
    )
    costs = cost_table(frame, GPT_OSS_120B_PRICES)
    costs.to_csv(out / "costs.csv", index=False)
    ladder_costs = costs[costs["bucket"].isin(_POOLED_BUCKETS)]
    if not ladder_costs.empty:
        plot_cost_per_solved(ladder_costs, out / "cost_per_solved.png")
    if systems:
        break_even_table(systems, BREAK_EVEN_WORKTREES).to_csv(
            out / "breakeven.csv", index=False
        )
        holdout_table(systems).to_csv(out / "breakeven_holdout.csv", index=False)
        ladder = pooled[pooled["arm"] != Arm.A0.value]
        plot_break_even(
            BreakEvenModel.fit(systems),
            systems,
            ladder["search_calls"],
            out / "break_even.png",
            worktrees=BREAK_EVEN_WORKTREES,
        )


def arm_summary(frame: pd.DataFrame, *, n_resamples: int, seed: int) -> pd.DataFrame:
    """Returns one row per (bucket, arm), plus pooled rows with bucket "all"."""
    rows = []
    pooled = frame[frame["bucket"].isin(_POOLED_BUCKETS)]
    groups = [(POOLED, pooled), *frame.groupby("bucket")]
    for bucket, data in groups:
        for arm_value, episodes in data.groupby("arm"):
            success = arm_mean(
                episodes, "success", Arm(arm_value), n_resamples=n_resamples, seed=seed
            )
            calls = episodes["tool_calls"].sum()
            rows.append(
                {
                    "bucket": bucket,
                    "arm": arm_value,
                    "tasks": success.n_tasks,
                    "episodes": len(episodes),
                    "success": success.estimate,
                    "success_ci_low": success.ci_low,
                    "success_ci_high": success.ci_high,
                    "file_mrr": episodes["file_reciprocal_rank"].mean(),
                    "function_success": episodes["function_success"].mean(),
                    "submit_rate": episodes["submitted"].mean(),
                    "cap_hit_rate": episodes["cap_hit"].mean(),
                    "median_total_tokens": episodes["total_tokens"].median(),
                    "mean_turns": episodes["turns"].mean(),
                    "mean_search_calls": episodes["search_calls"].mean(),
                    "invalid_call_rate": episodes["invalid_tool_calls"].sum() / calls
                    if calls
                    else float("nan"),
                    "exposure_rate": episodes["exposure_turn"].notna().mean(),
                    "restricted_exposure_turns": episodes[
                        "restricted_exposure_turns"
                    ].mean(),
                }
            )
    return pd.DataFrame(rows)


def contrasts(
    subsets: Mapping[str, pd.DataFrame], *, n_resamples: int, seed: int
) -> pd.DataFrame:
    """Returns every ladder contrast for every metric and subset.

    The primary metric runs on every subset, the others on all tasks only.
    Contrasts without a single task that has the metric in both arms (function
    success in a bucket without function gold) are omitted.
    """
    rows = []
    for metric in CONTRAST_METRICS:
        for subset, data in subsets.items():
            if subset != POOLED and metric != "success":
                continue
            family: dict[str, dict[str, object]] = {}
            for bucket, bucket_rows in [(POOLED, data), *data.groupby("bucket")]:
                for baseline, treatment in CONTRASTS:
                    differences = paired_differences(
                        bucket_rows, metric, baseline, treatment
                    )
                    if differences.empty:
                        continue
                    result = bootstrap_mean(
                        differences, n_resamples=n_resamples, seed=seed
                    )
                    row = {
                        "metric": metric,
                        "subset": subset,
                        "bucket": bucket,
                        "contrast": contrast_name(baseline, treatment),
                        **dataclasses.asdict(result),
                    }
                    rows.append(row)
                    if bucket == POOLED:
                        family[row["contrast"]] = row
            p_values = {name: float(row["p_value"]) for name, row in family.items()}
            for name, adjusted in holm_adjust(p_values).items():
                family[name]["p_holm"] = adjusted
    return pd.DataFrame(rows)


def a0_solved_tasks(frame: pd.DataFrame) -> set[str]:
    """Returns the tasks arm A0 solves in at least A0_SOLVED_MIN_SAMPLES samples."""
    a0 = frame[frame["arm"] == Arm.A0.value]
    wins = a0.groupby("task_id")["success"].sum()
    return set(wins[wins >= A0_SOLVED_MIN_SAMPLES].index)


def contamination(frame: pd.DataFrame, *, n_resamples: int, seed: int) -> pd.DataFrame:
    """Returns A0's success and share of tasks solved without search, by bucket."""
    solved = a0_solved_tasks(frame)
    rows = []
    for bucket, data in frame[frame["arm"] == Arm.A0.value].groupby("bucket"):
        success = arm_mean(data, "success", Arm.A0, n_resamples=n_resamples, seed=seed)
        tasks = set(data["task_id"])
        rows.append(
            {
                "bucket": bucket,
                "tasks": len(tasks),
                "a0_success": success.estimate,
                "a0_success_ci_low": success.ci_low,
                "a0_success_ci_high": success.ci_high,
                "solved_without_search": len(tasks & solved) / len(tasks),
            }
        )
    return pd.DataFrame(rows)


def write_replay_summary(
    tasks: Sequence[Task], replay: Path, run: Path, out: Path
) -> None:
    """Writes `out/replay_summary.csv` from the replay CSV `replay`, whose
    episodes' wall times are read from `run`."""
    provenance_path = replay.with_name(replay.stem + ".provenance.json")
    provenance = (
        json.loads(provenance_path.read_text()) if provenance_path.is_file() else None
    )
    replay_summary(
        pd.read_csv(replay), tasks, episode_wall_times(run), provenance
    ).to_csv(out / "replay_summary.csv", index=False)


def read_measurements(path: Path) -> list[RepoMeasurement]:
    """Reads a JSON list of RepoMeasurement objects written by the benchmarks."""
    return [RepoMeasurement(**entry) for entry in json.loads(path.read_text())]


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="results directory (default: RUN)")
    parser.add_argument("--split", type=Split, default=Split.TEST, choices=list(Split))
    parser.add_argument("--systems", type=Path, help="JSON list of RepoMeasurement")
    parser.add_argument("--replay", type=Path, help="CSV written by harness.replay")
    args = parser.parse_args(argv)
    systems = read_measurements(args.systems) if args.systems else []
    tasks = read_tasks(args.tasks)
    analyze(
        tasks,
        read_run(args.run),
        args.out or args.run,
        split=args.split,
        systems=systems,
    )
    if args.replay:
        write_replay_summary(tasks, args.replay, args.run, args.out or args.run)
    inputs = [
        args.tasks,
        *sorted((args.run / "episodes").glob("*.jsonl")),
        *([args.systems] if args.systems else []),
        *([args.replay] if args.replay else []),
    ]
    manifest = {
        "inputs": {str(path): _sha256(path) for path in inputs},
        "split": args.split.value,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    ((args.out or args.run) / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return 0


def _clean_tasks(tasks: Iterable[Task]) -> set[str]:
    return {t.id for t in tasks if t.provenance.created_at >= MODEL_TRAINING_CUTOFF}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Scores the team episodes and compares F with R.

python -m teams.analyze --tasks T.jsonl RUN [RUN ...] > report.md
"""

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.metrics import file_reciprocal_rank, file_success, function_success
from teams.commands import SEARCH

BOOTSTRAP = 10_000
SEED = 0


def episode_metrics(record: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    answer = record.get("answer") or {}
    files = answer.get("files") or []
    functions = answer.get("functions") or []
    items = record["items"]
    usage = record["usage"]
    searches = [i for i in items if i["class"] == SEARCH]
    changed = set(record.get("changed") or [])
    return {
        "key": record["key"],
        "task_id": record["task_id"],
        "bucket": record["bucket"],
        "arm": record["arm"],
        "setting": record["setting"],
        "wall_s": record["wall_s"],
        "search_s": sum(i["t1"] - i["t0"] for i in searches),
        "searches_ferret": sum(i["type"] == "mcp_tool_call" for i in searches),
        "searches_shell": sum(i["type"] == "command_execution" for i in searches),
        "lists": sum(i["class"] == "list" for i in items),
        "reads": sum(i["class"] == "read" for i in items),
        "edits": sum(i["class"] == "edit" for i in items),
        "tokens": usage["input_tokens"] + usage["output_tokens"],
        "cached_tokens": usage["cached_input_tokens"],
        "uncached_input_tokens": usage["input_tokens"] - usage["cached_input_tokens"],
        "output_tokens": usage["output_tokens"],
        "success5": bool(files) and file_success(files, task["gold_paths"]),
        "rr": file_reciprocal_rank(files, task["gold_paths"]),
        "func10": (
            function_success(functions, task["gold_functions"])
            if task.get("gold_functions")
            else math.nan
        ),
        "edited_gold": bool(changed & set(task["gold_paths"])),
        "answered": bool(record.get("answer")),
        "timed_out": record["timed_out"],
        "slept_s": record.get("slept_s", 0.0),
    }


def load(runs: Sequence[Path], tasks: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        for line in (run / "episodes.jsonl").read_text().splitlines():
            if line:
                record = json.loads(line)
                if "infra_error" not in record:
                    rows.append(episode_metrics(record, tasks[record["task_id"]]))
    frame = pd.DataFrame(rows)
    # A rerun replaces the earlier record of its key.
    return frame.drop_duplicates("key", keep="last").reset_index(drop=True)


def paired(frame: pd.DataFrame, setting: str) -> pd.DataFrame:
    sub = frame[frame.setting == setting]
    return sub.pivot(index="task_id", columns="arm").dropna(
        subset=[("wall_s", "F"), ("wall_s", "R")]
    )


def ratio_ci(
    f: np.ndarray, r: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float, float]:
    """Ratio of geometric means F/R, its bootstrap 95% interval and p-value."""
    d = np.log(f) - np.log(r)
    idx = rng.integers(0, len(d), size=(BOOTSTRAP, len(d)))
    boots = d[idx].mean(axis=1)
    p = min(1.0, 2 * min((boots >= 0).mean(), (boots <= 0).mean()))
    return (
        math.exp(d.mean()),
        math.exp(np.percentile(boots, 2.5)),
        math.exp(np.percentile(boots, 97.5)),
        p,
    )


def diff_ci(
    f: np.ndarray, r: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    d = f.astype(float) - r.astype(float)
    idx = rng.integers(0, len(d), size=(BOOTSTRAP, len(d)))
    boots = d[idx].mean(axis=1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted


def report(frame: pd.DataFrame) -> str:
    lines = [
        "## Arms",
        "",
        "| Setting | Bucket | Arm | Episodes | Success@5 | Func@10 | Median wall s | "
        "Median search s | Searches (Ferret / shell) | Median tokens | Timeouts |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (setting, bucket, arm), g in frame.groupby(["setting", "bucket", "arm"]):
        lines.append(
            f"| {setting} | {bucket} | {arm} | {len(g)} | {g.success5.mean():.0%} | "
            f"{g.func10.mean():.0%} | {g.wall_s.median():.0f} | "
            f"{g.search_s.median():.1f} | "
            f"{g.searches_ferret.mean():.1f} / {g.searches_shell.mean():.1f} | "
            f"{g.tokens.median() / 1000:.0f}k | {int(g.timed_out.sum())} |"
        )
    rng = np.random.default_rng(SEED)
    lines += [
        "",
        "## F against R, paired by task",
        "",
        "| Setting | Endpoint | Tasks | F/R (or F-R) | 95% CI | p | Holm p |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    primary, rows = {}, []
    for setting in sorted(frame.setting.unique()):
        p = paired(frame, setting)
        n = len(p)
        if n < 2:
            continue
        for name, f, r in [
            ("search time", 1 + p[("search_s", "F")], 1 + p[("search_s", "R")]),
            ("wall time", p[("wall_s", "F")], p[("wall_s", "R")]),
            ("tokens", p[("tokens", "F")], p[("tokens", "R")]),
        ]:
            est, lo, hi, pv = ratio_ci(f.to_numpy(float), r.to_numpy(float), rng)
            rows.append((setting, name, n, f"{est:.2f}x", f"[{lo:.2f}, {hi:.2f}]", pv))
            if name != "tokens":
                primary[f"{setting} {name}"] = pv
        est, lo, hi = diff_ci(
            p[("success5", "F")].to_numpy(), p[("success5", "R")].to_numpy(), rng
        )
        rows.append(
            (
                setting,
                "success@5",
                n,
                f"{est:+.0%}",
                f"[{lo:+.0%}, {hi:+.0%}]",
                math.nan,
            )
        )
    adjusted = holm(primary)
    for setting, name, n, est, ci, pv in rows:
        h = adjusted.get(f"{setting} {name}")
        p_text = "" if math.isnan(pv) else f"{pv:.4f}"
        holm_text = "" if h is None else f"{h:.4f}"
        lines.append(
            f"| {setting} | {name} | {n} | {est} | {ci} | {p_text} | {holm_text} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, action="append", required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args(argv)
    tasks = {}
    for path in args.tasks:
        for line in path.read_text().splitlines():
            if line:
                task = json.loads(line)
                tasks[task["id"]] = task
    frame = load(args.runs, tasks)
    if args.csv:
        frame.to_csv(args.csv, index=False)
    print(report(frame), end="")


if __name__ == "__main__":
    main()

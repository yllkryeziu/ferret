# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Build times and `systems.json` for the break-even model.

    uv run python -m harness.systems build --ferret FERRET --run RUN \\
        --snapshots EVAL/snapshots --out DATA [--snapshot SHA ...]
    uv run python -m harness.systems measure --replay replay.csv \\
        --builds DATA/builds.json --tasks tasks.jsonl --out systems.json

`build` makes a data directory that `harness.replay --data` can use: its
`snapshots/<sha>` are symlinks to the evaluation's snapshots, and its
`indexes/<sha>.fseg` are built here, by the binary being measured, so the
build times and the replayed searches come from the same code. Each index is
built twice from scratch. The second build is the warm one: the same page
cache state as the warm query timings. `builds.json` records, per snapshot,
the number of regular files N (`.git` excluded), both build times, and the
binary's SHA-256.

`measure` turns the replay CSV and `builds.json` into one `RepoMeasurement`
per snapshot: t_rg is the median ripgrep latency and t_idx the median raw
Ferret latency over the calls ripgrep answered (status 0 or 1), as the same
work with the same process start. Zoekt's build time is not in the model.
"""

import argparse
import csv
import dataclasses
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from analysis.breakeven import RepoMeasurement
from harness.replay import file_sha256, iter_search_calls
from tasks.prepare import index_path, snapshot_dir
from tasks.schema import Bucket, Task, read_tasks

logger = logging.getLogger(__name__)

SMALL = Bucket.S.value


def count_regular_files(root: Path) -> int:
    """Returns the regular files under `root`, without following symlinks
    and without the top-level `.git`."""
    total = 0
    pending = [root]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    if not (entry.name == ".git" and Path(entry.path).parent == root):
                        pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += 1
    return total


def timed_index(ferret: Path, root: Path, index: Path) -> float:
    """Builds a fresh index of `root`, deleting any earlier one first, and
    returns the wall seconds `ferret index` took."""
    index.unlink(missing_ok=True)
    index.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    subprocess.run(
        [str(ferret), "index", f"--root={root}", f"--index={index}"],
        check=True,
        capture_output=True,
    )
    return time.perf_counter() - start


def build_indexes(
    ferret: Path, snapshots_source: Path, snapshots: Sequence[str], data: Path
) -> dict[str, Any]:
    """Builds `data/snapshots/<sha>` links and `data/indexes/<sha>.fseg`,
    and returns the contents of `builds.json`.

    Raises:
        FileNotFoundError: if a snapshot is missing from `snapshots_source`.
    """
    result: dict[str, Any] = {
        "ferret": {"path": str(ferret.resolve()), "sha256": file_sha256(ferret)},
        "snapshots": {},
    }
    for snapshot in snapshots:
        source = snapshots_source / snapshot
        if not source.is_dir():
            raise FileNotFoundError(source)
        root = snapshot_dir(data, snapshot)
        root.parent.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or root.exists():
            root.unlink()
        root.symlink_to(source.resolve(), target_is_directory=True)
        index = index_path(data, snapshot)
        cold = timed_index(ferret, root, index)
        warm = timed_index(ferret, root, index)
        result["snapshots"][snapshot] = {
            "files": count_regular_files(root),
            "build_cold_seconds": cold,
            "build_seconds": warm,
            "index_bytes": index.stat().st_size,
        }
        logger.info("built %s: cold %.2f s, warm %.2f s", snapshot, cold, warm)
    return result


def measurement_name(task: Task) -> str:
    """The name of a snapshot in the break-even table: "repo@sha7" for the
    small repositories, and the repository's own name otherwise."""
    if task.bucket.value == SMALL:
        return f"{task.repo}@{task.snapshot[:7]}"
    return task.repo


def measure(
    rows: Sequence[dict[str, str]],
    builds: dict[str, Any],
    tasks: Sequence[Task],
) -> list[RepoMeasurement]:
    """Returns one `RepoMeasurement` per replayed snapshot, in build order.

    Raises:
        ValueError: if a replayed snapshot has no build, no task, or no call
            that ripgrep answered.
    """
    task_of = {task.snapshot: task for task in tasks}
    measurements = []
    for snapshot, build in builds["snapshots"].items():
        calls = [
            row
            for row in rows
            if row["snapshot"] == snapshot and row["rg_status"] in ("0", "1")
        ]
        if not calls:
            continue
        if snapshot not in task_of:
            raise ValueError(f"no task has snapshot {snapshot}")
        task = task_of[snapshot]
        measurements.append(
            RepoMeasurement(
                name=measurement_name(task),
                files=int(build["files"]),
                build_seconds=float(build["build_seconds"]),
                rg_seconds=statistics.median(float(r["rg_ms"]) for r in calls) / 1000,
                index_seconds=statistics.median(
                    float(r["ferret_raw_ms"]) for r in calls
                )
                / 1000,
                bucket=task.bucket.value,
            )
        )
    replayed = {row["snapshot"] for row in rows}
    missing = replayed - builds["snapshots"].keys()
    if missing:
        raise ValueError(f"replayed snapshots without a build: {sorted(missing)[:3]}")
    return measurements


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the process exit status."""
    parser = argparse.ArgumentParser(prog="python -m harness.systems")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build the indexes and builds.json")
    build.add_argument("--ferret", type=Path, required=True)
    build.add_argument("--run", type=Path, required=True, help="run to replay")
    build.add_argument("--snapshots", type=Path, required=True, help="source dir")
    build.add_argument("--out", type=Path, required=True, help="data directory")
    build.add_argument("--snapshot", action="append", default=[], metavar="SHA")
    report = commands.add_parser("measure", help="write systems.json")
    report.add_argument("--replay", type=Path, required=True, help="replay CSV")
    report.add_argument("--builds", type=Path, required=True)
    report.add_argument("--tasks", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.command == "build":
        snapshots = sorted({call.snapshot for call in iter_search_calls(args.run)})
        if args.snapshot:
            snapshots = [s for s in args.snapshot if s in snapshots]
        document = build_indexes(args.ferret, args.snapshots, snapshots, args.out)
        (args.out / "builds.json").write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
        return 0
    with args.replay.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    builds = json.loads(args.builds.read_text(encoding="utf-8"))
    measurements = measure(rows, builds, read_tasks(args.tasks))
    args.out.write_text(
        json.dumps([dataclasses.asdict(m) for m in measurements], indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

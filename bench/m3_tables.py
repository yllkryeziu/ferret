# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Renders the worktree and edit benchmarks (bench/README.md, 3 and 4).

    python -m bench.m3_tables g3 --ferret-log G3.log --zoekt ZOEKT.jsonl \\
        [--chain chain.json] --out g3.md
    python -m bench.m3_tables freshness --ferret-log FRESH.log \\
        --zoekt FRESH.jsonl --out fresh.md --csv fresh.csv --png fresh.png

The inputs are the logs of `workspace_benchmark --g3 | --freshness`
and the JSON lines of `zoekt-bench branches` and `zoekt-bench freshness`;
bench/run_m3.sh runs them. A table is refused (SystemExit) when its input is
incomplete, so a partial run never renders as a result.
"""

import argparse
import csv
import json
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

KV = re.compile(r"(\w+)=(\S+)")


def percentile(values: Sequence[float], pct: int) -> float:
    """Nearest rank over the sorted values, as the C++ benchmarks compute it."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, len(ordered) * pct // 100)]


def parse_kv(text: str) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for key, value in KV.findall(text):
        try:
            out[key] = float(value) if "." in value else int(value)
        except ValueError:
            out[key] = value
    return out


# --- G3 -------------------------------------------------------------------


def parse_g3_log(text: str) -> list[dict]:
    """The `G3 <label>: key=value ...` lines of workspace_benchmark --g3."""
    rows = []
    for line in text.splitlines():
        match = re.fullmatch(r"G3 (\S+): (.+)", line)
        if match:
            rows.append({"label": match[1], **parse_kv(match[2])})
    return rows


def parse_jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def mib(value: float) -> str:
    return f"{value / (1 << 20):.1f}"


def render_g3(ferret: list[dict], zoekt: list[dict], chain: dict | None = None) -> str:
    """The G3 table: Ferret per worktree, Zoekt per branch count.

    Ferret's rows are k = 1..K in order, plus a `same` row: one more worktree
    at the first commit. `chain` is the manifest of bench.synth_commits: the
    k-th worktree's diff to the previous one is chain["commits"][k-1].
    """
    numbered = [r for r in ferret if re.fullmatch(r"k\d+", r["label"])]
    same = [r for r in ferret if r["label"] == "same"]
    if not numbered:
        raise SystemExit("no k rows in the Ferret G3 log")
    expected = [f"k{i}" for i in range(1, len(numbered) + 1)]
    if [r["label"] for r in numbered] != expected:
        raise SystemExit(f"G3 rows are not k=1..{len(numbered)} in order")
    if zoekt and [r["k"] for r in zoekt] != list(range(1, len(zoekt) + 1)):
        raise SystemExit("Zoekt rows are not k=1..K in order")
    diffs = {}
    if chain:
        for commit in chain["commits"]:
            diffs[commit["k"]] = commit["bytes"]
    lines = [
        "# G3: sharing for k = 1..K worktrees",
        "",
        "## Ferret: registering the k-th worktree into one index",
        "",
        "| Worktree | Diff to k-1 (files / MiB) | Wall (ms) | Scan (ms) | Read (ms)"
        " | Ingest (ms) | CPU (ms) | Bytes read (MiB) | Segments written (MiB)"
        " | Net index change (MiB)"
        " | Flush + compaction (ms) | Segments |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    per_step = chain["files_per_step"] if chain else None

    def ferret_row(name: str, row: dict, diff: str) -> str:
        return (
            f"| {name} | {diff} | {row['wall_ms']} | {row['scan_ms']} "
            f"| {row['read_ms']} | {row['ingest_ms']} | {row['cpu_ms']} "
            f"| {mib(row['bytes_read'])} | {mib(row['segment_bytes_written'])} "
            f"| {mib(row['index_bytes_added'])} "
            f"| {row['flush_ms'] + row['compaction_wait_ms']} | {row['segments']} |"
        )

    for i, row in enumerate(numbered, start=1):
        if chain and i > 1:
            diff = f"{per_step} / {mib(diffs.get(i, 0))}"
        elif chain:
            diff = "(the base tree)"
        else:
            diff = "n/a"
        lines.append(ferret_row(f"k={i}", row, diff))
    for row in same:
        lines.append(ferret_row("same commit as k=1", row, "0 / 0.0"))
    if same and numbered:
        written = same[0]["segment_bytes_written"]
        base = numbered[0]["segment_bytes_written"]
        lines += [
            "",
            f"A worktree at the commit of k=1 writes {written} bytes of "
            f"segments, {100 * written / max(base, 1):.3f}% of the first "
            f"registration's {base} bytes (net change of the published index "
            f"after compaction: {same[0]['index_bytes_added']} bytes).",
        ]
    if chain and len(numbered) > 2:
        later = numbered[1:]
        ratios = [
            r["segment_bytes_written"] / max(diffs.get(i, 0), 1)
            for i, r in enumerate(later, start=2)
        ]
        first = numbered[0]
        per_tree_byte = first["segment_bytes_written"] / max(first["bytes_ingested"], 1)
        lines += [
            "",
            "Segment bytes written per byte of changed file content (k >= 2): "
            f"min {min(ratios):.2f}, max {max(ratios):.2f}; the first "
            f"registration writes {per_tree_byte:.2f} per byte of its tree.",
        ]
    if zoekt:
        lines += [
            "",
            "## Zoekt multi-branch: one index of the first k branches",
            "",
            "Each row is a fresh full build of the first k branches into one"
            " index (git indexer rules: one document per distinct path and"
            " blob, listing its branches).",
            "",
            "| k | Documents | Added docs | Index (MiB) | Added (MiB) |"
            " Build wall (s) | Build CPU (s) | Unshared content (MiB) |"
            " Shared content (MiB) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        previous_docs = 0
        previous_bytes = 0
        for row in zoekt:
            lines.append(
                f"| {row['k']} | {row['docs']} | {row['docs'] - previous_docs} "
                f"| {mib(row['index_bytes'])} "
                f"| {mib(row['index_bytes'] - previous_bytes)} "
                f"| {row['wall_seconds']:.1f} | {row['cpu_seconds']:.1f} "
                f"| {mib(row['bytes_if_unshared'])} | {mib(row['bytes_indexed'])} |"
            )
            previous_docs = row["docs"]
            previous_bytes = row["index_bytes"]
    return "\n".join(lines) + "\n"


# --- Freshness --------------------------------------------------------------

FERRET_STAGES = {
    "ferret_notify": ("notify_us", "apply_us", "search_us"),
    "ferret_watcher": ("event_us", "apply_us", "search_us"),
}
ZOEKT_STAGES = ("commit_ns", "build_ns", "reload_ns")
STAGE_LABELS = {
    "ferret_notify": (
        "Notify call",
        "wait for the generation",
        "search that sees it",
    ),
    "ferret_watcher": (
        "write to Ferret's callback (FSEvents)",
        "callback to applied",
        "applied to seen by the poll",
    ),
    "zoekt": ("git commit", "delta build", "reload and search"),
}
PATH_NAMES = {
    "ferret_notify": "Ferret Notify",
    "ferret_watcher": "Ferret watcher",
    "zoekt": "Zoekt commit + delta + reload",
}


def parse_ferret_fresh(text: str) -> dict[str, list[dict]]:
    """`FRESH <path> trial=N total_us=... stage_us=...` lines, by path."""
    out: dict[str, list[dict]] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"FRESH (ferret_\w+) (.+)", line)
        if match:
            out.setdefault(match[1], []).append(parse_kv(match[2]))
    return out


def drop_warmup(rows: Sequence[dict], warmup: int) -> list[dict]:
    """The rows without the first `warmup` trials, by trial number.

    The first trial of a path pays cold page caches and, for Zoekt, the
    first `git commit` in a fresh clone (an index refresh of every file).
    """
    if warmup < 0:
        raise SystemExit("--warmup must be at least 0")
    ordered = sorted(rows, key=lambda r: int(r["trial"]))
    if len(ordered) <= warmup:
        raise SystemExit(f"{len(ordered)} trials leave none after {warmup} warm-up")
    return ordered[warmup:]


def freshness_series(
    ferret: dict[str, list[dict]], zoekt: list[dict], warmup: int = 0
) -> dict[str, dict[str, list[float]]]:
    """Per path, the total and each stage in milliseconds, one value per trial.

    The first `warmup` trials of each path are left out.
    """
    series: dict[str, dict[str, list[float]]] = {}
    for path, all_rows in ferret.items():
        if path not in FERRET_STAGES:
            raise SystemExit(f"unknown Ferret path {path}")
        rows = drop_warmup(all_rows, warmup)
        stages = FERRET_STAGES[path]
        entry = {"total": [r["total_us"] / 1000 for r in rows]}
        for key, label in zip(stages, STAGE_LABELS[path], strict=True):
            entry[label] = [r[key] / 1000 for r in rows]
        series[path] = entry
    if zoekt:
        rows = drop_warmup(zoekt, warmup)
        entry = {"total": [r["total_ns"] / 1e6 for r in rows]}
        for key, label in zip(ZOEKT_STAGES, STAGE_LABELS["zoekt"], strict=True):
            entry[label] = [r[key] / 1e6 for r in rows]
        series["zoekt"] = entry
    return series


def warmup_totals(
    ferret: dict[str, list[dict]], zoekt: list[dict], warmup: int
) -> dict[str, list[float]]:
    """Per path, the totals in milliseconds of the trials left out as warm-up."""
    out: dict[str, list[float]] = {}
    for path, rows in ferret.items():
        ordered = sorted(rows, key=lambda r: int(r["trial"]))[:warmup]
        out[path] = [r["total_us"] / 1000 for r in ordered]
    if zoekt:
        ordered = sorted(zoekt, key=lambda r: int(r["trial"]))[:warmup]
        out["zoekt"] = [r["total_ns"] / 1e6 for r in ordered]
    return out


def cdf_points(values: Iterable[float]) -> list[tuple[float, float]]:
    """(value, fraction of trials at or below it), ascending; ends at 1.0."""
    ordered = sorted(values)
    n = len(ordered)
    return [(v, (i + 1) / n) for i, v in enumerate(ordered)]


def write_cdf_csv(path: Path, series: dict[str, dict[str, list[float]]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "latency_ms", "cumulative_fraction"])
        for name in PATH_NAMES:
            if name in series:
                for value, fraction in cdf_points(series[name]["total"]):
                    writer.writerow([name, f"{value:.4f}", f"{fraction:.6f}"])


def plot_cdf(path: Path, series: dict[str, dict[str, list[float]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, label in PATH_NAMES.items():
        if name not in series:
            continue
        points = cdf_points(series[name]["total"])
        ax.step(
            [p[0] for p in points],
            [p[1] for p in points],
            where="post",
            label=f"{label} (n={len(points)})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("edit to searchable (ms, log scale)")
    ax.set_ylabel("fraction of edits")
    ax.set_ylim(0, 1.01)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_freshness(
    series: dict[str, dict[str, list[float]]],
    warmup: dict[str, list[float]] | None = None,
) -> str:
    if not series:
        raise SystemExit("no freshness trials")
    lines = ["# Freshness: edit to searchable", ""]
    if warmup and any(warmup.values()):
        dropped = "; ".join(
            f"{PATH_NAMES[name]} " + ", ".join(f"{v:.2f}" for v in warmup[name]) + " ms"
            for name in PATH_NAMES
            if warmup.get(name)
        )
        count = max(len(v) for v in warmup.values())
        lines += [
            f"The first {count} trial(s) of each path are warm-up and left out"
            f" of every row below: {dropped}.",
            "",
        ]
    lines += [
        "| Path | n | p50 (ms) | p90 (ms) | p99 (ms) | max (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in PATH_NAMES.items():
        if name not in series:
            continue
        values = series[name]["total"]
        lines.append(
            f"| {label} | {len(values)} | {percentile(values, 50):.2f} "
            f"| {percentile(values, 90):.2f} | {percentile(values, 99):.2f} "
            f"| {max(values):.2f} |"
        )
    lines += [
        "",
        "## Stages (p50 / p99, ms)",
        "",
        "| Path | Stage | p50 | p99 | max |",
        "|---|---|---:|---:|---:|",
    ]
    for name, label in PATH_NAMES.items():
        if name not in series:
            continue
        for stage, values in series[name].items():
            if stage == "total":
                continue
            lines.append(
                f"| {label} | {stage} | {percentile(values, 50):.2f} "
                f"| {percentile(values, 99):.2f} | {max(values):.2f} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    g3 = sub.add_parser("g3")
    g3.add_argument("--ferret-log", type=Path, required=True)
    g3.add_argument("--zoekt", type=Path)
    g3.add_argument("--chain", type=Path)
    g3.add_argument("--out", type=Path, required=True)
    fresh = sub.add_parser("freshness")
    fresh.add_argument("--ferret-log", type=Path, required=True)
    fresh.add_argument("--zoekt", type=Path)
    fresh.add_argument("--out", type=Path, required=True)
    fresh.add_argument("--csv", type=Path)
    fresh.add_argument("--png", type=Path)
    fresh.add_argument(
        "--warmup", type=int, default=0, help="first trials of each path to leave out"
    )
    args = parser.parse_args(argv)

    if args.command == "g3":
        rows = parse_g3_log(args.ferret_log.read_text())
        zoekt = parse_jsonl(args.zoekt.read_text()) if args.zoekt else []
        chain = json.loads(args.chain.read_text()) if args.chain else None
        text = render_g3(rows, zoekt, chain)
    else:
        zoekt = parse_jsonl(args.zoekt.read_text()) if args.zoekt else []
        ferret = parse_ferret_fresh(args.ferret_log.read_text())
        series = freshness_series(ferret, zoekt, args.warmup)
        text = render_freshness(series, warmup_totals(ferret, zoekt, args.warmup))
        if args.csv:
            write_cdf_csv(args.csv, series)
        if args.png:
            plot_cdf(args.png, series)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

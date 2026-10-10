# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Renders bench/run.sh's output as one page of tables and charts.

    python -m bench.report RESULTS_ROOT

RESULTS_ROOT holds one directory per corpus, each written by bench/run.sh.
The page (README.md) and its charts (*.png) are written to RESULTS_ROOT.
A step whose output is missing is left out of the page.
"""

import json
import sys
from pathlib import Path
from statistics import median

from bench import m3_tables

COLORS = {"Ferret": "#C2633F", "Zoekt": "#4E5763", "ripgrep": "#A39E93"}
ENGINE_NAMES = {"ferret": "Ferret", "zoekt": "Zoekt", "ripgrep": "ripgrep"}


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * pct // 100))
    return ordered[int(rank) - 1]


def ms(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f} ms"
    if value >= 10:
        return f"{value:.0f} ms"
    return f"{value:.1f} ms"


def seconds(value: float) -> str:
    return f"{value:,.0f} s" if value >= 10 else f"{value:.1f} s"


def size(value: float) -> str:
    if value == 0:
        return "0 bytes"
    if value >= 1 << 30:
        return f"{value / (1 << 30):.1f} GiB"
    return f"{value / (1 << 20):,.0f} MiB"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def machine(directory: Path) -> dict[str, str]:
    out = {}
    for line in (directory / "machine.txt").read_text().splitlines():
        key, _, value = line.partition(": ")
        out[key] = value
    return out


def corpus_name(directory: Path) -> str:
    return machine(directory)["corpus"].split(" at ")[0]


# --- 1. Speed --------------------------------------------------------------


def speed(directory: Path) -> dict | None:
    """Per engine: per-query milliseconds, and the queries whose matches
    equal ripgrep's. A query's time is the median of its runs."""
    if not (directory / "speed-zoekt.jsonl").exists():
        return None
    rows = read_jsonl(directory / "speed.jsonl")
    runs: dict[str, dict[str, list[float]]] = {}
    digests: dict[str, dict[str, str]] = {}
    for row in rows:
        if "register" in row:
            continue
        if "error" in row:
            raise SystemExit(
                f"{directory}: {row['engine']} {row['query']}: {row['error']}"
            )
        runs.setdefault(row["engine"], {}).setdefault(row["query"], []).append(
            row["us"] / 1000
        )
        digests.setdefault(row["engine"], {})[row["query"]] = row["digest"]
    for row in read_jsonl(directory / "speed-zoekt.jsonl"):
        runs.setdefault("zoekt", {})[row["id"]] = [t / 1e6 for t in row["latencies_ns"]]
        digests.setdefault("zoekt", {})[row["id"]] = row["result_digest"]
    times = {e: {q: median(v) for q, v in per.items()} for e, per in runs.items()}
    reference = digests["ripgrep"]
    same = {
        e: sum(digests[e].get(q) == d for q, d in reference.items())
        for e in ("ferret", "zoekt")
    }
    zoekt_build = json.loads((directory / "speed-zoekt-build.json").read_text())
    return {
        "times": times,
        "same": same,
        "queries": len(reference),
        "files": zoekt_build["files_listed"],
        "index_bytes": {
            "ferret": int((directory / "speed-ferret-index-bytes").read_text()),
            "zoekt": zoekt_build["index_bytes"],
        },
    }


def speed_table(name: str, s: dict) -> str:
    """Ferret's search times are without the wait for recent edits, like
    Zoekt's; one row adds the wait that Ferret's server makes by default."""
    t = s["times"]
    search = ("ferret_nowait", "zoekt", "ripgrep")
    n = s["queries"]
    rows = [
        (name, "Ferret", "Zoekt", "ripgrep"),
        ("Median search", *(ms(median(t[e].values())) for e in search)),
        ("95th percentile", *(ms(percentile(list(t[e].values()), 95)) for e in search)),
        (
            "Median, first waiting until recent edits are visible",
            ms(median(t["ferret"].values())),
            "never sees uncommitted edits",
            "always reads the files",
        ),
        (
            "Same matches as ripgrep",
            f"{s['same']['ferret']} of {n}",
            f"{s['same']['zoekt']} of {n}",
            "",
        ),
        (
            "Index size (build times: see 3)",
            size(s["index_bytes"]["ferret"]),
            size(s["index_bytes"]["zoekt"]),
            "no index",
        ),
    ]
    return markdown_table(rows)


def markdown_table(rows: list[tuple[str, ...]]) -> str:
    """The first row is the header; the first column is left-aligned."""
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    lines.insert(1, "|---|" + "---:|" * (len(rows[0]) - 1))
    return "\n".join(lines) + "\n"


def plot_speed(path: Path, results: list[tuple[str, dict]]) -> None:
    plt = pyplot()
    fig, axes = plt.subplots(
        len(results), 1, figsize=(8, 2.3 * len(results)), squeeze=False
    )
    for ax, (name, s) in zip(axes[:, 0], results, strict=True):
        for row, engine in enumerate(("ripgrep", "zoekt", "ferret_nowait")):
            values = sorted(s["times"][engine].values())
            label = ENGINE_NAMES[engine.removesuffix("_nowait")]
            jitter = [((i * 7919) % 101 / 101 - 0.5) * 0.45 for i in range(len(values))]
            ax.scatter(
                values, [row + j for j in jitter], s=7, alpha=0.55, color=COLORS[label]
            )
            mid = median(values)
            ax.plot([mid, mid], [row - 0.32, row + 0.32], color="#2A2926", lw=2)
            ax.annotate(
                ms(mid),
                (mid, row + 0.34),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#2A2926",
            )
        ax.set_yticks(range(3), ["ripgrep", "Zoekt", "Ferret"])
        ax.set_ylim(-0.6, 2.75)
        ax.set_xscale("log")
        plain_log_ticks(ax.xaxis)
        ax.set_title(
            f"{name}: one dot per search, line at the median", fontsize=10, loc="left"
        )
        ax.grid(True, axis="x", which="major", alpha=0.3)
    axes[-1, 0].set_xlabel("time to return every match (ms, log scale)")
    save(fig, path)


# --- 2. Agents at once -------------------------------------------------------


def agents(directory: Path) -> dict | None:
    """Per engine and agent count: every search's milliseconds."""
    path = directory / "agents.jsonl"
    if not path.exists():
        return None
    out: dict[str, dict[int, list[float]]] = {}
    for row in read_jsonl(path):
        if "register" in row:
            continue
        if "error" in row:
            raise SystemExit(
                f"{directory}: {row['engine']} {row['query']}: {row['error']}"
            )
        out.setdefault(row["engine"], {}).setdefault(row["agents"], []).append(
            row["us"] / 1000
        )
    return out


def agents_table(name: str, a: dict) -> str:
    counts = sorted(a["ripgrep"])
    rows = [(f"{name}: agents at once", *(str(n) for n in counts))]
    for engine in ("ferret", "ripgrep"):
        cells = (ms(median(a[engine][n])) for n in counts)
        rows.append((f"{ENGINE_NAMES[engine]}, median search", *cells))
    return markdown_table(rows)


def plot_agents(path: Path, name: str, a: dict) -> None:
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for engine in ("ripgrep", "ferret"):
        counts = sorted(a[engine])
        values = [median(a[engine][n]) for n in counts]
        label = ENGINE_NAMES[engine]
        ax.plot(counts, values, marker="o", color=COLORS[label], label=label, lw=2)
        for n, v in zip(counts, values, strict=True):
            ax.annotate(
                ms(v),
                (n, v),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks(counts, [str(n) for n in counts])
    ax.set_yscale("log")
    plain_log_ticks(ax.yaxis)
    ax.set_xlabel("agents searching at once, each in its own worktree")
    ax.set_ylabel("median search (ms, log scale)")
    ax.set_title(
        f"{name}: every match, each agent its own queries", fontsize=10, loc="left"
    )
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="center right")
    save(fig, path)


# --- 3. Worktrees ------------------------------------------------------------


def worktrees(directory: Path) -> dict | None:
    """Per worktree k: Ferret's time and index growth to make it searchable,
    and Zoekt's rebuild of the first k branches."""
    if not (directory / "g3-zoekt.jsonl").exists():
        return None
    ferret = {
        r["label"]: r
        for r in m3_tables.parse_g3_log((directory / "g3-ferret.log").read_text())
    }
    zoekt = sorted(read_jsonl(directory / "g3-zoekt.jsonl"), key=lambda r: r["k"])
    chain = json.loads((directory / "g3-chain.json").read_text())
    rows = []
    previous = 0
    for z, commit in zip(zoekt, chain["commits"], strict=True):
        f = ferret[f"k{z['k']}"]
        rows.append(
            {
                "k": z["k"],
                "files": commit["files"],
                "ferret_s": f["wall_ms"] / 1000,
                "ferret_bytes": f["index_bytes_added"],
                "zoekt_s": z["wall_seconds"],
                "zoekt_bytes": z["index_bytes"] - previous,
            }
        )
        previous = z["index_bytes"]
    same = ferret["same"]
    return {
        "rows": rows,
        "same_s": same["wall_ms"] / 1000,
        "same_bytes": same["index_bytes_added"],
    }


def worktrees_table(name: str, w: dict) -> str:
    first, rest = w["rows"][0], w["rows"][1:]
    rows = [
        (
            name,
            "Ferret: time",
            "Ferret: disk added",
            "Zoekt: time",
            "Zoekt: disk added",
        ),
        (
            "First worktree (the whole repository)",
            seconds(first["ferret_s"]),
            size(first["ferret_bytes"]),
            seconds(first["zoekt_s"]),
            size(first["zoekt_bytes"]),
        ),
    ]
    if rest:
        files = int(median(r["files"] for r in rest))
        label = f"Each further worktree, {files:,} files changed"
        rows.append(
            (
                f"{label} (median of {len(rest)})",
                seconds(median(r["ferret_s"] for r in rest)),
                size(median(r["ferret_bytes"] for r in rest)),
                seconds(median(r["zoekt_s"] for r in rest)),
                size(median(r["zoekt_bytes"] for r in rest)),
            )
        )
    rows.append(
        (
            "A worktree at the same commit as the first",
            seconds(w["same_s"]),
            size(w["same_bytes"]),
            "",
            "",
        )
    )
    return markdown_table(rows)


def plot_worktrees(path: Path, name: str, w: dict) -> None:
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ks = [r["k"] for r in w["rows"]]
    for label, key in (("Zoekt", "zoekt_s"), ("Ferret", "ferret_s")):
        values = [r[key] for r in w["rows"]]
        ax.plot(ks, values, marker="o", color=COLORS[label], label=label, lw=2)
    ax.set_yscale("log")
    plain_log_ticks(ax.yaxis)
    ax.set_xticks(ks)
    ax.set_xlabel("worktree (each at a commit that changes a few hundred files)")
    ax.set_ylabel("time until searchable (s, log scale)")
    ax.set_title(f"{name}: adding worktrees one at a time", fontsize=10, loc="left")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="center right")
    save(fig, path)


# --- 4. Edits ----------------------------------------------------------------


def edits(directory: Path) -> dict | None:
    """Milliseconds from writing a file until a search finds the edit."""
    if not (directory / "freshness-zoekt.jsonl").exists():
        return None
    series = m3_tables.freshness_series(
        m3_tables.parse_ferret_fresh((directory / "freshness-ferret.log").read_text()),
        read_jsonl(directory / "freshness-zoekt.jsonl"),
        warmup=1,
    )
    return {
        "Ferret": series["ferret_watcher"]["total"],
        "Zoekt": series["zoekt"]["total"],
    }


def edits_table(name: str, e: dict) -> str:
    def cells(values: list[float]) -> tuple[str, str]:
        return ms(median(values)), ms(percentile(values, 99))

    return markdown_table(
        [
            (f"{name}: edit until a search finds it", "Median", "99th percentile"),
            ("Ferret (sees the file change)", *cells(e["Ferret"])),
            ("Zoekt (commit, update the index, reload)", *cells(e["Zoekt"])),
            ("ripgrep (reads the files at every search)", "0 ms", "0 ms"),
        ]
    )


def plot_edits(path: Path, name: str, e: dict) -> None:
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7, 2.4))
    labels = ["ripgrep", "Zoekt", "Ferret"]
    mids = [0.0, median(e["Zoekt"]), median(e["Ferret"])]
    highs = [0.0, percentile(e["Zoekt"], 99), percentile(e["Ferret"], 99)]
    ax.barh(labels, mids, color=[COLORS[x] for x in labels], height=0.55)
    for y, (mid, high) in enumerate(zip(mids, highs, strict=True)):
        if high:
            ax.plot([mid, high], [y, y], color="#2A2926", lw=1.5)
        text = ms(mid) if mid else "0 ms: no index"
        ax.annotate(
            text,
            (max(mid, high), y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, max(highs) * 1.25)
    ax.set_xlabel(
        "edit until a search finds it (ms): median, line to the 99th percentile"
    )
    ax.set_title(f"{name}: searching your own edit", fontsize=10, loc="left")
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, path)


# --- Page ----------------------------------------------------------------


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    return plt


def plain_log_ticks(axis) -> None:
    """Labels a log axis 0.1, 1, 10, 1,000 instead of powers of ten."""
    from matplotlib.ticker import FuncFormatter

    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,g}"))


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)


def render(root: Path) -> str:
    corpora = sorted(p for p in root.iterdir() if (p / "machine.txt").exists())
    if not corpora:
        raise SystemExit(f"no corpus directories under {root}")
    if slept := sorted(root.glob("*/*.SLEPT")):
        raise SystemExit(f"the machine slept during a step: {slept[0]}")
    info = machine(corpora[0])
    out = [
        "# Benchmarks\n",
        f"Ferret {info['ferret']} on a {info['machine']}, {info['power']}; "
        f"{info['ripgrep']}; Zoekt {info['zoekt']}. Raw output: "
        + ", ".join(f"[{c.name}/]({c.name})" for c in corpora)
        + ". Rendered by `python -m bench.report`; bench/README.md has the method.\n",
    ]
    names = {c: corpus_name(c) for c in corpora}

    found = [(names[c], s) for c in corpora if (s := speed(c))]
    if found:
        out.append("## 1. How long does a search take?\n")
        out.append(
            f"{found[0][1]['queries']} searches per repository, each returning every "
            "matching line.\n"
        )
        out += [speed_table(n, s) for n, s in found]
        plot_speed(root / "speed.png", found)
        out.append("![Search times](speed.png)\n")
    for number, title, step, table, plot in (
        (
            2,
            "Does search slow down when agents search at once?",
            agents,
            agents_table,
            plot_agents,
        ),
        (
            3,
            "What does another worktree cost?",
            worktrees,
            worktrees_table,
            plot_worktrees,
        ),
        (4, "How soon can an agent find its own edit?", edits, edits_table, plot_edits),
    ):
        found = [(c, names[c], d) for c in corpora if (d := step(c))]
        if not found:
            continue
        out.append(f"## {number}. {title}\n")
        for c, n, d in found:
            out.append(table(n, d))
            png = f"{step.__name__}-{c.name}.png"
            plot(root / png, n, d)
            out.append(f"![{title}]({png})\n")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    root = Path(argv[0])
    (root / "README.md").write_text(render(root))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

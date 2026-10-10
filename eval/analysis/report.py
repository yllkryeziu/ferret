# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Renders the analysis outputs into the generated tables of a Markdown report.

The report keeps its prose by hand, but every number in it lives between a
pair of markers and is regenerated from `analysis.analyze` output:

    <!-- BEGIN GENERATED: contrasts -->
    ...replaced on every run...
    <!-- END GENERATED: contrasts -->

    uv run python -m analysis.report --results results/ --findings report.md
    uv run python -m analysis.report --results results/ \
        --findings report.md --check

The document chooses which blocks it shows (see `BLOCKS`); a marker naming
an unknown block, or a block whose input file is missing, is an error rather
than a silently stale table. --check exits non-zero when the document is not
what the results produce, which catches hand-edited numbers.
"""

import argparse
import math
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.bootstrap import paired_bootstrap, paired_effects
from analysis.breakeven import HOLDOUT_FACTOR
from analysis.multiplicity import holm
from analysis.power import variance_split
from analysis.protocol import CONTRASTS

BREAK_EVEN_WORKTREES = (1, 10)
POOLED = "all"
_BUCKET_ORDER = ("S", "L", "XL")

_BLOCK = re.compile(
    r"(<!-- BEGIN GENERATED: (?P<name>[a-z-]+) -->\n)"
    r"(?P<body>.*?)(<!-- END GENERATED: (?P=name) -->)",
    re.DOTALL,
)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Returns a Markdown table, right-aligning numeric-looking columns."""
    align = ["---:" if _numeric_column(rows, i) else "---" for i in range(len(headers))]
    lines = [headers, align, *rows]
    return "\n".join("| " + " | ".join(cells) + " |" for cells in lines) + "\n"


def percent(value: float) -> str:
    """Formats a proportion as a percentage, "41.3%"."""
    return "n/a" if math.isnan(value) else f"{100 * value:.1f}%"


def interval(low: float, high: float, scale: float = 100.0, digits: int = 1) -> str:
    """Formats a confidence interval, "[38.0, 44.9]"."""
    return f"[{scale * low:.{digits}f}, {scale * high:.{digits}f}]"


def p_value(value: float) -> str:
    """Formats a p-value to three decimals, or "<0.001"."""
    if math.isnan(value):
        return ""
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def dollars(value: float) -> str:
    """Formats dollars with four decimals; infinities read as "none solved"."""
    if math.isinf(value):
        return "none solved"
    return "n/a" if math.isnan(value) else f"${value:.4f}"


def render_arm_summary(results: Path) -> str:
    """Success, tokens, and tool use per bucket and arm."""
    table = pd.read_csv(results / "arm_summary.csv")
    rows = [
        [
            row.bucket,
            row.arm,
            str(row.tasks),
            (
                f"{percent(row.success)} "
                f"{interval(row.success_ci_low, row.success_ci_high)}"
            ),
            percent(row.function_success),
            f"{row.median_total_tokens / 1000:.1f}k",
            f"{row.mean_search_calls:.1f}",
            percent(row.invalid_call_rate),
            percent(row.exposure_rate),
        ]
        for row in table.itertuples()
    ]
    return markdown_table(
        ["Bucket", "Arm", "Tasks", "Success (Acc@5) [95% CI]", "Function Acc@10",
         "Median tokens", "Searches", "Invalid calls", "Gold exposed"],
        rows,
    )  # fmt: skip


def render_contrasts(results: Path) -> str:
    """The primary endpoint: ladder contrasts in success, Holm-adjusted when pooled."""
    return _contrast_table(results, "success", "all", "Δ success (pp) [95% CI]", 100.0)


def render_mechanism(results: Path) -> str:
    """Co-primary mechanism endpoints: restricted mean time to first gold exposure."""
    table = _contrasts(results)
    rows = []
    for metric, label, scale in (
        ("restricted_exposure_turns", "turns", 1.0),
        ("restricted_exposure_tokens", "k tokens", 1e-3),
    ):
        for row in table[
            (table.metric == metric) & (table.bucket == POOLED)
        ].itertuples():
            rows.append(
                [
                    label,
                    row.contrast,
                    (
                        f"{scale * row.estimate:+.2f} "
                        f"{interval(row.ci_low, row.ci_high, scale, 2)}"
                    ),
                    p_value(row.p_value),
                    p_value(row.p_holm),
                ]
            )
    return markdown_table(
        ["Time to gold", "Contrast", "Δ [95% CI]", "p", "p (Holm)"], rows
    )


def render_tokens(results: Path) -> str:
    """Token contrasts, estimated on the log scale and shown as percent change."""
    table = _contrasts(results)
    rows = [
        [
            row.bucket,
            row.contrast,
            f"{_percent_change(row.estimate)} "
            f"[{_percent_change(row.ci_low)}, {_percent_change(row.ci_high)}]",
            p_value(row.p_value),
            p_value(row.p_holm),
        ]
        for row in table[table.metric == "log_total_tokens"].itertuples()
    ]
    return markdown_table(
        ["Bucket", "Contrast", "Δ tokens [95% CI]", "p", "p (Holm)"], rows
    )


# The k = 3 pilot's split, which sized the main run. Its tool arms had no
# working search.
PILOT_VARIANCE = {"A2-A1": (0.62, 0.78), "A3-A2": (0.70, 0.92), "A4-A3": (0.74, 0.98)}


def render_variance(results: Path) -> str:
    """Paired log-token SD at k = 1 and its within-task share, next to the
    pilot values the sample size was chosen from."""
    frame = pd.read_csv(results / "episodes.csv")
    rows = []
    for baseline, arm in CONTRASTS:
        contrast = f"{arm}-{baseline}"
        split = variance_split(frame, arm, baseline)
        pilot_sd, pilot_within = PILOT_VARIANCE[contrast]
        rows.append(
            [
                contrast,
                str(split.tasks),
                f"{split.paired_sd:.2f}",
                f"{pilot_sd:.2f}",
                percent(split.within_fraction),
                percent(pilot_within),
            ]
        )
    return markdown_table(
        ["Contrast", "Tasks", "Paired SD", "Pilot SD", "Within-task share",
         "Pilot share"],
        rows,
    )  # fmt: skip


def render_sensitivity(results: Path) -> str:
    """The primary contrasts on the clean and search-needed subsets."""
    table = _contrasts(results)
    rows = [
        [
            row.subset,
            row.contrast,
            str(row.n_tasks),
            f"{100 * row.estimate:+.1f} {interval(row.ci_low, row.ci_high)}",
            p_value(row.p_holm),
        ]
        for row in table[
            (table.metric == "success")
            & (table.subset != "all")
            & (table.bucket == POOLED)
        ].itertuples()
    ]
    return markdown_table(
        ["Subset", "Contrast", "Tasks", "Δ success (pp) [95% CI]", "p (Holm)"], rows
    )


def render_glmm(results: Path) -> str:
    """GLMM posterior means with 95% intervals, on the logit scale."""
    table = pd.read_csv(results / "glmm.csv")
    rows = [
        [
            row.term,
            f"{row.mean:+.3f}",
            "" if math.isnan(row.sd) else interval(row.ci_low, row.ci_high, 1.0, 3),
        ]
        for row in table.itertuples()
    ]
    return markdown_table(["Term", "Posterior mean", "95% interval"], rows)


def render_contamination(results: Path) -> str:
    """Zero-tool success per bucket, the contamination probe."""
    table = pd.read_csv(results / "contamination.csv")
    rows = [
        [
            row.bucket,
            str(row.tasks),
            (
                f"{percent(row.a0_success)} "
                f"{interval(row.a0_success_ci_low, row.a0_success_ci_high)}"
            ),
            percent(row.solved_without_search),
        ]
        for row in table.itertuples()
    ]
    return markdown_table(
        ["Bucket", "Tasks", "A0 success [95% CI]", "Solved without search"], rows
    )


def render_costs(results: Path) -> str:
    """Dollars per episode and per solved task, and the change from A1."""
    table = pd.read_csv(results / "costs.csv")
    rows = [
        [
            row.bucket,
            row.arm,
            dollars(row.cost_per_episode),
            dollars(row.cost_per_solved),
            ""
            if math.isnan(row.delta_cost_per_solved)
            else f"{row.delta_cost_per_solved:+.4f}",
        ]
        for row in table.itertuples()
    ]
    return markdown_table(
        ["Bucket", "Arm", "$ / episode", "$ / solved", "Δ $ / solved vs A1"], rows
    )


def _queries(value: float) -> str:
    """Formats a break-even query count; infinity reads as "never"."""
    return "never" if math.isinf(value) else f"{value:,.0f}"


def _queries_range(values: pd.Series) -> str:
    """The median of `values`, with their range when there are several."""
    median = _queries(float(values.median()))
    if len(values) == 1:
        return median
    return (
        f"{median} ({_queries(float(values.min()))} to {_queries(float(values.max()))})"
    )


def render_replay(results: Path) -> str:
    """Per bucket: latency percentiles of each engine on the replayed
    searches, agreement of the match sets, and the projected wall time."""
    table = pd.read_csv(results / "replay_summary.csv")

    def latency(row: pd.Series, name: str) -> str:
        cells = [row[f"{name}_{p}_ms"] for p in ("p50", "p90", "p99")]
        return " / ".join("n/a" if math.isnan(c) else f"{c:,.1f}" for c in cells)

    def ratio(value: float) -> str:
        return "n/a" if math.isnan(value) else f"{value:.3f}"

    rows = [
        [
            row["bucket"],
            f"{row['calls_replayed']:,} of {row['calls_in_run']:,.0f}",
            latency(row, "ferret"),
            latency(row, "ferret_raw"),
            latency(row, "rg"),
            latency(row, "zoekt"),
            f"{row['rg_identical']:,} of {row['rg_comparable']:,}"
            + (f" ({row['rg_errors']:,} errors)" if row["rg_errors"] else ""),
            f"{row['zoekt_identical']:,} of {row['zoekt_comparable']:,}"
            + (f" ({row['zoekt_errors']:,} errors)" if row["zoekt_errors"] else ""),
            ratio(row["ratio_rg_over_ferret"]),
            ratio(row["ratio_zoekt_over_ferret"]),
            "n/a"
            if math.isnan(row["zoekt_build_seconds"])
            else f"{row['zoekt_build_seconds']:,.1f}",
        ]
        for _, row in table.iterrows()
    ]
    return markdown_table(
        [
            "Bucket",
            "Calls replayed",
            "Ferret as run, ms (p50 / p90 / p99)",
            "Ferret raw, ms",
            "ripgrep, ms",
            "Zoekt, ms",
            "Same as ripgrep",
            "Same as Zoekt",
            "Wall time, rg / Ferret",
            "Wall time, Zoekt / Ferret",
            "Zoekt build, s",
        ],
        rows,
    )


def render_break_even(results: Path) -> str:
    """Per bucket: size, build and query times, and the break-even query
    counts of the measured snapshots."""
    table = pd.read_csv(results / "breakeven.csv").fillna({"bucket": ""})
    episodes = pd.read_csv(results / "episodes.csv")
    ladder = episodes[episodes["arm"].isin(["A1", "A2", "A3", "A4"])]
    searches = ladder.groupby("bucket")["search_calls"].median()
    buckets = sorted(
        table["bucket"].unique(),
        key=lambda b: (_BUCKET_ORDER.index(b) if b in _BUCKET_ORDER else 99, b),
    )
    rows = []
    for bucket in buckets:
        data = table[table["bucket"] == bucket]
        repos = data.drop_duplicates("repo")
        first = data[data["worktrees"] == BREAK_EVEN_WORKTREES[0]]
        rows.append(
            [
                bucket or "all",
                f"{len(repos):,}",
                f"{repos['files'].median():,.0f}",
                f"{repos['build_seconds'].median():,.1f}",
                f"{1000 * repos['rg_seconds'].median():,.1f}",
                f"{1000 * repos['index_seconds'].median():,.1f}",
                *(
                    _queries_range(
                        data.loc[data["worktrees"] == w, "break_even_queries"]
                    )
                    for w in BREAK_EVEN_WORKTREES
                ),
                f"{int(np.isinf(first['break_even_queries']).sum())}",
                f"{searches[bucket]:.0f}" if bucket in searches else "n/a",
            ]
        )
    return markdown_table(
        [
            "Bucket",
            "Snapshots",
            "Median N",
            "Median B, s",
            "Median t_rg, ms",
            "Median t_idx, ms",
            *(f"Q* (W = {w}), median (range)" for w in BREAK_EVEN_WORKTREES),
            "Never pay",
            "Searches per episode",
        ],
        rows,
    )


def render_break_even_holdout(results: Path) -> str:
    """The held-out snapshot's predicted and measured model quantities."""
    table = pd.read_csv(results / "breakeven_holdout.csv")
    if table.empty:
        return "No held-out snapshot was measured.\n"
    names = {
        "build_seconds": "Build time B, s",
        "rg_seconds": "ripgrep latency t_rg, s",
        "index_seconds": "Ferret latency t_idx, s",
        "break_even_queries": "Q* (W = 1)",
    }
    rows = [
        [
            row["repo"],
            names[row["quantity"]],
            _queries(row["predicted"])
            if row["quantity"] == "break_even_queries"
            else f"{row['predicted']:.4g}",
            _queries(row["measured"])
            if row["quantity"] == "break_even_queries"
            else f"{row['measured']:.4g}",
            "∞" if math.isinf(row["factor"]) else f"{row['factor']:.2f}x",
        ]
        for _, row in table.iterrows()
    ]
    verdicts = table[table["quantity"] == "break_even_queries"]["validated"]
    verdict = (
        "The model is validated: every held-out Q* is within a factor of "
        f"{HOLDOUT_FACTOR:g} of its prediction."
        if verdicts.astype(bool).all()
        else "The model is not validated: a held-out Q* is more than a factor "
        f"of {HOLDOUT_FACTOR:g} from its prediction."
    )
    fitted = int(table["fitted_on"].iloc[0])
    return f"Fitted on {fitted} other snapshots. {verdict}\n\n" + markdown_table(
        ["Snapshot", "Quantity", "Predicted", "Measured", "Factor"], rows
    )


# Block name -> renderer; the renderer's docstring says what the block shows.
BLOCKS: dict[str, Callable[[Path], str]] = {
    "arm-summary": render_arm_summary,
    "contrasts": render_contrasts,
    "mechanism": render_mechanism,
    "tokens": render_tokens,
    "variance": render_variance,
    "sensitivity": render_sensitivity,
    "glmm": render_glmm,
    "contamination": render_contamination,
    "costs": render_costs,
    "replay": render_replay,
    "break-even": render_break_even,
    "break-even-holdout": render_break_even_holdout,
}


def render_findings(document: str, results: Path) -> str:
    """Returns `document` with every generated block re-rendered from `results`.

    Raises:
        ValueError: if a block name is unknown.
        FileNotFoundError: if a block's input file is missing.
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in BLOCKS:
            raise ValueError(
                f"unknown generated block {name!r}; known: {sorted(BLOCKS)}"
            )
        return match.group(1) + BLOCKS[name](results) + match.group(4)

    return _BLOCK.sub(replace, document)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the process exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--findings", type=Path, required=True)
    parser.add_argument(
        "--check", action="store_true", help="fail if the document is stale"
    )
    args = parser.parse_args(argv)
    current = args.findings.read_text(encoding="utf-8")
    rendered = render_findings(current, args.results)
    if args.check:
        if rendered != current:
            print(f"{args.findings} is stale; rerun analysis.report", file=sys.stderr)
            return 1
        return 0
    args.findings.write_text(rendered, encoding="utf-8")
    return 0


def _contrasts(results: Path) -> pd.DataFrame:
    return pd.read_csv(results / "contrasts.csv")


def _contrast_table(
    results: Path, metric: str, subset: str, label: str, scale: float
) -> str:
    table = _contrasts(results)
    selected = table[(table.metric == metric) & (table.subset == subset)]
    rows = [
        [
            row.bucket,
            row.contrast,
            str(row.n_tasks),
            f"{scale * row.estimate:+.1f} {interval(row.ci_low, row.ci_high, scale)}",
            p_value(row.p_value),
            p_value(row.p_holm),
        ]
        for row in selected.itertuples()
    ]
    return markdown_table(["Bucket", "Contrast", "Tasks", label, "p", "p (Holm)"], rows)


def _percent_change(log_ratio: float) -> str:
    return f"{100 * math.expm1(log_ratio):+.1f}%"


def _numeric_column(rows: Sequence[Sequence[str]], index: int) -> bool:
    cells = [row[index] for row in rows if row[index]]
    return bool(cells) and all(re.match(r"^[-+$<]?[\d.,]", cell) for cell in cells)


def contrast_table(
    frame: pd.DataFrame,
    metric: str,
    *,
    contrasts: Sequence[tuple[str, str]] = CONTRASTS,
    draws: int = 10_000,
) -> pd.DataFrame:
    """Returns cluster intervals and two-sided bootstrap p-values with Holm."""
    rows = []
    p_values = []
    for baseline, treatment in contrasts:
        estimate, low, high = paired_bootstrap(
            frame, baseline, treatment, metric, draws=draws
        )
        # Null-centered bootstrap distribution uses effect minus estimate.
        effects = paired_effects(frame, baseline, treatment, metric).to_numpy()
        rng = np.random.default_rng(0)
        centered = (
            rng.choice(effects, (draws, len(effects)), replace=True).mean(axis=1)
            - estimate
        )
        p = min(
            1.0, (1 + np.count_nonzero(abs(centered) >= abs(estimate))) / (draws + 1)
        )
        p_values.append(p)
        rows.append(
            {
                "contrast": f"{treatment}-{baseline}",
                "effect": estimate,
                "low": low,
                "high": high,
                "p": p,
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm"] = holm(p_values, family_size=len(CONTRASTS))
    return result


if __name__ == "__main__":
    sys.exit(main())

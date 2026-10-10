# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""The break-even model and the dollar cost of each arm.

Latency and tokens are kept in separate models because they have no common
unit. Latency: an index pays for itself after

    Q*(N, W) = (B(N) + E·u) / (W · (t_rg(N) - t_idx(N)))

queries, where B is the build time for a repository of N files, E·u the cost
of ingesting E edits at u seconds each, W the number of worktrees sharing the
index, and t_rg, t_idx the per-query latencies of ripgrep and the index. Each
of B, t_rg, and t_idx is fitted as a power law a·N^b through the measured
repositories, so the model can predict a held-out repository (LLVM) before it
is measured. Tokens: dollars per episode and per solved task, by bucket and
arm, and their difference from the raw-grep arm A1.

Figures are drawn with matplotlib's Agg canvas directly, so importing this
module never touches pyplot's global backend state.
"""

import dataclasses
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from analysis.protocol import LADDER, Arm, TokenPrices

# Reference data-viz palette: categorical slots 1-2 and the light chart chrome.
_SERIES = ("#2a78d6", "#eb6834")
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_DPI = 150


def break_even_queries(
    build_seconds: npt.ArrayLike,
    rg_seconds: npt.ArrayLike,
    index_seconds: npt.ArrayLike,
    *,
    worktrees: int = 1,
    edits: int = 0,
    ingest_seconds_per_edit: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Returns Q*, elementwise; +inf where the index is not faster than ripgrep."""
    saving = worktrees * (
        np.asarray(rg_seconds, float) - np.asarray(index_seconds, float)
    )
    cost = np.asarray(build_seconds, float) + edits * ingest_seconds_per_edit
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(saving > 0, cost / saving, np.inf)


@dataclasses.dataclass(frozen=True, slots=True)
class PowerLaw:
    """y = coefficient · N^exponent."""

    coefficient: float
    exponent: float

    def __call__(self, files: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return self.coefficient * np.asarray(files, float) ** self.exponent

    @classmethod
    def fit(cls, files: Sequence[float], values: Sequence[float]) -> "PowerLaw":
        """Least-squares fit of log y on log N.

        Raises:
            ValueError: unless there are two distinct sizes and all inputs are
                positive.
        """
        x, y = np.log(np.asarray(files, float)), np.log(np.asarray(values, float))
        if len(set(files)) < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError(
                "power-law fit needs two distinct sizes and positive values"
            )
        exponent, intercept = np.polyfit(x, y, 1)
        return cls(coefficient=math.exp(intercept), exponent=float(exponent))


@dataclasses.dataclass(frozen=True, slots=True)
class RepoMeasurement:
    """Systems-benchmark numbers for one repository, in seconds.

    Attributes:
        name: Repository name, such as "linux".
        files: Tracked files N.
        build_seconds: Index build wall time B.
        rg_seconds: Median warm ripgrep latency per replayed agent query.
        index_seconds: Median warm Ferret latency per replayed agent query.
        bucket: The task bucket ("S", "L", "XL") the repository belongs to;
            empty in files written before buckets were recorded.
    """

    name: str
    files: int
    build_seconds: float
    rg_seconds: float
    index_seconds: float
    bucket: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class BreakEvenModel:
    """Q*(N, W) with B, t_rg, and t_idx interpolated as power laws in N."""

    build: PowerLaw
    rg: PowerLaw
    index: PowerLaw

    @classmethod
    def fit(cls, measurements: Sequence[RepoMeasurement]) -> "BreakEvenModel":
        """Fits the three power laws through the measured repositories."""
        files = [m.files for m in measurements]
        return cls(
            build=PowerLaw.fit(files, [m.build_seconds for m in measurements]),
            rg=PowerLaw.fit(files, [m.rg_seconds for m in measurements]),
            index=PowerLaw.fit(files, [m.index_seconds for m in measurements]),
        )

    def queries(
        self,
        files: npt.ArrayLike,
        *,
        worktrees: int = 1,
        edits: int = 0,
        ingest_seconds_per_edit: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Returns the predicted Q* for repositories of `files` files."""
        return break_even_queries(
            self.build(files),
            self.rg(files),
            self.index(files),
            worktrees=worktrees,
            edits=edits,
            ingest_seconds_per_edit=ingest_seconds_per_edit,
        )


def break_even_table(
    measurements: Iterable[RepoMeasurement], worktrees: Iterable[int]
) -> pd.DataFrame:
    """Returns the measured Q* of every repository for every W, without edits,
    with the inputs (seconds) it was computed from."""
    worktrees = list(worktrees)
    return pd.DataFrame(
        [
            {
                "repo": m.name,
                "bucket": m.bucket,
                "files": m.files,
                "build_seconds": m.build_seconds,
                "rg_seconds": m.rg_seconds,
                "index_seconds": m.index_seconds,
                "worktrees": w,
                "break_even_queries": float(
                    break_even_queries(
                        m.build_seconds, m.rg_seconds, m.index_seconds, worktrees=w
                    )
                ),
            }
            for m in measurements
            for w in worktrees
        ]
    )


# The held-out snapshot's Q* must be within this factor of the prediction.
HOLDOUT_FACTOR = 2.0
HOLDOUT_BUCKET = "L"
HOLDOUT_COLUMNS = (
    "repo",
    "bucket",
    "files",
    "worktrees",
    "quantity",
    "predicted",
    "measured",
    "factor",
    "fitted_on",
    "validated",
)


def _factor(predicted: float, measured: float) -> float:
    """Returns how many times larger the larger of the two is (1 for equal,
    including two infinities, and infinity for one infinity)."""
    if predicted == measured:
        return 1.0
    if not (math.isfinite(predicted) and math.isfinite(measured)):
        return math.inf
    if predicted <= 0 or measured <= 0:
        return math.inf
    return max(predicted / measured, measured / predicted)


def holdout_table(
    measurements: Sequence[RepoMeasurement],
    *,
    holdout_bucket: str = HOLDOUT_BUCKET,
    worktrees: int = 1,
) -> pd.DataFrame:
    """Predicts the held-out snapshot from a fit on all the others.

    The three power laws (B, t_rg, t_idx) are fitted on every measurement
    outside `holdout_bucket`; each held-out measurement is then predicted at
    its own N. One row per quantity (build, rg, index, break_even) with the
    prediction, the measurement, and the factor between them; the Q* row has
    `validated`, whether its factor is at most `HOLDOUT_FACTOR`. The model is
    validated only if every held-out snapshot is. Empty if there is no
    held-out measurement.

    Raises:
        ValueError: if the others do not give two distinct sizes.
    """
    held = [m for m in measurements if m.bucket == holdout_bucket]
    train = [m for m in measurements if m.bucket != holdout_bucket]
    rows = []
    if not held:
        return pd.DataFrame(columns=list(HOLDOUT_COLUMNS))
    model = BreakEvenModel.fit(train)
    for m in held:
        predicted = {
            "build_seconds": float(model.build(m.files)),
            "rg_seconds": float(model.rg(m.files)),
            "index_seconds": float(model.index(m.files)),
            "break_even_queries": float(model.queries(m.files, worktrees=worktrees)),
        }
        measured = {
            "build_seconds": m.build_seconds,
            "rg_seconds": m.rg_seconds,
            "index_seconds": m.index_seconds,
            "break_even_queries": float(
                break_even_queries(
                    m.build_seconds, m.rg_seconds, m.index_seconds, worktrees=worktrees
                )
            ),
        }
        for quantity, value in predicted.items():
            factor = _factor(value, measured[quantity])
            rows.append(
                {
                    "repo": m.name,
                    "bucket": m.bucket,
                    "files": m.files,
                    "worktrees": worktrees,
                    "quantity": quantity,
                    "predicted": value,
                    "measured": measured[quantity],
                    "factor": factor,
                    "fitted_on": len(train),
                    "validated": factor <= HOLDOUT_FACTOR
                    if quantity == "break_even_queries"
                    else None,
                }
            )
    return pd.DataFrame(rows, columns=list(HOLDOUT_COLUMNS))


def cost_table(frame: pd.DataFrame, prices: TokenPrices) -> pd.DataFrame:
    """Returns dollars per episode and per solved task, by bucket and arm.

    `delta_cost_per_solved` is the difference from arm A1 in the same bucket
    (NaN for A0 and where either arm solved nothing). Cost per solved task is
    total cost over total successes, infinite when nothing was solved.
    """
    costs = frame.assign(
        cost=prices.cost(
            frame["input_tokens"],
            frame["output_tokens"],
            frame.get("cached_input_tokens", 0),
        ),
        solved=frame["success"].astype(int),
    )
    table = (
        costs.groupby(["bucket", "arm"])
        .agg(episodes=("cost", "size"), cost=("cost", "sum"), solved=("solved", "sum"))
        .reset_index()
    )
    table["cost_per_episode"] = table["cost"] / table["episodes"]
    table["cost_per_solved"] = table["cost"] / table["solved"]  # x / 0 is inf.
    baseline = table[table["arm"] == Arm.A1.value].set_index("bucket")[
        "cost_per_solved"
    ]
    table["delta_cost_per_solved"] = (
        table["cost_per_solved"] - table["bucket"].map(baseline)
    ).where(table["arm"].isin([arm.value for arm in LADDER]))
    table["delta_cost_per_solved"] = pd.to_numeric(
        table["delta_cost_per_solved"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    return table.drop(columns=["cost"])


def plot_break_even(
    model: BreakEvenModel,
    measurements: Sequence[RepoMeasurement],
    searches_per_task: npt.ArrayLike,
    path: Path,
    *,
    worktrees: Sequence[int] = (1, 10),
) -> None:
    """Draws Q* against repository size, one line per W, as a PNG at `path`.

    Measured repositories are dots on each line's colour, one label per
    bucket (the dot of median size) rather than one per repository; the
    grey band is the interquartile range of searches per task in the agent
    runs, with its median as a line, so the reader sees where observed search
    volume crosses the break-even curve.
    """
    if len(worktrees) > len(_SERIES):
        raise ValueError(
            f"at most {len(_SERIES)} worktree counts, got {len(worktrees)}"
        )
    sizes = [m.files for m in measurements]
    span = np.geomspace(min(sizes) / 2, max(sizes) * 2, 200)
    figure, axes = _figure(6.4, 4.0)
    searches = np.asarray(searches_per_task, float)
    low, median, high = np.percentile(searches, [25, 50, 75])
    axes.axhspan(low, high, color=_INK_SECONDARY, alpha=0.10, linewidth=0)
    axes.axhline(median, color=_INK_SECONDARY, linewidth=1.0)
    axes.annotate(
        "searches per task (median, IQR)",
        (span[0], high),
        xytext=(2, 3),
        textcoords="offset points",
        color=_INK_SECONDARY,
        fontsize=8,
    )
    for color, w in zip(_SERIES, worktrees, strict=False):
        axes.plot(
            span,
            model.queries(span, worktrees=w),
            color=color,
            linewidth=1.5,
            label=f"W = {w}",
        )
        measured = break_even_queries(
            [m.build_seconds for m in measurements],
            [m.rg_seconds for m in measurements],
            [m.index_seconds for m in measurements],
            worktrees=w,
        )
        axes.plot(
            sizes,
            measured,
            linestyle="none",
            marker="o",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor=_SURFACE,
            markeredgewidth=1.5,
        )
    top = break_even_queries(
        [m.build_seconds for m in measurements],
        [m.rg_seconds for m in measurements],
        [m.index_seconds for m in measurements],
        worktrees=worktrees[0],
    )
    labelled = _label_points(measurements)
    for m, y in zip(measurements, top, strict=True):
        if np.isfinite(y) and m in labelled:
            axes.annotate(
                _label(m, measurements), (m.files, y), xytext=(5, 5),
                textcoords="offset points", color=_INK, fontsize=8,
            )  # fmt: skip
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("Repository size (tracked files)")
    axes.set_ylabel("Break-even queries Q*")
    axes.set_title("Queries until the index pays for its build", loc="left", color=_INK)
    axes.legend(frameon=False, labelcolor=_INK_SECONDARY)
    _save(figure, path)


def _label_points(measurements: Sequence[RepoMeasurement]) -> list[RepoMeasurement]:
    """Returns the measurements that get a label: one per bucket (the one
    of median size), or every one when buckets were not recorded."""
    if not all(m.bucket for m in measurements):
        return list(measurements)
    chosen = []
    for bucket in dict.fromkeys(m.bucket for m in measurements):
        group = sorted(
            (m for m in measurements if m.bucket == bucket), key=lambda m: m.files
        )
        chosen.append(group[(len(group) - 1) // 2])
    return chosen


def _label(m: RepoMeasurement, measurements: Sequence[RepoMeasurement]) -> str:
    if not m.bucket:
        return m.name
    count = sum(other.bucket == m.bucket for other in measurements)
    return (
        f"{m.name} (bucket {m.bucket})"
        if count == 1
        else (f"bucket {m.bucket}: {count} repos")
    )


def plot_cost_per_solved(costs: pd.DataFrame, path: Path) -> None:
    """Draws, per bucket, each ladder arm's extra dollars per solved task over A1.

    One small multiple per bucket; bars grow from zero, so arms that save
    money point left. Arm names sit on the axis and values at the bar tips,
    so the single colour carries no meaning of its own.
    """
    arms = [arm.value for arm in LADDER[1:]]
    buckets = sorted(costs["bucket"].unique())
    figure = Figure(figsize=(3.2 * len(buckets), 2.4), dpi=_DPI, facecolor=_SURFACE)
    FigureCanvasAgg(figure)
    panels = figure.subplots(1, len(buckets), squeeze=False, sharex=True)[0]
    for axes, bucket in zip(panels, buckets, strict=True):
        _style(axes)
        rows = costs[costs["bucket"] == bucket].set_index("arm")
        values = [float(rows["delta_cost_per_solved"].get(arm, np.nan)) for arm in arms]
        positions = np.arange(len(arms))
        axes.barh(positions, np.nan_to_num(values), height=0.5, color=_SERIES[0])
        axes.axvline(0, color=_AXIS, linewidth=1.0)
        for position, value in zip(positions, values, strict=True):
            missing, negative = math.isnan(value), value < 0
            axes.annotate(
                "n/a" if missing else f"{value:+.3f}",
                (0 if missing else value, position),
                xytext=(-4 if negative else 4, 0),
                textcoords="offset points",
                ha="right" if negative else "left",
                va="center",
                color=_INK,
                fontsize=8,
            )
        axes.set_yticks(positions, [f"{arm} vs A1" for arm in arms])
        axes.invert_yaxis()
        axes.set_title(f"Bucket {bucket}", loc="left", color=_INK, fontsize=9)
        axes.set_xlabel("Δ $ per solved task", color=_INK_SECONDARY)
    _save(figure, path)


def _figure(width: float, height: float) -> tuple[Figure, Axes]:
    figure = Figure(figsize=(width, height), dpi=_DPI, facecolor=_SURFACE)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    _style(axes)
    return figure, axes


def _style(axes: Axes) -> None:
    """Applies the recessive chrome: hairline solid grid, muted ticks, no box."""
    axes.set_facecolor(_SURFACE)
    axes.grid(True, which="major", color=_GRID, linewidth=0.8, linestyle="-")
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(_AXIS)
    axes.tick_params(colors=_INK_MUTED, labelcolor=_INK_SECONDARY, labelsize=8)
    axes.xaxis.label.set_color(_INK_SECONDARY)
    axes.yaxis.label.set_color(_INK_SECONDARY)


def _save(figure: Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, facecolor=_SURFACE)


def queries_to_breakeven(
    build_s: float, ingest_s: float, worktrees: int, scan_s: float, indexed_s: float
) -> float:
    """Returns searches per worktree; infinity if indexing never saves time."""
    if min(build_s, ingest_s, scan_s, indexed_s) < 0 or worktrees < 1:
        raise ValueError("times must be nonnegative and worktrees positive")
    saving = scan_s - indexed_s
    return (build_s + ingest_s) / (worktrees * saving) if saving > 0 else math.inf

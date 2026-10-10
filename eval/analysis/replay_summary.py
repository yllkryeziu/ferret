# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""The replay's per-bucket summary (the report's "replay" block).

`harness.replay` writes one CSV row per replayed search call. This module
groups the rows by the bucket of their snapshot and reports, per bucket:

* the calls replayed out of the calls in the run (from the replay's
  provenance file);
* the p50, p90 and p99 latency in ms of Ferret as the agent ran it, Ferret
  raw, ripgrep and Zoekt;
* how many comparable calls returned the same match set as ripgrep, and as
  Zoekt (ripgrep errors, and Zoekt errors and truncated results, are not
  comparable and are counted apart);
* the median projected episode wall-time ratio, had every search taken
  ripgrep's (or Zoekt's) time instead of Ferret's;
* Zoekt's build time.
"""

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from harness.replay import projected_wall_time_ratios
from tasks.schema import Task

ENGINES = (
    ("ferret", "ferret_ms"),
    ("ferret_raw", "ferret_raw_ms"),
    ("rg", "rg_ms"),
    ("zoekt", "zoekt_ms"),
)
BUCKET_ORDER = ("S", "L", "XL")


def _percentiles(values: pd.Series) -> tuple[float, float, float]:
    values = values.dropna().astype(float)
    if values.empty:
        return (float("nan"),) * 3
    p50, p90, p99 = np.percentile(values, [50, 90, 99])
    return float(p50), float(p90), float(p99)


def _median_ratio(
    rows: pd.DataFrame, wall_times: Mapping[str, float], column: str
) -> float:
    ratios = projected_wall_time_ratios(rows.to_dict("records"), wall_times, column)
    return statistics.median(ratios) if ratios else float("nan")


def replay_summary(
    rows: pd.DataFrame,
    tasks: Sequence[Task],
    wall_times: Mapping[str, float],
    provenance: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Returns one row per bucket (S, L, XL first) of the replay `rows`.

    `wall_times` maps episode ids to their wall time in the run, and
    `provenance` is the replay's provenance file, whose per-snapshot call
    counts give the calls in the run (NaN without it).

    Raises:
        ValueError: if a replayed snapshot belongs to no task.
    """
    bucket_of = {task.snapshot: task.bucket.value for task in tasks}
    unknown = set(rows["snapshot"]) - bucket_of.keys()
    if unknown:
        raise ValueError(f"no task has snapshot {sorted(unknown)[0]}")
    rows = rows.assign(bucket=rows["snapshot"].map(bucket_of))
    if "zoekt_status" not in rows:
        rows = rows.assign(zoekt_status="none", zoekt_ms=np.nan, zoekt_build_s=np.nan)
    buckets = sorted(
        rows["bucket"].unique(),
        key=lambda b: (BUCKET_ORDER.index(b) if b in BUCKET_ORDER else 99, b),
    )
    counts = (provenance or {}).get("snapshots", {})
    out = []
    for bucket in buckets:
        data = rows[rows["bucket"] == bucket]
        snapshots = set(data["snapshot"])
        row: dict[str, Any] = {
            "bucket": bucket,
            "snapshots": len(snapshots),
            "calls_replayed": len(data),
            "calls_in_run": sum(counts[s]["calls_in_run"] for s in snapshots)
            if snapshots <= counts.keys()
            else float("nan"),
        }
        zoekt_ok = data[data["zoekt_status"] == "ok"]
        for name, column in ENGINES:
            source = zoekt_ok if name == "zoekt" else data
            row |= dict(
                zip(
                    (f"{name}_p50_ms", f"{name}_p90_ms", f"{name}_p99_ms"),
                    _percentiles(source[column]),
                    strict=True,
                )
            )
        rg = data[data["rg_status"].isin([0, 1])]
        zoekt = zoekt_ok[~zoekt_ok["zoekt_truncated"].astype(bool)]
        row |= {
            "rg_comparable": len(rg),
            "rg_errors": len(data) - len(rg),
            "rg_identical": int(
                ((rg["only_ferret"] == 0) & (rg["only_rg"] == 0)).sum()
            ),
            "zoekt_comparable": len(zoekt),
            "zoekt_identical": int(
                (
                    (zoekt["zoekt_only_vs_ferret"] == 0)
                    & (zoekt["ferret_only_vs_zoekt"] == 0)
                ).sum()
            ),
            "zoekt_errors": int(data["zoekt_status"].str.startswith("error").sum()),
            "zoekt_truncated": len(zoekt_ok) - len(zoekt),
            "ratio_rg_over_ferret": _median_ratio(rg, wall_times, "rg_ms"),
            "ratio_zoekt_over_ferret": _median_ratio(zoekt_ok, wall_times, "zoekt_ms")
            if len(zoekt_ok)
            else float("nan"),
            "zoekt_build_seconds": float(
                data.groupby("snapshot")["zoekt_build_s"].first().median()
            ),
        }
        out.append(row)
    return pd.DataFrame(out)

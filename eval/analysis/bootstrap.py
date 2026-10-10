# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Paired cluster bootstrap by task, the primary inference.

Samples of one task are correlated (the same problem, the same repository),
so the task is the unit of resampling. Each task's samples are first averaged
per arm; a contrast is the mean over tasks of the per-task difference, which
pairs the arms within each task and weights every task equally. Resampling
tasks with replacement gives a percentile confidence interval, and resampling
the differences shifted to mean zero gives a two-sided p-value for "no
difference" with the usual +1 correction, so p is never exactly zero. Tasks
missing either arm, or with the metric undefined in every sample (NaN), are
dropped from that contrast.

Reference: B. Efron, R. Tibshirani, An Introduction to the Bootstrap,
Chapman & Hall, 1993, §6 (percentile intervals) and §16.4 (shifted-null
tests); A. C. Davison, D. V. Hinkley, Bootstrap Methods and their
Application, 1997, §3.8 (resampling clusters).
"""

import dataclasses

import numpy as np
import numpy.typing as npt
import pandas as pd

from analysis.protocol import Arm

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
CONFIDENCE = 0.95


@dataclasses.dataclass(frozen=True, slots=True)
class BootstrapEstimate:
    """A mean over tasks with its bootstrap interval and p-value against zero."""

    estimate: float
    ci_low: float
    ci_high: float
    p_value: float
    n_tasks: int


def bootstrap_mean(
    values: npt.ArrayLike,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = CONFIDENCE,
) -> BootstrapEstimate:
    """Bootstraps the mean of per-task values.

    Runs in O(n_resamples · n) time and memory.

    Raises:
        ValueError: if `values` is empty or contains NaN.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.isnan(values).any():
        raise ValueError(f"bootstrap needs finite per-task values, got {values.size}")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n_resamples, values.size))].mean(
        axis=1
    )
    estimate = float(values.mean())
    tail = (1 - confidence) / 2
    ci_low, ci_high = np.quantile(means, [tail, 1 - tail])
    extreme = np.count_nonzero(np.abs(means - estimate) >= abs(estimate))
    return BootstrapEstimate(
        estimate=estimate,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=(extreme + 1) / (n_resamples + 1),
        n_tasks=int(values.size),
    )


def task_means(frame: pd.DataFrame, metric: str, arm: Arm) -> pd.Series:
    """Returns the mean of `metric` over each task's samples in `arm`, by task_id."""
    rows = frame[frame["arm"] == arm.value]
    return rows.groupby("task_id")[metric].mean().dropna()


def arm_mean(
    frame: pd.DataFrame,
    metric: str,
    arm: Arm,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapEstimate:
    """Bootstraps the task-weighted mean of `metric` in one arm."""
    return bootstrap_mean(
        task_means(frame, metric, arm), n_resamples=n_resamples, seed=seed
    )


def paired_differences(
    frame: pd.DataFrame, metric: str, baseline: Arm, treatment: Arm
) -> pd.Series:
    """Returns treatment minus baseline task means, for tasks that have both."""
    treated = task_means(frame, metric, treatment)
    return (treated - task_means(frame, metric, baseline)).dropna()


def paired_contrast(
    frame: pd.DataFrame,
    metric: str,
    baseline: Arm,
    treatment: Arm,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapEstimate:
    """Bootstraps the mean over tasks of treatment minus baseline.

    Args:
        frame: One row per episode with columns task_id, arm, and `metric`.
        metric: The column to compare.
        baseline: The arm subtracted.
        treatment: The arm compared against it.
        n_resamples: Bootstrap resamples.
        seed: Seeds the resampling, so results are reproducible.

    Raises:
        ValueError: if no task has the metric in both arms.
    """
    differences = paired_differences(frame, metric, baseline, treatment)
    return bootstrap_mean(differences, n_resamples=n_resamples, seed=seed)


def paired_effects(
    frame: pd.DataFrame, baseline: str, treatment: str, metric: str
) -> pd.Series:
    """Averages replicates within each task before pairing arms."""
    means = (
        frame[frame.arm.isin((baseline, treatment))]
        .groupby(["task_id", "arm"])[metric]
        .mean()
        .unstack()
    )
    if baseline not in means or treatment not in means or means.isna().any().any():
        raise ValueError("both arms must have observations for every task")
    return means[treatment] - means[baseline]


def paired_bootstrap(
    frame: pd.DataFrame,
    baseline: str,
    treatment: str,
    metric: str,
    *,
    draws: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Returns the mean difference and percentile 95% interval over tasks."""
    effects = paired_effects(frame, baseline, treatment, metric).to_numpy()
    if not len(effects) or draws < 1:
        raise ValueError("bootstrap requires tasks and positive draws")
    rng = np.random.default_rng(seed)
    samples = rng.choice(effects, size=(draws, len(effects)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(effects.mean()), float(low), float(high)

# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Power and sample size for the paired success contrasts.

The model behind the numbers: a task's success probability in the baseline
arm is θ ~ Beta(a, b), with a = b = 0.6 by default (most tasks are nearly
always or nearly never solved). The treatment shifts every task's log-odds by
the same δ, so p(θ) = θe^δ / (1 - θ + θe^δ), and δ is chosen to make the
average effect E[p(θ) - θ] equal the target. Each arm runs k samples per task.
The per-task mean difference d then has

    E[d] = E[p(θ) - θ],
    Var[d] = E[θ(1 - θ) + p(θ)(1 - p(θ))] / k + Var[p(θ) - θ],

and the test on n tasks is a two-sided one-sample t-test on d, the
large-sample equivalent of the paired cluster bootstrap, so power follows the
noncentral t distribution with n - 1 degrees of freedom and noncentrality
E[d]·√n / SD[d]. Expectations over θ use 64-point Gauss-Jacobi quadrature,
exact to rounding because p(θ) is a rational function without poles on
[0, 1]. The pilot re-estimates a and b before the protocol is frozen.
"""

import dataclasses
import math
from collections.abc import Callable, Iterable

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import optimize, special, stats
from scipy.stats import norm

from analysis.protocol import SAMPLES_PER_ARM

ALPHA = 0.05 / 3

_QUADRATURE_POINTS = 64


@dataclasses.dataclass(frozen=True, slots=True)
class DifficultyPrior:
    """Beta(alpha, beta) distribution of baseline success probability over tasks."""

    alpha: float = 0.6
    beta: float = 0.6

    def expect(
        self, g: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    ) -> float:
        """Returns E[g(θ)] for θ drawn from this prior."""
        # roots_jacobi(n, p, q) integrates against (1 - x)^p (1 + x)^q on
        # [-1, 1]; θ = (1 + x) / 2 turns that into the Beta(q + 1, p + 1) kernel.
        x, w = special.roots_jacobi(_QUADRATURE_POINTS, self.beta - 1, self.alpha - 1)
        return float(np.dot(w, g((1 + x) / 2)) / w.sum())


DEFAULT_PRIOR = DifficultyPrior()


def logit_shift(effect: float, prior: DifficultyPrior = DEFAULT_PRIOR) -> float:
    """Returns the uniform log-odds shift δ whose average effect is `effect`.

    Raises:
        ValueError: if no shift reaches `effect` (it must lie strictly between
            -E[θ] and 1 - E[θ]).
    """
    baseline = prior.expect(lambda theta: theta)
    if not -baseline < effect < 1 - baseline:
        raise ValueError(
            f"effect {effect} is unreachable from mean success {baseline:.3f}"
        )
    return optimize.brentq(
        lambda delta: (
            prior.expect(lambda theta: _shifted(theta, delta)) - baseline - effect
        ),
        -50.0,
        50.0,
        xtol=1e-12,
    )


def paired_difference_sd(
    effect: float,
    samples_per_task: int = SAMPLES_PER_ARM,
    prior: DifficultyPrior = DEFAULT_PRIOR,
) -> float:
    """Returns SD[d], the SD of one task's mean treatment-minus-baseline success."""
    delta = logit_shift(effect, prior)

    def within(theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        p = _shifted(theta, delta)
        return (theta * (1 - theta) + p * (1 - p)) / samples_per_task

    difference_variance = prior.expect(
        lambda theta: (_shifted(theta, delta) - theta - effect) ** 2
    )
    return math.sqrt(prior.expect(within) + difference_variance)


def power(
    n_tasks: int,
    effect: float,
    *,
    samples_per_task: int = SAMPLES_PER_ARM,
    alpha: float = ALPHA,
    prior: DifficultyPrior = DEFAULT_PRIOR,
) -> float:
    """Returns the probability that the paired test rejects at level `alpha`."""
    sd = paired_difference_sd(effect, samples_per_task, prior)
    df = n_tasks - 1
    critical = stats.t.ppf(1 - alpha / 2, df)
    noncentrality = effect * math.sqrt(n_tasks) / sd
    return float(
        stats.nct.sf(critical, df, noncentrality)
        + stats.nct.cdf(-critical, df, noncentrality)
    )


def required_tasks(
    effect: float,
    *,
    target_power: float = 0.8,
    samples_per_task: int = SAMPLES_PER_ARM,
    alpha: float = ALPHA,
    prior: DifficultyPrior = DEFAULT_PRIOR,
) -> int:
    """Returns the smallest number of tasks whose power reaches `target_power`."""

    def enough(n: int) -> bool:
        return (
            power(
                n, effect, samples_per_task=samples_per_task, alpha=alpha, prior=prior
            )
            >= target_power
        )

    high = 4
    while not enough(high):
        high *= 2
    low = high // 2  # Power grows with n, so the answer lies in (low, high].
    while high - low > 1:
        middle = (low + high) // 2
        if enough(middle):
            high = middle
        else:
            low = middle
    return high


def power_table(
    n_tasks: Iterable[int],
    effects: Iterable[float],
    *,
    samples_per_task: int = SAMPLES_PER_ARM,
    alpha: float = ALPHA,
    prior: DifficultyPrior = DEFAULT_PRIOR,
) -> pd.DataFrame:
    """Returns power for every (n_tasks, effect) pair; one row per pair."""
    effects = list(effects)
    return pd.DataFrame(
        [
            {
                "n_tasks": n,
                "effect": effect,
                "power": power(
                    n,
                    effect,
                    samples_per_task=samples_per_task,
                    alpha=alpha,
                    prior=prior,
                ),
            }
            for n in n_tasks
            for effect in effects
        ]
    )


def _shifted(theta: npt.NDArray[np.float64], delta: float) -> npt.NDArray[np.float64]:
    """Returns θ with its log-odds shifted by δ, without evaluating logit(θ)."""
    scale = math.exp(delta)
    return theta * scale / (1 - theta + theta * scale)


def tasks_for_effect(
    reduction: float,
    paired_sd: float = 1.0,
    *,
    k: int = 1,
    within_fraction: float = 1.0,
    contrasts: int = 3,
    power: float = 0.8,
    alpha: float = 0.05,
) -> int:
    """Normal-approximation pairs; k only helps the within-task variance.

    Pilot v2 has k=1 and cannot identify the within-task fraction. The
    optimistic default treats all observed variance as within-task.
    Bonferroni alpha/contrasts bounds Holm's required sample size.
    """
    if (
        not 0 < reduction < 1
        or paired_sd <= 0
        or k < 1
        or not 0 <= within_fraction <= 1
    ):
        raise ValueError("invalid effect, variance, k, or variance fraction")
    variance = paired_sd**2 * (1 - within_fraction + within_fraction / k)
    critical = norm.ppf(1 - alpha / (2 * contrasts)) + norm.ppf(power)
    return math.ceil(variance * (critical / -math.log1p(-reduction)) ** 2)


@dataclasses.dataclass(frozen=True, slots=True)
class VarianceSplit:
    """The paired log-token variance of one contrast at k = 1, split in two.

    Attributes:
        paired_sd: SD of one task's k = 1 paired log-token difference.
        within_fraction: The share of that variance from resampling a task,
            which k samples per arm divide by k; the rest is between tasks.
        tasks: Tasks with at least two samples in both arms.
    """

    paired_sd: float
    within_fraction: float
    tasks: int


def variance_split(
    frame: pd.DataFrame, arm: str, baseline: str, column: str = "log_total_tokens"
) -> VarianceSplit:
    """Estimates `VarianceSplit` for `arm` - `baseline` from k > 1 episodes.

    Samples of different arms are independent, so a k = 1 paired difference
    has variance σ²_between + σ²_arm + σ²_baseline, where each σ² of an arm
    is its pooled within-task sample variance. The difference of task means
    over k samples has variance σ²_between + (σ²_arm + σ²_baseline) / k,
    which gives σ²_between by the method of moments, floored at zero.
    """
    rows = frame[frame["arm"].isin((arm, baseline))]
    stats_by = rows.groupby(["task_id", "arm"])[column].agg(["mean", "var", "count"])
    wide = stats_by.unstack("arm").dropna()
    wide = wide[(wide["count"][arm] >= 2) & (wide["count"][baseline] >= 2)]
    if len(wide) < 2:
        raise ValueError("need at least two tasks with k >= 2 in both arms")

    def pooled(name: str) -> float:
        degrees = wide["count"][name] - 1
        return float((wide["var"][name] * degrees).sum() / degrees.sum())

    within = pooled(arm) + pooled(baseline)
    k = float(np.mean([wide["count"][arm].mean(), wide["count"][baseline].mean()]))
    differences = wide["mean"][arm] - wide["mean"][baseline]
    between = max(float(differences.var(ddof=1)) - within / k, 0.0)
    total = between + within
    return VarianceSplit(math.sqrt(total), within / total, len(wide))


def main() -> None:
    """Prints the pre-registration's pilot-v2 sensitivity table."""
    for k in (1, 3):
        for reduction in (0.16, 0.12):
            tasks = tasks_for_effect(reduction, k=k)
            print(f"k={k}, reduction={reduction:.0%}: {tasks} tasks")


def fit_difficulty_prior(frame: pd.DataFrame) -> DifficultyPrior:
    """Fits task difficulty from paired A1/A2/A3 binary pilot outcomes.

    Conditional on task difficulty, arms are independent Bernoulli trials;
    A2 and A3 have fitted log-odds offsets relative to A1. A single A1
    sample per task alone cannot identify the Beta concentration, so the
    paired arms provide the necessary within-task correlation. This is a
    sensitivity model, not a causal estimate of arm effects.
    """
    observations = frame[frame.arm.isin(("A1", "A2", "A3"))].pivot(
        index="task_id", columns="arm", values="success"
    )
    if observations.isna().any().any() or len(observations) < 2:
        raise ValueError("prior fit needs paired A1/A2/A3 outcomes on two tasks")
    outcomes = observations[["A1", "A2", "A3"]].to_numpy(dtype=float)

    def negative_log_likelihood(params: npt.NDArray[np.float64]) -> float:
        prior = DifficultyPrior(math.exp(params[0]), math.exp(params[1]))
        x, weights = special.roots_jacobi(
            _QUADRATURE_POINTS, prior.beta - 1, prior.alpha - 1
        )
        theta = (1 + x) / 2
        weights /= weights.sum()
        probabilities = np.column_stack(
            [theta, _shifted(theta, params[2]), _shifted(theta, params[3])]
        )
        likelihoods = (
            np.prod(
                np.where(outcomes[:, None, :] == 1, probabilities, 1 - probabilities),
                axis=2,
            )
            @ weights
        )
        return float(-np.log(likelihoods).sum())

    result = optimize.minimize(
        negative_log_likelihood,
        np.zeros(4),
        method="L-BFGS-B",
        bounds=[(-4, 4), (-4, 4), (-6, 6), (-6, 6)],
    )
    if not result.success:
        raise ValueError(f"prior fit failed: {result.message}")
    return DifficultyPrior(math.exp(result.x[0]), math.exp(result.x[1]))


if __name__ == "__main__":
    main()

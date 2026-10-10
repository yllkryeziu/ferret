# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""The secondary model: a binomial GLMM, success ~ arm * bucket + (1 | task).

Arms use backward-difference coding, so each arm coefficient is directly one
pre-registered ladder contrast on the logit scale (A2-A1, A3-A2, A4-A3);
buckets use sum-to-zero coding, so those contrasts are averaged over buckets
and the arm-by-bucket terms are each bucket's deviation from the average.

statsmodels has no frequentist binomial GLMM, so this fits
`BinomialBayesMixedGLM` by variational Bayes with its default priors (fixed
effects N(0, 2²), log random-effect SD N(0, 1)). The mean-field posterior
understates uncertainty somewhat, which is one reason the protocol makes the
paired cluster bootstrap primary and this model a robustness check: it
reports posterior means with mean ± 1.96 SD intervals, and the SD of the task
intercepts, which measures how much task difficulty varies.

Reference: statsmodels.genmod.bayes_mixed_glm; J. C. Pinheiro, D. M. Bates,
Mixed-Effects Models in S and S-PLUS, 2000.
"""

import dataclasses
import itertools
import math

import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from analysis.protocol import LADDER, contrast_name

GLMM_SEED = 0

_Z_95 = 1.959963984540054
# Backward-difference coding labels level k+1 minus level k as "D.<level k>";
# over the ladder these are exactly the pre-registered CONTRASTS.
_DIFF_TERMS = {
    f"D.{baseline}": contrast_name(baseline, treatment)
    for baseline, treatment in itertools.pairwise(LADDER)
}


@dataclasses.dataclass(frozen=True, slots=True)
class GlmmFit:
    """Posterior summaries of the fixed effects and the task-intercept SD.

    Attributes:
        fixed_effects: One row per term (term, mean, sd, ci_low, ci_high), on
            the logit scale. Terms are "Intercept", the contrasts such as
            "A2-A1", "bucket[L]", and interactions such as "A2-A1:bucket[L]".
        task_sd: Posterior mean of the SD of the per-task intercepts.
    """

    fixed_effects: pd.DataFrame
    task_sd: float


def fit_success_glmm(frame: pd.DataFrame, *, seed: int = GLMM_SEED) -> GlmmFit:
    """Fits the GLMM to the ladder arms of an episode frame.

    Args:
        frame: One row per episode with columns task_id, bucket, arm, success.
            Rows of other arms (A0) are ignored.
        seed: Seeds the variational optimizer's starting point.

    Raises:
        ValueError: if a ladder arm has no episodes.
    """
    arms = [arm.value for arm in LADDER]
    data = frame[frame["arm"].isin(arms)].copy()
    missing = sorted(set(arms) - set(data["arm"]))
    if missing:
        raise ValueError(f"glmm: no episodes in arms {missing}")
    data["success"] = data["success"].astype(float)
    formula = f"success ~ C(arm, Diff, levels={arms!r})"
    if data["bucket"].nunique() > 1:
        formula += " * C(bucket, Sum)"
    model = BinomialBayesMixedGLM.from_formula(
        formula, {"task": "0 + C(task_id)"}, data
    )
    result = model.fit_vb(rng=np.random.default_rng(seed))
    mean = np.asarray(result.fe_mean)
    sd = np.asarray(result.fe_sd)
    effects = pd.DataFrame(
        {
            "term": [_readable(name) for name in model.exog_names],
            "mean": mean,
            "sd": sd,
            "ci_low": mean - _Z_95 * sd,
            "ci_high": mean + _Z_95 * sd,
        }
    )
    return GlmmFit(fixed_effects=effects, task_sd=math.exp(float(result.vcp_mean[0])))


def _readable(term: str) -> str:
    """Renames patsy's terms, e.g. "C(arm, Diff, ...)[D.A1]" to "A2-A1"."""
    parts = []
    for factor in term.split(":"):
        label = factor.rsplit("[", 1)[-1].rstrip("]")
        if factor.startswith("C(arm"):
            parts.append(_DIFF_TERMS[label])
        elif factor.startswith("C(bucket"):
            parts.append(f"bucket[{label.removeprefix('S.')}]")
        else:
            parts.append(factor)
    return ":".join(parts)


def fit_success(frame: pd.DataFrame):
    """Fits a variational Bayes binomial GLMM with a task random intercept."""
    if frame.empty:
        raise ValueError("success GLMM requires observations")
    data = frame.assign(success=frame.success.astype(int))
    model = BinomialBayesMixedGLM.from_formula(
        "success ~ C(arm) * C(bucket)", {"task": "0 + C(task_id)"}, data
    )
    return model.fit_vb()

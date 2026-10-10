# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Holm's step-down correction over a family of pre-registered contrasts.

Holm controls the family-wise error rate under any dependence between the
tests, which matters here because the ladder contrasts share arms (A2 is in
both A2-A1 and A3-A2). It is uniformly more powerful than Bonferroni.

Reference: S. Holm, A simple sequentially rejective multiple test procedure,
Scandinavian Journal of Statistics 6(2), 1979.
"""

from collections.abc import Mapping, Sequence

from statsmodels.stats.multitest import multipletests


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Returns Holm-adjusted p-values, keyed like `p_values`.

    With the m raw p-values sorted ascending, the i-th adjusted value is
    max over j <= i of min(1, (m - j + 1) · p_(j)); a contrast is rejected
    at level alpha exactly when its adjusted p-value is at most alpha.
    """
    if not p_values:
        return {}
    names = list(p_values)
    _, adjusted, _, _ = multipletests([p_values[name] for name in names], method="holm")
    return {name: float(value) for name, value in zip(names, adjusted, strict=True)}


def holm(p_values: Sequence[float], *, family_size: int | None = None) -> list[float]:
    """Returns adjusted p-values in the original contrast order."""
    if any(not 0 <= p <= 1 for p in p_values):
        raise ValueError("p-values must be between zero and one")
    family_size = len(p_values) if family_size is None else family_size
    if family_size < len(p_values):
        raise ValueError("family size must include all reported contrasts")
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(
        sorted(range(len(p_values)), key=p_values.__getitem__)
    ):
        running = max(running, (family_size - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted

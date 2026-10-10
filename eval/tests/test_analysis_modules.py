# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Small known-answer tests for each statistical and break-even module."""

import math

import numpy as np
import pandas as pd
import pytest

from analysis.bootstrap import paired_bootstrap, paired_effects
from analysis.breakeven import queries_to_breakeven
from analysis.glmm import fit_success
from analysis.multiplicity import holm
from analysis.power import tasks_for_effect, variance_split
from analysis.report import contrast_table


def _frame():
    return pd.DataFrame(
        [
            {
                "task_id": task,
                "bucket": bucket,
                "arm": arm,
                "sample": sample,
                "success": int((task + sample + int(arm == "A2")) % 3 != 0),
                "total_tokens": 100 + 10 * task + 50 * int(arm == "A2"),
                "restricted_exposure_turns": 4 + int(arm == "A2"),
            }
            for task in range(12)
            for bucket in ("S", "L")
            if bucket == ("S" if task < 6 else "L")
            for arm in ("A1", "A2")
            for sample in (0, 1, 2)
        ]
    )


def test_bootstrap_resamples_tasks_not_individual_episodes():
    frame = _frame()
    assert list(paired_effects(frame, "A1", "A2", "total_tokens")) == [50] * 12
    assert paired_bootstrap(frame, "A1", "A2", "total_tokens", draws=100) == (
        50,
        50,
        50,
    )
    with pytest.raises(ValueError):
        paired_effects(frame[frame.arm == "A1"], "A1", "A2", "total_tokens")


def test_holm_adjusts_in_sorted_step_down_order():
    assert holm([0.04, 0.001, 0.03]) == [0.06, 0.003, 0.06]
    with pytest.raises(ValueError):
        holm([-0.1])


def test_power_exposes_unidentified_variance_fraction():
    assert tasks_for_effect(0.16, k=1) == 345
    assert tasks_for_effect(0.12, k=3) == 214
    assert tasks_for_effect(0.16, k=3, within_fraction=0) == 345


def test_break_even_handles_sharing_and_no_saving():
    assert queries_to_breakeven(100, 20, 2, 3, 1) == 30
    assert math.isinf(queries_to_breakeven(100, 0, 1, 1, 1))


def test_report_reuses_paired_effect_and_holm():
    result = contrast_table(
        _frame(), "total_tokens", contrasts=[("A1", "A2")], draws=100
    )
    assert result.iloc[0]["effect"] == 50
    assert result.iloc[0]["low"] == result.iloc[0]["high"] == 50
    assert result.iloc[0]["p_holm"] < 0.05


def test_glmm_uses_random_task_intercept_and_arm_by_bucket():
    result = fit_success(_frame())
    assert any("C(arm)[T.A2]:C(bucket)" in name for name in result.model.exog_names)
    assert len(result.random_effects()) == 12


def test_restored_success_power_and_cluster_bootstrap():
    from analysis.bootstrap import arm_mean, paired_contrast
    from analysis.power import DifficultyPrior, fit_difficulty_prior, required_tasks
    from analysis.protocol import Arm

    frame = _frame()
    assert (
        paired_contrast(frame, "total_tokens", Arm.A1, Arm.A2, n_resamples=100).estimate
        == 50
    )
    assert arm_mean(frame, "total_tokens", Arm.A1, n_resamples=100).n_tasks == 12
    prior = DifficultyPrior(1, 1)
    assert prior.expect(lambda theta: theta) == pytest.approx(0.5)
    assert required_tasks(0.1, prior=prior, samples_per_task=3) < required_tasks(
        0.1, prior=prior, samples_per_task=1
    )
    paired = frame[frame["sample"] == 0][["task_id", "arm", "success"]]
    synthetic = pd.concat(
        [paired, paired[paired.arm == "A2"].assign(arm="A3")], ignore_index=True
    )
    assert fit_difficulty_prior(synthetic).alpha > 0


def test_restored_breakeven_power_law_and_costs():
    from analysis.breakeven import PowerLaw, break_even_queries, cost_table
    from analysis.protocol import TokenPrices

    assert break_even_queries(100, 3, 1, worktrees=2) == pytest.approx(25)
    assert math.isinf(break_even_queries(100, 1, 2))
    # Cached input is priced separately, not billed at the full input rate.
    frame = _frame().assign(input_tokens=100, cached_input_tokens=50, output_tokens=20)
    costs = cost_table(frame, TokenPrices(1, 10, 0.1))
    assert costs.loc[costs.arm == "A1", "cost_per_episode"].iloc[0] == pytest.approx(
        (50 + 5 + 200) / 1e6
    )
    assert PowerLaw.fit([100, 200, 400], [10, 20, 40])(800) == pytest.approx(80)


def test_parked_holm_mapping_and_finding_renderer():
    from analysis.multiplicity import holm_adjust
    from analysis.report import markdown_table

    assert holm_adjust({"b": 0.04, "a": 0.001}) == {"b": 0.04, "a": 0.002}
    assert "| Column |" in markdown_table(["Column"], [["value"]])


def test_restored_glmm_ladder_returns_contrast_labels():
    from analysis.glmm import fit_success_glmm

    rows = [
        {
            "task_id": str(task),
            "bucket": "S" if task < 10 else "L",
            "arm": arm,
            "success": int((task * 7 + sample * 5 + int(arm[1])) % 9 < 4),
        }
        for task in range(20)
        for sample in range(3)
        for arm in ("A1", "A2", "A3", "A4")
    ]
    fit = fit_success_glmm(pd.DataFrame(rows))
    assert {"A2-A1", "A3-A2", "A4-A3"} <= set(fit.fixed_effects.term)
    assert fit.task_sd > 0


def test_restored_cost_figure_and_findings_render(tmp_path):
    from analysis.breakeven import cost_table, plot_cost_per_solved
    from analysis.protocol import TokenPrices
    from analysis.report import render_findings

    costs = cost_table(
        _frame().assign(input_tokens=100, cached_input_tokens=50, output_tokens=20),
        TokenPrices(1, 10, 0.1),
    )
    plot_cost_per_solved(costs, tmp_path / "cost.png")
    assert (tmp_path / "cost.png").stat().st_size > 1000
    pd.DataFrame(
        [
            {
                "bucket": "S",
                "arm": "A1",
                "tasks": 2,
                "success": 0.5,
                "success_ci_low": 0.1,
                "success_ci_high": 0.9,
                "function_success": 1.0,
                "median_total_tokens": 1000,
                "mean_search_calls": 2.0,
                "invalid_call_rate": 0.0,
                "exposure_rate": 0.5,
            }
        ]
    ).to_csv(tmp_path / "arm_summary.csv", index=False)
    block = (
        "<!-- BEGIN GENERATED: arm-summary -->\nstale\n"
        "<!-- END GENERATED: arm-summary -->"
    )
    generated = render_findings(block, tmp_path)
    assert "stale" not in generated
    assert "A1" in generated


def test_variance_split_recovers_simulated_components():
    rng = np.random.default_rng(7)
    rows = []
    for task in range(400):
        effect = rng.normal(0.0, 0.6)  # between-task SD of the difference
        base = rng.normal(10.0, 1.0)
        for sample in range(3):
            rows.append((f"t{task}", "A1", sample, base + rng.normal(0, 0.5)))
            rows.append((f"t{task}", "A3", sample, base + effect + rng.normal(0, 0.5)))
    frame = pd.DataFrame(rows, columns=["task_id", "arm", "sample", "log_total_tokens"])
    split = variance_split(frame, "A3", "A1")
    # between 0.36, within 0.25 + 0.25: total 0.86, within share 0.58.
    assert split.tasks == 400
    assert split.paired_sd == pytest.approx(math.sqrt(0.86), rel=0.08)
    assert split.within_fraction == pytest.approx(0.5 / 0.86, abs=0.06)


def test_variance_split_needs_repeated_samples():
    frame = pd.DataFrame(
        [
            ("a", "A1", 0, 1.0),
            ("a", "A3", 0, 2.0),
            ("b", "A1", 0, 1.0),
            ("b", "A3", 0, 3.0),
        ],
        columns=["task_id", "arm", "sample", "log_total_tokens"],
    )
    with pytest.raises(ValueError, match="k >= 2"):
        variance_split(frame, "A3", "A1")


def test_variance_block_and_holm_adjusted_tokens(tmp_path):
    from analysis.report import render_findings

    rng = np.random.default_rng(3)
    rows = [
        {"task_id": f"t{task}", "arm": arm, "sample": sample,
         "log_total_tokens": 10 + 0.1 * task + rng.normal(0, 0.5)}
        for task in range(30)
        for arm in ("A1", "A2", "A3", "A4")
        for sample in range(3)
    ]  # fmt: skip
    pd.DataFrame(rows).to_csv(tmp_path / "episodes.csv", index=False)
    pd.DataFrame(
        [
            {"metric": "log_total_tokens", "subset": "all", "bucket": bucket,
             "contrast": "A4-A3", "estimate": -0.088, "ci_low": -0.153,
             "ci_high": -0.021, "p_value": 0.0093, "n_tasks": 120,
             "p_holm": holm_p}
            for bucket, holm_p in (("all", 0.0281), ("S", float("nan")))
        ]
    ).to_csv(tmp_path / "contrasts.csv", index=False)  # fmt: skip
    document = "".join(
        f"<!-- BEGIN GENERATED: {name} -->\nstale\n<!-- END GENERATED: {name} -->\n"
        for name in ("variance", "tokens")
    )
    generated = render_findings(document, tmp_path)
    assert "stale" not in generated
    variance_rows = [line for line in generated.splitlines() if "| 30 |" in line]
    assert [line.split(" | ")[0] for line in variance_rows] == [
        "| A2-A1",
        "| A3-A2",
        "| A4-A3",
    ]
    assert "| all | A4-A3 | -8.4% [-14.2%, -2.1%] | 0.009 | 0.028 |" in generated
    assert "| S | A4-A3 | -8.4% [-14.2%, -2.1%] | 0.009 |  |" in generated

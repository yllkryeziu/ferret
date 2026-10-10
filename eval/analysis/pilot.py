# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Summarizes a pilot run per arm.

    uv run python -m analysis.pilot --tasks pilot.jsonl --run RUN_DIR \\
        [--input-price DOLLARS --output-price DOLLARS \\
         [--cached-input-price DOLLARS]]

prints one row per arm and writes RUN_DIR/episodes.csv, one row of
`analysis.metrics` per episode. Prices are US dollars per million tokens.
A pilot checks the harness and sizes the main run; it tests nothing, so the
summary reports plain means without intervals.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from analysis.harness_episodes import read_run
from analysis.metrics import metrics_frame
from analysis.protocol import TokenPrices
from tasks.schema import read_tasks


def arm_summary(frame: pd.DataFrame, prices: TokenPrices | None) -> pd.DataFrame:
    """Returns one row per arm of the per-episode metrics in `frame`."""
    grouped = frame.groupby("arm")
    summary = pd.DataFrame(
        {
            "episodes": grouped.size(),
            "success": grouped["success"].mean(),
            "file_mrr": grouped["file_reciprocal_rank"].mean(),
            "submitted": grouped["submitted"].mean(),
            "cap_hit": grouped["cap_hit"].mean(),
            "turns": grouped["turns"].mean(),
            "tool_calls": grouped["tool_calls"].mean(),
            "search_calls": grouped["search_calls"].mean(),
            "invalid_calls": grouped["invalid_tool_calls"].mean(),
            "exposed": grouped["exposure_turn"].count() / grouped.size(),
            "exposure_turns_rm": grouped["restricted_exposure_turns"].mean(),
            "input_tokens": grouped["input_tokens"].mean(),
            "cached_input_tokens": grouped["cached_input_tokens"].mean(),
            "output_tokens": grouped["output_tokens"].mean(),
            "median_tokens": grouped["total_tokens"].median(),
        }
    )
    if prices is not None:
        summary["dollars"] = prices.cost(
            summary["input_tokens"],
            summary["output_tokens"],
            summary["cached_input_tokens"],
        )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the process exit status."""
    args = _parse_args(argv)
    if (args.input_price is None) != (args.output_price is None) or (
        args.cached_input_price is not None and args.input_price is None
    ):
        print(
            "give both --input-price and --output-price, or neither; "
            "--cached-input-price needs them",
            file=sys.stderr,
        )
        return 2
    tasks = {task.id: task for task in read_tasks(args.tasks)}
    frame = metrics_frame(read_run(args.run), tasks)
    frame.to_csv(args.run / "episodes.csv", index=False)
    prices = (
        None
        if args.input_price is None
        else TokenPrices(args.input_price, args.output_price, args.cached_input_price)
    )
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(arm_summary(frame, prices).to_string())
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m analysis.pilot")
    parser.add_argument("--tasks", type=Path, required=True, help="task file")
    parser.add_argument("--run", type=Path, required=True, help="run directory")
    parser.add_argument("--input-price", type=float, help="$ per 1M input tokens")
    parser.add_argument("--output-price", type=float, help="$ per 1M output tokens")
    parser.add_argument(
        "--cached-input-price", type=float, help="$ per 1M cached input tokens"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())

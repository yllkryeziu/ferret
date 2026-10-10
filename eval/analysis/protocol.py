# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""The analysis constants, fixed before the test split runs.

Everything here is frozen, together with the rest of eval/analysis, before
the test split runs; changing any of it afterwards breaks the
pre-registration. Task-side constants live next to the code that uses them
(`tasks.split`, `tasks.paths`).
"""

import dataclasses
import enum


class Arm(enum.StrEnum):
    """Experimental arms; only the output shape of the search tool differs."""

    A0 = "A0"  # Zero-tool: task text plus the top-level listing.
    A1 = "A1"  # raw: path:line:text in path order, 8 KiB cap.
    A2 = "A2"  # line: ranked match lines.
    A3 = "A3"  # context: ranked, ±2 lines, counts, budget.
    A4 = "A4"  # route: A3 plus the structural router.


# The nested ladder of tool arms, in order.
LADDER = (Arm.A1, Arm.A2, Arm.A3, Arm.A4)
# Pre-registered contrasts as (baseline, treatment); each adds one feature.
CONTRASTS = ((Arm.A1, Arm.A2), (Arm.A2, Arm.A3), (Arm.A3, Arm.A4))

SAMPLES_PER_ARM = 3
# Success: every gold file among the first FILE_TOP_K submitted files.
FILE_TOP_K = 5
FUNCTION_TOP_K = 10
# Restricted-mean horizons for time to first gold exposure.
TURN_HORIZON = 40
TOKEN_HORIZON = 400_000


@dataclasses.dataclass(frozen=True, slots=True)
class TokenPrices:
    """US dollars per million tokens; cached input costs the input price
    unless `cached_input_per_million` says otherwise."""

    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None

    def cost(
        self,
        input_tokens: float,
        output_tokens: float,
        cached_input_tokens: float = 0.0,
    ) -> float:
        """Returns the dollar cost of the given token counts, where
        `cached_input_tokens` is the cached part of `input_tokens`."""
        cached_price = (
            self.input_per_million
            if self.cached_input_per_million is None
            else self.cached_input_per_million
        )
        return (
            (input_tokens - cached_input_tokens) * self.input_per_million
            + cached_input_tokens * cached_price
            + output_tokens * self.output_per_million
        ) / 1e6


def contrast_name(baseline: Arm, treatment: Arm) -> str:
    """Returns the label of a contrast, such as "A2-A1"."""
    return f"{treatment}-{baseline}"

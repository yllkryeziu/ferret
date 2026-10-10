# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""The experiment arms.

A1..A3 form a ladder in which each arm changes one aspect of the search
output: A2-A1 isolates ranking and A3-A2 context. They offer the same five
tools and differ only in `ferret search --shape`. A4, which adds the
structural router, arrives with the router. A0 is the zero-tool
contamination probe: the model sees the task and the top-level listing, is
offered only `submit`, and gets one turn.
"""

import dataclasses
import enum
import types
from collections.abc import Mapping

from harness.tools import ALL_TOOLS, SUBMIT, ToolSpec


class Shape(enum.StrEnum):
    """How search results are rendered (`ferret search --shape`, §9.7)."""

    RAW = "raw"
    LINE = "line"
    CONTEXT = "context"


@dataclasses.dataclass(frozen=True, slots=True)
class Arm:
    """One experimental condition.

    Attributes:
        name: "A0" through "A4".
        shape: The search output shape, or None for the zero-tool arm.
    """

    name: str
    shape: Shape | None
    router: bool = False

    @property
    def tools(self) -> tuple[ToolSpec, ...]:
        """The tools offered, in presentation order."""
        return ALL_TOOLS if self.shape is not None else (SUBMIT,)


ARMS: Mapping[str, Arm] = types.MappingProxyType(
    {
        "A0": Arm("A0", shape=None),
        "A1": Arm("A1", shape=Shape.RAW),
        "A2": Arm("A2", shape=Shape.LINE),
        "A3": Arm("A3", shape=Shape.CONTEXT),
        "A4": Arm("A4", shape=Shape.CONTEXT, router=True),
    }
)

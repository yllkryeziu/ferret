# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Loads the model-facing prompts from eval/conf/prompts.yaml.

Prompts live in YAML, mini-swe-agent's configuration format, so the text can
be reviewed on its own; their hash is logged in every trajectory.
"""

import dataclasses
import hashlib
from pathlib import Path
from typing import Self

import yaml

from harness.tools import canonical_json

DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "conf" / "prompts.yaml"


@dataclasses.dataclass(frozen=True, slots=True)
class Prompts:
    """The prompts shared by every arm.

    Attributes:
        system_template: Jinja template of the system message.
        instance_template: Jinja template of the first user message; sees
            `task` and `top_level_listing`.
        forced_submit_message: The user message of the one forced-submit
            turn after a cap is hit.
    """

    system_template: str
    instance_template: str
    forced_submit_message: str

    @classmethod
    def load(cls, path: Path = DEFAULT_PROMPTS_PATH) -> Self:
        """Reads prompts from a YAML file with exactly the three fields.

        Raises:
            ValueError: on a missing, unknown, or non-string field.
        """
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        fields = [field.name for field in dataclasses.fields(cls)]
        if not isinstance(data, dict) or sorted(data) != sorted(fields):
            raise ValueError(f"{path}: expected exactly the fields {fields}")
        if not all(isinstance(data[name], str) for name in fields):
            raise ValueError(f"{path}: every field must be a string")
        return cls(**data)

    def sha256(self) -> str:
        """Returns the SHA-256 of the canonical JSON of all three prompts."""
        encoded = canonical_json(dataclasses.asdict(self)).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

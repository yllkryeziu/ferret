# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""The rewriting model that turns fix commits into bug reports.

`tasks.rewrite` requires a rewriter that differs from the agent model, so a
model that saw the commit message never also solves the task. This one is
gpt-6-sol on the same Azure OpenAI v1 endpoint as the agent, called through
litellm's Responses API with low reasoning effort. Credentials are read from
AZURE_API_KEY and AZURE_API_BASE on each request, as in `models.azure`, and
never stored. The module avoids mini-swe-agent, which task preparation does
not need.
"""

import dataclasses
import hashlib
import os
from collections.abc import Mapping
from typing import Any, Self

import litellm

from models.endpoint import AGENT_MODEL, check_credentials
from tasks.rewrite import REWRITE_PROMPT

DEFAULT_REWRITER_MODEL = "gpt-6-sol"

REASONING_EFFORT = "low"

NUM_RETRIES = 3


@dataclasses.dataclass(slots=True)
class AzureRewriter:
    """A `tasks.rewrite.Rewriter` on the Azure endpoint. Not thread-safe.

    Attributes:
        model: The model name on the endpoint; never the agent model.
        input_tokens: Input tokens used by all calls so far.
        output_tokens: Output tokens used by all calls so far.
    """

    model: str = DEFAULT_REWRITER_MODEL
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.model == AGENT_MODEL:
            raise ValueError(f"the rewriter must not be the agent model {self.model}")

    @classmethod
    def from_environment(
        cls,
        model: str = DEFAULT_REWRITER_MODEL,
        environ: Mapping[str, str] = os.environ,
    ) -> Self:
        """Returns a rewriter after checking that the credentials are set.

        Raises:
            ValueError: naming, but never echoing, the missing variables, or
                if `model` is the agent model.
        """
        check_credentials(environ)
        return cls(model)

    @property
    def rewriter_id(self) -> str:
        """Names the model, its settings, and the prompt, for task provenance."""
        prompt = hashlib.sha256(REWRITE_PROMPT.encode()).hexdigest()[:12]
        return f"azure/{self.model}/effort={REASONING_EFFORT}/prompt={prompt}"

    def __call__(self, prompt: str) -> str:
        """Returns the model's reply to `prompt`, or "" if it gave no text."""
        response = litellm.responses(
            model=f"openai/{self.model}",
            input=prompt,
            reasoning={"effort": REASONING_EFFORT},
            num_retries=NUM_RETRIES,
            api_base=os.environ["AZURE_API_BASE"],
            api_key=os.environ["AZURE_API_KEY"],
        )
        usage = _field(response, "usage")
        if usage is not None:
            self.input_tokens += _field(usage, "input_tokens") or 0
            self.output_tokens += _field(usage, "output_tokens") or 0
        return _output_text(response)


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _output_text(response: Any) -> str:
    """Concatenates the text parts of the response's message items."""
    parts = []
    for item in _field(response, "output") or ():
        if _field(item, "type") != "message":
            continue
        for content in _field(item, "content") or ():
            if _field(content, "type") == "output_text":
                parts.append(_field(content, "text") or "")
    return "".join(parts).strip()

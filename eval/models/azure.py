# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""gpt-6-luna through the Azure OpenAI v1 endpoint, the model of record.

The v1 endpoint speaks the OpenAI API, so requests go through litellm's
Responses API with the `openai/` provider and the endpoint as `api_base`.
Reasoning items come back with each response, and
`ToolCallingResponsesModel` sends them back on every turn. Sampling is the
same in every arm: reasoning effort medium and temperature 1.0 (the model
rejects top_p). `tool_choice` is never sent, so a turn without a tool call
gets the harness's fixed FormatError, as in every arm.

Credentials are read from AZURE_API_KEY and AZURE_API_BASE when each request
is made and are passed to litellm for that request only. They are never part
of the model config, which mini-swe-agent serializes into the trajectory, nor
of a log line.
"""

import copy
import dataclasses
import os
import types
from collections.abc import Mapping, Sequence
from typing import Any, Self

from harness.arms import Arm
from harness.tools import ToolSpec
from models.endpoint import AGENT_MODEL, check_credentials
from models.litellm_models import ToolCallingResponsesModel
from models.toolcalls import OBSERVATION_TEMPLATE, ApiFlavor

DEFAULT_MODEL = AGENT_MODEL

SAMPLING: Mapping[str, Any] = types.MappingProxyType(
    {
        "reasoning": {"effort": "medium"},
        "temperature": 1.0,
    }
)


# Seconds before one model request is abandoned. Without it litellm waits
# far longer, and a stalled HTTPS response hung pilot episodes for over 40
# minutes. mini-swe-agent retries a timed-out request, so the episode
# continues. It is transport, not sampling, so it stays out of SAMPLING and
# the run's pinned configuration.
REQUEST_TIMEOUT_S = 600.0


class AzureResponsesModel(ToolCallingResponsesModel):
    """Sends the Azure credentials with each request. Not thread-safe."""

    def _query(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        return super()._query(
            messages,
            api_base=os.environ["AZURE_API_BASE"],
            api_key=os.environ["AZURE_API_KEY"],
            timeout=REQUEST_TIMEOUT_S,
            **kwargs,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class AzureModelFactory:
    """Builds the Azure model for each episode.

    Attributes:
        model: The model name on the endpoint.
    """

    model: str = DEFAULT_MODEL

    @classmethod
    def from_environment(
        cls, model: str = DEFAULT_MODEL, environ: Mapping[str, str] = os.environ
    ) -> Self:
        """Returns a factory after checking that the credentials are set.

        Raises:
            ValueError: naming, but never echoing, the missing variables.
        """
        check_credentials(environ)
        return cls(model)

    @property
    def model_id(self) -> str:
        """The litellm model id."""
        return f"openai/{self.model}"

    def build(
        self, arm: Arm, episode_id: str, *, tools: Sequence[ToolSpec] | None = None
    ) -> AzureResponsesModel:
        """Returns a fresh model offering `arm`'s tools."""
        del episode_id
        return AzureResponsesModel(
            tools=tools if tools is not None else arm.tools,
            model_name=self.model_id,
            model_kwargs=copy.deepcopy(dict(SAMPLING)),
            # Cost is computed from token counts in the analysis.
            cost_tracking="ignore_errors",
            observation_template=OBSERVATION_TEMPLATE,
        )

    def provenance(self) -> dict[str, Any]:
        """Describes the model for the trajectory; contains no secrets."""
        return {
            "kind": "azure",
            "model_id": self.model_id,
            "api": ApiFlavor.RESPONSES.value,
            "sampling": copy.deepcopy(dict(SAMPLING)),
        }

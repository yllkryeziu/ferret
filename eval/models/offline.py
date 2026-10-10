# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Model-protocol plumbing shared by the models that call no API."""

from collections.abc import Mapping
from typing import Any

from minisweagent.models.utils.actions_toolcall import (
    format_toolcall_observation_messages as format_chat_observations,
)
from minisweagent.models.utils.actions_toolcall_response import (
    format_toolcall_observation_messages as format_responses_observations,
)

from models.toolcalls import OBSERVATION_TEMPLATE, ApiFlavor


class OfflineModel:
    """Base for the scripted and replay models; subclasses define `query`.

    Observations are formatted exactly as the litellm model of the same
    flavor formats them, so offline trajectories have the same shape as live
    ones. Not thread-safe.
    """

    def __init__(self, *, api_flavor: ApiFlavor, config: Mapping[str, Any]) -> None:
        self.api_flavor = api_flavor
        self.config = dict(config)

    def format_message(self, **kwargs: Any) -> dict[str, Any]:
        """Returns the message as given, like mini-swe-agent's litellm models."""
        return dict(kwargs)

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Formats tool outputs as tool-result messages of this flavor."""
        formatter = (
            format_chat_observations
            if self.api_flavor is ApiFlavor.CHAT
            else format_responses_observations
        )
        return formatter(
            actions=message.get("extra", {}).get("actions", []),
            outputs=outputs,
            observation_template=OBSERVATION_TEMPLATE,
            template_vars=template_vars,
        )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        """Returns `kwargs`; offline models contribute no template variables."""
        return dict(kwargs)

    def serialize(self) -> dict[str, Any]:
        """Describes the model for mini-swe-agent's trajectory format."""
        return {
            "info": {
                "config": {
                    "model": dict(self.config),
                    "model_type": f"{type(self).__module__}.{type(self).__name__}",
                }
            }
        }

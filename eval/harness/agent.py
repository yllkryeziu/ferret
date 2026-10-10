# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""mini-swe-agent's DefaultAgent plus the episode caps of the protocol.

An episode ends at the first of: a valid `submit`, the tool-call budget
running out, or the context passing its token cap. In both cap cases the
model gets exactly one more turn, preceded by the same forced-submit message
in every arm, and the episode then ends whether or not it submitted.

The caps are applied in `query`, the hook DefaultAgent documents for this
purpose; its run and step loop is inherited unchanged. The tool-call budget
itself is enforced by `FerretEnvironment`, which sees every call.
"""

from typing import Any

from minisweagent import Model
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import FormatError, LimitsExceeded

from harness.ferret_env import FerretEnvironment

# The `extra.forced_submit` value of the forced-submit message: why it was sent.
FORCED_BY_TOOL_CALLS = "tool_calls"
FORCED_BY_CONTEXT = "context"


class FerretAgent(DefaultAgent):
    """DefaultAgent with the context cap and the forced-submit turn.

    Not thread-safe; create one per episode.
    """

    def __init__(
        self,
        model: Model,
        env: FerretEnvironment,
        *,
        max_context_tokens: int,
        forced_submit_message: str,
        **kwargs: Any,
    ) -> None:
        """Creates the agent.

        Args:
            model: A harness model; its messages carry `extra.usage`.
            env: The episode's environment, which owns the tool-call budget.
            max_context_tokens: Cap on the tokens of the last model call,
                prompt plus completion, which approximates the next prompt.
            forced_submit_message: Text of the forced-submit user message.
            **kwargs: DefaultAgent's AgentConfig fields.
        """
        super().__init__(model, env, **kwargs)
        self.env: FerretEnvironment = env
        self._max_context_tokens = max_context_tokens
        self._forced_submit_message = forced_submit_message
        self._context_tokens = 0
        self._forced = False

    def query(self) -> dict:
        """Applies the caps, then queries the model as DefaultAgent does."""
        if self._forced:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        reason = None
        if self.env.calls_remaining <= 0:
            reason = FORCED_BY_TOOL_CALLS
        elif self._context_tokens >= self._max_context_tokens:
            reason = FORCED_BY_CONTEXT
        if reason is not None:
            self._forced = True
            self.env.end_exploration()
            self.add_messages(
                self.model.format_message(
                    role="user",
                    content=self._forced_submit_message,
                    extra={"forced_submit": reason},
                )
            )
        try:
            message = super().query()
        except FormatError as error:
            self._context_tokens = _context_tokens(error.messages[0])
            raise
        self._context_tokens = _context_tokens(message)
        return message


def _context_tokens(message: dict) -> int:
    usage = message.get("extra", {}).get("usage") or {}
    return usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

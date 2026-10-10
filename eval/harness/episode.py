# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Runs one episode: one task under one arm with one model instance.

The episode wires the task's sandbox and the arm's search backend into a
`FerretEnvironment`, runs the agent to completion, and returns the episode's
fields for the trajectory together with the messages.
"""

import dataclasses
import datetime
import time
from typing import Any, Protocol

from minisweagent import Model

from harness.agent import FerretAgent
from harness.arms import Arm
from harness.ferret_env import FerretEnvironment
from harness.prompts import Prompts
from harness.sandbox import RepoSandbox
from harness.search import SearchBackend
from harness.tools import ToolSpec, tool_set_sha256
from harness.trajectory import FORMAT_VERSION, summarize


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Task:
    """The fields of a task record the harness needs.

    Attributes:
        task_id: Unique task id.
        prompt: The problem statement shown to the model.
        snapshot: Names the snapshot directory and the index directory.
        bucket: The task stratum; bucket "P" runs arm A0 only.
        split: "dev" or "test", if the task file says.
    """

    task_id: str
    prompt: str
    snapshot: str
    bucket: str | None = None
    split: str | None = None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeCaps:
    """The per-episode limits of the protocol, identical in every arm.

    Attributes:
        max_tool_calls: Tool-call budget; A0 gets none.
        max_context_tokens: Context cap, in tokens of the last model call.
        max_consecutive_format_errors: Unparseable responses in a row that
            end the episode (mini-swe-agent's limit).
    """

    max_tool_calls: int = 40
    max_context_tokens: int = 96_000
    max_consecutive_format_errors: int = 3


class HarnessModel(Model, Protocol):
    """A mini-swe-agent Model that also names its message flavor."""

    api_flavor: str


def run_episode(
    *,
    task: Task,
    arm: Arm,
    model: HarnessModel,
    sandbox: RepoSandbox,
    search: SearchBackend | None,
    caps: EpisodeCaps,
    prompts: Prompts,
    tools: tuple[ToolSpec, ...] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Runs the episode to completion.

    Args:
        task: The task.
        arm: The arm; its tools are offered to `model`.
        model: A fresh model for this episode.
        sandbox: The task's snapshot.
        search: The search backend; required iff the arm has search.
        caps: The episode limits.
        prompts: The prompts.

    Returns:
        The episode's trajectory fields and the full message list.

    Raises:
        Exception: whatever the model or the search backend raises beyond
            the failures reported to the model.
    """
    if task.bucket == "P" and arm.name != "A0":
        raise ValueError(f"probe {task.task_id} permits only arm A0")
    if (arm.shape is None) != (search is None):
        raise ValueError(
            f"arm {arm.name}: a search backend is required iff it has search"
        )
    offered = tools if tools is not None else arm.tools
    started_at = datetime.datetime.now(datetime.UTC)
    start = time.monotonic()
    env = FerretEnvironment(
        sandbox=sandbox,
        tools=offered,
        search=search,
        max_tool_calls=caps.max_tool_calls if search is not None else 0,
    )
    agent = FerretAgent(
        model,
        env,
        max_context_tokens=caps.max_context_tokens,
        forced_submit_message=prompts.forced_submit_message,
        system_template=prompts.system_template,
        instance_template=prompts.instance_template,
        step_limit=0,
        cost_limit=0.0,
        max_consecutive_format_errors=caps.max_consecutive_format_errors,
    )
    agent.run(task.prompt, top_level_listing=sandbox.list_dir("."))
    fields = {
        "format_version": FORMAT_VERSION,
        "task_id": task.task_id,
        "snapshot": task.snapshot,
        "arm": arm.name,
        "shape": arm.shape,
        "tools": [tool.name for tool in offered],
        "tool_schema_sha256": tool_set_sha256(offered),
        "server_search_schema_sha256": (
            getattr(search, "schema_sha256", None) if search is not None else None
        ),
        "prompts_sha256": prompts.sha256(),
        "api_flavor": model.api_flavor,
        "search": search.describe() if search is not None else None,
        "caps": dataclasses.asdict(caps),
        "started_at": started_at.isoformat(),
        "wall_time_s": time.monotonic() - start,
        **summarize(agent.messages),
    }
    return fields, agent.messages

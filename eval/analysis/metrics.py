# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Per-episode metrics.

* Success (primary): the agent submitted, and every gold file is among the
  first `FILE_TOP_K` distinct submitted paths. Episodes that never submit
  fail.
* File reciprocal rank: 1/r for the first gold file at rank r among the
  submitted paths, 0 if none.
* Function success: every gold function is among the first `FUNCTION_TOP_K`
  submitted functions; undefined (NaN) for tasks without function gold. A
  submitted "path::name" matches gold "path::qualified.name" when the paths
  are equal and the name equals the qualified name or its last segment.
* Tokens: input and output tokens billed over all turns; turns: model calls.
* Exposure: the first turn whose tool observations contain a gold path as a
  whole path, and the cumulative tokens through that turn. An episode that
  never shows a gold path has no exposure; the restricted means cap the time
  to exposure at `TURN_HORIZON` turns and `TOKEN_HORIZON` tokens, counting
  never-exposed episodes at the cap. That is exact rather than censored,
  because an episode that has ended can no longer expose anything.
* Invalid tool calls: calls the environment rejected before running them.
"""

import dataclasses
import math
import posixpath
import re
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from analysis.protocol import FILE_TOP_K, FUNCTION_TOP_K, TOKEN_HORIZON, TURN_HORIZON
from analysis.trajectory import Episode
from tasks.schema import Task, split_function_id


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeMetrics:
    """The metrics of one episode; see the module docstring for definitions."""

    task_id: str
    bucket: str
    arm: str
    sample: int
    submitted: bool
    success: bool
    file_reciprocal_rank: float
    function_success: float  # 0.0, 1.0, or NaN when the task has no function gold.
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int
    turns: int
    tool_calls: int
    invalid_tool_calls: int
    search_calls: int
    read_calls: int
    cap_hit: bool
    exposure_turn: float  # 1-based, or NaN if never exposed.
    exposure_tokens: float  # Cumulative tokens through exposure, or NaN.
    restricted_exposure_turns: float
    restricted_exposure_tokens: float


def normalize_path(path: str) -> str:
    """Returns `path` as a normalized repository-relative path ("./a//b" -> "a/b")."""
    normalized = posixpath.normpath(path.strip())
    return "" if normalized == "." else normalized


def ranked_files(submitted: Iterable[str]) -> list[str]:
    """Returns the distinct normalized submitted paths, in submission order."""
    seen: dict[str, None] = {}
    for path in submitted:
        seen.setdefault(normalize_path(path), None)
    seen.pop("", None)
    return list(seen)


def file_success(submitted: Sequence[str], gold_paths: Iterable[str]) -> bool:
    """Returns whether every gold path is among the first FILE_TOP_K submitted."""
    return set(gold_paths) <= set(ranked_files(submitted)[:FILE_TOP_K])


def file_reciprocal_rank(submitted: Sequence[str], gold_paths: Iterable[str]) -> float:
    """Returns 1/rank of the first submitted gold path, or 0 if none."""
    gold = set(gold_paths)
    for rank, path in enumerate(ranked_files(submitted), start=1):
        if path in gold:
            return 1.0 / rank
    return 0.0


def function_success(submitted: Sequence[str], gold_functions: Iterable[str]) -> bool:
    """Returns whether every gold function is among the first FUNCTION_TOP_K."""
    candidates = set()
    for function_id in submitted[:FUNCTION_TOP_K]:
        try:
            path, name = split_function_id(function_id)
        except ValueError:
            continue  # A malformed entry matches nothing.
        candidates.add((normalize_path(path), name))
    return all(
        (path, name) in candidates or (path, re.split(r"::|\.", name)[-1]) in candidates
        for path, name in map(split_function_id, gold_functions)
    )


def first_exposure(
    episode: Episode, gold_paths: Iterable[str]
) -> tuple[int, int] | None:
    """Returns (turn, cumulative tokens) when a gold path first appears, or None.

    A path appears when it occurs in an observation not preceded by a path
    character and not followed by one, so "a/foo.c" does not match
    "a/foo.cc" or "xa/foo.c"; a leading "/" or "./" is allowed.
    """
    patterns = [
        re.compile(rf"(?<![\w.-]){re.escape(path)}(?![\w-]|\.\w)")
        for path in gold_paths
    ]
    tokens = 0
    for number, turn in enumerate(episode.turns, start=1):
        tokens += turn.input_tokens + turn.output_tokens
        for call in turn.tool_calls:
            if any(pattern.search(call.observation) for pattern in patterns):
                return number, tokens
    return None


def score_episode(episode: Episode, task: Task) -> EpisodeMetrics:
    """Computes the metrics of `episode` against `task`'s gold."""
    submission = episode.submission
    files = submission.files if submission else ()
    functions = submission.functions if submission else ()
    calls = [call for turn in episode.turns for call in turn.tool_calls]
    input_tokens = sum(turn.input_tokens for turn in episode.turns)
    cached_input_tokens = sum(turn.cached_input_tokens for turn in episode.turns)
    output_tokens = sum(turn.output_tokens for turn in episode.turns)
    exposure = first_exposure(episode, task.gold_paths)
    if exposure is None:
        exposure_turn = exposure_tokens = math.nan
        restricted_turns, restricted_tokens = float(TURN_HORIZON), float(TOKEN_HORIZON)
    else:
        exposure_turn, exposure_tokens = exposure
        restricted_turns = float(min(exposure_turn, TURN_HORIZON))
        restricted_tokens = float(min(exposure_tokens, TOKEN_HORIZON))
    return EpisodeMetrics(
        task_id=task.id,
        bucket=task.bucket.value,
        arm=episode.arm.value,
        sample=episode.sample,
        submitted=submission is not None,
        success=submission is not None and file_success(files, task.gold_paths),
        file_reciprocal_rank=file_reciprocal_rank(files, task.gold_paths),
        function_success=float(function_success(functions, task.gold_functions))
        if task.gold_functions
        else math.nan,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        turns=len(episode.turns),
        tool_calls=len(calls),
        invalid_tool_calls=sum(not call.valid for call in calls),
        search_calls=sum(call.name == "search" for call in calls),
        read_calls=sum(call.name == "read" for call in calls),
        cap_hit=episode.cap_hit,
        exposure_turn=exposure_turn,
        exposure_tokens=exposure_tokens,
        restricted_exposure_turns=restricted_turns,
        restricted_exposure_tokens=restricted_tokens,
    )


def metrics_frame(
    episodes: Iterable[Episode], tasks: Mapping[str, Task]
) -> pd.DataFrame:
    """Scores every episode; one row per episode, columns as in EpisodeMetrics.

    Adds `log_total_tokens`, the scale on which token contrasts are estimated.

    Raises:
        KeyError: if an episode names a task missing from `tasks`.
    """
    rows = [dataclasses.asdict(score_episode(e, tasks[e.task_id])) for e in episodes]
    frame = pd.DataFrame(
        rows, columns=[f.name for f in dataclasses.fields(EpisodeMetrics)]
    )
    frame["log_total_tokens"] = np.log(frame["total_tokens"].clip(lower=1))
    return frame

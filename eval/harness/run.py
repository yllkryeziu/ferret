# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Runs the agent experiment: every (task, arm, sample) episode, resumably.

    cd eval && uv run python -m harness.run --tasks TASKS.jsonl --out RUN_DIR \\
        --data DATA_DIR --ferret FERRET --model azure

DATA_DIR is laid out by `tasks.prepare materialize`. Each trajectory is
written atomically to RUN_DIR/episodes/<episode>.jsonl, so rerunning the
same command resumes an interrupted run: finished episodes are skipped and
unfinished ones start over. RUN_DIR/run.json pins the configuration; a rerun
with a different one is refused instead of being mixed into the same
directory, and a lock keeps two runs out of one directory.

Episodes run in a deterministic pseudo-random order that depends only on the
seed and the episode identities: tasks are shuffled, and each task's arms and
samples are shuffled and scheduled together, so all arms of a task run in the
same window of serving conditions.
"""

import argparse
import concurrent.futures
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

# mini-swe-agent reads these when first imported: no startup banner, and no
# .env from the user's configuration directory, whose settings would leak
# into the run.
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
os.environ.setdefault("MSWEA_GLOBAL_CONFIG_DIR", tempfile.mkdtemp(prefix="mswea-"))

import minisweagent

from harness.arms import ARMS, Arm
from harness.episode import EpisodeCaps, HarnessModel, Task, run_episode
from harness.mcp_client import McpClient, McpError, check_search_canary
from harness.mcp_servers import ManagedServers
from harness.prompts import Prompts
from harness.sandbox import RepoSandbox
from harness.search import FerretCliSearch
from harness.tools import ToolSpec, canonical_json, tool_set_sha256
from harness.trajectory import episode_path, write_episode
from tasks.prepare import index_path, listing_dir, snapshot_dir
from tasks.schema import Bucket, read_tasks

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UNSAFE_ID_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]")


class ModelFactory(Protocol):
    """Makes a fresh model per episode and describes it for provenance."""

    def build(
        self, arm: Arm, episode_id: str, *, tools: Sequence[ToolSpec] | None = None
    ) -> HarnessModel: ...

    def provenance(self) -> dict[str, Any]: ...


class RunDirectoryError(Exception):
    """The output directory holds a run with another configuration, or is in
    use by another run."""


@dataclasses.dataclass(frozen=True, slots=True)
class EpisodeKey:
    """Identifies one episode."""

    task_id: str
    arm: str
    sample: int

    @property
    def episode_id(self) -> str:
        """A file-name-safe, collision-free id, readable when possible."""
        safe = _UNSAFE_ID_CHARACTERS.sub("_", self.task_id)
        if safe != self.task_id:
            safe += "-" + hashlib.sha256(self.task_id.encode("utf-8")).hexdigest()[:8]
        return f"{safe}__{self.arm}__s{self.sample}"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentConfig:
    """Everything that determines a run's episodes except tasks and model.

    Attributes:
        arms: Arm names to run.
        samples: Samples per (task, arm).
        seed: Seeds the episode order.
        data_dir: Holds the snapshots and indexes (`tasks.prepare`).
        ferret: The ferret binary; needed when an arm has search.
        caps: The episode limits.
        prompts: The prompts.
    """

    arms: tuple[str, ...]
    samples: int
    seed: int
    data_dir: Path
    ferret: Path | None
    search_backend: str = "cli"
    caps: EpisodeCaps = dataclasses.field(default_factory=EpisodeCaps)
    prompts: Prompts = dataclasses.field(default_factory=Prompts.load)

    def to_json(self, model: Mapping[str, Any]) -> dict[str, Any]:
        """Returns the configuration pinned in run.json."""
        return {
            "model": dict(model),
            "arms": list(self.arms),
            "samples": self.samples,
            "seed": self.seed,
            "data_dir": str(self.data_dir),
            "ferret": None if self.ferret is None else str(self.ferret),
            "search_backend": self.search_backend,
            "caps": dataclasses.asdict(self.caps),
            "prompts_sha256": self.prompts.sha256(),
            "tool_schema_sha256": {
                arm: tool_set_sha256(ARMS[arm].tools) for arm in self.arms
            },
        }


@dataclasses.dataclass(frozen=True, slots=True)
class RunReport:
    """How many episodes a run finished, skipped as done, and failed."""

    completed: int
    skipped: int
    failed: int


def load_tasks(path: Path, *, split: str | None = None) -> list[Task]:
    """Reads a task file (`tasks.schema`), keeping the fields the harness needs.

    Args:
        path: The task file.
        split: If set, only tasks of this split are returned.

    Raises:
        ValueError: naming the line of a malformed task, or a duplicate id.
    """
    return [
        Task(
            task_id=task.id,
            prompt=task.prompt,
            snapshot=task.snapshot,
            bucket=task.bucket.value,
            split=task.split.value,
        )
        for task in read_tasks(path)
        if split is None or task.split.value == split
    ]


def plan_episodes(
    tasks: Sequence[Task], arms: Sequence[str], samples: int, seed: int
) -> list[EpisodeKey]:
    """Returns every episode to run, in execution order.

    The order depends only on `seed` and the episode keys, so adding tasks
    does not reorder the others. Contamination-probe tasks get arm A0 only.
    """
    plan: list[EpisodeKey] = []
    for task in sorted(tasks, key=lambda task: _rank(seed, "task", task.task_id)):
        if task.bucket == Bucket.P and "A0" not in arms:
            raise ValueError(f"probe {task.task_id} permits only arm A0")
        task_arms = ("A0",) if task.bucket == Bucket.P else arms
        window = [
            EpisodeKey(task.task_id, arm, sample)
            for arm in task_arms
            for sample in range(samples)
        ]
        window.sort(key=lambda key: _rank(seed, "episode", key.episode_id))
        plan.extend(window)
    return plan


def _check_mcp_flags(config: ExperimentConfig) -> None:
    """Rejects incompatible binaries before starting servers or episodes."""
    assert config.ferret is not None
    help_text = subprocess.run(
        [str(config.ferret), "mcp", "--helpfull"],
        capture_output=True,
        text=True,
        check=False,
    )
    listing = help_text.stdout + help_text.stderr
    required = ["--fixed_shape", "--spawn_daemon"]
    if any(ARMS[arm].router for arm in config.arms):
        required.append("--router")
    missing = [flag for flag in required if flag not in listing]
    if missing:
        raise ValueError(
            f"ferret mcp --helpfull missing required flags: {', '.join(missing)}"
        )


def run_experiment(
    config: ExperimentConfig,
    tasks: Sequence[Task],
    factory: ModelFactory,
    out_dir: Path,
    *,
    workers: int,
) -> RunReport:
    """Runs every planned episode that has no trajectory in `out_dir` yet.

    A failed episode is logged and leaves no trajectory, so the next run
    retries it.

    Raises:
        RunDirectoryError: if `out_dir` holds a run with another
            configuration or is in use.
        ValueError: if a snapshot or index is missing.
    """
    if config.search_backend == "mcp" and any(ARMS[arm].shape for arm in config.arms):
        _check_mcp_flags(config)
    plan = plan_episodes(tasks, config.arms, config.samples, config.seed)
    _check_inputs(config, tasks, plan)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunDirectoryError(f"another run is using {out_dir}") from None
        run_config = config.to_json(factory.provenance())
        config_sha256 = _pin_config(out_dir, run_config)
        for stale in (out_dir / "episodes").glob(".*.tmp"):
            stale.unlink()
        pending = [
            key for key in plan if not episode_path(out_dir, key.episode_id).exists()
        ]
        logger.info("%d episodes planned, %d to run", len(plan), len(pending))
        if config.search_backend == "mcp" and pending:
            assert config.ferret is not None
            snapshots = sorted(
                {
                    next(t.snapshot for t in tasks if t.task_id == key.task_id)
                    for key in pending
                    if ARMS[key.arm].shape
                }
            )
            servers = (
                ManagedServers(
                    config.ferret,
                    [snapshot_dir(config.data_dir, s) for s in snapshots],
                    out_dir / "mcp-indexes",
                )
                if snapshots
                else None
            )
        else:
            servers = None
        try:
            if servers is not None:
                servers.warm_up()
            return _run_pending(
                config,
                tasks,
                factory,
                out_dir,
                pending,
                plan,
                config_sha256,
                workers,
                servers,
            )
        finally:
            if servers is not None:
                servers.close()


def _run_pending(
    config: ExperimentConfig,
    tasks: Sequence[Task],
    factory: ModelFactory,
    out_dir: Path,
    pending: list[EpisodeKey],
    plan: list[EpisodeKey],
    config_sha256: str,
    workers: int,
    servers: ManagedServers | None,
) -> RunReport:
    """Runs episodes after the shared servers have started."""
    tool_specs: dict[str, ToolSpec] = {}
    hashes: dict[str, str] = {}
    searched = [key for key in pending if ARMS[key.arm].shape]
    if config.search_backend == "mcp" and searched:
        assert config.ferret is not None
        by_id = {t.task_id: t for t in tasks}
        # Every arm must search every snapshot with model-shaped arguments
        # before any episode runs.
        roots = sorted(
            {
                snapshot_dir(config.data_dir, by_id[key.task_id].snapshot)
                for key in searched
            }
        )
        for name in config.arms:
            arm = ARMS[name]
            if arm.shape is None:
                continue
            for root in roots:
                with McpClient(
                    config.ferret,
                    root,
                    out_dir / "mcp-indexes",
                    arm.shape,
                    router=arm.router,
                    daemon_alive=(lambda root=root: servers.assert_alive(root))
                    if servers
                    else None,
                ) as client:
                    if hashes.setdefault(name, client.schema_sha256) != (
                        client.schema_sha256
                    ):
                        raise ValueError(
                            f"{name} search schema differs across snapshots"
                        )
                    tool_specs[name] = client.tool
                    try:
                        check_search_canary(client, root)
                    except McpError as error:
                        raise ValueError(f"{name}: {error}") from error
        logger.info(
            "search canary passed for %d arms x %d snapshots", len(hashes), len(roots)
        )
        if len(set(hashes.values())) > 1:
            raise ValueError(f"server search tool schema differs across arms: {hashes}")
        schema_path = out_dir / "server-tools.json"
        if schema_path.exists() and json.loads(schema_path.read_text()) != hashes:
            raise ValueError("server search tool schema changed since this run")
        schema_path.write_text(json.dumps(hashes, indent=2) + "\n")
    runner = _EpisodeRunner(
        config=config,
        tasks={task.task_id: task for task in tasks},
        factory=factory,
        out_dir=out_dir,
        run_provenance={"config_sha256": config_sha256, **_harness_provenance()},
        tool_specs=tool_specs,
        servers=servers,
    )
    failed = 0
    finished = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(runner.run, key) for key in pending]
        try:
            for future in concurrent.futures.as_completed(futures):
                finished += 1
                if not future.result():
                    failed += 1
                problem = _health_problem(finished, failed, *runner.search_counts())
                if problem is not None:
                    logger.error("aborting: %s; finishing running episodes", problem)
                    executor.shutdown(wait=True, cancel_futures=True)
                    raise ValueError(f"run aborted: {problem}")
        except KeyboardInterrupt:
            logger.warning("interrupted; finishing running episodes")
            executor.shutdown(wait=True, cancel_futures=True)
            raise
    return RunReport(
        completed=len(pending) - failed, skipped=len(plan) - len(pending), failed=failed
    )


# Circuit breakers for failures no single episode reveals. Valid runs had no
# failed episodes and 0.4% search errors; the first main run, whose every
# search failed, would have stopped at once.
_MIN_FAILED_EPISODES = 5
_MAX_FAILED_FRACTION = 0.1
_MIN_SEARCHES = 50
_MAX_SEARCH_ERROR_FRACTION = 0.25


def _health_problem(
    finished: int, failed: int, searches: int, search_errors: int
) -> str | None:
    """Why the run must stop, or None while it looks healthy."""
    if failed >= _MIN_FAILED_EPISODES and failed > _MAX_FAILED_FRACTION * finished:
        return f"{failed} of {finished} finished episodes failed"
    if (
        searches >= _MIN_SEARCHES
        and search_errors > _MAX_SEARCH_ERROR_FRACTION * searches
    ):
        return f"{search_errors} of {searches} searches returned errors"
    return None


class _EpisodeRunner:
    """Runs and records single episodes. Thread-safe."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        tasks: Mapping[str, Task],
        factory: ModelFactory,
        out_dir: Path,
        run_provenance: Mapping[str, Any],
        tool_specs: Mapping[str, Any] | None = None,
        servers: ManagedServers | None = None,
    ) -> None:
        self._config = config
        self._tasks = tasks
        self._factory = factory
        self._out_dir = out_dir
        self._run_provenance = dict(run_provenance)
        self._tool_specs = tool_specs or {}
        self._servers = servers
        self._sandboxes: dict[str, RepoSandbox] = {}
        self._sandboxes_lock = threading.Lock()
        self._counts_lock = threading.Lock()
        self._searches = 0
        self._search_errors = 0

    def search_counts(self) -> tuple[int, int]:
        """Searches and search errors in the episodes recorded so far."""
        with self._counts_lock:
            return self._searches, self._search_errors

    def run(self, key: EpisodeKey) -> bool:
        """Runs one episode and writes its trajectory; False if it failed."""
        try:
            record, messages = self._run(key)
            write_episode(episode_path(self._out_dir, key.episode_id), record, messages)
        except Exception as error:
            failure = episode_path(self._out_dir, key.episode_id).with_suffix(
                ".failure.json"
            )
            failure.parent.mkdir(parents=True, exist_ok=True)
            failure.write_text(
                canonical_json({"episode_id": key.episode_id, "reason": str(error)})
            )
            logger.exception("episode %s failed", key.episode_id)
            return False
        statuses = [
            call["status"]
            for step in record.get("steps", [])
            for call in step.get("tool_calls", [])
            if call["tool"] == "search"
        ]
        with self._counts_lock:
            self._searches += len(statuses)
            self._search_errors += statuses.count("error")
        logger.info(
            "episode %s: %s after %d model calls",
            key.episode_id,
            record["exit_status"],
            record["totals"]["model_calls"],
        )
        return True

    def _run(self, key: EpisodeKey) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        config = self._config
        task = self._tasks[key.task_id]
        arm = ARMS[key.arm]
        search = None
        if arm.shape is not None:
            assert config.ferret is not None
            if config.search_backend == "mcp":
                search = McpClient(
                    config.ferret,
                    snapshot_dir(config.data_dir, task.snapshot),
                    self._out_dir / "mcp-indexes",
                    arm.shape,
                    router=arm.router,
                    daemon_alive=(
                        lambda: self._servers.assert_alive(
                            snapshot_dir(config.data_dir, task.snapshot)
                        )
                    )
                    if self._servers
                    else None,
                )
            else:
                search = FerretCliSearch(
                    binary=config.ferret,
                    index=index_path(config.data_dir, task.snapshot),
                    shape=arm.shape,
                )
        try:
            tools = (
                tuple(
                    self._tool_specs.get(key.arm, tool)
                    if tool.name == "search"
                    else tool
                    for tool in arm.tools
                )
                if search is not None
                else arm.tools
            )
            fields, messages = run_episode(
                task=task,
                arm=arm,
                model=self._factory.build(arm, key.episode_id, tools=tools),
                sandbox=self._sandbox(task),
                search=search,
                caps=config.caps,
                prompts=config.prompts,
                tools=tools,
            )
        finally:
            if isinstance(search, McpClient):
                search.close()
        record = {
            "episode_id": key.episode_id,
            "sample": key.sample,
            "model": self._factory.provenance(),
            **fields,
            "run": self._run_provenance,
        }
        return record, messages

    def _sandbox(self, task: Task) -> RepoSandbox:
        root = _task_root(self._config.data_dir, task)
        with self._sandboxes_lock:
            if str(root) not in self._sandboxes:
                self._sandboxes[str(root)] = RepoSandbox(root)
            return self._sandboxes[str(root)]


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the process exit status."""
    args = _parse_args(argv)
    if args.dry_run:
        tasks = load_tasks(args.tasks, split=args.split)
        count = len(plan_episodes(tasks, args.arms, args.samples, args.seed))
        print(f"dry run: {len(tasks)} tasks, {count} episodes; no model calls")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    _configure_logging(args.out / "run.log")
    try:
        factory = _make_factory(args)
        config = ExperimentConfig(
            arms=args.arms,
            samples=args.samples,
            seed=args.seed,
            data_dir=args.data,
            ferret=args.ferret,
            search_backend=args.search_backend
            or ("cli" if args.model == "scripted" else "mcp"),
        )
        tasks = load_tasks(args.tasks, split=args.split)
        report = run_experiment(config, tasks, factory, args.out, workers=args.workers)
    except (ValueError, RunDirectoryError) as error:
        logger.error("%s", error)
        return 2
    logger.info(
        "done: %d completed, %d already done, %d failed",
        report.completed,
        report.skipped,
        report.failed,
    )
    return 0 if report.failed == 0 else 1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m harness.run",
        description="Run the agent experiment.",
    )
    parser.add_argument(
        "--tasks", type=Path, required=True, help="task JSON Lines file"
    )
    parser.add_argument("--out", type=Path, required=True, help="run output directory")
    parser.add_argument(
        "--data", type=Path, required=True, help="snapshots and indexes"
    )
    parser.add_argument("--ferret", type=Path, help="the ferret binary")
    parser.add_argument("--search-backend", choices=("mcp", "cli"))
    parser.add_argument(
        "--arms", type=_arm_list, default=tuple(ARMS), help="comma-separated arm names"
    )
    parser.add_argument(
        "--samples", type=int, default=3, help="samples per (task, arm)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="plan without calling a model"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8, help="concurrent episodes")
    parser.add_argument("--split", choices=("dev", "test"), help="run only this split")
    parser.add_argument("--model", choices=("azure", "scripted"), required=True)
    parser.add_argument("--azure-model", help="model name on the Azure endpoint")
    parser.add_argument("--script", type=Path, help="script file for --model=scripted")
    return parser.parse_args(argv)


def _arm_list(text: str) -> tuple[str, ...]:
    arms = tuple(dict.fromkeys(arm.strip() for arm in text.split(",") if arm.strip()))
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown or not arms:
        raise argparse.ArgumentTypeError(f"arms must be among {', '.join(ARMS)}")
    return arms


def _make_factory(args: argparse.Namespace) -> ModelFactory:
    """Builds the chosen model's factory.

    Model modules are imported here rather than at the top because importing
    litellm fetches its model-cost map over the network unless told not to:
    offline runs never import it, and live runs use the map bundled with the
    pinned litellm instead of whatever the remote file says that day.
    """
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    if args.model == "azure":
        from models.azure import DEFAULT_MODEL, AzureModelFactory

        return AzureModelFactory.from_environment(args.azure_model or DEFAULT_MODEL)
    from models.scripted import ScriptedModelFactory

    if args.script is None:
        raise ValueError("--model=scripted needs --script")
    return ScriptedModelFactory.from_file(args.script)


def _configure_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(log_path)],
        force=True,
    )
    # httpx logs every request URL at INFO; litellm is verbose at INFO.
    for name in ("httpx", "LiteLLM"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _task_root(data: Path, task: Task) -> Path:
    """Returns the only files a task's arm may see."""
    return (
        listing_dir(data, task.snapshot)
        if task.bucket == Bucket.P
        else snapshot_dir(data, task.snapshot)
    )


def _check_inputs(
    config: ExperimentConfig, tasks: Sequence[Task], plan: Sequence[EpisodeKey]
) -> None:
    by_id = {task.task_id: task for task in tasks}
    roots = {_task_root(config.data_dir, by_id[key.task_id]) for key in plan}
    searched = {by_id[key.task_id].snapshot for key in plan if ARMS[key.arm].shape}
    problems = [
        f"missing snapshot or listing {root}"
        for root in sorted(roots)
        if not root.is_dir()
    ]
    if searched and config.ferret is None:
        problems.append("arms with search need --ferret")
    problems += [
        f"missing index {index_path(config.data_dir, snapshot)}"
        for snapshot in sorted(searched)
        if config.search_backend == "cli"
        and not index_path(config.data_dir, snapshot).exists()
    ]
    if problems:
        raise ValueError("; ".join(problems))


def _pin_config(out_dir: Path, run_config: Mapping[str, Any]) -> str:
    """Writes run.json, or checks it against `run_config`; returns its hash."""
    config_sha256 = hashlib.sha256(
        canonical_json(run_config).encode("utf-8")
    ).hexdigest()
    manifest = out_dir / "run.json"
    if manifest.exists():
        pinned = json.loads(manifest.read_text(encoding="utf-8"))
        if pinned.get("config_sha256") != config_sha256:
            raise RunDirectoryError(
                f"{out_dir} holds a run with another configuration; see {manifest}"
            )
    else:
        document = {"config_sha256": config_sha256, "config": run_config}
        manifest.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return config_sha256


def _harness_provenance() -> dict[str, Any]:
    return {
        "harness_commit": _git(["rev-parse", "HEAD"]),
        "harness_dirty": _dirty(),
        "minisweagent_version": minisweagent.__version__,
        "litellm_version": importlib.metadata.version("litellm"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def _dirty() -> bool | None:
    status = _git(["status", "--porcelain", "--untracked-files=no"])
    return None if status is None else bool(status)


def _git(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), *arguments],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _rank(seed: int, *parts: str) -> bytes:
    """A seeded sort key that is stable across Python versions and platforms."""
    return hashlib.sha256("\0".join((str(seed), *parts)).encode("utf-8")).digest()


if __name__ == "__main__":
    sys.exit(main())

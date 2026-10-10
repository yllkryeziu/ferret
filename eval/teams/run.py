# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Runs the team episodes: Codex with (F) and without (R) Ferret, one agent at
a time (solo) or four at once in four worktrees of one snapshot (team).

Each episode appends one record to
RUN/episodes.jsonl; its raw Codex events, each stamped with the seconds since
the episode started, are in RUN/episodes/KEY/events.jsonl.

    python -m teams.run --tasks T.jsonl --setting team --out RUN \\
        --worktrees linux=/wt/linux-0,/wt/linux-1,... --worktrees chromium=...
"""

import argparse
import concurrent.futures
import contextlib
import dataclasses
import datetime
import hashlib
import json
import logging
import os
import random
import signal
import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from teams.commands import SEARCH, classify

logger = logging.getLogger(__name__)

ARMS = ("F", "R")
TEAM_SIZE = 4
EPISODE_TIMEOUT_S = 1800
CODEX_HOME = Path.home() / "ferret-data/codex-eval/home"

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "files": {"type": "array", "items": {"type": "string"}},
        "functions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["files", "functions"],
    "additionalProperties": False,
}

PROMPT = """<problem_report>
{report}
</problem_report>

Find where the fix for this problem belongs in this repository and make the fix. \
You cannot build the code or run tests.

When you are done, answer with the files that need to change, most likely first, \
and, if you can, the functions that need to change, each written as path::function.
"""
FERRET_HINT = (
    "\nFor searching code, use the `search` tool of the `ferret` MCP server "
    "instead of grep or rg.\n"
)


@dataclasses.dataclass(frozen=True)
class Episode:
    task: dict[str, Any]
    arm: str
    setting: str
    batch: int
    slot: Path

    @property
    def key(self) -> str:
        task = self.task["id"].replace("/", "_")
        return f"{self.setting}__{task}__{self.arm}"


def render_prompt(task: dict[str, Any], arm: str) -> str:
    return PROMPT.format(report=task["prompt"].strip()) + (
        FERRET_HINT if arm == "F" else ""
    )


def schedule(
    tasks: Sequence[dict[str, Any]],
    setting: str,
    worktrees: dict[str, list[Path]],
    seed: int,
) -> list[list[list[Episode]]]:
    """Returns batches; each batch is one list of concurrent episodes per arm.

    Tasks are shuffled per repository, cut into batches of TEAM_SIZE (team) or
    1 (solo), and batches alternate between repositories. Within a batch both
    arms run back to back, in a seeded random order.
    """
    rng = random.Random(seed)
    size = TEAM_SIZE if setting == "team" else 1
    per_repo: dict[str, list[list[dict[str, Any]]]] = {}
    for repo in sorted({t["repo"] for t in tasks}):
        group = sorted((t for t in tasks if t["repo"] == repo), key=lambda t: t["id"])
        rng.shuffle(group)
        per_repo[repo] = [group[i : i + size] for i in range(0, len(group), size)]
    ordered: list[list[dict[str, Any]]] = []
    while any(per_repo.values()):
        for repo in sorted(per_repo):
            if per_repo[repo]:
                ordered.append(per_repo[repo].pop(0))
    batches = []
    for number, group in enumerate(ordered):
        arms = list(ARMS)
        rng.shuffle(arms)
        batches.append(
            [
                [
                    Episode(task, arm, setting, number, worktrees[task["repo"]][i])
                    for i, task in enumerate(group)
                ]
                for arm in arms
            ]
        )
    return batches


def index_lock(slot: Path) -> Path:
    git_dir = subprocess.run(
        ["git", "-C", str(slot), "rev-parse", "--absolute-git-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(git_dir) / "index.lock"


def reset(slot: Path, attempts: int = 15) -> list[str]:
    """Discards every change an episode made to its worktree.

    A git command that Codex killed at its command timeout can leave
    `index.lock` behind (on Chromium, `git status` can outlast it). The reset
    retries for 30 s, then removes a lock that is still there. Returns what
    it had to do, for the episode record.
    """
    notes = []
    lock = index_lock(slot)
    for attempt in range(attempts):
        result = subprocess.run(
            ["git", "-C", str(slot), "reset", "-q", "--hard"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            break
        notes.append(f"reset failed: {result.stderr.strip().splitlines()[0]}")
        if attempt == attempts - 2 and lock.exists():
            lock.unlink()
            notes.append("removed a stale index.lock")
        time.sleep(2)
    else:
        raise RuntimeError(f"git reset failed in {slot}: {notes}")
    subprocess.run(["git", "-C", str(slot), "clean", "-q", "-f", "-d"], check=True)
    return notes


def codex_argv(episode: Episode, args: argparse.Namespace, out: Path) -> list[str]:
    argv = [
        args.codex,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "-C",
        str(episode.slot),
        "--output-schema",
        str(args.out / "answer_schema.json"),
        "-o",
        str(out / "answer.json"),
    ]
    if episode.arm == "F":
        server = [
            "mcp",
            f"--index_dir={args.index_dir}",
            "--fixed_shape",
            "--shape=context",
            "--router",
            "--spawn_daemon=false",
            str(episode.slot),
        ]
        argv += [
            "-c",
            f"mcp_servers.ferret.command={json.dumps(str(args.ferret))}",
            "-c",
            f"mcp_servers.ferret.args={json.dumps(server)}",
            "-c",
            "mcp_servers.ferret.startup_timeout_sec=120",
            "-c",
            "mcp_servers.ferret.tool_timeout_sec=300",
        ]
    return [*argv, "-"]


def tool_items(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pairs started and completed tool items into timed records."""
    started: dict[str, float] = {}
    items = []
    for stamped in events:
        event, t = stamped["event"], stamped["t"]
        item = event.get("item") or {}
        kind = item.get("type")
        if kind not in {
            "command_execution",
            "mcp_tool_call",
            "file_change",
            "collab_tool_call",
        }:
            continue
        if event.get("type") == "item.started":
            started[item["id"]] = t
            continue
        if event.get("type") != "item.completed":
            continue
        record = {"type": kind, "t0": started.get(item["id"], t), "t1": t}
        if kind == "command_execution":
            record |= {
                "class": classify(item.get("command", "")),
                "command": item.get("command", "")[:500],
                "exit_code": item.get("exit_code"),
            }
        elif kind == "mcp_tool_call":
            record |= {
                "class": SEARCH
                if item.get("tool") in {"search", "find_symbol"}
                else "other",
                "server": item.get("server"),
                "tool": item.get("tool"),
                "arguments": item.get("arguments"),
                "error": bool(item.get("error")),
            }
        else:
            record["class"] = "edit" if kind == "file_change" else "other"
        items.append(record)
    return items


def run_episode(episode: Episode, args: argparse.Namespace) -> dict[str, Any]:
    out = args.out / "episodes" / episode.key
    out.mkdir(parents=True, exist_ok=True)
    reset_notes = reset(episode.slot)
    prompt = render_prompt(episode.task, episode.arm)
    env = {**os.environ, "CODEX_HOME": str(CODEX_HOME)}
    events: list[dict[str, Any]] = []
    started_at = datetime.datetime.now(datetime.UTC).isoformat()
    wall0, mono0 = time.time(), time.monotonic()
    load = os.getloadavg()[0]
    with (out / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(
            codex_argv(episode, args, out),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=env,
            start_new_session=True,
        )
        assert process.stdin and process.stdout
        process.stdin.write(prompt.encode())
        process.stdin.close()
        timed_out = threading.Event()

        def kill() -> None:
            timed_out.set()
            os.killpg(process.pid, signal.SIGKILL)

        timer = threading.Timer(EPISODE_TIMEOUT_S, kill)
        timer.start()
        with (out / "events.jsonl").open("w") as log:
            for line in process.stdout:
                t = time.monotonic() - mono0
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    event = {"type": "unparsed", "line": line.decode(errors="replace")}
                stamped = {"t": round(t, 3), "event": event}
                events.append(stamped)
                log.write(json.dumps(stamped) + "\n")
        exit_code = process.wait()
        timer.cancel()
        # Codex can leave children behind (a shell command, the MCP client).
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
    mono_s, wall_s = time.monotonic() - mono0, time.time() - wall0
    usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    for stamped in events:
        if stamped["event"].get("type") == "turn.completed":
            for name in usage:
                usage[name] += stamped["event"].get("usage", {}).get(name, 0)
    answer, answer_error = None, None
    try:
        answer = json.loads((out / "answer.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        answer_error = str(error)
    changed = subprocess.run(
        [
            "git",
            "-C",
            str(episode.slot),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    (out / "diff.patch").write_text(
        subprocess.run(
            ["git", "-C", str(episode.slot), "diff"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    )
    reset_notes += reset(episode.slot)
    return {
        "key": episode.key,
        "task_id": episode.task["id"],
        "bucket": episode.task["bucket"],
        "repo": episode.task["repo"],
        "arm": episode.arm,
        "setting": episode.setting,
        "batch": episode.batch,
        "slot": str(episode.slot),
        "started_at": started_at,
        "wall_s": round(mono_s, 3),
        # macOS's monotonic clock stops during sleep; the wall clock does not.
        "slept_s": round(max(0.0, wall_s - mono_s), 3),
        "load_at_start": load,
        "exit_code": exit_code,
        "timed_out": timed_out.is_set(),
        "usage": usage,
        "answer": answer,
        "answer_error": answer_error,
        "changed": [line[3:] for line in changed],
        "items": tool_items(events),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "reset_notes": reset_notes,
    }


def run_episode_safely(episode: Episode, args: argparse.Namespace) -> dict[str, Any]:
    """Runs an episode; an exception becomes an infrastructure-failure record,
    which a restarted run reruns."""
    try:
        return run_episode(episode, args)
    except Exception as error:
        logger.exception("episode %s failed", episode.key)
        return {
            "key": episode.key,
            "task_id": episode.task["id"],
            "arm": episode.arm,
            "setting": episode.setting,
            "infra_error": f"{type(error).__name__}: {error}",
        }


def warm_git(worktrees: dict[str, list[Path]]) -> None:
    """Refreshes each worktree's index stat data, so an agent's first
    `git status` does not rehash the tree inside Codex's command timeout."""
    for slots in worktrees.values():
        for slot in slots:
            start = time.monotonic()
            reset(slot)
            subprocess.run(
                ["git", "-C", str(slot), "status", "--porcelain"],
                capture_output=True,
                check=True,
            )
            logger.info("git status %s in %.1f s", slot, time.monotonic() - start)


def resident_kb(pids: Sequence[int]) -> int:
    total = 0
    for pid in pids:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        total += int(out.strip() or 0)
    return total


def start_ferret(args: argparse.Namespace, worktrees: dict[str, list[Path]]) -> Any:
    """Starts one daemon per repository and registers every worktree."""
    from harness.mcp_client import McpClient, wait_until_indexed
    from harness.mcp_servers import ManagedServers

    servers = ManagedServers(
        args.ferret, [slots[0] for slots in worktrees.values()], args.index_dir
    )
    for slots in worktrees.values():
        for slot in slots:
            start = time.monotonic()
            with McpClient(
                args.ferret, slot, args.index_dir, "context", timeout_s=3600
            ) as client:
                wait_until_indexed(client, 3600)
            logger.info("registered %s in %.1f s", slot, time.monotonic() - start)
    return servers


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--setting", choices=("solo", "team"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--worktrees",
        action="append",
        required=True,
        help="REPO=PATH,PATH,... (the first is the solo slot)",
    )
    parser.add_argument(
        "--ferret", type=Path, default=Path.home() / "ferret-data/codex-eval/bin/ferret"
    )
    parser.add_argument(
        "--index_dir", type=Path, default=Path.home() / "ferret-data/codex-eval/index"
    )
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="first N batches only")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    worktrees = {}
    for spec in args.worktrees:
        repo, paths = spec.split("=", 1)
        worktrees[repo] = [Path(p).resolve() for p in paths.split(",")]
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line]
    tasks = [t for t in tasks if t["repo"] in worktrees]
    batches = schedule(tasks, args.setting, worktrees, args.seed)
    if args.limit:
        batches = batches[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "answer_schema.json").write_text(json.dumps(ANSWER_SCHEMA))
    done = set()
    records = args.out / "episodes.jsonl"
    if records.exists():
        for line in records.read_text().splitlines():
            record = json.loads(line) if line else {}
            if record and "infra_error" not in record:
                done.add(record["key"])
    codex_version = subprocess.run(
        [args.codex, "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    (args.out / "run.json").write_text(
        json.dumps(
            {
                "tasks": str(args.tasks),
                "tasks_sha256": sha256_file(args.tasks),
                "setting": args.setting,
                "seed": args.seed,
                "codex": codex_version,
                "codex_config_sha256": sha256_file(CODEX_HOME / "config.toml"),
                "ferret": str(args.ferret),
                "ferret_sha256": sha256_file(args.ferret),
                "worktrees": {k: [str(p) for p in v] for k, v in worktrees.items()},
                "batches": len(batches),
                "argv": list(argv) if argv else None,
            },
            indent=2,
        )
    )

    warm_git(worktrees)
    with start_ferret(args, worktrees) as servers:
        for batch in batches:
            for group in batch:
                group = [e for e in group if e.key not in done]
                if not group:
                    continue
                logger.info(
                    "batch %d arm %s: %s",
                    group[0].batch,
                    group[0].arm,
                    [e.task["id"] for e in group],
                )
                with concurrent.futures.ThreadPoolExecutor(len(group)) as pool:
                    results = list(
                        pool.map(lambda e: run_episode_safely(e, args), group)
                    )
                rss = resident_kb([child.pid for child in servers._children])
                with records.open("a") as log:
                    for record in results:
                        record |= {"concurrency": len(group), "daemon_rss_kb": rss}
                        log.write(json.dumps(record) + "\n")
                        if "infra_error" in record:
                            logger.error("%s: %s", record["key"], record["infra_error"])
                            continue
                        logger.info(
                            "%s exit=%s wall=%.0fs tokens=%d answer=%s",
                            record["key"],
                            record["exit_code"],
                            record["wall_s"],
                            record["usage"]["input_tokens"]
                            + record["usage"]["output_tokens"],
                            bool(record["answer"]),
                        )


if __name__ == "__main__":
    main()

# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Prepares tasks for a run: builds or samples them, then snapshots and
indexes each one.

LocBench tasks are sampled, then materialized:

    uv run python -m tasks.prepare sample --locbench locbench.jsonl \\
        --split dev --count 20 --out pilot.jsonl
    uv run python -m tasks.prepare materialize --tasks pilot.jsonl \\
        --data ~/ferret-data/eval --ferret ../bazel-bin/ferret/cli/ferret

Commit-derived tasks are mined from a local clone (`tasks.commit_tasks`),
materialized, and only then assessed and capped against their snapshot
(`tasks.lexical`), which the lexical oracle reads from the data directory:

    uv run python -m tasks.prepare commits --repo ~/ferret-data/linux.git \\
        --name linux --bucket L --snapshot v7.3-rc1 --head master \\
        --model-cutoff 2026-08-30 --split dev --out linux.jsonl \\
        --rejections linux-rejections.jsonl
    uv run python -m tasks.prepare materialize --tasks linux.jsonl \\
        --data DATA --ferret FERRET --url-template file://$HOME/ferret-data/linux.git
    uv run python -m tasks.prepare lexical --tasks linux.jsonl --data DATA \\
        --out linux-capped.jsonl

`materialize` lays out the data directory the harness reads:

    DATA/cache/<repo>.git        commits fetched at depth 1 (`tasks.fetch`)
    DATA/snapshots/<snapshot>/   history-free snapshots (`tasks.snapshot`)
    DATA/indexes/<snapshot>.fseg Ferret indexes of the snapshots

Every step skips work already done, and each snapshot appears under its
final name only once complete, so an interrupted run resumes by rerunning.
"""

import argparse
import collections
import dataclasses
import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from models.rewriter import DEFAULT_REWRITER_MODEL, AzureRewriter
from tasks.benchmarks import (
    SAMPLE_SALT,
    SWEBENCH_PROBE_TASKS,
    load_locbench,
    load_swebench_verified,
)
from tasks.commit_tasks import CommitTaskBuild, CommitTaskConfig, build_commit_tasks
from tasks.fetch import CommitCache
from tasks.git import Git
from tasks.lexical import LexicalOracle, apply_cap, iter_text_files, prompt_identifiers
from tasks.rewrite import Rewriter
from tasks.schema import Bucket, Split, Task, read_tasks, write_tasks
from tasks.snapshot import export_snapshot, missing_gold_paths
from tasks.split import hash_order

logger = logging.getLogger(__name__)

GITHUB_URL = "https://github.com/{repo}.git"


def sample_tasks(tasks: Sequence[Task], split: Split, count: int) -> list[Task]:
    """Returns a seeded simple random sample of `count` tasks of `split`.

    The sample is a prefix of a fixed permutation, so a larger count only
    adds tasks.
    """
    in_split = [task for task in tasks if task.split is split]
    return hash_order(in_split, key=lambda task: task.id, salt=SAMPLE_SALT)[:count]


def mine_commit_tasks(
    git: Git,
    config: CommitTaskConfig,
    rewriter: Rewriter,
    out: Path,
    rejections: Path,
    answerability_screen: Rewriter | None = None,
    task_ids: frozenset[str] | None = None,
    *,
    dry_run: bool = False,
) -> CommitTaskBuild:
    """Builds commit-derived tasks, writing them to `out` and the rejected
    commits, one JSON object per line, to `rejections`."""
    build = build_commit_tasks(
        git, config, rewriter, answerability_screen, task_ids, dry_run=dry_run
    )
    if dry_run:
        with out.open("w", encoding="utf-8") as lines:
            for candidate in build.tasks:
                lines.write(json.dumps(dataclasses.asdict(candidate)) + "\n")
    else:
        write_tasks(out, build.tasks)
    with rejections.open("w", encoding="utf-8") as lines:
        for rejection in build.rejections:
            lines.write(json.dumps(dataclasses.asdict(rejection)) + "\n")
    return build


def assess_and_cap(tasks: Sequence[Task], data: Path) -> tuple[list[Task], list[Task]]:
    """Sets `lexically_solvable` from each task's materialized snapshot, then
    applies the per-bucket cap on solvable tasks (`tasks.lexical.apply_cap`).

    Returns:
        (kept, dropped).

    Raises:
        FileNotFoundError: if a snapshot has not been materialized.
    """
    by_snapshot: dict[str, list[Task]] = collections.defaultdict(list)
    for task in tasks:
        by_snapshot[task.snapshot].append(task)
    solvable = {}
    for snapshot, group in by_snapshot.items():
        root = snapshot_dir(data, snapshot)
        if not root.is_dir():
            raise FileNotFoundError(f"snapshot {snapshot} is not materialized")
        vocabulary = set().union(*(prompt_identifiers(task.prompt) for task in group))
        oracle = LexicalOracle(iter_text_files(root), vocabulary)
        for task in group:
            solvable[task.id] = oracle.is_solvable(task.prompt, task.gold_paths)
    return apply_cap(
        dataclasses.replace(task, lexically_solvable=solvable[task.id])
        for task in tasks
    )


def snapshot_dir(data: Path, snapshot: str) -> Path:
    """Where the snapshot of commit `snapshot` lives under `data`."""
    return data / "snapshots" / snapshot


def listing_dir(data: Path, snapshot: str) -> Path:
    """Returns the top-level-only listing directory for an A0 probe."""
    return data / "listings" / snapshot


def index_path(data: Path, snapshot: str) -> Path:
    """Where the index of the snapshot of commit `snapshot` lives under `data`."""
    return data / "indexes" / f"{snapshot}.fseg"


def materialize(
    tasks: Sequence[Task],
    data: Path,
    ferret: Path,
    *,
    url_template: str = GITHUB_URL,
) -> None:
    """Fetches, snapshots, and indexes the commit of every task.

    Args:
        tasks: The tasks.
        data: The data directory (module docstring).
        ferret: The ferret binary.
        url_template: Formats a task's repository into its fetch URL.

    Raises:
        ValueError: if a task's gold path does not exist at its snapshot.
        GitError: if fetching or exporting fails.
        subprocess.CalledProcessError: if indexing fails.
    """
    cache = CommitCache(data / "cache")
    for task in tasks:
        destination = snapshot_dir(data, task.snapshot)
        if not destination.exists():
            source = cache.fetch(
                task.repo, url_template.format(repo=task.repo), task.snapshot
            )
            missing = missing_gold_paths(source, task.snapshot, task.gold_paths)
            if missing:
                raise ValueError(
                    f"task {task.id}: gold paths {missing} not in snapshot"
                )
            partial = destination.with_name(f".{destination.name}.partial")
            shutil.rmtree(partial, ignore_errors=True)
            export_snapshot(source, task.snapshot, partial)
            partial.rename(destination)
            logger.info("snapshot %s of %s", task.snapshot, task.repo)
        index = index_path(data, task.snapshot)
        if not index.exists():
            index.parent.mkdir(parents=True, exist_ok=True)
            # `ferret index` writes to a temporary file and renames it.
            subprocess.run(
                [str(ferret), "index", f"--root={destination}", f"--index={index}"],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            logger.info("indexed %s", task.snapshot)


def materialize_listings(tasks: Sequence[Task], data: Path) -> None:
    """Fetches only the probe base trees and exports their top-level names.

    No working-tree blobs are written and no index is built. Re-running skips
    complete listings; the temporary directory is renamed only when ready.
    """
    cache = CommitCache(data / "cache")
    for task in tasks:
        if task.bucket is not Bucket.P:
            raise ValueError(f"task {task.id}: listings are only for bucket P")
        destination = listing_dir(data, task.snapshot)
        if destination.exists():
            continue
        source = cache.fetch(
            task.repo, GITHUB_URL.format(repo=task.repo), task.snapshot
        )
        partial = destination.with_name(f".{destination.name}.partial")
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        entries = source.text("ls-tree", "-z", task.snapshot).split("\0")
        for entry in entries:
            if not entry:
                continue
            metadata, name = entry.split("\t", 1)
            if name.casefold() == ".git":
                continue
            if metadata.startswith("040000 tree "):
                (partial / name).mkdir()
            elif metadata.startswith(("100644 blob ", "100755 blob ", "120000 blob ")):
                (partial / name).touch()
        partial.rename(destination)
        logger.info("listed %s of %s", task.snapshot, task.repo)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the process exit status."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # litellm logs every request at INFO.
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    if args.command == "sample":
        tasks = sample_tasks(_sample_source(args), args.split, args.count)
        if len(tasks) < args.count:
            logger.error("only %d tasks in split %s", len(tasks), args.split)
            return 1
        write_tasks(args.out, tasks)
        logger.info("wrote %d tasks to %s", len(tasks), args.out)
    elif args.command == "commits":
        return _mine(args)
    elif args.command == "probe":
        tasks = load_swebench_verified(args.swebench)
        selected = hash_order(tasks, key=lambda task: task.id, salt=SAMPLE_SALT)[
            :SWEBENCH_PROBE_TASKS
        ]
        if len(selected) < SWEBENCH_PROBE_TASKS:
            raise ValueError(f"only {len(selected)} eligible probe tasks")
        write_tasks(args.out, selected)
    elif args.command == "listings":
        materialize_listings(read_tasks(args.tasks), args.data)
    elif args.command == "materialize":
        materialize(
            read_tasks(args.tasks),
            args.data,
            args.ferret,
            url_template=args.url_template,
        )
    else:
        kept, dropped = assess_and_cap(read_tasks(args.tasks), args.data)
        write_tasks(args.out, kept)
        logger.info(
            "kept %d tasks (%d lexically solvable); dropped %d over the cap",
            len(kept),
            sum(bool(task.lexically_solvable) for task in kept),
            len(dropped),
        )
    return 0


def _sample_source(args: argparse.Namespace) -> list[Task]:
    """Reads the tasks `sample` draws from: LocBench rows or a task file."""
    if args.locbench is not None:
        return load_locbench(args.locbench)
    if args.swebench is not None:
        return load_swebench_verified(args.swebench)
    return read_tasks(args.tasks)


def read_task_ids(path: Path) -> frozenset[str]:
    """Reads one JSON object with a nonempty `id` per line; rejects duplicates."""
    ids: set[str] = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        row = json.loads(line)
        task_id = row.get("id") if isinstance(row, dict) else None
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{path}:{number}: missing task id")
        if task_id in ids:
            raise ValueError(f"{path}:{number}: duplicate task id {task_id}")
        ids.add(task_id)
    return frozenset(ids)


def _mine(args: argparse.Namespace) -> int:
    rewriter = (
        (lambda message: message)
        if args.dry_run
        else AzureRewriter.from_environment(args.rewriter_model)
    )
    config = CommitTaskConfig(
        repo_name=args.name,
        bucket=args.bucket,
        snapshot=args.snapshot,
        model_cutoff=args.model_cutoff,
        rewriter_id="dry-run" if args.dry_run else rewriter.rewriter_id,
        head=args.head,
        split=args.split,
    )
    build = mine_commit_tasks(
        Git(args.repo),
        config,
        rewriter,
        args.out,
        args.rejections,
        AzureRewriter.from_environment(args.answerability_model)
        if args.answerability_model and not args.dry_run
        else None,
        read_task_ids(args.only_task_ids) if args.only_task_ids else None,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(
            f"dry run: {len(build.tasks)} candidates would reach the rewriter; "
            "no model calls"
        )
    reasons = collections.Counter(rejection.reason for rejection in build.rejections)
    logger.info(
        "wrote %d tasks to %s; rejected %s; rewriter used %d input and %d output "
        "tokens",
        len(build.tasks),
        args.out,
        {reason.value: count for reason, count in reasons.most_common()} or "none",
        0 if args.dry_run else rewriter.input_tokens,
        0 if args.dry_run else rewriter.output_tokens,
    )
    return 0


def _date(text: str) -> datetime:
    """Parses an ISO 8601 date or time, taking one without an offset as UTC."""
    value = datetime.fromisoformat(text)
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m tasks.prepare")
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="sample tasks of one split")
    source = sample.add_mutually_exclusive_group(required=True)
    source.add_argument("--locbench", type=Path, help="LocBench rows")
    source.add_argument("--swebench", type=Path, help="SWE-bench Verified rows")
    source.add_argument("--tasks", type=Path, help="a task file")
    sample.add_argument("--split", type=Split, choices=list(Split), required=True)
    sample.add_argument("--count", type=int, required=True)
    sample.add_argument("--out", type=Path, required=True)
    commits = commands.add_parser(
        "commits", help="mine fix commits after a snapshot into tasks"
    )
    commits.add_argument(
        "--dry-run",
        action="store_true",
        help="mine structural candidates and rejections without model calls",
    )
    commits.add_argument("--repo", type=Path, required=True, help="a local clone")
    commits.add_argument("--name", required=True, help="repository name, e.g. linux")
    commits.add_argument("--bucket", type=Bucket, choices=list(Bucket), required=True)
    commits.add_argument("--snapshot", required=True, help="the snapshot revision")
    commits.add_argument("--head", default="HEAD", help="last revision searched")
    commits.add_argument(
        "--model-cutoff",
        type=_date,
        required=True,
        help="fixes authored at or before this date are skipped",
    )
    commits.add_argument("--split", type=Split, choices=list(Split))
    commits.add_argument("--rewriter-model", default=DEFAULT_REWRITER_MODEL)
    commits.add_argument(
        "--only-task-ids", type=Path, help="rewrite only IDs in this task JSONL"
    )
    commits.add_argument(
        "--answerability-model", help="optional second model for YES/NO screen"
    )
    commits.add_argument("--out", type=Path, required=True)
    commits.add_argument("--rejections", type=Path, required=True)
    probe = commands.add_parser("probe", help="sample 50 Verified A0 tasks")
    probe.add_argument("--swebench", type=Path, required=True)
    probe.add_argument("--out", type=Path, required=True)
    listings = commands.add_parser("listings", help="fetch only A0 probe listings")
    listings.add_argument("--tasks", type=Path, required=True)
    listings.add_argument("--data", type=Path, required=True)
    materialize_parser = commands.add_parser(
        "materialize", help="fetch, snapshot, and index the tasks' commits"
    )
    materialize_parser.add_argument("--tasks", type=Path, required=True)
    materialize_parser.add_argument("--data", type=Path, required=True)
    materialize_parser.add_argument("--ferret", type=Path, required=True)
    materialize_parser.add_argument(
        "--url-template",
        default=GITHUB_URL,
        help="formats a task's repo into its fetch URL (default: %(default)s)",
    )
    lexical = commands.add_parser(
        "lexical", help="assess lexical solvability and apply the cap"
    )
    lexical.add_argument("--tasks", type=Path, required=True)
    lexical.add_argument("--data", type=Path, required=True)
    lexical.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())

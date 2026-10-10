# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Replays the agents' searches on other engines.

Arms differ in how search results are shaped, not in which engine finds
them, so comparing engines needs no further LLM episodes: every successful
`search` call in a run is rerun here, after the run, on an otherwise idle
machine.

    uv run python -m harness.replay --run RUN --data DATA \\
        --ferret ../bazel-bin/ferret/cli/ferret --out RUN/replay.csv \\
        [--zoekt zoekt-bench \\
         --ferret_bench ../bazel-bin/bench/ferret/ferret_bench]

Each call is timed warm (one untimed run, then the median of `repeats`) as

  * `ferret_ms`: `ferret search` with the arm's shape, as the agent ran it;
  * `ferret_raw_ms`: `ferret search --shape=raw`, every match in path order;
  * `rg_ms`: ripgrep over the snapshot, every match, in its default
    parallel mode (so in no fixed order);
  * `zoekt_ms`, with `--zoekt`: Zoekt, every match, from a shard index of the
    snapshot (below).

The match sets of the raw runs, as path:line pairs, are compared. ripgrep is
told to search what Ferret indexes: tracked files, hidden ones included,
up to 1 MiB. Ferret also skips files with over 20,000 distinct trigrams
(`ferret/text/eligibility.h`), so matches only ripgrep finds are expected in
generated tables; the CSV keeps counts so they can be traced.

Zoekt is driven through `zoekt-bench` (bench/zoekt), which opens the shards
once and times only the library's Search call. Per snapshot, `ferret_bench
files` lists the files under the `zoekt` eligibility policy, so Zoekt indexes
every listed file, and `zoekt-bench build` indexes them once; the index is
reused while the list's hash is unchanged. All calls of a snapshot then run in
one `zoekt-bench search` process, exhaustive, warmed up once and repeated
`repeats` times. A call with a `path_glob` is scoped to exactly the files
that Ferret's own glob matcher admits (`ferret_bench files --path_glob`),
passed to Zoekt as a file-name set.

A scope is one `files_from` file per distinct glob, made by `ferret_bench
files --from_list` from the snapshot's listing, so no file is read again and
no query repeats a file list. A query Zoekt cannot build (a regex Go's
`regexp/syntax` rejects) has `zoekt_status` "error: ..."; such rows stay in
the CSV, are counted, and are left out of Zoekt's comparisons. With
`--discard_zoekt_indexes`, a snapshot's Zoekt index and scope files are
deleted right after its Zoekt searches, which bounds the disk use by the
largest single index.

Large runs sample whole episodes (`sample_calls`): per snapshot, the episodes
that have a search call, sorted by id and shuffled with `random.Random(seed)`,
are taken in that order while the snapshot's call count stays at most
`--max_calls_per_snapshot` (always at least one episode). The sampled
episode ids, the caps, and the SHA-256 of every binary are recorded in the
provenance file. The raw Ferret run passes `--max_lines` so that no cap
applies, as ripgrep prints every match.

The Zoekt timing favors Zoekt, which is the safe direction for a claim that
Ferret is faster: it has no process start (Ferret and ripgrep are timed as
processes), and the glob matching and set construction happen outside the
timed region. Zoekt runs with GOMAXPROCS 4 and Ferret with one search worker
per performance core plus the calling thread; both are recorded in the
provenance file next to the CSV (`replay_provenance`). A Zoekt result that
reaches its match cap is truncated, not a mismatch.
"""

import argparse
import collections
import csv
import dataclasses
import hashlib
import json
import logging
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from harness.arms import Shape
from harness.search import FerretCliSearch
from harness.trajectory import is_model_turn, is_observation, read_episode
from tasks.prepare import index_path, snapshot_dir

logger = logging.getLogger(__name__)

DEFAULT_REPEATS = 3

# Zoekt's GOMAXPROCS, which also sizes its shard search parallelism; the value
# the earlier systems benchmark used.
ZOEKT_GOMAXPROCS = 4

# Outputs are compared in full, so no output budget applies.
_UNLIMITED_BYTES = 1 << 40
_UNLIMITED_LINES = (1 << 63) - 1

_LOCATION = re.compile(r"^(.+?):(\d+):", re.MULTILINE)


@dataclasses.dataclass(frozen=True, slots=True)
class SearchCall:
    """One successful `search` call of a trajectory.

    Attributes:
        episode_id: The episode it belongs to.
        arm: The episode's arm.
        snapshot: The commit whose tree was searched.
        shape: The arm's output shape.
        arguments: The validated tool arguments.
        latency_s: How long the call took in the run.
    """

    episode_id: str
    arm: str
    snapshot: str
    shape: Shape
    arguments: Mapping[str, Any]
    latency_s: float


def iter_search_calls(run_dir: Path) -> Iterator[SearchCall]:
    """Yields the successful search calls of every trajectory in `run_dir`.

    Each model turn's actions are answered by one observation each, in order.
    """
    for path in sorted((run_dir / "episodes").glob("*.jsonl")):
        record, messages = read_episode(path)
        if record.get("shape") is None:
            continue
        pending: collections.deque[Mapping[str, Any]] = collections.deque()
        for message in messages:
            extra = message.get("extra") or {}
            if is_model_turn(message):
                pending = collections.deque(extra.get("actions") or ())
            elif is_observation(message) and pending:
                action = pending.popleft()
                if action.get("tool") == "search" and extra.get("status") == "ok":
                    yield SearchCall(
                        episode_id=record["episode_id"],
                        arm=record["arm"],
                        snapshot=record["snapshot"],
                        shape=Shape(record["shape"]),
                        arguments=action["arguments"],
                        latency_s=float(extra.get("tool_latency_s", 0.0)),
                    )


def ripgrep_globs(path_glob: str) -> list[str]:
    """Returns the ripgrep `--glob` values that admit exactly the files
    Ferret's `path_glob` admits.

    Ferret follows .gitignore rules (ferret/git/path_glob.h): a file passes
    when the glob matches its path or any directory above it, while ripgrep's
    `--glob` admits only files the glob itself matches. So a glob gives two
    forms: itself (unless a trailing '/' makes it directory-only), and a
    directory form admitting everything below a matching directory. A glob
    with no '/' besides a trailing one matches at any depth, so its directory
    form starts with `**/`; an anchored one does not. A leading '!' applies
    to both forms.
    """
    negate = path_glob.startswith("!")
    body = path_glob[1:] if negate else path_glob
    core = body.rstrip("/")
    forms = [] if body.endswith("/") else [body]
    forms.append((core if "/" in core else f"**/{core}") + "/**")
    return [("!" if negate else "") + form for form in forms]


def ripgrep_argv(rg: Path, arguments: Mapping[str, Any]) -> list[str]:
    """Returns the ripgrep command line equivalent to a `search` call.

    It runs in the snapshot root and prints "path:line:text" lines.
    """
    argv = [
        str(rg),
        "--no-config",
        "--no-ignore",
        "--hidden",
        "--max-filesize=1M",
        "--line-number",
        "--with-filename",
        "--no-heading",
        "--color=never",
    ]
    if not arguments.get("regex"):
        argv.append("--fixed-strings")
    if not arguments.get("case_sensitive"):
        argv.append("--ignore-case")
    if arguments.get("path_glob"):
        argv.extend(f"--glob={g}" for g in ripgrep_globs(arguments["path_glob"]))
    # Last, so that no directory form above can admit files under .git.
    argv.append("--glob=!/.git")
    return [*argv, "--", arguments["pattern"]]


def ferret_argv(ferret: Path, index: Path, call: SearchCall, shape: Shape) -> list[str]:
    """Returns the `ferret search` command line of `call` with `shape`."""
    argv = FerretCliSearch(binary=ferret, index=index, shape=shape).argv(call.arguments)
    if shape is Shape.RAW:
        # The pattern and the "--" before it stay last.
        argv[-2:-2] = [
            f"--max_bytes={_UNLIMITED_BYTES}",
            f"--max_lines={_UNLIMITED_LINES}",
        ]
    return argv


def locations(output: str) -> set[str]:
    """Returns the "path:line" of every "path:line:text" line of `output`."""
    return {f"{path}:{line}" for path, line in _LOCATION.findall(output)}


@dataclasses.dataclass(frozen=True, slots=True)
class Timed:
    """A command's output and its warm median latency."""

    output: str
    returncode: int
    median_ms: float


def time_command(argv: Sequence[str], cwd: Path, repeats: int) -> Timed:
    """Runs `argv` once untimed, then `repeats` times, and keeps the median."""
    # Without a path, ripgrep searches stdin when stdin is a pipe or a file,
    # as it is under a detached launcher, so no command inherits stdin.
    completed = subprocess.run(
        argv, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True, check=False
    )
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        subprocess.run(
            argv, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True, check=False
        )
        samples.append((time.perf_counter() - start) * 1000)
    return Timed(
        completed.stdout.decode("utf-8", "replace"),
        completed.returncode,
        statistics.median(samples),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ZoektConfig:
    """How to replay on Zoekt.

    Attributes:
        zoekt: The `zoekt-bench` executable.
        ferret_bench: The `ferret_bench` executable, which lists the files.
        work: Directory of the per-snapshot file lists and Zoekt indexes.
        discard_indexes: Delete a snapshot's Zoekt index and scope files
            right after its Zoekt searches.
    """

    zoekt: Path
    ferret_bench: Path
    work: Path
    discard_indexes: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class ZoektResult:
    """One call's result on Zoekt.

    Attributes:
        locations: The matching "path:line" pairs.
        median_ms: The warm median of Zoekt's Search call; None on an error.
        truncated: Whether Zoekt stopped early, so `locations` is partial.
        error: Why Zoekt could not run the call, or None.
    """

    locations: frozenset[str]
    median_ms: float | None
    truncated: bool
    error: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ZoektSnapshot:
    """What the Zoekt replay of one snapshot produced.

    Attributes:
        results: The result of every call, keyed by the call's number.
        build_seconds: How long `zoekt-bench build` took (the recorded time
            when an index of the same file list was reused), or None.
    """

    results: Mapping[int, ZoektResult]
    build_seconds: float | None


def _run_checked(argv: Sequence[str]) -> str:
    """Runs `argv` and returns its stdout; raises if it exits non-zero."""
    completed = subprocess.run(
        argv, stdin=subprocess.DEVNULL, capture_output=True, check=False, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{argv[0]} failed: {completed.stderr.strip()}")
    return completed.stdout


def list_files(
    ferret_bench: Path, root: Path, out: Path, path_glob: str | None = None
) -> list[str]:
    """Writes to `out`, and returns, the files of the snapshot at `root` that
    Zoekt indexes, keeping only those `path_glob` admits when it is given."""
    argv = [str(ferret_bench), "files", f"--root={root}", f"--out={out}"]
    if path_glob:
        argv.append(f"--path_glob={path_glob}")
    _run_checked(argv)
    return out.read_text(encoding="utf-8").splitlines()


def filter_listing(ferret_bench: Path, listing: Path, glob: str, out: Path) -> None:
    """Writes to `out` the paths of `listing` that `glob` admits, without
    reading any file (`ferret_bench files --from_list`)."""
    _run_checked(
        [
            str(ferret_bench),
            "files",
            f"--from_list={listing}",
            f"--out={out}",
            f"--path_glob={glob}",
        ]
    )


def ensure_zoekt_index(
    config: ZoektConfig, snapshot: str, root: Path
) -> tuple[Path, float | None]:
    """Returns the Zoekt index directory of a snapshot and the seconds its
    build took, building it unless one built from the same file list (by
    hash) is already there."""
    directory = config.work / snapshot
    directory.mkdir(parents=True, exist_ok=True)
    listing, stamp, index, timing = (
        directory / "files.txt",
        directory / "files.sha256",
        directory / "index",
        directory / "build_seconds.txt",
    )
    list_files(config.ferret_bench, root, listing)
    digest = hashlib.sha256(listing.read_bytes()).hexdigest()
    if index.is_dir() and stamp.is_file() and stamp.read_text() == digest:
        seconds = float(timing.read_text()) if timing.is_file() else None
        return index, seconds
    stamp.unlink(missing_ok=True)
    timing.unlink(missing_ok=True)
    shutil.rmtree(index, ignore_errors=True)
    logger.info("building the Zoekt index of %s", snapshot)
    start = time.perf_counter()
    _run_checked(
        [
            str(config.zoekt),
            "build",
            f"--root={root}",
            f"--files={listing}",
            f"--out={index}",
            f"--name={snapshot}",
        ]
    )
    seconds = time.perf_counter() - start
    stamp.write_text(digest)
    timing.write_text(repr(seconds))
    return index, seconds


def zoekt_query_spec(
    number: int, arguments: Mapping[str, Any], files_from: Path | None
) -> dict[str, Any]:
    """Returns the zoekt-bench query spec of call `number`; `files_from` is
    the scope file of a call with a path glob, or None."""
    spec: dict[str, Any] = {
        "id": str(number),
        "class": "replay",
        "kind": "regex" if arguments.get("regex") else "literal",
        "pattern": arguments["pattern"],
        "case_sensitive": bool(arguments.get("case_sensitive")),
    }
    if files_from is not None:
        spec["files_from"] = str(files_from)
    return spec


def _one_line(message: str, limit: int = 200) -> str:
    return " ".join(message.split())[:limit]


def replay_zoekt_snapshot(
    config: ZoektConfig,
    snapshot: str,
    root: Path,
    calls: Mapping[int, SearchCall],
    repeats: int,
) -> ZoektSnapshot:
    """Replays `calls`, keyed by their number, on Zoekt over one snapshot in a
    single `zoekt-bench search` process.

    A call whose scope cannot be made, or that Zoekt cannot build, gets a
    result with an `error`; the other calls are unaffected.
    """
    index, build_seconds = ensure_zoekt_index(config, snapshot, root)
    directory = (config.work / snapshot).resolve()
    scopes_dir = directory / "scopes"
    scopes_dir.mkdir(exist_ok=True)
    scopes: dict[str, Path | str] = {}
    specs = []
    failed: dict[int, str] = {}
    for number, call in calls.items():
        glob = call.arguments.get("path_glob")
        if glob and glob not in scopes:
            scope = scopes_dir / f"{len(scopes)}.txt"
            try:
                filter_listing(
                    config.ferret_bench, directory / "files.txt", glob, scope
                )
                scopes[glob] = scope
            except RuntimeError as error:
                scopes[glob] = f"scope: {_one_line(str(error))}"
        scope_or_error = scopes[glob] if glob else None
        if isinstance(scope_or_error, str):
            failed[number] = scope_or_error
            continue
        specs.append(zoekt_query_spec(number, call.arguments, scope_or_error))
    results: dict[int, ZoektResult] = {
        number: ZoektResult(frozenset(), None, False, error)
        for number, error in failed.items()
    }
    if specs:
        queries = directory / "queries.jsonl"
        queries.write_text(
            "".join(json.dumps(spec) + "\n" for spec in specs), encoding="utf-8"
        )
        output = _run_checked(
            [
                str(config.zoekt),
                "search",
                f"--index={index}",
                f"--queries={queries}",
                "--mode=exhaustive",
                "--full_results",
                "--context=0",
                f"--gomaxprocs={ZOEKT_GOMAXPROCS}",
                "--warmup=1",
                f"--runs={repeats}",
            ]
        )
        for line in output.splitlines():
            record = json.loads(line)
            if record.get("error"):
                results[int(record["id"])] = ZoektResult(
                    frozenset(), None, False, _one_line(record["error"])
                )
                continue
            # A query without matches has "results": null.
            pairs = record["results"] or ()
            results[int(record["id"])] = ZoektResult(
                locations=frozenset(f"{path}:{number}" for path, number in pairs),
                median_ms=statistics.median(record["latencies_ns"]) / 1e6,
                truncated=bool(record["truncated"]),
            )
    if config.discard_indexes:
        discard_zoekt_snapshot(config, snapshot)
    return ZoektSnapshot(results, build_seconds)


def discard_zoekt_snapshot(config: ZoektConfig, snapshot: str) -> None:
    """Deletes a snapshot's Zoekt index and scope files, which are the bulk of
    its disk use; the file list and the build time stay."""
    directory = config.work / snapshot
    (directory / "files.sha256").unlink(missing_ok=True)
    shutil.rmtree(directory / "index", ignore_errors=True)
    shutil.rmtree(directory / "scopes", ignore_errors=True)
    (directory / "queries.jsonl").unlink(missing_ok=True)


def ferret_search_threads() -> int:
    """Returns the search workers `ferret search` starts on this machine:
    the performance cores, as `PerformanceCoreCount` counts them. The calling
    thread verifies too."""
    if platform.system() == "Darwin":
        for name in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
            probe = subprocess.run(
                ["sysctl", "-n", name], capture_output=True, text=True, check=False
            )
            if probe.returncode == 0 and int(probe.stdout) > 0:
                return int(probe.stdout)
    return os.cpu_count() or 1


def file_sha256(path: Path) -> str:
    """Returns the SHA-256 of the file at `path`, as lowercase hex."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_provenance(binaries: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    """Returns every binary's resolved path and SHA-256."""
    result = {}
    for name, binary in binaries.items():
        resolved = shutil.which(binary) or str(binary)
        result[name] = {
            "path": str(Path(resolved).resolve()),
            "sha256": file_sha256(Path(resolved)),
        }
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class Sample:
    """The calls a replay runs.

    Attributes:
        calls: The sampled calls, grouped by snapshot.
        episodes: The sampled episode ids, per snapshot.
        calls_in_run: Every snapshot's call count in the run.
        episodes_in_run: Every snapshot's count of episodes with a call.
    """

    calls: list[SearchCall]
    episodes: dict[str, list[str]]
    calls_in_run: dict[str, int]
    episodes_in_run: dict[str, int]


def sample_calls(
    calls: Sequence[SearchCall],
    max_calls_per_snapshot: int | None = None,
    seed: int = 0,
) -> Sample:
    """Takes whole episodes per snapshot, at most `max_calls_per_snapshot`
    calls of them (but always at least one episode).

    The episodes of a snapshot that have a call are sorted by id, shuffled
    with `random.Random(seed)`, and taken in that order while the call count
    stays within the cap. Whole episodes keep the projected wall-time ratios
    valid. With no cap, every call is taken.
    """
    by_snapshot: dict[str, dict[str, list[SearchCall]]] = {}
    for call in calls:
        by_snapshot.setdefault(call.snapshot, {}).setdefault(
            call.episode_id, []
        ).append(call)
    sample = Sample([], {}, {}, {})
    for snapshot, episodes in by_snapshot.items():
        ids = sorted(episodes)
        random.Random(seed).shuffle(ids)
        taken: list[str] = []
        total = 0
        for episode_id in ids:
            size = len(episodes[episode_id])
            if (
                max_calls_per_snapshot is not None
                and taken
                and total + size > max_calls_per_snapshot
            ):
                break
            taken.append(episode_id)
            total += size
        sample.episodes[snapshot] = taken
        sample.calls_in_run[snapshot] = sum(map(len, episodes.values()))
        sample.episodes_in_run[snapshot] = len(episodes)
        for episode_id in taken:
            sample.calls.extend(episodes[episode_id])
    return sample


def replay_provenance(
    zoekt: ZoektConfig | None,
    *,
    binaries: Mapping[str, Path] | None = None,
    caps: Mapping[str, Any] | None = None,
    sample: Sample | None = None,
) -> dict[str, Any]:
    """Returns what a reader of the replay CSV needs to interpret it."""
    provenance: dict[str, Any] = {
        "ferret_search_workers": ferret_search_threads(),
        "ferret_timing": "process, including start",
        "rg_timing": "process, including start",
        "ferret_raw": "no cap on lines, files, or bytes",
    }
    if zoekt is not None:
        provenance |= {
            "zoekt_gomaxprocs": ZOEKT_GOMAXPROCS,
            "zoekt_timing": "in-process Search call only; no process start, "
            "and glob matching and file-set construction outside the timed "
            "region, which favors Zoekt",
            "zoekt_scope": "exact file set of Ferret's own path_glob matcher",
            "zoekt_file_policy": "zoekt (ferret_bench files)",
            "zoekt_indexes_discarded": zoekt.discard_indexes,
        }
    if binaries is not None:
        provenance["binaries"] = binary_provenance(binaries)
    if caps is not None:
        provenance["caps"] = dict(caps)
    if sample is not None:
        provenance["snapshots"] = {
            snapshot: {
                "calls_in_run": sample.calls_in_run[snapshot],
                "calls_replayed": sum(
                    call.snapshot == snapshot for call in sample.calls
                ),
                "episodes_in_run": sample.episodes_in_run[snapshot],
                "episodes_sampled": sample.episodes[snapshot],
            }
            for snapshot in sample.episodes
        }
    return provenance


ZOEKT_COLUMNS = (
    "zoekt_status",
    "zoekt_ms",
    "zoekt_matches",
    "zoekt_truncated",
    "zoekt_only_vs_ferret",
    "ferret_only_vs_zoekt",
    "zoekt_only_vs_rg",
    "rg_only_vs_zoekt",
    "zoekt_build_s",
)


def zoekt_columns(
    result: ZoektResult,
    ferret_matches: set[str],
    rg_matches: set[str],
    build_seconds: float | None = None,
) -> dict[str, Any]:
    """Returns the CSV columns of a call's Zoekt result, whose match set is
    compared with the raw Ferret and the ripgrep sets. A call Zoekt could not
    run has its error as `zoekt_status` and no measurements."""
    build = None if build_seconds is None else round(build_seconds, 3)
    if result.error is not None:
        return dict.fromkeys(ZOEKT_COLUMNS) | {
            "zoekt_status": f"error: {result.error}",
            "zoekt_truncated": False,
            "zoekt_build_s": build,
        }
    assert result.median_ms is not None
    return {
        "zoekt_status": "ok",
        "zoekt_ms": round(result.median_ms, 3),
        "zoekt_matches": len(result.locations),
        "zoekt_truncated": result.truncated,
        "zoekt_only_vs_ferret": len(result.locations - ferret_matches),
        "ferret_only_vs_zoekt": len(ferret_matches - result.locations),
        "zoekt_only_vs_rg": len(result.locations - rg_matches),
        "rg_only_vs_zoekt": len(rg_matches - result.locations),
        "zoekt_build_s": build,
    }


def replay(
    calls: Sequence[SearchCall],
    data: Path,
    ferret: Path,
    rg: Path,
    repeats: int = DEFAULT_REPEATS,
    zoekt: ZoektConfig | None = None,
) -> list[dict[str, Any]]:
    """Reruns `calls` with Ferret and ripgrep, and with Zoekt if `zoekt` is
    given; returns one CSV row per call, grouped by snapshot.

    One snapshot is finished (Zoekt first, so its index can be discarded)
    before the next starts, which keeps only one snapshot's results in memory.
    """
    by_snapshot: dict[str, dict[int, SearchCall]] = collections.defaultdict(dict)
    for number, call in enumerate(calls):
        by_snapshot[call.snapshot][number] = call
    rows: list[dict[str, Any]] = []
    for snapshot, snapshot_calls in by_snapshot.items():
        root = snapshot_dir(data, snapshot)
        index = index_path(data, snapshot)
        zoekt_snapshot = (
            replay_zoekt_snapshot(zoekt, snapshot, root, snapshot_calls, repeats)
            if zoekt
            else None
        )
        for number, call in snapshot_calls.items():
            as_run = time_command(
                ferret_argv(ferret, index, call, call.shape), root, repeats
            )
            raw = time_command(
                ferret_argv(ferret, index, call, Shape.RAW), root, repeats
            )
            grep = time_command(ripgrep_argv(rg, call.arguments), root, repeats)
            ferret_matches, rg_matches = locations(raw.output), locations(grep.output)
            row = {
                "snapshot": snapshot,
                "episode_id": call.episode_id,
                "arm": call.arm,
                "pattern": call.arguments["pattern"],
                "regex": bool(call.arguments.get("regex")),
                "case_sensitive": bool(call.arguments.get("case_sensitive")),
                "path_glob": call.arguments.get("path_glob") or "",
                "run_ms": round(call.latency_s * 1000, 3),
                "ferret_ms": round(as_run.median_ms, 3),
                "ferret_raw_ms": round(raw.median_ms, 3),
                "rg_ms": round(grep.median_ms, 3),
                "rg_status": grep.returncode,
                "ferret_matches": len(ferret_matches),
                "rg_matches": len(rg_matches),
                "only_ferret": len(ferret_matches - rg_matches),
                "only_rg": len(rg_matches - ferret_matches),
            }
            if zoekt_snapshot is not None:
                row |= zoekt_columns(
                    zoekt_snapshot.results[number],
                    ferret_matches,
                    rg_matches,
                    zoekt_snapshot.build_seconds,
                )
            rows.append(row)
            if len(rows) % 50 == 0:
                logger.info("replayed %d of %d calls", len(rows), len(calls))
    return rows


def episode_wall_times(run_dir: Path) -> dict[str, float]:
    """Returns the wall time of every episode of `run_dir`, by episode id."""
    wall_times = {}
    for path in sorted((run_dir / "episodes").glob("*.jsonl")):
        record, _ = read_episode(path)
        wall_times[record["episode_id"]] = float(record["wall_time_s"])
    return wall_times


def projected_wall_time_ratios(
    rows: Sequence[Mapping[str, Any]],
    run_dir: Path | Mapping[str, float],
    column: str,
) -> list[float]:
    """Returns, per episode with a comparable call, its wall time had every
    such search taken `column`'s time instead of Ferret's, over its wall
    time. `run_dir` is the run, or its episodes' wall times already read."""
    extra_s: collections.Counter[str] = collections.Counter()
    for row in rows:
        extra_s[row["episode_id"]] += (row[column] - row["ferret_ms"]) / 1000
    wall_times = (
        run_dir if isinstance(run_dir, Mapping) else episode_wall_times(run_dir)
    )
    return [
        (wall_times[episode_id] + extra) / wall_times[episode_id]
        for episode_id, extra in sorted(extra_s.items())
        if episode_id in wall_times
    ]


def zoekt_comparable(row: Mapping[str, Any]) -> bool:
    """Whether Zoekt ran the call to completion: no error, no truncation."""
    return row.get("zoekt_status", "ok") == "ok" and not row["zoekt_truncated"]


def summarize_zoekt(
    rows: Sequence[Mapping[str, Any]], run_dir: Path | Mapping[str, float]
) -> dict[str, Any]:
    """Zoekt's agreement with Ferret and ripgrep, and its projected wall time.

    A truncated result is counted apart: its match set is partial, so it says
    nothing about agreement. A call Zoekt could not run (`zoekt_status` is an
    error) is counted apart too and left out of everything else. ripgrep
    errors are left out of the comparison with ripgrep only.
    """
    ran = [row for row in rows if row.get("zoekt_status", "ok") == "ok"]
    complete = [row for row in ran if zoekt_comparable(row)]
    rg_comparable = [row for row in complete if row["rg_status"] in (0, 1)]
    ratios = projected_wall_time_ratios(ran, run_dir, "zoekt_ms")
    return {
        "errors": len(rows) - len(ran),
        "truncated": len(ran) - len(complete),
        "identical_to_ferret": sum(
            row["zoekt_only_vs_ferret"] == row["ferret_only_vs_zoekt"] == 0
            for row in complete
        ),
        "mismatches_vs_ferret": sum(
            row["zoekt_only_vs_ferret"] != 0 or row["ferret_only_vs_zoekt"] != 0
            for row in complete
        ),
        "mismatches_vs_rg": sum(
            row["zoekt_only_vs_rg"] != 0 or row["rg_only_vs_zoekt"] != 0
            for row in rg_comparable
        ),
        "median_wall_time_ratio_zoekt_over_ferret": statistics.median(ratios)
        if ratios
        else None,
    }


def summarize(rows: Sequence[Mapping[str, Any]], run_dir: Path) -> dict[str, Any]:
    """Latency percentiles, match agreement, and the episode wall time had
    each search taken ripgrep's time (or Zoekt's, if the rows have it)
    instead of Ferret's."""

    def percentiles(column: str) -> dict[str, float]:
        values = [float(row[column]) for row in rows if row[column] is not None]
        if len(values) == 1:
            return dict.fromkeys(("p50", "p90", "p99"), values[0])
        cuts = statistics.quantiles(values, n=100, method="inclusive")
        return {"p50": cuts[49], "p90": cuts[89], "p99": cuts[98]}

    comparable = [row for row in rows if row["rg_status"] in (0, 1)]
    wall_times = episode_wall_times(run_dir)
    ratios = projected_wall_time_ratios(comparable, wall_times, "rg_ms")
    columns = ["run_ms", "ferret_ms", "ferret_raw_ms", "rg_ms"]
    has_zoekt = any("zoekt_ms" in row for row in rows)
    if has_zoekt:
        columns.append("zoekt_ms")
    summary = {
        "calls": len(rows),
        "latency_ms": {column: percentiles(column) for column in columns},
        "rg_errors": len(rows) - len(comparable),
        "identical_match_sets": sum(
            row["only_ferret"] == row["only_rg"] == 0 for row in comparable
        ),
        "median_wall_time_ratio_rg_over_ferret": statistics.median(ratios)
        if ratios
        else None,
    }
    if has_zoekt:
        summary["zoekt"] = summarize_zoekt(rows, wall_times)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the process exit status."""
    parser = argparse.ArgumentParser(prog="python -m harness.replay")
    parser.add_argument("--run", type=Path, required=True, help="run directory")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ferret", type=Path, required=True)
    parser.add_argument("--rg", type=Path, default=Path("rg"))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--zoekt", type=Path, help="zoekt-bench executable; adds Zoekt to the replay"
    )
    parser.add_argument(
        "--ferret_bench",
        type=Path,
        help="ferret_bench executable, which lists Zoekt's files (with --zoekt)",
    )
    parser.add_argument(
        "--max_calls_per_snapshot",
        type=int,
        help="sample whole episodes per snapshot up to this many calls",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of the sample")
    parser.add_argument(
        "--snapshot",
        action="append",
        default=[],
        metavar="SHA",
        help="replay only this snapshot (repeatable); they run in this order",
    )
    parser.add_argument(
        "--discard_zoekt_indexes",
        action="store_true",
        help="delete a snapshot's Zoekt index right after its Zoekt searches",
    )
    parser.add_argument("--out", type=Path, required=True, help="CSV of calls")
    args = parser.parse_args(argv)
    if (args.zoekt is None) != (args.ferret_bench is None):
        parser.error("--zoekt and --ferret_bench go together")
    if args.discard_zoekt_indexes and args.zoekt is None:
        parser.error("--discard_zoekt_indexes needs --zoekt")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    zoekt = (
        ZoektConfig(
            args.zoekt,
            args.ferret_bench,
            args.out.parent / "zoekt-work",
            args.discard_zoekt_indexes,
        )
        if args.zoekt
        else None
    )
    calls = list(iter_search_calls(args.run))
    if args.snapshot:
        rank = {snapshot: i for i, snapshot in enumerate(args.snapshot)}
        calls = sorted(
            (call for call in calls if call.snapshot in rank),
            key=lambda call: rank[call.snapshot],
        )
    sample = sample_calls(calls, args.max_calls_per_snapshot, args.seed)
    logger.info(
        "replaying %d of %d calls in %d snapshots",
        len(sample.calls),
        len(calls),
        len(sample.episodes),
    )
    rows = replay(sample.calls, args.data, args.ferret, args.rg, args.repeats, zoekt)
    if not rows:
        logger.error("no search calls in %s", args.run)
        return 1
    with args.out.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    binaries = {"ferret": args.ferret, "rg": args.rg}
    if zoekt:
        binaries |= {"zoekt-bench": zoekt.zoekt, "ferret_bench": zoekt.ferret_bench}
    caps = {
        "max_calls_per_snapshot": args.max_calls_per_snapshot,
        "seed": args.seed,
        "repeats": args.repeats,
        "snapshots": args.snapshot,
        "run": str(args.run),
    }
    provenance = args.out.with_name(args.out.stem + ".provenance.json")
    provenance.write_text(
        json.dumps(
            replay_provenance(zoekt, binaries=binaries, caps=caps, sample=sample),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summarize(rows, args.run))
    return 0


if __name__ == "__main__":
    sys.exit(main())

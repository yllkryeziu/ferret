# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for harness.replay: rerunning a run's searches on other engines."""

import csv
import dataclasses
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from harness import replay as replay_module
from harness.arms import Shape
from harness.replay import (
    ZOEKT_GOMAXPROCS,
    SearchCall,
    ZoektConfig,
    ferret_argv,
    iter_search_calls,
    locations,
    replay,
    replay_provenance,
    ripgrep_argv,
    sample_calls,
    summarize,
)
from harness.trajectory import FORMAT_VERSION

_RG = shutil.which("rg")


def _write_episode(
    run_dir: Path, episode_id: str, shape, messages, snapshot: str = "a" * 40
) -> None:
    record = {
        "record": "episode",
        "format_version": FORMAT_VERSION,
        "episode_id": episode_id,
        "arm": "A3" if shape else "A0",
        "snapshot": snapshot,
        "shape": shape,
        "wall_time_s": 10.0,
    }
    path = run_dir / "episodes" / f"{episode_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in [record, *messages]))


def _turn(*actions):
    return {"object": "response", "extra": {"actions": list(actions)}}


def _observation(tool, status="ok", latency=0.02):
    return {
        "type": "function_call_output",
        "extra": {"tool": tool, "status": status, "tool_latency_s": latency},
    }


def test_pairs_each_search_with_its_observation(tmp_path):
    search = {"tool": "search", "arguments": {"pattern": "tcp_window"}}
    rejected = {"tool": "search", "arguments": {"pattern": "("}}
    read = {"tool": "read", "arguments": {"path": "net/tcp.c"}}
    _write_episode(
        tmp_path,
        "e1",
        "context",
        [
            {"role": "user", "content": "find it"},
            _turn(read, search),
            _observation("read"),
            _observation("search", latency=0.5),
            _turn(rejected),
            _observation("search", status="error"),
        ],
    )
    _write_episode(tmp_path, "e0", None, [_turn(search), _observation("search")])
    assert list(iter_search_calls(tmp_path)) == [
        SearchCall(
            episode_id="e1",
            arm="A3",
            snapshot="a" * 40,
            shape=Shape.CONTEXT,
            arguments={"pattern": "tcp_window"},
            latency_s=0.5,
        )
    ]


def test_ferret_raw_runs_have_no_output_budget():
    call = SearchCall("e", "A3", "a" * 40, Shape.CONTEXT, {"pattern": "-x"}, 0.0)
    argv = ferret_argv(Path("ferret"), Path("i.fseg"), call, Shape.RAW)
    assert argv[-4:] == [
        f"--max_bytes={1 << 40}",
        f"--max_lines={(1 << 63) - 1}",
        "--",
        "-x",
    ]
    assert "--shape=raw" in argv
    as_run = ferret_argv(Path("ferret"), Path("i.fseg"), call, Shape.CONTEXT)
    assert not any(arg.startswith(("--max_bytes", "--max_lines")) for arg in as_run)


@pytest.mark.skipif(_RG is None, reason="ripgrep is not installed")
def test_ripgrep_searches_what_ferret_indexes(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("Needle\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.c").write_text("x\nNeedle(1)\n")
    (tmp_path / "src" / ".hidden.h").write_text("needle\n")
    (tmp_path / "huge.txt").write_text("needle\n" * (1 << 18))
    (tmp_path / ".gitignore").write_text("*.h\n")

    def run(**arguments):
        argv = ripgrep_argv(Path(_RG), arguments)
        completed = subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True)
        return locations(completed.stdout)

    assert run(pattern="needle") == {"src/a.c:2", "src/.hidden.h:1"}
    assert run(pattern="needle", case_sensitive=True) == {"src/.hidden.h:1"}
    assert run(pattern="Needle(", path_glob="*.c") == {"src/a.c:2"}
    assert run(pattern=r"Needle\(\d\)", regex=True) == {"src/a.c:2"}


@pytest.mark.skipif(_RG is None, reason="ripgrep is not installed")
def test_timed_ripgrep_searches_files_when_stdin_is_a_pipe(tmp_path):
    # Under a detached launcher stdin is a pipe, and ripgrep given no path
    # searches it instead of the directory: every call found nothing.
    (tmp_path / "a.c").write_text("needle\n")
    read, write = os.pipe()
    saved = os.dup(0)
    os.dup2(read, 0)
    try:
        timed = replay_module.time_command(
            ripgrep_argv(Path(_RG), {"pattern": "needle"}), tmp_path, repeats=1
        )
    finally:
        os.dup2(saved, 0)
        for fd in (read, write, saved):
            os.close(fd)
    assert timed.returncode == 0
    assert locations(timed.output) == {"a.c:1"}


def test_locations():
    output = "a/b.c:12:int x;\nc.h:3:y:z\nnot a match\n"
    assert locations(output) == {"a/b.c:12", "c.h:3"}


def test_summarize_projects_wall_time_with_ripgrep(tmp_path):
    _write_episode(tmp_path, "e1", "context", [])
    rows = [
        {
            "episode_id": "e1",
            "run_ms": 20.0,
            "ferret_ms": 10.0,
            "ferret_raw_ms": 8.0,
            "rg_ms": 1010.0,
            "rg_status": status,
            "only_ferret": 0,
            "only_rg": only_rg,
        }
        for status, only_rg in [(0, 0), (1, 2), (2, 0)]
    ]
    summary = summarize(rows, tmp_path)
    assert summary["calls"] == 3
    assert summary["rg_errors"] == 1
    assert summary["identical_match_sets"] == 1
    assert summary["latency_ms"]["rg_ms"]["p50"] == 1010.0
    # Two comparable calls, 1 s slower each, on a 10 s episode.
    assert summary["median_wall_time_ratio_rg_over_ferret"] == pytest.approx(1.2)


# The fake engines read their answers from one JSON table next to them, keyed
# by pattern: {"ferret": {pattern: [[path, line], ...]}, "rg": ..., "zoekt":
# ..., "truncated": [pattern, ...]}, and log every zoekt-bench call.
_FAKE_ENGINE = """\
    import fnmatch, json, pathlib, sys
    here = pathlib.Path(__file__).parent
    table = json.loads((here / "table.json").read_text())
    argv = sys.argv[1:]
    name = pathlib.Path(__file__).name

    def flag(prefix):
        return next((a[len(prefix):] for a in argv if a.startswith(prefix)), None)

    def emit_lines(matches):
        for path, line in matches:
            print(f"{path}:{line}:text")

    if name == "ferret":
        if "--shape=raw" in argv:
            matches = table["ferret"].get(argv[-1], [])
            # Like the real binary, stop at 10,000 lines unless told not to.
            if not any(a.startswith("--max_lines=") for a in argv):
                matches = matches[:10000]
            emit_lines(matches)
    elif name == "rg":
        emit_lines(table["rg"].get(argv[-1], []))
    elif name == "ferret_bench":
        glob = flag("--path_glob=")
        paths = ["a.c", "b.h", "sub/c.c"]
        source = flag("--from_list=")
        if source:
            paths = pathlib.Path(source).read_text().splitlines()
        if glob == "!":
            sys.exit("invalid glob")
        if glob:
            paths = [p for p in paths if fnmatch.fnmatch(p, glob)]
        pathlib.Path(flag("--out=")).write_text("".join(p + "\\n" for p in paths))
        print("{}")
    elif name == "zoekt-bench":
        with (here / "calls.log").open("a") as log:
            log.write(" ".join(a for a in argv if not a.startswith("--") or
                               a.startswith("--gomaxprocs")) + "\\n")
        if argv[0] == "build":
            out = pathlib.Path(flag("--out="))
            out.mkdir(parents=True)
            (out / "shard.zoekt").write_text("x" * 100)
        else:
            for line in pathlib.Path(flag("--queries=")).read_text().splitlines():
                spec = json.loads(line)
                if spec["pattern"] in table.get("zoekt_errors", []):
                    print(json.dumps({
                        "id": spec["id"],
                        "latencies_ns": None,
                        "truncated": False,
                        "error": "query %s: error parsing regexp" % spec["id"],
                    }))
                    continue
                found = table["zoekt"].get(spec["pattern"], [])
                if "files_from" in spec:
                    scope = pathlib.Path(spec["files_from"]).read_text().split()
                    found = [m for m in found if m[0] in scope]
                print(json.dumps({
                    "id": spec["id"],
                    "latencies_ns": [2_000_000, 6_000_000, 4_000_000],
                    "truncated": spec["pattern"] in table["truncated"],
                    "results": found,
                }))
"""


def _fake_engines(directory: Path, table) -> ZoektConfig:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "table.json").write_text(json.dumps(table))
    for name in ("ferret", "rg", "ferret_bench", "zoekt-bench"):
        script = directory / name
        script.write_text(f"#!{sys.executable}\n" + textwrap.dedent(_FAKE_ENGINE))
        script.chmod(0o755)
    return ZoektConfig(
        zoekt=directory / "zoekt-bench",
        ferret_bench=directory / "ferret_bench",
        work=directory / "zoekt-work",
    )


def _call(pattern, **arguments):
    return SearchCall(
        "e1", "A3", "b" * 40, Shape.CONTEXT, {"pattern": pattern, **arguments}, 0.02
    )


def _replay_with_fakes(tmp_path, table, calls, repeats=3):
    config = _fake_engines(tmp_path / "bin", table)
    (tmp_path / "data" / "snapshots" / ("b" * 40)).mkdir(parents=True)
    rows = replay(
        calls,
        tmp_path / "data",
        tmp_path / "bin" / "ferret",
        tmp_path / "bin" / "rg",
        repeats,
        config,
    )
    return config, rows


def test_replay_compares_zoekt_with_ferret_and_ripgrep(tmp_path):
    both = [["a.c", 1], ["sub/c.c", 5]]
    table = {
        "ferret": {"alpha": both, "beta": both, "gamma": both, "delta": both},
        "rg": {"alpha": both, "beta": both, "gamma": both, "delta": both},
        "zoekt": {
            "alpha": [*both, ["b.h", 9]],
            "beta": [*both, ["b.h", 9]],
            "gamma": [["a.c", 1], ["a.c", 2]],
            "delta": both[:1],
        },
        "truncated": ["delta"],
    }
    config, rows = _replay_with_fakes(
        tmp_path,
        table,
        [
            _call("alpha"),
            _call("beta", path_glob="*.c"),
            _call("gamma", regex=True),
            _call("delta"),
        ],
    )
    by_pattern = {row["pattern"]: row for row in rows}

    # An unscoped call sees the extra b.h line Zoekt reports.
    assert by_pattern["alpha"]["zoekt_matches"] == 3
    assert by_pattern["alpha"]["zoekt_only_vs_ferret"] == 1
    assert by_pattern["alpha"]["ferret_only_vs_zoekt"] == 0
    # A scoped call is restricted to the files the glob admits, so the b.h
    # line is dropped and the sets agree.
    beta = by_pattern["beta"]
    assert beta["zoekt_matches"] == 2
    assert (beta["zoekt_only_vs_ferret"], beta["ferret_only_vs_zoekt"]) == (0, 0)
    assert (beta["zoekt_only_vs_rg"], beta["rg_only_vs_zoekt"]) == (0, 0)
    # A mismatch in both directions.
    gamma = by_pattern["gamma"]
    assert (gamma["zoekt_only_vs_ferret"], gamma["ferret_only_vs_zoekt"]) == (1, 1)
    assert (gamma["zoekt_only_vs_rg"], gamma["rg_only_vs_zoekt"]) == (1, 1)
    assert gamma["zoekt_truncated"] is False
    # The median of the fake's three latencies, in milliseconds.
    assert beta["zoekt_ms"] == 4.0
    assert by_pattern["delta"]["zoekt_truncated"] is True

    specs = [
        json.loads(line)
        for line in (config.work / ("b" * 40) / "queries.jsonl")
        .read_text()
        .splitlines()
    ]
    # The scope is a file, not a list repeated in the query.
    assert all("files" not in spec for spec in specs)
    scopes = [
        Path(spec["files_from"]).read_text().split() if "files_from" in spec else None
        for spec in specs
    ]
    assert scopes == [None, ["a.c", "sub/c.c"], None, None]
    assert [spec["kind"] for spec in specs] == [
        "literal",
        "literal",
        "regex",
        "literal",
    ]
    assert not any(spec["case_sensitive"] for spec in specs)


def test_summarize_counts_truncated_results_apart_from_mismatches(tmp_path):
    _write_episode(tmp_path, "e1", "context", [])

    def row(**overrides):
        return {
            "episode_id": "e1",
            "run_ms": 20.0,
            "ferret_ms": 10.0,
            "ferret_raw_ms": 8.0,
            "rg_ms": 20.0,
            "rg_status": 0,
            "only_ferret": 0,
            "only_rg": 0,
            "zoekt_ms": 1010.0,
            "zoekt_truncated": False,
            "zoekt_only_vs_ferret": 0,
            "ferret_only_vs_zoekt": 0,
            "zoekt_only_vs_rg": 0,
            "rg_only_vs_zoekt": 0,
        } | overrides

    rows = [
        row(),
        row(zoekt_only_vs_ferret=1, zoekt_only_vs_rg=1),
        row(zoekt_truncated=True, ferret_only_vs_zoekt=40, rg_only_vs_zoekt=40),
        row(rg_status=2, rg_only_vs_zoekt=3),
    ]
    zoekt = summarize(rows, tmp_path)["zoekt"]
    assert zoekt["truncated"] == 1
    assert zoekt["identical_to_ferret"] == 2
    assert zoekt["mismatches_vs_ferret"] == 1
    # The ripgrep error is left out of the ripgrep comparison.
    assert zoekt["mismatches_vs_rg"] == 1
    # Four calls, 1 s slower each, on a 10 s episode.
    assert zoekt["median_wall_time_ratio_zoekt_over_ferret"] == pytest.approx(1.4)
    assert "zoekt_ms" in summarize(rows, tmp_path)["latency_ms"]


def test_zoekt_index_is_built_once_per_file_list(tmp_path):
    table = {"ferret": {}, "rg": {}, "zoekt": {}, "truncated": []}
    config, _ = _replay_with_fakes(tmp_path, table, [_call("alpha"), _call("beta")])
    log = tmp_path / "bin" / "calls.log"

    def rerun():
        args = (tmp_path / "data", tmp_path / "bin" / "ferret", tmp_path / "bin" / "rg")
        replay([_call("alpha")], *args, 1, config)
        return log.read_text().splitlines().count("build")

    assert rerun() == 1
    # A different file list is a different index.
    (config.work / ("b" * 40) / "files.sha256").write_text("stale")
    assert rerun() == 2


def test_zoekt_runs_with_the_recorded_thread_count(tmp_path):
    table = {"ferret": {}, "rg": {}, "zoekt": {}, "truncated": []}
    _replay_with_fakes(tmp_path, table, [_call("alpha")])
    log = (tmp_path / "bin" / "calls.log").read_text()
    assert f"search --gomaxprocs={ZOEKT_GOMAXPROCS}" in log
    provenance = replay_provenance(ZoektConfig(Path("z"), Path("f"), Path("w")))
    assert provenance["zoekt_gomaxprocs"] == 4
    assert provenance["ferret_search_workers"] >= 1
    assert "favors Zoekt" in provenance["zoekt_timing"]
    assert "zoekt_gomaxprocs" not in replay_provenance(None)


def _calls_of(snapshot, episodes):
    """SearchCalls for {episode_id: number_of_calls}."""
    return [
        SearchCall(episode, "A3", snapshot, Shape.CONTEXT, {"pattern": f"p{i}"}, 0.0)
        for episode, count in episodes.items()
        for i in range(count)
    ]


def test_sampler_takes_whole_episodes_under_the_cap_deterministically():
    episodes = {f"e{i:02d}": 1 + i % 4 for i in range(30)}
    calls = _calls_of("x" * 40, episodes) + _calls_of("y" * 40, {"only": 3})
    sample = sample_calls(calls, max_calls_per_snapshot=20, seed=0)

    taken = sample.episodes["x" * 40]
    # Whole episodes: every call of a taken episode, and no other call.
    assert [c.episode_id for c in sample.calls if c.snapshot == "x" * 40] == [
        episode for episode in taken for _ in range(episodes[episode])
    ]
    total = sum(episodes[episode] for episode in taken)
    assert total <= 20
    # The next episode in the shuffled order would not have fitted.
    shuffled = sorted(episodes)
    random.Random(0).shuffle(shuffled)
    assert taken == shuffled[: len(taken)]
    assert total + episodes[shuffled[len(taken)]] > 20
    # Deterministic, and a different seed picks differently.
    assert sample_calls(calls, 20, 0) == sample
    assert sample_calls(calls, 20, 1).episodes["x" * 40] != taken
    # The counts of the whole run are kept.
    assert sample.calls_in_run == {"x" * 40: sum(episodes.values()), "y" * 40: 3}
    assert sample.episodes_in_run == {"x" * 40: 30, "y" * 40: 1}


def test_sampler_always_takes_one_episode_and_everything_without_a_cap():
    calls = _calls_of("x" * 40, {"big": 9, "bigger": 12})
    sample = sample_calls(calls, max_calls_per_snapshot=5)
    assert len(sample.episodes["x" * 40]) == 1
    assert len(sample.calls) in (9, 12)
    assert len(sample_calls(calls).calls) == 21
    # A cap that holds one episode but not two takes exactly one.
    assert len(sample_calls(calls, 14).calls) in (9, 12)


def test_replay_raw_run_is_not_capped_at_ten_thousand_lines(tmp_path):
    many = [["a.c", i] for i in range(1, 10_501)]
    table = {
        "ferret": {"alpha": many},
        "rg": {"alpha": many},
        "zoekt": {"alpha": many},
        "truncated": [],
    }
    _, rows = _replay_with_fakes(tmp_path, table, [_call("alpha")])
    (row,) = rows
    assert row["ferret_matches"] == row["rg_matches"] == 10_500
    assert row["only_ferret"] == row["only_rg"] == 0
    assert row["ferret_only_vs_zoekt"] == row["zoekt_only_vs_ferret"] == 0


def test_a_query_zoekt_cannot_build_is_a_status_not_a_failure(tmp_path):
    both = [["a.c", 1]]
    table = {
        "ferret": {"alpha": both, "bad": both, "gone": both},
        "rg": {"alpha": both, "bad": both, "gone": both},
        "zoekt": {"alpha": both},
        "truncated": [],
        "zoekt_errors": ["bad"],
    }
    # "gone" has an invalid glob, so no scope can be made for it.
    _, rows = _replay_with_fakes(
        tmp_path,
        table,
        [_call("bad", regex=True), _call("alpha"), _call("gone", path_glob="!")],
    )
    by_pattern = {row["pattern"]: row for row in rows}
    assert len(rows) == 3
    assert by_pattern["alpha"]["zoekt_status"] == "ok"
    assert by_pattern["alpha"]["zoekt_matches"] == 1
    for pattern in ("bad", "gone"):
        row = by_pattern[pattern]
        assert row["zoekt_status"].startswith("error: ")
        assert row["zoekt_ms"] is None
        assert row["zoekt_only_vs_ferret"] is None
        # Ferret and ripgrep were still replayed.
        assert row["ferret_matches"] == row["rg_matches"] == 1
    assert "scope" in by_pattern["gone"]["zoekt_status"]

    _write_episode(tmp_path, "e1", "context", [])
    summary = summarize(rows, tmp_path)
    assert summary["calls"] == 3
    zoekt = summary["zoekt"]
    assert zoekt["errors"] == 2
    assert zoekt["identical_to_ferret"] == 1
    assert zoekt["mismatches_vs_ferret"] == 0
    # The percentiles cover the one call Zoekt ran.
    assert summary["latency_ms"]["zoekt_ms"]["p50"] == 4.0


def test_discarding_zoekt_indexes_deletes_them_after_the_search(tmp_path):
    table = {"ferret": {}, "rg": {}, "zoekt": {}, "truncated": []}
    config = _fake_engines(tmp_path / "bin", table)
    kept = _fake_engines(tmp_path / "bin", table)
    discarding = dataclasses.replace(config, discard_indexes=True)
    (tmp_path / "data" / "snapshots" / ("b" * 40)).mkdir(parents=True)
    args = (tmp_path / "data", tmp_path / "bin" / "ferret", tmp_path / "bin" / "rg")

    replay([_call("alpha", path_glob="*.c")], *args, 1, kept)
    directory = kept.work / ("b" * 40)
    assert (directory / "index" / "shard.zoekt").exists()
    assert list((directory / "scopes").glob("*.txt"))

    shutil.rmtree(kept.work)
    rows = replay([_call("alpha", path_glob="*.c")], *args, 1, discarding)
    assert not (directory / "index").exists()
    assert not (directory / "scopes").exists()
    assert not (directory / "files.sha256").exists()
    # The results and the build time were taken before the delete.
    assert rows[0]["zoekt_status"] == "ok"
    assert rows[0]["zoekt_build_s"] > 0
    # A second replay rebuilds rather than trusting a stamp for a missing index.
    replay([_call("alpha")], *args, 1, discarding)
    assert (tmp_path / "bin" / "calls.log").read_text().splitlines().count("build") == 3


def test_main_samples_restricts_and_records_provenance(tmp_path):
    run = tmp_path / "run"
    snapshot_b, snapshot_c = "b" * 40, "c" * 40
    search = {"tool": "search", "arguments": {"pattern": "alpha"}}
    for snapshot, ids in ((snapshot_b, ("e1", "e2", "e3")), (snapshot_c, ("e4",))):
        for episode_id in ids:
            _write_episode(
                run,
                episode_id,
                "context",
                [_turn(search, search), _observation("search"), _observation("search")],
                snapshot=snapshot,
            )
    table = {"ferret": {}, "rg": {}, "zoekt": {}, "truncated": []}
    config = _fake_engines(tmp_path / "bin", table)
    for snapshot in (snapshot_b, snapshot_c):
        (tmp_path / "data" / "snapshots" / snapshot).mkdir(parents=True)
    out = tmp_path / "out" / "replay.csv"
    out.parent.mkdir()

    status = replay_module.main(
        [
            f"--run={run}",
            f"--data={tmp_path / 'data'}",
            f"--ferret={config.work.parent / 'ferret'}",
            f"--rg={config.work.parent / 'rg'}",
            f"--zoekt={config.zoekt}",
            f"--ferret_bench={config.ferret_bench}",
            "--repeats=1",
            "--max_calls_per_snapshot=4",
            "--seed=3",
            f"--snapshot={snapshot_b}",
            "--discard_zoekt_indexes",
            f"--out={out}",
        ]
    )
    assert status == 0
    with out.open() as handle:
        rows = list(csv.DictReader(handle))
    # Two whole episodes of two calls each, of the one snapshot asked for.
    assert len(rows) == 4
    assert {row["snapshot"] for row in rows} == {snapshot_b}
    assert len({row["episode_id"] for row in rows}) == 2
    provenance = json.loads(out.with_name("replay.provenance.json").read_text())
    sampled = provenance["snapshots"][snapshot_b]
    assert sampled["calls_in_run"] == 6
    assert sampled["calls_replayed"] == 4
    assert sampled["episodes_in_run"] == 3
    assert sorted(sampled["episodes_sampled"]) == sorted(
        {row["episode_id"] for row in rows}
    )
    assert snapshot_c not in provenance["snapshots"]
    assert provenance["caps"]["max_calls_per_snapshot"] == 4
    assert provenance["caps"]["seed"] == 3
    assert provenance["zoekt_indexes_discarded"] is True
    expected = hashlib.sha256((config.work.parent / "ferret").read_bytes()).hexdigest()
    assert provenance["binaries"]["ferret"]["sha256"] == expected
    assert set(provenance["binaries"]) == {
        "ferret",
        "rg",
        "zoekt-bench",
        "ferret_bench",
    }

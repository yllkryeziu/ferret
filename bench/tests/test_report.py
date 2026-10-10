# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  (fixtures are written one per line)
"""Tests of bench/report.py on a small fake results directory."""

import json

import pytest

from bench import report


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def search(engine, query, us, digest="d", agents=1, run=0):
    return {
        "engine": engine,
        "query": query,
        "class": "identifier",
        "us": us,
        "lines": 1,
        "digest": digest,
        "agents": agents,
        "agent": 0,
        "run": run,
    }


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "results"
    directory = root / "linux"
    directory.mkdir(parents=True)
    (directory / "machine.txt").write_text(
        "date: 2026-10-04T12:00:00Z\nferret: abc1234\ncorpus: linux-v7.3-rc1 at cee9395\n"
        "machine: Mac15,12, 16 GiB, 8 cores, macOS 15.6.1\npower: AC Power\n"
        "ripgrep: ripgrep 15.2.0\nzoekt: v0\n"
    )
    rows = [{"register": "/x", "ms": 30000}]
    for q, digest in (("q1", "a"), ("q2", "b")):
        rows += [search("ferret", q, 13000 + r, digest, run=r) for r in range(3)]
        rows += [
            search("ferret_nowait", q, 1000 * (r + 1), digest, run=r) for r in range(3)
        ]
        rows.append(search("ripgrep", q, 2_000_000, digest))
    write_jsonl(directory / "speed.jsonl", rows)
    write_jsonl(
        directory / "speed-zoekt.jsonl",
        [
            {"id": "q1", "latencies_ns": [3e6, 1e6, 2e6], "result_digest": "a"},
            {"id": "q2", "latencies_ns": [5e6, 5e6, 5e6], "result_digest": "other"},
        ],
    )
    (directory / "speed-zoekt-build.json").write_text(
        json.dumps({"files_listed": 95759, "index_bytes": 3 << 30})
    )
    (directory / "speed-ferret-index-bytes").write_text(str(2 << 30))
    agents = [search(e, "q1", us, agents=n) for e, us, n in (
        ("ferret", 14000, 1), ("ferret", 15000, 2), ("ripgrep", 2_000_000, 1), ("ripgrep", 7_000_000, 2),
    )]  # fmt: skip
    write_jsonl(directory / "agents.jsonl", agents)
    return root


def test_percentile_is_nearest_rank():
    assert report.percentile([1, 2, 3, 4], 50) == 2
    assert report.percentile(list(range(1, 201)), 95) == 190
    assert report.percentile([7], 99) == 7


def test_speed_uses_medians_and_compares_digests(corpus):
    s = report.speed(corpus / "linux")
    assert s["times"]["ferret_nowait"] == {"q1": 2.0, "q2": 2.0}
    assert s["times"]["zoekt"]["q1"] == 2.0
    assert s["same"] == {"ferret": 2, "zoekt": 1}
    table = report.speed_table("linux", s)
    assert "| linux | Ferret | Zoekt | ripgrep |" in table
    assert "| Median search | 2.0 ms | 3.5 ms | 2,000 ms |" in table
    assert "| Median, first waiting until recent edits are visible | 13 ms |" in table
    assert "| Same matches as ripgrep | 2 of 2 | 1 of 2 |  |" in table
    assert "| Index size (build times: see 3) | 2.0 GiB | 3.0 GiB | no index |" in table


def test_agents_table_has_one_column_per_count(corpus):
    table = report.agents_table("linux", report.agents(corpus / "linux"))
    assert "| linux: agents at once | 1 | 2 |" in table
    assert "| Ferret, median search | 14 ms | 15 ms |" in table
    assert "| ripgrep, median search | 2,000 ms | 7,000 ms |" in table


def test_errors_refuse_the_page(corpus):
    path = corpus / "linux" / "agents.jsonl"
    path.write_text(
        path.read_text()
        + json.dumps({**search("ripgrep", "q1", 1), "error": "boom"})
        + "\n"
    )
    with pytest.raises(SystemExit, match="boom"):
        report.agents(corpus / "linux")


def test_render_leaves_out_missing_steps(corpus):
    page = report.render(corpus)
    assert "## 1. How long does a search take?" in page
    assert "## 2. Does search slow down" in page
    assert "## 3." not in page and "## 4." not in page
    assert (corpus / "speed.png").exists() and (corpus / "agents-linux.png").exists()


def test_a_sleep_refuses_the_page(corpus):
    (corpus / "linux" / "speed.SLEPT").write_text("2026-10-05 13:14:20 +0200 Sleep\n")
    with pytest.raises(SystemExit, match="slept"):
        report.render(corpus)

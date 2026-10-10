# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  (fixtures are written as one-line tables)
"""Tests of the worktree and freshness table renderers: G3 and the freshness CDF."""

import csv
import json

import pytest

from bench import m3_tables as m3


def g3_line(label, **changes):
    values = {
        "wall_ms": 100, "scan_ms": 10, "read_ms": 40, "ingest_ms": 50,
        "cpu_ms": 300, "bytes_read": 4 << 20, "bytes_ingested": 4 << 20,
        "segment_bytes_written": 2 << 20, "index_bytes_added": 1 << 20, "flush_ms": 7, "compaction_wait_ms": 5, "segments": 3,
    }  # fmt: skip
    values.update(changes)
    return f"G3 {label}: " + " ".join(f"{k}={v}" for k, v in values.items())


def g3_log(count=3, same=True):
    lines = [
        g3_line(
            "k1",
            wall_ms=900,
            bytes_read=100 << 20,
            segment_bytes_written=80 << 20,
            bytes_ingested=100 << 20,
        )
    ]
    for k in range(2, count + 1):
        lines.append(
            g3_line(
                f"k{k}", wall_ms=50, bytes_read=1 << 20, segment_bytes_written=1 << 20
            )
        )
    if same:
        lines.append(
            g3_line(
                "same",
                wall_ms=20,
                bytes_read=0,
                segment_bytes_written=4096,
                index_bytes_added=-4096,
            )
        )
    return "\n".join(["noise line", *lines, "other: wall_ms=5"])  # fmt: skip


def zoekt_rows(count=3):
    return [
        {
            "k": k, "docs": 1000 + 10 * (k - 1), "index_bytes": (50 + k) << 20,
            "wall_seconds": 10.0 + k, "cpu_seconds": 40.0 + k,
            "bytes_if_unshared": 100 * k << 20, "bytes_indexed": 101 << 20,
        }
        for k in range(1, count + 1)
    ]  # fmt: skip


def chain(count=3):
    return {
        "files_per_step": 10,
        "commits": [{"k": 1, "bytes": 0}] + [{"k": k, "bytes": 1 << 20} for k in range(2, count + 1)],
    }  # fmt: skip


def test_percentile_is_nearest_rank():
    values = list(range(1, 101))
    assert m3.percentile(values, 50) == 51
    assert m3.percentile(values, 99) == 100
    assert m3.percentile(values, 90) == 91
    assert m3.percentile([7], 99) == 7
    assert m3.percentile([3, 1, 2], 50) == 2


def test_parse_g3_log_reads_only_g3_lines():
    rows = m3.parse_g3_log(g3_log())
    assert [r["label"] for r in rows] == ["k1", "k2", "k3", "same"]
    assert rows[0]["wall_ms"] == 900


def test_g3_table_has_a_row_per_k_a_same_row_and_zoekt():
    rows = m3.parse_g3_log(g3_log(count=3))
    text = m3.render_g3(rows, zoekt_rows(3), chain(3))
    assert "| k=1 |" in text and "| k=3 |" in text
    assert "| 12 | 3 |" in text  # flush 7 + compaction wait 5
    assert "| same commit as k=1 |" in text
    assert "writes 4096 bytes of segments" in text
    assert "## Zoekt multi-branch" in text
    assert "| 3 | 1020 | 10 |" in text  # k, docs, added docs
    assert "10 / 1.0" in text  # the diff of k=2: 10 files, 1 MiB
    # Bytes added per changed byte: 1.00 for both later rows.
    assert "min 1.00, max 1.00" in text


def test_g3_rejects_rows_out_of_order_or_missing():
    rows = m3.parse_g3_log(g3_log(count=3))
    with pytest.raises(SystemExit):
        m3.render_g3([rows[0], rows[2]], [], None)
    with pytest.raises(SystemExit):
        m3.render_g3([], [], None)
    broken = zoekt_rows(3)
    broken[1]["k"] = 5
    with pytest.raises(SystemExit):
        m3.render_g3(rows, broken, None)


def test_g3_table_without_zoekt_or_chain():
    text = m3.render_g3(m3.parse_g3_log(g3_log(2, same=False)), [], None)
    assert "## Zoekt" not in text
    assert "| k=2 | n/a |" in text


FRESH = "\n".join(
    [f"FRESH ferret_notify trial={i} total_us={200 + i} notify_us=10 apply_us={150 + i} search_us=40" for i in range(10)]
    + [f"FRESH ferret_watcher trial={i} total_us={11000 + 100 * i} event_us=10500 apply_us=300 search_us=100" for i in range(10)]
    + ["unrelated line"]
)  # fmt: skip


def zoekt_fresh(n=10):
    return [
        {"trial": i, "engine": "zoekt", "commit_ns": 5_000_000, "build_ns": 20_000_000 + i * 1_000_000,
         "reload_ns": 2_000_000, "total_ns": 27_000_000 + i * 1_000_000}
        for i in range(n)
    ]  # fmt: skip


def test_freshness_series_and_table():
    series = m3.freshness_series(m3.parse_ferret_fresh(FRESH), zoekt_fresh())
    assert set(series) == {"ferret_notify", "ferret_watcher", "zoekt"}
    assert series["ferret_notify"]["total"][0] == 0.2
    assert series["zoekt"]["total"][9] == 36.0
    assert series["zoekt"]["delta build"][0] == 20.0
    text = m3.render_freshness(series)
    assert "| Ferret Notify | 10 | 0.20 |" in text  # p50 of 200..209 is 205
    assert (
        "| Zoekt commit + delta + reload | 10 | 32.00 | 36.00 | 36.00 | 36.00 |" in text
    )
    assert "| Ferret watcher | write to Ferret's callback (FSEvents) | 10.50 |" in text
    assert "| Zoekt commit + delta + reload | git commit | 5.00 |" in text


def test_warmup_leaves_out_the_first_trials_of_every_path():
    ferret = m3.parse_ferret_fresh(FRESH)
    # Trials in a different order must not change which ones are warm-up.
    zoekt = list(reversed(zoekt_fresh()))
    series = m3.freshness_series(ferret, zoekt, warmup=1)
    assert len(series["ferret_notify"]["total"]) == 9
    assert series["ferret_notify"]["total"][0] == 0.201
    assert series["ferret_watcher"]["total"][0] == 11.1
    assert series["zoekt"]["total"][0] == 28.0
    assert series["zoekt"]["delta build"][0] == 21.0
    dropped = m3.warmup_totals(ferret, zoekt, 1)
    assert dropped == {
        "ferret_notify": [0.2],
        "ferret_watcher": [11.0],
        "zoekt": [27.0],
    }
    text = m3.render_freshness(series, dropped)
    assert "The first 1 trial(s) of each path are warm-up" in text
    assert "Ferret Notify 0.20 ms" in text
    assert "| Ferret Notify | 9 |" in text


def test_warmup_must_leave_trials():
    with pytest.raises(SystemExit):
        m3.freshness_series(m3.parse_ferret_fresh(FRESH), [], warmup=10)
    with pytest.raises(SystemExit):
        m3.freshness_series(m3.parse_ferret_fresh(FRESH), [], warmup=-1)


def test_freshness_p50_matches_the_nearest_rank():
    series = m3.freshness_series(m3.parse_ferret_fresh(FRESH), [])
    text = m3.render_freshness(series)
    # Totals 200..209 us: p50 = index 5 = 205 us = 0.205 ms, shown as 0.21 (or 0.20).
    row = next(
        line for line in text.splitlines() if line.startswith("| Ferret Notify | 10")
    )
    assert row.split("|")[3].strip() in ("0.20", "0.21")
    assert row.split("|")[6].strip() == "0.21"  # max 209 us


def test_cdf_points_are_sorted_and_end_at_one():
    points = m3.cdf_points([5, 1, 3, 3])
    assert points == [(1, 0.25), (3, 0.5), (3, 0.75), (5, 1.0)]
    assert m3.cdf_points([2])[0] == (2, 1.0)


def test_cdf_csv_and_png(tmp_path):
    series = m3.freshness_series(m3.parse_ferret_fresh(FRESH), zoekt_fresh())
    out = tmp_path / "cdf.csv"
    m3.write_cdf_csv(out, series)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 30
    zoekt = [r for r in rows if r["path"] == "zoekt"]
    assert [float(r["cumulative_fraction"]) for r in zoekt][-1] == 1.0
    assert float(zoekt[0]["latency_ms"]) == 27.0
    png = tmp_path / "cdf.png"
    m3.plot_cdf(png, series)
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_freshness_needs_trials():
    with pytest.raises(SystemExit):
        m3.render_freshness({})


def test_cli_end_to_end(tmp_path):
    ferret_log = tmp_path / "g3.log"
    ferret_log.write_text(g3_log())
    zoekt = tmp_path / "z.jsonl"
    zoekt.write_text("\n".join(json.dumps(r) for r in zoekt_rows()) + "\n")
    chain_json = tmp_path / "chain.json"
    chain_json.write_text(json.dumps(chain()))
    out = tmp_path / "out" / "g3.md"
    assert m3.main(["g3", "--ferret-log", str(ferret_log), "--zoekt", str(zoekt),
                    "--chain", str(chain_json), "--out", str(out)]) == 0  # fmt: skip
    assert out.read_text().startswith("# G3")

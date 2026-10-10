# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tasks.lexical: the single-grep oracle and the solvable cap."""

import dataclasses
from datetime import UTC, datetime
from fractions import Fraction

import pytest

from tasks.lexical import (
    LexicalOracle,
    apply_cap,
    assess,
    iter_text_files,
    prompt_identifiers,
)
from tasks.schema import Bucket, PromptOrigin, Provenance, Source, Split, Task


def test_prompt_identifiers():
    # Words shorter than four characters or starting with a digit are skipped.
    assert prompt_identifiers("Calling Foo_Bar(x) in 2024 hits 3dfx and ab.") == {
        "calling",
        "foo_bar",
        "hits",
    }


def test_iter_text_files_follows_eligibility(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.c").write_text("int a;")
    (tmp_path / "blob.bin").write_bytes(b"\0binary")
    (tmp_path / "huge.txt").write_bytes(b"x" * ((1 << 20) + 1))
    (tmp_path / "link.c").symlink_to("src/a.c")
    assert list(iter_text_files(tmp_path)) == [("src/a.c", b"int a;")]


def _oracle() -> LexicalOracle:
    files = [
        ("net/tcp.c", b"tcp_retransmit and TCP_Window"),
        ("net/udp.c", b"udp only"),
        ("net/both.c", b"tcp_retransmit tcp_window"),
        ("docs/tcp.txt", b"tcp_retransmit"),
    ]
    # The vocabulary covers every prompt the tests check.
    vocabulary = prompt_identifiers(
        "tcp_retransmit stalls when tcp_window shrinks slow"
    )
    return LexicalOracle(files, vocabulary)


def test_ranks_files_by_distinct_identifiers_with_pessimistic_ties():
    oracle = _oracle()
    prompt = "tcp_retransmit stalls when tcp_window shrinks"
    assert oracle.is_solvable(prompt, ["net/tcp.c"], k=2)
    # net/tcp.c and net/both.c tie at 2; with k = 1 the tie counts against it.
    assert not oracle.is_solvable(prompt, ["net/tcp.c"], k=1)
    # A gold file with no identifier, or one missing from the snapshot, fails.
    assert not oracle.is_solvable(prompt, ["net/udp.c"], k=5)
    assert not oracle.is_solvable(prompt, ["net/gone.c"], k=5)
    assert not oracle.is_solvable("it is slow", ["net/tcp.c"], k=5)
    with pytest.raises(KeyError):
        oracle.is_solvable("unknownword", ["net/tcp.c"])


def _task(task_id: str, solvable: bool | None) -> Task:
    return Task(
        id=task_id,
        bucket=Bucket.L,
        repo="linux",
        snapshot="0" * 40,
        prompt="tcp_retransmit stalls",
        gold_paths=("net/tcp.c",),
        gold_functions=(),
        provenance=Provenance(
            source=Source.COMMIT,
            source_id=task_id,
            created_at=datetime(2026, 9, 1, tzinfo=UTC),
            prompt_origin=PromptOrigin.REWRITE,
            rewriter="test",
        ),
        lexically_solvable=solvable,
        split=Split.DEV,
    )


def test_assess_sets_the_flag():
    [task] = assess([_task("linux/a", None)], _oracle())
    assert task.lexically_solvable is True


def test_cap_keeps_at_most_the_allowed_share_of_solvable_tasks():
    tasks = [_task(f"linux/u{n}", False) for n in range(7)] + [
        _task(f"linux/s{n}", True) for n in range(5)
    ]
    kept, dropped = apply_cap(tasks, max_fraction=Fraction(3, 10))
    # floor(0.3 * 7 / 0.7) = 3 solvable tasks may stay.
    assert sum(task.lexically_solvable for task in kept) == 3
    assert len(dropped) == 2
    assert [task.id for task in kept if not task.lexically_solvable] == [
        f"linux/u{n}" for n in range(7)
    ]
    assert apply_cap(tasks, max_fraction=Fraction(3, 10)) == (kept, dropped)
    with pytest.raises(ValueError, match="not assessed"):
        apply_cap([dataclasses.replace(tasks[0], lexically_solvable=None)])

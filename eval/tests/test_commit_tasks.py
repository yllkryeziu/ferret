# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tasks.commit_tasks: mining fix commits into tasks."""

import dataclasses
from datetime import UTC, datetime

import pytest

from tasks.commit_tasks import (
    CommitTaskConfig,
    RejectReason,
    build_commit_tasks,
    is_fix_commit,
)
from tasks.schema import Bucket, PromptOrigin, Source, Split
from tasks.split import assign_split
from tasks.testing.toy_repo import ToyRepo

_PARSE_C = """\
static int
parse_header(const char *p)
{
	return p[1];
}

int parse_body(const char *p)
{
	return p[2];
}
"""
_READER_C = "int read_block(int fd)\n{\n\treturn fd;\n}\n"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("net: fix a leak in the tx path", True),
        ("net: Fixed the tx path", True),
        ('Add a helper\n\nFixes: 0123456789ab ("net: add tx")', True),
        ('Revert "net: fix a leak"', False),
        ("Reland net: fix a leak", False),
        ("Roll src/third_party/foo and fix nothing", False),
        ("tests: add a fixture for tx", False),
    ],
)
def test_is_fix_commit(message, expected):
    assert is_fix_commit(message) is expected


def _rewriter(prompt: str) -> str:
    if "crash in helper" in prompt:
        return "  "
    if "leak in reader" in prompt:
        return "The read_block path leaks memory on every call."
    return "Parsing a header returns the wrong byte."


@pytest.fixture
def history(tmp_path):
    """A snapshot S on 2026-08-10 and one commit per case after it."""
    repo = ToyRepo(tmp_path / "linux")
    commits = {}

    def commit(name, message, files, date, author_date=None):
        commits[name] = repo.commit(
            message,
            files,
            date=f"2026-{date}T12:00:00Z",
            author_date=None
            if author_date is None
            else f"2026-{author_date}T12:00:00Z",
        )

    four = {f"drivers/net/n{n}.c": f"int n{n};\n" for n in range(4)}
    commit(
        "culprit_before",
        "parse: add header parsing",
        {
            "src/parse.c": _PARSE_C,
            "src/util.c": "int util;\n",
            "src/reader.c": _READER_C,
            "src/helper.c": "int helper;\n",
            "drivers/net/a.c": "int a;\n",
            "drivers/net/c.c": "int c;\n",
            "include/x.h": "int x;\n",
            "docs/guide.rst": "Guide\n",
            **four,
        },
        "08-01",
    )
    commit("snapshot", "Linux 7.3-rc1", {"Makefile": "VERSION = 7\n"}, "08-10")
    commit("feature", "util: add counters", {"src/util.c": "int util, n;\n"}, "08-11")
    commit(
        "good",
        f"parse: fix off-by-one in header parsing\n\nThe byte read was wrong.\n\n"
        f'Fixes: {commits["culprit_before"][:12]} ("parse: add header parsing")\n'
        "Signed-off-by: Dev <dev@example.com>",
        {"src/parse.c": _PARSE_C.replace("p[1]", "p[0]")},
        "08-12",
    )
    commit(
        "syzbot",
        "net: fix a use-after-free\n\nReported-by: syzbot+123@example.com",
        {"drivers/net/a.c": "int a = 1;\n"},
        "08-13",
    )
    commit(
        "added",
        "parse: fix by adding a table",
        {"src/table.c": "int table;\n", "src/parse.c": _PARSE_C + "\n"},
        "08-14",
    )
    commit("docs", "docs: fix a typo", {"docs/guide.rst": "Guide.\n"}, "08-15")
    commit(
        "four",
        "net: fix four drivers",
        {path: text + "\n" for path, text in four.items()},
        "08-16",
    )
    commit("changed", "util: fix a leak", {"src/util.c": "int util;\n"}, "08-17")
    commit("culprit_after", "x: add a helper", {"include/x.h": "int x, y;\n"}, "08-18")
    commit(
        "introduced",
        f"c: fix misuse of the x helper\n\nFixes: {commits['culprit_after'][:12]}",
        {"drivers/net/c.c": "int c = 2;\n"},
        "08-19",
    )
    commit(
        "empty", "helper: fix crash in helper", {"src/helper.c": "int h;\n"}, "08-20"
    )
    commit(
        "leak",
        "reader: fix leak in reader",
        {"src/reader.c": _READER_C.replace("fd;", "0;")},
        "08-21",
    )
    commit(
        "old",
        "reader: fix an old bug",
        {"include/x.h": "int x, y, z;\n"},
        "08-22",
        author_date="08-01",
    )
    commit("revert", 'Revert "parse: fix off-by-one"', {"Makefile": "V=7\n"}, "08-23")
    commit("late", "net: fix a late bug", {"drivers/net/a.c": "int a = 3;\n"}, "09-05")
    return repo, commits


def test_mines_fixes_after_the_snapshot(history):
    repo, commits = history
    config = CommitTaskConfig(
        repo_name="linux",
        bucket=Bucket.L,
        snapshot=commits["snapshot"],
        model_cutoff=datetime(2026, 8, 5, tzinfo=UTC),
        rewriter_id="test-rewriter-v1",
    )
    build = build_commit_tasks(repo.git, config, _rewriter)
    assert build.snapshot == commits["snapshot"]

    [task] = build.tasks
    good = commits["good"]
    assert task.id == f"linux/{good[:12]}"
    assert task.bucket is Bucket.L
    assert task.repo == "linux"
    assert task.snapshot == commits["snapshot"]
    assert task.prompt == "Parsing a header returns the wrong byte."
    assert task.gold_paths == ("src/parse.c",)
    assert task.gold_functions == ("src/parse.c::parse_header",)
    assert task.provenance.source is Source.COMMIT
    assert task.provenance.source_id == good
    assert task.provenance.created_at == datetime(2026, 8, 12, 12, tzinfo=UTC)
    assert task.provenance.prompt_origin is PromptOrigin.REWRITE
    assert task.provenance.rewriter == "test-rewriter-v1"
    assert task.lexically_solvable is None

    assert [(r.commit, r.reason, r.detail) for r in build.rejections] == [
        (commits["syzbot"], RejectReason.SYZBOT, ""),
        (commits["added"], RejectReason.NOT_IN_PLACE, "src/table.c"),
        (commits["docs"], RejectReason.NO_SOURCE_CHANGE, ""),
        (commits["four"], RejectReason.TOO_MANY_FILES, "4 gold candidates"),
        (commits["changed"], RejectReason.CHANGED_SINCE_SNAPSHOT, "src/util.c"),
        (
            commits["introduced"],
            RejectReason.INTRODUCED_AFTER_SNAPSHOT,
            commits["culprit_after"][:12],
        ),
        (commits["empty"], RejectReason.EMPTY_REWRITE, ""),
        (commits["leak"], RejectReason.LEAK, "read_block"),
    ]


def test_builds_only_the_requested_split(history):
    repo, commits = history
    config = CommitTaskConfig(
        repo_name="linux",
        bucket=Bucket.L,
        snapshot=commits["snapshot"],
        model_cutoff=datetime(2026, 8, 5, tzinfo=UTC),
        rewriter_id="test",
    )
    everything = build_commit_tasks(repo.git, config, _rewriter)
    prompts = []

    def counting_rewriter(prompt: str) -> str:
        prompts.append(prompt)
        return _rewriter(prompt)

    for split in Split:
        build = build_commit_tasks(
            repo.git, dataclasses.replace(config, split=split), counting_rewriter
        )
        assert build.tasks == tuple(t for t in everything.tasks if t.split is split)
        assert {r.commit for r in build.rejections} == {
            r.commit
            for r in everything.rejections
            if assign_split(f"linux/{r.commit[:12]}") is split
        }
    # Each candidate was rewritten once, in the split it belongs to.
    assert len(prompts) == 3


def test_rejects_an_unknown_snapshot(history):
    repo, _ = history
    config = CommitTaskConfig(
        repo_name="linux",
        bucket=Bucket.L,
        snapshot="v99.0",
        model_cutoff=datetime(2026, 8, 5, tzinfo=UTC),
        rewriter_id="test",
    )
    with pytest.raises(ValueError, match="does not name a commit"):
        build_commit_tasks(repo.git, config, _rewriter)


def test_rewrites_only_previous_task_ids_before_calling_model(history):
    repo, commits = history
    config = CommitTaskConfig(
        repo_name="linux",
        bucket=Bucket.L,
        snapshot=commits["snapshot"],
        model_cutoff=datetime(2026, 8, 5, tzinfo=UTC),
        rewriter_id="test",
    )
    prompts = []

    def rewriter(prompt):
        prompts.append(prompt)
        return _rewriter(prompt)

    selected = frozenset({f"linux/{commits['good'][:12]}"})
    result = build_commit_tasks(repo.git, config, rewriter, task_ids=selected)
    assert {task.id for task in result.tasks} == selected
    assert not result.rejections
    assert len(prompts) == 1


def test_chromium_dry_run_filters_automated_reverts_and_trailers(tmp_path):
    repo = ToyRepo(tmp_path / "chromium")
    base = repo.commit(
        "Base",
        {"src/widget.cc": "int widget() { return 1; }\n"},
        date="2026-09-23T00:00:00Z",
    )
    repo.commit(
        "Roll library from 1 to 2: fix crash",
        {"src/roll.txt": "roll\n"},
        date="2026-09-24T00:00:00Z",
    )
    repo.commit(
        'Revert "fix widget crash"',
        {"src/revert.txt": "revert\n"},
        date="2026-09-25T00:00:00Z",
    )
    repo.commit(
        "Fix crash when opening a widget\n\nBug: 12345\nChange-Id: I123",
        {"src/widget.cc": "int widget() { return 4; }\n"},
        date="2026-09-26T00:00:00Z",
    )
    build = build_commit_tasks(
        repo.git,
        CommitTaskConfig(
            repo_name="chromium",
            bucket=Bucket.XL,
            snapshot=base,
            model_cutoff=datetime(2026, 8, 30, tzinfo=UTC),
            rewriter_id="test",
        ),
        lambda _: pytest.fail("dry run called rewriter"),
        dry_run=True,
    )
    assert [item.reason for item in build.rejections] == [
        RejectReason.AUTOMATED,
        RejectReason.REVERT_RELAND,
    ]
    assert len(build.tasks) == 1
    assert build.tasks[0].message == "Fix crash when opening a widget"

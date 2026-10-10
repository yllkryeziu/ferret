# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tasks.prepare: sampling tasks and materializing their snapshots."""

import dataclasses
import json
import stat
from datetime import UTC, datetime

import pytest

from tasks import prepare
from tasks.commit_tasks import CommitTaskConfig, RejectReason
from tasks.git import Git, tree_blobs
from tasks.prepare import (
    assess_and_cap,
    index_path,
    materialize,
    mine_commit_tasks,
    sample_tasks,
    snapshot_dir,
)
from tasks.schema import (
    Bucket,
    PromptOrigin,
    Provenance,
    Source,
    Split,
    Task,
    read_tasks,
    write_tasks,
)
from tasks.split import assign_split
from tasks.testing.toy_repo import ToyRepo

# Records its arguments and writes the index file `ferret index` would.
_FAKE_FERRET = """\
#!/bin/sh
echo "$@" >> "$(dirname "$0")/calls.log"
for argument in "$@"; do
  case "$argument" in --index=*) echo index > "${argument#--index=}" ;; esac
done
"""


def _task(task_id: str, snapshot: str, gold_paths=("src/app.py",)) -> Task:
    return Task(
        id=task_id,
        bucket=Bucket.S,
        repo="toy/upstream",
        snapshot=snapshot,
        prompt="The app crashes.",
        gold_paths=tuple(gold_paths),
        gold_functions=(),
        provenance=Provenance(
            source=Source.LOCBENCH,
            source_id=task_id,
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            prompt_origin=PromptOrigin.ISSUE,
        ),
        lexically_solvable=None,
        split=assign_split(task_id),
    )


def test_sample_is_a_seeded_prefix_of_one_split():
    tasks = [_task(f"locbench/t{n}", "0" * 40) for n in range(200)]
    dev = sample_tasks(tasks, Split.DEV, 10)
    assert len(dev) == 10
    assert all(task.split is Split.DEV for task in dev)
    assert sample_tasks(tasks, Split.DEV, 10) == dev
    assert sample_tasks(tasks, Split.DEV, 20)[:10] == dev
    assert sample_tasks(list(reversed(tasks)), Split.DEV, 10) == dev


@pytest.fixture
def fake_ferret(tmp_path):
    path = tmp_path / "bin" / "ferret"
    path.parent.mkdir()
    path.write_text(_FAKE_FERRET)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_materializes_each_snapshot_once(tmp_path, fake_ferret):
    repo = ToyRepo(tmp_path / "remotes" / "toy" / "upstream")
    first = repo.commit("One", {"src/app.py": "one\n"}, date="2025-01-01T00:00:00Z")
    second = repo.commit("Two", {"src/app.py": "two\n"}, date="2025-01-02T00:00:00Z")
    tasks = [_task("a", first), _task("b", first), _task("c", second)]
    data = tmp_path / "data"
    url_template = f"file://{tmp_path}/remotes/{{repo}}"

    materialize(tasks, data, fake_ferret, url_template=url_template)
    for commit in (first, second):
        snapshot = Git(snapshot_dir(data, commit))
        assert tree_blobs(snapshot, "HEAD") == tree_blobs(repo.git, commit)
        assert index_path(data, commit).read_text() == "index\n"
    calls = (fake_ferret.parent / "calls.log").read_text().splitlines()
    assert calls == [
        f"index --root={snapshot_dir(data, first)} --index={index_path(data, first)}",
        f"index --root={snapshot_dir(data, second)} --index={index_path(data, second)}",
    ]
    assert not list((data / "snapshots").glob(".*"))

    # A rerun finds everything done, even with the remote gone.
    materialize(tasks, data, fake_ferret, url_template="file:///nonexistent/{repo}")
    assert len((fake_ferret.parent / "calls.log").read_text().splitlines()) == 2


def test_rejects_gold_paths_missing_from_the_snapshot(tmp_path, fake_ferret):
    repo = ToyRepo(tmp_path / "remotes" / "toy" / "upstream")
    commit = repo.commit("One", {"src/app.py": "one\n"}, date="2025-01-01T00:00:00Z")
    task = _task("a", commit, gold_paths=("src/app.py", "src/gone.py"))
    with pytest.raises(ValueError, match=r"gold paths \['src/gone.py'\]"):
        materialize(
            [task],
            tmp_path / "data",
            fake_ferret,
            url_template=f"file://{tmp_path}/remotes/{{repo}}",
        )
    assert not snapshot_dir(tmp_path / "data", commit).exists()


def _fix_history(tmp_path):
    """A snapshot and two fixes after it: one valid, one touching docs only."""
    repo = ToyRepo(tmp_path / "linux")
    snapshot = repo.commit(
        "Linux 7.3-rc1",
        {"net/tcp.c": "int tcp_window;\n", "docs/net.rst": "Net\n"},
        date="2026-08-30T12:00:00Z",
    )
    fixes = [
        repo.commit(
            "net: fix a stall",
            {"net/tcp.c": "int tcp_window = 1;\n"},
            date="2026-09-01T12:00:00Z",
        ),
        repo.commit(
            "docs: fix a typo",
            {"docs/net.rst": "Net.\n"},
            date="2026-09-02T12:00:00Z",
        ),
    ]
    return repo, snapshot, fixes


def test_mine_writes_tasks_and_rejections(tmp_path):
    repo, snapshot, [fix, docs] = _fix_history(tmp_path)
    config = CommitTaskConfig(
        repo_name="linux",
        bucket=Bucket.L,
        snapshot=snapshot,
        model_cutoff=datetime(2026, 8, 1, tzinfo=UTC),
        rewriter_id="test",
    )
    out, rejections = tmp_path / "tasks.jsonl", tmp_path / "rejections.jsonl"

    build = mine_commit_tasks(
        repo.git, config, lambda _: "Downloads stall.", out, rejections
    )
    assert read_tasks(out) == list(build.tasks)
    [task] = build.tasks
    assert task.provenance.source_id == fix
    assert [json.loads(line) for line in rejections.read_text().splitlines()] == [
        {"commit": docs, "reason": RejectReason.NO_SOURCE_CHANGE, "detail": ""}
    ]


def test_commits_command_uses_the_azure_rewriter(tmp_path, monkeypatch):
    repo, snapshot, [fix, _] = _fix_history(tmp_path)

    class FakeRewriter:
        rewriter_id = "fake-rewriter"
        input_tokens = output_tokens = 0

        def __call__(self, prompt):
            return "Downloads stall."

    monkeypatch.setattr(
        prepare.AzureRewriter, "from_environment", lambda model: FakeRewriter()
    )
    out = tmp_path / "tasks.jsonl"
    status = prepare.main(
        [
            "commits",
            f"--repo={repo.git.repo}",
            "--name=linux",
            "--bucket=L",
            f"--snapshot={snapshot}",
            "--model-cutoff=2026-08-01",
            f"--out={out}",
            f"--rejections={tmp_path / 'rejections.jsonl'}",
        ]
    )
    assert status == 0
    [task] = read_tasks(out)
    assert task.provenance.source_id == fix
    assert task.provenance.rewriter == "fake-rewriter"
    # An offset-free cutoff is UTC, and a fix authored exactly then is skipped.
    assert (
        prepare.main(
            [
                "commits",
                f"--repo={repo.git.repo}",
                "--name=linux",
                "--bucket=L",
                f"--snapshot={snapshot}",
                "--model-cutoff=2026-09-01T12:00:00",
                f"--out={out}",
                f"--rejections={tmp_path / 'rejections.jsonl'}",
            ]
        )
        == 0
    )
    assert read_tasks(out) == []


def test_assess_and_cap_reads_the_materialized_snapshot(tmp_path, fake_ferret):
    repo = ToyRepo(tmp_path / "remotes" / "linux")
    snapshot = repo.commit(
        "Snapshot",
        {
            "net/tcp.c": "tcp_retransmit tcp_window\n",
            "net/udp.c": "udp_send\n",
        },
        date="2026-08-30T12:00:00Z",
    )
    solvable = _task("linux/solvable", snapshot, gold_paths=("net/tcp.c",))
    solvable = dataclasses.replace(solvable, prompt="tcp_retransmit stalls.")
    unsolvable = dataclasses.replace(
        _task("linux/unsolvable", snapshot, gold_paths=("net/udp.c",)),
        prompt="Sending datagrams fails.",
    )
    data = tmp_path / "data"
    with pytest.raises(FileNotFoundError, match="not materialized"):
        assess_and_cap([solvable], data)
    materialize(
        [solvable], data, fake_ferret, url_template=f"file://{tmp_path}/remotes/linux"
    )

    # One unsolvable task allows floor(0.3 * 1 / 0.7) = 0 solvable ones.
    kept, dropped = assess_and_cap([solvable, unsolvable], data)
    assert [(task.id, task.lexically_solvable) for task in kept] == [
        ("linux/unsolvable", False)
    ]
    assert [(task.id, task.lexically_solvable) for task in dropped] == [
        ("linux/solvable", True)
    ]

    tasks, out = tmp_path / "tasks.jsonl", tmp_path / "capped.jsonl"
    write_tasks(tasks, [solvable, unsolvable])
    argv = ["lexical", f"--tasks={tasks}", f"--data={data}", f"--out={out}"]
    assert prepare.main(argv) == 0
    assert read_tasks(out) == kept


def test_sample_command_draws_from_a_task_file(tmp_path):
    tasks = [_task(f"linux/t{n}", "0" * 40) for n in range(50)]
    path, out = tmp_path / "tasks.jsonl", tmp_path / "sample.jsonl"
    write_tasks(path, tasks)
    argv = ["sample", f"--tasks={path}", "--split=dev", "--count=3", f"--out={out}"]
    assert prepare.main(argv) == 0
    assert read_tasks(out) == sample_tasks(tasks, Split.DEV, 3)
    assert prepare.main([*argv[:3], "--count=50", f"--out={out}"]) == 1


def test_only_task_ids_dry_run_never_constructs_model(tmp_path, monkeypatch, capsys):
    ids = tmp_path / "ids.jsonl"
    ids.write_text('{"id":"linux/one"}\n{"id":"linux/two"}\n')

    def no_model(*args, **kwargs):
        raise AssertionError("dry run constructed a model")

    from tasks.testing.toy_repo import ToyRepo

    repo = ToyRepo(tmp_path / "repo")
    base = repo.commit("base", {"a.py": "old\n"}, date="2026-08-10T00:00:00Z")
    monkeypatch.setattr(prepare.AzureRewriter, "from_environment", no_model)
    assert (
        prepare.main(
            [
                "commits",
                "--repo",
                str(repo.git.repo),
                "--name",
                "linux",
                "--bucket",
                "L",
                "--snapshot",
                base,
                "--model-cutoff",
                "2026-08-30",
                "--split",
                "dev",
                "--only-task-ids",
                str(ids),
                "--out",
                str(tmp_path / "out.jsonl"),
                "--rejections",
                str(tmp_path / "reject.jsonl"),
                "--dry-run",
            ]
        )
        == 0
    )
    # The new dry run mines instead of just validating the CLI. The empty
    # selection still writes both output files without constructing a model.
    assert "0 candidates would reach the rewriter" in capsys.readouterr().out
    assert (tmp_path / "out.jsonl").read_text() == ""
    assert (tmp_path / "reject.jsonl").read_text() == ""
    ids.write_text('{"id":"linux/one"}\n{"id":"linux/one"}\n')
    with pytest.raises(ValueError, match="duplicate task id"):
        prepare.read_task_ids(ids)


def test_probe_listings_need_no_snapshot_or_index(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    from tasks.prepare import listing_dir, materialize_listings
    from tasks.schema import Bucket, PromptOrigin, Provenance, Source, Split, Task
    from tasks.testing.toy_repo import ToyRepo

    repo = ToyRepo(tmp_path / "upstream")
    base = repo.commit(
        "Base",
        {"src/app.py": "print(1)\n", "README.md": "Hi\n"},
        date="2024-01-01T00:00:00Z",
    )
    monkeypatch.setattr(prepare, "GITHUB_URL", f"file://{repo.git.repo}")
    task = Task(
        id="swebench_verified/one",
        bucket=Bucket.P,
        repo="toy/upstream",
        snapshot=base,
        prompt="The app fails.",
        gold_paths=("src/app.py",),
        gold_functions=(),
        lexically_solvable=None,
        split=Split.TEST,
        provenance=Provenance(
            source=Source.SWEBENCH_VERIFIED,
            source_id="one",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            prompt_origin=PromptOrigin.ISSUE,
        ),
    )
    data = tmp_path / "data"
    materialize_listings([task], data)
    assert sorted(path.name for path in listing_dir(data, base).iterdir()) == [
        "README.md",
        "src",
    ]
    assert (listing_dir(data, base) / "src").is_dir()
    assert not (listing_dir(data, base) / "src/app.py").exists()
    assert not (data / "snapshots").exists()
    assert not (data / "indexes").exists()
    materialize_listings([task], data)


def test_probe_command_samples_fifty_by_seeded_hash(tmp_path):
    from tasks.benchmarks import SAMPLE_SALT
    from tasks.split import hash_order

    rows = tmp_path / "verified.jsonl"
    with rows.open("w") as out:
        for number in range(52):
            out.write(
                json.dumps(
                    {
                        "instance_id": f"repo-issue-{number}",
                        "repo": "toy/repo",
                        "base_commit": "0" * 40,
                        "problem_statement": "An issue.",
                        "patch": (
                            "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-a\n+b\n"
                        ),
                        "created_at": "2024-01-01T00:00:00Z",
                    }
                )
                + "\n"
            )
    out = tmp_path / "probe.jsonl"
    assert prepare.main(["probe", "--swebench", str(rows), "--out", str(out)]) == 0
    tasks = read_tasks(out)
    assert len(tasks) == 50
    assert tasks == hash_order(tasks, key=lambda task: task.id, salt=SAMPLE_SALT)

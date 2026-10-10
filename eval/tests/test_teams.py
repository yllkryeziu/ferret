# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from teams.commands import LIST, OTHER, READ, SEARCH, classify, programs, script_of
from teams.run import TEAM_SIZE, render_prompt, schedule, tool_items


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("/bin/zsh -lc 'rg -n \"foo\" src'", SEARCH),
        ("/bin/zsh -lc 'rg --files | rg xfs'", SEARCH),
        ("/bin/zsh -lc \"git grep -n 'x' -- '*.c'\"", SEARCH),
        ("/bin/zsh -lc 'cd fs && grep -rn recover . 2>&1 | head -20'", SEARCH),
        ("/bin/zsh -lc 'find . -name \"*.c\" | xargs grep -l foo'", SEARCH),
        ("/bin/zsh -lc 'find . -name \"*.c\" -exec grep -l foo {} +'", SEARCH),
        ("/bin/zsh -lc 'timeout 30 rg foo'", SEARCH),
        ("/bin/zsh -lc 'LC_ALL=C grep -r foo .'", SEARCH),
        ("/bin/zsh -lc 'find fs/xfs -name \"*.c\"'", LIST),
        ("/bin/zsh -lc 'ls fs/xfs'", LIST),
        ("/bin/zsh -lc 'git ls-files | wc -l'", LIST),
        ("/bin/zsh -lc \"sed -n '1,80p' fs/xfs/xfs_rmap_item.c\"", READ),
        ("/bin/zsh -lc 'nl -ba a.c | sed -n 1,20p'", READ),
        ("/bin/zsh -lc 'git status --short'", OTHER),
        ("/bin/zsh -lc \"python3 -c 'print(1)'\"", OTHER),
    ],
)
def test_classify(command, expected):
    assert classify(command) == expected


def test_script_and_programs():
    assert script_of("/bin/zsh -lc 'rg x'") == "rg x"
    assert script_of("rg x") == "rg x"
    assert programs("cd a && git grep -n x | head; ls") == [
        "cd",
        "git-grep",
        "head",
        "ls",
    ]
    # An unbalanced quote falls back to splitting on whitespace.
    assert classify('/bin/zsh -lc "rg \'unterminated"') == SEARCH


def _tasks(repo, n):
    return [{"id": f"{repo}/{i:02d}", "repo": repo, "prompt": "p"} for i in range(n)]


def test_schedule_team_batches_alternate_repos_and_cover_each_task_per_arm():
    worktrees = {r: [Path(f"/{r}-{i}") for i in range(TEAM_SIZE)] for r in ("a", "b")}
    tasks = _tasks("a", 9) + _tasks("b", 4)
    batches = schedule(tasks, "team", worktrees, seed=0)
    assert [b[0][0].task["repo"] for b in batches] == ["a", "b", "a", "a"]
    for arm in ("F", "R"):
        ids = [e.task["id"] for b in batches for g in b for e in g if e.arm == arm]
        assert sorted(ids) == sorted(t["id"] for t in tasks)
    for batch in batches:
        assert {g[0].arm for g in batch} == {"F", "R"}
        for group in batch:
            assert [e.slot for e in group] == worktrees[group[0].task["repo"]][
                : len(group)
            ]
    assert schedule(tasks, "team", worktrees, seed=0) == batches


def test_schedule_solo_uses_the_first_slot():
    worktrees = {"a": [Path("/a-0"), Path("/a-1")]}
    batches = schedule(_tasks("a", 3), "solo", worktrees, seed=1)
    assert len(batches) == 3
    assert all(len(g) == 1 and g[0].slot == Path("/a-0") for b in batches for g in b)


def test_prompt_differs_only_by_the_ferret_hint():
    task = {"prompt": " report \n"}
    r, f = render_prompt(task, "R"), render_prompt(task, "F")
    assert f.startswith(r) and "ferret" in f[len(r) :] and "ferret" not in r
    assert "<problem_report>\nreport\n</problem_report>" in r


def test_tool_items_pairs_start_and_completion():
    def ev(t, kind, item):
        return {"t": t, "event": {"type": kind, "item": item}}

    command = {"id": "1", "type": "command_execution", "command": "/bin/zsh -lc 'rg x'"}
    call = {
        "id": "2",
        "type": "mcp_tool_call",
        "server": "ferret",
        "tool": "search",
        "arguments": {"pattern": "x"},
    }
    events = [
        ev(1.0, "item.started", command),
        ev(3.5, "item.completed", command | {"exit_code": 0}),
        ev(4.0, "item.started", call),
        ev(4.1, "item.completed", call),
        ev(5.0, "item.completed", {"id": "3", "type": "agent_message", "text": "hi"}),
    ]
    items = tool_items(events)
    assert [(i["type"], i["class"], i["t1"] - i["t0"]) for i in items] == [
        ("command_execution", SEARCH, 2.5),
        ("mcp_tool_call", SEARCH, pytest.approx(0.1)),
    ]


def _record(task_id, arm, setting, wall, search, tokens, files):
    return {
        "key": f"{setting}__{task_id}__{arm}",
        "task_id": task_id,
        "bucket": "L",
        "arm": arm,
        "setting": setting,
        "wall_s": wall,
        "timed_out": False,
        "usage": {"input_tokens": tokens, "cached_input_tokens": 0, "output_tokens": 0},
        "answer": {"files": files, "functions": []},
        "changed": ["a.c"],
        "items": [
            {
                "type": "mcp_tool_call" if arm == "F" else "command_execution",
                "class": SEARCH,
                "t0": 0.0,
                "t1": search,
            }
        ],
    }


def test_analyze_scores_and_pairs(tmp_path):
    import json

    from teams.analyze import load, paired, report

    tasks = {
        f"t{i}": {"id": f"t{i}", "gold_paths": ["a.c"], "gold_functions": []}
        for i in range(3)
    }
    records = []
    for i in range(3):
        records.append(_record(f"t{i}", "F", "team", 100, 0.1, 1000, ["a.c"]))
        records.append(_record(f"t{i}", "R", "team", 200, 9.0, 2000, ["b.c"]))
    # A rerun of the same key replaces the first record.
    records.append(_record("t0", "R", "team", 200, 9.0, 2000, ["b.c"]))
    (tmp_path / "episodes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    frame = load([tmp_path], tasks)
    assert len(frame) == 6
    assert (
        frame[frame.arm == "F"].success5.all()
        and not frame[frame.arm == "R"].success5.any()
    )
    assert frame[frame.arm == "F"].edited_gold.all()
    assert len(paired(frame, "team")) == 3
    text = report(frame)
    assert "| team | wall time | 3 | 0.50x | [0.50, 0.50] |" in text
    assert "| team | success@5 | 3 | +100% |" in text


def test_reset_removes_a_stale_index_lock(tmp_path):
    import subprocess

    from teams.run import index_lock, reset

    def git(*args):
        subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        )

    git("init", "-q")
    (tmp_path / "a.c").write_text("x\n")
    git("add", "a.c")
    git("-c", "user.name=n", "-c", "user.email=e@x", "commit", "-qm", "a")
    (tmp_path / "a.c").write_text("edited\n")
    (tmp_path / "new.c").write_text("new\n")
    index_lock(tmp_path).touch()
    notes = reset(tmp_path, attempts=2)
    assert "removed a stale index.lock" in notes
    assert (tmp_path / "a.c").read_text() == "x\n"
    assert not (tmp_path / "new.c").exists()

# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures, and a hermetic environment for mini-swe-agent and litellm.

The environment variables are set before any test module imports those
packages: mini-swe-agent otherwise prints a banner and loads a .env file
from the user's configuration directory, and litellm otherwise fetches its
model-cost map over the network.
"""

import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

os.environ["MSWEA_SILENT_STARTUP"] = "1"
os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = tempfile.mkdtemp(prefix="mswea-config-")
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"

# tasks.prepare imports litellm, so these come after the environment.
from tasks import schema
from tasks.prepare import index_path, listing_dir, snapshot_dir


@pytest.fixture
def toy_repo(tmp_path: Path) -> Path:
    """A small snapshot with the hazards the sandbox must handle.

    Layout, relative to the returned root:
        README.md, src/app.py, src/util.py, lib/a.c, lib/b.c
        big.txt          450 short lines
        wide.txt         one 40,000-byte line
        blob.bin         binary (NUL in the first bytes)
        .git/config      git metadata with a secret marker
        vendor/.git/HEAD nested git metadata
        link_ok          -> src/app.py
        link_dir         -> src
        link_git         -> .git
        link_out         -> ../outside/secret.txt
        link_out_dir     -> ../outside
        link_dangling    -> missing.txt
        fifo             a named pipe
    The directory `tmp_path / "outside"` holds secret.txt, outside the root.
    """
    root = tmp_path / "repo"
    files = {
        "README.md": "# Toy\nA toy repository.\n",
        "src/app.py": "import util\n\n\ndef main():\n    util.helper()\n",
        "src/util.py": "def helper():\n    return 42\n",
        "lib/a.c": "#include <stdio.h>\n\nint parse_header(void) {\n  return 0;\n}\n",
        "lib/b.c": "void g(void) {\n  parse_header();\n}\n",
        "big.txt": "".join(f"line {n}\n" for n in range(1, 451)),
        "wide.txt": "x" * 40_000 + "\n",
        ".git/config": "[core]\n\tsecret = GIT-SECRET\n",
        "vendor/.git/HEAD": "ref: refs/heads/GIT-SECRET\n",
        "vendor/lib.c": "int vendored;\n",
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "blob.bin").write_bytes(b"\x7fELF\x00\x01\x02binary")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("OUTSIDE-SECRET\n")
    (root / "link_ok").symlink_to("src/app.py")
    (root / "link_dir").symlink_to("src")
    (root / "link_git").symlink_to(".git")
    (root / "link_out").symlink_to("../outside/secret.txt")
    (root / "link_out_dir").symlink_to("../outside")
    (root / "link_dangling").symlink_to("missing.txt")
    os.mkfifo(root / "fifo")
    return root


RUN_SNAPSHOT = "0123456789abcdef0123456789abcdef01234567"
_RUN_SCRIPT = {
    "default": [
        [{"tool": "search", "arguments": {"pattern": "helper"}}],
        [{"tool": "submit", "arguments": {"files": ["src/util.py"]}}],
    ],
    "A0": [[{"tool": "submit", "arguments": {"files": ["src/app.py"]}}]],
}


def _run_task(task_id: str, bucket: schema.Bucket) -> schema.Task:
    return schema.Task(
        id=task_id,
        bucket=bucket,
        repo="toy/repo",
        snapshot=RUN_SNAPSHOT,
        prompt="helper() returns the wrong value.",
        gold_paths=("src/util.py",),
        gold_functions=(),
        provenance=schema.Provenance(
            source=schema.Source.LOCBENCH,
            source_id=task_id,
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            prompt_origin=schema.PromptOrigin.ISSUE,
        ),
        lexically_solvable=None,
        split=schema.Split.DEV,
    )


@pytest.fixture
def run_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Inputs of a two-task run: data directory, ferret, task file, script.

    The ferret is a fake that prints one match; the script searches and then
    submits the gold file, except in arm A0, where it submits another file.
    Task "probe/toy-2" is in the contamination-probe bucket.
    """
    data = tmp_path / "data"
    snapshot = snapshot_dir(data, RUN_SNAPSHOT)
    (snapshot / "src").mkdir(parents=True)
    (snapshot / "src/app.py").write_text("import util\n")
    (snapshot / "src/util.py").write_text("def helper():\n    return 41\n")
    listing = listing_dir(data, RUN_SNAPSHOT)
    (listing / "src").mkdir(parents=True)
    index = index_path(data, RUN_SNAPSHOT)
    index.parent.mkdir(parents=True)
    index.write_text("index\n")
    ferret = tmp_path / "ferret"
    ferret.write_text(f"#!{sys.executable}\nprint('src/util.py:1:def helper():')\n")
    ferret.chmod(0o755)
    tasks = tmp_path / "tasks.jsonl"
    schema.write_tasks(
        tasks,
        [
            _run_task("locbench/toy-1", schema.Bucket.S),
            _run_task("probe/toy-2", schema.Bucket.P),
        ],
    )
    script = tmp_path / "script.json"
    script.write_text(json.dumps(_RUN_SCRIPT))
    return data, ferret, tasks, script

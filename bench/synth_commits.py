# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic commits for the freshness benchmarks.

The corpora are shallow clones, so there is no history to check out. These
commands make it in a disposable copy: seeded, with realistic small edits,
committed with a fixed identity and date. The same repository content, file
list, seed and flags give the same commit ids.

    python -m bench.synth_commits chain REPO --files LIST --seed 1 \\
        --steps 9 --diff-fraction 0.005 --out chain.json
    python -m bench.synth_commits churn REPO --files LIST --seed 1 \\
        --files-changed 5000 --out churn.json

`chain` makes commits C2..C(steps+1) on top of the current HEAD (C1). Step n
edits its own slice of the eligible files, so worktree n (at Cn) differs from
worktree 1 by (n - 1) slices, and from worktree n - 1 by exactly one. The
commits are branches m3/k1 (the base) to m3/k<steps+1>.

`churn` makes branch m3/churn on top of HEAD, one commit "a week of changes":
80% of the files edited, 10% deleted, 10% added (copies of listed files, edited,
under synthetic/), and a marker file whose line the benchmark probes for.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

# A file qualifies for editing if it is valid UTF-8 text below this size.
MAX_EDIT_BYTES = 256 * 1024
# The line a churn commit adds, for the benchmark to probe.
MARKER_PATH = "synthetic/m3_churn_marker.txt"
MARKER_PREFIX = "ferret_m3_churn_marker_"
FIXED_DATE = "2026-01-01T00:00:00+0000"

COMMENT_PREFIX = {
    ".c": "//", ".cc": "//", ".h": "//", ".cpp": "//", ".hpp": "//",
    ".go": "//", ".rs": "//", ".java": "//", ".js": "//", ".ts": "//",
    ".py": "#", ".sh": "#", ".bzl": "#", ".yaml": "#", ".yml": "#",
    ".toml": "#", ".txt": "#", ".mk": "#", ".pl": "#", ".rb": "#",
}  # fmt: skip

# Stands for the comment prefix, which edit_file knows from the extension.
PLACEHOLDER = "\x01PREFIX\x01"
DIGITS = re.compile(r"\d+")


def edit_text(content: str, rng: random.Random, suffix: str) -> str:
    """Return content with one to three small edits; never equal to content.

    The edits are what a source change looks like at the line level: a
    duplicated line with a changed number or name, a deleted line, a changed
    number, and a short comment line where the language is known. `suffix`
    (unique per file and step) makes a duplicated line differ.
    """
    lines = content.splitlines(keepends=True)
    if not lines:
        return f"{suffix}\n"
    if not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    ops = ["dup", "number", "delete", "comment"]
    for _ in range(rng.randint(1, 3)):
        op = rng.choice(ops)
        i = rng.randrange(len(lines))
        line = lines[i]
        if op == "dup" and line.strip():
            if DIGITS.search(line):
                changed = DIGITS.sub(lambda m: str(int(m[0]) + 1), line, count=1)
            else:
                changed = line.rstrip("\n") + f"_{suffix}\n"
            lines.insert(i + 1, changed)
        elif op == "number":
            for j in range(i, i + len(lines)):
                at = j % len(lines)
                if DIGITS.search(lines[at]):
                    lines[at] = DIGITS.sub(
                        lambda m: str(int(m[0]) * 3 + 7), lines[at], count=1
                    )
                    break
        elif op == "delete" and len(lines) > 3 and line.strip():
            del lines[i]
        elif op == "comment":
            lines.insert(i, f"{PLACEHOLDER} {suffix}\n")
    return "".join(lines)


def edit_file(path: str, content: str, rng: random.Random, suffix: str) -> str:
    """edit_text for a file at `path`; always differs from `content`."""
    prefix = COMMENT_PREFIX.get(os.path.splitext(path)[1].lower())
    edited = edit_text(content, rng, suffix)
    if prefix is None:
        # No known comment syntax: drop the placeholder lines.
        edited = "".join(
            ln for ln in edited.splitlines(keepends=True) if PLACEHOLDER not in ln
        )
    else:
        edited = edited.replace(PLACEHOLDER, prefix)
    if edited == content:
        edited = content + ("" if content.endswith("\n") or not content else "\n")
        edited += f"{suffix}\n" if prefix is None else f"{prefix} {suffix}\n"
    return edited


def read_text(repo: Path, path: str) -> str | None:
    """The file as strict UTF-8 text, or None if it does not qualify."""
    full = repo / path
    try:
        if full.is_symlink() or not full.is_file():
            return None
        if full.stat().st_size > MAX_EDIT_BYTES:
            return None
        data = full.read_bytes()
        if not data or b"\0" in data:
            return None
        return data.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def select_editable(repo: Path, files: Sequence[str], seed: int, count: int):
    """`count` qualifying files and their text, in a seeded order.

    Repeatable: the order depends on the sorted list and the seed only.
    Raises SystemExit if fewer than `count` files qualify.
    """
    order = sorted(files)
    random.Random(seed).shuffle(order)
    picked: list[tuple[str, str]] = []
    for path in order:
        if len(picked) == count:
            break
        text = read_text(repo, path)
        if text is not None:
            picked.append((path, text))
    if len(picked) < count:
        raise SystemExit(
            f"only {len(picked)} of the {len(files)} listed files can be edited; "
            f"{count} are needed"
        )
    return picked


def git(repo: Path, *args: str, check: bool = True) -> str:
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="ferret-synth",
        GIT_AUTHOR_EMAIL="synth@localhost",
        GIT_COMMITTER_NAME="ferret-synth",
        GIT_COMMITTER_EMAIL="synth@localhost",
        GIT_AUTHOR_DATE=FIXED_DATE,
        GIT_COMMITTER_DATE=FIXED_DATE,
    )
    result = subprocess.run(
        [
            "git", "-C", str(repo),
            "-c", "core.hooksPath=/dev/null",
            "-c", "commit.gpgSign=false",
            "-c", "gc.auto=0",
            "-c", "core.autocrlf=false",
            *args,
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )  # fmt: skip
    if check and result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "--quiet", "--no-verify", "--message", message)
    return git(repo, "rev-parse", "HEAD")


def read_list(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if ln]


def make_chain(
    repo: Path, files: Sequence[str], seed: int, steps: int, per_step: int
) -> dict:
    """Commits C2..C(steps+1) on the current HEAD; see the module comment."""
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "--quiet", "--detach", base)
    git(repo, "branch", "--force", "m3/k1", base)
    picked = select_editable(repo, files, seed, steps * per_step)
    commits = [{"k": 1, "branch": "m3/k1", "commit": base, "files": 0, "bytes": 0}]
    for step in range(steps):
        changed_bytes = 0
        for path, text in picked[step * per_step : (step + 1) * per_step]:
            rng = random.Random(f"{seed}:{step}:{path}")
            edited = edit_file(path, text, rng, f"m3s{step}x{rng.randrange(10**6)}")
            (repo / path).write_text(edited, encoding="utf-8")
            changed_bytes += len(edited.encode())
        commit = commit_all(repo, f"synthetic step {step + 1} (seed {seed})")
        k = step + 2
        git(repo, "branch", "--force", f"m3/k{k}", commit)
        commits.append(
            {
                "k": k,
                "branch": f"m3/k{k}",
                "commit": commit,
                "files": per_step,
                "bytes": changed_bytes,
            }
        )
    git(repo, "checkout", "--quiet", "--force", "--detach", base)
    return {
        "kind": "chain",
        "seed": seed,
        "base": base,
        "files_per_step": per_step,
        "commits": commits,
    }


def make_churn(repo: Path, files: Sequence[str], seed: int, changed: int) -> dict:
    """Branch m3/churn: one commit of `changed` files, 80/10/10 edit/delete/add."""
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "--quiet", "--detach", base)
    deleted = changed // 10
    added = changed // 10
    edited = changed - deleted - added
    picked = select_editable(repo, files, seed, edited + deleted + added)
    for i, (path, text) in enumerate(picked[:edited]):
        rng = random.Random(f"{seed}:churn:{path}")
        (repo / path).write_text(
            edit_file(path, text, rng, f"m3c{i}"), encoding="utf-8"
        )
    for path, _ in picked[edited : edited + deleted]:
        (repo / path).unlink()
    for i, (path, text) in enumerate(picked[edited + deleted :]):
        rng = random.Random(f"{seed}:churn-add:{path}")
        name = f"synthetic/w{seed}/n{i:05d}{os.path.splitext(path)[1]}"
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(edit_file(path, text, rng, f"m3a{i}"), encoding="utf-8")
    marker = f"{MARKER_PREFIX}{seed}"
    (repo / MARKER_PATH).parent.mkdir(parents=True, exist_ok=True)
    (repo / MARKER_PATH).write_text(f"{marker}\n", encoding="utf-8")
    commit = commit_all(repo, f"synthetic week of changes (seed {seed})")
    git(repo, "branch", "--force", "m3/churn", commit)
    git(repo, "checkout", "--quiet", "--force", "--detach", base)
    return {
        "kind": "churn",
        "seed": seed,
        "base": base,
        "branch": "m3/churn",
        "commit": commit,
        "edited": edited,
        "deleted": deleted,
        "added": added + 1,
        "marker": marker,
        "marker_path": MARKER_PATH,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("chain", "churn"):
        p = sub.add_parser(name)
        p.add_argument("repo", type=Path)
        p.add_argument("--files", type=Path, required=True)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--out", type=Path)
        if name == "chain":
            p.add_argument("--steps", type=int, default=9)
            p.add_argument("--diff-fraction", type=float, default=0.005)
            p.add_argument("--diff-files", type=int, default=0)
        else:
            p.add_argument("--files-changed", type=int, default=5000)
    args = parser.parse_args(argv)
    files = read_list(args.files)
    if args.command == "chain":
        per_step = args.diff_files or max(1, round(len(files) * args.diff_fraction))
        result = make_chain(args.repo, files, args.seed, args.steps, per_step)
    else:
        result = make_churn(args.repo, files, args.seed, args.files_changed)
    text = json.dumps(result, indent=1) + "\n"
    if args.out:
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Builds localization tasks from fix commits (buckets L and XL).

The experiment takes commit-derived tasks from a few history-free snapshots
per repository instead of one checkout per fix: a Chromium checkout is 5 GB,
and 150 of them do not fit on the benchmark machine. A task is therefore a
fix commit F that lands within `window` after snapshot S, and the agent
searches S, not F's parent. That is only sound when the bug exists in S
exactly as the fixer saw it, which the pipeline checks. For each non-merge
commit F in S..head:

1. Candidacy: F lands within `window` after S, was authored after the model's
   training cutoff, is a fix (a Fixes:/Fixed: trailer or "fix" in the
   subject; reverts, relands, and rolls are not), and its task falls in the
   requested split, if any. Other commits are skipped silently; every later
   check logs a `Rejection` instead.
2. Gold paths: the gold-candidate files (`tasks.paths`) F modifies in place,
   1 to MAX_GOLD_PATHS of them. A fix that adds, deletes, or renames a gold
   candidate has no well-defined location in S and is rejected.
3. Validity at S: every gold file has the same blob in S as in F's parent,
   and no commit named by a `Fixes:` trailer is missing from S's history (a
   culprit the fetched history cannot resolve is assumed to predate S).
4. Gold functions: the definitions enclosing F's changed lines, found by
   `tasks.definitions` in the parent's copy of each C-family gold file.
5. Prompt: the rewriter turns F's message into a symptom-level report, and
   `tasks.leakage` rejects any report that names a gold path component or a
   gold function. Syzbot reports are rejected before rewriting, since their
   stack traces name the gold function.

Lexical solvability is assessed later, against the exported snapshot
(`tasks.lexical`), because it needs file contents rather than history.
"""

from __future__ import annotations

import dataclasses
import enum
import re
from collections.abc import Iterator
from datetime import datetime, timedelta

from tasks.definitions import changed_definitions
from tasks.diff import parse_unified_diff
from tasks.git import Git, tree_blobs
from tasks.leakage import PathComponentIndex, find_leaks
from tasks.paths import MAX_GOLD_PATHS, is_c_family, is_gold_candidate
from tasks.rewrite import Rewriter, rewrite_prompt, strip_trailers
from tasks.schema import (
    FUNCTION_SEPARATOR,
    Bucket,
    PromptOrigin,
    Provenance,
    Source,
    Split,
    Task,
)
from tasks.split import assign_split

DEFAULT_WINDOW = timedelta(days=21)

_FIX_TRAILER = re.compile(r"^(Fixes|Fixed):\s*\S", re.MULTILINE | re.IGNORECASE)
_CULPRIT_TRAILER = re.compile(
    r"^Fixes:\s*([0-9a-f]{8,40})\b", re.MULTILINE | re.IGNORECASE
)
_FIX_SUBJECT = re.compile(r"\bfix(es|ed)?\b", re.IGNORECASE)
_NOT_A_FIX_SUBJECT = re.compile(r"(Revert|Reland|Roll)\b")
_SYZBOT = re.compile(r"^Reported-by:.*\bsyzbot\b", re.MULTILINE | re.IGNORECASE)
# One log record per commit: fields split by US, records ended by RS.
_LOG_FORMAT = "--format=%H%x1f%P%x1f%aI%x1f%cI%x1f%ae%x1f%B%x1e"


class RejectReason(enum.StrEnum):
    """Why a fix commit did not become a task."""

    AUTOMATED = "automated"
    REVERT_RELAND = "revert_reland"
    NO_SYMPTOM = "no_symptom"
    SYZBOT = "syzbot"
    NOT_IN_PLACE = "not_in_place"
    NO_SOURCE_CHANGE = "no_source_change"
    TOO_MANY_FILES = "too_many_files"
    CHANGED_SINCE_SNAPSHOT = "changed_since_snapshot"
    INTRODUCED_AFTER_SNAPSHOT = "introduced_after_snapshot"
    EMPTY_REWRITE = "empty_rewrite"
    LEAK = "leak"
    UNANSWERABLE = "unanswerable"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CommitTaskConfig:
    """What to mine from one repository around one snapshot.

    Attributes:
        repo_name: Task repo name and id prefix, such as "linux".
        bucket: L for Linux, XL for Chromium.
        snapshot: Revision of the snapshot commit S.
        model_cutoff: Fixes authored at or before this instant are excluded.
        rewriter_id: Identifies the rewriting model and prompt version.
        head: Upper end of the commit range searched for fixes.
        window: How long after S a fix may land.
        split: Only tasks of this split, or None for all. The split is known
            from the commit id alone, so other fixes are skipped before the
            costly rewrite.
    """

    repo_name: str
    bucket: Bucket
    snapshot: str
    model_cutoff: datetime
    rewriter_id: str
    head: str = "HEAD"
    window: timedelta = DEFAULT_WINDOW
    split: Split | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Rejection:
    """A fix commit that failed a check, and why; the protocol publishes these."""

    commit: str
    reason: RejectReason
    detail: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class CommitTaskBuild:
    """The outcome of mining one snapshot."""

    snapshot: str
    tasks: tuple[Task | RewriteCandidate, ...]
    rejections: tuple[Rejection, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _Commit:
    id: str
    parents: tuple[str, ...]
    author_date: datetime
    committer_date: datetime
    message: str
    author: str


@dataclasses.dataclass(frozen=True, slots=True)
class RewriteCandidate:
    """A structurally valid fix awaiting a symptom-level rewrite."""

    commit: str
    gold_paths: tuple[str, ...]
    message: str


def _chromium_exclusion(commit: _Commit) -> RejectReason | None:
    """Rejects automated Chromium changes and reverts before fix selection."""
    subject = commit.message.split("\n", 1)[0]
    if re.match(r"(?:Revert|Reland)\b", subject, re.IGNORECASE):
        return RejectReason.REVERT_RELAND
    if (
        re.match(
            r"(?:Roll|Re-roll|Update translations|Update.*expectations)\b",
            subject,
            re.IGNORECASE,
        )
        or re.search(
            r"(?:bot|autoroll|automation|updater)@|\[auto[- ]?(?:roll|update)\]",
            commit.author,
            re.IGNORECASE,
        )
        or re.search(r"^Bot-Commit:", commit.message, re.MULTILINE)
    ):
        return RejectReason.AUTOMATED
    return None


def _has_symptom(message: str) -> bool:
    """Checks for explanatory prose beyond a bug reference or fix label."""
    prose = strip_trailers(message)
    prose = re.sub(r"(?:https?://\S+|\b(?:crbug\.com/)?\d{5,}\b)", "", prose)
    return bool(
        re.search(
            r"[A-Za-z]{3,}.*\b(?:when|if|fails?|crash(?:es)?|wrong|broken|error|missing|not|instead|cannot|incorrect|regression)\b",
            prose,
            re.IGNORECASE,
        )
    )


def is_fix_commit(message: str) -> bool:
    """Returns whether a commit message describes a bug fix."""
    subject = message.split("\n", 1)[0]
    if _NOT_A_FIX_SUBJECT.match(subject):
        return False
    return bool(_FIX_TRAILER.search(message) or _FIX_SUBJECT.search(subject))


def build_commit_tasks(
    git: Git,
    config: CommitTaskConfig,
    rewriter: Rewriter,
    answerability_screen: Rewriter | None = None,
    task_ids: frozenset[str] | None = None,
    *,
    dry_run: bool = False,
) -> CommitTaskBuild:
    """Mines the fix commits after one snapshot into tasks.

    Tasks come out in commit order with `lexically_solvable` unset.

    Raises:
        ValueError: if `config.snapshot` names no commit.
        GitError: if a git command fails.
    """
    snapshot = git.resolve(config.snapshot)
    if snapshot is None:
        raise ValueError(
            f"snapshot {config.snapshot} does not name a commit in {git.repo}"
        )
    snapshot_date = datetime.fromisoformat(
        git.text("show", "-s", "--format=%cI", snapshot).strip()
    )
    components = PathComponentIndex(
        path
        for path in git.text("ls-tree", "-r", "-z", "--name-only", snapshot).split("\0")
        if path
    )
    tasks, rejections = [], []
    for commit in _log(
        git, f"{snapshot}..{config.head}", first_parent=config.bucket is Bucket.XL
    ):
        if task_ids is not None and _task_id(config, commit.id) not in task_ids:
            continue
        if not snapshot_date < commit.committer_date <= snapshot_date + config.window:
            continue
        if commit.author_date <= config.model_cutoff:
            continue
        if config.bucket is Bucket.XL:
            exclusion = _chromium_exclusion(commit)
            if exclusion is not None:
                rejections.append(Rejection(commit.id, exclusion))
                continue
        if not is_fix_commit(commit.message):
            continue
        if config.split is not None and (
            assign_split(_task_id(config, commit.id)) is not config.split
        ):
            continue
        outcome = _build_task(
            git,
            config,
            snapshot,
            components,
            commit,
            rewriter,
            answerability_screen,
            dry_run=dry_run,
        )
        (rejections if isinstance(outcome, Rejection) else tasks).append(outcome)
    return CommitTaskBuild(snapshot, tuple(tasks), tuple(rejections))


def _build_task(
    git: Git,
    config: CommitTaskConfig,
    snapshot: str,
    components: PathComponentIndex,
    commit: _Commit,
    rewriter: Rewriter,
    answerability_screen: Rewriter | None = None,
    *,
    dry_run: bool = False,
) -> Task | Rejection | RewriteCandidate:
    def reject(reason: RejectReason, detail: str = "") -> Rejection:
        return Rejection(commit.id, reason, detail)

    if _SYZBOT.search(commit.message):
        return reject(RejectReason.SYZBOT)
    gold, moved = [], []
    for status, paths in _changes(git, commit.id):
        candidates = [path for path in paths if is_gold_candidate(path)]
        (gold if status == "M" else moved).extend(candidates)
    if moved:
        return reject(RejectReason.NOT_IN_PLACE, ", ".join(sorted(moved)))
    if not gold:
        return reject(RejectReason.NO_SOURCE_CHANGE)
    if len(gold) > MAX_GOLD_PATHS:
        return reject(RejectReason.TOO_MANY_FILES, f"{len(gold)} gold candidates")
    gold_paths = tuple(sorted(gold))
    parent = commit.parents[0]
    at_snapshot = tree_blobs(git, snapshot, gold_paths)
    at_parent = tree_blobs(git, parent, gold_paths)
    changed = [
        path for path in gold_paths if at_snapshot.get(path) != at_parent.get(path)
    ]
    if changed:
        return reject(RejectReason.CHANGED_SINCE_SNAPSHOT, ", ".join(changed))
    late = [
        culprit
        for culprit in _CULPRIT_TRAILER.findall(commit.message)
        if (resolved := git.resolve(culprit)) is not None
        and not git.test("merge-base", "--is-ancestor", resolved, snapshot)
    ]
    if late:
        return reject(RejectReason.INTRODUCED_AFTER_SNAPSHOT, ", ".join(late))
    if config.bucket is Bucket.XL and not _has_symptom(commit.message):
        return reject(RejectReason.NO_SYMPTOM)
    if dry_run:
        return RewriteCandidate(commit.id, gold_paths, strip_trailers(commit.message))
    gold_functions = _gold_functions(git, parent, commit.id, gold_paths)
    report = rewriter(rewrite_prompt(commit.message)).strip()
    if not report:
        return reject(RejectReason.EMPTY_REWRITE)
    leaks = find_leaks(report, gold_paths, gold_functions, components)
    if leaks:
        return reject(RejectReason.LEAK, ", ".join(leaks))
    if answerability_screen is not None and not screen_answerability(
        report, answerability_screen
    ):
        return reject(RejectReason.UNANSWERABLE)
    task_id = _task_id(config, commit.id)
    return Task(
        id=task_id,
        bucket=config.bucket,
        repo=config.repo_name,
        snapshot=snapshot,
        prompt=report,
        gold_paths=gold_paths,
        gold_functions=gold_functions,
        provenance=Provenance(
            source=Source.COMMIT,
            source_id=commit.id,
            created_at=commit.author_date,
            prompt_origin=PromptOrigin.REWRITE,
            rewriter=config.rewriter_id,
        ),
        lexically_solvable=None,
        split=assign_split(task_id),
    )


def _task_id(config: CommitTaskConfig, commit: str) -> str:
    return f"{config.repo_name}/{commit[:12]}"


def _log(
    git: Git, revision_range: str, *, first_parent: bool = False
) -> Iterator[_Commit]:
    """Yields non-merge commits oldest first, optionally on the first parent."""
    output = git.text(
        "log",
        "--no-merges",
        "--reverse",
        *(("--first-parent",) if first_parent else ()),
        _LOG_FORMAT,
        revision_range,
    )
    for record in output.split("\x1e"):
        record = record.lstrip("\n")
        if not record:
            continue
        commit_id, parents, author_date, committer_date, author, message = record.split(
            "\x1f", 5
        )
        yield _Commit(
            id=commit_id,
            parents=tuple(parents.split()),
            author_date=datetime.fromisoformat(author_date),
            committer_date=datetime.fromisoformat(committer_date),
            message=message,
            author=author,
        )


def _changes(git: Git, commit: str) -> Iterator[tuple[str, tuple[str, ...]]]:
    """Yields (status letter, paths) for each file `commit` changes.

    Renames and copies carry both paths; every other status carries one.
    """
    fields = git.text(
        "diff-tree", "-r", "-z", "-M", "--no-commit-id", "--name-status", commit
    )
    tokens = fields.split("\0")
    index = 0
    while index < len(tokens) and tokens[index]:
        status = tokens[index][0]
        width = 2 if status in "RC" else 1
        yield status, tuple(tokens[index + 1 : index + 1 + width])
        index += 1 + width


def _gold_functions(
    git: Git, parent: str, commit: str, gold_paths: tuple[str, ...]
) -> tuple[str, ...]:
    """Returns "path::name" for the definitions changed in C-family gold files."""
    c_paths = [path for path in gold_paths if is_c_family(path)]
    if not c_paths:
        return ()
    diff = git.text(
        "diff", "-U0", "--no-color", "--no-ext-diff", "--no-textconv", "--no-renames",
        "--src-prefix=a/", "--dst-prefix=b/", parent, commit, "--", *c_paths,
    )  # fmt: skip
    functions = set()
    for file_diff in parse_unified_diff(diff):
        path = file_diff.old_path
        if path is None:
            continue
        source = git.text("cat-file", "blob", f"{parent}:{path}")
        functions.update(
            f"{path}{FUNCTION_SEPARATOR}{name}"
            for name in changed_definitions(source, file_diff.hunks)
        )
    return tuple(sorted(functions))


def screen_answerability(report: str, judge: Rewriter) -> bool:
    """Asks a second model if the report gives enough clues, without showing gold."""
    decision = (
        judge(
            "Does this bug report identify the affected product or hardware and "
            "symptoms precisely enough to localize a fault? Reply YES or NO only.\n"
            + report
        )
        .strip()
        .upper()
    )
    if decision not in ("YES", "NO"):
        raise ValueError(f"answerability judge must return YES or NO, got {decision!r}")
    return decision == "YES"

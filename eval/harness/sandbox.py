# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Read-only access to one repository snapshot for `read`, `ls`, and `find`.

The agent must not see anything outside the snapshot, and must not see git
metadata, which could leak history (the fix commit) even in a history-free
snapshot. Every path goes through `RepoSandbox.resolve`:

* absolute paths are refused;
* the path is resolved with `os.path.realpath`, so `..` and symlinks are
  followed, and the result must lie inside the resolved root;
* any component named `.git`, compared case-insensitively because APFS is,
  in the requested or the resolved path reads as nonexistent.

`ls` and `find` list exactly the entries that `resolve` accepts, so the agent
never sees a name it cannot open. Snapshots are immutable during a run, so a
check-then-open race cannot be exploited; files are still opened with
O_NOFOLLOW and O_NONBLOCK so that a symlink or a FIFO can never be read.
"""

import errno
import os
import re
import stat
import threading
from collections.abc import Iterable
from pathlib import PurePosixPath

from harness.tools import LIST_MAX_ENTRIES, READ_PAGE_BYTES, READ_PAGE_LINES

# Files with a NUL byte in their first 8 KiB are binary, matching Ferret's
# eligibility rule.
_BINARY_PROBE_BYTES = 8 * 1024

# Room kept inside READ_PAGE_BYTES for the footer line of a page, besides the
# file's path.
_FOOTER_RESERVE_BYTES = 96

_TRUNCATED_LINE_MARK = " [line truncated]"


class SandboxError(Exception):
    """A refused or failed operation; `message` is shown to the model.

    Attributes:
        message: Model-facing text. It never contains an absolute path.
        rejected: True if the call violated the sandbox policy (escape
            attempts, `.git`), False for ordinary failures such as a missing
            file.
    """

    def __init__(self, message: str, *, rejected: bool) -> None:
        super().__init__(message)
        self.message = message
        self.rejected = rejected


class RepoSandbox:
    """Read-only view of a repository snapshot.

    Thread-safe: the snapshot is immutable, and the lazily built file list is
    guarded by a lock, so one instance can serve concurrent episodes.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        """Creates a sandbox rooted at `root`.

        Raises:
            ValueError: if `root` is not a directory.
        """
        resolved = os.path.realpath(root)
        if not os.path.isdir(resolved):
            raise ValueError(f"snapshot root {root} is not a directory")
        self._root = resolved
        self._root_prefix = os.path.join(resolved, "")
        self._files: tuple[str, ...] | None = None
        self._files_lock = threading.Lock()

    @property
    def root(self) -> str:
        """The resolved absolute path of the snapshot."""
        return self._root

    def resolve(self, path: str) -> str:
        """Maps a repository-relative path to an absolute path inside the root.

        Raises:
            SandboxError: if the path is absolute, escapes the root, or names
                git metadata. The path need not exist.
        """
        if path.startswith("/"):
            raise SandboxError(
                f"error: path must be relative to the repository root: {path}",
                rejected=True,
            )
        if _names_git(path.split("/")):
            raise SandboxError(_not_found(path), rejected=True)
        resolved = os.path.realpath(os.path.join(self._root, path))
        if resolved != self._root and not resolved.startswith(self._root_prefix):
            raise SandboxError(
                f"error: path is outside the repository: {path}", rejected=True
            )
        if _names_git(resolved[len(self._root_prefix) :].split(os.sep)):
            raise SandboxError(_not_found(path), rejected=True)
        return resolved

    def read_page(self, path: str, start_line: int) -> str:
        """Returns one page of a text file as numbered lines plus a footer.

        A page holds up to READ_PAGE_LINES lines and at most READ_PAGE_BYTES
        of UTF-8 including the footer, which names the file by its
        repository-relative path after symlinks and the next start line. A
        line too long for an empty page is shown truncated. Invalid UTF-8 is
        shown as U+FFFD.

        Raises:
            SandboxError: if the path is refused, is not a regular file, is
                binary, or `start_line` is past the end of the file.
        """
        resolved = self.resolve(path)
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
        try:
            fd = os.open(resolved, flags)
        except OSError as error:
            raise _os_error(path, error) from None
        # Checked before fdopen, which refuses a directory with its own error.
        mode = os.fstat(fd).st_mode
        if not stat.S_ISREG(mode):
            os.close(fd)
            if stat.S_ISDIR(mode):
                raise SandboxError(f"error: is a directory: {path}", rejected=False)
            raise SandboxError(f"error: not a regular file: {path}", rejected=False)
        with os.fdopen(fd, "rb") as file:
            if b"\0" in file.read(_BINARY_PROBE_BYTES):
                raise SandboxError(f"error: binary file: {path}", rejected=False)
            file.seek(0)
            shown = _display(os.path.relpath(resolved, self._root))
            return _render_page(path, shown, file, start_line)

    def list_dir(self, path: str) -> str:
        """Returns the sorted entries of a directory, one per line.

        Directories end with '/'. Git metadata, special files, and symlinks
        that `resolve` refuses or that dangle are omitted.

        Raises:
            SandboxError: if the path is refused or is not a directory.
        """
        resolved = self.resolve(path)
        try:
            names = os.listdir(resolved)
        except NotADirectoryError:
            raise SandboxError(
                f"error: not a directory: {path}", rejected=False
            ) from None
        except OSError as error:
            raise _os_error(path, error) from None
        entries = []
        for name in sorted(names):
            if name.casefold() == ".git":
                continue
            relative = os.path.relpath(os.path.join(resolved, name), self._root)
            kind = self._kind(relative)
            if kind is not None:
                entries.append(_display(name) + ("/" if kind == "dir" else ""))
        if not entries:
            return "[empty directory]"
        return _capped(entries, "entries")

    def find(self, glob: str) -> str:
        """Returns the sorted paths of files matching `glob`, one per line.

        `*` and `?` match within a component, but a matched directory
        includes its descendants, as in search's gitignore-style glob.
        `**` crosses directories; a glob without '/' matches at any depth.
        """
        match_whole_path = "/" in glob
        pieces = _literal_pieces(glob)
        memo: dict[str, bool] = {}
        matches = [
            path
            for path in self._file_list()
            if all(piece in path for piece in pieces)
            and _matches_glob(path, glob, match_whole_path, memo)
        ]
        if not matches:
            return "[no matching files]"
        return _capped([_display(path) for path in matches], "matches")

    def _kind(self, relative: str) -> str | None:
        """Returns "dir" or "file" if `relative` is accessible, else None."""
        try:
            info = os.stat(self.resolve(relative))
        except (SandboxError, OSError):
            return None
        if stat.S_ISDIR(info.st_mode):
            return "dir"
        if stat.S_ISREG(info.st_mode):
            return "file"
        return None

    def _file_list(self) -> tuple[str, ...]:
        with self._files_lock:
            if self._files is None:
                self._files = self._walk()
            return self._files

    def _walk(self) -> tuple[str, ...]:
        """Lists every accessible regular file, without following directory
        symlinks, so that each file appears once under its real directory."""
        files = []
        pending = [""]
        while pending:
            directory = pending.pop()
            try:
                scanner = os.scandir(os.path.join(self._root, directory))
            except OSError:
                continue
            with scanner:
                for entry in scanner:
                    if entry.name.casefold() == ".git":
                        continue
                    relative = f"{directory}/{entry.name}" if directory else entry.name
                    if entry.is_symlink():
                        if self._kind(relative) == "file":
                            files.append(relative)
                    elif entry.is_dir(follow_symlinks=False):
                        pending.append(relative)
                    elif entry.is_file(follow_symlinks=False):
                        files.append(relative)
        files.sort()
        return tuple(files)


def _names_git(components: list[str]) -> bool:
    return any(component.casefold() == ".git" for component in components)


def _not_found(path: str) -> str:
    return f"error: no such file or directory: {path}"


def _os_error(path: str, error: OSError) -> SandboxError:
    """Converts an OSError without leaking the absolute path it carries."""
    if error.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
        return SandboxError(_not_found(path), rejected=False)
    return SandboxError(f"error: cannot open {path}: {error.strerror}", rejected=False)


def _display(name: str) -> str:
    """Makes a file name from os.listdir safe to put in a JSON request."""
    return name.encode("utf-8", "surrogateescape").decode("utf-8", "replace")


def _capped(lines: list[str], noun: str) -> str:
    shown = lines[:LIST_MAX_ENTRIES]
    if len(lines) > LIST_MAX_ENTRIES:
        shown.append(f"[showing {LIST_MAX_ENTRIES} of {len(lines)} {noun}]")
    return "\n".join(shown)


def _render_page(path: str, shown: str, lines: Iterable[bytes], start_line: int) -> str:
    """Renders the page of `lines`, the raw lines of the file requested as
    `path`, at `start_line`; the footer names the file as `shown`."""
    budget = READ_PAGE_BYTES - _FOOTER_RESERVE_BYTES - len(shown.encode("utf-8"))
    page: list[str] = []
    used = 0
    last = 0
    total = 0
    full = False
    for number, raw in enumerate(lines, start=1):
        total = number
        if full or number < start_line:
            continue
        text = raw.rstrip(b"\n").removesuffix(b"\r").decode("utf-8", "replace")
        rendered = f"{number:6d}\t{text}\n"
        size = len(rendered.encode("utf-8"))
        if used + size > budget:
            if not page:
                room = budget - len(_TRUNCATED_LINE_MARK) - 1
                head = rendered.encode("utf-8")[:room].decode("utf-8", "ignore")
                page.append(f"{head}{_TRUNCATED_LINE_MARK}\n")
                last = number
            full = True
            continue
        page.append(rendered)
        used += size
        last = number
        full = len(page) == READ_PAGE_LINES
    if total == 0:
        return f"[{shown}: empty file]"
    if start_line > total:
        raise SandboxError(
            f"error: start_line {start_line} is past the end of {path} ({total} lines)",
            rejected=False,
        )
    footer = f"[{shown}: lines {start_line}-{last} of {total}"
    footer += f"; next start_line={last + 1}]" if last < total else "]"
    return "".join(page) + footer


def _literal_pieces(glob: str) -> list[str]:
    """Returns substrings that every path matching `glob` contains.

    `full_match` matches the pattern `str(PurePosixPath(glob))`, in which
    only `*`, `?`, and `[` are special and every other character matches
    itself, case-sensitively. A candidate is a prefix or a component of the
    path, so each maximal run of plain characters between `*`, `?`, and `/`
    occurs in the path. A pattern with `[` gets no pieces.
    """
    pattern = str(PurePosixPath(glob))
    if "[" in pattern:
        return []
    return [piece for piece in re.split(r"[*?/]", pattern) if piece and piece != "."]


def _matches_glob(
    path: str, glob: str, whole_path: bool, memo: dict[str, bool] | None = None
) -> bool:
    """Matches files and their ancestor directories, as search does.

    `memo` caches `full_match` per candidate for one glob, since ancestors
    and names repeat across the paths of one `find`.
    """
    parts = path.split("/")
    for length in range(1, len(parts) + 1):
        candidate = "/".join(parts[:length]) if whole_path else parts[length - 1]
        hit = None if memo is None else memo.get(candidate)
        if hit is None:
            hit = PurePosixPath(candidate).full_match(glob)
            if memo is not None:
                memo[candidate] = hit
        if hit:
            return True
    return False

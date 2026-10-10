# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Owns one persistent daemon per snapshot for concurrent MCP episodes."""

import hashlib
import logging
import os
import socket
import subprocess
import time
from pathlib import Path

from harness.mcp_client import WARM_UP_TIMEOUT_S, McpClient, wait_until_indexed
from harness.search import _child_environment

logger = logging.getLogger(__name__)


class ManagedServers:
    """Starts daemons before clients can register; reaps only our own children."""

    def __init__(self, binary: Path, roots: list[Path], index_dir: Path) -> None:
        self.index_dir = index_dir
        self._binary = binary
        self._children: list[subprocess.Popen[bytes]] = []
        self._roots: dict[Path, subprocess.Popen[bytes]] = {}
        try:
            for root in roots:
                canonical = root.resolve()
                # Git worktrees key their daemon by the common Git directory.
                git = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(canonical),
                        "rev-parse",
                        "--path-format=absolute",
                        "--git-common-dir",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                common = (
                    Path(git.stdout.strip()).resolve()
                    if git.returncode == 0
                    else canonical
                )
                key = hashlib.sha256(str(common).encode()).hexdigest()[:16]
                sock = Path(os.environ.get("TMPDIR", "/tmp")) / (
                    f"ferret-{os.geteuid()}/ferret-{key}.sock"
                )
                if _listening(sock):
                    raise RuntimeError(f"MCP daemon already serves {canonical}")
                # The harness reaps its daemons, and a snapshot's first task
                # may come hours into a shuffled run, so they never idle out.
                child = subprocess.Popen(
                    [
                        str(binary),
                        "serve",
                        f"--index_dir={index_dir}",
                        "--idle_timeout=inf",
                        str(canonical),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=_child_environment(),
                )
                self._children.append(child)
                self._roots[canonical] = child
                deadline = time.monotonic() + 30
                while not _listening(sock):
                    if child.poll() is not None:
                        raise RuntimeError(f"MCP daemon exited for {canonical}")
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"MCP daemon startup timed out: {canonical}")
                    time.sleep(0.05)
        except BaseException:
            self.close()
            raise

    def warm_up(self, timeout_s: float = WARM_UP_TIMEOUT_S) -> None:
        """Registers each root through one client and waits until it is
        indexed, so no episode or schema check starts on a partial index."""
        for root in self._roots:
            start = time.monotonic()
            with McpClient(
                self._binary, root, self.index_dir, "context", timeout_s=timeout_s
            ) as client:
                status = wait_until_indexed(client, timeout_s)
            logger.info(
                "indexed %s in %.1f s: %s", root, time.monotonic() - start, status
            )

    def assert_alive(self, root: Path) -> None:
        """Fails an episode if its owned daemon died, without restarting it."""
        child = self._roots[root.resolve()]
        if child.poll() is not None:
            raise RuntimeError(f"MCP daemon exited for {root}")

    def close(self) -> None:
        """Terminates and reaps every daemon this manager started."""
        for child in self._children:
            if child.poll() is None:
                child.terminate()
        for child in self._children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        self._children.clear()
        self._roots.clear()

    def __enter__(self) -> "ManagedServers":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _listening(path: Path) -> bool:
    try:
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(0.25)
            connection.connect(str(path))
        return True
    except OSError:
        return False

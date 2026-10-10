# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""A bounded, owned stdio MCP session for Ferret's search tool."""

import hashlib
import json
import os
import re
import select
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from harness.search import NO_MATCHES, SearchResult, _child_environment
from harness.tools import ToolSpec, canonical_json


class McpError(RuntimeError):
    """The MCP transport or protocol failed; the episode must be recorded as failed."""


class McpRpcError(McpError):
    """A server JSON-RPC error, distinct from a broken transport."""


# Search errors that no model argument can correct: the daemon or the root
# setup is broken, so the episode fails instead of recording the error as a
# tool result. The worktree errors failed every search of the first main run.
_INFRASTRUCTURE_ERRORS = (
    "daemon unavailable",
    "daemon exited",
    "daemon connection",
    "worktree must be a registered root",
    "worktree is required",
)


class McpClient:
    """Owns one MCP process and closes it even after a failed request."""

    def __init__(
        self,
        binary: Path,
        root: Path,
        index_dir: Path,
        shape: str,
        *,
        timeout_s: float = 60.0,
        protocol_version: str = "2025-11-25",
        router: bool = False,
        fixed_shape: bool = True,
        daemon_alive: Callable[[], None] | None = None,
    ) -> None:
        self.shape = shape
        self._daemon_alive = daemon_alive
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._id = 0
        self._buffer = bytearray()
        self._process = subprocess.Popen(
            [
                str(binary),
                "mcp",
                *(["--fixed_shape", "--spawn_daemon=false"] if fixed_shape else []),
                *(["--router"] if router else []),
                f"--shape={shape}",
                "--serve_consistency=none",
                f"--index_dir={index_dir}",
                str(root),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_child_environment(),
        )
        self._binary = binary
        self._root = root
        self._index_dir = index_dir
        self._version = protocol_version
        try:
            # Prefer the parked client's stateless modern discovery. Older
            # servers reject the probe; only then use the legacy handshake.
            try:
                self._version = "2026-07-28"
                self._exchange("server/discover", {})
            except McpRpcError:
                self._version = protocol_version
                initialized = self._exchange(
                    "initialize",
                    {
                        "protocolVersion": protocol_version,
                        "capabilities": {},
                        "clientInfo": {"name": "ferret-eval", "version": "1"},
                    },
                    meta=False,
                )
                self._version = initialized["protocolVersion"]
                self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            tools = []
            cursor = None
            while True:
                page = self._exchange(
                    "tools/list", {} if cursor is None else {"cursor": cursor}
                )
                tools.extend(page["tools"])
                cursor = page.get("nextCursor")
                if not cursor:
                    break
            search = next(tool for tool in tools if tool["name"] == "search")
            if fixed_shape and "shape" in search["inputSchema"].get("properties", {}):
                raise McpError("fixed-shape MCP server exposes a model-supplied shape")
            self.tool = ToolSpec("search", search["description"], search["inputSchema"])
            self.schema_sha256 = hashlib.sha256(
                canonical_json(
                    {
                        "name": search["name"],
                        "description": search["description"],
                        "inputSchema": search["inputSchema"],
                    }
                ).encode()
            ).hexdigest()
        except (KeyError, StopIteration, TypeError, ValueError, McpError) as error:
            self.close()
            raise McpError(f"MCP startup failed: {error}") from error

    def _send(self, payload: dict[str, Any]) -> None:
        assert self._process.stdin is not None
        try:
            self._process.stdin.write((canonical_json(payload) + "\n").encode())
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise McpError(f"MCP server exited: {error}") from error

    def _exchange(
        self, method: str, params: dict[str, Any], *, meta: bool = True
    ) -> dict[str, Any]:
        self._id += 1
        if meta and self._version >= "2026-07-28":
            params = {
                **params,
                "_meta": {
                    "io.modelcontextprotocol/protocolVersion": self._version,
                    "io.modelcontextprotocol/clientCapabilities": {},
                    "io.modelcontextprotocol/clientInfo": {
                        "name": "ferret-eval",
                        "version": "1",
                    },
                },
            }
        self._send(
            {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}
        )
        deadline = time.monotonic() + self.timeout_s
        while True:
            reply = self._receive(deadline, method)
            if "method" in reply:
                if "id" in reply:
                    self._send(
                        {
                            "jsonrpc": "2.0",
                            "id": reply["id"],
                            "error": {
                                "code": -32601,
                                "message": "client does not support server requests",
                            },
                        }
                    )
                continue
            if reply.get("id") != self._id:
                if "id" in reply:
                    raise McpError(f"MCP {method} returned a mismatched id")
                continue
            if "error" in reply:
                raise McpRpcError(f"MCP {method}: {reply['error']}")
            return reply["result"]

    def _receive(self, deadline: float, method: str) -> dict[str, Any]:
        """Reads a bounded JSON line without blocking past the request deadline."""
        assert self._process.stdout is not None
        fd = self._process.stdout.fileno()
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._buffer[:newline])
                del self._buffer[: newline + 1]
                if not line.strip():
                    continue
                try:
                    reply = json.loads(line)
                except ValueError as error:
                    raise McpError("MCP server sent invalid JSON") from error
                if not isinstance(reply, dict):
                    raise McpError("MCP server sent a non-object message")
                return reply
            if len(self._buffer) > 64 << 20:
                raise McpError("MCP server sent a line longer than 64 MiB")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise McpError(f"MCP {method} timed out after {self.timeout_s:g} s")
            readable, _, _ = select.select([fd], [], [], remaining)
            if not readable:
                continue
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                raise McpError(f"MCP server exited during {method}")
            self._buffer.extend(chunk)

    def search(self, arguments: Mapping[str, Any]) -> SearchResult:
        """Calls search; transport failures abort the episode."""
        result = self.call_search(arguments)
        text = "\n".join(
            block["text"] for block in result["content"] if block.get("type") == "text"
        )
        error = result.get("isError", False)
        if error and any(marker in text.lower() for marker in _INFRASTRUCTURE_ERRORS):
            raise McpError(f"MCP search infrastructure failure: {text}")
        return SearchResult(f"error: {text}" if error else (text or NO_MATCHES), error)

    def call_search(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Returns the raw tools/call result of one search."""
        with self._lock:
            if self._daemon_alive is not None:
                self._daemon_alive()
            result = self._exchange(
                "tools/call", {"name": "search", "arguments": dict(arguments)}
            )
            if self._daemon_alive is not None:
                self._daemon_alive()
            return result

    def index_status(self) -> list[dict[str, Any]]:
        """Returns index_status's structured status of each registered root."""
        with self._lock:
            result = self._exchange(
                "tools/call", {"name": "index_status", "arguments": {}}
            )
        if result.get("isError", False):
            raise McpError(f"index_status failed: {result.get('content')}")
        return list(result["structuredContent"]["worktrees"])

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "ferret_mcp",
            "binary": str(self._binary),
            "root": str(self._root),
            "index_dir": str(self._index_dir),
            "shape": self.shape,
            "tool_schema_sha256": self.schema_sha256,
        }

    def close(self) -> None:
        """Reaps the child even if it hung or crashed."""
        if self._process.stdin:
            self._process.stdin.close()
        try:
            self._process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        if self._process.stdout:
            self._process.stdout.close()

    def __enter__(self) -> "McpClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# A daemon indexes a snapshot when its first client registers it, before that
# client's discovery returns: minutes on Linux or Chromium, far past a search
# call's timeout.
WARM_UP_TIMEOUT_S = 1800.0


def is_indexed(worktrees: Sequence[Mapping[str, Any]]) -> bool:
    """Whether every root's visible files are indexed and their symbols
    extracted, so no episode sees a partial index or partial definitions."""
    return bool(worktrees) and all(
        status["queued_paths"] == 0
        and status["applied_generation"] == status["assigned_generation"]
        and status["pending_symbols"] == 0
        for status in worktrees
    )


def wait_until_indexed(
    client: "McpClient",
    timeout_s: float = WARM_UP_TIMEOUT_S,
    poll_s: float = 2.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Polls index_status until `is_indexed`; returns the final status."""
    deadline = clock() + timeout_s
    while True:
        worktrees = client.index_status()
        if is_indexed(worktrees):
            return worktrees
        if clock() >= deadline:
            raise McpError(f"index not ready after {timeout_s:g} s: {worktrees}")
        sleep(poll_s)


# gpt-6-luna fills every advertised search property. The first argument set
# keeps each default and sends "" for a string without one; the second uses
# its most common values in the first main run, with "." for such strings.
_TYPICAL_SEARCH_ARGUMENTS = {
    "path_glob": "**",
    "max_files": 20,
    "max_bytes": 12000,
    "context_lines": 3,
}


def canary_arguments(schema: Mapping[str, Any], pattern: str) -> list[dict[str, Any]]:
    """Model-shaped search arguments for every property of `schema`."""
    properties = schema.get("properties", {})
    sets = []
    for filler, overrides in (("", {}), (".", _TYPICAL_SEARCH_ARGUMENTS)):
        arguments: dict[str, Any] = {}
        for name, spec in properties.items():
            if name in overrides:
                arguments[name] = overrides[name]
            elif "default" in spec:
                arguments[name] = spec["default"]
            elif spec.get("type") == "boolean":
                arguments[name] = False
            elif spec.get("type") == "integer":
                arguments[name] = max(1, spec.get("minimum", 1))
            else:
                arguments[name] = filler
        arguments["pattern"] = pattern
        sets.append(arguments)
    return sets


_WORD = re.compile(rb"[A-Za-z_][A-Za-z0-9_]{5,}")


def canary_patterns(root: Path, limit: int = 5) -> list[str]:
    """Words from the first text files at the top of `root`; the canary
    tries several in case an index skips a file."""
    patterns = []
    for path in sorted(root.iterdir()):
        if len(patterns) >= limit:
            break
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1 << 20:
            continue
        data = path.read_bytes()
        try:
            data.decode()
        except UnicodeDecodeError:
            continue
        word = _WORD.search(data)
        if word is not None and word.group().decode() not in patterns:
            patterns.append(word.group().decode())
    return patterns


def check_search_canary(client: "McpClient", root: Path) -> None:
    """Raises McpError unless model-shaped searches succeed and find a word of
    the snapshot, so a server that fails every search stops the run before
    any episode."""
    patterns = canary_patterns(root)
    if not patterns:
        raise McpError(f"no canary word in the top-level files of {root}")
    for index in range(2):
        found = False
        for pattern in patterns:
            arguments = canary_arguments(client.tool.parameters, pattern)[index]
            result = client.call_search(arguments)
            if result.get("isError", False):
                raise McpError(
                    f"canary search {canonical_json(arguments)} failed: "
                    f"{result.get('content')}"
                )
            if result.get("structuredContent", {}).get("matching_files", 0) > 0:
                found = True
                break
        if not found:
            raise McpError(f"canary searches for {patterns} in {root} found nothing")

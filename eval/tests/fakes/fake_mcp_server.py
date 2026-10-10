#!/usr/bin/env python3
# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""A scripted newline-delimited MCP server; root/mode file selects behavior."""

import json
import sys
from pathlib import Path

if "--helpfull" in sys.argv:
    print("--fixed_shape --spawn_daemon --router")
    sys.exit(0)
mode = (Path(sys.argv[-1]) / "mode").read_text().strip()
assert "--fixed_shape" in sys.argv
assert "--spawn_daemon=false" in sys.argv
shape = next(arg.partition("=")[2] for arg in sys.argv if arg.startswith("--shape="))
router = "--router" in sys.argv
response_schema_properties = {"pattern", "regex", "case_sensitive", "path_glob"}
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request or "method" not in request:
        continue
    method = request["method"]
    if method == "server/discover" and mode == "legacy":
        print(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "error": {"code": -32601, "message": "method not found"},
                }
            ),
            flush=True,
        )
        continue
    if method == "server/discover":
        response = {
            "supportedVersions": ["2026-07-28"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "1"},
        }
    elif method == "initialize":
        response = {
            "protocolVersion": "2024-11-05" if mode == "legacy" else "2026-07-28",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "1"},
        }
    elif (
        mode == "modern"
        and request["params"]
        .get("_meta", {})
        .get("io.modelcontextprotocol/protocolVersion")
        != "2026-07-28"
    ):
        print(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "error": {"code": -32602, "message": "missing version"},
                }
            ),
            flush=True,
        )
        continue
    elif method == "tools/list":
        response = {
            "tools": [
                {
                    "name": "search",
                    "description": "fixed",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "regex": {"type": "boolean"},
                            "case_sensitive": {"type": "boolean"},
                            "path_glob": {"type": "string"},
                        },
                        "required": ["pattern"],
                        "additionalProperties": False,
                    },
                },
                {"name": "index_status", "description": "secret", "inputSchema": {}},
            ]
        }
        if mode == "paginated":
            if request["params"].get("cursor") == "page-2":
                response["tools"] = [response["tools"][0]]
            else:
                response["tools"] = [response["tools"][1]]
                response["nextCursor"] = "page-2"
        if mode == "different":
            response["tools"][0]["description"] = shape
        if mode == "worktree_trap":
            response["tools"][0]["inputSchema"]["properties"]["worktree"] = {
                "type": "string"
            }
        if mode == "has_shape":
            response["tools"][0]["inputSchema"]["properties"]["shape"] = {
                "type": "string",
                "default": shape,
            }
    elif method == "tools/call":
        arguments = request["params"]["arguments"]
        pattern = arguments["pattern"]
        allowed = response_schema_properties | (
            {"worktree"} if mode == "worktree_trap" else set()
        )
        if set(arguments) - allowed:
            response = {
                "content": [{"type": "text", "text": "invalid params"}],
                "isError": True,
            }
            print(
                json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": response}),
                flush=True,
            )
            continue
        if arguments.get("worktree", sys.argv[-1]) != sys.argv[-1]:
            response = {
                "content": [
                    {"type": "text", "text": "worktree must be a registered root"}
                ],
                "isError": True,
            }
            print(
                json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": response}),
                flush=True,
            )
            continue
        if pattern == "daemon_crash":
            (Path(sys.argv[-1]) / "daemon_dead").write_text("dead")
        if (Path(sys.argv[-1]) / "daemon_dead").exists():
            response = {
                "content": [{"type": "text", "text": "daemon unavailable"}],
                "isError": True,
            }
            print(
                json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": response}),
                flush=True,
            )
            continue
        if pattern == "__hang__":
            continue
        if pattern == "__server_request__":
            print(
                json.dumps(
                    {"jsonrpc": "2.0", "id": "server-1", "method": "roots/list"}
                ),
                flush=True,
            )
        if pattern in ("crash", "__crash__"):
            sys.exit(3)
        response = {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "bad regex"
                        if pattern == "error"
                        else json.dumps(arguments, sort_keys=True)
                        if pattern == "echo_args"
                        else f"route:{pattern}"
                        if router and pattern.startswith("symbol:")
                        else f"{shape}:{pattern}"
                    ),
                }
            ],
            "isError": pattern == "error",
        }
        if mode != "legacy":
            response["structuredContent"] = {
                "text": f"{shape}:{pattern}",
                "matching_files": 0 if mode == "no_matches" else 1,
            }
    else:
        response = {}
    print(
        json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": response}),
        flush=True,
    )

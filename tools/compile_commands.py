# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Convert Bazel CppCompile aquery actions into clangd compilation commands.

Bazel's Apple compiler wrapper and module maps only work inside Bazel actions.
Use the host compiler and the same include roots and definitions without
layering_check, while resolving external repositories in Bazel's execution root.
"""

import argparse
import json
from pathlib import Path


def include_path(path: str, workspace: Path, execution_root: Path) -> str:
    if path == ".":
        return str(workspace)
    if path.startswith(("external/", "bazel-out/")):
        return str(execution_root / path)
    return str(workspace / path)


def compile_command(
    action: dict, workspace: Path, execution_root: Path, compiler: str
) -> dict:
    args = action["arguments"]
    source = args[args.index("-c") + 1]
    command = [compiler, "-fsyntax-only"]
    i = 1
    while i < len(args):
        flag = args[i]
        if flag in ("-iquote", "-I", "-isystem"):
            # Bazel's -iquote roots must also find angle-bracket includes.
            command.extend(["-I", include_path(args[i + 1], workspace, execution_root)])
            i += 2
        elif flag.startswith("-I") and len(flag) > 2:
            command.extend(["-I", include_path(flag[2:], workspace, execution_root)])
            i += 1
        elif flag.startswith(("-D", "-U", "-std=", "-fno-exceptions")):
            command.append(flag)
            i += 1
        else:
            i += 1
    command.append(str(workspace / source))
    return {
        "directory": str(workspace),
        "file": str(workspace / source),
        "arguments": command,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aquery", type=Path)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("execution_root", type=Path)
    parser.add_argument("compiler")
    args = parser.parse_args()
    actions = json.loads(args.aquery.read_text())["actions"]
    commands = [
        compile_command(action, args.workspace, args.execution_root, args.compiler)
        for action in actions
    ]
    (args.workspace / "compile_commands.json").write_text(
        json.dumps(commands, indent=2) + "\n"
    )
    print(f"Wrote {len(commands)} commands to compile_commands.json")


if __name__ == "__main__":
    main()

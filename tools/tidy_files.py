# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Select compilation units and non-excluded headers for tools/tidy.sh."""

import json
import re
from pathlib import Path
import sys


_SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx"})


def select_files(
    commands: list[dict], root: Path, selectors: list[Path], exclusions: list[Path]
) -> tuple[list[Path], str]:
    """Returns selected sources and a header filter excluding deferred packages."""
    sources = set()
    packages = set()
    for command in commands:
        file = Path(command["file"]).resolve()
        if not file.is_relative_to(root):
            continue
        relative = file.relative_to(root)
        if relative.parts[0] not in ("ferret", "bench"):
            continue
        if any(file.is_relative_to(exclusion) for exclusion in exclusions):
            continue
        if file.suffix not in _SOURCE_SUFFIXES:
            continue
        packages.add("/".join(relative.parts[:2]))
        if not selectors or any(
            file == path or file.is_relative_to(path) for path in selectors
        ):
            sources.add(file)
    # The header filter covers all active packages, not just the selected
    # translation units, so a scoped run also checks their direct headers.
    prefix = re.escape(str(root)) + "/"
    header_filter = "^" + prefix + "(?:" + "|".join(map(re.escape, sorted(packages))) + ")/"
    return sorted(sources), header_filter


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2])
    header_filter_output = Path(sys.argv[3])
    selectors = [Path(path).resolve() for path in sys.argv[4:]]
    for path in selectors:
        if not path.is_relative_to(root) or not path.exists():
            raise SystemExit(f"tidy: not a workspace file or directory: {path}")
    exclusions = [
        root / line.strip()
        for line in (root / "tools/tidy_exclude.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    commands = json.loads((root / "compile_commands.json").read_text())
    files, header_filter = select_files(commands, root, selectors, exclusions)
    output.write_bytes(b"".join(str(file).encode() + b"\0" for file in files))
    header_filter_output.write_text(header_filter)


if __name__ == "__main__":
    main()

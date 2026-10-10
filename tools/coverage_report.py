# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Summarize a Bazel LCOV report as line coverage per first-party package.

tools/coverage.sh passes the combined report and the `bazel query` labels of
the source files of non-testonly //ferret libraries and binaries. Only those
files count: tests, test-only libraries, generated files, and external
repositories are ignored. A line is covered when any record for its file hit
it, so a header compiled into several test binaries counts each line once.
"""

import sys
from collections import defaultdict
from pathlib import Path


def source_paths(labels: list[str]) -> set[str]:
    """Returns the workspace-relative paths of //ferret source-file labels.

    The query already leaves out tests and testonly targets, so every //ferret
    label counts, whatever its file name.
    """
    paths = set()
    for label in labels:
        label = label.strip()
        if label.startswith("//ferret/") and ":" in label:
            package, name = label[2:].split(":", 1)
            paths.add(f"{package}/{name}")
    return paths


def line_hits(report: str, sources: set[str]) -> dict[str, dict[int, int]]:
    """Returns the highest hit count of each instrumented line in `sources`.

    LCOV source paths are relative to Bazel's execution root, so generated
    files (under bazel-out/) and external repositories never match.
    """
    lines: dict[str, dict[int, int]] = defaultdict(dict)
    source = None
    for record in report.splitlines():
        if record.startswith("SF:"):
            source = record[3:] if record[3:] in sources else None
        elif record.startswith("DA:") and source is not None:
            number, hits = map(int, record[3:].split(",")[:2])
            lines[source][number] = max(hits, lines[source].get(number, 0))
        elif record == "end_of_record":
            source = None
    return lines


def package_totals(lines: dict[str, dict[int, int]]) -> dict[str, tuple[int, int]]:
    """Returns (covered, instrumented) line counts per ferret/<dir> package."""
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for path, hits in lines.items():
        package = "/".join(path.split("/")[:2])
        totals[package][0] += sum(count > 0 for count in hits.values())
        totals[package][1] += len(hits)
    return {package: (covered, total) for package, (covered, total) in totals.items()}


def format_table(totals: dict[str, tuple[int, int]]) -> str:
    """Formats per-package and overall line coverage, sorted by package."""
    rows = sorted(totals.items())
    covered = sum(covered for covered, _ in totals.values())
    total = sum(total for _, total in totals.values())
    rows.append(("overall", (covered, total)))
    width = max(len(package) for package, _ in rows)
    result = [f"{'package':<{width}}  {'lines':>6}  {'covered':>7}  coverage"]
    for package, (covered, total) in rows:
        percent = 100 * covered / total
        result.append(f"{package:<{width}}  {total:>6}  {covered:>7}  {percent:7.1f}%")
    return "\n".join(result)


def unmeasured(sources: set[str], lines: dict[str, dict[int, int]]) -> list[str]:
    """Returns the sorted .cc files in `sources` that no report record covers.

    A source that no test links has no coverage map, so its lines are unknown
    rather than uncovered. A header without records declares no code.
    """
    return sorted(
        path for path in sources if path.endswith(".cc") and path not in lines
    )


def main() -> None:
    report, labels = map(Path, sys.argv[1:])
    sources = source_paths(labels.read_text().splitlines())
    lines = line_hits(report.read_text(), sources)
    if not lines:
        raise SystemExit(f"coverage: no first-party production lines in {report}")
    print(format_table(package_totals(lines)))
    for path in unmeasured(sources, lines):
        print(f"not measured: no test links {path}")


if __name__ == "__main__":
    main()

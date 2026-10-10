# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-package LCOV summary."""

import unittest

from coverage_report import (
    format_table,
    line_hits,
    package_totals,
    source_paths,
    unmeasured,
)


class SourcePathsTest(unittest.TestCase):
    def test_keeps_every_ferret_label_and_drops_the_rest(self) -> None:
        # The query decides what is production code, so a benchmark source it
        # returns is kept and later reported as not measured.
        paths = source_paths(
            [
                "//ferret/base:env.cc",
                "//ferret/base:env.h\n",
                "//ferret/watch:internal/event_batcher.cc",
                "//ferret/index:index_ingest_benchmark.cc",
                "//bench:runner.cc",
                "@abseil-cpp//absl/strings:str_cat.cc",
                "//ferret/base",
            ]
        )
        self.assertEqual(
            paths,
            {
                "ferret/base/env.cc",
                "ferret/base/env.h",
                "ferret/watch/internal/event_batcher.cc",
                "ferret/index/index_ingest_benchmark.cc",
            },
        )


class LineHitsTest(unittest.TestCase):
    def test_merges_records_by_line_and_skips_other_files(self) -> None:
        report = """\
SF:ferret/base/env.h
DA:4,0
DA:5,2
end_of_record
SF:ferret/base/env.h
DA:4,3,abcdef
DA:5,0
DA:9,0
end_of_record
SF:bazel-out/darwin_arm64-fastbuild/bin/ferret/base/env.h
DA:1,1
end_of_record
SF:external/abseil-cpp+/ferret/base/env.h
DA:1,1
end_of_record
SF:ferret/base/env_test.cc
DA:1,1
end_of_record
DA:7,1
"""
        self.assertEqual(
            line_hits(report, {"ferret/base/env.h"}),
            {"ferret/base/env.h": {4: 3, 5: 2, 9: 0}},
        )


class PackageTotalsTest(unittest.TestCase):
    def test_groups_subdirectories_under_their_package(self) -> None:
        totals = package_totals(
            {
                "ferret/watch/watcher.cc": {1: 1, 2: 0},
                "ferret/watch/internal/event_batcher.cc": {3: 5},
                "ferret/base/env.cc": {1: 0},
            }
        )
        self.assertEqual(totals, {"ferret/watch": (2, 3), "ferret/base": (0, 1)})

    def test_formats_sorted_rows_and_overall(self) -> None:
        table = format_table({"ferret/watch": (2, 3), "ferret/base": (1, 1)})
        self.assertEqual(
            table.splitlines(),
            [
                "package        lines  covered  coverage",
                "ferret/base        1        1    100.0%",
                "ferret/watch       3        2     66.7%",
                "overall            4        3     75.0%",
            ],
        )


class UnmeasuredTest(unittest.TestCase):
    def test_names_sources_without_records_but_not_headers(self) -> None:
        sources = {
            "ferret/base/env.cc",
            "ferret/base/env.h",
            "ferret/index/index_ingest_benchmark.cc",
            "ferret/symbols/linux_corpus.cc",
        }
        self.assertEqual(
            unmeasured(sources, {"ferret/base/env.cc": {1: 0}}),
            [
                "ferret/index/index_ingest_benchmark.cc",
                "ferret/symbols/linux_corpus.cc",
            ],
        )


if __name__ == "__main__":
    unittest.main()

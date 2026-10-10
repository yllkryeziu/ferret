# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for compilation-unit selection and temporary tidy exclusions."""

import re
import tempfile
import unittest
from pathlib import Path

from tidy_files import select_files


class SelectFilesTest(unittest.TestCase):
    def test_selects_sources_and_excludes_deferred_packages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths = [
                root / "ferret/index/segment.cc",
                root / "ferret/index/segment_test.cc",
                root / "ferret/search/engine.cc",
                root / "bench/ferret/bench.cc",
                root / "bench/tools/tool.c",
                root / "outside.cc",
                root / "ferret/index/segment.h",
            ]
            commands = [{"file": str(path)} for path in paths]
            exclusions = [root / "ferret/search", root / "bench/ferret"]
            files, header_filter = select_files(commands, root, [], exclusions)
            self.assertEqual(files, [paths[4], paths[0], paths[1]])
            self.assertTrue(re.search(header_filter, str(root / "ferret/index/segment.h")))
            self.assertFalse(re.search(header_filter, str(root / "ferret/search/engine.h")))
            self.assertFalse(re.search(header_filter, str(root / "bench/ferret/bench.h")))

    def test_selectors_match_files_or_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            files = [root / "ferret/base/env.cc", root / "ferret/index/segment.cc"]
            commands = [{"file": str(path)} for path in files]
            selected, _ = select_files(commands, root, [root / "ferret/base"], [])
            self.assertEqual(selected, files[:1])
            selected, _ = select_files(commands, root, [files[1]], [])
            self.assertEqual(selected, files[1:])
            selected, _ = select_files(commands, root, [root / "ferret/search"], [])
            self.assertEqual(selected, [])


if __name__ == "__main__":
    unittest.main()

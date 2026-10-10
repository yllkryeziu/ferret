# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Which changed files count as gold locations.

A localization target is a source file the fix changes. Tests, documentation,
build files, and data are excluded: an agent that is asked where a bug lives
should not be scored on the regression test or the changelog entry that came
with the fix. Every task source applies the same rules, and every task keeps
between 1 and `MAX_GOLD_PATHS` gold files, the LocAgent convention for file
Acc@5.
"""

import posixpath
import re

# Tasks whose fix changes more gold files than this are not localization tasks
# a top-5 file ranking can solve cleanly; every source drops them.
MAX_GOLD_PATHS = 3

_SOURCE_EXTENSIONS = frozenset({
    ".c", ".cc", ".cpp", ".cxx", ".go", ".h", ".hh", ".hpp", ".hxx", ".java",
    ".js", ".jsx", ".kt", ".m", ".mm", ".py", ".rs", ".S", ".swift", ".ts", ".tsx",
})  # fmt: skip

# C and its descendants: `tasks.definitions` can find function spans in them.
_C_FAMILY_EXTENSIONS = frozenset({
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".m", ".mm",
})  # fmt: skip

# Directory names that hold tests or documentation in Linux, Chromium, and the
# LocBench repositories. Compared case-insensitively.
_TEST_OR_DOC_DIRS = frozenset({
    "doc", "docs", "documentation", "example", "examples", "selftests", "test",
    "test_data", "testdata", "testing", "tests", "unittests",
})  # fmt: skip

# File stems of tests: test_foo.py, foo_test.cc, foo_unittest.cc,
# foo_browsertest.cc, foo_kunit.c, FooTest.java, foo.test.ts, foo_fuzzer.cc.
_TEST_STEM = re.compile(
    r"^test_|[_.](unit|browser|api|ui|perf|integration)?tests?$|_kunit$|Test$|_fuzzer$"
)


def is_test_or_doc(path: str) -> bool:
    """Returns whether `path` looks like a test, test data, or documentation."""
    directories = path.split("/")[:-1]
    if any(directory.lower() in _TEST_OR_DOC_DIRS for directory in directories):
        return True
    stem, _ = posixpath.splitext(posixpath.basename(path))
    return _TEST_STEM.search(stem) is not None


def is_gold_candidate(path: str) -> bool:
    """Returns whether a change to `path` can make it a gold location."""
    _, extension = posixpath.splitext(path)
    parts = path.lower().split("/")
    generated = (
        any(part in ("gen", "generated", "out") for part in parts[:-1])
        or re.search(r"(?:_generated|_autogen|\.pb)\.[^.]+$", parts[-1]) is not None
    )
    return (
        extension in _SOURCE_EXTENSIONS and not is_test_or_doc(path) and not generated
    )


def is_c_family(path: str) -> bool:
    """Returns whether `path` is a C, C++, or Objective-C source or header."""
    _, extension = posixpath.splitext(path)
    return extension in _C_FAMILY_EXTENSIONS

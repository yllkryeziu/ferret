# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tasks.paths: which changed files count as gold locations."""

import pytest

from tasks.paths import is_c_family, is_gold_candidate, is_test_or_doc


@pytest.mark.parametrize(
    "path",
    [
        "django/db/models/query.py",
        "drivers/net/ethernet/intel/e1000/e1000_main.c",
        "include/linux/sched.h",
        "chrome/browser/ui/browser.cc",
        "arch/x86/entry/entry_64.S",
        "src/latest.py",  # "test" inside a word is not a test marker.
        "src/contest.rs",
        "web/app.tsx",
    ],
)
def test_source_files_are_gold_candidates(path):
    assert is_gold_candidate(path)


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_query.py",
        "django/tests/models.py",
        "tools/testing/selftests/net/tls.c",
        "Documentation/conf.py",  # Directories compare case-insensitively.
        "examples/demo.py",
        "pkg/testdata/input.go",
        "src/test_parser.py",
        "src/parser_test.go",
        "base/strings_unittest.cc",
        "chrome/browser/tab_browsertest.cc",
        "lib/list_kunit.c",
        "src/main/java/ParserTest.java",
        "web/parser.test.ts",
        "web/parser.spec_tests.js",
        "net/url_fuzzer.cc",
    ],
)
def test_tests_and_docs_are_not_gold_candidates(path):
    assert is_test_or_doc(path)
    assert not is_gold_candidate(path)


@pytest.mark.parametrize(
    "path",
    ["README.md", "BUILD.gn", "Makefile", "setup.cfg", "data/table.json", "a.txt"],
)
def test_non_source_files_are_not_gold_candidates(path):
    assert not is_test_or_doc(path)
    assert not is_gold_candidate(path)


def test_c_family():
    assert is_c_family("kernel/sched/core.c")
    assert is_c_family("base/files/file_path.h")
    assert is_c_family("ios/app.mm")
    assert not is_c_family("arch/x86/entry/entry_64.S")
    assert not is_c_family("django/db/models/query.py")


def test_chromium_generated_paths_are_not_gold():
    from tasks.paths import is_gold_candidate

    for path in (
        "out/gen/foo.cc",
        "src/generated/foo.h",
        "src/service.pb.cc",
        "components/theme_autogen.cc",
        "DEPS",
        "ui/translations/messages.xtb",
    ):
        assert not is_gold_candidate(path)
    assert is_gold_candidate("components/theme.cc")

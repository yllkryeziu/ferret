# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the compilation database converter."""

import unittest
from pathlib import Path

from compile_commands import compile_command, include_path


class CompileCommandsTest(unittest.TestCase):
    def test_include_roots(self):
        workspace = Path("/src/ferret")
        execution_root = Path("/cache/bazel/execroot/_main")
        self.assertEqual(include_path(".", workspace, execution_root), "/src/ferret")
        self.assertEqual(
            include_path("external/abseil-cpp+", workspace, execution_root),
            "/cache/bazel/execroot/_main/external/abseil-cpp+",
        )
        self.assertEqual(
            include_path("bazel-out/config/bin", workspace, execution_root),
            "/cache/bazel/execroot/_main/bazel-out/config/bin",
        )

    def test_removes_bazel_only_flags_but_preserves_includes_and_defines(self):
        action = {
            "arguments": [
                "wrapped_clang_pp",
                "-iquote",
                ".",
                "-iquote",
                "external/re2+",
                "-iquote",
                "bazel-out/config/bin",
                "-Iexternal/googletest+/googletest/include",
                "-I",
                "external/abseil-cpp+",
                "-isystem",
                "external/+tree_sitter+tree_sitter/lib/include",
                "-std=c++20",
                "-DDEBUG",
                "-ULEGACY",
                "-fno-exceptions",
                "-isysroot",
                "__BAZEL_XCODE_SDKROOT__",
                "-fmodules-strict-decluse",
                "-Xclang",
                "-fmodule-map-file=foo",
                "-c",
                "ferret/search/searcher.cc",
                "-o",
                "searcher.o",
            ]
        }
        result = compile_command(
            action,
            Path("/src/ferret"),
            Path("/cache/execroot/_main"),
            "/usr/bin/clang++",
        )
        self.assertEqual(result["file"], "/src/ferret/ferret/search/searcher.cc")
        self.assertEqual(
            result["arguments"],
            [
                "/usr/bin/clang++",
                "-fsyntax-only",
                "-I",
                "/src/ferret",
                "-I",
                "/cache/execroot/_main/external/re2+",
                "-I",
                "/cache/execroot/_main/bazel-out/config/bin",
                "-I",
                "/cache/execroot/_main/external/googletest+/googletest/include",
                "-I",
                "/cache/execroot/_main/external/abseil-cpp+",
                "-I",
                "/cache/execroot/_main/external/+tree_sitter+tree_sitter/lib/include",
                "-std=c++20",
                "-DDEBUG",
                "-ULEGACY",
                "-fno-exceptions",
                "/src/ferret/ferret/search/searcher.cc",
            ],
        )


if __name__ == "__main__":
    unittest.main()

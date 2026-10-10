# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tasks.diff: parsing the unified diffs git and benchmarks produce."""

import pytest

from tasks.diff import FileDiff, Hunk, parse_unified_diff

_GIT_DIFF = """\
diff --git a/src/app.py b/src/app.py
index 1111111..2222222 100644
--- a/src/app.py
+++ b/src/app.py
@@ -3,6 +3,7 @@ def main():
 context
-old
+new
+added
 context
 context
 context
 context
@@ -20 +21 @@ def helper():
-x = 1
+x = 2
diff --git a/src/new.py b/src/new.py
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/src/new.py
@@ -0,0 +1,2 @@
+line one
+line two
diff --git a/src/gone.py b/src/gone.py
deleted file mode 100644
index 4444444..0000000
--- a/src/gone.py
+++ /dev/null
@@ -1 +0,0 @@
-bye
diff --git a/logo.png b/logo.png
index 5555555..6666666 100644
Binary files a/logo.png and b/logo.png differ
diff --git a/old_name.py b/new_name.py
similarity index 90%
rename from old_name.py
rename to new_name.py
--- a/old_name.py
+++ b/new_name.py
@@ -1,2 +1,2 @@
-a
+b
 c
"""


def test_parses_files_and_hunks():
    assert parse_unified_diff(_GIT_DIFF) == [
        FileDiff(
            "src/app.py",
            "src/app.py",
            (Hunk(3, 6, 3, 7), Hunk(20, 1, 21, 1)),
        ),
        FileDiff(None, "src/new.py", (Hunk(0, 0, 1, 2),)),
        FileDiff("src/gone.py", None, (Hunk(1, 1, 0, 0),)),
        FileDiff("old_name.py", "new_name.py", (Hunk(1, 2, 1, 2),)),
    ]


def test_changed_old_lines():
    assert list(Hunk(3, 2, 3, 5).changed_old_lines()) == [3, 4]
    assert list(Hunk(7, 0, 8, 2).changed_old_lines()) == []


def test_hunk_lines_that_look_like_headers_are_body():
    diff = """\
--- a/notes.py
+++ b/notes.py
@@ -1,2 +1,2 @@
--- a/fake.py
-+++ b/fake.py
+real
+line
"""
    assert parse_unified_diff(diff) == [
        FileDiff("notes.py", "notes.py", (Hunk(1, 2, 1, 2),))
    ]


def test_skips_the_message_of_a_format_patch_mail():
    mail = """\
From 0123456789abcdef0123456789abcdef01234567 Mon Sep 17 00:00:00 2001
From: Dev <dev@example.com>
Subject: [PATCH] Fix the parser

The old parser read one byte too far.
---
 src/parse.c | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)

diff --git a/src/parse.c b/src/parse.c
--- a/src/parse.c
+++ b/src/parse.c
@@ -10,1 +10,1 @@
-  return p[n];
+  return p[n - 1];
--
2.49.0
"""
    assert parse_unified_diff(mail) == [
        FileDiff("src/parse.c", "src/parse.c", (Hunk(10, 1, 10, 1),))
    ]


def test_no_newline_markers_do_not_count_as_lines():
    diff = """\
--- a/end.py
+++ b/end.py
@@ -1 +1 @@
-old
\\ No newline at end of file
+new
\\ No newline at end of file
--- a/next.py
+++ b/next.py
@@ -1 +1 @@
-a
+b
"""
    assert [file.old_path for file in parse_unified_diff(diff)] == [
        "end.py",
        "next.py",
    ]


def test_decodes_quoted_paths_and_drops_timestamps():
    diff = """\
--- "a/caf\\303\\251 \\"menu\\".py"
+++ "b/caf\\303\\251 \\"menu\\".py"
@@ -1 +1 @@
-a
+b
--- plain.c\t2026-01-01 00:00:00.000000000 +0000
+++ plain.c\t2026-01-02 00:00:00.000000000 +0000
@@ -1 +1 @@
-a
+b
"""
    files = parse_unified_diff(diff)
    assert [file.old_path for file in files] == ['café "menu".py', "plain.c"]
    assert [file.new_path for file in files] == ['café "menu".py', "plain.c"]


def test_rejects_malformed_hunks():
    with pytest.raises(ValueError, match="malformed hunk header"):
        parse_unified_diff("--- a/x\n+++ b/x\n@@ -1,a +1 @@\n")
    with pytest.raises(ValueError, match="truncated hunk"):
        parse_unified_diff("--- a/x\n+++ b/x\n@@ -1,3 +1,3 @@\n a\n")

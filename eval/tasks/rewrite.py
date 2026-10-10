# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Turns a fix commit's message into a symptom-level bug report.

Linux and Chromium commit subjects usually name the subsystem, the file, and
the function they change ("drm/amdgpu: fix NULL deref in amdgpu_vm_bo_update"),
so using them as prompts would turn localization into a single grep. A
rewriting model, which must differ from the agent model, sees the message
without its trailers and writes the report a user would have filed. The
rewriter is any callable from prompt to text, so the pipeline runs on an LLM
in production and on a scripted function in tests; leakage checks
(`tasks.leakage`) reject whatever slips through.
"""

import re
from collections.abc import Callable

# A rewriting model: takes the full rewrite prompt, returns the report text.
type Rewriter = Callable[[str], str]

REWRITE_PROMPT = """\
You turn a fix commit into the bug report a user would have filed before the \
fix existed. The user saw the problem but has never read the source code.

Write 2 to 6 sentences: what the user did, what happened, and what they \
expected instead. Keep error messages and log lines a user would see, but \
remove function names, file names, and addresses from them.

Keep product, device, hardware, and driver names a real user would know, \
even when the commit subject names a subsystem. For example, "Elastic Fabric \
Adapter" is useful, but "efa" is a source path component and must not appear.\

Never mention: gold file or directory names or their path components; function, \
variable, type, macro, or configuration-option names; commit ids; or how the \
problem was fixed.

Commit message:
<<<
{message}
>>>

Reply with the bug report only."""

_TRAILER = re.compile(r"[A-Za-z][A-Za-z0-9-]*:\s*\S")


def strip_trailers(message: str) -> str:
    """Removes the trailing "Key: value" block (Signed-off-by, Bug, Fixes, ...).

    Trailers carry people, links, and commit ids, none of which belong in a
    user's report; they are the final paragraph when every line in it is a
    trailer or an indented continuation.
    """
    lines = message.strip().splitlines()
    end = len(lines)
    while end > 1:
        index = end - 1
        while index > 0 and not lines[index].strip():
            index -= 1
        while index > 0 and lines[index].startswith((" ", "\t")):
            index -= 1
        if not _TRAILER.match(lines[index]):
            break
        # Continuation lines belong to the last trailer, not the prose.
        start = index
        while start > 0 and (
            lines[start - 1].startswith((" ", "\t")) or _TRAILER.match(lines[start - 1])
        ):
            start -= 1
        if start == 0:
            break
        end = start
        while end > 0 and not lines[end - 1].strip():
            end -= 1
    return "\n".join(lines[:end]).strip()


def rewrite_prompt(message: str) -> str:
    """Returns the prompt that asks a rewriter to turn `message` into a report."""
    return REWRITE_PROMPT.format(message=strip_trailers(message))

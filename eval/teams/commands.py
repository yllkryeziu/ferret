# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Classifies the shell commands a Codex agent runs.

Codex runs each command as `/bin/zsh -lc SCRIPT`. A script is split into its
simple commands, and the script takes the first class in `CLASSES` order
that any of them has: `rg foo | head` is a search, `cat a.c | grep x` too.
"""

import shlex

SEARCH = "search"
LIST = "list"
READ = "read"
OTHER = "other"
CLASSES = (SEARCH, LIST, READ, OTHER)

_SEARCH_PROGRAMS = frozenset({"rg", "grep", "egrep", "fgrep", "ag", "ack", "git-grep"})
_LIST_PROGRAMS = frozenset({"find", "fd", "ls", "tree", "git-ls-files"})
_READ_PROGRAMS = frozenset({"cat", "sed", "head", "tail", "nl", "less", "more", "bat"})
# Words that run the next word as the program.
_PREFIXES = frozenset({"sudo", "time", "command", "env", "nice", "xargs", "timeout"})
_SEPARATORS = frozenset({"|", "||", "&&", ";", "&", "\n", "(", ")"})


def script_of(command: str) -> str:
    """Returns the script of `sh -c SCRIPT` or `sh -lc SCRIPT`, else `command`."""
    try:
        argv = shlex.split(command)
    except ValueError:
        return command
    shell = argv[0].rsplit("/", 1)[-1] if argv else ""
    if len(argv) == 3 and shell in {"sh", "bash", "zsh"} and argv[1] in {"-c", "-lc"}:
        return argv[2]
    return command


def programs(script: str) -> list[str]:
    """Returns the program of each simple command in `script`.

    `git grep` and `git ls-files` are reported as `git-grep` and
    `git-ls-files`. Leading `VAR=value` words and the `_PREFIXES` are
    skipped; `timeout` also skips its duration.
    """
    # "2>&1" is a redirection, not a background "&".
    text = script.replace(">&", "> ").replace("\n", " ; ")
    lexer = shlex.shlex(text, posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        tokens = text.split()
    result = []
    words: list[str] = []
    for token in [*tokens, ";"]:
        if token in _SEPARATORS or set(token) <= set(";&|()"):
            if words:
                result.append(_program(words))
                # `find ... -exec grep ...` runs grep on every file found.
                if result[-1] == "find":
                    result.extend(
                        _program(words[i + 1 :])
                        for i, word in enumerate(words)
                        if word in {"-exec", "-execdir"}
                    )
            words = []
        else:
            words.append(token)
    return [program for program in result if program]


def _program(words: list[str]) -> str:
    i = 0
    while i < len(words):
        word = words[i]
        if (
            "=" in word
            and not word.startswith("-")
            and word.split("=", 1)[0].isidentifier()
        ):
            i += 1
        elif word in _PREFIXES:
            i += 1
            if word == "timeout" and i < len(words):
                i += 1
            while i < len(words) and words[i].startswith("-"):
                i += 1
        else:
            break
    if i >= len(words):
        return ""
    name = words[i].rsplit("/", 1)[-1]
    if name == "git":
        rest = [w for w in words[i + 1 :] if not w.startswith("-")]
        if rest and rest[0] in {"grep", "ls-files"}:
            return f"git-{rest[0]}"
    return name


def classify(command: str) -> str:
    """Returns the class of a Codex `command_execution` command."""
    found = set()
    for program in programs(script_of(command)):
        if program in _SEARCH_PROGRAMS:
            found.add(SEARCH)
        elif program in _LIST_PROGRAMS:
            found.add(LIST)
        elif program in _READ_PROGRAMS:
            found.add(READ)
    return next((c for c in CLASSES if c in found), OTHER)

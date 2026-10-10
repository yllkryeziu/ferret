# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tasks.definitions: which top-level definitions a diff changes."""

from tasks.definitions import DefinitionSpan, changed_definitions, definition_spans
from tasks.diff import Hunk

_KERNEL_C = """\
// SPDX-License-Identifier: GPL-2.0
#include <linux/module.h>
#define TWICE(x) \\
	((x) * 2)

/*
 * parse_header() - reads a header
 */
static int
parse_header(const char *p)
{
	if (!p)
		goto out;
	return p[0];
out:
	return -EINVAL;
}
EXPORT_SYMBOL(parse_header);

static const struct item table[] = {
	{ 1, 2 },
};

struct device_state {
	int flags;
};

int helper(void) { return TWICE(1); }
"""


def test_finds_kernel_style_definitions():
    assert definition_spans(_KERNEL_C) == [
        DefinitionSpan("parse_header", 9, 17),
        DefinitionSpan("table", 20, 22),
        DefinitionSpan("device_state", 24, 26),
        DefinitionSpan("helper", 28, 28),
    ]


def test_finds_cpp_definitions():
    source = """\
namespace net {

template <typename T>
bool Parser<T>::Parse(const std::string& input) {
  return !input.empty();
}

Parser& Parser::operator=(const Parser& other) {
  return *this;
}

class Resolver final : public Base {
 public:
  void Run();
};

}  // namespace net
"""
    names = [span.name for span in definition_spans(source)]
    assert names == ["Parser::Parse", "Parser::operator=", "Resolver"]


def test_maps_changed_lines_to_definitions():
    # Line 14 ("return p[0];") is in parse_header; line 22 ends the table.
    assert changed_definitions(_KERNEL_C, [Hunk(14, 1, 14, 1)]) == {"parse_header"}
    assert changed_definitions(_KERNEL_C, [Hunk(14, 1, 14, 2), Hunk(22, 1, 23, 1)]) == {
        "parse_header",
        "table",
    }
    # A replaced EXPORT_SYMBOL line belongs to no definition.
    assert changed_definitions(_KERNEL_C, [Hunk(18, 1, 18, 1)]) == set()


def test_insertions_count_only_inside_a_definition():
    # Inserting after line 12 lands between two lines of parse_header.
    assert changed_definitions(_KERNEL_C, [Hunk(12, 0, 13, 1)]) == {"parse_header"}
    # Inserting after line 17, the closing brace, lands outside it.
    assert changed_definitions(_KERNEL_C, [Hunk(17, 0, 18, 5)]) == set()

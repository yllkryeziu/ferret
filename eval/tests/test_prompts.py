# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for harness.prompts."""

import pytest

from harness.prompts import Prompts


def test_loads_the_default_prompts():
    prompts = Prompts.load()
    assert "{{ task }}" in prompts.instance_template
    assert "{{ top_level_listing }}" in prompts.instance_template
    assert "submit" in prompts.forced_submit_message
    assert prompts.sha256() == Prompts.load().sha256()


@pytest.mark.parametrize(
    "text",
    [
        "system_template: a\ninstance_template: b\n",
        "system_template: a\ninstance_template: b\nforced_submit_message: c\nx: d\n",
        "system_template: a\ninstance_template: b\nforced_submit_message: 3\n",
        "- a\n",
    ],
)
def test_rejects_malformed_files(tmp_path, text):
    path = tmp_path / "prompts.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        Prompts.load(path)

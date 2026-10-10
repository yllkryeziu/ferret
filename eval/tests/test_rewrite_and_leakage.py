# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tasks.rewrite and tasks.leakage: prompts that do not give away
their answer."""

from tasks.leakage import PathComponentIndex, find_leaks, path_components
from tasks.rewrite import rewrite_prompt, strip_trailers

_MESSAGE = """\
ACPI: video: fix brightness keys on the X13

Pressing the brightness keys printed "acpi_video0: Failed to set
brightness" and nothing changed.

Fixes: 0123456789ab ("ACPI: video: add backlight quirks")
Reported-by: A User <user@example.com>
Link: https://lore.kernel.org/r/20260901
  continued-link-line
Signed-off-by: A Maintainer <maintainer@example.com>
"""


def test_strips_the_trailer_block():
    assert strip_trailers(_MESSAGE) == (
        "ACPI: video: fix brightness keys on the X13\n\n"
        'Pressing the brightness keys printed "acpi_video0: Failed to set\n'
        'brightness" and nothing changed.'
    )


def test_keeps_a_final_paragraph_that_is_not_all_trailers():
    message = "Fix a crash\n\nNote: this also happens on boot.\nIt is rare."
    assert strip_trailers(message) == message
    # A one-paragraph message is a subject, never a trailer block.
    assert strip_trailers("Fixes: the crash on boot") == "Fixes: the crash on boot"


def test_rewrite_prompt_holds_the_message_without_trailers():
    prompt = rewrite_prompt(_MESSAGE)
    assert "Failed to set" in prompt
    assert "Signed-off-by" not in prompt
    assert "0123456789ab" not in prompt


def test_path_components():
    assert path_components("drivers/acpi/Video_Detect.c") == {
        "drivers",
        "acpi",
        "video_detect.c",
        "video_detect",
    }


def _index(generic_count: int) -> PathComponentIndex:
    paths = [f"drivers/acpi/file{n}.c" for n in range(generic_count)]
    return PathComponentIndex([*paths, "drivers/acpi/video_detect.c"])


def test_specific_path_components_leak_as_whole_words():
    components = _index(150)
    gold = ["drivers/acpi/video_detect.c"]
    assert find_leaks("The VIDEO_DETECT logic misfires.", gold, [], components) == [
        "video_detect"
    ]
    # "acpi" and "drivers" name more than 100 files, so they are too generic.
    assert find_leaks("An ACPI driver misfires.", gold, [], components) == []
    assert find_leaks("my_video_detect_hook misfires", gold, [], components) == []


def test_function_names_leak_by_segment_and_case():
    components = _index(0)
    gold_paths = ["drivers/acpi/video_detect.c"]
    functions = ["drivers/acpi/video_detect.c::Backlight::Update"]
    assert find_leaks("Update is slow", gold_paths, functions, components) == ["Update"]
    assert find_leaks("an update is slow", gold_paths, functions, components) == []
    assert find_leaks("Backlight::Update hangs", gold_paths, functions, components) == [
        "Backlight",
        "Backlight::Update",
        "Update",
    ]


def test_user_known_product_is_not_a_gold_path_component():
    from tasks.rewrite import REWRITE_PROMPT

    paths = PathComponentIndex(["drivers/infiniband/hw/efa/driver.c"])
    gold = ["drivers/infiniband/hw/efa/driver.c"]
    assert "Elastic Fabric Adapter" in REWRITE_PROMPT
    assert (
        find_leaks(
            "Elastic Fabric Adapter crashes while removing the device.", gold, [], paths
        )
        == []
    )
    assert find_leaks("efa crashes while removing the device.", gold, [], paths) == [
        "efa"
    ]


def test_answerability_screen_off_by_default_and_scripted_judge():
    from tasks.commit_tasks import screen_answerability

    observations = []

    def judge(prompt):
        observations.append(prompt)
        return "YES" if "Elastic Fabric Adapter" in prompt else "NO"

    assert screen_answerability("Elastic Fabric Adapter fails to initialize", judge)
    assert not screen_answerability("The device crashes sometimes", judge)
    assert "efa" not in observations[0]
    with __import__("pytest").raises(ValueError, match="YES or NO"):
        screen_answerability("irrelevant", lambda _: "maybe")


def test_strips_gerrit_trailers_across_blank_lines():
    message = (
        "Fix crash when opening a window\n\nA window crashes on startup.\n\n\n"
        "Bug: 12345\nChange-Id: Iabc\nReviewed-on: https://example.test/123\n"
        "\nCr-Commit-Position: refs/heads/main@{#123}"
    )
    assert strip_trailers(message) == (
        "Fix crash when opening a window\n\nA window crashes on startup."
    )


def test_strips_final_continuation_lines():
    assert (
        strip_trailers(
            "Fix the crash\n\nThe window closes unexpectedly.\n\n"
            "Reviewed-on: https://example.test/1\n  continued"
        )
        == "Fix the crash\n\nThe window closes unexpectedly."
    )

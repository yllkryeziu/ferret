# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Q_mix generator on a small reproducible corpus."""

import collections
import json

import pytest

from bench.queries import gen_mix


@pytest.fixture
def corpus(tmp_path):
    """Build definition names in each of the three log-scale DF bands."""
    root = tmp_path / "corpus"
    root.mkdir()
    files = []
    # With eight documents, DF=1, 2, and 8 occupy distinct log bands.
    for index in range(8):
        path = f"file{index}.py"
        lines = []
        for n in range(12):
            lines += [
                f"def Rare{index}Alpha{n}():",
                f"    return 'value-{index}-{n}!' + 'other: word'",
            ]
            if index < 2:
                lines += [
                    f"def MiddleBeta{n}():",
                    "    return 'middle-value' + 'other: word'",
                ]
            if index == 0:
                lines += [
                    f"def CommonGamma{n}():",
                    "    return 'common-value' + 'other: word'",
                ]
            lines += [
                f"use = 'word {n} -> token {index}; item ! > next'",
                "if fd io nr ab: return fd + io",
            ]
        (root / path).write_text("\n".join(lines) + "\n", encoding="ascii")
        files.append(path)
    # References raise the middle names to DF=2 and common names to DF=8.
    for index in range(1, 8):
        path = root / files[index]
        references = [f"CommonGamma{n}" for n in range(12)]
        if index == 1:
            references += [f"MiddleBeta{n}" for n in range(12)]
        path.write_text(path.read_text() + "\n".join(references) + "\n")
    file_list = tmp_path / "corpus.files"
    file_list.write_text("\n".join(files) + "\n")
    return root, file_list, files


def test_mix_is_deterministic_and_has_requested_shares(corpus, tmp_path):
    root, file_list, files = corpus
    outputs = []
    for seed in (17, 17, 18):
        out = tmp_path / f"mix-{len(outputs)}.jsonl"
        assert (
            gen_mix.main(
                [
                    "--root",
                    str(root),
                    "--files",
                    str(file_list),
                    "--corpus",
                    "tiny",
                    "--out",
                    str(out),
                    "--count",
                    "50",
                    "--seed",
                    str(seed),
                ]
            )
            == 0
        )
        outputs.append(
            (
                out.read_bytes(),
                json.loads(out.with_name(out.stem + ".meta.json").read_text()),
            )
        )
    assert outputs[0] == outputs[1]
    assert outputs[0][0] != outputs[2][0]
    records = [json.loads(line) for line in outputs[0][0].splitlines()]
    assert len(records) == 50
    assert len({record["id"] for record in records}) == 50
    assert collections.Counter(r["class"] for r in records) == dict(
        zip(
            (cls for cls, _ in gen_mix.CLASS_SHARES),
            gen_mix.allocate(50, [share for _, share in gen_mix.CLASS_SHARES]),
            strict=True,
        )
    )
    contents = [(root / path).read_bytes() for path in files]
    for record in records:
        if record["class"] in {"identifier", "literal"}:
            assert any(record["pattern"].encode() in data for data in contents)
        elif record["class"] == "zero":
            assert all(record["pattern"].encode() not in data for data in contents)

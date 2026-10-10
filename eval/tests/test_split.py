# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tasks.split: the salted-hash dev/test split and sampling."""

from fractions import Fraction

from tasks.schema import Split
from tasks.split import assign_split, hash_order, salted_hash

_IDS = [f"locbench/task-{n}" for n in range(5000)]


def test_salted_hash_is_a_stable_64_bit_value():
    value = salted_hash("salt", "key")
    assert value == salted_hash("salt", "key")
    assert 0 <= value < 1 << 64
    assert value != salted_hash("other salt", "key")
    # The separator keeps ("ab", "c") and ("a", "bc") apart.
    assert salted_hash("ab", "c") != salted_hash("a", "bc")


def test_assigns_about_a_fifth_to_dev():
    dev = sum(assign_split(task_id) is Split.DEV for task_id in _IDS)
    # Binomial(5000, 1/5) has mean 1000 and standard deviation about 28.
    assert 850 < dev < 1150


def test_split_depends_only_on_the_id_and_salt():
    assert [assign_split(task_id) for task_id in _IDS[:100]] == [
        assign_split(task_id) for task_id in _IDS[:100]
    ]
    assert [assign_split(task_id) for task_id in _IDS] != [
        assign_split(task_id, salt="another salt") for task_id in _IDS
    ]


def test_extreme_fractions():
    assert all(
        assign_split(task_id, dev_fraction=Fraction(0)) is Split.TEST
        for task_id in _IDS[:200]
    )
    assert all(
        assign_split(task_id, dev_fraction=Fraction(1)) is Split.DEV
        for task_id in _IDS[:200]
    )


def test_hash_order_is_a_salted_permutation():
    order = hash_order(_IDS, key=str, salt="s")
    assert sorted(order) == sorted(_IDS)
    assert order == hash_order(reversed(_IDS), key=str, salt="s")
    assert order != _IDS
    assert order != hash_order(_IDS, key=str, salt="t")


def test_adding_items_keeps_the_relative_order_of_a_sample():
    before = hash_order(_IDS[:1000], key=str, salt="s")
    after = hash_order(_IDS, key=str, salt="s")
    sampled = set(before)
    assert [task_id for task_id in after if task_id in sampled] == before

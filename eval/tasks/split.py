# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic dev/test split and sampling by salted hash of the task id.

A task's split depends only on its own id and the salt, never on which other
tasks exist, so adding tasks never moves an existing task between dev and
test. Per bucket the dev share is Binomial(n, 1/5) rather than exactly 20%:
an exact quota would need a rank cut-off, and a rank cut-off moves when tasks
are added. The salt is fixed before any test run, so it cannot be chosen
after seeing results.
"""

import hashlib
from collections.abc import Callable, Iterable
from fractions import Fraction

from tasks.schema import Split

SPLIT_SALT = "ferret-eval-split-v1"
DEV_FRACTION = Fraction(1, 5)

_HASH_BITS = 64


def salted_hash(salt: str, key: str) -> int:
    """Returns a uniform 64-bit integer derived from SHA-256(salt, key)."""
    digest = hashlib.sha256(f"{salt}\0{key}".encode()).digest()
    return int.from_bytes(digest[: _HASH_BITS // 8], "big")


def assign_split(
    task_id: str, *, salt: str = SPLIT_SALT, dev_fraction: Fraction = DEV_FRACTION
) -> Split:
    """Returns DEV for a `dev_fraction` share of ids and TEST for the rest.

    The comparison is exact integer arithmetic, so the threshold has no
    floating-point rounding.
    """
    scaled = salted_hash(salt, task_id) * dev_fraction.denominator
    return Split.DEV if scaled < dev_fraction.numerator << _HASH_BITS else Split.TEST


def hash_order[T](items: Iterable[T], key: Callable[[T], str], salt: str) -> list[T]:
    """Returns `items` sorted by the salted hash of `key(item)`.

    This is a uniformly random permutation that is fixed by the salt, so the
    first n items are a reproducible simple random sample of size n. Ties
    between colliding hashes fall back to the key itself.
    """
    return sorted(items, key=lambda item: (salted_hash(salt, key(item)), key(item)))

import random

import pytest

from rt_model_inference.iterators import (
    first_and_last_nonoutlier,
    first_and_last_outlier,
    hampel_identifier_nonwindowed,
    obatched,
)


def test_obatched_yields_overlapping_batches_and_trailing_partial() -> None:
    batches = list(obatched(range(6), batch_size=3, overlap=1))

    assert batches == [(0, 1, 2), (2, 3, 4), (4, 5)]


def test_obatched_yields_non_overlapping_batches_when_overlap_is_zero() -> None:
    batches = list(obatched(range(6), batch_size=3, overlap=0))

    assert batches == [(0, 1, 2), (3, 4, 5)]


def test_obatched_rejects_infeasible_overlap() -> None:
    with pytest.raises(ValueError, match="infeasible batch size"):
        _ = list(obatched(range(4), batch_size=3, overlap=3))

    with pytest.raises(ValueError, match="infeasible batch size"):
        _ = list(obatched(range(4), batch_size=3, overlap=-1))


def test_obatched_does_not_emit_overlap_only_tail() -> None:
    batches = list(obatched(range(5), batch_size=3, overlap=1))

    assert batches == [(0, 1, 2), (2, 3, 4)]


def _mk_outlier_data(
    seed: str | int, length: int = 1000
) -> tuple[list[int], list[bool]]:
    rng = random.Random(seed)
    data = [rng.randint(0, 20) for _ in range(length)]
    expected = length * [False]
    i = rng.randint(1, 10)
    while i < length:
        data[i] = rng.randint(40, 50)
        expected[i] = True
        i += rng.randint(1, 10)
    return (data, expected)


def test_hampel_identifies_obvious_outliers() -> None:
    data, expected = _mk_outlier_data("Hampel")
    assert list(hampel_identifier_nonwindowed(data)) == expected


def test_first_and_last_nonoutlier() -> None:
    data, expected = _mk_outlier_data("Hampel")
    first, last = first_and_last_nonoutlier(hampel_identifier_nonwindowed(data))
    assert first is not None
    assert last is not None

    assert expected[first] is False
    assert expected[last] is False
    assert first <= last

    assert expected[0:first] == first * [True]
    assert expected[last + 1 :] == (len(expected) - 1 - last) * [True]


def test_first_and_last_outlier() -> None:
    data, expected = _mk_outlier_data("Hampel")
    first, last = first_and_last_outlier(hampel_identifier_nonwindowed(data))
    assert first is not None
    assert last is not None

    assert expected[first] is True
    assert expected[last] is True
    assert first <= last

    assert expected[0:first] == first * [False]
    assert expected[last + 1 :] == (len(expected) - 1 - last) * [False]

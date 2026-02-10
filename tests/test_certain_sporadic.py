import random
from bisect import bisect_left, bisect_right
from itertools import batched

import pytest

from rt_model_inference.certain_sporadic import (
    infer_delta_max,
    infer_delta_min,
    infer_sporadic_model,
    max_releases,
    min_releases,
)
from rt_model_inference.extractors import (
    DeltaMaxExtractor,
    DeltaMinExtractor,
    SporadicExtractor,
)
from rt_model_inference.time import EPSILON, Instant
from rt_model_inference.validate import (
    dmax_curve_is_tight,
    dmax_curve_lower_bounds_releases,
    dmin_curve_is_tight,
    dmin_curve_upper_bounds_releases,
    find_closed_interval_containing,
    max_interval_containing,
)
from rt_model_inference.validate import (
    test_intervals as considered_intervals,
)


def test_min_releases() -> None:
    dmax = [9, 19, 29, 39, 49, 59]

    assert min_releases(dmax, 5) == 0
    assert min_releases(dmax, 9) == 0
    assert min_releases(dmax, 10) == 1
    assert min_releases(dmax, 11) == 1
    assert min_releases(dmax, 19) == 1
    assert min_releases(dmax, 20) == 2
    assert min_releases(dmax, 21) == 2
    assert min_releases(dmax, 29) == 2
    assert min_releases(dmax, 30) == 3


def test_min_releases_returns_none_for_empty_curve() -> None:
    assert min_releases([], 5) is None


def test_max_releases() -> None:
    dmin = [0, 1, 11, 21, 31, 41, 51, 52]

    assert max_releases(dmin, 0) == 0
    assert max_releases(dmin, 1) == 1
    assert max_releases(dmin, 3) == 1
    assert max_releases(dmin, 9) == 1
    assert max_releases(dmin, 10) == 1
    assert max_releases(dmin, 11) == 2
    assert max_releases(dmin, 19) == 2
    assert max_releases(dmin, 20) == 2
    assert max_releases(dmin, 21) == 3
    assert max_releases(dmin, 29) == 3
    assert max_releases(dmin, 30) == 3
    assert max_releases(dmin, 31) == 4


def test_max_releases_returns_none_for_empty_curve() -> None:
    assert max_releases([], 5) is None


def test_periodic_dmin() -> None:
    nmax = 5
    period = 50
    releases = range(123, 1000, period)

    dmin = infer_delta_min(releases, nmax=nmax)

    assert len(dmin) == nmax + 1
    assert dmin == [0] + list(range(1, 1 + nmax * period, period))
    assert dmin_curve_is_tight(dmin, releases)


def test_irregular_bursty_and_sparse_releases() -> None:
    nmax = 4
    releases = [0, 2, 4, 100, 101, 200]

    dmin = infer_delta_min(releases, nmax=nmax)

    # Shortest intervals for 2 and 3 releases come from the early burst,
    # while 4 releases require spanning into the sparse region.
    assert dmin == [0, 1, 2, 5, 100]
    assert dmin_curve_is_tight(dmin, releases)


def test_irregular_with_duplicate_timestamps() -> None:
    nmax = 3
    releases = [5, 5, 6, 10, 10, 10]

    dmin = infer_delta_min(releases, nmax=nmax)

    # Duplicate timestamps allow 2 or 3 releases within the minimum interval.
    assert dmin == [0, 1, 1, 1]
    assert dmin_curve_is_tight(dmin, releases)


def test_empty_release_sequence() -> None:
    dmin = infer_delta_min([])

    assert dmin == []


def test_non_monotonic_releases_raise() -> None:
    releases = [0, 5, 3, 6]

    with pytest.raises(ValueError, match="releases must be monotonic"):
        _ = infer_delta_min(releases)


def test_nmax_greater_than_release_count() -> None:
    nmax = 10
    releases = [0, 3, 7]

    dmin = infer_delta_min(releases, nmax=nmax)

    assert dmin == [0, 1, 4, 8]
    assert dmin_curve_is_tight(dmin, releases)


def test_infer_delta_min_rejects_nmax_below_two() -> None:
    with pytest.raises(ValueError, match="nmax must be at least 2"):
        _ = infer_delta_min([], nmax=1)


def test_infer_sporadic_model_extracts_minimum_separation() -> None:
    releases = [5, 8, 8, 14, 25]

    min_separation = infer_sporadic_model(releases)

    assert min_separation == 0


def test_infer_sporadic_model_matches_delta_min_two_release_interval() -> None:
    releases = [0, 2, 4, 100, 101, 200]

    min_separation = infer_sporadic_model(releases)
    dmin = infer_delta_min(releases, nmax=2)

    assert min_separation == dmin[2] - EPSILON


def test_infer_sporadic_model_rejects_single_observation() -> None:
    with pytest.raises(ValueError, match="at least two"):
        _ = infer_sporadic_model([7])


def test_periodic_dmax() -> None:
    nmax = 5
    period = 50
    releases = list(range(123, 1000, period))

    dmax = infer_delta_max(releases, nmax=nmax)

    assert len(dmax) == nmax + 1
    assert dmax == [49, 99, 149, 199, 249, 299]
    assert dmax_curve_is_tight(dmax, releases)


def test_irregular_bursty_and_sparse_releases_dmax() -> None:
    nmax = 4
    releases = [0, 2, 4, 100, 101, 200]

    dmax = infer_delta_max(releases, nmax=nmax)

    # Largest intervals for each exact release count span sparse regions.
    assert dmax == [98, 99, 195, 197, 199]
    assert dmax_curve_is_tight(dmax, releases)


def test_irregular_with_duplicate_timestamps_dmax() -> None:
    nmax = 3
    releases = [5, 5, 6, 10, 10, 10]

    dmax = infer_delta_max(releases, nmax=nmax)

    # No gaps observed other than (6, 10)
    assert dmax == [3, 4, 4, 5]
    assert dmax_curve_is_tight(dmax, releases)


def test_empty_release_sequence_dmax() -> None:
    dmax = infer_delta_max([])

    assert dmax == []


def test_non_monotonic_releases_raise_dmax() -> None:
    releases = [0, 5, 3, 6]

    with pytest.raises(ValueError, match="releases must be monotonic"):
        _ = infer_delta_max(releases)


def test_nmax_greater_than_release_count_dmax() -> None:
    nmax = 10
    releases = [0, 3, 7]

    dmax = infer_delta_max(releases, nmax=nmax)

    assert dmax == [3, 6, 7, 8]
    assert dmax_curve_is_tight(dmax, releases)


def test_infer_delta_max_rejects_nonpositive_nmax() -> None:
    with pytest.raises(ValueError, match="nmax must be positive"):
        _ = infer_delta_max([], nmax=0)


def generate_random_release_sequence(
    length: int,
    seed: int | str,
    maxgap: int = 10,
    skip: int = 0,
) -> list[int]:
    rng = random.Random(seed)
    current = 0
    releases: list[int] = []
    # discard `skip` samples
    _ = [rng.randint(0, maxgap) for _ in range(skip)]
    # use the next `length` samples
    for _ in range(length):
        # Permit zero-step increments so duplicates are represented.
        current += rng.randint(0, maxgap)
        releases.append(current)
    return releases


RELEASES: list[list[Instant]] = [
    list(range(123, 1000, 50)),
    [0, 2, 4, 100, 101, 200],
    [5, 5, 6, 10, 10, 10],
    [0, 3, 7],
    [21, 260, 260, 260],
    [92, 94, 94, 219, 229],
    [28, 35, 35, 249, 258],
    [7, 9, 9, 9, 222, 225, 225],
    [279, 3828, 3829, 3829, 3876],
]
SEEDS = [
    "Anderson",
    "Baruah",
    "Jeffay",
    "Baker",
    "Rajkumar",
    "Burns",
    "Davis",
    "Buttazzo",
    "Di Natale",
    "Lipari",
]

for seed in SEEDS:
    for offset in range(0, 1000, 113):
        releases = generate_random_release_sequence(
            59, seed=f"{seed}:{offset}", maxgap=(10 + offset // 100)
        )
        RELEASES.append(releases)
RELEASES.sort()

for a, b in zip(RELEASES, RELEASES[1:]):
    if a == b:
        assert False, f"duplicate: {a}"


@pytest.mark.parametrize(
    ("releases", "nmax"),
    [
        *[(releases, None) for releases in RELEASES],
        *[(releases, 10) for releases in RELEASES],
    ],
)
def test_inferred_curves_are_correct_for_all_release_pairs(
    releases: list[int], nmax: int | None
) -> None:
    dmin = infer_delta_min(releases, nmax=nmax)
    dmax = infer_delta_max(releases, nmax=nmax)

    assert dmin_curve_upper_bounds_releases(dmin, releases)
    assert dmin_curve_is_tight(dmin, releases)
    assert dmax_curve_lower_bounds_releases(dmax, releases)
    assert dmax_curve_is_tight(dmax, releases)


def test_dmin_curve_validation_rejects_too_many_releases() -> None:
    releases = list(range(10, 1001, 10))
    dmin = infer_delta_min(releases, nmax=5)

    for i in range(len(releases)):
        one_more = list(releases)
        one_more.insert(i, releases[i] - 1)
        assert not dmin_curve_upper_bounds_releases(dmin, one_more)

    one_more = list(releases)
    one_more.append(releases[-1] + 1)
    assert not dmin_curve_upper_bounds_releases(dmin, one_more)


def test_dmax_curve_validation_rejects_too_few_releases() -> None:
    releases = list(range(10, 1001, 10))
    dmax = infer_delta_max(releases, nmax=5)

    for i in range(1, len(releases) - 1):
        one_fewer = list(releases)
        del one_fewer[i]
        assert not dmax_curve_lower_bounds_releases(dmax, one_fewer)


@pytest.mark.parametrize("releases", RELEASES)
def test_considered_intervals(releases: list[int]) -> None:
    for (x, y), claimed_count in considered_intervals(
        releases, releases[-1] - releases[0], 11
    ):
        actual_count = bisect_right(releases, y) - bisect_left(releases, x)
        assert claimed_count == actual_count


def test_dmin_curve_tightness_rejects_loose_sequence() -> None:
    releases = list(range(10, 1001, 10))
    dmin = infer_delta_min(releases, nmax=10)

    assert find_closed_interval_containing(releases, 11, 2) == (10, 20)
    assert find_closed_interval_containing(releases, 21, 3) == (10, 30)
    assert find_closed_interval_containing(releases, 31, 4) == (10, 40)

    assert find_closed_interval_containing(releases, 10, 2) is None
    assert find_closed_interval_containing(releases, 20, 3) is None
    assert find_closed_interval_containing(releases, 30, 4) is None

    assert dmin_curve_is_tight(dmin, releases)

    for i in range(len(dmin)):
        dmin_loose = list(dmin)
        dmin_loose[i] -= 1
        assert not dmin_curve_is_tight(dmin_loose, releases)


def test_dmax_curve_tightness_rejects_loose_sequence() -> None:
    releases = list(range(10, 1001, 10))
    dmax = infer_delta_max(releases, nmax=10)

    assert max_interval_containing(releases, 0)[0] == 9
    assert max_interval_containing(releases, 1)[0] == 19
    assert max_interval_containing(releases, 2)[0] == 29
    assert max_interval_containing(releases, 3)[0] == 39
    assert max_interval_containing(releases, 4)[0] == 49

    assert dmax_curve_is_tight(dmax, releases)

    for i in range(len(dmax)):
        dmax_loose = list(dmax)
        dmax_loose[i] += 1
        assert not dmax_curve_is_tight(dmax_loose, releases)


@pytest.mark.parametrize("releases", RELEASES)
@pytest.mark.parametrize("num_chunks", [2, 5])
def test_sporadic_extractor_matches_one_shot_inference(
    releases: list[Instant], num_chunks: int
) -> None:
    extractor = SporadicExtractor()
    observed: list[Instant] = []
    chunk_size = max(1, len(releases) // num_chunks)

    for batch in batched(releases, chunk_size):
        extractor.feed(batch)
        observed.extend(batch)

        expected = None if len(observed) < 2 else infer_sporadic_model(observed)
        assert extractor.current_model == expected


@pytest.mark.parametrize(
    ("releases", "nmax"),
    [
        *[(releases, 3) for releases in RELEASES],
        *[(releases, 10) for releases in RELEASES],
    ],
)
@pytest.mark.parametrize("num_chunks", [2, 5])
def test_delta_min_extractor_matches_one_shot_inference(
    releases: list[Instant], nmax: int | None, num_chunks: int
) -> None:
    extractor = DeltaMinExtractor(nmax=nmax)
    observed: list[Instant] = []
    chunk_size = max(1, len(releases) // num_chunks)

    for batch in batched(releases, chunk_size):
        extractor.feed(batch)
        observed.extend(batch)

        assert dmin_curve_upper_bounds_releases(extractor.current_model, observed)
        assert extractor.current_model == infer_delta_min(observed, nmax=nmax)


def test_delta_min_extractor_rejects_nmax_below_two() -> None:
    with pytest.raises(ValueError, match="nmax must be at least 2"):
        _ = DeltaMinExtractor(nmax=1)


@pytest.mark.parametrize(
    ("releases", "nmax"),
    [
        *[(releases, 3) for releases in RELEASES],
        *[(releases, 10) for releases in RELEASES],
    ],
)
@pytest.mark.parametrize("num_chunks", [2, 5])
def test_delta_max_extractor_matches_one_shot_inference(
    releases: list[Instant], nmax: int | None, num_chunks: int
) -> None:
    extractor = DeltaMaxExtractor(nmax=nmax)
    observed: list[Instant] = []
    chunk_size = max(1, len(releases) // num_chunks)

    for batch in batched(releases, chunk_size):
        extractor.feed(batch)
        observed.extend(batch)

        assert dmax_curve_lower_bounds_releases(extractor.current_model, observed)
        assert extractor.current_model == infer_delta_max(observed, nmax=nmax)


def test_delta_max_extractor_rejects_nonpositive_nmax() -> None:
    with pytest.raises(ValueError, match="nmax must be positive"):
        _ = DeltaMaxExtractor(nmax=0)

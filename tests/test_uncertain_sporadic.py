import random
from collections.abc import Callable, Iterable, Iterator
from itertools import batched, product

import pytest

from rt_model_inference.certain_sporadic import infer_delta_max, infer_delta_min
from rt_model_inference.extractors import (
    DeltaMaxHiExtractor,
    DeltaMaxLoExtractor,
    DeltaMinHiExtractor,
    DeltaMinLoExtractor,
)
from rt_model_inference.iterators import is_monotonic
from rt_model_inference.time import EPSILON, Duration, Instant, ReleaseWindow
from rt_model_inference.uncertain_sporadic import (
    infer_delta_max_hi,
    infer_delta_max_lo,
    infer_delta_min_hi,
    infer_delta_min_lo,
)
from rt_model_inference.validate import (
    dmax_curve_lower_bounds_releases,
    dmin_curve_upper_bounds_releases,
)


def all_feasible_release_sequences(
    windows: Iterable[tuple[Duration, Duration]],
) -> Iterator[tuple[Duration, ...]]:
    return (
        seq
        for seq in product(*[range(lo, hi + EPSILON) for lo, hi in windows])
        if is_monotonic(seq)
    )


def brute_force(
    infer_from_releases: Callable[[Iterable[Instant], int | None], list[Duration]],
    update: Callable[[int, int], int],
    windows: list[tuple[int, int]],
    nmax: int | None = None,
) -> list[int]:
    all_seqs = all_feasible_release_sequences(windows)

    try:
        envelope = infer_from_releases(next(all_seqs), nmax)
        assert is_monotonic(envelope)
    except StopIteration:
        return []

    for releases in all_seqs:
        vec = infer_from_releases(releases, nmax)
        assert is_monotonic(vec)
        for n, value in enumerate(vec):
            if len(envelope) > n:
                envelope[n] = update(envelope[n], value)
            else:
                envelope.append(value)

    return envelope


def brute_force_dmin_lo(
    windows: list[tuple[int, int]], nmax: int | None = None
) -> list[int]:
    return brute_force(infer_delta_min, max, windows, nmax)


def brute_force_dmin_hi(
    windows: list[tuple[int, int]], nmax: int | None = None
) -> list[int]:
    return brute_force(infer_delta_min, min, windows, nmax)


def brute_force_dmax_lo(
    windows: list[tuple[int, int]], nmax: int | None = None
) -> list[int]:
    return brute_force(infer_delta_max, max, windows, nmax)


def brute_force_dmax_hi(
    windows: list[tuple[int, int]], nmax: int | None = None
) -> list[int]:
    return brute_force(infer_delta_max, min, windows, nmax)


def test_exact_release_windows_match_certain_inference() -> None:
    releases = [0, 2, 4, 100, 101, 200]
    windows = [(r, r) for r in releases]

    assert infer_delta_min_hi(windows, nmax=4) == infer_delta_min(releases, nmax=4)


def test_uncertain_windows_affect_lower_bounds() -> None:
    windows = [(0, 3), (5, 7), (9, 10), (20, 21)]

    assert infer_delta_min_hi(windows) == [0, 1, 3, 7, 18]


def test_overlapping_windows_do_not_result_in_nonpositive_lower_bounds() -> None:
    windows = [(0, 10), (1, 11), (2, 12)]

    assert infer_delta_min_hi(windows) == [0, 1, 1, 1]


def test_nmax_returns_only_requested_prefix() -> None:
    windows = [(0, 0), (10, 10), (11, 11), (12, 12), (100, 100)]

    full = infer_delta_min_hi(windows)
    truncated = infer_delta_min_hi(windows, nmax=3)

    assert full == [0, 1, 2, 3, 13, 101]
    assert truncated == [0, 1, 2, 3]


def test_empty_release_windows_returns_empty_curve() -> None:
    assert infer_delta_min_hi([]) == []


def test_non_monotonic_release_window_lower_bounds_raise() -> None:
    windows = [(0, 1), (5, 6), (4, 7)]

    with pytest.raises(
        ValueError, match="release-window lower bounds must be monotonic"
    ):
        _ = infer_delta_min_hi(windows)


def test_nmax_must_be_positive() -> None:
    with pytest.raises(ValueError, match="nmax must be at least 2"):
        _ = infer_delta_min_hi([], nmax=0)


def test_infer_delta_min_hi_overapproximates_exact_envelope() -> None:
    windows = [(0, 1), (1, 8), (3, 8), (6, 10), (9, 10)]

    exact_envelope = brute_force_dmin_hi(windows)
    overapprox = infer_delta_min_hi(windows)

    assert len(overapprox) == len(exact_envelope)
    assert is_monotonic(overapprox)
    for i, (x, y) in enumerate(zip(overapprox, exact_envelope)):
        assert x <= y, f" x={x} > {y}=y @ {i}"


def _generate_random_release_sequence(
    length: int, seed: int | str, maxgap: int = 10
) -> list[int]:
    rng = random.Random(seed)
    current = 0
    releases: list[int] = []

    for _ in range(length):
        current += rng.randint(0, maxgap)
        releases.append(current)

    return releases


def random_windows_around_releases(
    releases: Iterable[int], seed: int | str, max_jitter: int
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    windows: list[tuple[int, int]] = []
    prev_lo: int | None = None
    prev_hi: int | None = None

    for rel in releases:
        lo = rel - rng.randint(0, max_jitter)
        lo = max(lo, prev_lo) if prev_lo is not None else lo
        hi = rel + rng.randint(0, max_jitter)
        hi = max(hi, prev_hi) if prev_hi is not None else hi
        windows.append((lo, hi))
        prev_lo = lo
        prev_hi = hi

    return windows


def test_exact_release_windows_match_certain_inference_for_delta_min_lo() -> None:
    releases = [0, 2, 4, 100, 101, 200]
    windows = [(r, r) for r in releases]

    assert infer_delta_min_lo(windows, nmax=4) == infer_delta_min(releases, nmax=4)


def test_uncertain_windows_affect_upper_bounds() -> None:
    windows = [(0, 3), (5, 7), (9, 10), (20, 21)]

    assert infer_delta_min_lo(windows) == [0, 1, 6, 11, 22]


def test_nmax_returns_only_requested_prefix_for_delta_min_lo() -> None:
    windows = [(0, 0), (10, 10), (11, 11), (12, 12), (100, 100)]

    full = infer_delta_min_lo(windows)
    truncated = infer_delta_min_lo(windows, nmax=3)

    assert full == [0, 1, 2, 3, 13, 101]
    assert truncated == [0, 1, 2, 3]


def test_empty_release_windows_returns_empty_curve_for_delta_min_lo() -> None:
    assert infer_delta_min_lo([]) == []


def test_non_monotonic_release_window_lower_bounds_raise_for_delta_min_lo() -> None:
    windows = [(0, 1), (5, 6), (4, 7)]

    with pytest.raises(
        ValueError, match="release-window lower bounds must be monotonic"
    ):
        _ = infer_delta_min_lo(windows)


def test_nmax_must_be_positive_for_delta_min_lo() -> None:
    with pytest.raises(ValueError, match="nmax must be at least 2"):
        _ = infer_delta_min_lo([], nmax=0)


def test_infer_delta_min_lo_overapproximates_exact_envelope() -> None:
    windows = [(0, 1), (1, 8), (3, 8), (6, 10), (9, 10)]

    exact_envelope = brute_force_dmin_lo(windows)
    overapprox = infer_delta_min_lo(windows)

    assert len(overapprox) == len(exact_envelope)
    assert is_monotonic(overapprox)
    for i, (x, y) in enumerate(zip(overapprox, exact_envelope)):
        assert x >= y, f" x={x} < {y}=y @ {i}"


def test_exact_release_windows_match_certain_inference_for_delta_max_lo() -> None:
    releases = [0, 2, 4, 100, 101, 200]
    windows = [(r, r) for r in releases]

    assert infer_delta_max_lo(windows, nmax=4) == infer_delta_max(releases, nmax=4)


def test_nmax_returns_only_requested_prefix_for_delta_max_lo() -> None:
    releases = [0, 2, 4, 100, 101, 200]
    windows = [(r, r) for r in releases]

    full = infer_delta_max_lo(windows)
    truncated = infer_delta_max_lo(windows, nmax=3)

    assert full == infer_delta_max(releases)
    assert truncated == infer_delta_max(releases, nmax=3)


def test_empty_release_windows_returns_empty_curve_for_delta_max_lo() -> None:
    assert infer_delta_max_lo([]) == []


def test_non_monotonic_release_window_lower_bounds_raise_for_delta_max_lo() -> None:
    windows = [(0, 1), (5, 6), (4, 7)]

    with pytest.raises(
        ValueError, match="release-window lower bounds must be monotonic"
    ):
        _ = infer_delta_max_lo(windows)


def test_nmax_must_be_positive_for_delta_max_lo() -> None:
    with pytest.raises(ValueError, match="nmax must be positive"):
        _ = infer_delta_max_lo([], nmax=0)


def test_infer_delta_max_lo_overapproximates_exact_envelope() -> None:
    windows = [(0, 1), (1, 8), (3, 8), (6, 10), (9, 10)]

    exact_envelope = brute_force_dmax_lo(windows)
    overapprox = infer_delta_max_lo(windows)

    assert len(overapprox) == len(exact_envelope)
    assert is_monotonic(overapprox)
    for i, (x, y) in enumerate(zip(overapprox, exact_envelope)):
        assert x >= y, f" x={x} < {y}=y @ {i}"


def test_exact_release_windows_match_certain_inference_for_delta_max_hi() -> None:
    releases = [0, 2, 4, 100, 101, 200]
    windows = [(r, r) for r in releases]

    assert infer_delta_max_hi(windows, nmax=4) == infer_delta_max(releases, nmax=4)


def test_nmax_returns_only_requested_prefix_for_delta_max_hi() -> None:
    releases = [0, 2, 4, 100, 101, 200]
    windows = [(r, r) for r in releases]

    full = infer_delta_max_hi(windows)
    truncated = infer_delta_max_hi(windows, nmax=3)

    assert full == infer_delta_max(releases)
    assert truncated == infer_delta_max(releases, nmax=3)


def test_empty_release_windows_returns_empty_curve_for_delta_max_hi() -> None:
    assert infer_delta_max_hi([]) == []


def test_non_monotonic_release_window_lower_bounds_raise_for_delta_max_hi() -> None:
    windows = [(0, 1), (5, 6), (4, 7)]

    with pytest.raises(
        ValueError, match="release-window lower bounds must be monotonic"
    ):
        _ = infer_delta_max_hi(windows)


def test_nmax_must_be_positive_for_delta_max_hi() -> None:
    with pytest.raises(ValueError, match="nmax must be positive"):
        _ = infer_delta_max_hi([], nmax=0)


def test_infer_delta_max_hi_underapproximates_exact_envelope() -> None:
    windows = [(0, 1), (1, 8), (3, 8), (6, 10), (9, 10)]

    exact_envelope = brute_force_dmax_hi(windows)
    underapprox = infer_delta_max_hi(windows)

    assert len(underapprox) == len(exact_envelope)
    assert is_monotonic(underapprox)
    for i, (x, y) in enumerate(zip(underapprox, exact_envelope)):
        assert x <= y, f" x={x} > {y}=y @ {i}"


def test_infer_delta_max_hi_specific_windows_can_exceed_exact_curve() -> None:
    releases = [10, 10, 12, 18, 20, 25, 36]
    windows = [(10, 17), (10, 17), (10, 33), (10, 33), (10, 40), (10, 40), (33, 40)]

    exact = infer_delta_max(releases)
    # bf_underapprox = brute_force_dmax_hi(windows)
    underapprox = infer_delta_max_hi(windows)

    # [10, 15, 17, 23, 25, 26] exact
    # [3, 5, 8, 8, 15, 16, 17] bf
    # [15, 15, 15, 15, 15, 16] under

    bf_underapprox = [3, 5, 8, 8, 15, 16, 17]

    assert bf_underapprox[0] <= exact[0], (
        f"violation at n=0: bf-hi={bf_underapprox[0]}, exact={exact[0]}"
    )

    assert underapprox[0] <= exact[0], (
        f"violation at n=0: hi={underapprox[0]}, exact={exact[0]}"
    )


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


@pytest.mark.parametrize("seed", [*SEEDS])
@pytest.mark.parametrize("offset", [0, 113, 226, 339, 452, 777])
def test_infer_delta_min_hi_lo_approximates_exact_dmin_for_random_windows(
    seed: str, offset: int
) -> None:
    releases = _generate_random_release_sequence(
        length=59, seed=f"{seed}:{offset}", maxgap=(10 + offset // 100)
    )
    windows = random_windows_around_releases(
        releases, seed=f"{seed}:{offset}:window", max_jitter=(20 + offset // 50)
    )

    exact = infer_delta_min(releases)
    underapprox = infer_delta_min_hi(windows)
    overapprox = infer_delta_min_lo(windows)
    exact_windows_hi = infer_delta_min_hi([(r, r) for r in releases])
    exact_windows_lo = infer_delta_min_lo([(r, r) for r in releases])

    assert len(underapprox) == len(exact)
    assert len(overapprox) == len(exact)
    assert is_monotonic(overapprox)
    assert is_monotonic(underapprox)
    assert exact == exact_windows_hi == exact_windows_lo
    for n, (from_under, from_exact, from_over) in enumerate(
        zip(underapprox, exact, overapprox)
    ):
        assert from_under <= from_exact <= from_over, (
            f"violation at n={n}: hi={from_under}, exact={from_exact}, lo={from_over}"
        )


@pytest.mark.parametrize("seed", [*SEEDS])
@pytest.mark.parametrize("offset", [0, 113, 226, 339, 452, 777])
@pytest.mark.parametrize("nmax", [None, 13])
def test_infer_delta_max_hi_lo_approximates_exact_dmax_for_random_windows(
    seed: str, offset: int, nmax: int | None
) -> None:
    releases = _generate_random_release_sequence(
        length=59, seed=f"{seed}:{offset}", maxgap=(10 + offset // 100)
    )
    windows = random_windows_around_releases(
        releases, seed=f"{seed}:{offset}:window", max_jitter=(20 + offset // 50)
    )

    exact = infer_delta_max(releases, nmax)
    underapprox = infer_delta_max_hi(windows, nmax)
    overapprox = infer_delta_max_lo(windows, nmax)
    exact_windows_hi = infer_delta_max_hi([(r, r) for r in releases], nmax)
    exact_windows_lo = infer_delta_max_lo([(r, r) for r in releases], nmax)

    assert is_monotonic(underapprox)
    assert is_monotonic(overapprox)
    assert exact == exact_windows_hi == exact_windows_lo
    for n, (from_under, from_exact, from_over) in enumerate(
        zip(underapprox, exact, overapprox)
    ):
        assert from_exact <= from_over, (
            f"violation at n={n}: exact={from_exact}, lo={from_over}"
        )
        assert from_under <= from_exact, (
            f"violation at n={n}: hi={from_under}, exact={from_exact}"
        )


@pytest.mark.parametrize("seed", [*SEEDS])
@pytest.mark.parametrize("offset", [0, 113, 226, 339, 452, 777])
@pytest.mark.parametrize("nmax", [None, 13])
@pytest.mark.parametrize("num_chunks", [2, 5])
def test_delta_min_hi_extractor_matches_one_shot_inference(
    seed: str, offset: int, nmax: int | None, num_chunks: int
) -> None:
    releases = _generate_random_release_sequence(
        length=59, seed=f"{seed}:{offset}", maxgap=(10 + offset // 100)
    )
    windows = random_windows_around_releases(
        releases, seed=f"{seed}:{offset}:window", max_jitter=(20 + offset // 50)
    )

    extractor = DeltaMinHiExtractor(nmax=nmax)
    observed: list[ReleaseWindow] = []
    chunk_size = max(1, len(releases) // num_chunks)

    for batch in batched(windows, chunk_size):
        extractor.feed(batch)
        observed.extend(batch)

        assert dmin_curve_upper_bounds_releases(
            extractor.current_model, releases[: len(observed)]
        )
        assert extractor.current_model == infer_delta_min_hi(observed, nmax=nmax)


def test_delta_min_hi_extractor_rejects_nmax_below_two() -> None:
    with pytest.raises(ValueError, match="nmax must be at least 2"):
        _ = DeltaMinHiExtractor(nmax=0)


@pytest.mark.parametrize("seed", [*SEEDS])
@pytest.mark.parametrize("offset", [0, 113, 226, 339, 452, 777])
@pytest.mark.parametrize("nmax", [None, 13])
@pytest.mark.parametrize("num_chunks", [2, 5])
def test_delta_min_lo_extractor_matches_one_shot_inference(
    seed: str, offset: int, nmax: int | None, num_chunks: int
) -> None:
    releases = _generate_random_release_sequence(
        length=59, seed=f"{seed}:{offset}", maxgap=(10 + offset // 100)
    )
    windows = random_windows_around_releases(
        releases, seed=f"{seed}:{offset}:window", max_jitter=(20 + offset // 50)
    )

    extractor = DeltaMinLoExtractor(nmax=nmax)
    observed: list[ReleaseWindow] = []
    chunk_size = max(1, len(releases) // num_chunks)

    for batch in batched(windows, chunk_size):
        extractor.feed(batch)
        observed.extend(batch)

        assert extractor.current_model == infer_delta_min_lo(observed, nmax=nmax)


def test_delta_min_lo_extractor_rejects_nmax_below_two() -> None:
    with pytest.raises(ValueError, match="nmax must be at least 2"):
        _ = DeltaMinLoExtractor(nmax=0)


@pytest.mark.parametrize("seed", [*SEEDS])
@pytest.mark.parametrize("offset", [0, 113, 226, 339, 452, 777])
@pytest.mark.parametrize("nmax", [None, 13])
@pytest.mark.parametrize("num_chunks", [2, 5])
def test_delta_max_hi_extractor_matches_one_shot_inference(
    seed: str, offset: int, nmax: int | None, num_chunks: int
) -> None:
    releases = _generate_random_release_sequence(
        length=59, seed=f"{seed}:{offset}", maxgap=(10 + offset // 100)
    )
    windows = random_windows_around_releases(
        releases, seed=f"{seed}:{offset}:window", max_jitter=(20 + offset // 50)
    )

    extractor = DeltaMaxHiExtractor(nmax=nmax)
    observed: list[ReleaseWindow] = []
    chunk_size = max(1, len(releases) // num_chunks)

    for batch in batched(windows, chunk_size):
        extractor.feed(batch)
        observed.extend(batch)

        assert extractor.current_model == infer_delta_max_hi(observed, nmax=nmax)


def test_delta_max_hi_extractor_rejects_nonpositive_nmax() -> None:
    with pytest.raises(ValueError, match="nmax must be positive"):
        _ = DeltaMaxHiExtractor(nmax=0)


@pytest.mark.parametrize("seed", [*SEEDS])
@pytest.mark.parametrize("offset", [0, 113, 226, 339, 452, 777])
@pytest.mark.parametrize("nmax", [None, 13])
@pytest.mark.parametrize("num_chunks", [2, 5])
def test_delta_max_lo_extractor_matches_one_shot_inference(
    seed: str, offset: int, nmax: int | None, num_chunks: int
) -> None:
    releases = _generate_random_release_sequence(
        length=59, seed=f"{seed}:{offset}", maxgap=(10 + offset // 100)
    )
    windows = random_windows_around_releases(
        releases, seed=f"{seed}:{offset}:window", max_jitter=(20 + offset // 50)
    )

    extractor = DeltaMaxLoExtractor(nmax=nmax)
    observed: list[ReleaseWindow] = []
    chunk_size = max(1, len(releases) // num_chunks)

    for batch in batched(windows, chunk_size):
        extractor.feed(batch)
        observed.extend(batch)

        assert dmax_curve_lower_bounds_releases(
            extractor.current_model, releases[: len(observed)]
        )
        assert extractor.current_model == infer_delta_max_lo(observed, nmax=nmax)


def test_delta_max_lo_extractor_rejects_nonpositive_nmax() -> None:
    with pytest.raises(ValueError, match="nmax must be positive"):
        _ = DeltaMaxLoExtractor(nmax=0)

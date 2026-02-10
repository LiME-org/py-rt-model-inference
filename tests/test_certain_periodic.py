import math
import random
from itertools import batched

import pytest

from rt_model_inference.certain_periodic import (
    Batch,
    PeriodicModel,
    batch_gaps,
    batch_jitter,
    batch_last_processed_index,
    batch_mean_gap,
    batch_min_jitter_model,
    batch_model,
    batch_offset,
    batch_update,
    clean_batch,
    derived_model_candidates,
    infer_periodic_model,
    rounded_at_each_granularity,
    spaced_period_candidates,
    trailing_zeroes,
)
from rt_model_inference.extractors import PeriodicExtractor
from rt_model_inference.iterators import evenly_spaced_around, monotonic
from rt_model_inference.time import Duration, Instant
from rt_model_inference.validate import model_is_conservative

SAMPLE_BATCH: Batch = ((0, 3), (1, 14), (2, 24), (3, 35))


def test_batch_gaps_and_mean_gap() -> None:
    assert batch_gaps(SAMPLE_BATCH) == [11, 10, 11]
    assert math.isclose(batch_mean_gap(SAMPLE_BATCH), 32 / 3)


def test_batch_mean_gap_for_empty_and_singleton_batches() -> None:
    assert batch_mean_gap(tuple()) == 0
    assert batch_mean_gap(((0, 42),)) == 0


def test_clean_batch_keeps_full_batch_without_edge_outliers() -> None:
    assert clean_batch(SAMPLE_BATCH) == SAMPLE_BATCH


def test_clean_batch_trims_outlier_edges() -> None:
    batch: Batch = tuple(enumerate([0, 100, 102, 105, 107, 207]))

    assert clean_batch(batch) == ((1, 100), (2, 102), (3, 105), (4, 107))


def test_batch_offset_jitter_and_model() -> None:
    offset = batch_offset(SAMPLE_BATCH, period=10)
    jitter = batch_jitter(SAMPLE_BATCH, offset=offset, period=10)
    model = batch_model(SAMPLE_BATCH, period=10)

    assert offset == 3
    assert jitter == 2
    assert model == PeriodicModel(period=10, offset=3, jitter=2)


def test_batch_min_jitter_model_for_exactly_periodic_batch() -> None:
    batch: Batch = tuple(enumerate([7, 20, 33, 46, 59]))

    assert batch_min_jitter_model(batch) == PeriodicModel(period=13, offset=7, jitter=0)


def test_batch_min_jitter_model_uses_rounded_mean_when_search_window_collapses():
    model = batch_min_jitter_model(SAMPLE_BATCH, search_range=(1.0, 1.0))

    assert model == batch_model(SAMPLE_BATCH, period=11)


def test_evenly_spaced_around_includes_center_and_bounds() -> None:
    points = list(evenly_spaced_around(center=10.0, extension=2.0, n=5))

    assert points == [10.0, 8.0, 9.333333333333334, 10.666666666666666, 12.0]


def test_spaced_period_candidates_uses_unique_rounded_periods() -> None:
    min_jitter_model = PeriodicModel(period=10, offset=3, jitter=2)

    candidates = spaced_period_candidates(
        min_jitter_model=min_jitter_model,
        n_candidates=5,
        candidate_dispersion=2.0,
    )
    assert set(candidates) == {6, 9, 10, 11, 14}


def test_rounded_at_each_granularity() -> None:
    rounded = list(rounded_at_each_granularity(period=123, adjustments=(-1, 0, 1)))
    assert rounded == [110, 120, 130, 100, 200]

    rounded = list(
        rounded_at_each_granularity(period=993, adjustments=(-2, -1, 0, 1, 2))
    )
    assert rounded == [970, 980, 990, 1010, 700, 800, 900, 1100, 1000]

    rounded = list(rounded_at_each_granularity(period=998334, adjustments=(-1, 0, 1)))
    assert rounded == [
        998320,
        998330,
        998340,
        998200,
        998300,
        998400,
        997000,
        998000,
        999000,
        980000,
        990000,
        800000,
        900000,
        1000000,
    ]

    rounded = list(
        rounded_at_each_granularity(period=991605, adjustments=(-2, -1, 0, 1, 2))
    )
    assert rounded == [
        991580,
        991590,
        991610,
        991620,
        991400,
        991500,
        991600,
        991700,
        991800,
        989000,
        991000,
        992000,
        993000,
        970000,
        980000,
        990000,
        1010000,
        700000,
        800000,
        900000,
        1100000,
        1000000,
    ]

    rounded = list(
        rounded_at_each_granularity(period=49992, adjustments=(-2, -1, 0, 1, 2))
    )
    assert rounded == [
        49970,
        49980,
        49990,
        50010,
        49700,
        49800,
        49900,
        50100,
        47000,
        48000,
        49000,
        51000,
        20000,
        30000,
        40000,
        50000,
        60000,
    ]

    rounded = list(rounded_at_each_granularity(period=10000000, adjustments=(0, 1)))
    assert rounded == [
        10000010,
        10000100,
        10001000,
        10010000,
        10100000,
        11000000,
        10000000,
    ]

    rng = random.Random("test rounded_at_each_granularity")
    for _ in range(10000):
        period = rng.randint(1, 1000000)
        rounded = list(
            rounded_at_each_granularity(period, adjustments=(-2, -1, 0, 1, 2))
        )
        assert len(rounded) == len(set(rounded))
        assert rounded == sorted(rounded, key=trailing_zeroes)


def test_rounded_at_each_granularity_stops_for_small_period() -> None:
    assert list(rounded_at_each_granularity(period=10, adjustments=(-1, 0, 1))) == []


def test_batch_derived_model_candidates_updates_period_offset_and_jitter() -> None:
    batch: Batch = ((5, 50), (6, 62), (7, 74))
    model_candidates = [
        PeriodicModel(period=10, offset=40, jitter=3),
        PeriodicModel(14, 30, 5),
        PeriodicModel(period=20, offset=5, jitter=8),
    ]

    derived = list(
        derived_model_candidates(
            batch_last_processed_index(batch, 2),
            running_mean_period=13,
            min_jitter_period=22,
            model_candidates=model_candidates,
        )
    )

    assert derived == [
        PeriodicModel(period=13, offset=30, jitter=11),
        PeriodicModel(period=22, offset=-7, jitter=20),
    ]


def test_batch_update_adjusts_offset_and_jitter() -> None:
    updated = batch_update(
        SAMPLE_BATCH, mc_old=PeriodicModel(period=10, offset=5, jitter=1)
    )

    assert updated == PeriodicModel(period=10, offset=3, jitter=3)


def test_batch_update_keeps_offset_when_recomputed_offset_is_later() -> None:
    updated = batch_update(
        SAMPLE_BATCH, mc_old=PeriodicModel(period=10, offset=2, jitter=1)
    )

    assert updated == PeriodicModel(period=10, offset=2, jitter=3)


def test_infer_periodic_model_for_exact_periodic_releases() -> None:
    releases = [7 + 13 * i for i in range(100)]

    model = infer_periodic_model(
        releases,
        batch_size=11,
        overlap=3,
        n_candidates=20,
        candidate_dispersion=2.0,
        rounding_adjustments=(-1, 0, 1),
    )

    assert model == PeriodicModel(period=13, offset=7, jitter=0)


def test_infer_periodic_model_for_jittered_periodic_releases() -> None:
    releases = [3 + 10 * i + (i % 3) for i in range(90)]

    model = infer_periodic_model(
        releases,
        batch_size=8,
        overlap=1,
        n_candidates=30,
        candidate_dispersion=3.0,
        rounding_adjustments=(-2, -1, 0, 1, 2),
    )

    assert model == PeriodicModel(period=10, offset=3, jitter=2)


def test_infer_periodic_model_for_highly_jittered_periodic_releases() -> None:
    releases = [3 + 10 * i + (i % 27) for i in range(90)]

    model = infer_periodic_model(
        releases,
        batch_size=8,
        overlap=1,
        n_candidates=30,
        candidate_dispersion=3.0,
        rounding_adjustments=(-2, -1, 0, 1, 2),
    )

    assert model == PeriodicModel(period=10, offset=3, jitter=26)


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


def mk_releases(
    seed: str, well_known_periods: bool = False
) -> tuple[list[Instant], Duration]:
    rng = random.Random(seed)
    scale = 1000
    if well_known_periods:
        period = scale * rng.choice([1, 5, 10, 25, 50, 100, 125, 250, 500, 1000])
    else:
        period = rng.randint(10 * scale, 1000 * scale)
    offset = rng.randint(0, 50 * scale)
    jitter = rng.randint(0, 3 * period)

    jitters = [rng.randint(0, jitter) for _ in range(2500)]
    # need at least one release with zero jitter in the trace
    jitters[rng.randrange(0, len(jitters))] = 0
    releases = list(monotonic((offset + i * period + j for i, j in enumerate(jitters))))
    return releases, period


@pytest.mark.parametrize("seed", [*SEEDS])
@pytest.mark.parametrize("well_known_periods", [True, False])
def test_infer_periodic_model_for_randomized_periodic_releases(
    seed: str, well_known_periods: bool
) -> None:
    releases, period = mk_releases(seed, well_known_periods)

    batch_size = 64 if well_known_periods else 512

    model = infer_periodic_model(releases, batch_size=batch_size)
    assert model_is_conservative(model, releases)
    if well_known_periods:
        assert model.period == period

    ex = PeriodicExtractor(batch_size=batch_size)
    ex.feed(releases)
    assert ex.current_model == model


@pytest.mark.parametrize("seed", [*SEEDS])
@pytest.mark.parametrize("num_chunks", [7, 47])
def test_periodic_extractor_matches_one_shot_inference(
    seed: str, num_chunks: int
) -> None:
    releases, _ = mk_releases(seed)

    extractor = PeriodicExtractor(batch_size=512)
    observed: list[Instant] = []
    chunk_size = max(1, len(releases) // num_chunks)

    for batch in batched(releases, chunk_size):
        extractor.feed(batch)
        observed.extend(batch)

        assert model_is_conservative(extractor.current_model, observed)  # pyright: ignore[reportArgumentType]
        assert extractor.current_model == infer_periodic_model(observed, batch_size=512)

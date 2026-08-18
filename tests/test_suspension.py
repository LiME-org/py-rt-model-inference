import random
from itertools import batched

import pytest

from rt_model_inference import (
    BasicJobSuspensionBehavior,
    BasicSuspensionModel,
    Segment,
    SegmentedJobSuspensionBehavior,
    SegmentedSuspensionModel,
    basic_from_segmented_model,
    basic_from_segmented_observation,
    infer_max_suspension_model,
    infer_min_suspension_model,
    infer_segmented_suspension_model,
)
from rt_model_inference.extractors import (
    MaxSuspensionModelExtractor,
    MinSuspensionModelExtractor,
    SegmentedSuspensionModelExtractor,
)


def test_infer_maximum_observed_suspension_model() -> None:
    observations: list[BasicJobSuspensionBehavior] = [
        (3, 10),
        (1, 7),
        (4, 1),
        (1, 5),
        (5, 2),
    ]

    assert infer_max_suspension_model(observations) == [
        BasicSuspensionModel(0, 0),
        BasicSuspensionModel(5, 10),
        BasicSuspensionModel(6, 17),
        BasicSuspensionModel(10, 18),
        BasicSuspensionModel(11, 23),
        BasicSuspensionModel(14, 25),
    ]


def test_infer_maximum_observed_suspension_model_honors_nmax() -> None:
    observations: list[BasicJobSuspensionBehavior] = [
        (3, 10),
        (1, 7),
        (4, 1),
        (1, 5),
        (5, 2),
    ]

    assert infer_max_suspension_model(observations, nmax=3) == [
        BasicSuspensionModel(0, 0),
        BasicSuspensionModel(5, 10),
        BasicSuspensionModel(6, 17),
        BasicSuspensionModel(10, 18),
    ]


def test_infer_minimum_observed_suspension_model() -> None:
    observations: list[BasicJobSuspensionBehavior] = [
        (3, 10),
        (1, 7),
        (4, 1),
        (1, 5),
        (5, 2),
    ]

    assert infer_min_suspension_model(observations) == [
        BasicSuspensionModel(0, 0),
        BasicSuspensionModel(1, 1),
        BasicSuspensionModel(4, 6),
        BasicSuspensionModel(6, 8),
        BasicSuspensionModel(9, 15),
        BasicSuspensionModel(14, 25),
    ]


def test_infer_minimum_observed_suspension_model_honors_nmax() -> None:
    observations: list[BasicJobSuspensionBehavior] = [
        (3, 10),
        (1, 7),
        (4, 1),
        (1, 5),
        (5, 2),
    ]

    assert infer_min_suspension_model(observations, nmax=3) == [
        BasicSuspensionModel(0, 0),
        BasicSuspensionModel(1, 1),
        BasicSuspensionModel(4, 6),
        BasicSuspensionModel(6, 8),
    ]


def test_infer_maximum_observed_suspension_model_for_empty_observations() -> None:
    assert infer_max_suspension_model([]) == []


def test_infer_minimum_observed_suspension_model_for_empty_observations() -> None:
    assert infer_min_suspension_model([]) == []


def test_infer_maximum_observed_suspension_model_rejects_invalid_nmax() -> None:
    with pytest.raises(ValueError, match="nmax must be positive"):
        _ = infer_max_suspension_model([(1, 2)], nmax=0)


def test_infer_minimum_observed_suspension_model_rejects_invalid_nmax() -> None:
    with pytest.raises(ValueError, match="nmax must be positive"):
        _ = infer_min_suspension_model([(1, 2)], nmax=0)


def test_infer_maximum_observed_suspension_model_rejects_negative_counts() -> None:
    with pytest.raises(
        ValueError, match="resource-use observations must be non-negative"
    ):
        _ = infer_max_suspension_model([(1, 2), (-1, 3)])


def test_infer_minimum_observed_suspension_model_rejects_negative_counts() -> None:
    with pytest.raises(
        ValueError, match="resource-use observations must be non-negative"
    ):
        _ = infer_min_suspension_model([(1, 2), (-1, 3)])


def test_infer_maximum_observed_suspension_model_rejects_negative_durations() -> None:
    with pytest.raises(
        ValueError, match="resource-use observations must be non-negative"
    ):
        _ = infer_max_suspension_model([(1, 2), (3, -1)])


def test_infer_minimum_observed_suspension_model_rejects_negative_durations() -> None:
    with pytest.raises(
        ValueError, match="resource-use observations must be non-negative"
    ):
        _ = infer_min_suspension_model([(1, 2), (3, -1)])


def test_segmented_to_basic_for_empty_observation() -> None:
    assert basic_from_segmented_observation([]) == (0, 0)


def test_segmented_to_basic_observations() -> None:
    segments: SegmentedJobSuspensionBehavior = [(3, 5), (7, 11), (13, 17)]

    assert basic_from_segmented_observation(segments) == (2, 23)

    assert basic_from_segmented_observation([(0, 10)]) == (0, 0)
    assert basic_from_segmented_observation([(0, 10), (99, 123)]) == (1, 99)


def test_segmented_to_basic_for_empty_model() -> None:
    assert basic_from_segmented_model([]) == BasicSuspensionModel(0, 0)


def test_segmented_to_basic_model() -> None:
    model: SegmentedSuspensionModel = [Segment(3, 5), Segment(7, 11), Segment(13, 17)]

    assert basic_from_segmented_model(model) == BasicSuspensionModel(2, 23)

    assert basic_from_segmented_model([Segment(0, 10)]) == BasicSuspensionModel(0, 0)
    assert basic_from_segmented_model(
        [Segment(0, 10), Segment(99, 123)]
    ) == BasicSuspensionModel(1, 99)


def test_infer_segmented_suspension_model_for_empty_observations() -> None:
    assert infer_segmented_suspension_model([]) == []


def test_infer_segmented_suspension_model_takes_per_segment_maxima() -> None:
    observations: list[SegmentedJobSuspensionBehavior] = [
        [(1, 9), (8, 2)],
        [(3, 4)],
        [(2, 7), (5, 6), (11, 13)],
    ]

    assert infer_segmented_suspension_model(iter(observations)) == [
        Segment(max_suspension_time=3, max_execution_time=9),
        Segment(max_suspension_time=8, max_execution_time=6),
        Segment(max_suspension_time=11, max_execution_time=13),
    ]


def test_infer_segmented_suspension_model_honors_max_segments() -> None:
    observations: list[SegmentedJobSuspensionBehavior] = [
        [(1, 2)],
        [(3, 4), (5, 6)],
    ]

    assert infer_segmented_suspension_model(observations, max_segments=2) == [
        Segment(max_suspension_time=3, max_execution_time=4),
        Segment(max_suspension_time=5, max_execution_time=6),
    ]
    assert infer_segmented_suspension_model(observations, max_segments=1) is None


def test_infer_segmented_suspension_model_rejects_negative_suspension_time() -> None:
    with pytest.raises(
        ValueError, match="segment suspension-time observations must be non-negative"
    ):
        _ = infer_segmented_suspension_model([[(1, 2)], [(-1, 3)]])


def test_infer_segmented_suspension_model_rejects_negative_execution_time() -> None:
    with pytest.raises(
        ValueError, match="segment execution-time observations must be non-negative"
    ):
        _ = infer_segmented_suspension_model([[(1, 2)], [(3, -1)]])


@pytest.mark.parametrize(
    "seed", ["Segmented self-suspensions", "are", "also", "tricky."]
)
@pytest.mark.parametrize("max_segments", [None, 7, 12])
@pytest.mark.parametrize("num_chunks", [1, 3, 17])
def test_segmented_suspension_model_extractor_matches_one_shot_inference(
    seed: str, max_segments: int | None, num_chunks: int
) -> None:
    rng = random.Random(seed)
    observations: list[SegmentedJobSuspensionBehavior] = [
        [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(rng.randint(1, 8))]
        for _ in range(251)
    ]
    extractor = SegmentedSuspensionModelExtractor(max_segments=max_segments)
    observed: list[SegmentedJobSuspensionBehavior] = []
    chunk_size = max(1, len(observations) // num_chunks)

    assert extractor.current_model == []

    for batch in batched(observations, chunk_size):
        extractor.feed(batch)
        observed.extend(batch)

        assert extractor.current_model == infer_segmented_suspension_model(
            observed, max_segments=max_segments
        )


@pytest.mark.parametrize("seed", ["Self-suspensions", "are", "tricky."])
@pytest.mark.parametrize("nmax", [None, 1, 7, 32])
@pytest.mark.parametrize("num_chunks", [2, 5, 100])
def test_maximum_observed_suspension_model_extractor_matches_one_shot_inference(
    seed: str, nmax: int | None, num_chunks: int
) -> None:
    rng = random.Random(seed)
    observations = [(rng.randint(0, 10), rng.randint(0, 100)) for _ in range(517)]
    extractor = MaxSuspensionModelExtractor(nmax=nmax)
    observed: list[BasicJobSuspensionBehavior] = []
    chunk_size = max(1, len(observations) // num_chunks)

    assert extractor.current_model == []

    for batch in batched(observations, chunk_size):
        extractor.feed(batch)
        observed.extend(batch)

        assert extractor.current_model == infer_max_suspension_model(
            observed, nmax=nmax
        )


@pytest.mark.parametrize("seed", ["Self-suspensions", "are", "tricky."])
@pytest.mark.parametrize("nmax", [None, 1, 7, 32])
@pytest.mark.parametrize("num_chunks", [2, 5, 100])
def test_minimum_observed_suspension_model_extractor_matches_one_shot_inference(
    seed: str, nmax: int | None, num_chunks: int
) -> None:
    rng = random.Random(seed)
    observations = [(rng.randint(0, 10), rng.randint(0, 100)) for _ in range(517)]
    extractor = MinSuspensionModelExtractor(nmax=nmax)
    observed: list[BasicJobSuspensionBehavior] = []
    chunk_size = max(1, len(observations) // num_chunks)

    assert extractor.current_model == []

    for batch in batched(observations, chunk_size):
        extractor.feed(batch)
        observed.extend(batch)

        assert extractor.current_model == infer_min_suspension_model(
            observed, nmax=nmax
        )


def test_maximum_observed_suspension_model_extractor_rejects_invalid_nmax() -> None:
    with pytest.raises(ValueError, match="nmax must be positive"):
        _ = MaxSuspensionModelExtractor(nmax=0)


def test_minimum_observed_suspension_model_extractor_rejects_invalid_nmax() -> None:
    with pytest.raises(ValueError, match="nmax must be positive"):
        _ = MinSuspensionModelExtractor(nmax=0)

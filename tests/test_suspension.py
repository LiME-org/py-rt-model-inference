import random
from itertools import batched

import pytest

import rt_model_inference
from rt_model_inference.extractors import (
    MaxSuspensionModelExtractor,
    MinSuspensionModelExtractor,
)
from rt_model_inference.suspension import (
    BasicJobSuspensionBehavior,
    BasicSuspensionModel,
    infer_max_suspension_model,
    infer_min_suspension_model,
)


def test_suspension_apis_are_exported_from_package_root() -> None:
    observation: rt_model_inference.BasicJobSuspensionBehavior = (1, 2)

    assert rt_model_inference.BasicSuspensionModel is BasicSuspensionModel
    assert rt_model_inference.infer_max_suspension_model([observation]) == [
        BasicSuspensionModel(0, 0),
        BasicSuspensionModel(1, 2),
    ]
    assert rt_model_inference.infer_min_suspension_model([observation]) == [
        BasicSuspensionModel(0, 0),
        BasicSuspensionModel(1, 2),
    ]


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

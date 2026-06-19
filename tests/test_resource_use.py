import random
from itertools import batched

import pytest

from rt_model_inference import (
    ResourceAmount,
    infer_max_resource_use,
    infer_min_resource_use,
)
from rt_model_inference.extractors import MaxResourceUseExtractor, MinResourceUseExtractor


def test_infer_maximum_observed_resource_use() -> None:
    observations: list[ResourceAmount] = [3, 1, 4, 1, 5]

    assert infer_max_resource_use(observations) == [0, 5, 6, 10, 11, 14]


def test_infer_maximum_observed_resource_use_honors_nmax() -> None:
    observations: list[ResourceAmount] = [3, 1, 4, 1, 5]

    assert infer_max_resource_use(observations, nmax=3) == [0, 5, 6, 10]


def test_infer_minimum_observed_resource_use() -> None:
    observations: list[ResourceAmount] = [3, 1, 4, 1, 5]

    assert infer_min_resource_use(observations) == [0, 1, 4, 6, 9, 14]


def test_infer_minimum_observed_resource_use_honors_nmax() -> None:
    observations: list[ResourceAmount] = [3, 1, 4, 1, 5]

    assert infer_min_resource_use(observations, nmax=3) == [0, 1, 4, 6]


def brute_force_maximum_observed_resource_use(
    observations: list[ResourceAmount], nmax: int | None = None
) -> list[ResourceAmount]:
    if len(observations) == 0:
        return []

    max_width = len(observations) if nmax is None else min(nmax, len(observations))
    return [
        0,
        *[
            max(
                sum(observations[start : start + width])
                for start in range(len(observations) - width + 1)
            )
            for width in range(1, max_width + 1)
        ],
    ]


def brute_force_minimum_observed_resource_use(
    observations: list[ResourceAmount], nmax: int | None = None
) -> list[ResourceAmount]:
    if len(observations) == 0:
        return []

    max_width = len(observations) if nmax is None else min(nmax, len(observations))
    return [
        0,
        *[
            min(
                sum(observations[start : start + width])
                for start in range(len(observations) - width + 1)
            )
            for width in range(1, max_width + 1)
        ],
    ]


SEEDS = [
    "Wilhelm",
    "Puaut",
    "Reineke",
    "Bernat",
    "Ferdinand",
    "Mitra",
    "Puschner",
    "Staschulat",
]


@pytest.mark.parametrize("seed", [*SEEDS])
@pytest.mark.parametrize("nmax", [None, 1, 7, 32])
def test_infer_maximum_observed_resource_use_matches_brute_force(
    seed: str, nmax: int | None
) -> None:
    rng = random.Random(seed)
    observations = [rng.randint(0, 100) for _ in range(250)]

    assert infer_max_resource_use(
        observations, nmax=nmax
    ) == brute_force_maximum_observed_resource_use(observations, nmax=nmax)


@pytest.mark.parametrize("seed", [*SEEDS])
@pytest.mark.parametrize("nmax", [None, 1, 7, 32])
def test_infer_minimum_observed_resource_use_matches_brute_force(
    seed: str, nmax: int | None
) -> None:
    rng = random.Random(seed)
    observations = [rng.randint(0, 100) for _ in range(250)]

    assert infer_min_resource_use(
        observations, nmax=nmax
    ) == brute_force_minimum_observed_resource_use(observations, nmax=nmax)


def test_infer_maximum_observed_resource_use_for_empty_observations() -> None:
    assert infer_max_resource_use([]) == []


def test_infer_minimum_observed_resource_use_for_empty_observations() -> None:
    assert infer_min_resource_use([]) == []


def test_infer_maximum_observed_resource_use_rejects_invalid_nmax() -> None:
    with pytest.raises(ValueError, match="nmax must be positive"):
        _ = infer_max_resource_use([1, 2, 3], nmax=0)


def test_infer_minimum_observed_resource_use_rejects_invalid_nmax() -> None:
    with pytest.raises(ValueError, match="nmax must be positive"):
        _ = infer_min_resource_use([1, 2, 3], nmax=0)


def test_infer_maximum_observed_resource_use_rejects_negative_observations() -> None:
    with pytest.raises(ValueError, match="resource-use observations must be non-negative"):
        _ = infer_max_resource_use([1, -1, 3])


def test_infer_minimum_observed_resource_use_rejects_negative_observations() -> None:
    with pytest.raises(ValueError, match="resource-use observations must be non-negative"):
        _ = infer_min_resource_use([1, -1, 3])


@pytest.mark.parametrize("seed", [*SEEDS])
@pytest.mark.parametrize("nmax", [None, 1, 7, 32])
@pytest.mark.parametrize("num_chunks", [2, 5, 100])
def test_maximum_observed_resource_use_extractor_matches_one_shot_inference(
    seed: str, nmax: int | None, num_chunks: int
) -> None:
    rng = random.Random(seed)
    observations = [rng.randint(0, 100) for _ in range(517)]
    extractor = MaxResourceUseExtractor(nmax=nmax)
    observed: list[ResourceAmount] = []
    chunk_size = max(1, len(observations) // num_chunks)

    assert extractor.current_model == []

    for batch in batched(observations, chunk_size):
        extractor.feed(batch)
        observed.extend(batch)

        assert extractor.current_model == infer_max_resource_use(observed, nmax=nmax)


@pytest.mark.parametrize("seed", [*SEEDS])
@pytest.mark.parametrize("nmax", [None, 1, 7, 32])
@pytest.mark.parametrize("num_chunks", [2, 5, 100])
def test_minimum_observed_resource_use_extractor_matches_one_shot_inference(
    seed: str, nmax: int | None, num_chunks: int
) -> None:
    rng = random.Random(seed)
    observations = [rng.randint(0, 100) for _ in range(517)]
    extractor = MinResourceUseExtractor(nmax=nmax)
    observed: list[ResourceAmount] = []
    chunk_size = max(1, len(observations) // num_chunks)

    assert extractor.current_model == []

    for batch in batched(observations, chunk_size):
        extractor.feed(batch)
        observed.extend(batch)

        assert extractor.current_model == infer_min_resource_use(observed, nmax=nmax)


def test_maximum_observed_resource_use_extractor_rejects_invalid_nmax() -> None:
    with pytest.raises(ValueError, match="nmax must be positive"):
        _ = MaxResourceUseExtractor(nmax=0)


def test_minimum_observed_resource_use_extractor_rejects_invalid_nmax() -> None:
    with pytest.raises(ValueError, match="nmax must be positive"):
        _ = MinResourceUseExtractor(nmax=0)


def test_maximum_observed_resource_use_extractor_rejects_negative_observations() -> None:
    extractor = MaxResourceUseExtractor()

    with pytest.raises(ValueError, match="resource-use observations must be non-negative"):
        extractor.feed([-1])

    assert extractor.current_model == []


def test_minimum_observed_resource_use_extractor_rejects_negative_observations() -> None:
    extractor = MinResourceUseExtractor()

    with pytest.raises(ValueError, match="resource-use observations must be non-negative"):
        extractor.feed([-1])

    assert extractor.current_model == []

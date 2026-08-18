import random
from itertools import batched

import pytest

from rt_model_inference import (
    BagOfSegmentsModel,
    BasicJobSuspensionBehavior,
    BasicSuspensionModel,
    Segment,
    SegmentedJobSuspensionBehavior,
    SegmentedSuspensionModel,
    basic_from_segmented_model,
    basic_from_segmented_observation,
    infer_bos_suspension_model,
    infer_max_suspension_model,
    infer_min_suspension_model,
    infer_segmented_suspension_model,
)
from rt_model_inference.extractors import (
    BOSSuspensionModelExtractor,
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


def test_infer_bos_suspension_model_for_empty_observations() -> None:
    assert infer_bos_suspension_model([]) == []


def test_infer_bos_suspension_model_retains_non_dominated_models() -> None:
    observations: list[SegmentedJobSuspensionBehavior] = [
        [(1, 10)],
        [(2, 9)],
        [(0, 8)],  # already covered by the first model
        [(3, 11), (4, 5)],
    ]

    model: BagOfSegmentsModel | None = infer_bos_suspension_model(observations)

    assert model == [
        [Segment(max_suspension_time=1, max_execution_time=10)],
        [Segment(max_suspension_time=2, max_execution_time=9)],
        [
            Segment(max_suspension_time=3, max_execution_time=11),
            Segment(max_suspension_time=4, max_execution_time=5),
        ],
    ]


def test_infer_bos_suspension_model_honors_max_models() -> None:
    observations: list[SegmentedJobSuspensionBehavior] = [
        [(1, 9), (8, 2)],
        [(3, 4)],
        [(2, 7), (5, 6), (11, 13)],
    ]

    assert infer_bos_suspension_model(observations, max_models=1) == [
        [
            Segment(max_suspension_time=3, max_execution_time=9),
            Segment(max_suspension_time=8, max_execution_time=6),
            Segment(max_suspension_time=11, max_execution_time=13),
        ]
    ]


def test_infer_bos_suspension_model_honors_max_segments() -> None:
    observations: list[SegmentedJobSuspensionBehavior] = [
        [(1, 2)],
        [(3, 4), (5, 6)],
    ]

    assert infer_bos_suspension_model(observations, max_segments=2) is not None
    assert infer_bos_suspension_model(observations, max_segments=1) is None


@pytest.mark.parametrize("max_segments", [0, -1])
def test_infer_bos_suspension_model_rejects_invalid_max_segments(
    max_segments: int,
) -> None:
    with pytest.raises(ValueError, match="max_segments must be positive"):
        _ = infer_bos_suspension_model([], max_segments=max_segments)


@pytest.mark.parametrize("max_models", [0, -1])
def test_infer_bos_suspension_model_rejects_invalid_max_models(
    max_models: int,
) -> None:
    with pytest.raises(ValueError, match="max_models must be positive"):
        _ = infer_bos_suspension_model([], max_models=max_models)


def test_infer_bos_suspension_model_rejects_negative_suspension_time() -> None:
    with pytest.raises(
        ValueError, match="segment suspension-time observations must be non-negative"
    ):
        _ = infer_bos_suspension_model([[(1, 2)], [(-1, 3)]])


def test_infer_bos_suspension_model_rejects_negative_execution_time() -> None:
    with pytest.raises(
        ValueError, match="segment execution-time observations must be non-negative"
    ):
        _ = infer_bos_suspension_model([[(1, 2)], [(3, -1)]])


@pytest.mark.parametrize(
    "seed", ["BOS", "streaming", "avoids", "quadratic", "latency", "spikes."]
)
@pytest.mark.parametrize("max_segments", [None, 5, 10])
@pytest.mark.parametrize("max_models", [None, 1, 2, 5, 10])
@pytest.mark.parametrize("num_chunks", [1, 7])
def test_bos_suspension_model_extractor_matches_one_shot_inference(
    seed: str,
    max_segments: int | None,
    max_models: int | None,
    num_chunks: int | None,
) -> None:
    rng = random.Random(seed)
    observations: list[SegmentedJobSuspensionBehavior] = [
        [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(rng.randint(1, 8))]
        for _ in range(261)
    ]
    extractor = BOSSuspensionModelExtractor(
        max_segments=max_segments, max_models=max_models
    )
    observed: list[SegmentedJobSuspensionBehavior] = []
    if num_chunks is not None:
        chunk_size = max(1, len(observations) // num_chunks)
    else:
        chunk_size = 1

    assert extractor.current_model == []

    for batch in batched(observations, chunk_size):
        extractor.feed(batch)
        observed.extend(batch)

        assert extractor.current_model == infer_bos_suspension_model(
            observed, max_segments=max_segments, max_models=max_models
        )


def test_bos_suspension_model_extractor_preserves_merge_tie_breaking() -> None:
    observations: list[SegmentedJobSuspensionBehavior] = [
        [(15, 7)],
        [(24, 3)],
        [(19, 5)],
        [(12, 9)],
        [(1, 11)],
    ]
    extractor = BOSSuspensionModelExtractor(max_models=3)

    extractor.feed(observations)

    expected = [
        [Segment(max_suspension_time=15, max_execution_time=11)],
        [Segment(max_suspension_time=19, max_execution_time=5)],
        [Segment(max_suspension_time=24, max_execution_time=3)],
    ]
    assert infer_bos_suspension_model(observations, max_models=3) == expected
    assert extractor.current_model == expected


def test_bos_suspension_model_extractor_is_callable_and_returns_a_copy() -> None:
    extractor = BOSSuspensionModelExtractor()
    extractor([[(1, 2)]])

    current = extractor.current_model
    assert current == [[Segment(1, 2)]]
    assert current is not None
    current[0].append(Segment(3, 4))
    current.append([])

    assert extractor.current_model == [[Segment(1, 2)]]


def test_bos_suspension_model_extractor_becomes_terminal_on_segment_overflow() -> None:
    extractor = BOSSuspensionModelExtractor(max_segments=1, max_models=2)

    extractor.feed([[(1, 2)], [(3, 4), (5, 6)], [(-1, -1)]])
    assert extractor.current_model is None

    extractor.feed([[(-1, -1)]])
    assert extractor.current_model is None


@pytest.mark.parametrize("max_segments", [0, -1])
def test_bos_suspension_model_extractor_rejects_invalid_max_segments(
    max_segments: int,
) -> None:
    with pytest.raises(ValueError, match="max_segments must be positive"):
        _ = BOSSuspensionModelExtractor(max_segments=max_segments)


@pytest.mark.parametrize("max_models", [0, -1])
def test_bos_suspension_model_extractor_rejects_invalid_max_models(
    max_models: int,
) -> None:
    with pytest.raises(ValueError, match="max_models must be positive"):
        _ = BOSSuspensionModelExtractor(max_models=max_models)


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        ([(-1, 2)], "segment suspension-time observations must be non-negative"),
        ([(1, -2)], "segment execution-time observations must be non-negative"),
    ],
)
def test_bos_suspension_model_extractor_rejects_negative_observations(
    observation: SegmentedJobSuspensionBehavior, message: str
) -> None:
    extractor = BOSSuspensionModelExtractor()

    with pytest.raises(ValueError, match=message):
        extractor.feed([observation])

    assert extractor.current_model == []


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


@pytest.mark.parametrize(
    "seed", ["BOS", "and", "segmented", "extractors", "agree", "for max_models=1."]
)
@pytest.mark.parametrize("max_segments", [None, 5, 10])
def test_bos_suspension_model_matches_segmented_suspension_model_inference(
    seed: str,
    max_segments: int | None,
) -> None:
    rng = random.Random(seed)
    observations: list[SegmentedJobSuspensionBehavior] = [
        [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(rng.randint(1, 8))]
        for _ in range(261)
    ]

    bos = infer_bos_suspension_model(
        observations, max_segments=max_segments, max_models=1
    )

    segmented = infer_segmented_suspension_model(
        observations, max_segments=max_segments
    )

    assert segmented == bos or bos is not None and segmented == bos[0]


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

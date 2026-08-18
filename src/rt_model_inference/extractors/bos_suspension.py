"""Continuously update Bag-of-Segments self-suspension models."""

from collections.abc import Iterable
from itertools import zip_longest

from rt_model_inference.suspension import (
    BagOfSegmentsModel,
    Segment,
    SegmentedJobSuspensionBehavior,
    SegmentedSuspensionModel,
)


def covers(
    model: SegmentedSuspensionModel, segments: SegmentedJobSuspensionBehavior
) -> bool:
    "Check if a given `model` covers a given observed vector of `segments`."
    if len(model) < len(segments):
        return False
    for ms, os in zip(model, segments):
        if ms.max_execution_time < os[1] or ms.max_suspension_time < os[0]:
            return False
    return True


def merge_cost(
    model_a: SegmentedSuspensionModel,
    model_b: SegmentedSuspensionModel,
) -> int:
    """Estimate the increase in pessimism needed for `model_a` to cover `model_b`.

    The cost function penalizes increases in execution-time bounds more than increases
    in suspension bounds since over-utilization usually results in more pessimism in
    schedulability tests.
    """
    total = 0
    for seg_a, seg_b in zip_longest(model_a, model_b):
        if seg_a is not None and seg_b is not None:
            total += max(0, seg_b.max_execution_time - seg_a.max_execution_time) ** 2
            total += max(0, seg_b.max_suspension_time - seg_a.max_suspension_time)
        elif seg_b is not None:
            total += seg_b.max_execution_time**2 + seg_b.max_suspension_time
    return total


def merge_models(
    target: SegmentedSuspensionModel, source: SegmentedSuspensionModel
) -> SegmentedSuspensionModel:
    """Return the component-wise union of two segmented models."""
    overlap = [
        Segment(
            max(target_segment[0], source_segment[0]),
            max(target_segment[1], source_segment[1]),
        )
        for target_segment, source_segment in zip(target, source)
    ]
    return overlap + target[len(overlap) :] + source[len(overlap) :]


class BOSSuspensionModelExtractor:
    """Streaming extractor for Bag-of-Segments self-suspension models.

    With `max_models` set, the cheapest merge target for each source model is
    approximated incrementally to avoid a quadratic latency spike on compaction.
    """

    def __init__(
        self,
        max_segments: int | None = None,
        max_models: int | None = None,
    ):
        if max_segments is not None and max_segments <= 0:
            raise ValueError("max_segments must be positive")
        if max_models is not None and max_models <= 0:
            raise ValueError("max_models must be positive")

        self._max_segments: int | None = max_segments
        self._max_models: int | None = max_models
        self._models: dict[int, SegmentedSuspensionModel] = {}
        self._best_merges: dict[int, tuple[int, int, int]] = {}
        self._next_model_id: int = 0
        self._terminal: bool = False

    def _consider_target(self, source_id: int, target_id: int) -> None:
        candidate = (
            merge_cost(self._models[target_id], self._models[source_id]),
            -target_id,
            -source_id,
        )
        current = self._best_merges.get(source_id)
        if current is None or candidate < current:
            self._best_merges[source_id] = candidate

    def _add_model(self, model: SegmentedSuspensionModel) -> None:
        model_id = self._next_model_id
        self._models[model_id] = model
        self._next_model_id += 1

        if self._max_models is not None and len(self._models) > 1:
            for source_id in (sid for sid in self._models if sid != model_id):
                self._consider_target(source_id, model_id)
                self._consider_target(model_id, source_id)

    def _consolidate(self) -> None:
        _cost, neg_target_id, neg_source_id = min(self._best_merges.values())
        target_id = -neg_target_id
        source_id = -neg_source_id

        merged = merge_models(self._models[target_id], self._models[source_id])

        # remove the source
        del self._models[source_id]
        del self._best_merges[source_id]

        # remove the target
        del self._models[target_id]
        del self._best_merges[target_id]

        # add the merged model
        self._add_model(merged)

    def feed(self, observations: Iterable[SegmentedJobSuspensionBehavior]) -> None:
        "Update the current model estimate based on newly observed segments."
        if self._terminal:
            return

        for observed in observations:
            if len(observed) == 0:
                raise ValueError("each observation must contain at least one segment")

            if self._max_segments is not None and len(observed) > self._max_segments:
                self._terminal = True
                self._models.clear()
                self._best_merges.clear()
                return

            for suspension_time, execution_time in observed:
                if suspension_time < 0:
                    raise ValueError(
                        "segment suspension-time observations must be non-negative"
                    )
                if execution_time < 0:
                    raise ValueError(
                        "segment execution-time observations must be non-negative"
                    )

            # first check if this observation is already accounted for by an existing model
            if any(covers(model, observed) for model in self._models.values()):
                continue

            # it's not covered, so let's add a new model to account for this observation
            self._add_model([Segment(*segment) for segment in observed])

            # make sure we don't violate the space limit
            if self._max_models is not None and len(self._models) > self._max_models:
                self._consolidate()

    def __call__(self, observations: Iterable[SegmentedJobSuspensionBehavior]) -> None:
        self.feed(observations)

    @property
    def current_model(self) -> BagOfSegmentsModel | None:
        "Query the model extracted so far."
        if self._terminal:
            return None
        else:
            return sorted(
                (list(model) for model in self._models.values()),
                key=lambda m: (len(m), m),
            )

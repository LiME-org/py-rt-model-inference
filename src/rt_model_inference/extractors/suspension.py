"""Continuously updating streaming extractors for self-suspension models."""

from collections.abc import Iterable

from rt_model_inference.extractors.resource_use import (
    MaxResourceUseExtractor,
    MinResourceUseExtractor,
)
from rt_model_inference.suspension import (
    BasicJobSuspensionBehavior,
    BasicSuspensionModel,
    Segment,
    SegmentedJobSuspensionBehavior,
    SegmentedSuspensionModel,
)


class MaxSuspensionModelExtractor:
    "Streaming extractor for over-approximated basic self-suspension models."

    def __init__(self, nmax: int | None = None):
        self._counts: MaxResourceUseExtractor = MaxResourceUseExtractor(nmax=nmax)
        self._durations: MaxResourceUseExtractor = MaxResourceUseExtractor(nmax=nmax)

    def feed(self, observations: Iterable[BasicJobSuspensionBehavior]) -> None:
        "Update the current model estimate based on newly observed suspensions."
        for count, duration in observations:
            self._counts.feed([count])
            self._durations.feed([duration])

    def __call__(self, observations: Iterable[BasicJobSuspensionBehavior]) -> None:
        self.feed(observations)

    @property
    def current_model(self) -> list[BasicSuspensionModel]:
        "Query the model extracted so far."
        return [
            BasicSuspensionModel(count, duration)
            for count, duration in zip(
                self._counts.current_model, self._durations.current_model
            )
        ]


class MinSuspensionModelExtractor:
    "Streaming extractor for under-approximated basic self-suspension models."

    def __init__(self, nmax: int | None = None):
        self._counts: MinResourceUseExtractor = MinResourceUseExtractor(nmax=nmax)
        self._durations: MinResourceUseExtractor = MinResourceUseExtractor(nmax=nmax)

    def feed(self, observations: Iterable[BasicJobSuspensionBehavior]) -> None:
        "Update the current model estimate based on newly observed suspensions."
        for count, duration in observations:
            self._counts.feed([count])
            self._durations.feed([duration])

    def __call__(self, observations: Iterable[BasicJobSuspensionBehavior]) -> None:
        self.feed(observations)

    @property
    def current_model(self) -> list[BasicSuspensionModel]:
        "Query the model extracted so far."
        return [
            BasicSuspensionModel(count, duration)
            for count, duration in zip(
                self._counts.current_model, self._durations.current_model
            )
        ]


class SegmentedSuspensionModelExtractor:
    "Streaming extractor for segmented self-suspension models."

    def __init__(self, max_segments: int | None = None):
        if max_segments is not None and max_segments <= 0:
            raise ValueError("max_segments must be positive")
        self._max_segments: int | None = max_segments
        self._model: SegmentedSuspensionModel | None = []

    def feed(self, observations: Iterable[SegmentedJobSuspensionBehavior]) -> None:
        "Update the current model estimate based on newly observed segments."
        if self._model is None:
            return

        for observed in observations:
            if len(observed) == 0:
                raise ValueError("each observation must contain at least one segment")

            if self._max_segments is not None and len(observed) > self._max_segments:
                self._model = None
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

            overlap = [
                Segment(
                    max_suspension_time=max(
                        segment.max_suspension_time, observation[0]
                    ),
                    max_execution_time=max(segment.max_execution_time, observation[1]),
                )
                for segment, observation in zip(self._model, observed)
            ]
            additional = [
                Segment(*observation) for observation in observed[len(overlap) :]
            ]
            tail = self._model[len(overlap) :]
            self._model = overlap + additional + tail

    def __call__(self, observations: Iterable[SegmentedJobSuspensionBehavior]) -> None:
        self.feed(observations)

    @property
    def current_model(self) -> SegmentedSuspensionModel | None:
        "Query the model extracted so far."
        return None if self._model is None else list(self._model)

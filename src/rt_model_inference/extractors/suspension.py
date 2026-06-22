"""Continuously updating streaming extractors for basic self-suspension models."""

from collections.abc import Iterable

from rt_model_inference.extractors.resource_use import (
    MaxResourceUseExtractor,
    MinResourceUseExtractor,
)
from rt_model_inference.suspension import (
    BasicJobSuspensionBehavior,
    BasicSuspensionModel,
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

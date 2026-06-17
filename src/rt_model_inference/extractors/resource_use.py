"""Continuously updating streaming extractors for resource-use models."""

from collections import deque
from collections.abc import Iterable

from rt_model_inference.resource_use import ResourceAmount


class MaxResourceUseExtractor:
    "Streaming extractor for maximum observed resource-use vectors such as WCET(n)."

    def __init__(self, nmax: int | None = None):
        if nmax is not None and nmax <= 0:
            raise ValueError("nmax must be positive")
        self._moru: list[ResourceAmount] = [0]
        self._buffer: deque[ResourceAmount] = deque(maxlen=nmax)

    def feed(self, observations: Iterable[ResourceAmount]) -> None:
        "Update the current model estimate based on newly observed resource use."
        for observation_i in observations:
            self._buffer.append(observation_i)
            total = 0
            for n_observations, observation in enumerate(
                reversed(self._buffer), start=1
            ):
                total += observation
                if len(self._moru) == n_observations:
                    self._moru.append(total)
                else:
                    self._moru[n_observations] = max(self._moru[n_observations], total)

    def __call__(self, observations: Iterable[ResourceAmount]) -> None:
        self.feed(observations)

    @property
    def current_model(self) -> list[ResourceAmount]:
        "Query the model extracted so far."
        if len(self._buffer) == 0:
            # we didn't see any observations
            return []
        else:
            return list(self._moru)


class MinResourceUseExtractor:
    "Streaming extractor for minimum observed resource-use vectors such as BCET(n)."

    def __init__(self, nmax: int | None = None):
        if nmax is not None and nmax <= 0:
            raise ValueError("nmax must be positive")
        self._moru: list[ResourceAmount] = [0]
        self._buffer: deque[ResourceAmount] = deque(maxlen=nmax)

    def feed(self, observations: Iterable[ResourceAmount]) -> None:
        "Update the current model estimate based on newly observed resource use."
        for observation_i in observations:
            self._buffer.append(observation_i)
            total = 0
            for n_observations, observation in enumerate(
                reversed(self._buffer), start=1
            ):
                total += observation
                if len(self._moru) == n_observations:
                    self._moru.append(total)
                else:
                    self._moru[n_observations] = min(self._moru[n_observations], total)

    def __call__(self, observations: Iterable[ResourceAmount]) -> None:
        self.feed(observations)

    @property
    def current_model(self) -> list[ResourceAmount]:
        "Query the model extracted so far."
        if len(self._buffer) == 0:
            # we didn't see any observations
            return []
        else:
            return list(self._moru)

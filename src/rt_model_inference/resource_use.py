"""Inference of resource-use models."""

from collections import deque
from collections.abc import Iterable

ResourceAmount = int


def infer_max_resource_use(
    observations: Iterable[ResourceAmount], nmax: int | None = None
) -> list[ResourceAmount]:
    """Given a sequence of per-job resource-consumption observations (such as
    the processor time consumed by each job), infer a model upper-bounding the
    maximum joint observed resource consumption of consecutive jobs.

    Returns a vector `v` such that `v[n]` indicates the maximum cumulative
    resource use of `n` consecutive observations.

    If `nmax` is provided, a vector of length at most `nmax + 1` is returned
    (`nmax` must be at least 1 if specified).
    """

    if nmax is not None and nmax <= 0:
        raise ValueError("nmax must be positive")

    # moru <=> "maximum observed resource use"
    moru: list[ResourceAmount] = [0]
    buffer: deque[ResourceAmount] = deque(maxlen=nmax)

    for observation_i in observations:
        if observation_i < 0:
            raise ValueError("resource-use observations must be non-negative")
        buffer.append(observation_i)
        total = 0
        for n_observations, observation in enumerate(reversed(buffer), start=1):
            total += observation
            if len(moru) == n_observations:
                moru.append(total)
            else:
                moru[n_observations] = max(moru[n_observations], total)

    if len(buffer) == 0:
        # we didn't see any observations
        return []
    else:
        return moru


def infer_min_resource_use(
    observations: Iterable[ResourceAmount], nmax: int | None = None
) -> list[ResourceAmount]:
    """Given a sequence of per-job resource-consumption observations (such as
    the processor time consumed by each job), infer a model tracking the
    minimum joint observed resource consumption of consecutive jobs.

    Returns a vector `v` such that `v[n]` indicates the minimum cumulative
    resource use of `n` consecutive observations.

    If `nmax` is provided, a vector of length at most `nmax + 1` is returned
    (`nmax` must be at least 1 if specified).
    """

    if nmax is not None and nmax <= 0:
        raise ValueError("nmax must be positive")

    # moru <=> "minimum observed resource use"
    moru: list[ResourceAmount] = [0]
    buffer: deque[ResourceAmount] = deque(maxlen=nmax)

    for observation_i in observations:
        if observation_i < 0:
            raise ValueError("resource-use observations must be non-negative")
        buffer.append(observation_i)
        total = 0
        for n_observations, observation in enumerate(reversed(buffer), start=1):
            total += observation
            if len(moru) == n_observations:
                moru.append(total)
            else:
                moru[n_observations] = min(moru[n_observations], total)

    if len(buffer) == 0:
        # we didn't see any observations
        return []
    else:
        return moru

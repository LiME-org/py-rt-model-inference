from collections.abc import Iterable
from itertools import tee
from typing import NamedTuple, TypeAlias

from rt_model_inference.resource_use import (
    infer_max_resource_use,
    infer_min_resource_use,
)
from rt_model_inference.time import Duration


class BasicSuspensionModel(NamedTuple):
    number_of_suspensions: int
    cumulative_duration: Duration


# Every job is characterized by
# (1) the number of times it self-suspended, and
# (2) the total duration of self-suspension.
BasicJobSuspensionBehavior: TypeAlias = tuple[int, Duration]


def infer_max_suspension_model(
    observations: Iterable[BasicJobSuspensionBehavior], nmax: int | None = None
) -> list[BasicSuspensionModel]:
    """Given a sequence of per-job suspension-behavior observations, infer a
    model upper-bounding the joint suspension behavior of consecutive jobs.

    Returns a vector `v` such that `v[n]` indicates the component-wise maximum
    suspension count and cumulative self-duration of `n` jobs.

    If `nmax` is provided, a vector of length at most `nmax + 1` is returned
    (`nmax` must be at least 1 if specified).
    """
    # Internally, we reuse the existing resource-use facilities by interpreting
    # "suspension count" and "suspension time" as two kinds of resources.
    obs1, obs2 = tee(observations, 2)
    counts = infer_max_resource_use((o[0] for o in obs1), nmax=nmax)
    durations = infer_max_resource_use((o[1] for o in obs2), nmax=nmax)
    return [BasicSuspensionModel(c, d) for (c, d) in zip(counts, durations)]


def infer_min_suspension_model(
    observations: Iterable[BasicJobSuspensionBehavior], nmax: int | None = None
) -> list[BasicSuspensionModel]:
    """Given a sequence of per-job suspension-behavior observations, infer a
    model lower-bounding the joint suspension behavior of consecutive jobs.

    Returns a vector `v` such that `v[n]` indicates the component-wise minimum
    suspension count and cumulative self-suspension duration of `n` jobs.

    If `nmax` is provided, a vector of length at most `nmax + 1` is returned
    (`nmax` must be at least 1 if specified).
    """
    # Internally, we reuse the existing resource-use facilities by interpreting
    # "suspension count" and "suspension time" as two kinds of resources.
    obs1, obs2 = tee(observations, 2)
    counts = infer_min_resource_use((o[0] for o in obs1), nmax=nmax)
    durations = infer_min_resource_use((o[1] for o in obs2), nmax=nmax)
    return [BasicSuspensionModel(c, d) for (c, d) in zip(counts, durations)]

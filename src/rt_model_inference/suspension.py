from collections.abc import Iterable, Iterator
from itertools import permutations, tee, zip_longest
from typing import NamedTuple

from rt_model_inference.resource_use import (
    infer_max_resource_use,
    infer_min_resource_use,
)
from rt_model_inference.time import Duration

# Every job is characterized by
# (1) the number of times it self-suspended, and
# (2) the total duration of self-suspension.
BasicJobSuspensionBehavior = tuple[int, Duration]

# A job's execution can be understood as a list of observed segments ("suspension time", "execution time")
ObservedSegment = tuple[Duration, Duration]
SegmentedJobSuspensionBehavior = list[ObservedSegment]


def basic_from_segmented_observation(
    segments: SegmentedJobSuspensionBehavior,
) -> BasicJobSuspensionBehavior:
    "Convert a detailed observation to the corresponding basic observation of job suspension behavior."
    if segments:
        # special case: first segment has nonzero self-suspension time <=> release jitter
        # Release jitter counts against the total self-suspension time, but is not counted
        # as a _separate_ self-suspension because it occurs before the job starts execution.
        return (len(segments) - 1, sum(s[0] for s in segments))
    else:
        return (0, 0)


class BasicSuspensionModel(NamedTuple):
    number_of_suspensions: int
    cumulative_duration: Duration


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


class Segment(NamedTuple):
    """A suspension segment is a self-suspension of bounded maximum length followed by
    an execution segment of bounded maximum length.
    """

    max_suspension_time: Duration
    max_execution_time: Duration


# The segmented suspension model is simply a vector of segments.
SegmentedSuspensionModel = list[Segment]


def basic_from_segmented_model(
    model: SegmentedSuspensionModel,
) -> BasicSuspensionModel:
    "Convert a segmented suspension model to the corresponding basic model."
    if model:
        # special case: first segment has nonzero self-suspension time <=> release jitter
        # Release jitter counts against the total self-suspension time, but is not counted
        # as a _separate_ self-suspension because it occurs before the job starts execution.
        return BasicSuspensionModel(
            len(model) - 1, sum(s.max_suspension_time for s in model)
        )
    else:
        return BasicSuspensionModel(0, 0)


def infer_segmented_suspension_model(
    observations: Iterable[SegmentedJobSuspensionBehavior],
    max_segments: int | None = None,
) -> SegmentedSuspensionModel | None:
    """Given a sequence of per-job suspension segments, infer a single
    segmented suspension model that over-approximates the behavior of all
    observed jobs.

    If `max_segments` is provided, the returned model will include at most
    that many segments (`max_segments` must be at least 1 if specified). This
    allows the inference to run with strictly bounded memory footprint.

    If a job with more segments is encountered, `None` is returned to indicate
    that the input could not be safely over-approximated within the given
    `max_segments` bound.

    By default, `max_segments` is `None`, which means that the number of
    segments is unconstrained.
    """

    if max_segments is not None and max_segments <= 0:
        raise ValueError("max_segments must be positive")

    # The model inferred so far
    model: SegmentedSuspensionModel = []

    def merge(segments: SegmentedJobSuspensionBehavior) -> Iterator[Segment]:
        for ms, os in zip_longest(model, segments):
            if ms is not None and os is not None:
                yield Segment(
                    max_suspension_time=max(ms.max_suspension_time, os[0]),
                    max_execution_time=max(ms.max_execution_time, os[1]),
                )
            elif os is not None:
                yield Segment(max_suspension_time=os[0], max_execution_time=os[1])
            else:
                assert ms is not None
                yield ms

    for observed in observations:
        if len(observed) == 0:
            raise ValueError("each observation must contain at least one segment")

        if max_segments is not None and len(observed) > max_segments:
            # we cannot represent this observation; give up
            return None

        # sanity-check input
        for susp, exec in observed:
            if susp < 0:
                raise ValueError(
                    "segment suspension-time observations must be non-negative"
                )
            if exec < 0:
                raise ValueError(
                    "segment execution-time observations must be non-negative"
                )

        # update current model merging the observed segments
        model = list(merge(observed))

    return model


# The Bag of Segments (BOS) model due to Brandenburg et al. (2025) is a collection of
# segmented models.
BagOfSegmentsModel = list[SegmentedSuspensionModel]


def infer_bos_suspension_model(
    observations: Iterable[SegmentedJobSuspensionBehavior],
    max_segments: int | None = None,
    max_models: int | None = None,
) -> list[SegmentedSuspensionModel] | None:
    """Given a sequence of per-job suspension segments, infer a "bag" (i.e.,
    multiset) of segmented suspension models that jointly over-approximate
    the behavior of all observed jobs.

    See the LiME paper (Brandenburg et al., RTAS 2025) for a formal definition.

    If `max_segments` is provided, the returned model will include at most
    that many segments (`max_segments` must be at least 1 if specified).
    If a job with more segments is encountered, `None` is returned to indicate
    that the input could not be safely over-approximated within the given
    `max_segments` bound. By default, `max_segments` is `None`, which means
    that the number of segments is unconstrained.

    If `max_models` is provided, the returned multiset will not include more
    segmented models than specified (`max_models` must be at least 1 if
    specified).
    """

    if max_segments is not None and max_segments <= 0:
        raise ValueError("max_segments must be positive")

    if max_models is not None and max_models <= 0:
        raise ValueError("max_models must be positive")

    # The models inferred so far.
    bag: list[SegmentedSuspensionModel] = []

    # A counter of added models, for deterministic tie-breaking
    # when merging models. Used only if max_models is not None.
    model_count: int = 0
    model_ids: list[int] = []

    def add(model: SegmentedSuspensionModel):
        "Add a model to the bag, keeping track of the model ID if necessary."
        nonlocal model_count
        bag.append(model)
        if max_models is not None:
            model_ids.append(model_count)
            model_count += 1

    def remove(i: int, j: int):
        "Temove two models at positions `i` and `j` in the bag, respectively."
        i, j = max((i, j), (j, i))
        del bag[i]
        del model_ids[i]
        del bag[j]
        del model_ids[j]

    def cost(
        model: SegmentedSuspensionModel,
        segments: SegmentedJobSuspensionBehavior | SegmentedSuspensionModel,
    ) -> int:
        """Heuristic to estimate the cost of merging two models (or one
        observation into a model).

        NB: The cost heuristic has a bias towards increasing suspension length
        (rather than increasing execution costs). The rationale is that
        over-utilization typically has a worse impact on schedulability than
        pessimism in suspension lengths.
        """
        total: int = 0
        for ms, os in zip_longest(model, segments):
            if ms is not None and os is not None:
                # NB: We square the execution-time increase to bias the algorithm
                # towards favoring increases in the suspension bound.
                total += max(0, os[1] - ms.max_execution_time) ** 2
                total += max(0, os[0] - ms.max_suspension_time)
            elif os is not None:
                total += os[1] ** 2 + os[0]
        return total

    def merge(
        target: SegmentedSuspensionModel, source: SegmentedSuspensionModel
    ) -> Iterator[Segment]:
        "Merge two models such that the resulting model over-approximates both."
        for ts, ss in zip_longest(target, source):
            if ts is not None and ss is not None:
                yield Segment(max(ts[0], ss[0]), max(ts[1], ss[1]))
            elif ss is not None:
                yield ss
            else:
                assert ts is not None
                yield ts

    def consolidate():
        "Reduce the number of models in the bag by one via a least-cost merge."

        # First, let's find the two models that minimize the cost of merging.
        #
        # NB: we use the model ID to ensure deterministic tie-breaking, favoring newer
        # models over older ones.
        i, j = min(
            permutations(range(len(bag)), 2),
            key=lambda p: (
                cost(bag[p[0]], bag[p[1]]),
                -model_ids[p[0]],
                -model_ids[p[1]],
            ),
        )
        # Obtain the merged model.
        merged = list(merge(bag[i], bag[j]))
        # remove the two models that were merged into one
        remove(i, j)
        # add the merged model
        add(merged)

    # The main algorithm: add every observation not yet covered by an existing model
    # as a new model, and perform least-cost merges when we reach the bag-size limit.
    for observed in observations:
        if len(observed) == 0:
            raise ValueError("each observation must contain at least one segment")

        if max_segments is not None and len(observed) > max_segments:
            # we cannot represent this observation; give up
            return None

        # sanity-check input
        for susp, exec in observed:
            if susp < 0:
                raise ValueError(
                    "segment suspension-time observations must be non-negative"
                )
            if exec < 0:
                raise ValueError(
                    "segment execution-time observations must be non-negative"
                )

        # Create a model covering exactly the current observation.
        model = [
            Segment(max_suspension_time=o[0], max_execution_time=o[1]) for o in observed
        ]

        if 0 < min((cost(m, observed) for m in bag), default=1):
            # The best fit has cost greater than zero, which means that the
            # observation is not yet over-approximated. Thus, we add the new
            # model representing the current observation to the bag.
            add(model)

            # Check if we now violate the bag-size limit.
            if max_models is not None and len(bag) > max_models:
                # The tricky case: we are out of space in the bag, so we
                # need to consolidate.
                consolidate()

    return sorted(bag, key=lambda m: (len(m), m))

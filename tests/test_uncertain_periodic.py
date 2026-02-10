import random
from collections.abc import Iterable

import pytest

from rt_model_inference import infer_periodic_model
from rt_model_inference.extractors import (
    CertainFitPeriodicExtractor,
    PeriodicExtractor,
    PossibleFitPeriodicExtractor,
)
from rt_model_inference.iterators import monotonic
from rt_model_inference.time import Duration
from rt_model_inference.uncertain_periodic import (
    infer_certain_fit_periodic_model,
    infer_possible_fit_periodic_model,
)
from rt_model_inference.validate import model_covers_all, model_intersects_all


def as_exact_windows(releases: Iterable[int]) -> list[tuple[int, int]]:
    return [(release, release) for release in releases]


def random_windows_around_releases(
    releases: Iterable[int], seed: int | str, max_jitter: int
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    windows: list[tuple[int, int]] = []
    prev_lo: int | None = None
    prev_hi: int | None = None

    for rel in releases:
        lo = rel - rng.randint(0, max_jitter)
        lo = max(lo, prev_lo) if prev_lo is not None else lo
        hi = rel + rng.randint(0, max_jitter)
        hi = max(hi, prev_hi) if prev_hi is not None else hi
        windows.append((lo, hi))
        prev_lo = lo
        prev_hi = hi

    return windows


SEEDS = [
    "Anderson",
    "Baruah",
    "Jeffay",
    "Baker",
    "Rajkumar",
    "Burns",
    "Davis",
    "Buttazzo",
    "Di Natale",
    "Lipari",
]

UNCERTAINTY = [None, 17, 93, 501]


@pytest.mark.parametrize("seed", [*SEEDS])
@pytest.mark.parametrize("uncertainty", [*UNCERTAINTY])
def test_infer_uncertain_periodic_models_for_randomized_periodic_releases(
    seed: str, uncertainty: Duration | None
) -> None:
    rng = random.Random(seed)
    scale = 1000
    period = rng.randint(10 * scale, 1000 * scale)
    offset = rng.randint(0, 50 * scale)
    jitter = rng.randint(0, 3 * period)

    jitters = [rng.randint(0, jitter) for _ in range(2500)]
    # need at least one release with zero jitter in the trace
    jitters[rng.randrange(0, len(jitters))] = 0
    releases = list(monotonic((offset + i * period + j for i, j in enumerate(jitters))))

    if uncertainty is None:
        windows = as_exact_windows(releases)
    else:
        windows = list(
            random_windows_around_releases(
                releases, seed=f"{seed}:windows", max_jitter=scale * uncertainty
            )
        )

    cf_model = infer_certain_fit_periodic_model(windows, batch_size=512)
    pf_model = infer_possible_fit_periodic_model(windows, batch_size=768)

    assert model_covers_all(cf_model, windows)
    assert model_intersects_all(pf_model, windows)

    exact = infer_periodic_model(releases, batch_size=512)
    if uncertainty is None:
        assert exact == cf_model
        assert exact == pf_model
    else:
        assert exact.offset >= cf_model.offset
        assert exact.jitter <= cf_model.jitter
        assert exact.offset <= pf_model.offset
        assert exact.jitter >= pf_model.jitter

    ex_exact = PeriodicExtractor(batch_size=512)
    ex_exact.feed(releases)
    assert ex_exact.current_model == exact

    ex_cf = CertainFitPeriodicExtractor(batch_size=512)
    ex_cf.feed(windows)
    assert ex_cf.current_model == cf_model

    ex_pf = PossibleFitPeriodicExtractor(batch_size=768)
    ex_pf.feed(windows)
    assert ex_pf.current_model == pf_model


@pytest.mark.parametrize("seed", [*SEEDS])
@pytest.mark.parametrize("uncertainty", [*UNCERTAINTY])
def test_recover_well_known_periods_despite_uncertainty(
    seed: str, uncertainty: Duration | None
) -> None:
    rng = random.Random(seed)
    scale = 1000
    period = scale * rng.choice([1, 5, 10, 25, 50, 100, 125, 250, 500, 1000])
    offset = rng.randint(0, 50 * scale)
    jitter = rng.randint(0, 3 * period)

    jitters = [rng.randint(0, jitter) for _ in range(2250)]
    # need at least one release with zero jitter in the trace
    jitters[rng.randrange(0, len(jitters))] = 0
    releases = list(monotonic((offset + i * period + j for i, j in enumerate(jitters))))

    if uncertainty is None:
        windows = as_exact_windows(releases)
    else:
        windows = list(
            random_windows_around_releases(
                releases, seed=f"{seed}:windows", max_jitter=scale * uncertainty
            )
        )

    cf_model = infer_certain_fit_periodic_model(windows, batch_size=100)
    pf_model = infer_possible_fit_periodic_model(windows, batch_size=512)

    assert model_covers_all(cf_model, windows)
    assert model_intersects_all(pf_model, windows)
    assert cf_model.period == period
    assert pf_model.period == period

    exact = infer_periodic_model(releases, batch_size=100)
    if uncertainty is None:
        assert exact == cf_model
        assert exact == pf_model
    else:
        assert exact.offset >= cf_model.offset
        assert exact.period == cf_model.period
        assert exact.jitter <= cf_model.jitter
        assert exact.offset <= pf_model.offset
        assert exact.period == pf_model.period
        assert exact.jitter >= pf_model.jitter

    ex_exact = PeriodicExtractor(batch_size=100)
    ex_exact.feed(releases)
    assert ex_exact.current_model == exact

    ex_cf = CertainFitPeriodicExtractor(batch_size=100)
    ex_cf.feed(windows)
    assert ex_cf.current_model == cf_model

    ex_pf = PossibleFitPeriodicExtractor(batch_size=512)
    ex_pf.feed(windows)
    assert ex_pf.current_model == pf_model

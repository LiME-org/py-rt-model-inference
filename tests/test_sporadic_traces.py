import csv
import os
from itertools import batched
from pathlib import Path

import pytest

from rt_model_inference import (
    infer_certain_fit_periodic_model,
    infer_delta_max,
    infer_delta_max_hi,
    infer_delta_max_lo,
    infer_delta_min,
    infer_delta_min_hi,
    infer_delta_min_lo,
    infer_periodic_model,
    infer_possible_fit_periodic_model,
)
from rt_model_inference.extractors import (
    CertainFitPeriodicExtractor,
    DeltaMaxExtractor,
    DeltaMaxHiExtractor,
    DeltaMaxLoExtractor,
    DeltaMinExtractor,
    DeltaMinHiExtractor,
    DeltaMinLoExtractor,
    PeriodicExtractor,
    PossibleFitPeriodicExtractor,
)
from rt_model_inference.time import Instant, ReleaseWindow
from rt_model_inference.validate import (
    dmax_curve_lower_bounds_releases,
    dmin_curve_upper_bounds_releases,
    model_covers_all,
    model_intersects_all,
    model_is_conservative,
)

DEFAULT_TRACE_DIR = Path(__file__).parent / "traces" / "sporadic"
TRACE_DIR = Path(os.getenv("SPORADIC_TRACES", default=DEFAULT_TRACE_DIR))
ALL_TRACE_FILE_NAMES = sorted(trace_file.name for trace_file in TRACE_DIR.glob("*.csv"))
TRACE_FILE_NAMES = (
    ALL_TRACE_FILE_NAMES if os.getenv("TEST_ALL_TRACES") else ALL_TRACE_FILE_NAMES[:50]
)


def load_release_trace(trace_file: Path) -> list[tuple[Instant, ReleaseWindow]]:
    with trace_file.open() as trace_stream:
        return [
            (int(row[0]), (int(row[1]), int(row[2])))
            for row in csv.reader(trace_stream)
        ]


SPORADIC_CUTOFF = None if os.getenv("TEST_ALL_TRACES") else 250


@pytest.mark.parametrize("trace_file_name", TRACE_FILE_NAMES)
def test_sporadic_model_inference_on_sporadic_trace_file(
    trace_file_name: str, nmax: int = 5, cutoff: int | None = SPORADIC_CUTOFF
):
    releases_and_windows = load_release_trace(TRACE_DIR / trace_file_name)
    if cutoff is not None:
        releases_and_windows = releases_and_windows[:cutoff]
    releases = list((r for r, _ in releases_and_windows))
    windows = list((w for _, w in releases_and_windows))

    dmin = infer_delta_min(releases, nmax=nmax)
    dmax = infer_delta_max(releases, nmax=nmax)

    assert dmin_curve_upper_bounds_releases(dmin, releases)
    assert dmax_curve_lower_bounds_releases(dmax, releases)

    dmin_hi = infer_delta_min_hi(windows, nmax=nmax)
    dmin_lo = infer_delta_min_lo(windows, nmax=nmax)
    dmax_hi = infer_delta_max_hi(windows, nmax=nmax)
    dmax_lo = infer_delta_max_lo(windows, nmax=nmax)

    assert len(dmin) == len(dmin_hi) == len(dmin_lo)
    assert len(dmax) == len(dmax_hi) == len(dmax_lo)

    assert dmin_curve_upper_bounds_releases(dmin, releases)
    assert dmin_curve_upper_bounds_releases(dmin_hi, releases)
    assert dmax_curve_lower_bounds_releases(dmax, releases)
    assert dmax_curve_lower_bounds_releases(dmax_lo, releases)

    for x_lo, x, x_hi in zip(dmin_lo, dmin, dmin_hi):
        assert x_lo >= x >= x_hi

    for x_lo, x, x_hi in zip(dmax_lo, dmax, dmax_hi):
        assert x_hi <= x <= x_lo

    ex_dmin = DeltaMinExtractor(nmax=nmax)
    ex_dmin.feed(releases)
    assert ex_dmin.current_model == dmin

    ex_dmax = DeltaMaxExtractor(nmax=nmax)
    ex_dmax.feed(releases)
    assert ex_dmax.current_model == dmax

    ex_dmin_hi = DeltaMinHiExtractor(nmax=nmax)
    ex_dmin_hi.feed(windows)
    assert ex_dmin_hi.current_model == dmin_hi

    ex_dmax_lo = DeltaMaxLoExtractor(nmax=nmax)
    ex_dmax_lo.feed(windows)
    assert ex_dmax_lo.current_model == dmax_lo

    ex_dmin_lo = DeltaMinLoExtractor(nmax=nmax)
    ex_dmin_lo.feed(windows)
    assert ex_dmin_lo.current_model == dmin_lo

    ex_dmax_hi = DeltaMaxHiExtractor(nmax=nmax)
    ex_dmax_hi.feed(windows)
    assert ex_dmax_hi.current_model == dmax_hi


# allow access to "private" fields in the test below
# pyright: reportPrivateUsage=false


@pytest.mark.parametrize("trace_file_name", TRACE_FILE_NAMES)
def test_periodic_model_inference_on_sporadic_trace_file(trace_file_name: str):

    # As these traces are not periodic, this tests mainly model divergence.
    releases_and_windows = load_release_trace(TRACE_DIR / trace_file_name)
    releases = list((r for r, _ in releases_and_windows))
    windows = list((w for _, w in releases_and_windows))

    BS = 128
    exact = infer_periodic_model(releases, batch_size=BS)
    assert model_is_conservative(exact, releases)

    cf_model = infer_certain_fit_periodic_model(windows, batch_size=BS)
    assert model_covers_all(cf_model, windows)

    pf_model = infer_possible_fit_periodic_model(windows, batch_size=BS)
    assert model_intersects_all(pf_model, windows)

    ex_exact = PeriodicExtractor(batch_size=BS)
    mc_count = None
    for batch in batched(releases, BS // 2):
        ex_exact.feed(batch)
        # ensure model candidate count is monotonic
        assert mc_count is None or len(ex_exact._candidates) <= mc_count
        if mc_count is not None or ex_exact._candidates:
            mc_count = len(ex_exact._candidates)
    assert ex_exact.current_model == exact

    ex_cf = CertainFitPeriodicExtractor(batch_size=BS)
    mc_count = None
    for batch in batched(windows, BS // 2):
        ex_cf.feed(batch)
        # ensure model candidate count is monotonic
        assert mc_count is None or len(ex_cf._candidates) <= mc_count
        if mc_count is not None or ex_cf._candidates:
            mc_count = len(ex_cf._candidates)

    assert ex_cf.current_model == cf_model

    ex_pf = PossibleFitPeriodicExtractor(batch_size=BS)
    mc_count = None
    for batch in batched(windows, BS // 2):
        ex_pf.feed(batch)
        # ensure model candidate count is monotonic
        assert mc_count is None or len(ex_pf._candidates) <= mc_count
        if mc_count is not None or ex_pf._candidates:
            mc_count = len(ex_pf._candidates)
    assert ex_pf.current_model == pf_model

import csv
import os
import re
from pathlib import Path

import pytest

from rt_model_inference import (
    infer_certain_fit_periodic_model,
    infer_delta_max,
    infer_delta_max_lo,
    infer_delta_min,
    infer_delta_min_hi,
    infer_possible_fit_periodic_model,
)
from rt_model_inference.certain_periodic import infer_periodic_model
from rt_model_inference.extractors import (
    CertainFitPeriodicExtractor,
    DeltaMaxExtractor,
    DeltaMaxLoExtractor,
    DeltaMinExtractor,
    DeltaMinHiExtractor,
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

DEFAULT_TRACE_DIR = Path(__file__).parent / "traces" / "periodic"
TRACE_DIR = Path(os.getenv("PERIODIC_TRACES", default=DEFAULT_TRACE_DIR))
ALL_TRACE_FILE_NAMES = sorted(trace_file.name for trace_file in TRACE_DIR.glob("*.csv"))
TRACE_FILE_NAMES = (
    ALL_TRACE_FILE_NAMES if os.getenv("TEST_ALL_TRACES") else ALL_TRACE_FILE_NAMES[:50]
)
PERIOD_FROM_NAME = re.compile(r"period=(\d+)")


def load_release_trace(trace_file: Path) -> list[tuple[Instant, ReleaseWindow]]:
    with trace_file.open() as trace_stream:
        return [
            (int(row[0]), (int(row[1]), int(row[2])))
            for row in csv.reader(trace_stream)
        ]


def expected_period(trace_file_name: str) -> int | None:
    if match := PERIOD_FROM_NAME.search(trace_file_name):
        return int(match.group(1))
    return None


@pytest.mark.parametrize("trace_file_name", TRACE_FILE_NAMES)
def test_periodic_model_inference_on_trace_file(trace_file_name: str):
    releases_and_windows = load_release_trace(TRACE_DIR / trace_file_name)
    releases = list((r for r, _ in releases_and_windows))
    windows = list((w for _, w in releases_and_windows))
    expected = expected_period(trace_file_name)

    one_millisecond = 1_000_000

    exact = infer_periodic_model(
        releases,
        batch_size=128,
        negligible_jitter_threshold=one_millisecond,
    )
    cf_model = infer_certain_fit_periodic_model(
        windows,
        batch_size=512,
        jitter_selection_threshold=1.65,
    )
    pf_model = infer_possible_fit_periodic_model(
        windows,
        batch_size=512,
        negligible_jitter_threshold=one_millisecond,
    )

    assert model_is_conservative(exact, releases)
    assert model_covers_all(cf_model, windows)
    assert model_intersects_all(pf_model, windows)

    if expected is not None:
        assert exact.period == expected

    assert exact.period == cf_model.period
    assert exact.offset >= cf_model.offset
    assert exact.jitter <= cf_model.jitter

    assert exact.period == pf_model.period
    assert exact.offset <= pf_model.offset
    assert exact.jitter >= pf_model.jitter

    ex_exact = PeriodicExtractor(
        batch_size=128,
        negligible_jitter_threshold=one_millisecond,
    )
    ex_exact.feed(releases)
    assert ex_exact.current_model == exact

    ex_cf = CertainFitPeriodicExtractor(
        batch_size=512,
        jitter_selection_threshold=1.65,
    )
    ex_cf.feed(windows)
    assert ex_cf.current_model == cf_model

    ex_pf = PossibleFitPeriodicExtractor(
        batch_size=512,
        negligible_jitter_threshold=one_millisecond,
    )
    ex_pf.feed(windows)
    assert ex_pf.current_model == pf_model


SPORADIC_CUTOFF = None if os.getenv("TEST_ALL_TRACES") else 250


@pytest.mark.parametrize("trace_file_name", TRACE_FILE_NAMES)
def test_sporadic_model_inference_on_periodic_trace_file(
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
    dmax_lo = infer_delta_max_lo(windows, nmax=nmax)

    assert len(dmin) == len(dmin_hi)
    assert len(dmax) == len(dmax_lo)

    assert dmin_curve_upper_bounds_releases(dmin_hi, releases)
    assert dmax_curve_lower_bounds_releases(dmax_lo, releases)

    for a, b in zip(dmin, dmin_hi):
        assert a >= b

    for a, b in zip(dmax, dmax_lo):
        assert a <= b

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

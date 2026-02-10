from rt_model_inference import (
    PeriodicModel,
    infer_certain_fit_periodic_model,
    infer_delta_max,
    infer_delta_max_hi,
    infer_delta_max_lo,
    infer_delta_min,
    infer_delta_min_hi,
    infer_delta_min_lo,
    infer_periodic_model,
    infer_possible_fit_periodic_model,
    infer_sporadic_model,
    max_releases,
    min_releases,
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
    SporadicExtractor,
)
from rt_model_inference.time import Duration, Instant, ReleaseWindow

RELEASES: list[Instant] = [
    135,
    249,
    354,
    473,
    526,
    657,
    729,
    823,
    935,
    1041,
    1144,
    1258,
    1368,
    1434,
    1534,
    1653,
    1753,
    1834,
    1944,
    2057,
]


def test_infer_periodic_model_example():
    pm = infer_periodic_model(RELEASES)

    assert isinstance(pm, PeriodicModel)
    assert pm.offset == 123
    assert pm.period == 100
    assert pm.jitter == 50

    ex = PeriodicExtractor()
    ex.feed(RELEASES[:3])
    cm = ex.current_model
    assert isinstance(cm, PeriodicModel)
    assert cm.offset == 134
    assert cm.period == 110
    assert cm.jitter == 5

    ex.feed(RELEASES[3:])
    cm = ex.current_model
    assert isinstance(cm, PeriodicModel)
    assert cm.offset == 123
    assert cm.period == 100
    assert cm.jitter == 50


def test_infer_sporadic_model_example():
    min_sep = infer_sporadic_model(RELEASES)

    assert isinstance(min_sep, Duration)
    assert min_sep == 53

    ex_sp = SporadicExtractor()
    ex_sp.feed(RELEASES)
    assert ex_sp.current_model == min_sep


def test_infer_delta_min_max_example():
    dmin = infer_delta_min(RELEASES, nmax=5)

    assert isinstance(dmin, list) and all(
        isinstance(dmin[i], Duration) for i in range(len(dmin))
    )
    assert dmin == [0, 1, 54, 167, 257, 351]
    assert max_releases(dmin, 100) == 2
    assert max_releases(dmin, 300) == 4

    dmax = infer_delta_max(RELEASES, nmax=5)

    assert isinstance(dmax, list) and all(
        isinstance(dmax[i], Duration) for i in range(len(dmax))
    )
    assert dmax == [130, 223, 337, 434, 544, 638]
    assert min_releases(dmax, 100) == 0
    assert min_releases(dmax, 300) == 2

    ex_min = DeltaMinExtractor(nmax=5)
    ex_min.feed(RELEASES)
    assert ex_min.current_model == dmin

    ex_max = DeltaMaxExtractor(nmax=5)
    ex_max.feed(RELEASES)
    assert ex_max.current_model == dmax


RELEASE_WINDOWS: list[ReleaseWindow] = [
    (117, 145),
    (242, 277),
    (332, 356),
    (454, 489),
    (505, 554),
    (642, 666),
    (728, 732),
    (818, 846),
    (933, 949),
    (1020, 1066),
    (1131, 1161),
    (1255, 1259),
    (1342, 1379),
    (1419, 1446),
    (1511, 1536),
    (1647, 1654),
    (1743, 1763),
    (1812, 1857),
    (1919, 1965),
    (2049, 2068),
]


def test_infer_uncertain_periodic_model_example():
    cf = infer_certain_fit_periodic_model(RELEASE_WINDOWS)

    assert isinstance(cf, PeriodicModel)
    assert cf.offset == 105
    assert cf.period == 100
    assert cf.jitter == 84

    pf = infer_possible_fit_periodic_model(RELEASE_WINDOWS)

    assert isinstance(pf, PeriodicModel)
    assert pf.offset == 132
    assert pf.period == 100
    assert pf.jitter == 23

    ex_cf = CertainFitPeriodicExtractor()
    ex_cf.feed(RELEASE_WINDOWS)
    assert ex_cf.current_model == cf

    ex_pf = PossibleFitPeriodicExtractor()
    ex_pf.feed(RELEASE_WINDOWS)
    assert ex_pf.current_model == pf


def test_infer_delta_min_max_lo_hi_example():
    dmin_hi = infer_delta_min_hi(RELEASE_WINDOWS, nmax=5)

    assert isinstance(dmin_hi, list) and all(
        isinstance(dmin_hi[i], Duration) for i in range(len(dmin_hi))
    )
    assert dmin_hi == [0, 1, 17, 133, 229, 330]
    assert max_releases(dmin_hi, 100) == 2
    assert max_releases(dmin_hi, 300) == 4

    dmin_lo = infer_delta_min_lo(RELEASE_WINDOWS, nmax=5)

    assert isinstance(dmin_lo, list) and all(
        isinstance(dmin_lo[i], Duration) for i in range(len(dmin_lo))
    )
    assert dmin_lo == [0, 1, 91, 192, 279, 393]
    assert max_releases(dmin_lo, 100) == 2
    assert max_releases(dmin_lo, 300) == 4

    dmax_lo = infer_delta_max_lo(RELEASE_WINDOWS, nmax=5)

    assert isinstance(dmax_lo, list) and all(
        isinstance(dmax_lo[i], Duration) for i in range(len(dmax_lo))
    )
    assert dmax_lo == [160, 255, 371, 453, 560, 655]
    assert min_releases(dmax_lo, 100) == 0
    assert min_releases(dmax_lo, 300) == 2

    dmax_hi = infer_delta_max_hi(RELEASE_WINDOWS, nmax=5)

    assert isinstance(dmax_hi, list) and all(
        isinstance(dmax_hi[i], Duration) for i in range(len(dmax_hi))
    )
    assert dmax_hi == [110, 206, 308, 408, 522, 609]
    assert min_releases(dmax_hi, 100) == 0
    assert min_releases(dmax_hi, 300) == 2

    ex_dmin_hi = DeltaMinHiExtractor(nmax=5)
    ex_dmin_hi.feed(RELEASE_WINDOWS)
    assert ex_dmin_hi.current_model == dmin_hi

    ex_dmin_lo = DeltaMinLoExtractor(nmax=5)
    ex_dmin_lo.feed(RELEASE_WINDOWS)
    assert ex_dmin_lo.current_model == dmin_lo

    ex_dmax_hi = DeltaMaxHiExtractor(nmax=5)
    ex_dmax_hi.feed(RELEASE_WINDOWS)
    assert ex_dmax_hi.current_model == dmax_hi

    ex_dmax_lo = DeltaMaxLoExtractor(nmax=5)
    ex_dmax_lo.feed(RELEASE_WINDOWS)
    assert ex_dmax_lo.current_model == dmax_lo

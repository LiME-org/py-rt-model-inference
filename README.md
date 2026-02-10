# Real-Time Task Model Inference

This package provides Python reference implementations of the real-time task model inference algorithms used by the [Linux Real-Time Model Extractor (LiME)](https://lime.mpi-sws.org).

## Implemented Algorithms

The library implements all arrival-model inference algorithms used by LiME.

For _certain_ (i.e., exact) release times:

- `infer_sporadic_model`: infer the _minimum separation_ parameter of the sporadic task model from a sequence of releases.
- `infer_delta_min`: infer _delta-min_ curves from a sequence of releases, to be used with `max_releases` to derive _upper_ bounds on the _maximum_ number of releases in any interval of a given length
- `infer_delta_max`: infer _delta-max_ curves from a sequence of releases, to be used with `min_releases` to derive _lower_ bounds on the _minimum_ number of releases in any interval of a given length
- `infer_periodic_model`: infer periodic _(offset, period, jitter)_ models from a sequence of releases.

For _uncertain_ release windows:

- `infer_delta_min_hi` / `infer_delta_min_lo`: infer under-/over-approximations of delta-min from a sequence of release windows (can be used with `max_releases`).
- `infer_delta_max_hi` / `infer_delta_max_lo`: under-/over-approximations of delta-max from a sequence of release windows (can be used with `min_releases`).
- `infer_certain_fit_periodic_model`: infer a periodic model from a sequence of release windows that _fully covers_ every release window.
- `infer_possible_fit_periodic_model`: infer a periodic model from a sequence of release windows that _intersects_ every release window.

The above-mentioned APIs are "one-shot" procedures, consuming all given input in one go and producing a model as a result. Additionally, the library provides _streaming extractor_ variants of all these algorithms, which consume input continuously and can be queried at any time to obtain the model inferred _so far_. The streaming extractor classes can be imported from `rt_model_inference.extractors` and are named as follows:

- `SporadicExtractor`
- `DeltaMinExtractor`, `DeltaMinHiExtractor`, and `DeltaMinLoExtractor`
- `DeltaMaxExtractor`, `DeltaMaxHiExtractor`, and `DeltaMaxLoExtractor`
- `PeriodicExtractor`, `CertainFitPeriodicExtractor`, and `PossibleFitPeriodicExtractor`

## Attribution

When using these algorithms for academic work, please cite the following two papers:

1. B. Brandenburg, C. Courtaud, F. Marković, and B. Ye, “LiME: The Linux Real-Time Task Model Extractor”, *Proceedings of the 31st IEEE Real-Time and Embedded Technology and Applications Symposium (RTAS 2025)*, May 2025.
2. B. Ye, F. Marković, and B. Brandenburg, “Framework-Agnostic Model Inference for Intra-Thread Real-Time Tasks“, *Proceedings of the 32nd IEEE Real-Time and Embedded Technology and Applications Symposium (RTAS 2026)*, May 2026.

## Usage

The algorithms assume a *discrete-time model*, so all time values passed to the library must be integers. The library provides `Instant` and `Duration` in `rt_model_inference.time`, which are both aliases of `int`.

```python
from rt_model_inference.time import Instant, Duration
```

### Arrival Model Inference from Exact Release Times

Consider a sequence of observed exact release times:

```python
RELEASES: list[Instant] = [135, 249, 354, 473, 526, 657, 729, 823, 935, 1041, 1144, 1258, 1368, 1434, 1534, 1653, 1753, 1834, 1944, 2057]
```

#### Periodic Models

A periodic model explaining all releases can be recovered with `infer_periodic_model()`:

```python
from rt_model_inference import infer_periodic_model, PeriodicModel

pm = infer_periodic_model(RELEASES)

assert isinstance(pm, PeriodicModel)
assert pm.offset == 123
assert pm.period == 100
assert pm.jitter == 50
```

The corresponding streaming extractor `PeriodicExtractor` can be given the input piecemeal and allows querying the model extracted so far:

```python
from rt_model_inference.extractors import PeriodicExtractor

ex = PeriodicExtractor()

# process the first 3 observations
ex.feed(RELEASES[:3])

# query the current model estimate based on the input seen so far
cm = ex.current_model
assert isinstance(cm, PeriodicModel)
assert cm.offset == 134
assert cm.period == 110
assert cm.jitter == 5

# consume the rest of the observations
ex.feed(RELEASES[3:])

# final model is the same as the one obtained by infer_periodic_model()
cm = ex.current_model
assert isinstance(cm, PeriodicModel)
assert cm.offset == 123
assert cm.period == 100
assert cm.jitter == 50
```

#### Sporadic Models

The classic sporadic task model's scalar minimum-release separation can be obtained with `infer_sporadic_model()`:

```python
from rt_model_inference import infer_sporadic_model

min_sep = infer_sporadic_model(RELEASES)

assert isinstance(min_sep, Duration)
assert min_sep == 53
```

The functions `infer_delta_min()` and `infer_delta_max()` characterize the given release sequence with arrival curves. Arrival curves are represented as delta-min and delta-max vectors (of type `list[Duration]`). Both `infer_delta_min()` and `infer_delta_max()` take an optional keyword parameter `nmax` controlling the length of the extracted delta-min/max prefix. 

```python
from rt_model_inference import infer_delta_min, infer_delta_max

dmin = infer_delta_min(RELEASES, nmax=5)
assert isinstance(dmin, list) and all(isinstance(dmin[i], Duration) for i in range(len(dmin)))
assert dmin == [0, 1, 54, 167, 257, 351]

dmax = infer_delta_max(RELEASES, nmax=5)
assert isinstance(dmax, list) and all(isinstance(dmax[i], Duration) for i in range(len(dmax)))
assert dmax == [130, 223, 337, 434, 544, 638]
```

The extracted delta-min and delta-max vectors can be interpreted with `max_releases()` and `min_releases()`, respectively. 

```python
from rt_model_inference import max_releases, min_releases

# In any interval of length 100, at least 0 and at most 2 releases are observed.
assert min_releases(dmax, 100) == 0
assert max_releases(dmin, 100) == 2

# In any interval of length 300, at least 2 and at most 4 releases are observed.
assert min_releases(dmax, 300) == 2
assert max_releases(dmin, 300) == 4
```

The corresponding streaming extractors are `SporadicExtractor`, `DeltaMinExtractor`, and `DeltaMaxExtractor`:

```python
from rt_model_inference.extractors import SporadicExtractor, DeltaMinExtractor, DeltaMaxExtractor

ex_sp = SporadicExtractor()
ex_sp.feed(RELEASES)
assert ex_sp.current_model == min_sep

ex_min = DeltaMinExtractor(nmax=5)
ex_min.feed(RELEASES)
assert ex_min.current_model == dmin

ex_max = DeltaMaxExtractor(nmax=5)
ex_max.feed(RELEASES)
assert ex_max.current_model == dmax
```

### Arrival Model Inference from Uncertain Release Windows

Suppose that instead of exact release-time observations, we only have the following release-window observations:

```python
from rt_model_inference.time import ReleaseWindow

RELEASE_WINDOWS: list[ReleaseWindow] = [(117, 145), (242, 277), (332, 356), (454, 489), (505, 554), (642, 666), (728, 732), (818, 846), (933, 949), (1020, 1066), (1131, 1161), (1255, 1259), (1342, 1379), (1419, 1446), (1511, 1536), (1647, 1654), (1743, 1763), (1812, 1857), (1919, 1965), (2049, 2068)]
```

#### Periodic Models

Certain-fit and possible-fit periodic models can be extracted with `infer_certain_fit_periodic_model()` and `infer_possible_fit_periodic_model()`, respectively:

```python
from rt_model_inference import infer_certain_fit_periodic_model, infer_possible_fit_periodic_model

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
```

The certain-fit model conservatively over-approximates the ground-truth model (lower offset, larger jitter), whereas the possible-fit under-approximates the ground-truth model (larger offset, lower jitter).

The corresponding streaming extractors are `CertainFitPeriodicExtractor` and `PossibleFitPeriodicExtractor`:

```python
from rt_model_inference.extractors import CertainFitPeriodicExtractor, PossibleFitPeriodicExtractor

ex_cf = CertainFitPeriodicExtractor()
ex_cf.feed(RELEASE_WINDOWS)
assert ex_cf.current_model == cf

ex_pf = PossibleFitPeriodicExtractor()
ex_pf.feed(RELEASE_WINDOWS)
assert ex_pf.current_model == pf
```

#### Sporadic Models

Over- and under-approximations of delta-min and delta-max vectors can be obtained with `infer_delta_min_hi()`, `infer_delta_min_lo()`, `infer_delta_max_hi()`, and `infer_delta_max_lo()`. For example:

```python
dmin_hi = infer_delta_min_hi(RELEASE_WINDOWS, nmax=5)

assert isinstance(dmin_hi, list) and all(isinstance(dmin_hi[i], Duration) for i in range(len(dmin_hi)))
assert dmin_hi == [0, 1, 17, 133, 229, 330]
assert max_releases(dmin_hi, 100) == 2
assert max_releases(dmin_hi, 300) == 4

dmax_lo = infer_delta_max_lo(RELEASE_WINDOWS, nmax=5)

assert isinstance(dmax_lo, list) and all(isinstance(dmax_lo[i], Duration) for i in range(len(dmax_lo)))
assert dmax_lo == [160, 255, 371, 453, 560, 655]
assert min_releases(dmax_lo, 100) == 0
assert min_releases(dmax_lo, 300) == 2
```

The extracted `delta-min-hi` vector safely over-approximates the ground-truth delta-min vector, and the extracted `delta-max-lo` vector safely under-approximates the ground-truth delta-max vector.


The corresponding streaming extractors are `DeltaMinHiExtractor`, `DeltaMinLoExtractor`, `DeltaMaxHiExtractor` and `DeltaMaxLoExtractor`. For example:

```python
ex_dmin_hi = DeltaMinHiExtractor(nmax=5)
ex_dmin_hi.feed(RELEASE_WINDOWS)
assert ex_dmin_hi.current_model == dmin_hi

ex_dmax_lo = DeltaMaxLoExtractor(nmax=5)
ex_dmax_lo.feed(RELEASE_WINDOWS)
assert ex_dmax_lo.current_model == dmax_lo
```

The above examples are all checked in `tests/test_examples.py`. See also the other unit tests for further usage examples.

## Development

We recommend using the [`uv` package manager for Python](https://docs.astral.sh/uv/).

### Quick Start

Install all dependencies with the `uv` package manager:

```
uv sync
```

### Running the Package's CLI

The library provides a small CLI, mainly for testing and demonstration purposes.

To run the CLI, use the standard module entry point:

```bash
uv run python -m rt_model_inference --help
```

The CLI reads release times from stdin by default (or from a file path argument):

- if there is one number per line, the value is interpreted as a (certain) release time;
- if there are two numbers per line, the two values are interpreted as defining an (uncertain) release window.

To exercise periodic inference with generated sample input:

```bash
seq 27 30 1000 | uv run python -m rt_model_inference -m periodic --json
```

To apply the CLI tool to one of the CSV files in `tests/traces/periodic` or `tests/traces/sporadic`, use `awk` to convert the input to the expected format.

With release windows:

```bash
awk -F, '{print $2 " " $3}'  tests/traces/periodic/rust-p-10task_u30_mu22_ms_v13_task=07_period=16000000.csv | uv run python -m rt_model_inference -m periodic-certain-fit --json
```

Exact release times:

```bash
awk -F, '{print $1}'  tests/traces/periodic/rust-p-10task_u30_mu22_ms_v13_task=07_period=16000000.csv | uv run python -m rt_model_inference -m periodic --json
```

Exercise the streaming extractors:

```bash
awk -F, '{print $2 " " $3}'  tests/traces/sporadic/cpp-ar-mixed_20task_u90_mu22_ms_v22_task=05.csv | uv run python -m rt_model_inference -m delta-min-hi --n-max 10 --stream 25
```

```bash
awk -F, '{print $2 " " $3}'  tests/traces/sporadic/cpp-ar-mixed_20task_u90_mu22_ms_v22_task=05.csv | uv run python -m rt_model_inference -m delta-max-lo --n-max 10 --stream 25
```

### Run the Tests

Run the standard tests:

```
uv run pytest -n auto
```

To run tests on all traces included in the repository, set the `TEST_ALL_TRACES` environment variable.

```
TEST_ALL_TRACES=1 uv run pytest -n auto
```


### Linter and Type Checking

This package uses type annotations and the `ruff` linter.

Run `ruff`:

```
uvx ruff check
```

Check all types:

```
uvx basedpyright
```

Both should report no issues.




## License

This library is free software and released under the MIT license.

## Feedback, Questions, Patches

Please use the [project's GitLab issue tracker](https://gitlab.mpi-sws.org/LiME/py-rt-model-inference/-/issues) or contact [Björn Brandenburg](https://people.mpi-sws.org/~bbb/).

A [Github mirror](https://github.com/LiME-org/py-rt-model-inference) is available for those who prefer Github.

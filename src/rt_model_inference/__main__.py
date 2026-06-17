"""Command-line interface for real-time model inference."""

import argparse
import json
import sys
from collections.abc import Iterable, Iterator, Sequence
from enum import StrEnum
from functools import partial
from itertools import batched
from typing import cast, no_type_check

from rt_model_inference.extractors import (
    CertainFitPeriodicExtractor,
    DeltaMaxExtractor,
    DeltaMaxHiExtractor,
    DeltaMaxLoExtractor,
    DeltaMinExtractor,
    DeltaMinHiExtractor,
    DeltaMinLoExtractor,
    MaxResourceUseExtractor,
    MinResourceUseExtractor,
    PeriodicExtractor,
    PossibleFitPeriodicExtractor,
    SporadicExtractor,
)

from . import (
    infer_certain_fit_periodic_model,
    infer_delta_max,
    infer_delta_max_hi,
    infer_delta_max_lo,
    infer_delta_min,
    infer_delta_min_hi,
    infer_delta_min_lo,
    infer_max_resource_use,
    infer_min_resource_use,
    infer_periodic_model,
    infer_possible_fit_periodic_model,
    infer_sporadic_model,
)
from .time import ReleaseWindow


class ModelName(StrEnum):
    DELTA_MIN = "delta-min"
    DELTA_MAX = "delta-max"
    DELTA_MIN_HI = "delta-min-hi"
    DELTA_MIN_LO = "delta-min-lo"
    DELTA_MAX_HI = "delta-max-hi"
    DELTA_MAX_LO = "delta-max-lo"
    SPORADIC = "sporadic"
    PERIODIC = "periodic"
    PERIODIC_CERTAIN_FIT = "periodic-certain-fit"
    PERIODIC_POSSIBLE_FIT = "periodic-possible-fit"
    MAX_RESOURCE_USE = "max-resource-use"
    MIN_RESOURCE_USE = "min-resource-use"


MODEL_NAMES: tuple[str, ...] = tuple(model.value for model in ModelName)


def range_checked_int(
    value: str, min: int | None = None, max: int | None = None
) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer value: {value!r}") from exc
    if min is not None and parsed < min:
        raise argparse.ArgumentTypeError(f"value must be at least {min}")
    if max is not None and parsed > max:
        raise argparse.ArgumentTypeError(f"value must be at most {max}")
    return parsed


def parse_cmdline(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m rt_model_inference",
        description="Infer a real-time arrival model from release times.",
    )
    _ = parser.add_argument(
        "-m",
        "--model",
        choices=MODEL_NAMES,
        default=None,
        help=(
            "Model to infer. Defaults to delta-min in release mode and "
            "max-resource-use in resource-use mode."
        ),
    )
    _ = parser.add_argument(
        "-r",
        "--resource-use",
        action="store_true",
        help=(
            "Infer a resource-use vector from one resource consumption amount "
            "per input line. Defaults to max-resource-use unless --model is "
            "min-resource-use."
        ),
    )
    _ = parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help=(
            "Input file path. Release modes expect one release time or release "
            "window per line; resource-use mode expects one resource amount per "
            "line. Use '-' or omit for stdin."
        ),
    )
    _ = parser.add_argument(
        "-n",
        "--n-max",
        type=partial(range_checked_int, min=1),
        default=None,
        help=("Maximum number of jobs (`nmax`) for vector-valued inference models."),
    )
    _ = parser.add_argument(
        "-s",
        "--stream",
        type=partial(range_checked_int, min=0),
        default=None,
        metavar="N",
        help=(
            "Use streaming extractors and emit the current model estimate every N "
            "samples (0 emits only the final estimate)."
        ),
    )
    _ = parser.add_argument(
        "--json",
        action="store_true",
        help="Emit inferred model in JSON format.",
    )
    return parser.parse_args(argv)


def parse_release_windows(lines: Iterable[str]) -> Iterator[ReleaseWindow]:
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped == "":
            continue
        parts = stripped.split()
        if len(parts) == 1:
            try:
                value = int(parts[0])
            except ValueError as exc:
                raise ValueError(
                    f"invalid release time at line {line_number}: {stripped!r}"
                ) from exc
            yield (value, value)
            continue
        if len(parts) != 2:
            raise ValueError(
                f"invalid release window at line {line_number}: {stripped!r}"
            )
        try:
            lo = int(parts[0])
            hi = int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"invalid release window at line {line_number}: {stripped!r}"
            ) from exc
        if lo > hi:
            raise ValueError(
                f"invalid release window at line {line_number}: lower bound exceeds upper bound"
            )
        yield (lo, hi)


def parse_resource_amounts(lines: Iterable[str]) -> Iterator[int]:
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            yield int(stripped)
        except ValueError as exc:
            raise ValueError(
                f"invalid resource amount at line {line_number}: {stripped!r}"
            ) from exc


def exact_release_times(
    model: ModelName, release_windows: Iterable[ReleaseWindow]
) -> Iterator[int]:
    for lo, hi in release_windows:
        if lo != hi:
            raise ValueError(
                f"inexact release windows are not supported by the {model} inference heuristic"
            )
        yield lo


@no_type_check
def vector_model(
    inference_algorithm,
    requires_exact_releases: bool,
    release_windows: Iterable[ReleaseWindow],
    model: ModelName,
    n_max: int | None,
    output_json: bool,
) -> str:
    if requires_exact_releases:
        vec = inference_algorithm(
            exact_release_times(model, release_windows), nmax=n_max
        )
    else:
        vec = inference_algorithm(release_windows, nmax=n_max)
    if output_json:
        return json.dumps({"model": model, "vector": vec})
    else:
        return " ".join(str(d) for d in vec)


@no_type_check
def periodic_model(
    inference_algorithm,
    requires_exact_releases: bool,
    release_windows: Iterable[ReleaseWindow],
    model: ModelName,
    _n_max: int | None,
    output_json: bool,
) -> str:
    if requires_exact_releases:
        pm = inference_algorithm(exact_release_times(model, release_windows))
    else:
        pm = inference_algorithm(release_windows)
    if output_json:
        return json.dumps(
            {
                "model": model,
                "offset": pm.offset,
                "period": pm.period,
                "maximum jitter": pm.jitter,
            }
        )
    else:
        return f"{pm.offset} {pm.period} {pm.jitter}"


def sporadic_model(
    release_windows: Iterable[ReleaseWindow],
    model: ModelName,
    _n_max: int | None,
    output_json: bool,
) -> str:
    releases = exact_release_times(model, release_windows)
    minimum_separation = infer_sporadic_model(releases)
    if output_json:
        return json.dumps(
            {
                "model": model,
                "minimum separation": minimum_separation,
            }
        )
    else:
        return str(minimum_separation)


def resource_use_model(
    observations: Iterable[int],
    model: ModelName,
    n_max: int | None,
    output_json: bool,
) -> str:
    if model == ModelName.MAX_RESOURCE_USE:
        model_name = "maximum cumulative resource use"
        vec = infer_max_resource_use(observations, nmax=n_max)
    elif model == ModelName.MIN_RESOURCE_USE:
        model_name = "minimum cumulative resource use"
        vec = infer_min_resource_use(observations, nmax=n_max)
    else:
        raise ValueError(f"not a resource-use model: {model}")
    if output_json:
        return json.dumps({"model": model_name, "vector": vec})
    else:
        return " ".join(str(resource_amount) for resource_amount in vec)


INFERENCE_ALGORITHMS = {
    ModelName.DELTA_MIN: partial(vector_model, infer_delta_min, True),
    ModelName.DELTA_MAX: partial(vector_model, infer_delta_max, True),
    ModelName.DELTA_MIN_HI: partial(vector_model, infer_delta_min_hi, False),
    ModelName.DELTA_MIN_LO: partial(vector_model, infer_delta_min_lo, False),
    ModelName.DELTA_MAX_HI: partial(vector_model, infer_delta_max_hi, False),
    ModelName.DELTA_MAX_LO: partial(vector_model, infer_delta_max_lo, False),
    ModelName.SPORADIC: sporadic_model,
    ModelName.PERIODIC: partial(periodic_model, infer_periodic_model, True),
    ModelName.PERIODIC_CERTAIN_FIT: partial(
        periodic_model, infer_certain_fit_periodic_model, False
    ),
    ModelName.PERIODIC_POSSIBLE_FIT: partial(
        periodic_model, infer_possible_fit_periodic_model, False
    ),
}


@no_type_check
def streaming_vector_model(
    extractor,
    requires_exact_releases: bool,
    release_windows: Iterable[ReleaseWindow],
    model: ModelName,
    n_max: int | None,
    output_json: bool,
    output_every: int,
) -> Iterator[str]:
    if requires_exact_releases:
        input = exact_release_times(model, release_windows)
    else:
        input = release_windows

    @no_type_check
    def as_str(vec):
        if output_json:
            return json.dumps({"model": model, "vector": vec})
        else:
            return " ".join(str(d) for d in vec)

    ex = extractor(nmax=n_max)
    first = True
    if output_json:
        yield "["

    if output_every > 0:
        for batch in batched(input, output_every):
            ex.feed(batch)
            if output_json:
                yield f"  {', ' if not first else ''}{as_str(ex.current_model)}"
                first = False
            else:
                yield as_str(ex.current_model)
    else:
        ex.feed(input)
        if output_json:
            yield f"  {as_str(ex.current_model)}"
        else:
            yield as_str(ex.current_model)

    if output_json:
        yield "]"


@no_type_check
def streaming_periodic_model(
    extractor,
    requires_exact_releases: bool,
    release_windows: Iterable[ReleaseWindow],
    model: ModelName,
    _n_max: int | None,
    output_json: bool,
    output_every: int,
) -> Iterator[str]:
    if requires_exact_releases:
        input = exact_release_times(model, release_windows)
    else:
        input = release_windows

    @no_type_check
    def as_str(pm):
        if output_json:
            return json.dumps(
                {
                    "model": model,
                    "offset": pm.offset,
                    "period": pm.period,
                    "maximum jitter": pm.jitter,
                }
            )
        else:
            return f"{pm.offset} {pm.period} {pm.jitter}"

    ex = extractor()
    first = True
    if output_json:
        yield "["

    if output_every > 0:
        for batch in batched(input, output_every):
            ex.feed(batch)
            if output_json:
                yield f"  {', ' if not first else ''}{as_str(ex.current_model)}"
                first = False
            else:
                yield as_str(ex.current_model)
    else:
        ex.feed(input)
        if output_json:
            yield f"  {as_str(ex.current_model)}"
        else:
            yield as_str(ex.current_model)

    if output_json:
        yield "]"


def streaming_sporadic_model(
    release_windows: Iterable[ReleaseWindow],
    model: ModelName,
    _n_max: int | None,
    output_json: bool,
    output_every: int,
) -> Iterator[str]:
    releases = exact_release_times(model, release_windows)
    ex = SporadicExtractor()
    first = True
    if output_json:
        yield "["

    @no_type_check
    def as_str(min_sep):
        if output_json:
            return json.dumps(
                {
                    "model": model,
                    "minimum separation": min_sep,
                }
            )
        else:
            return str(min_sep)

    if output_every > 0:
        for batch in batched(releases, output_every):
            ex.feed(batch)
            if output_json:
                yield f"  {', ' if not first else ''}{as_str(ex.current_model)}"
                first = False
            else:
                yield as_str(ex.current_model)
    else:
        ex.feed(releases)
        if output_json:
            yield f"  {as_str(ex.current_model)}"
        else:
            yield as_str(ex.current_model)

    if output_json:
        yield "]"


def streaming_resource_use_model(
    observations: Iterable[int],
    model: ModelName,
    n_max: int | None,
    output_json: bool,
    output_every: int,
) -> Iterator[str]:
    if model == ModelName.MAX_RESOURCE_USE:
        model_name = "maximum cumulative resource use"
        extractor = MaxResourceUseExtractor
    elif model == ModelName.MIN_RESOURCE_USE:
        model_name = "minimum cumulative resource use"
        extractor = MinResourceUseExtractor
    else:
        raise ValueError(f"not a resource-use model: {model}")

    @no_type_check
    def as_str(vec):
        if output_json:
            return json.dumps({"model": model_name, "vector": vec})
        else:
            return " ".join(str(resource_amount) for resource_amount in vec)

    ex = extractor(nmax=n_max)
    first = True
    if output_json:
        yield "["

    if output_every > 0:
        for batch in batched(observations, output_every):
            ex.feed(batch)
            if output_json:
                yield f"  {', ' if not first else ''}{as_str(ex.current_model)}"
                first = False
            else:
                yield as_str(ex.current_model)
    else:
        ex.feed(observations)
        if output_json:
            yield f"  {as_str(ex.current_model)}"
        else:
            yield as_str(ex.current_model)

    if output_json:
        yield "]"


STREAMING_INFERENCE_ALGORITHMS = {
    ModelName.DELTA_MIN: partial(streaming_vector_model, DeltaMinExtractor, True),
    ModelName.DELTA_MAX: partial(streaming_vector_model, DeltaMaxExtractor, True),
    ModelName.DELTA_MIN_HI: partial(streaming_vector_model, DeltaMinHiExtractor, False),
    ModelName.DELTA_MIN_LO: partial(streaming_vector_model, DeltaMinLoExtractor, False),
    ModelName.DELTA_MAX_HI: partial(streaming_vector_model, DeltaMaxHiExtractor, False),
    ModelName.DELTA_MAX_LO: partial(streaming_vector_model, DeltaMaxLoExtractor, False),
    ModelName.SPORADIC: streaming_sporadic_model,
    ModelName.PERIODIC: partial(streaming_periodic_model, PeriodicExtractor, True),
    ModelName.PERIODIC_CERTAIN_FIT: partial(
        streaming_periodic_model, CertainFitPeriodicExtractor, False
    ),
    ModelName.PERIODIC_POSSIBLE_FIT: partial(
        streaming_periodic_model, PossibleFitPeriodicExtractor, False
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_cmdline(argv)
    resource_use_flag = cast(bool, args.resource_use)
    raw_model = cast(str | None, args.model)
    model = (
        ModelName(raw_model)
        if raw_model is not None
        else ModelName.MAX_RESOURCE_USE
        if resource_use_flag
        else ModelName.DELTA_MIN
    )
    resource_use = resource_use_flag or model in (
        ModelName.MAX_RESOURCE_USE,
        ModelName.MIN_RESOURCE_USE,
    )
    input_path = cast(str, args.input)
    n_max = cast(int | None, args.n_max)
    stream = cast(int | None, args.stream)
    output_json = cast(bool, args.json)

    input_stream = sys.stdin
    if input_path != "-":
        try:
            input_stream = open(input_path, "r", encoding="utf-8")
        except OSError as exc:
            print(
                f"error: unable to open input file {input_path!r}: {exc}",
                file=sys.stderr,
            )
            return 2

    try:
        if resource_use:
            observations = parse_resource_amounts(input_stream)
            if stream is not None:
                for output in streaming_resource_use_model(
                    observations,
                    model,
                    n_max,
                    output_json,
                    stream,
                ):
                    print(output)
            else:
                print(resource_use_model(observations, model, n_max, output_json))
        elif stream is not None:
            if model in STREAMING_INFERENCE_ALGORITHMS:
                alg = STREAMING_INFERENCE_ALGORITHMS[model]
                for output in alg(
                    parse_release_windows(input_stream),
                    model,
                    n_max,
                    output_json,
                    stream,
                ):
                    print(output)
            else:
                print(f"unsupported streaming model: {model}", file=sys.stderr)
                return 3
        else:
            if model in INFERENCE_ALGORITHMS:
                alg = INFERENCE_ALGORITHMS[model]
                print(
                    alg(parse_release_windows(input_stream), model, n_max, output_json)
                )
            else:
                print(f"unsupported model: {model}", file=sys.stderr)
                return 3
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

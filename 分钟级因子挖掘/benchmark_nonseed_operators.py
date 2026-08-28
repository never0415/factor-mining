"""Benchmark every used, non-zero-cost operator lacking a calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from min_gp.benchmark_seed_operators import (
    DEFAULT_SHAPE, _atomic_json, _environment, _parameter_profiles,
)
from min_gp.dsl import (
    OperatorNode, SemanticType, benchmark_callable, calibration_document,
    operator_spec_fingerprint,
)
from min_gp.factors.catalog import build_factor_catalog
from min_gp.operators import build_operator_registry


def _walk(node):
    yield node
    if isinstance(node, OperatorNode):
        for child in node.children:
            yield from _walk(child)


def benchmark_target_specs(registry):
    used = set()
    for entry in build_factor_catalog():
        root, _ = entry.genome.expression(registry)
        used.update(
            node.name for node in _walk(root)
            if isinstance(node, OperatorNode)
        )
    return tuple(
        registry.get(name) for name in sorted(used)
        if registry.get(name).cost > 0
        and not name.startswith("seed_")
        and (
            registry.get(name).calibration is None
            or registry.get(name).calibration.source
            == "nonseed_operator_batch_benchmark"
        )
    )


def remaining_target_specs(registry):
    """Stable set for alternatives plus parameter-specific old baselines.

    Parameterized in-source calibrations cover only one exact profile.  This
    batch measures every domain profile and persists the slowest one, making
    the resulting record safe for strict admission across the full domain.
    """
    return tuple(
        spec for spec in registry.specs()
        if spec.cost > 0 and not spec.name.startswith("seed_")
        and (
            spec.calibration is None
            or spec.calibration.source
            == "remaining_operator_batch_benchmark"
            or bool(spec.calibration.parameter_values)
        )
    )


def _random(shape, device, generator, *, positive=False):
    value = torch.rand(*shape, device=device, generator=generator)
    return value + 1 if positive else value * 2 - 1


def _input(kind, shape, device, generator):
    I, D, M = shape["I"], shape["D"], shape["M"]
    minute = (I, D, M)
    daily = (I, D)
    if kind in {
        SemanticType.MINUTE_OPEN, SemanticType.MINUTE_HIGH,
        SemanticType.MINUTE_LOW, SemanticType.MINUTE_CLOSE,
        SemanticType.MINUTE_PRICE, SemanticType.MINUTE_VOLUME,
        SemanticType.MINUTE_AMOUNT, SemanticType.MINUTE_HIGH_AMOUNT,
        SemanticType.MINUTE_LOW_AMOUNT,
    }:
        actual_shape = (
            daily if kind in {
                SemanticType.MINUTE_HIGH_AMOUNT,
                SemanticType.MINUTE_LOW_AMOUNT,
            } else minute
        )
        return _random(actual_shape, device, generator, positive=True)
    if kind in {
        SemanticType.MINUTE_RETURN, SemanticType.MINUTE_SIGNAL,
        SemanticType.MINUTE_ACTIVITY,
    }:
        return _random(minute, device, generator) * 0.02
    if kind in {
        SemanticType.MINUTE_AMOUNT_SHARE,
        SemanticType.MINUTE_VOLUME_SHARE,
    }:
        return _random(minute, device, generator, positive=True)
    if kind == SemanticType.MINUTE_PRICE_STATE:
        return torch.randint(0, 3, minute, device=device, generator=generator)
    if kind == SemanticType.MINUTE_MASK:
        return torch.rand(*minute, device=device, generator=generator) > 0.8
    if kind == SemanticType.SPECTRUM:
        return _random((I, D, M // 2 + 1), device, generator, positive=True)
    if kind in {
        SemanticType.DAILY_RAW_FACTOR, SemanticType.DAILY_PATH_SPEED,
        SemanticType.DAILY_ACTIVITY, SemanticType.DAILY_FACTOR,
        SemanticType.DAILY_RETURN,
    }:
        return _random(daily, device, generator)
    if kind in {
        SemanticType.DAILY_PRICE,
        SemanticType.DAILY_FLOAT_MARKET_CAP,
    }:
        return _random(daily, device, generator, positive=True)
    if kind == SemanticType.MARKET_DAILY_PRICE:
        return _random((D,), device, generator, positive=True)
    if kind == SemanticType.PAIR_SIMILARITY:
        return _random((I, I, D), device, generator)
    if kind == SemanticType.OLS_STATISTICS:
        return (
            _random((I, D, 7), device, generator),
            _random(daily, device, generator, positive=True),
        )
    if kind == SemanticType.MINUTE_INDEX:
        return torch.full((I, D), M // 2, device=device, dtype=torch.long)
    if kind == SemanticType.MINUTE_LEFT_INDEX:
        return torch.full((I, D), M // 4, device=device, dtype=torch.long)
    if kind == SemanticType.MINUTE_RIGHT_INDEX:
        return torch.full((I, D), 3 * M // 4, device=device, dtype=torch.long)
    raise TypeError(f"no non-seed calibration input for {kind.value}")


def _valid_profiles(spec):
    profiles = []
    for params in _parameter_profiles(spec):
        if (
            "period_low" in params and "period_high" in params
            and params["period_low"] >= params["period_high"]
        ):
            continue
        profiles.append(params)
    return tuple(profiles)


def benchmark_missing_specs(
    *, device, shape, warmup=1, repeats=3, progress=True,
    target="anchor",
):
    registry = build_operator_registry()
    if target == "anchor":
        specs = benchmark_target_specs(registry)
        source = "nonseed_operator_batch_benchmark"
    elif target == "remaining":
        specs = remaining_target_specs(registry)
        source = "remaining_operator_batch_benchmark"
    else:
        raise ValueError(f"unknown benchmark target {target!r}")
    records = {}
    generator = torch.Generator(device=device)
    generator.manual_seed(20260821)
    for number, spec in enumerate(specs, 1):
        inputs = tuple(
            _input(kind, shape, device, generator) for kind in spec.input_types
        )
        profiles = _valid_profiles(spec)
        slowest_seconds, largest_peak, slowest_params = -1.0, None, None
        try:
            for params in profiles:
                def invoke(current=dict(params)):
                    with torch.inference_mode():
                        return spec.implementation(*inputs, **current)

                measured = benchmark_callable(
                    invoke, shape, device=str(device), warmup=warmup,
                    repeats=repeats,
                    source=source,
                )
                if measured.seconds > slowest_seconds:
                    slowest_seconds = measured.seconds
                    slowest_params = dict(params)
                if measured.peak_bytes is not None:
                    largest_peak = max(largest_peak or 0, measured.peak_bytes)
            records[spec.name] = {
                "status": "ok",
                "spec_fingerprint": operator_spec_fingerprint(spec),
                "reference_shape": dict(shape),
                "seconds": slowest_seconds,
                "peak_bytes": largest_peak,
                "device": (
                    torch.cuda.get_device_name(device)
                    if str(device).startswith("cuda") else "CPU"
                ),
                "source": source,
                "profiles_tested": len(profiles),
                "slowest_parameters": slowest_params,
            }
        except (RuntimeError, ValueError, TypeError) as exc:
            records[spec.name] = {
                "status": "error",
                "spec_fingerprint": operator_spec_fingerprint(spec),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        finally:
            del inputs
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        if progress:
            record = records[spec.name]
            detail = (
                f"{record['seconds']:.6f}s"
                if record["status"] == "ok" else record["error_type"]
            )
            print(
                f"[nonseed-calibration {number:2d}/{len(specs)}] "
                f"{spec.name}: {detail}", flush=True,
            )
    return calibration_document(records, _environment(device))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument(
        "--target", choices=("anchor", "remaining"), default="anchor"
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--instruments", type=int, default=DEFAULT_SHAPE["I"])
    parser.add_argument("--days", type=int, default=DEFAULT_SHAPE["D"])
    parser.add_argument("--minutes", type=int, default=DEFAULT_SHAPE["M"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA calibration requested but CUDA is unavailable")
    document = benchmark_missing_specs(
        device=torch.device(args.device),
        shape={"I": args.instruments, "D": args.days, "M": args.minutes},
        warmup=args.warmup, repeats=args.repeats,
        target=args.target,
    )
    output = args.output or str(
        Path(__file__).resolve().parent / "calibrations" /
        (
            "nonseed_operators.json" if args.target == "anchor"
            else "remaining_operators.json"
        )
    )
    _atomic_json(output, document)
    failed = {
        name: record for name, record in document["operators"].items()
        if record["status"] != "ok"
    }
    print(
        f"wrote {len(document['operators']) - len(failed)}/"
        f"{len(document['operators'])} calibrations to "
        f"{Path(output).resolve()}", flush=True,
    )
    if failed:
        print(json.dumps(failed, ensure_ascii=False, indent=2), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Benchmark every physical ``seed_*`` OperatorSpec and persist the results."""

from __future__ import annotations

import argparse
from itertools import product
import json
import os
from pathlib import Path
import platform
import time

import torch

from min_gp.dsl import (
    SemanticType, benchmark_callable, calibration_document,
    operator_spec_fingerprint,
)
from min_gp.operators import build_operator_registry


DEFAULT_SHAPE = {"I": 24, "D": 60, "M": 240}


def _tensor(kind, shape, device, generator):
    I, D, M = shape["I"], shape["D"], shape["M"]
    if kind == SemanticType.LEGACY_MINUTE:
        return torch.randn(I, D, M, device=device, generator=generator)
    if kind == SemanticType.LEGACY_DAILY:
        return torch.randn(I, D, device=device, generator=generator)
    if kind == SemanticType.SAME_MINUTE_HISTORY:
        return torch.randn(I, M, D, device=device, generator=generator)
    if kind == SemanticType.SESSION_MASK:
        return torch.arange(M, device=device) % 2 == 0
    if kind == SemanticType.SCALAR:
        return 1
    raise TypeError(f"no calibration input generator for {kind.value}")


def _parameter_profiles(spec):
    if not spec.parameter_domains:
        return ({},)
    names = tuple(spec.parameter_domains)
    return tuple(
        dict(zip(names, values))
        for values in product(*(spec.parameter_domains[name] for name in names))
    )


def _environment(device):
    cuda = str(device).startswith("cuda")
    return {
        "created_at_unix": time.time(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if cuda else "CPU",
        "compute_capability": (
            list(torch.cuda.get_device_capability(device)) if cuda else None
        ),
        "dtype": "float32",
    }


def benchmark_seed_specs(
    *, device, shape, warmup=1, repeats=3, progress=True,
):
    registry = build_operator_registry()
    specs = [spec for spec in registry.specs() if spec.name.startswith("seed_")]
    records = {}
    generator = torch.Generator(device=device)
    generator.manual_seed(20260820)
    context = {
        "minute": torch.randn(
            shape["I"], shape["D"], shape["M"],
            device=device, generator=generator,
        )
    }
    for number, spec in enumerate(specs, 1):
        inputs = tuple(
            _tensor(kind, shape, device, generator) for kind in spec.input_types
        )
        best_seconds, best_peak, best_params = -1.0, None, None
        profiles = _parameter_profiles(spec)
        try:
            for params in profiles:
                def invoke(current=dict(params)):
                    kwargs = dict(current)
                    if spec.passes_context:
                        kwargs["_context"] = context
                    with torch.inference_mode():
                        return spec.implementation(*inputs, **kwargs)

                measured = benchmark_callable(
                    invoke, shape, device=str(device), warmup=warmup,
                    repeats=repeats, source="seed_operator_batch_benchmark",
                )
                if measured.seconds > best_seconds:
                    best_seconds = measured.seconds
                    best_params = dict(params)
                if measured.peak_bytes is not None:
                    best_peak = max(best_peak or 0, measured.peak_bytes)
            records[spec.name] = {
                "status": "ok",
                "spec_fingerprint": operator_spec_fingerprint(spec),
                "reference_shape": dict(shape),
                "seconds": best_seconds,
                "peak_bytes": best_peak,
                "device": (
                    torch.cuda.get_device_name(device)
                    if str(device).startswith("cuda") else "CPU"
                ),
                "source": "seed_operator_batch_benchmark",
                "profiles_tested": len(profiles),
                "slowest_parameters": best_params,
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
                f"[seed-calibration {number:3d}/{len(specs)}] "
                f"{spec.name}: {detail}", flush=True,
            )
    return calibration_document(records, _environment(device))


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=str(
            Path(__file__).resolve().parent
            / "calibrations" / "seed_operators.json"
        )
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
    shape = {"I": args.instruments, "D": args.days, "M": args.minutes}
    document = benchmark_seed_specs(
        device=torch.device(args.device), shape=shape,
        warmup=args.warmup, repeats=args.repeats,
    )
    _atomic_json(args.output, document)
    ok = sum(
        record["status"] == "ok"
        for record in document["operators"].values()
    )
    print(
        f"wrote {ok}/{len(document['operators'])} calibrations to "
        f"{Path(args.output).resolve()}", flush=True,
    )
    if ok != len(document["operators"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

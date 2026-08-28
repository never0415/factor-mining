"""Versioned persistence for measured operator calibrations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from min_gp.dsl.registry import CostCalibration, OperatorRegistry, OperatorSpec


SCHEMA_VERSION = 1


def operator_spec_fingerprint(spec: OperatorSpec) -> str:
    payload = {
        "name": spec.name,
        "inputs": [value.value for value in spec.input_types],
        "output": spec.output_type.value,
        "parameter_domains": {
            name: list(domain)
            for name, domain in sorted(spec.parameter_domains.items())
        },
        "complexity": dict(sorted(spec.complexity.items())),
        "memory_complexity": dict(sorted(spec.memory_complexity.items())),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def calibration_document(records, environment):
    return {
        "schema_version": SCHEMA_VERSION,
        "environment": dict(environment),
        "operators": dict(sorted(records.items())),
    }


def load_calibration_file(
    registry: OperatorRegistry, path: str | Path, *, strict: bool = True,
) -> tuple[str, ...]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported calibration schema: {document.get('schema_version')!r}"
        )
    loaded = []
    for name, record in document.get("operators", {}).items():
        try:
            spec = registry.get(name)
        except KeyError:
            if strict:
                raise
            continue
        expected = operator_spec_fingerprint(spec)
        if record.get("spec_fingerprint") != expected:
            if strict:
                raise ValueError(f"stale calibration for operator {name}")
            continue
        if record.get("status") != "ok":
            continue
        registry.apply_calibration(
            name,
            CostCalibration(
                reference_shape=record["reference_shape"],
                seconds=float(record["seconds"]),
                peak_bytes=record.get("peak_bytes"),
                device=record.get("device", "unknown"),
                source=record.get("source", str(path)),
                # The benchmark uses a conservative parameter profile and the
                # result is intentionally valid for every domain value.
                parameter_values={},
            ),
        )
        loaded.append(name)
    return tuple(sorted(loaded))

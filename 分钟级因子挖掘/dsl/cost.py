"""Measured operator-cost extrapolation and GP admission budgets."""

from dataclasses import dataclass
import math
import time
from typing import Callable, Mapping

import torch

from min_gp.dsl.node import ConstantNode, LeafNode, OperatorNode
from min_gp.dsl.registry import CostCalibration, OperatorRegistry
from min_gp.dsl.types import ExecutionScope


@dataclass(frozen=True)
class CostEstimate:
    cost_units: int
    calibrated_seconds: float
    calibrated_peak_bytes: int | None
    uncalibrated_operators: tuple[str, ...]
    unmeasured_peak_operators: tuple[str, ...]
    resident_input_bytes: int = 0
    uncalibrated_cost_units: int = 0
    # Work is charged once per node, but a date-chunked tree recomputes its
    # halo in every chunk. Without this the estimate misses that entirely: a
    # tree measured at 4.9s really took 93.8s behind a three-day chunk.
    halo_amplification: float = 1.0

    @property
    def fully_calibrated(self) -> bool:
        return not self.uncalibrated_operators

    @property
    def peak_fully_calibrated(self) -> bool:
        return not self.unmeasured_peak_operators

    @property
    def total_peak_bytes(self) -> int:
        """Resident input tensors plus calibrated incremental workspace."""
        return self.resident_input_bytes + (self.calibrated_peak_bytes or 0)


@dataclass(frozen=True)
class ExecutionBudget:
    """Hard admission limits checked before an expression is evaluated."""

    max_cost_units: int | None = None
    max_estimated_seconds: float | None = None
    max_peak_bytes: int | None = None
    max_uncalibrated_cost_units: int | None = None
    allow_uncalibrated: bool = True

    def accepts(self, estimate: CostEstimate) -> bool:
        if (
            self.max_cost_units is not None
            and estimate.cost_units > self.max_cost_units
        ):
            return False
        if (
            self.max_estimated_seconds is not None
            and estimate.calibrated_seconds > self.max_estimated_seconds
        ):
            return False
        if (
            self.max_peak_bytes is not None
            and estimate.total_peak_bytes > self.max_peak_bytes
        ):
            return False
        if (
            self.max_uncalibrated_cost_units is not None
            and estimate.uncalibrated_cost_units
            > self.max_uncalibrated_cost_units
        ):
            return False
        if not self.allow_uncalibrated:
            if (
                self.max_estimated_seconds is not None
                and not estimate.fully_calibrated
            ):
                return False
            if (
                self.max_peak_bytes is not None
                and not estimate.peak_fully_calibrated
            ):
                return False
        return True


def _axis_ratio(axis, target, reference):
    if axis.startswith("log"):
        base = axis[3:].lstrip("_()")
        base = base.rstrip(")")
        if base not in target or base not in reference:
            return 1.0
        numerator = math.log2(max(2, target[base]))
        denominator = math.log2(max(2, reference[base]))
        return numerator / denominator
    if axis not in target or axis not in reference:
        return 1.0
    return target[axis] / reference[axis]


def _scale(complexity, target_shape, reference_shape):
    scale = 1.0
    for axis, exponent in complexity.items():
        scale *= _axis_ratio(
            axis, target_shape, reference_shape
        ) ** exponent
    return scale


def extrapolate_calibration(
    calibration, complexity, target_shape, memory_complexity=None,
    memory_target_shape=None,
):
    seconds = calibration.seconds * _scale(
        complexity, target_shape, calibration.reference_shape
    )
    peak = (
        None if calibration.peak_bytes is None
        else max(1, round(calibration.peak_bytes * _scale(
            memory_complexity or complexity,
            memory_target_shape or target_shape,
            calibration.reference_shape,
        )))
    )
    return seconds, peak


def estimate_expression_cost(
    root: LeafNode | OperatorNode,
    registry: OperatorRegistry,
    target_shape: Mapping[str, int],
    peak_shape: Mapping[str, int] | None = None,
    element_budget: int | None = None,
) -> CostEstimate:
    """Sum runtime and take peak memory across unique nodes in a cached DAG.

    Runtime is then scaled by the halo redundancy the chunk planner would
    incur at this shape, so the number a budget sees is the number the run
    will actually pay.
    """
    units = 0
    seconds = 0.0
    peaks = []
    unknown = []
    unknown_peak = []
    unknown_units = 0
    seen = set()

    def visit(node):
        nonlocal units, seconds, unknown_units
        if isinstance(node, (LeafNode, ConstantNode)) or id(node) in seen:
            return
        seen.add(id(node))
        spec = registry.get(node.name)
        units += spec.cost
        if spec.calibration is None and spec.cost == 0:
            pass
        elif spec.calibration is None or not spec.calibration.matches(node.params):
            unknown.append(spec.name)
            unknown_peak.append(spec.name)
            unknown_units += spec.cost
        else:
            elapsed, peak = extrapolate_calibration(
                spec.calibration, spec.complexity, target_shape,
                spec.memory_complexity, peak_shape,
            )
            seconds += elapsed
            if peak is not None:
                peaks.append(peak)
            else:
                unknown_peak.append(spec.name)
        for child in node.children:
            visit(child)

    visit(root)
    amplification = 1.0
    if isinstance(root, OperatorNode):
        from min_gp.dsl.execution import DEFAULT_CHUNK_ELEMENTS, plan_chunks_for_shape
        try:
            plan = plan_chunks_for_shape(
                (root,), registry, target_shape,
                DEFAULT_CHUNK_ELEMENTS if element_budget is None else element_budget,
            )
            amplification = plan.amplification
        except (ValueError, KeyError):
            amplification = 1.0
    return CostEstimate(
        cost_units=units,
        calibrated_seconds=seconds * amplification,
        calibrated_peak_bytes=max(peaks) if peaks else None,
        uncalibrated_operators=tuple(sorted(set(unknown))),
        unmeasured_peak_operators=tuple(sorted(set(unknown_peak))),
        uncalibrated_cost_units=unknown_units,
        halo_amplification=amplification,
    )


def chunk_peak_shape(
    root: LeafNode | OperatorNode,
    registry: OperatorRegistry,
    target_shape: Mapping[str, int],
    chunk_rows: int,
) -> dict[str, int]:
    """Largest tensor grid resident under the scope-aware chunk planner."""
    shape = dict(target_shape)
    instruments, days = shape.get("I"), shape.get("D")
    if not instruments or not days or instruments * days <= chunk_rows:
        return shape
    if isinstance(root, LeafNode):
        shape["I"], shape["D"] = 1, min(chunk_rows, instruments * days)
        return shape
    scope = root.execution_scope_with(registry)
    if scope & (ExecutionScope.FULL_CROSS_SECTION | ExecutionScope.HISTORY):
        chunk_days = max(1, chunk_rows // instruments)
        history = root.history_days_with(registry)
        resident_days = days if history is None else chunk_days + history
        shape["D"] = min(days, resident_days)
        return shape
    shape["I"], shape["D"] = 1, min(chunk_rows, instruments * days)
    return shape


def benchmark_callable(
    function: Callable[[], object],
    reference_shape: Mapping[str, int],
    *,
    device: str,
    warmup: int = 1,
    repeats: int = 3,
    source: str = "local_benchmark",
    parameter_values: Mapping[str, object] | None = None,
) -> CostCalibration:
    """Measure a ready-to-run callable, including CUDA synchronization."""
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be >= 0 and repeats must be positive")
    uses_cuda = str(device).startswith("cuda")
    for _ in range(warmup):
        function()
    baseline_bytes = None
    if uses_cuda:
        torch.cuda.synchronize(device)
        baseline_bytes = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repeats):
        if uses_cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        function()
        if uses_cuda:
            torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - start)
    samples.sort()
    peak = (
        max(0, torch.cuda.max_memory_allocated(device) - baseline_bytes)
        if uses_cuda else None
    )
    return CostCalibration(
        reference_shape=dict(reference_shape),
        seconds=samples[len(samples) // 2],
        peak_bytes=peak,
        device=str(device),
        source=source,
        parameter_values=dict(parameter_values or {}),
    )

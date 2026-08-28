"""One cost-admission path shared by every production GP engine."""

from dataclasses import replace
import inspect

import torch

from min_gp.dsl import (
    CostEstimate, ExecutionBudget, chunk_peak_shape,
    estimate_expression_cost,
)


DEFAULT_MAX_COST_UNITS = 200
DEFAULT_MAX_ESTIMATED_SECONDS = 60.0
DEFAULT_MAX_UNCALIBRATED_COST_UNITS = 50


def execution_budget(config):
    return ExecutionBudget(
        max_cost_units=getattr(config, "max_cost_units", DEFAULT_MAX_COST_UNITS),
        max_estimated_seconds=getattr(
            config, "max_estimated_seconds", DEFAULT_MAX_ESTIMATED_SECONDS
        ),
        max_peak_bytes=getattr(config, "max_peak_bytes", None),
        max_uncalibrated_cost_units=getattr(
            config, "max_uncalibrated_cost_units",
            DEFAULT_MAX_UNCALIBRATED_COST_UNITS,
        ),
        allow_uncalibrated=getattr(config, "allow_uncalibrated_cost", False),
    )


def _target_shape(context, fwd_ret):
    instruments, days = fwd_ret.shape
    minutes = max((
        value.shape[-1] for value in context.values()
        if isinstance(value, torch.Tensor) and value.ndim >= 3
        and value.shape[:2] == (instruments, days)
    ), default=1)
    return {"I": instruments, "D": days, "M": minutes}


def _resident_cuda_bytes(context):
    seen, total = set(), 0
    for value in context.values():
        if not isinstance(value, torch.Tensor) or value.device.type != "cuda":
            continue
        storage = value.untyped_storage()
        identity = storage.data_ptr()
        if identity not in seen:
            seen.add(identity)
            total += storage.nbytes()
    return total


def estimate_genome_cost(genome, registry, context, fwd_ret, chunk_rows):
    """Estimate typed trees; expose legacy templates as explicitly unknown."""
    cost_expression = getattr(genome, "cost_expression", None)
    expression = cost_expression or getattr(genome, "expression", None)
    if expression is None or (
        cost_expression is None
        and len(inspect.signature(expression).parameters) == 0
    ):
        raw_complexity = getattr(genome, "complexity", 1)
        units = raw_complexity() if callable(raw_complexity) else raw_complexity
        name = f"legacy_template:{type(genome).__name__}"
        return CostEstimate(
            cost_units=int(units), calibrated_seconds=0.0,
            calibrated_peak_bytes=None,
            uncalibrated_operators=(name,),
            unmeasured_peak_operators=(name,),
            resident_input_bytes=_resident_cuda_bytes(context),
            uncalibrated_cost_units=int(units),
        )
    root, expression_registry = expression(registry)
    target = _target_shape(context, fwd_ret)
    pending = [root]
    lag_counts = []
    while pending:
        node = pending.pop()
        params = getattr(node, "params", {})
        if "lags" in params:
            lag_counts.append(int(params["lags"]) + 2)
        pending.extend(getattr(node, "children", ()))
    if lag_counts:
        target["P"] = max(lag_counts)
    estimate = estimate_expression_cost(
        root, expression_registry, target,
        peak_shape=chunk_peak_shape(
            root, expression_registry, target, chunk_rows
        ),
    )
    return replace(
        estimate, resident_input_bytes=_resident_cuda_bytes(context)
    )


class CostAdmission:
    """Small stateful facade used by GP evaluators and offspring loops."""

    def __init__(self, config, registry, context, fwd_ret):
        self.config = config
        self.registry = registry
        self.context = context
        self.fwd_ret = fwd_ret
        self.budget = execution_budget(config)

    def estimate(self, genome):
        return estimate_genome_cost(
            genome, self.registry, self.context, self.fwd_ret,
            getattr(self.config, "chunk_rows", 4096),
        )

    def accepts(self, genome):
        return self.budget.accepts(self.estimate(genome))

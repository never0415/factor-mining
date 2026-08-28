"""Scope-aware bounded evaluation for expressions producing daily factors."""

from collections.abc import Mapping
from typing import Any

import torch

from min_gp.dsl.node import ConstantNode, LeafNode, OperatorNode
from min_gp.dsl.registry import OperatorRegistry
from min_gp.dsl.types import ExecutionScope, SemanticType


def _leaves(node):
    if isinstance(node, ConstantNode):
        return
    if isinstance(node, LeafNode):
        yield node
        return
    for child in node.children:
        yield from _leaves(child)


def _unique_leaves(roots):
    seen = set()
    for root in roots:
        for leaf in _leaves(root):
            key = (leaf.name, leaf.output_type)
            if key not in seen:
                seen.add(key)
                yield leaf


def _grid_shape(roots, context: Mapping[str, Any]) -> tuple[int, int]:
    candidates = []
    for leaf in _unique_leaves(roots):
        value = context[leaf.name]
        if not isinstance(value, torch.Tensor):
            continue
        if leaf.output_type == SemanticType.PAIR_SIMILARITY:
            if value.ndim != 3 or value.shape[0] != value.shape[1]:
                raise ValueError(f"{leaf.name} must be (I,I,D)")
            candidates.append((value.shape[0], value.shape[2]))
        elif leaf.output_type == SemanticType.MARKET_DAILY_PRICE:
            continue
        elif value.ndim >= 2:
            candidates.append((value.shape[0], value.shape[1]))
    if not candidates:
        raise ValueError("cannot infer the (I,D) grid from expression leaves")
    if any(shape != candidates[0] for shape in candidates[1:]):
        raise ValueError(f"inconsistent leaf grids: {candidates}")
    return candidates[0]


# A date chunk costs `chunk_days + history` days of work but only publishes
# `chunk_days`, so a chunk shorter than its own halo spends most of its time
# recomputing. Sizing chunks in *rows* charged minute-wide leaves and daily
# leaves the same price, which left daily trees on three-day chunks behind a
# thirty-nine-day halo: a 270x slowdown over evaluating the 7 MiB grid whole.
DEFAULT_CHUNK_ELEMENTS = 1 << 22        # ~4M floats ≈ 16 MiB per leaf slice
MIN_HALO_MULTIPLE = 4                   # bounds halo redundancy at ~1.25x


def _elements_per_day(roots, context, instruments) -> int:
    """Leaf elements touched per calendar day, summed over distinct leaves."""
    total = 0
    for leaf in _unique_leaves(roots):
        value = context[leaf.name]
        if not isinstance(value, torch.Tensor):
            continue
        if leaf.output_type == SemanticType.MARKET_DAILY_PRICE:
            total += 1
        elif leaf.output_type == SemanticType.PAIR_SIMILARITY:
            total += value.shape[0] * value.shape[1]
        else:
            width = 1
            for size in value.shape[2:]:
                width *= size
            total += instruments * width
    return max(total, 1)


class ChunkPlan:
    """How a daily expression will actually be split, and what that costs.

    Shared with the cost model so a budget cannot accept a genome whose real
    price is dominated by halo recomputation the estimate never saw.
    """

    __slots__ = ("mode", "chunk_days", "history", "days", "amplification")

    def __init__(self, mode, chunk_days, history, days):
        self.mode = mode
        self.chunk_days = chunk_days
        self.history = history
        self.days = days
        if mode != "date" or not chunk_days:
            self.amplification = 1.0
        elif history is None:
            # An unbounded operator re-reads the whole prefix each chunk.
            chunks = -(-days // chunk_days)
            self.amplification = (chunks + 1) / 2 * chunk_days * chunks / max(days, 1)
        else:
            self.amplification = (chunk_days + history) / chunk_days

    def __repr__(self):
        return (
            f"ChunkPlan(mode={self.mode!r}, chunk_days={self.chunk_days}, "
            f"history={self.history}, amplification={self.amplification:.2f})"
        )


def plan_chunks(
    roots,
    registry,
    instruments: int,
    days: int,
    elements_per_day: int,
    element_budget: int = DEFAULT_CHUNK_ELEMENTS,
    min_halo_multiple: int = MIN_HALO_MULTIPLE,
) -> ChunkPlan:
    """Decide whole-grid, date-chunked or row-chunked evaluation."""
    if elements_per_day * days <= element_budget:
        return ChunkPlan("whole", days, 0, days)

    scope = ExecutionScope.LOCAL
    histories = []
    for root in roots:
        scope |= root.execution_scope_with(registry)
        histories.append(root.history_days_with(registry))
    if not scope & (ExecutionScope.FULL_CROSS_SECTION | ExecutionScope.HISTORY):
        return ChunkPlan("row", 0, 0, days)

    history = None if any(value is None for value in histories) else max(histories)
    chunk_days = max(1, element_budget // elements_per_day)
    if history:
        # Memory says one thing, redundancy another; take the larger chunk and
        # accept the extra slice, because the halo is already being loaded.
        chunk_days = max(chunk_days, min_halo_multiple * history)
    chunk_days = min(chunk_days, days)
    return ChunkPlan("date", chunk_days, history, days)


_MINUTE_WIDE_PREFIXES = ("minute", "legacy_minute", "same_minute")


def elements_per_day_for_shape(roots, shape: Mapping[str, int]) -> int:
    """Leaf elements per day inferred from types, for costing without tensors."""
    instruments = int(shape.get("I", 1))
    minutes = int(shape.get("M", 1))
    total = 0
    for leaf in _unique_leaves(roots):
        kind = leaf.output_type.value
        if leaf.output_type == SemanticType.MARKET_DAILY_PRICE:
            total += 1
        elif leaf.output_type == SemanticType.PAIR_SIMILARITY:
            total += instruments * instruments
        elif kind.startswith(_MINUTE_WIDE_PREFIXES):
            total += instruments * minutes
        else:
            total += instruments
    return max(total, 1)


def plan_chunks_for_shape(
    roots,
    registry,
    shape: Mapping[str, int],
    element_budget: int = DEFAULT_CHUNK_ELEMENTS,
) -> ChunkPlan:
    """The plan a run of this shape would use, without materialising it."""
    instruments = int(shape.get("I", 1))
    days = int(shape.get("D", 1))
    return plan_chunks(
        roots, registry, instruments, days,
        elements_per_day_for_shape(roots, shape), element_budget,
    )


def _slice_dates(value, value_type, start, stop, instruments, days):
    if not isinstance(value, torch.Tensor):
        return value
    if value_type == SemanticType.PAIR_SIMILARITY:
        return value[:, :, start:stop]
    if value_type == SemanticType.MARKET_DAILY_PRICE:
        if value.shape[0] != days:
            raise ValueError("market daily leaf does not match the date grid")
        return value[start:stop]
    if value.ndim >= 2 and value.shape[:2] == (instruments, days):
        return value[:, start:stop, ...]
    return value


def _slice_rows(value, value_type, start, stop, instruments, days):
    if not isinstance(value, torch.Tensor):
        return value
    if value_type in (SemanticType.PAIR_SIMILARITY, SemanticType.MARKET_DAILY_PRICE):
        raise ValueError(
            f"{value_type.value} cannot be evaluated in independent row chunks"
        )
    if value.ndim >= 2 and value.shape[:2] == (instruments, days):
        return value.reshape(instruments * days, *value.shape[2:])[start:stop]
    return value


def _context_for(roots, context, slicer, *args):
    return {
        leaf.name: slicer(context[leaf.name], leaf.output_type, *args)
        for leaf in _unique_leaves(roots)
    }


def evaluate_daily_expression(
    root: OperatorNode,
    context: Mapping[str, Any],
    registry: OperatorRegistry,
    chunk_rows: int = 4096,
    element_budget: int | None = None,
) -> torch.Tensor:
    return evaluate_daily_expressions(
        (root,), context, registry, chunk_rows, element_budget
    )[0]


def evaluate_daily_expressions(
    roots: tuple[OperatorNode, ...],
    context: Mapping[str, Any],
    registry: OperatorRegistry,
    chunk_rows: int = 4096,
    element_budget: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Evaluate an ``(I,D)`` expression without violating its axis semantics.

    Local, history-free trees may flatten stock-days and split rows.  A tree
    that needs either the complete cross-section or daily history is split by
    date while retaining every instrument.  History chunks receive the exact
    preceding halo; an unbounded recursive operator receives the whole prefix.
    """
    if not roots:
        raise ValueError("at least one expression root is required")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    instruments, days = _grid_shape(roots, context)
    elements_per_day = _elements_per_day(roots, context, instruments)
    if element_budget is None:
        # `chunk_rows` is a minute-row budget; read it as the element budget it
        # implies so narrow daily leaves are not charged a minute-wide price.
        element_budget = max(chunk_rows * 240, DEFAULT_CHUNK_ELEMENTS)
    plan = plan_chunks(
        roots, registry, instruments, days, elements_per_day, element_budget
    )
    if plan.mode == "whole":
        cache = {}
        results = tuple(root.evaluate(context, registry, cache) for root in roots)
        for result in results:
            if result.shape != (instruments, days):
                raise ValueError(
                    f"daily expression returned {tuple(result.shape)}, expected "
                    f"{(instruments, days)}"
                )
        return results

    reference = next(
        value for value in context.values() if isinstance(value, torch.Tensor)
    )
    outputs = tuple(
        torch.full(
            (instruments, days), float("nan"), device=reference.device,
            dtype=torch.float32,
        )
        for _ in roots
    )

    if plan.mode == "date":
        chunk_days, history = plan.chunk_days, plan.history
        for start in range(0, days, chunk_days):
            stop = min(start + chunk_days, days)
            halo_start = 0 if history is None else max(0, start - history)
            window = _context_for(
                roots, context, _slice_dates, halo_start, stop,
                instruments, days,
            )
            cache = {}
            values = tuple(
                root.evaluate(window, registry, cache).float() for root in roots
            )
            for output, value in zip(outputs, values):
                expected = (instruments, stop - halo_start)
                if value.shape != expected:
                    raise ValueError(
                        f"date-chunk expression returned {tuple(value.shape)}, "
                        f"expected {expected}"
                    )
                offset = start - halo_start
                output[:, start:stop] = value[
                    :, offset:offset + stop - start
                ]
        return outputs

    rows = instruments * days
    flats = tuple(output.reshape(rows) for output in outputs)
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        window = _context_for(
            roots, context, _slice_rows, start, stop, instruments, days
        )
        cache = {}
        values = tuple(
            root.evaluate(window, registry, cache).float() for root in roots
        )
        for flat, value in zip(flats, values):
            if value.shape != (stop - start,):
                raise ValueError(
                    f"row-chunk expression returned {tuple(value.shape)}, "
                    f"expected {(stop - start,)}"
                )
            flat[start:stop] = value
    return outputs

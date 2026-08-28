"""Typed, evolvable operator-slot genome for every handbook factor.

Each reproduction template is an anchor fill.  The selected core factor,
cross-sectional transform and low-frequency transform are operator-valued
genes, while each operator's numeric choices remain its parameter genes.
"""

from dataclasses import dataclass
from typing import Any

from min_gp.dsl import (
    LeafNode, OperatorNode, SemanticType, evaluate_daily_expression,
)
from min_gp.factors.event_skeleton import Slot
from min_gp.factors.tide_skeleton import (
    CompleteTideSkeletonGenome, complete_tide_anchor,
)
from min_gp.factors.climb_skeleton import (
    ClimbMountainSkeletonGenome, climb_mountain_anchor,
)
from min_gp.factors.rushing_skeleton import (
    RushingForwardSkeletonGenome, rushing_forward_anchor,
)
from min_gp.factors.handbook_composed import (
    ComposedHandbookGenome, composed_handbook_anchor,
)
from min_gp.operators import build_operator_registry


CORE_LEAVES = {
    "handbook_hidden_flower": ("close", "volume"),
    "handbook_long_short_battle": ("high", "low", "close", "volume"),
    "handbook_equal_treatment": ("open", "close", "volume"),
    "handbook_dark_flow": ("open", "high", "low", "volume"),
    "handbook_raw_panic": ("daily_close", "market_close"),
    "handbook_raw_panic_intraday": (
        "daily_close", "market_close", "close",
    ),
    "handbook_rushing_forward": (
        "amount_share", "volume_share", "up_volume_down_price_mask",
    ),
    "handbook_water_boat": (
        "high_amount", "low_amount", "float_market_cap",
    ),
    "handbook_cooperation_effect": (
        "volume_share", "price_state", "daily_return", "pair_similarity",
    ),
}

FACTOR_CORE = {
    "hidden_flower": "handbook_hidden_flower",
    "long_short_battle": "handbook_long_short_battle",
    "equal_treatment": "handbook_equal_treatment",
    "dark_flow": "handbook_dark_flow",
    "raw_panic": "handbook_raw_panic",
    "rushing_forward": "handbook_rushing_forward",
    "water_boat": "handbook_water_boat",
    "cooperation_effect": "handbook_cooperation_effect",
}

ANCHOR_PARAMS = {
    "complete_tide": dict(neighborhood=9, exclude_edges=15, smooth_window=20),
    "climb_mountain": dict(window=5, smooth_window=20),
    "hidden_flower": dict(
        lags=5, smooth_window=20, align_component_directions=True,
    ),
    "long_short_battle": dict(return_window=5, smooth_window=20),
    "equal_treatment": dict(
        response_window=5, exclude_edges=15, smooth_window=20,
    ),
    "dark_flow": dict(
        bins=48, lookback=5, multiple=1.0, smooth_window=20,
    ),
    "raw_panic": dict(smooth_window=20),
    "rushing_forward": dict(smooth_window=20),
    "water_boat": {},
    "cooperation_effect": dict(peer_count=30, smooth_window=20),
}


def handbook_core_candidates(registry, available_fields=None):
    # Whole-factor wrappers remain registered only as numerical baselines.
    # Production GP must search the fine-grained trees instead.
    return ()


def handbook_slot_candidates(registry, kind, available_fields=None):
    raw, daily = SemanticType.DAILY_RAW_FACTOR, SemanticType.DAILY_FACTOR
    if kind == "core":
        return handbook_core_candidates(registry, available_fields)
    if kind == "cross_section":
        return registry.matching(raw, input_types=(raw,))
    if kind == "low_frequency":
        return registry.matching(daily, input_types=(raw,))
    raise ValueError(f"unknown handbook slot: {kind}")


@dataclass(frozen=True)
class HandbookSkeletonGenome:
    core: Slot
    cross_section: Slot
    low_frequency: Slot
    anchor_name: str | None = None

    def __post_init__(self):
        self.expression()

    def expression(self, registry=None):
        registry = registry or build_operator_registry()
        spec = registry.get(self.core.operator)
        try:
            names = CORE_LEAVES[self.core.operator]
        except KeyError as exc:
            raise TypeError(f"{self.core.operator!r} is not a handbook core") from exc
        leaves = tuple(
            LeafNode(name, value_type)
            for name, value_type in zip(names, spec.input_types)
        )
        core = OperatorNode(
            self.core.operator, leaves, self.core.kwargs
        ).bind(registry)
        adjusted = OperatorNode(
            self.cross_section.operator, (core,), self.cross_section.kwargs
        ).bind(registry)
        root = OperatorNode(
            self.low_frequency.operator, (adjusted,),
            self.low_frequency.kwargs,
        ).bind(registry)
        return root, registry

    def evaluate(
        self, context: dict[str, Any], registry=None, chunk_rows: int = 4096
    ):
        missing = set(self.required_fields) - set(context)
        if missing:
            raise ValueError(f"handbook genome requires fields {sorted(missing)}")
        root, registry = self.expression(registry)
        return evaluate_daily_expression(root, context, registry, chunk_rows)

    @property
    def execution_scope(self):
        root, registry = self.expression()
        return root.execution_scope_with(registry)

    @property
    def history_days(self):
        root, registry = self.expression()
        return root.history_days_with(registry)

    @property
    def intraday_lookahead_minutes(self):
        root, registry = self.expression()
        return root.intraday_lookahead_with(registry)

    @property
    def complexity_profile(self):
        root, registry = self.expression()
        return root.complexity_profile_with(registry)

    @property
    def required_fields(self):
        return tuple(sorted(CORE_LEAVES[self.core.operator]))

    @property
    def operator_slots(self):
        return (self.core.operator, self.cross_section.operator,
                self.low_frequency.operator)

    @property
    def complexity(self):
        root, registry = self.expression()
        return root.complexity_with(registry)

    def to_dict(self):
        def slot(value):
            return [value.operator, dict(value.params)]
        return {
            "kind": "handbook_skeleton",
            "anchor_name": self.anchor_name,
            "core": slot(self.core),
            "cross_section": slot(self.cross_section),
            "low_frequency": slot(self.low_frequency),
        }

    @classmethod
    def from_dict(cls, payload):
        if payload.get("kind") == "complete_tide_skeleton":
            return CompleteTideSkeletonGenome.from_dict(payload)
        if payload.get("kind") == "climb_mountain_skeleton":
            return ClimbMountainSkeletonGenome.from_dict(payload)
        if payload.get("kind") == "composed_handbook":
            return ComposedHandbookGenome.from_dict(payload)
        if payload.get("kind") == "rushing_forward_skeleton":
            return RushingForwardSkeletonGenome.from_dict(payload)
        def slot(value):
            return Slot.of(value[0], **value[1])
        return cls(
            core=slot(payload["core"]),
            cross_section=slot(payload["cross_section"]),
            low_frequency=slot(payload["low_frequency"]),
            anchor_name=payload.get("anchor_name"),
        )

    def __str__(self):
        root, _ = self.expression()
        return str(root)


def handbook_anchor(name: str, **overrides):
    if name == "complete_tide":
        params = dict(ANCHOR_PARAMS[name])
        params.update(overrides)
        return complete_tide_anchor(**params)
    if name == "climb_mountain":
        params = dict(ANCHOR_PARAMS[name])
        params.update(overrides)
        return climb_mountain_anchor(**params)
    if name == "rushing_forward":
        params = dict(ANCHOR_PARAMS[name])
        params.update(overrides)
        return rushing_forward_anchor(**params)
    if name in FACTOR_CORE:
        params = dict(ANCHOR_PARAMS[name])
        params.update(overrides)
        return composed_handbook_anchor(name, **params)
    try:
        core = FACTOR_CORE[name]
    except KeyError as exc:
        raise ValueError(f"unknown handbook anchor: {name}") from exc
    params = dict(ANCHOR_PARAMS[name])
    params.update(overrides)
    return HandbookSkeletonGenome(
        core=Slot.of(core, **params),
        cross_section=Slot.of("cross_section_identity"),
        low_frequency=Slot.of("daily_identity"),
        anchor_name=name,
    )


def handbook_seed_population():
    return tuple(
        handbook_anchor(name)
        for name in ("complete_tide", "climb_mountain", *FACTOR_CORE)
    )

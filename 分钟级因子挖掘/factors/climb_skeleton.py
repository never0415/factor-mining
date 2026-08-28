"""Fine-grained, typed and recombinable Climb-Mountain genome."""

from dataclasses import dataclass
from typing import Any

from min_gp.dsl import (
    LeafNode, OperatorNode, SemanticType, chunk_peak_shape,
    estimate_expression_cost, evaluate_daily_expression,
)
from min_gp.factors.event_skeleton import Slot
from min_gp.operators import build_operator_registry


CLIMB_SLOT_OPERATORS = {
    "dispersion": (
        "rolling_ohlc_dispersion", "rolling_ohlc_range_dispersion",
    ),
    "returns": ("close_minute_return", "close_minute_log_return"),
    "ratio": ("safe_signal_ratio", "signed_signal_ratio"),
    "state": ("extreme_high_state", "extreme_low_state"),
    "measurement": ("conditional_covariance",),
    "cross_section": (
        "cross_section_identity", "cross_section_distance",
        "cross_section_rank",
    ),
    "temporal": (
        "daily_identity", "smooth_daily", "rolling_daily_std",
        "mean_std_blend",
    ),
}


def climb_slot_candidates(registry, kind):
    try:
        names = CLIMB_SLOT_OPERATORS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown climb slot: {kind}") from exc
    return tuple(registry.get(name) for name in names)


@dataclass(frozen=True)
class ClimbMountainSkeletonGenome:
    dispersion: Slot
    returns: Slot
    ratio: Slot
    state: Slot
    measurement: Slot
    cross_section: Slot
    temporal: Slot
    anchor_name: str | None = "climb_mountain"

    def __post_init__(self):
        self.expression()

    def expression(self, registry=None):
        registry = registry or build_operator_registry()
        open_ = LeafNode("open", SemanticType.MINUTE_OPEN)
        high = LeafNode("high", SemanticType.MINUTE_HIGH)
        low = LeafNode("low", SemanticType.MINUTE_LOW)
        close = LeafNode("close", SemanticType.MINUTE_CLOSE)
        dispersion = OperatorNode(
            self.dispersion.operator, (open_, high, low, close),
            self.dispersion.kwargs,
        ).bind(registry)
        returns = OperatorNode(
            self.returns.operator, (close,), self.returns.kwargs
        ).bind(registry)
        ratio = OperatorNode(
            self.ratio.operator, (returns, dispersion), self.ratio.kwargs
        ).bind(registry)
        state = OperatorNode(
            self.state.operator, (dispersion,), self.state.kwargs
        ).bind(registry)
        raw = OperatorNode(
            self.measurement.operator, (dispersion, ratio, state),
            self.measurement.kwargs,
        ).bind(registry)
        adjusted = OperatorNode(
            self.cross_section.operator, (raw,), self.cross_section.kwargs
        ).bind(registry)
        root = OperatorNode(
            self.temporal.operator, (adjusted,), self.temporal.kwargs
        ).bind(registry)
        return root, registry

    def evaluate(self, context: dict[str, Any], registry=None, chunk_rows=4096):
        missing = set(self.required_fields) - set(context)
        if missing:
            raise ValueError(f"climb mountain requires fields {sorted(missing)}")
        root, registry = self.expression(registry)
        # Materialise the intraday measurement once.  Re-running it for every
        # temporal halo would multiply the expensive OHLC unfold by ~D/chunk.
        adjusted_node = root.children[0]
        raw_node = adjusted_node.children[0]
        raw = evaluate_daily_expression(
            raw_node, context, registry, chunk_rows
        )
        adjusted = registry.get(self.cross_section.operator).implementation(
            raw, **self.cross_section.kwargs
        )
        return registry.get(self.temporal.operator).implementation(
            adjusted, **self.temporal.kwargs
        )

    @property
    def required_fields(self):
        return ("close", "high", "low", "open")

    @property
    def operator_slots(self):
        return tuple(
            getattr(self, name).operator for name in CLIMB_SLOT_OPERATORS
        )

    @property
    def complexity(self):
        root, registry = self.expression()
        return root.complexity_with(registry)

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

    def cost_estimate(self, target_shape, chunk_rows=None):
        root, registry = self.expression()
        peak = (
            None if chunk_rows is None
            else chunk_peak_shape(root, registry, target_shape, chunk_rows)
        )
        return estimate_expression_cost(
            root, registry, target_shape, peak_shape=peak
        )

    def to_dict(self):
        return {
            "kind": "climb_mountain_skeleton",
            "anchor_name": self.anchor_name,
            **{
                name: [getattr(self, name).operator,
                       dict(getattr(self, name).params)]
                for name in CLIMB_SLOT_OPERATORS
            },
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            **{
                name: Slot.of(payload[name][0], **payload[name][1])
                for name in CLIMB_SLOT_OPERATORS
            },
            anchor_name=payload.get("anchor_name"),
        )

    def __str__(self):
        root, _ = self.expression()
        return str(root)


def climb_mountain_anchor(window=5, smooth_window=20):
    return ClimbMountainSkeletonGenome(
        dispersion=Slot.of("rolling_ohlc_dispersion", window=window),
        returns=Slot.of("close_minute_return", horizon=1),
        ratio=Slot.of("safe_signal_ratio", floor=1e-12),
        state=Slot.of("extreme_high_state", sigma=1.0),
        measurement=Slot.of("conditional_covariance", min_count=3),
        cross_section=Slot.of("cross_section_identity"),
        temporal=Slot.of("mean_std_blend", window=smooth_window),
    )

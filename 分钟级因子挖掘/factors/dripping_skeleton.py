"""Operator-valued spectral skeleton for the Dripping-Stone family."""

from dataclasses import dataclass

from min_gp.dsl import (
    LeafNode, OperatorNode, SemanticType, evaluate_daily_expression,
)
from min_gp.factors.event_skeleton import Slot
from min_gp.operators import build_operator_registry


DRIPPING_SLOT_OPERATORS = {
    "session": ("select_regular_session",),
    "transform": ("volume_transform", "boxcox_grid_mle"),
    "clip": ("iqr_clip",),
    "detrend": ("detrend",),
    "window": ("spectral_window",),
    "spectrum": ("fft_power",),
    "reducer": ("band_power_ratio", "spectral_entropy"),
    "low_frequency": (
        "daily_identity", "mean_std_blend", "rolling_daily_std",
        "smooth_daily",
    ),
}


def dripping_slot_candidates(registry, kind):
    try:
        names = DRIPPING_SLOT_OPERATORS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown dripping slot: {kind}") from exc
    return tuple(registry.get(name) for name in names)


@dataclass(frozen=True)
class DrippingSkeletonGenome:
    session: Slot
    transform: Slot
    clip: Slot
    detrend: Slot
    window: Slot
    spectrum: Slot
    reducer: Slot
    low_frequency: Slot

    def expression(self, registry=None):
        registry = registry or build_operator_registry()
        node = LeafNode("volume", SemanticType.MINUTE_VOLUME)
        for value in (
            self.session, self.transform, self.clip, self.detrend,
            self.window, self.spectrum, self.reducer, self.low_frequency,
        ):
            node = OperatorNode(
                value.operator, (node,), value.kwargs
            ).bind(registry)
        return node, registry

    def evaluate(self, context, chunk_rows=4096, registry=None):
        volume = context["volume"]
        if volume.ndim != 3:
            raise ValueError("volume must be (instrument,date,minute)")
        registry = registry or build_operator_registry()
        root, _ = self.expression(registry)
        raw = evaluate_daily_expression(
            root.children[0], context, registry, chunk_rows
        )
        return registry.get(self.low_frequency.operator).implementation(
            raw, **self.low_frequency.kwargs
        )

    @property
    def required_fields(self):
        return ("volume",)

    @property
    def operator_slots(self):
        return tuple(
            getattr(self, kind).operator for kind in DRIPPING_SLOT_OPERATORS
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

    def to_dict(self):
        return {
            "kind": "dripping_skeleton",
            **{
                kind: [getattr(self, kind).operator,
                       dict(getattr(self, kind).params)]
                for kind in DRIPPING_SLOT_OPERATORS
            },
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(**{
            kind: Slot.of(payload[kind][0], **payload[kind][1])
            for kind in DRIPPING_SLOT_OPERATORS
        })

    def __str__(self):
        root, _ = self.expression()
        return str(root)


def dripping_stone_anchor():
    return DrippingSkeletonGenome(
        session=Slot.of("select_regular_session", session="all"),
        transform=Slot.of("volume_transform", mode="raw"),
        clip=Slot.of("iqr_clip", k=3.0, min_valid_ratio=0.95),
        detrend=Slot.of("detrend", method="demean"),
        window=Slot.of("spectral_window", window="hann"),
        spectrum=Slot.of("fft_power"),
        reducer=Slot.of(
            "band_power_ratio", period_low=2.0, period_high=5.0
        ),
        low_frequency=Slot.of("daily_identity"),
    )

"""Structure-preserving genome for the Rushing-Forward handbook island."""

from dataclasses import dataclass
from typing import Any

from min_gp.dsl import LeafNode, OperatorNode, SemanticType
from min_gp.factors.event_skeleton import Slot
from min_gp.operators import build_operator_registry


RUSHING_SLOT_OPERATORS = {
    "cross_section": (
        "cross_section_identity",
        "cross_section_rank",
        "cross_section_distance",
    ),
    "temporal": (
        "daily_identity",
        "smooth_daily",
        "rolling_daily_std",
        "mean_std_blend",
        "rank_smooth_daily",
    ),
}


def rushing_slot_candidates(registry, kind):
    try:
        names = RUSHING_SLOT_OPERATORS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown rushing-forward slot: {kind}") from exc
    return tuple(registry.get(name) for name in names)


@dataclass(frozen=True)
class RushingForwardSkeletonGenome:
    invert_mask: bool
    cross_section: Slot
    temporal: Slot
    anchor_name: str | None = "rushing_forward"

    def __post_init__(self):
        self.expression()

    def expression(self, registry=None):
        registry = registry or build_operator_registry()
        amount_share = LeafNode(
            "amount_share", SemanticType.MINUTE_AMOUNT_SHARE
        )
        volume_share = LeafNode(
            "volume_share", SemanticType.MINUTE_VOLUME_SHARE
        )
        mask = LeafNode(
            "up_volume_down_price_mask", SemanticType.MINUTE_MASK
        )
        if self.invert_mask:
            mask = OperatorNode("inverse_event", (mask,), {}).bind(registry)
        raw = OperatorNode(
            "rushing_imbalance", (amount_share, volume_share, mask), {}
        ).bind(registry)
        adjusted = OperatorNode(
            self.cross_section.operator, (raw,), self.cross_section.kwargs
        ).bind(registry)
        root = OperatorNode(
            self.temporal.operator, (adjusted,), self.temporal.kwargs
        ).bind(registry)
        return root, registry

    def evaluate(
        self, context: dict[str, Any], registry=None, chunk_rows: int = 4096
    ):
        missing = set(self.required_fields) - set(context)
        if missing:
            raise ValueError(
                f"rushing forward requires fields {sorted(missing)}"
            )
        registry = registry or build_operator_registry()
        mask = context["up_volume_down_price_mask"].bool()
        if self.invert_mask:
            mask = registry.get("inverse_event").implementation(mask)
        raw = registry.get("rushing_imbalance").implementation(
            context["amount_share"], context["volume_share"], mask
        )
        adjusted = registry.get(self.cross_section.operator).implementation(
            raw, **self.cross_section.kwargs
        )
        return registry.get(self.temporal.operator).implementation(
            adjusted, **self.temporal.kwargs
        )

    @property
    def required_fields(self):
        return (
            "amount_share",
            "up_volume_down_price_mask",
            "volume_share",
        )

    @property
    def operator_slots(self):
        prefix = ("inverse_event",) if self.invert_mask else ()
        return prefix + (
            "rushing_imbalance",
            self.cross_section.operator,
            self.temporal.operator,
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
            "kind": "rushing_forward_skeleton",
            "anchor_name": self.anchor_name,
            "invert_mask": self.invert_mask,
            "cross_section": [
                self.cross_section.operator, dict(self.cross_section.params)
            ],
            "temporal": [
                self.temporal.operator, dict(self.temporal.params)
            ],
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            invert_mask=bool(payload["invert_mask"]),
            cross_section=Slot.of(
                payload["cross_section"][0], **payload["cross_section"][1]
            ),
            temporal=Slot.of(
                payload["temporal"][0], **payload["temporal"][1]
            ),
            anchor_name=payload.get("anchor_name"),
        )

    def __str__(self):
        root, _ = self.expression()
        return str(root)


def rushing_forward_anchor(smooth_window=20):
    return RushingForwardSkeletonGenome(
        invert_mask=False,
        cross_section=Slot.of("cross_section_identity"),
        temporal=Slot.of(
            "smooth_daily", method="mean", window=smooth_window
        ),
    )


RUSHING_LEAF_FIELDS = ("amount_share", "volume_share", "up_volume_down_price_mask")


def rushing_imbalance_node(registry=None, invert_mask=False):
    """The bare Rushing-Forward imbalance as a typed subtree.

    ``RushingForwardSkeletonGenome`` fixes this core and exposes only a
    cross-section and a temporal slot around it -- two multiplied by three
    multiplied by five is thirty reachable structures, which a population of
    sixty enumerates several times over in one generation. Handing the same
    core to the seed-tree GP as a plain subtree keeps the economics and drops
    the ceiling: it outputs ``DAILY_RAW_FACTOR``, which is assignable to the
    generic ``LEGACY_DAILY`` the migrated seed operators consume, so any of
    them may wrap it and crossover may graft it into any daily slot.
    """
    registry = registry or build_operator_registry()
    mask = LeafNode("up_volume_down_price_mask", SemanticType.MINUTE_MASK)
    if invert_mask:
        mask = OperatorNode("inverse_event", (mask,), {}).bind(registry)
    return OperatorNode(
        "rushing_imbalance",
        (
            LeafNode("amount_share", SemanticType.MINUTE_AMOUNT_SHARE),
            LeafNode("volume_share", SemanticType.MINUTE_VOLUME_SHARE),
            mask,
        ),
        {},
    ).bind(registry), registry

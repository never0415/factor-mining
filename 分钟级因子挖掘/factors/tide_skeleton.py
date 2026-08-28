"""Fine-grained, typed and recombinable Complete-Tide genome."""

from dataclasses import dataclass
from typing import Any

from min_gp.dsl import (
    LeafNode, OperatorNode, SemanticType, chunk_peak_shape,
    estimate_expression_cost, evaluate_daily_expressions,
)
from min_gp.factors.event_skeleton import Slot
from min_gp.operators import build_operator_registry


TIDE_SLOT_OPERATORS = {
    "activity": ("rolling_volume_sum", "rolling_volume_std"),
    "pivot": ("locate_peak", "locate_trough"),
    "left_locator": ("locate_left_valley", "locate_left_peak"),
    "right_locator": ("locate_right_valley", "locate_right_peak"),
    "left_speed": ("left_path_return_speed", "left_path_log_speed"),
    "right_speed": ("right_path_return_speed", "right_path_log_speed"),
    "left_activity": ("left_activity_value", "left_log_activity_value"),
    "right_activity": ("right_activity_value", "right_log_activity_value"),
    "selector": (
        "strong_path_by_activity", "weak_path_by_activity",
        "average_path_speed",
    ),
    "cross_section": (
        "cross_section_identity", "cross_section_distance",
        "cross_section_rank",
    ),
    "temporal": (
        "daily_identity", "smooth_daily", "rolling_daily_std",
        "mean_std_blend",
    ),
    "combiner": ("equal_blend",),
}


def tide_slot_candidates(registry, kind):
    try:
        names = TIDE_SLOT_OPERATORS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown tide slot: {kind}") from exc
    return tuple(registry.get(name) for name in names)


@dataclass(frozen=True)
class TideBranch:
    selector: Slot
    cross_section: Slot
    temporal: Slot


@dataclass(frozen=True)
class CompleteTideSkeletonGenome:
    activity: Slot
    pivot: Slot
    left_locator: Slot
    right_locator: Slot
    left_speed: Slot
    right_speed: Slot
    left_activity: Slot
    right_activity: Slot
    strong: TideBranch
    weak: TideBranch
    combiner: Slot
    anchor_name: str | None = "complete_tide"

    def __post_init__(self):
        self.expression()

    def expression(self, registry=None):
        registry = registry or build_operator_registry()
        close = LeafNode("close", SemanticType.MINUTE_CLOSE)
        volume = LeafNode("volume", SemanticType.MINUTE_VOLUME)
        activity = OperatorNode(
            self.activity.operator, (volume,), self.activity.kwargs
        ).bind(registry)
        pivot = OperatorNode(
            self.pivot.operator, (activity,), self.pivot.kwargs
        ).bind(registry)
        left = OperatorNode(
            self.left_locator.operator, (activity, pivot),
            self.left_locator.kwargs,
        ).bind(registry)
        right = OperatorNode(
            self.right_locator.operator, (activity, pivot),
            self.right_locator.kwargs,
        ).bind(registry)
        left_speed = OperatorNode(
            self.left_speed.operator, (close, left, pivot),
            self.left_speed.kwargs,
        ).bind(registry)
        right_speed = OperatorNode(
            self.right_speed.operator, (close, pivot, right),
            self.right_speed.kwargs,
        ).bind(registry)
        left_activity = OperatorNode(
            self.left_activity.operator, (activity, left),
            self.left_activity.kwargs,
        ).bind(registry)
        right_activity = OperatorNode(
            self.right_activity.operator, (activity, right),
            self.right_activity.kwargs,
        ).bind(registry)

        def branch(fill):
            selected = OperatorNode(
                fill.selector.operator,
                (left_speed, right_speed, left_activity, right_activity),
                fill.selector.kwargs,
            ).bind(registry)
            adjusted = OperatorNode(
                fill.cross_section.operator, (selected,),
                fill.cross_section.kwargs,
            ).bind(registry)
            return OperatorNode(
                fill.temporal.operator, (adjusted,), fill.temporal.kwargs
            ).bind(registry)

        strong = branch(self.strong)
        weak = branch(self.weak)
        root = OperatorNode(
            self.combiner.operator, (strong, weak), self.combiner.kwargs
        ).bind(registry)
        return root, registry

    def evaluate(self, context: dict[str, Any], registry=None, chunk_rows=4096):
        missing = set(self.required_fields) - set(context)
        if missing:
            raise ValueError(f"complete tide requires fields {sorted(missing)}")
        root, registry = self.expression(registry)
        strong_tail, weak_tail = root.children
        strong_raw = strong_tail.children[0].children[0]
        weak_raw = weak_tail.children[0].children[0]
        strong_value, weak_value = evaluate_daily_expressions(
            (strong_raw, weak_raw), context, registry, chunk_rows
        )

        def finish(branch, raw):
            adjusted = registry.get(
                branch.cross_section.operator
            ).implementation(raw, **branch.cross_section.kwargs)
            return registry.get(branch.temporal.operator).implementation(
                adjusted, **branch.temporal.kwargs
            )

        strong = finish(self.strong, strong_value)
        weak = finish(self.weak, weak_value)
        return registry.get(self.combiner.operator).implementation(
            strong, weak, **self.combiner.kwargs
        )

    @property
    def required_fields(self):
        return ("close", "volume")

    @property
    def operator_slots(self):
        return (
            self.activity.operator, self.pivot.operator,
            self.left_locator.operator, self.right_locator.operator,
            self.left_speed.operator, self.right_speed.operator,
            self.left_activity.operator, self.right_activity.operator,
            self.strong.selector.operator, self.strong.cross_section.operator,
            self.strong.temporal.operator, self.weak.selector.operator,
            self.weak.cross_section.operator, self.weak.temporal.operator,
            self.combiner.operator,
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
        def slot(value):
            return [value.operator, dict(value.params)]

        def branch(value):
            return {
                "selector": slot(value.selector),
                "cross_section": slot(value.cross_section),
                "temporal": slot(value.temporal),
            }

        return {
            "kind": "complete_tide_skeleton",
            "anchor_name": self.anchor_name,
            **{
                name: slot(getattr(self, name))
                for name in (
                    "activity", "pivot", "left_locator", "right_locator",
                    "left_speed", "right_speed", "left_activity",
                    "right_activity", "combiner",
                )
            },
            "strong": branch(self.strong),
            "weak": branch(self.weak),
        }

    @classmethod
    def from_dict(cls, payload):
        def slot(value):
            return Slot.of(value[0], **value[1])

        def branch(value):
            return TideBranch(
                selector=slot(value["selector"]),
                cross_section=slot(value["cross_section"]),
                temporal=slot(value["temporal"]),
            )

        return cls(
            **{
                name: slot(payload[name])
                for name in (
                    "activity", "pivot", "left_locator", "right_locator",
                    "left_speed", "right_speed", "left_activity",
                    "right_activity", "combiner",
                )
            },
            strong=branch(payload["strong"]),
            weak=branch(payload["weak"]),
            anchor_name=payload.get("anchor_name"),
        )

    def __str__(self):
        root, _ = self.expression()
        return str(root)


def complete_tide_anchor(
    neighborhood=9, exclude_edges=15, smooth_window=20,
):
    return CompleteTideSkeletonGenome(
        activity=Slot.of("rolling_volume_sum", neighborhood=neighborhood),
        pivot=Slot.of("locate_peak", exclude_edges=exclude_edges),
        left_locator=Slot.of(
            "locate_left_valley", exclude_edges=exclude_edges
        ),
        right_locator=Slot.of(
            "locate_right_valley", exclude_edges=exclude_edges
        ),
        left_speed=Slot.of("left_path_return_speed"),
        right_speed=Slot.of("right_path_return_speed"),
        left_activity=Slot.of("left_activity_value"),
        right_activity=Slot.of("right_activity_value"),
        strong=TideBranch(
            selector=Slot.of("strong_path_by_activity"),
            cross_section=Slot.of("cross_section_identity"),
            temporal=Slot.of(
                "smooth_daily", method="mean", window=smooth_window
            ),
        ),
        weak=TideBranch(
            selector=Slot.of("weak_path_by_activity"),
            cross_section=Slot.of("cross_section_identity"),
            temporal=Slot.of("rolling_daily_std", window=smooth_window),
        ),
        combiner=Slot.of("equal_blend"),
    )

"""A slot-based skeleton shared by the handbook's event factors.

The hand-written templates in `event_factors.py` hard-code their operator
chain, so a search over them can only turn numeric knobs; the formula itself is
fixed. Here the chain is decomposed into parts and the *choice of part* becomes
the gene:

    detector -> statistic -> aggregator -> cross_section -> low_frequency ─┐
             └-> statistic2 -> aggregator -> cross_section -> low_frequency ┴-> combiner

适度冒险 is this skeleton with a two-branch fill (forward-window volatility plus
the spike minute's own return); 待著而救 is a single-branch fill with a
different detector. They stop being separate factors and become two points in
one space, which is what lets the search recombine them.

Slot candidates come from the operator registry's type signatures, so adding an
operator widens the search space with no table to update.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from min_gp.dsl import (
    LeafNode, OperatorNode, SemanticType, evaluate_daily_expression,
)
from min_gp.operators import build_operator_registry


MINUTE_SOURCES = {
    SemanticType.MINUTE_VOLUME: "volume",
    SemanticType.MINUTE_RETURN: "return",
    SemanticType.MINUTE_PRICE: "close",
}

DETECTOR_OUTPUT = SemanticType.MINUTE_MASK
STATISTIC_OUTPUT = SemanticType.MINUTE_SIGNAL
AGGREGATOR_INPUTS = (SemanticType.MINUTE_SIGNAL, SemanticType.MINUTE_MASK)
DAILY_RAW = SemanticType.DAILY_RAW_FACTOR
DAILY = SemanticType.DAILY_FACTOR


def _freeze(params: Mapping[str, Any] | None) -> tuple:
    return tuple(sorted((params or {}).items()))


@dataclass(frozen=True)
class Slot:
    """One filled slot: which operator, and with which parameters."""

    operator: str
    params: tuple = ()

    @classmethod
    def of(cls, operator: str, **params) -> "Slot":
        return cls(operator, _freeze(params))

    @property
    def kwargs(self) -> dict:
        return dict(self.params)

    def __str__(self) -> str:
        if not self.params:
            return self.operator
        inner = ", ".join(f"{k}={v!r}" for k, v in self.params)
        return f"{self.operator}({inner})"


def slot_candidates(
    registry, kind: str, available_fields: set[str] | frozenset[str] | None = None
) -> tuple:
    """Operators that can fill a named slot, taken from their type signatures."""
    if kind == "detector":
        return registry.matching(
            DETECTOR_OUTPUT, input_types=(SemanticType.MINUTE_VOLUME,)
        )
    if kind == "statistic":
        candidates = tuple(
            spec for spec in registry.matching(STATISTIC_OUTPUT, arity=1)
            if spec.input_types[0] in MINUTE_SOURCES
        )
        if available_fields is None:
            return candidates
        available_fields = set(available_fields)
        return tuple(
            spec for spec in candidates
            if (
                "volume" if spec.input_types[0] == SemanticType.MINUTE_VOLUME
                else "close"
            ) in available_fields
        )
    if kind == "aggregator":
        return registry.matching(DAILY_RAW, input_types=AGGREGATOR_INPUTS)
    if kind == "cross_section":
        return registry.matching(DAILY_RAW, input_types=(DAILY_RAW,))
    if kind == "low_frequency":
        return registry.matching(DAILY, input_types=(DAILY_RAW,))
    if kind == "combiner":
        return registry.matching(DAILY, input_types=(DAILY, DAILY))
    raise ValueError(f"unknown slot: {kind}")


@dataclass(frozen=True)
class Branch:
    """One measurement path: what to measure at the detected minutes."""

    statistic: Slot
    aggregator: Slot
    cross_section: Slot
    low_frequency: Slot

    def __str__(self) -> str:
        return (
            f"{self.low_frequency}<-{self.cross_section}<-"
            f"{self.aggregator}<-{self.statistic}"
        )


@dataclass(frozen=True)
class EventSkeletonGenome:
    """A complete fill of the skeleton. Hashable, so it can be a GP genome."""

    detector: Slot
    primary: Branch
    secondary: Branch | None = None
    combiner: Slot | None = None

    def __post_init__(self):
        if (self.secondary is None) != (self.combiner is None):
            raise ValueError(
                "a second branch needs a combiner, and a combiner needs a "
                "second branch"
            )
        # Fail here rather than at evaluation time: an unbuildable genome must
        # never reach the population.
        self.expression()

    # ── expression assembly ──

    def _minute_source(self, registry, statistic: Slot):
        """Feed a statistic the leaf its declared input type asks for."""
        spec = registry.get(statistic.operator)
        source_type = spec.input_types[0]
        if source_type == SemanticType.MINUTE_RETURN:
            close = LeafNode("close", SemanticType.MINUTE_PRICE)
            return OperatorNode(
                "minute_return", (close,), {"horizon": 1}
            ).bind(registry)
        if source_type not in MINUTE_SOURCES:
            raise TypeError(f"{statistic.operator}: unusable source {source_type}")
        return LeafNode(MINUTE_SOURCES[source_type], source_type)

    def _branch_expression(self, registry, branch: Branch, detector_node):
        source = self._minute_source(registry, branch.statistic)
        measured = OperatorNode(
            branch.statistic.operator, (source,), branch.statistic.kwargs
        ).bind(registry)
        daily = OperatorNode(
            branch.aggregator.operator, (measured, detector_node),
            branch.aggregator.kwargs,
        ).bind(registry)
        adjusted = OperatorNode(
            branch.cross_section.operator, (daily,), branch.cross_section.kwargs
        ).bind(registry)
        return OperatorNode(
            branch.low_frequency.operator, (adjusted,),
            branch.low_frequency.kwargs,
        ).bind(registry)

    def expression(self, registry=None):
        """Build and type-check the whole tree; raises on an illegal fill."""
        registry = registry or build_operator_registry()
        volume = LeafNode("volume", SemanticType.MINUTE_VOLUME)
        detector = OperatorNode(
            self.detector.operator, (volume,), self.detector.kwargs
        ).bind(registry)
        primary = self._branch_expression(registry, self.primary, detector)
        if self.secondary is None:
            return primary, registry
        secondary = self._branch_expression(registry, self.secondary, detector)
        combined = OperatorNode(
            self.combiner.operator, (primary, secondary), self.combiner.kwargs
        ).bind(registry)
        return combined, registry

    # ── evaluation ──

    def _branch_daily(self, registry, branch, detector_node, context, chunk):
        """Run the minute half using the expression's derived execution scope."""
        source = self._minute_source(registry, branch.statistic)
        measured = OperatorNode(
            branch.statistic.operator, (source,), branch.statistic.kwargs
        ).bind(registry)
        daily = OperatorNode(
            branch.aggregator.operator, (measured, detector_node),
            branch.aggregator.kwargs,
        ).bind(registry)
        return evaluate_daily_expression(daily, context, registry, chunk)

    def evaluate(
        self,
        minute_tensors: dict,
        chunk_rows: int = 4096,
        registry=None,
    ) -> torch.Tensor:
        """(I, D) factor from (I, D, M) minute tensors."""
        registry = registry or build_operator_registry()
        volume = minute_tensors["volume"]
        if volume.ndim != 3:
            raise ValueError(f"volume must be (I,D,M), got {tuple(volume.shape)}")
        I, D, M = volume.shape
        if self.needs_close and "close" not in minute_tensors:
            raise ValueError("this genome needs a 'close' minute tensor")
        detector = OperatorNode(
            self.detector.operator,
            (LeafNode("volume", SemanticType.MINUTE_VOLUME),),
            self.detector.kwargs,
        ).bind(registry)

        raw = self._branch_daily(
            registry, self.primary, detector, minute_tensors, chunk_rows
        )
        factor = self._daily_tail(registry, self.primary, raw)
        if self.secondary is None:
            return factor
        raw2 = self._branch_daily(
            registry, self.secondary, detector, minute_tensors, chunk_rows
        )
        other = self._daily_tail(registry, self.secondary, raw2)
        return registry.get(self.combiner.operator).implementation(
            factor, other, **self.combiner.kwargs
        )

    def _daily_tail(self, registry, branch, raw):
        """Cross-section and low-frequency steps need the whole (I, D) grid."""
        adjusted = registry.get(branch.cross_section.operator).implementation(
            raw, **branch.cross_section.kwargs
        )
        return registry.get(branch.low_frequency.operator).implementation(
            adjusted, **branch.low_frequency.kwargs
        )

    # ── metadata ──

    @property
    def needs_close(self) -> bool:
        registry = build_operator_registry()
        branches = [self.primary] + ([self.secondary] if self.secondary else [])
        return any(
            registry.get(branch.statistic.operator).input_types[0]
            != SemanticType.MINUTE_VOLUME
            for branch in branches
        )

    @property
    def required_fields(self) -> tuple[str, ...]:
        return ("close", "volume") if self.needs_close else ("volume",)

    @property
    def complexity(self) -> int:
        expression, registry = self.expression()
        return expression.complexity_with(registry)

    @property
    def execution_scope(self):
        expression, registry = self.expression()
        return expression.execution_scope_with(registry)

    @property
    def history_days(self):
        expression, registry = self.expression()
        return expression.history_days_with(registry)

    @property
    def intraday_lookahead_minutes(self):
        expression, registry = self.expression()
        return expression.intraday_lookahead_with(registry)

    @property
    def complexity_profile(self):
        expression, registry = self.expression()
        return expression.complexity_profile_with(registry)

    def to_dict(self) -> dict:
        def branch(value):
            if value is None:
                return None
            return {
                "statistic": [value.statistic.operator, dict(value.statistic.params)],
                "aggregator": [
                    value.aggregator.operator, dict(value.aggregator.params)
                ],
                "cross_section": [
                    value.cross_section.operator, dict(value.cross_section.params)
                ],
                "low_frequency": [
                    value.low_frequency.operator, dict(value.low_frequency.params)
                ],
            }
        return {
            "detector": [self.detector.operator, dict(self.detector.params)],
            "primary": branch(self.primary),
            "secondary": branch(self.secondary),
            "combiner": (
                None if self.combiner is None
                else [self.combiner.operator, dict(self.combiner.params)]
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EventSkeletonGenome":
        def slot(value):
            return None if value is None else Slot.of(value[0], **value[1])

        def branch(value):
            if value is None:
                return None
            return Branch(
                statistic=slot(value["statistic"]),
                aggregator=slot(value["aggregator"]),
                cross_section=slot(value["cross_section"]),
                low_frequency=slot(value["low_frequency"]),
            )
        return cls(
            detector=slot(payload["detector"]),
            primary=branch(payload["primary"]),
            secondary=branch(payload.get("secondary")),
            combiner=slot(payload.get("combiner")),
        )

    def __str__(self) -> str:
        expression, _ = self.expression()
        return str(expression)


# ──────────────────────────────────────────────
# Handbook anchors, expressed as slot fills
# ──────────────────────────────────────────────

def moderate_risk_anchor(
    sigma: float = 1.0,
    response_window: int = 5,
    smooth_window: int = 20,
    exclude_edges: int = 0,
    ddof: int = 0,
    standardize: bool = False,
) -> EventSkeletonGenome:
    """适度冒险 (section 3) as a skeleton fill, matching ModerateRiskTemplate."""
    def branch(statistic):
        return Branch(
            statistic=statistic,
            aggregator=Slot.of("masked_daily_mean_signal"),
            cross_section=Slot.of(
                "cross_section_distance", standardize=standardize
            ),
            low_frequency=Slot.of("mean_std_blend", window=smooth_window),
        )
    return EventSkeletonGenome(
        detector=Slot.of(
            "delta_sigma_event", transform="raw", sigma=sigma,
            direction="above", exclude_edges=exclude_edges, ddof=ddof,
        ),
        primary=branch(
            Slot.of("forward_window_std", window=response_window, ddof=ddof)
        ),
        secondary=branch(Slot.of("point_signal")),
        combiner=Slot.of("equal_blend"),
    )


def wait_rescue_anchor(
    top_k: int = 10,
    exclude_before: int = 15,
    min_gap: int = 5,
    follow_window: int = 5,
    smooth_window: int = 20,
) -> EventSkeletonGenome:
    """待著而救 (section 7) as a skeleton fill, matching WaitRescueTemplate."""
    return EventSkeletonGenome(
        detector=Slot.of(
            "topk_separated_events", k=top_k,
            exclude_before=exclude_before, min_gap=min_gap,
        ),
        primary=Branch(
            statistic=Slot.of("follow_ratio_series", window=follow_window),
            aggregator=Slot.of("masked_daily_mean_signal"),
            cross_section=Slot.of("cross_section_identity"),
            low_frequency=Slot.of("mean_std_blend", window=smooth_window),
        ),
    )


SKELETON_ANCHORS = {
    "moderate_risk": moderate_risk_anchor,
    "wait_rescue": wait_rescue_anchor,
}

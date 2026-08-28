"""Strongly typed event-factor templates from the reproduction handbook."""

from dataclasses import asdict, dataclass
from typing import Any

import torch

from min_gp.dsl import LeafNode, OperatorNode, SemanticType
from min_gp.operators import build_operator_registry
from min_gp.operators.event import (
    EVENT_SIGMAS,
    EDGE_EXCLUSIONS,
    DDOF_VALUES,
    EXCLUDE_OPEN_MINUTES,
    FORWARD_WINDOWS,
    MIN_EVENT_GAPS,
    TOP_K_EVENTS,
)
from min_gp.operators.temporal import SMOOTH_WINDOWS


@dataclass(frozen=True)
class ModerateRiskTemplate:
    """适度冒险: spike response volatility + spike-minute return."""

    sigma: float = 1.0
    response_window: int = 5
    smooth_window: int = 20
    exclude_edges: int = 0
    ddof: int = 0
    standardize_before_distance: bool = False
    direction: int = -1

    def __post_init__(self):
        if self.sigma not in EVENT_SIGMAS:
            raise ValueError(self.sigma)
        if self.response_window not in FORWARD_WINDOWS:
            raise ValueError(self.response_window)
        if self.smooth_window not in SMOOTH_WINDOWS[1:]:
            raise ValueError(self.smooth_window)
        if self.exclude_edges not in EDGE_EXCLUSIONS:
            raise ValueError(self.exclude_edges)
        if self.ddof not in DDOF_VALUES:
            raise ValueError(self.ddof)
        if self.direction != -1:
            raise ValueError("ModerateRisk paper direction is fixed at -1")

    def expressions(self):
        registry = build_operator_registry()
        close = LeafNode("close", SemanticType.MINUTE_PRICE)
        volume = LeafNode("volume", SemanticType.MINUTE_VOLUME)
        ret = OperatorNode("minute_return", (close,), {"horizon": 1}).bind(registry)
        delta = OperatorNode(
            "minute_delta_volume", (volume,), {"horizon": 1}
        ).bind(registry)
        spike = OperatorNode(
            "intraday_sigma_event", (delta,),
            {
                "sigma": self.sigma,
                "direction": "above",
                "exclude_edges": self.exclude_edges,
                "ddof": self.ddof,
            },
        ).bind(registry)
        response_std = OperatorNode(
            "forward_window_std", (ret,), {
                "window": self.response_window,
                "ddof": self.ddof,
            }
        ).bind(registry)
        daily_vol = OperatorNode(
            "masked_daily_mean_signal", (response_std, spike), {}
        ).bind(registry)
        daily_ret = OperatorNode(
            "masked_daily_mean_return", (ret, spike), {}
        ).bind(registry)
        return daily_vol, daily_ret, registry

    def cost_expression(self, registry=None):
        """Return the complete typed DAG used for production cost admission."""
        registry = registry or build_operator_registry()
        close = LeafNode("close", SemanticType.MINUTE_PRICE)
        volume = LeafNode("volume", SemanticType.MINUTE_VOLUME)
        ret = OperatorNode("minute_return", (close,), {"horizon": 1}).bind(registry)
        delta = OperatorNode(
            "minute_delta_volume", (volume,), {"horizon": 1}
        ).bind(registry)
        spike = OperatorNode(
            "intraday_sigma_event", (delta,),
            {
                "sigma": self.sigma,
                "direction": "above",
                "exclude_edges": self.exclude_edges,
                "ddof": self.ddof,
            },
        ).bind(registry)
        response = OperatorNode(
            "forward_window_std", (ret,),
            {"window": self.response_window, "ddof": self.ddof},
        ).bind(registry)
        raw_vol = OperatorNode(
            "masked_daily_mean_signal", (response, spike), {}
        ).bind(registry)
        raw_ret = OperatorNode(
            "masked_daily_mean_return", (ret, spike), {}
        ).bind(registry)
        distance_params = {
            "standardize": self.standardize_before_distance,
        }
        dist_vol = OperatorNode(
            "cross_section_distance", (raw_vol,), distance_params
        ).bind(registry)
        dist_ret = OperatorNode(
            "cross_section_distance", (raw_ret,), distance_params
        ).bind(registry)
        bright_vol = OperatorNode(
            "mean_std_blend", (dist_vol,), {"window": self.smooth_window}
        ).bind(registry)
        bright_ret = OperatorNode(
            "mean_std_blend", (dist_ret,), {"window": self.smooth_window}
        ).bind(registry)
        root = OperatorNode(
            "equal_blend", (bright_vol, bright_ret), {}
        ).bind(registry)
        return root, registry

    def evaluate(
        self, close: torch.Tensor, volume: torch.Tensor, chunk_rows: int = 4096
    ) -> torch.Tensor:
        if close.shape != volume.shape or close.ndim != 3:
            raise ValueError("close and volume must share shape (I,D,M)")
        I, D, M = close.shape
        close_rows, volume_rows = close.reshape(-1, M), volume.reshape(-1, M)
        # Build once to enforce the strong type contract, then share common
        # intermediates during execution instead of evaluating two trees twice.
        _, _, registry = self.expressions()
        op_return = registry.get("minute_return").implementation
        op_delta = registry.get("minute_delta_volume").implementation
        op_event = registry.get("intraday_sigma_event").implementation
        op_forward_std = registry.get("forward_window_std").implementation
        op_masked_mean = registry.get("masked_daily_mean_signal").implementation
        raw_vol = torch.full((I * D,), float("nan"), device=close.device)
        raw_ret = torch.full_like(raw_vol, float("nan"))
        for start in range(0, I * D, chunk_rows):
            stop = min(start + chunk_rows, I * D)
            ret = op_return(close_rows[start:stop], horizon=1)
            delta = op_delta(volume_rows[start:stop], horizon=1)
            spike = op_event(
                delta, sigma=self.sigma, direction="above",
                exclude_edges=self.exclude_edges, ddof=self.ddof,
            )
            response = op_forward_std(
                ret, window=self.response_window, ddof=self.ddof
            )
            raw_vol[start:stop] = op_masked_mean(response, spike)
            raw_ret[start:stop] = op_masked_mean(ret, spike)
        raw_vol, raw_ret = raw_vol.reshape(I, D), raw_ret.reshape(I, D)
        distance_spec = registry.get("cross_section_distance")
        kwargs = {"standardize": self.standardize_before_distance}
        dist_vol = distance_spec.implementation(raw_vol, **kwargs)
        dist_ret = distance_spec.implementation(raw_ret, **kwargs)
        low_freq = registry.get("mean_std_blend").implementation
        bright_vol = low_freq(dist_vol, self.smooth_window)
        bright_ret = low_freq(dist_ret, self.smooth_window)
        return registry.get("equal_blend").implementation(bright_vol, bright_ret)

    @property
    def complexity(self) -> int:
        a, b, registry = self.expressions()
        return (
            a.complexity_with(registry) + b.complexity_with(registry)
            + 2 * registry.get("cross_section_distance").cost
            + 2 * registry.get("mean_std_blend").cost
            + registry.get("equal_blend").cost
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            "moderate_risk("
            f"sigma={self.sigma}, response={self.response_window}, "
            f"smooth={self.smooth_window}, exclude_edges={self.exclude_edges}, "
            f"ddof={self.ddof}, "
            f"standardize={self.standardize_before_distance})"
        )


@dataclass(frozen=True)
class WaitRescueTemplate:
    """待著而救: volume following after separated top-volume events."""

    top_k: int = 10
    exclude_before: int = 15
    min_gap: int = 5
    follow_window: int = 5
    smooth_window: int = 20
    direction: int = -1

    def __post_init__(self):
        if self.top_k not in TOP_K_EVENTS:
            raise ValueError(self.top_k)
        if self.exclude_before not in EXCLUDE_OPEN_MINUTES:
            raise ValueError(self.exclude_before)
        if self.min_gap not in MIN_EVENT_GAPS:
            raise ValueError(self.min_gap)
        if self.follow_window not in FORWARD_WINDOWS:
            raise ValueError(self.follow_window)
        if self.smooth_window not in SMOOTH_WINDOWS[1:]:
            raise ValueError(self.smooth_window)
        if self.direction != -1:
            raise ValueError("WaitRescue paper direction is fixed at -1")

    def expression(self):
        registry = build_operator_registry()
        volume = LeafNode("volume", SemanticType.MINUTE_VOLUME)
        event = OperatorNode(
            "topk_separated_events", (volume,), {
                "k": self.top_k,
                "exclude_before": self.exclude_before,
                "min_gap": self.min_gap,
            },
        ).bind(registry)
        daily = OperatorNode(
            "follow_ratio", (volume, event), {"window": self.follow_window}
        ).bind(registry)
        return daily, registry

    def cost_expression(self, registry=None):
        """Include the temporal tail omitted from the raw evaluation tree."""
        registry = registry or build_operator_registry()
        volume = LeafNode("volume", SemanticType.MINUTE_VOLUME)
        event = OperatorNode(
            "topk_separated_events", (volume,), {
                "k": self.top_k,
                "exclude_before": self.exclude_before,
                "min_gap": self.min_gap,
            },
        ).bind(registry)
        daily = OperatorNode(
            "follow_ratio", (volume, event), {"window": self.follow_window}
        ).bind(registry)
        root = OperatorNode(
            "mean_std_blend", (daily,), {"window": self.smooth_window}
        ).bind(registry)
        return root, registry

    def evaluate(self, volume: torch.Tensor, chunk_rows: int = 4096) -> torch.Tensor:
        if volume.ndim != 3:
            raise ValueError("volume must be (I,D,M)")
        I, D, M = volume.shape
        rows = volume.reshape(-1, M)
        expression, registry = self.expression()
        raw = torch.full((I * D,), float("nan"), device=volume.device)
        for start in range(0, I * D, chunk_rows):
            stop = min(start + chunk_rows, I * D)
            raw[start:stop] = expression.evaluate(
                {"volume": rows[start:stop]}, registry
            )
        return registry.get("mean_std_blend").implementation(
            raw.reshape(I, D), self.smooth_window
        )

    @property
    def complexity(self) -> int:
        expression, registry = self.expression()
        return expression.complexity_with(registry) + registry.get("mean_std_blend").cost

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            "wait_rescue("
            f"top_k={self.top_k}, exclude_before={self.exclude_before}, "
            f"min_gap={self.min_gap}, follow={self.follow_window}, "
            f"smooth={self.smooth_window})"
        )

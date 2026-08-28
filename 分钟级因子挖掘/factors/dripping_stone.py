"""Strongly typed 滴水穿石 factor template.

The paper-faithful anchor is:
  volume -> median±3*IQR clip -> demean -> Hann window -> rFFT
         -> power share in the 2-5 minute period band.

Daily smoothing is intentionally optional because the source excerpt does not
state its window. A smoothed result is an improved variant, not the raw anchor.
"""

from dataclasses import asdict, dataclass
from typing import Any

import torch

from min_gp.dsl import LeafNode, OperatorNode, SemanticType
from min_gp.operators import build_operator_registry
from min_gp.operators.spectral import (
    DETREND_METHODS,
    IQR_MULTIPLIERS,
    PERIOD_BANDS,
    SPECTRAL_WINDOWS,
    VOLUME_TRANSFORMS,
    select_regular_session,
)
from min_gp.operators.temporal import SMOOTH_METHODS, SMOOTH_WINDOWS, smooth_daily


SESSIONS = ("all", "am", "pm")
MIN_VALID_RATIOS = (0.90, 0.95, 0.98, 1.0)


@dataclass(frozen=True)
class DrippingStoneTemplate:
    volume_transform: str = "raw"
    clip_k: float = 3.0
    min_valid_ratio: float = 0.95
    detrend: str = "demean"
    spectral_window: str = "hann"
    period_low: float = 2.0
    period_high: float = 5.0
    session: str = "all"
    smooth_method: str = "none"
    smooth_window: int = 1

    def __post_init__(self):
        if self.volume_transform not in VOLUME_TRANSFORMS:
            raise ValueError(self.volume_transform)
        if self.clip_k not in IQR_MULTIPLIERS:
            raise ValueError(self.clip_k)
        if self.min_valid_ratio not in MIN_VALID_RATIOS:
            raise ValueError(self.min_valid_ratio)
        if self.detrend not in DETREND_METHODS:
            raise ValueError(self.detrend)
        if self.spectral_window not in SPECTRAL_WINDOWS:
            raise ValueError(self.spectral_window)
        if (self.period_low, self.period_high) not in PERIOD_BANDS:
            raise ValueError((self.period_low, self.period_high))
        if self.session not in SESSIONS:
            raise ValueError(self.session)
        if self.smooth_method not in SMOOTH_METHODS:
            raise ValueError(self.smooth_method)
        if self.smooth_window not in SMOOTH_WINDOWS:
            raise ValueError(self.smooth_window)
        if self.smooth_method == "none" and self.smooth_window != 1:
            raise ValueError("smooth_window must be 1 when smoothing is disabled")

    @classmethod
    def paper_anchor(cls) -> "DrippingStoneTemplate":
        return cls()

    def raw_expression(self) -> tuple[OperatorNode, Any]:
        """Build and type-check the intraday part of the expression."""
        registry = build_operator_registry()
        leaf = LeafNode("volume", SemanticType.MINUTE_VOLUME)
        transform = OperatorNode(
            "volume_transform", (leaf,), {"mode": self.volume_transform}
        ).bind(registry)
        clipped = OperatorNode(
            "iqr_clip", (transform,), {
                "k": self.clip_k,
                "min_valid_ratio": self.min_valid_ratio,
            }
        ).bind(registry)
        detrended = OperatorNode(
            "detrend", (clipped,), {"method": self.detrend}
        ).bind(registry)
        windowed = OperatorNode(
            "spectral_window", (detrended,), {"window": self.spectral_window}
        ).bind(registry)
        spectrum = OperatorNode("fft_power", (windowed,), {}).bind(registry)
        ratio = OperatorNode(
            "band_power_ratio", (spectrum,), {
                "period_low": self.period_low,
                "period_high": self.period_high,
            }
        ).bind(registry)
        return ratio, registry

    def cost_expression(self, registry=None):
        """Build the complete session-to-daily DAG for cost admission."""
        registry = registry or build_operator_registry()
        volume = LeafNode("volume", SemanticType.MINUTE_VOLUME)
        selected = OperatorNode(
            "select_regular_session", (volume,), {"session": self.session}
        ).bind(registry)
        transformed = OperatorNode(
            "volume_transform", (selected,), {"mode": self.volume_transform}
        ).bind(registry)
        clipped = OperatorNode(
            "iqr_clip", (transformed,), {
                "k": self.clip_k,
                "min_valid_ratio": self.min_valid_ratio,
            },
        ).bind(registry)
        detrended = OperatorNode(
            "detrend", (clipped,), {"method": self.detrend}
        ).bind(registry)
        windowed = OperatorNode(
            "spectral_window", (detrended,), {"window": self.spectral_window}
        ).bind(registry)
        spectrum = OperatorNode("fft_power", (windowed,), {}).bind(registry)
        ratio = OperatorNode(
            "band_power_ratio", (spectrum,), {
                "period_low": self.period_low,
                "period_high": self.period_high,
            },
        ).bind(registry)
        if self.smooth_method == "none":
            return ratio, registry
        root = OperatorNode(
            "smooth_daily", (ratio,), {
                "method": self.smooth_method,
                "window": self.smooth_window,
            },
        ).bind(registry)
        return root, registry

    def evaluate(self, volume: torch.Tensor, chunk_rows: int = 4096) -> torch.Tensor:
        """Evaluate (I,D,M) volume to an (I,D) factor with bounded memory."""
        if volume.ndim != 3:
            raise ValueError(f"volume must be (I,D,M), got {tuple(volume.shape)}")
        selected = select_regular_session(volume, self.session)
        I, D, M = selected.shape
        rows = selected.reshape(I * D, M)
        expression, registry = self.raw_expression()
        result = torch.full(
            (I * D,), float("nan"), device=volume.device, dtype=torch.float32
        )
        for start in range(0, rows.shape[0], chunk_rows):
            stop = min(start + chunk_rows, rows.shape[0])
            result[start:stop] = expression.evaluate(
                {"volume": rows[start:stop]}, registry
            )
        daily = result.reshape(I, D)
        return smooth_daily(daily, self.smooth_method, self.smooth_window)

    @property
    def complexity(self) -> int:
        expression, registry = self.raw_expression()
        value = expression.complexity_with(registry)
        if self.smooth_method != "none":
            value += registry.get("smooth_daily").cost
        return value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        expression, _ = self.raw_expression()
        raw = f"session[{self.session}]({expression})"
        if self.smooth_method == "none":
            return raw
        return f"smooth_daily({raw}, method={self.smooth_method!r}, window={self.smooth_window})"

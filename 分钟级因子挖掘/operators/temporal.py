"""NaN-aware daily smoothing operators."""

import torch
import torch.nn.functional as F

from min_gp.dsl.registry import CostCalibration, OperatorRegistry, OperatorSpec
from min_gp.dsl.types import SemanticType
from min_gp.numeric.ranking import cross_section_rank


SMOOTH_METHODS = ("none", "mean", "ema")
SMOOTH_WINDOWS = (1, 5, 10, 20, 40)
# Longer than any level-smoothing window: rank smoothing exists to buy
# persistence, and 60/120 are where a weekly book stops rebalancing.
RANK_SMOOTH_WINDOWS = (1, 5, 10, 20, 40, 60, 120)


def _gtx1070(seconds, peak_bytes, **parameter_values):
    return CostCalibration(
        reference_shape={"I": 150, "D": 120, "M": 240},
        seconds=seconds,
        peak_bytes=peak_bytes,
        device="NVIDIA GeForce GTX 1070",
        source="local median of 7 runs, 2026-08-20",
        parameter_values=parameter_values,
    )


def _smooth_history(params):
    if params["method"] == "none" or params["window"] <= 1:
        return 0
    # The recursive EMA state depends on the complete preceding prefix.
    return None if params["method"] == "ema" else params["window"] - 1


def _window_history(params):
    return max(0, params["window"] - 1)


def smooth_daily(x: torch.Tensor, method: str, window: int) -> torch.Tensor:
    if method == "none" or window <= 1:
        return x
    x = x.float()
    if method == "mean":
        valid = torch.isfinite(x).to(x.dtype)
        xc = torch.nan_to_num(x)
        kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype)
        sums = F.conv1d(F.pad(xc.unsqueeze(1), (window - 1, 0)), kernel)[:, 0]
        counts = F.conv1d(F.pad(valid.unsqueeze(1), (window - 1, 0)), kernel)[:, 0]
        min_count = max(3, window // 2)
        out = sums / counts.clamp(min=1)
        return torch.where(
            counts >= min_count, out, torch.full_like(out, float("nan"))
        )
    if method == "ema":
        alpha = 2.0 / (window + 1.0)
        out = torch.full_like(x, float("nan"))
        state = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        initialized = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
        for d in range(x.shape[1]):
            value = x[:, d]
            valid = torch.isfinite(value)
            update = torch.where(
                initialized, alpha * value + (1.0 - alpha) * state, value
            )
            state = torch.where(valid, update, state)
            initialized |= valid
            out[:, d] = torch.where(valid, state, torch.full_like(state, float("nan")))
        return out
    raise ValueError(f"unknown smoothing method: {method}")


def rolling_daily_std(x: torch.Tensor, window: int) -> torch.Tensor:
    x = x.float()
    valid = torch.isfinite(x).to(x.dtype)
    clean = torch.nan_to_num(x)
    kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype)
    padded = (window - 1, 0)
    count = F.conv1d(F.pad(valid.unsqueeze(1), padded), kernel)[:, 0]
    total = F.conv1d(F.pad(clean.unsqueeze(1), padded), kernel)[:, 0]
    total2 = F.conv1d(F.pad((clean ** 2).unsqueeze(1), padded), kernel)[:, 0]
    mean = total / count.clamp(min=1)
    var = (total2 / count.clamp(min=1) - mean ** 2).clamp(min=0)
    min_count = max(3, window // 2)
    std = torch.sqrt(var)
    return torch.where(
        count >= min_count, std, torch.full_like(std, float("nan"))
    )


def mean_std_blend(x: torch.Tensor, window: int) -> torch.Tensor:
    mean = smooth_daily(x, "mean", window)
    std = rolling_daily_std(x, window)
    valid = torch.isfinite(mean) & torch.isfinite(std)
    value = 0.5 * (mean + std)
    return torch.where(valid, value, torch.full_like(value, float("nan")))


def equal_blend(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(a) & torch.isfinite(b)
    value = 0.5 * (a.float() + b.float())
    return torch.where(valid, value, torch.full_like(value, float("nan")))


def rank_smooth_daily(x: torch.Tensor, window: int) -> torch.Tensor:
    """Trailing mean of the daily cross-sectional rank.

    Portfolio turnover is driven by rank changes, not by level changes: a
    factor can be smooth in value and still reshuffle its ordering every
    period. Smoothing the rank damps the thing that actually triggers trades,
    which is what lets a genome satisfy the turnover gate without giving up its
    ordering information.
    """
    ranks = cross_section_rank(x.float())
    if window <= 1:
        return ranks
    valid = torch.isfinite(ranks).to(ranks.dtype)
    clean = torch.nan_to_num(ranks)
    kernel = torch.ones(1, 1, window, device=x.device, dtype=ranks.dtype)
    padded = (window - 1, 0)
    total = F.conv1d(F.pad(clean.unsqueeze(1), padded), kernel)[:, 0]
    count = F.conv1d(F.pad(valid.unsqueeze(1), padded), kernel)[:, 0]
    minimum = max(1, window // 2)
    out = total / count.clamp(min=1)
    return torch.where(count >= minimum, out, torch.full_like(out, float("nan")))


def _rank_smooth_history(params):
    return max(0, int(params.get("window", 1)) - 1)


def register_temporal_operators(registry: OperatorRegistry) -> None:
    registry.register(OperatorSpec(
        "rank_smooth_daily", (SemanticType.DAILY_RAW_FACTOR,),
        SemanticType.DAILY_FACTOR, rank_smooth_daily, cost=3,
        parameter_domains={"window": RANK_SMOOTH_WINDOWS},
        needs_full_cross_section=True,
        needs_history=True, history_days=_rank_smooth_history,
        complexity={"I": 1, "D": 1, "logI": 1},
        memory_complexity={"I": 1, "D": 1},
    ))
    registry.register(OperatorSpec(
        "smooth_daily", (SemanticType.DAILY_RAW_FACTOR,),
        SemanticType.DAILY_FACTOR, smooth_daily, cost=2,
        parameter_domains={"method": SMOOTH_METHODS, "window": SMOOTH_WINDOWS},
        needs_history=True, history_days=_smooth_history,
        complexity={"I": 1, "D": 1},
        memory_complexity={"I": 1, "D": 1},
        calibration=_gtx1070(
            0.000410699984, 524_288, method="mean", window=20
        ),
    ))
    registry.register(OperatorSpec(
        "rolling_daily_std", (SemanticType.DAILY_RAW_FACTOR,),
        SemanticType.DAILY_FACTOR, rolling_daily_std, cost=2,
        parameter_domains={"window": SMOOTH_WINDOWS[1:]},
        needs_history=True, history_days=_window_history,
        complexity={"I": 1, "D": 1},
        memory_complexity={"I": 1, "D": 1},
        calibration=_gtx1070(0.000574500009, 740_864, window=20),
    ))
    registry.register(OperatorSpec(
        "mean_std_blend", (SemanticType.DAILY_RAW_FACTOR,),
        SemanticType.DAILY_FACTOR, mean_std_blend, cost=3,
        parameter_domains={"window": SMOOTH_WINDOWS[1:]},
        needs_history=True, history_days=_window_history,
        complexity={"I": 1, "D": 1},
        memory_complexity={"I": 1, "D": 1},
        calibration=_gtx1070(0.001081100025, 813_056, window=20),
    ))
    registry.register(OperatorSpec(
        "equal_blend", (SemanticType.DAILY_FACTOR, SemanticType.DAILY_FACTOR),
        SemanticType.DAILY_FACTOR, equal_blend,
        complexity={"I": 1, "D": 1},
        memory_complexity={"I": 1, "D": 1},
        calibration=_gtx1070(0.000200600014, 235_008),
    ))

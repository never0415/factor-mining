"""Reusable intraday event operators."""

import torch

from min_gp.dsl.registry import OperatorRegistry, OperatorSpec
from min_gp.dsl.types import SemanticType


EVENT_SIGMAS = (0.5, 1.0, 1.5, 2.0)
FORWARD_WINDOWS = (3, 5, 10)
TOP_K_EVENTS = (5, 10, 15)
MIN_EVENT_GAPS = (3, 5, 10)
EXCLUDE_OPEN_MINUTES = (0, 15, 30)
EDGE_EXCLUSIONS = (0, 5, 10, 15)
DDOF_VALUES = (0, 1)
EVENT_DIRECTIONS = ("above", "below")
# log1p is the lambda -> 0 case of the Box-Cox transform the handbook asks for
# in section 10; a fitted per-series lambda is not implemented.
VOLUME_PRETRANSFORMS = ("raw", "log1p", "sqrt")
RELATIVE_LOOKBACKS = (3, 5, 10)
RELATIVE_MULTIPLES = (0.5, 1.0, 2.0)


def minute_return(close: torch.Tensor, horizon: int = 1) -> torch.Tensor:
    out = torch.full_like(close.float(), float("nan"))
    left, right = close[..., :-horizon].float(), close[..., horizon:].float()
    value = right / left.clamp(min=1e-12) - 1.0
    good = torch.isfinite(left) & torch.isfinite(right) & (left > 0)
    out[..., horizon:] = torch.where(
        good, value, torch.full_like(value, float("nan"))
    )
    return out


def minute_delta_volume(volume: torch.Tensor, horizon: int = 1) -> torch.Tensor:
    out = torch.full_like(volume.float(), float("nan"))
    left, right = volume[..., :-horizon].float(), volume[..., horizon:].float()
    value = right - left
    good = torch.isfinite(left) & torch.isfinite(right)
    out[..., horizon:] = torch.where(
        good, value, torch.full_like(value, float("nan"))
    )
    return out


def intraday_sigma_event(
    signal: torch.Tensor,
    sigma: float = 1.0,
    direction: str = "above",
    exclude_edges: int = 0,
    ddof: int = 0,
) -> torch.Tensor:
    valid = torch.isfinite(signal)
    if exclude_edges:
        valid = valid.clone()
        valid[..., :exclude_edges] = False
        valid[..., -exclude_edges:] = False
    clean = torch.nan_to_num(signal.float())
    count = valid.sum(-1, keepdim=True).clamp(min=1)
    # Excluded edge minutes are deliberately outside the estimation sample.
    # They therefore must be removed from both the numerator and denominator.
    mean = (clean * valid).sum(-1, keepdim=True) / count
    denom = (count - int(ddof)).clamp(min=1)
    var = (((clean - mean) ** 2) * valid).sum(-1, keepdim=True) / denom
    std = torch.sqrt(var.clamp(min=0))
    if direction == "above":
        event = signal > mean + float(sigma) * std
    elif direction == "below":
        event = signal < mean - float(sigma) * std
    else:
        raise ValueError(f"unknown event direction: {direction}")
    return event & valid


def forward_window_std(
    x: torch.Tensor, window: int = 5, ddof: int = 0
) -> torch.Tensor:
    """Std of [t, ..., t+window-1], aligned at event minute t."""
    x = x.float()
    out = torch.full_like(x, float("nan"))
    if x.shape[-1] < window:
        return out
    win = x.unfold(-1, window, 1)
    valid = torch.isfinite(win)
    count = valid.sum(-1)
    clean = torch.nan_to_num(win)
    mean = clean.sum(-1) / count.clamp(min=1)
    denom = (count - int(ddof)).clamp(min=1)
    var = (((clean - mean.unsqueeze(-1)) ** 2) * valid).sum(-1) / denom
    value = torch.sqrt(var.clamp(min=0))
    value = torch.where(
        count == window, value, torch.full_like(value, float("nan"))
    )
    out[..., :value.shape[-1]] = value
    return out


def masked_daily_mean(signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool() & torch.isfinite(signal)
    count = valid.sum(-1)
    value = (torch.nan_to_num(signal.float()) * valid).sum(-1) / count.clamp(min=1)
    return torch.where(count > 0, value, torch.full_like(value, float("nan")))


def topk_separated_events(
    volume: torch.Tensor,
    k: int = 10,
    exclude_before: int = 15,
    min_gap: int = 5,
) -> torch.Tensor:
    """Top-k volume minutes, then remove the later event when gap < min_gap."""
    original_shape = volume.shape
    rows = volume.float().reshape(-1, original_shape[-1])
    R, M = rows.shape
    minute = torch.arange(M, device=rows.device).view(1, M)
    eligible = torch.isfinite(rows) & (minute >= exclude_before)
    values = torch.where(eligible, rows, torch.full_like(rows, float("-inf")))
    k_eff = min(int(k), M)
    top_values, top_idx = torch.topk(values, k_eff, dim=1)
    order = torch.argsort(top_idx, dim=1)
    top_idx = torch.gather(top_idx, 1, order)
    top_values = torch.gather(top_values, 1, order)
    result = torch.zeros(R, M, device=rows.device, dtype=torch.bool)
    last = torch.full((R,), -10_000, device=rows.device, dtype=torch.long)
    row_idx = torch.arange(R, device=rows.device)
    for pos in range(k_eff):
        idx = top_idx[:, pos]
        valid = torch.isfinite(top_values[:, pos]) & ((idx - last) >= min_gap)
        if valid.any():
            result[row_idx[valid], idx[valid]] = True
            last = torch.where(valid, idx, last)
    return result.reshape(original_shape)


def follow_ratio_series(volume: torch.Tensor, window: int = 5) -> torch.Tensor:
    """Per-minute sum(V[t+1:t+window]) / V[t], aligned at t.

    Split out from `follow_ratio` because the series depends only on the
    window, while the event mask it is averaged over depends on three other
    parameters. An exhaustive sweep computes each series once and reuses it
    across every mask instead of rebuilding it per genome.
    """
    volume = volume.float()
    M = volume.shape[-1]
    ratio = torch.full_like(volume, float("nan"))
    if M <= window:
        return ratio
    future = volume[..., 1:].unfold(-1, window, 1)
    good_future = torch.isfinite(future).all(-1)
    future_sum = torch.nan_to_num(future).sum(-1)
    base = volume[..., :M - window]
    good = good_future & torch.isfinite(base) & (base > 0)
    values = future_sum / base.clamp(min=1e-12)
    ratio[..., :M - window] = torch.where(
        good, values, torch.full_like(values, float("nan"))
    )
    return ratio


def follow_ratio(
    volume: torch.Tensor, event_mask: torch.Tensor, window: int = 5
) -> torch.Tensor:
    """Mean over events of sum(V[t+1:t+window]) / V[t]."""
    return masked_daily_mean(follow_ratio_series(volume, window), event_mask)


# ──────────────────────────────────────────────
# Detector parts: minute volume -> event mask.
#
# Every handbook event factor starts by turning the volume series into a set of
# "interesting" minutes, and each report picks a different rule. Registering
# them under one signature makes the rule itself a searchable slot instead of a
# hard-coded call, so 适度冒险's spike test and 待著而救's top-k test become
# interchangeable parts rather than separate factors.
# ──────────────────────────────────────────────

def delta_sigma_event(
    volume: torch.Tensor,
    transform: str = "raw",
    sigma: float = 1.0,
    direction: str = "above",
    exclude_edges: int = 0,
    ddof: int = 0,
) -> torch.Tensor:
    """适度冒险 (section 3): minutes whose volume increment exceeds mu + sigma*std."""
    from min_gp.operators.spectral import volume_transform
    signal = minute_delta_volume(volume_transform(volume, transform), horizon=1)
    return intraday_sigma_event(
        signal, sigma=sigma, direction=direction,
        exclude_edges=exclude_edges, ddof=ddof,
    )


def relative_volume_event(
    volume: torch.Tensor,
    lookback: int = 5,
    multiple: float = 1.0,
    exclude_edges: int = 0,
) -> torch.Tensor:
    """暗流涌动 (section 12): V_t / mean(V_(t-lookback..t-1)) - 1 > multiple."""
    volume = volume.float()
    M = volume.shape[-1]
    event = torch.zeros_like(volume, dtype=torch.bool)
    if M <= lookback:
        return event
    past = volume[..., :-1].unfold(-1, lookback, 1)
    good_past = torch.isfinite(past).all(-1)
    baseline = torch.nan_to_num(past).mean(-1)
    current = volume[..., lookback:]
    ratio = current / baseline.clamp(min=1e-12) - 1.0
    hit = good_past & torch.isfinite(current) & (baseline > 0) & (ratio > multiple)
    event[..., lookback:] = hit
    if exclude_edges:
        event[..., :exclude_edges] = False
        event[..., -exclude_edges:] = False
    return event


# ──────────────────────────────────────────────
# Statistic parts: what is measured at (or after) an event minute.
# ──────────────────────────────────────────────

def point_signal(x: torch.Tensor) -> torch.Tensor:
    """Identity: score the event minute itself rather than a forward window.

    Section 3.3 stresses that the return component takes the spike minute's own
    return, not the following five minutes, so "no window" has to be one of the
    choices the search can make.
    """
    return x.float()


def forward_window_mean(x: torch.Tensor, window: int = 5) -> torch.Tensor:
    """Mean of [t, ..., t+window-1], aligned at event minute t."""
    x = x.float()
    out = torch.full_like(x, float("nan"))
    if x.shape[-1] < window:
        return out
    win = x.unfold(-1, window, 1)
    valid = torch.isfinite(win)
    count = valid.sum(-1)
    value = torch.nan_to_num(win).sum(-1) / count.clamp(min=1)
    value = torch.where(
        count == window, value, torch.full_like(value, float("nan"))
    )
    out[..., :value.shape[-1]] = value
    return out


def log_follow_ratio_series(volume: torch.Tensor, window: int = 5) -> torch.Tensor:
    """log1p of the follow ratio: the same idea on a symmetric scale."""
    return torch.log1p(follow_ratio_series(volume, window).clamp(min=0))


# ──────────────────────────────────────────────
# Aggregator parts: (signal, mask) -> one value per stock-day.
# ──────────────────────────────────────────────

def masked_daily_median(signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool() & torch.isfinite(signal)
    filled = torch.where(valid, signal.float(), torch.full_like(signal, float("nan")))
    value = torch.nanmedian(filled, dim=-1).values
    return torch.where(
        valid.any(-1), value, torch.full_like(value, float("nan"))
    )


def masked_daily_std(signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool() & torch.isfinite(signal)
    count = valid.sum(-1)
    clean = torch.nan_to_num(signal.float()) * valid
    mean = clean.sum(-1) / count.clamp(min=1)
    var = (((clean - mean.unsqueeze(-1)) ** 2) * valid).sum(-1) / count.clamp(min=1)
    value = torch.sqrt(var.clamp(min=0))
    return torch.where(count >= 2, value, torch.full_like(value, float("nan")))


def masked_daily_count(signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """How many events fired, ignoring the signal's values."""
    valid = mask.bool() & torch.isfinite(signal)
    return valid.sum(-1).to(torch.float32)


def register_event_operators(registry: OperatorRegistry) -> None:
    registry.register(OperatorSpec(
        "minute_return", (SemanticType.MINUTE_PRICE,), SemanticType.MINUTE_RETURN,
        minute_return, parameter_domains={"horizon": (1, 5)},
    ))
    registry.register(OperatorSpec(
        "minute_delta_volume", (SemanticType.MINUTE_VOLUME,),
        SemanticType.MINUTE_SIGNAL, minute_delta_volume,
        parameter_domains={"horizon": (1, 5)},
    ))
    registry.register(OperatorSpec(
        "intraday_sigma_event", (SemanticType.MINUTE_SIGNAL,),
        SemanticType.MINUTE_MASK, intraday_sigma_event,
        parameter_domains={
            "sigma": EVENT_SIGMAS,
            "direction": ("above", "below"),
            "exclude_edges": EDGE_EXCLUSIONS,
            "ddof": DDOF_VALUES,
        },
    ))
    registry.register(OperatorSpec(
        "forward_window_std", (SemanticType.MINUTE_RETURN,),
        SemanticType.MINUTE_SIGNAL, forward_window_std,
        parameter_domains={"window": FORWARD_WINDOWS, "ddof": DDOF_VALUES},
        intraday_lookahead_minutes=lambda p: p["window"] - 1,
    ))
    registry.register(OperatorSpec(
        "masked_daily_mean_signal",
        (SemanticType.MINUTE_SIGNAL, SemanticType.MINUTE_MASK),
        SemanticType.DAILY_RAW_FACTOR, masked_daily_mean,
    ))
    registry.register(OperatorSpec(
        "masked_daily_mean_return",
        (SemanticType.MINUTE_RETURN, SemanticType.MINUTE_MASK),
        SemanticType.DAILY_RAW_FACTOR, masked_daily_mean,
    ))
    registry.register(OperatorSpec(
        "topk_separated_events", (SemanticType.MINUTE_VOLUME,),
        SemanticType.MINUTE_MASK, topk_separated_events,
        parameter_domains={
            "k": TOP_K_EVENTS,
            "exclude_before": EXCLUDE_OPEN_MINUTES,
            "min_gap": MIN_EVENT_GAPS,
        },
    ))
    registry.register(OperatorSpec(
        "follow_ratio", (SemanticType.MINUTE_VOLUME, SemanticType.MINUTE_MASK),
        SemanticType.DAILY_RAW_FACTOR, follow_ratio,
        parameter_domains={"window": FORWARD_WINDOWS},
        intraday_lookahead_minutes=lambda p: p["window"],
    ))

    # ── detector slot: minute volume -> event mask ──
    registry.register(OperatorSpec(
        "delta_sigma_event", (SemanticType.MINUTE_VOLUME,),
        SemanticType.MINUTE_MASK, delta_sigma_event, cost=2,
        parameter_domains={
            "transform": VOLUME_PRETRANSFORMS,
            "sigma": EVENT_SIGMAS,
            "direction": EVENT_DIRECTIONS,
            "exclude_edges": EDGE_EXCLUSIONS,
            "ddof": DDOF_VALUES,
        },
    ))
    registry.register(OperatorSpec(
        "relative_volume_event", (SemanticType.MINUTE_VOLUME,),
        SemanticType.MINUTE_MASK, relative_volume_event, cost=2,
        parameter_domains={
            "lookback": RELATIVE_LOOKBACKS,
            "multiple": RELATIVE_MULTIPLES,
            "exclude_edges": EDGE_EXCLUSIONS,
        },
    ))

    # ── statistic slot ──
    registry.register(OperatorSpec(
        "point_signal", (SemanticType.MINUTE_RETURN,), SemanticType.MINUTE_SIGNAL,
        point_signal,
    ))
    registry.register(OperatorSpec(
        "forward_window_mean", (SemanticType.MINUTE_RETURN,),
        SemanticType.MINUTE_SIGNAL, forward_window_mean,
        parameter_domains={"window": FORWARD_WINDOWS},
        intraday_lookahead_minutes=lambda p: p["window"] - 1,
    ))
    registry.register(OperatorSpec(
        "follow_ratio_series", (SemanticType.MINUTE_VOLUME,),
        SemanticType.MINUTE_SIGNAL, follow_ratio_series,
        parameter_domains={"window": FORWARD_WINDOWS},
        intraday_lookahead_minutes=lambda p: p["window"],
    ))
    registry.register(OperatorSpec(
        "log_follow_ratio_series", (SemanticType.MINUTE_VOLUME,),
        SemanticType.MINUTE_SIGNAL, log_follow_ratio_series,
        parameter_domains={"window": FORWARD_WINDOWS},
        intraday_lookahead_minutes=lambda p: p["window"],
    ))

    # ── aggregator slot ──
    registry.register(OperatorSpec(
        "masked_daily_median",
        (SemanticType.MINUTE_SIGNAL, SemanticType.MINUTE_MASK),
        SemanticType.DAILY_RAW_FACTOR, masked_daily_median,
    ))
    registry.register(OperatorSpec(
        "masked_daily_std",
        (SemanticType.MINUTE_SIGNAL, SemanticType.MINUTE_MASK),
        SemanticType.DAILY_RAW_FACTOR, masked_daily_std,
    ))
    registry.register(OperatorSpec(
        "masked_daily_count",
        (SemanticType.MINUTE_SIGNAL, SemanticType.MINUTE_MASK),
        SemanticType.DAILY_RAW_FACTOR, masked_daily_count,
    ))

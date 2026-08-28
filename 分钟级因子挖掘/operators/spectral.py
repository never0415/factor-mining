"""Spectral operators for volume-periodicity factors such as 滴水穿石."""

import torch

from min_gp.dsl.registry import OperatorRegistry, OperatorSpec
from min_gp.dsl.types import SemanticType


VOLUME_TRANSFORMS = ("raw", "log1p", "sqrt")
DETREND_METHODS = ("demean", "linear")
SPECTRAL_WINDOWS = ("hann", "hamming", "none")
IQR_MULTIPLIERS = (1.5, 2.0, 3.0, 4.0)
PERIOD_BANDS = (
    (2.0, 3.0),
    (2.0, 4.0),
    (2.0, 5.0),
    (3.0, 5.0),
    (3.0, 8.0),
    (5.0, 10.0),
)


def select_regular_session(x: torch.Tensor, session: str = "all") -> torch.Tensor:
    """Return contiguous trading-minute samples, removing the legacy 11:30 gap.

    Legacy tensors have 241 slots: 09:30..11:30 and 13:00..14:59, but the
    source data contains 09:30..11:29 plus 13:00..14:59 (240 observations).
    """
    n = x.shape[-1]
    if n == 241:
        x = torch.cat((x[..., :120], x[..., 121:]), dim=-1)
    elif n != 240:
        raise ValueError(f"expected 240 or 241 minute slots, got {n}")
    if session == "all":
        return x
    if session == "am":
        return x[..., :120]
    if session == "pm":
        return x[..., 120:]
    raise ValueError(f"unknown session: {session}")


def volume_transform(x: torch.Tensor, mode: str) -> torch.Tensor:
    x = x.float()
    if mode == "raw":
        return x
    if mode == "log1p":
        return torch.log1p(x.clamp(min=0))
    if mode == "sqrt":
        return torch.sqrt(x.clamp(min=0))
    raise ValueError(f"unknown volume transform: {mode}")


def iqr_clip(x: torch.Tensor, k: float, min_valid_ratio: float = 0.95) -> torch.Tensor:
    """Per-row median ± k×IQR clipping with deterministic missing handling.

    Rows below min_valid_ratio remain all-NaN. Short gaps in otherwise complete
    rows are filled with the row median before the FFT; no cross-day fill occurs.
    """
    x = x.float()
    n = x.shape[-1]
    valid = torch.isfinite(x)
    enough = valid.sum(-1) >= int(n * min_valid_ratio + 0.999999)
    q25 = torch.nanquantile(x, 0.25, dim=-1, keepdim=True)
    q75 = torch.nanquantile(x, 0.75, dim=-1, keepdim=True)
    med = torch.nanmedian(x, dim=-1, keepdim=True).values
    spread = q75 - q25
    filled = torch.where(valid, x, med)
    clipped = torch.maximum(
        torch.minimum(filled, med + float(k) * spread),
        med - float(k) * spread,
    )
    return torch.where(
        enough.unsqueeze(-1), clipped, torch.full_like(clipped, float("nan"))
    )


def detrend_signal(x: torch.Tensor, method: str) -> torch.Tensor:
    x = x.float()
    mean = x.mean(-1, keepdim=True)
    centered = x - mean
    if method == "demean":
        return centered
    if method == "linear":
        t = torch.linspace(-1.0, 1.0, x.shape[-1], device=x.device, dtype=x.dtype)
        denom = (t * t).sum().clamp(min=1e-12)
        beta = (centered * t).sum(-1, keepdim=True) / denom
        return centered - beta * t
    raise ValueError(f"unknown detrend method: {method}")


def apply_spectral_window(x: torch.Tensor, window: str) -> torch.Tensor:
    if window == "none":
        return x
    if window == "hann":
        w = torch.hann_window(
            x.shape[-1], periodic=False, device=x.device, dtype=x.dtype
        )
    elif window == "hamming":
        w = torch.hamming_window(
            x.shape[-1], periodic=False, device=x.device, dtype=x.dtype
        )
    else:
        raise ValueError(f"unknown spectral window: {window}")
    return x * w


def fft_power(x: torch.Tensor) -> torch.Tensor:
    """One-sided power spectrum; input sampling interval is one trading minute."""
    return torch.abs(torch.fft.rfft(x.float(), dim=-1)) ** 2


def band_power_ratio(
    power: torch.Tensor, period_low: float, period_high: float
) -> torch.Tensor:
    """Power share whose period lies in [period_low, period_high] minutes."""
    if not (1.0 < period_low < period_high):
        raise ValueError(
            f"invalid period band: [{period_low}, {period_high}]"
        )
    n = 2 * (power.shape[-1] - 1)
    freq = torch.fft.rfftfreq(n, d=1.0, device=power.device)
    band = (freq >= 1.0 / period_high) & (freq <= 1.0 / period_low)
    non_dc = freq > 0
    numerator = power[..., band].sum(-1)
    denominator = power[..., non_dc].sum(-1)
    result = numerator / denominator.clamp(min=1e-20)
    good = torch.isfinite(denominator) & (denominator > 1e-20)
    return torch.where(good, result, torch.full_like(result, float("nan")))


def spectral_entropy(power: torch.Tensor) -> torch.Tensor:
    """Normalized entropy over non-zero frequencies, in [0, 1]."""
    p = power[..., 1:].float()
    total = p.sum(-1, keepdim=True)
    prob = p / total.clamp(min=1e-20)
    entropy = -(prob * torch.log(prob.clamp(min=1e-20))).sum(-1)
    scale = torch.log(torch.tensor(p.shape[-1], device=p.device, dtype=p.dtype))
    out = entropy / scale.clamp(min=1.0)
    return torch.where(
        total.squeeze(-1) > 1e-20, out, torch.full_like(out, float("nan"))
    )


def register_spectral_operators(registry: OperatorRegistry) -> None:
    registry.register(OperatorSpec(
        "select_regular_session", (SemanticType.MINUTE_VOLUME,),
        SemanticType.MINUTE_VOLUME, select_regular_session,
        parameter_domains={"session": ("all", "am", "pm")},
    ))
    registry.register(OperatorSpec(
        "volume_transform", (SemanticType.MINUTE_VOLUME,),
        SemanticType.MINUTE_VOLUME, volume_transform,
        parameter_domains={"mode": VOLUME_TRANSFORMS},
    ))
    registry.register(OperatorSpec(
        "iqr_clip", (SemanticType.MINUTE_VOLUME,), SemanticType.MINUTE_VOLUME,
        iqr_clip, parameter_domains={
            "k": IQR_MULTIPLIERS,
            "min_valid_ratio": (0.90, 0.95, 0.98, 1.0),
        },
    ))
    registry.register(OperatorSpec(
        "detrend", (SemanticType.MINUTE_VOLUME,), SemanticType.MINUTE_SIGNAL,
        detrend_signal, parameter_domains={"method": DETREND_METHODS},
    ))
    registry.register(OperatorSpec(
        "spectral_window", (SemanticType.MINUTE_SIGNAL,),
        SemanticType.MINUTE_SIGNAL, apply_spectral_window,
        parameter_domains={"window": SPECTRAL_WINDOWS},
    ))
    registry.register(OperatorSpec(
        "fft_power", (SemanticType.MINUTE_SIGNAL,), SemanticType.SPECTRUM,
        fft_power, cost=8,
    ))
    registry.register(OperatorSpec(
        "band_power_ratio", (SemanticType.SPECTRUM,),
        SemanticType.DAILY_RAW_FACTOR, band_power_ratio,
        parameter_domains={
            "period_low": tuple(sorted({b[0] for b in PERIOD_BANDS})),
            "period_high": tuple(sorted({b[1] for b in PERIOD_BANDS})),
        },
    ))
    registry.register(OperatorSpec(
        "spectral_entropy", (SemanticType.SPECTRUM,),
        SemanticType.DAILY_RAW_FACTOR, spectral_entropy, cost=2,
    ))

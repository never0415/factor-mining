"""Reusable OHLC dispersion, return-ratio and intraday-state operators."""

import torch

from min_gp.dsl import (
    CostCalibration, OperatorRegistry, OperatorSpec, SemanticType,
)
from min_gp.operators.event import minute_return


DISPERSION_WINDOWS = (3, 5, 10)
STATE_SIGMAS = (0.5, 1.0, 1.5, 2.0)


def _gtx1070(seconds, peak_bytes, **parameter_values):
    return CostCalibration(
        reference_shape={"I": 150, "D": 120, "M": 240},
        seconds=seconds,
        peak_bytes=peak_bytes,
        device="NVIDIA GeForce GTX 1070",
        source="local median of 7 runs, 2026-08-20",
        parameter_values=parameter_values,
    )


_CLIMB_CALIBRATIONS = {
    "rolling_ohlc_dispersion": _gtx1070(
        0.034306999994, 684_189_696, window=5
    ),
    "close_minute_return": _gtx1070(
        0.002652700001, 73_417_728, horizon=1
    ),
    "safe_signal_ratio": _gtx1070(
        0.001977399981, 56_160_256, floor=1e-12
    ),
    "extreme_high_state": _gtx1070(
        0.002414099989, 56_461_824, sigma=1.0
    ),
}


def rolling_ohlc_dispersion(open_, high, low, close, window=5):
    """Squared coefficient of variation of the last ``4*window`` OHLC values."""
    prices = torch.stack((open_, high, low, close), dim=-1).float()
    out = torch.full_like(close.float(), float("nan"))
    if close.shape[-1] < window:
        return out
    # ``-2`` is the minute axis both before and after (I,D) row flattening.
    win = prices.unfold(-2, window, 1)
    # (..., 4, window) -> chronological groups of four OHLC observations.
    flat = win.transpose(-1, -2).flatten(-2)
    good = torch.isfinite(flat).all(-1)
    value = (
        flat.std(-1, unbiased=False) / flat.mean(-1).clamp(min=1e-12)
    ).square()
    out[..., window - 1:] = torch.where(
        good, value, torch.full_like(value, float("nan"))
    )
    return out


def rolling_ohlc_range_dispersion(open_, high, low, close, window=5):
    """Alternative rolling mean of relative OHLC ranges."""
    del open_
    amplitude = (high.float() - low.float()) / close.float().clamp(min=1e-12)
    out = torch.full_like(amplitude, float("nan"))
    if amplitude.shape[-1] < window:
        return out
    win = amplitude.unfold(-1, window, 1)
    good = torch.isfinite(win).all(-1)
    value = win.mean(-1)
    out[..., window - 1:] = torch.where(
        good, value, torch.full_like(value, float("nan"))
    )
    return out


def close_minute_return(close, horizon=1):
    return minute_return(close, horizon)


def close_minute_log_return(close, horizon=1):
    simple = minute_return(close, horizon)
    return torch.log1p(simple.clamp(min=-1 + 1e-12))


def safe_signal_ratio(numerator, denominator, floor=1e-12):
    """Protected ratio while preserving NaNs in either source signal."""
    valid = torch.isfinite(numerator) & torch.isfinite(denominator)
    value = numerator.float() / denominator.float().clamp(min=floor)
    return torch.where(valid, value, torch.full_like(value, float("nan")))


def signed_signal_ratio(numerator, denominator, floor=1e-12):
    valid = torch.isfinite(numerator) & torch.isfinite(denominator)
    scale = denominator.float().abs().clamp(min=floor)
    value = numerator.float() / scale
    return torch.where(valid, value, torch.full_like(value, float("nan")))


def _extreme_state(signal, sigma, high_state):
    valid = torch.isfinite(signal)
    count = valid.sum(-1, keepdim=True).clamp(min=1)
    clean = signal.float().nan_to_num()
    mean = (clean * valid).sum(-1, keepdim=True) / count
    variance = (((clean - mean) ** 2) * valid).sum(-1, keepdim=True) / count
    boundary = float(sigma) * torch.sqrt(variance.clamp(min=0))
    selected = signal >= mean + boundary if high_state else signal <= mean - boundary
    return selected & valid


def extreme_high_state(signal, sigma=1.0):
    return _extreme_state(signal, sigma, True)


def extreme_low_state(signal, sigma=1.0):
    return _extreme_state(signal, sigma, False)


def register_intraday_operators(registry: OperatorRegistry):
    t = SemanticType
    ohlc = (t.MINUTE_OPEN, t.MINUTE_HIGH, t.MINUTE_LOW, t.MINUTE_CLOSE)
    for name, implementation in (
        ("rolling_ohlc_dispersion", rolling_ohlc_dispersion),
        ("rolling_ohlc_range_dispersion", rolling_ohlc_range_dispersion),
    ):
        registry.register(OperatorSpec(
            name, ohlc, t.MINUTE_SIGNAL, implementation, cost=4,
            parameter_domains={"window": DISPERSION_WINDOWS},
            complexity={"I": 1, "D": 1, "M": 1},
            memory_complexity={"I": 1, "D": 1, "M": 1},
            calibration=_CLIMB_CALIBRATIONS.get(name),
        ))
    for name, implementation in (
        ("close_minute_return", close_minute_return),
        ("close_minute_log_return", close_minute_log_return),
    ):
        registry.register(OperatorSpec(
            name, (t.MINUTE_CLOSE,), t.MINUTE_RETURN, implementation,
            parameter_domains={"horizon": (1, 5)},
            complexity={"I": 1, "D": 1, "M": 1},
            memory_complexity={"I": 1, "D": 1, "M": 1},
            calibration=_CLIMB_CALIBRATIONS.get(name),
        ))
    for name, implementation in (
        ("safe_signal_ratio", safe_signal_ratio),
        ("signed_signal_ratio", signed_signal_ratio),
    ):
        registry.register(OperatorSpec(
            name, (t.MINUTE_RETURN, t.MINUTE_SIGNAL), t.MINUTE_SIGNAL,
            implementation, cost=2,
            parameter_domains={"floor": (1e-12, 1e-8, 1e-4)},
            complexity={"I": 1, "D": 1, "M": 1},
            memory_complexity={"I": 1, "D": 1, "M": 1},
            calibration=_CLIMB_CALIBRATIONS.get(name),
        ))
    for name, implementation in (
        ("extreme_high_state", extreme_high_state),
        ("extreme_low_state", extreme_low_state),
    ):
        registry.register(OperatorSpec(
            name, (t.MINUTE_SIGNAL,), t.MINUTE_MASK, implementation, cost=2,
            parameter_domains={"sigma": STATE_SIGMAS},
            complexity={"I": 1, "D": 1, "M": 1},
            memory_complexity={"I": 1, "D": 1, "M": 1},
            calibration=_CLIMB_CALIBRATIONS.get(name),
        ))

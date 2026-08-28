"""Typed, reusable operators for the 59 original expression-tree seeds.

This module intentionally does not import :mod:`min_gp.expr`.  The old parser
remains available to the historical backtest CLI, while production typed trees
use these independently registered numerical kernels.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from min_gp.dsl.registry import OperatorRegistry, OperatorSpec
from min_gp.dsl.types import SemanticType
from min_gp.numeric.ranking import cross_section_rank


# 60 加入于 2026-08-28: 972 组参数网格显示 ts_std 窗口是换手的主要旋钮, 而换手
# 是唯一可靠延续到样本外的指标 (样本期-测试期秩相关 +0.993)。窗口从 20 拉到 60,
# 周均换手 57.5% -> 36.8% 而毛多空几乎不动。字母表原先止于 40, 意味着 GP 在结构上
# 就到不了低换手的那一片区域。这十个滚动算子共用本元组, 因此放宽对它们一并生效。
#
# 待办 — window=60 尚未做成本标定。加入新取值会改变这十个算子的 spec_fingerprint,
# 于是 test_seed_calibration / test_nonseed_calibration 会报
# cost_calibrated_operators 111 != 121 (少的正好是这十个)。已知且刻意暂缓:
#   * 报告与回测不受影响 —— 它们直接求值, 不走 gp/cost_control。
#   * 但 GP 搜索会受影响: 未标定算子落入 max_uncalibrated_cost_units 预算, 而
#     allow_uncalibrated 默认 False, 含 window=60 的个体会被挡掉。
# 重跑 v7 之前必须先补标定:
#   python -m min_gp.benchmark_seed_operators
WINDOWS = (1, 5, 10, 20, 40, 60)
QUANTILES = (0.25, 0.5, 0.75, 0.8)
MASK_STATS = tuple(range(9))


def _float(x):
    if not isinstance(x, torch.Tensor):
        return x
    return x if x.dtype.is_floating_point else x.to(torch.bfloat16)


def _binary(fn):
    def apply(a, b):
        if isinstance(a, torch.Tensor):
            a = _float(a)
        if isinstance(b, torch.Tensor):
            b = _float(b)
        return fn(a, b)
    return apply


def _comparison(predicate):
    def apply(a, b):
        tensor = a if isinstance(a, torch.Tensor) else b
        left = _float(a) if isinstance(a, torch.Tensor) else a
        right = _float(b) if isinstance(b, torch.Tensor) else b
        return predicate(left, right).to(_float(tensor).dtype)
    return apply


def _safe_div(a, b):
    a = _float(a) if isinstance(a, torch.Tensor) else a
    b = _float(b) if isinstance(b, torch.Tensor) else b
    if not isinstance(b, torch.Tensor):
        b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
    eps = 1e-6
    sign = torch.sign(b)
    protected = torch.where(sign == 0, torch.ones_like(sign), sign) * eps
    protected = torch.where(b.abs() >= eps, b, protected)
    return a / protected


def _sqrt(x):
    return torch.sqrt(torch.clamp(_float(x), min=0))


def _shift_minute(x, shift):
    # Legacy convention: +1 means take the next minute at the current minute.
    shift = int(shift)
    out = torch.zeros_like(x)
    if shift > 0:
        out[..., :-shift] = x[..., shift:]
    elif shift < 0:
        out[..., -shift:] = x[..., :shift]
    else:
        out.copy_(x)
    return out


def _intra_roll(x, window, std=False):
    x = _float(x)
    window = int(window)
    shape = x.shape
    rows = x.reshape(-1, 1, shape[-1])
    clean = torch.nan_to_num(rows)
    valid = torch.isfinite(rows).to(x.dtype)
    kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype)
    count = F.conv1d(F.pad(valid, (window - 1, 0)), kernel).clamp(min=1)
    mean = F.conv1d(F.pad(clean, (window - 1, 0)), kernel) / count
    if not std:
        return mean.reshape(shape)
    second = F.conv1d(F.pad(clean.square(), (window - 1, 0)), kernel) / count
    return torch.sqrt((second - mean.square()).clamp(min=0)).reshape(shape)


def _history_rows(x):
    """Flatten D2 or (I,M,D) without losing the same-minute grouping."""
    return x.reshape(-1, x.shape[-1]), x.shape


def _rolling(x, window, mode):
    x = _float(x)
    window = int(window)
    rows, shape = _history_rows(x)
    if window <= 1:
        return x.clone()
    clean = torch.nan_to_num(rows)
    valid = torch.isfinite(rows).to(rows.dtype)
    kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype)
    count = F.conv1d(F.pad(valid[:, None], (window - 1, 0)), kernel)[:, 0]
    if mode in ("sum", "mean"):
        out = F.conv1d(F.pad(clean[:, None], (window - 1, 0)), kernel)[:, 0]
        if mode == "mean":
            out = out / count.clamp(min=1)
    elif mode == "max":
        values = torch.nan_to_num(rows, nan=-float("inf"))
        out = F.max_pool1d(
            F.pad(values[:, None], (window - 1, 0), value=-float("inf")),
            window, 1,
        )[:, 0]
        out = torch.where(
            torch.isneginf(out), torch.full_like(out, float("nan")), out
        )
    else:
        raise ValueError(mode)
    out = torch.where(count >= 1, out, torch.full_like(out, float("nan")))
    return out.reshape(shape)


def _ts_std(x, window):
    mean = _rolling(x, window, "mean")
    second = _rolling(_float(x).square(), window, "mean")
    return torch.sqrt((second - mean.square()).clamp(min=0))


def _ts_delay(x, window):
    delay = int(window)
    out = torch.full_like(x, float("nan"))
    if delay > 0:
        out[..., delay:] = x[..., :-delay]
    elif delay < 0:
        out[..., :delay] = x[..., -delay:]
    else:
        out.copy_(x)
    return out


def _rolling_pair(x, y, window):
    x, y = _float(x), _float(y)
    dtype = torch.promote_types(x.dtype, y.dtype)
    x, y = x.to(dtype), y.to(dtype)
    xr, shape = _history_rows(x)
    yr, _ = _history_rows(y)
    valid = torch.isfinite(xr) & torch.isfinite(yr)
    clean_x, clean_y = torch.nan_to_num(xr), torch.nan_to_num(yr)
    terms = torch.stack(
        (clean_x, clean_y, clean_x * clean_y, clean_x.square(), clean_y.square()),
        dim=1,
    )
    kernel = torch.ones(5, 1, int(window), device=x.device, dtype=dtype)
    sums = F.conv1d(F.pad(terms, (int(window) - 1, 0)), kernel, groups=5)
    one = torch.ones(1, 1, int(window), device=x.device, dtype=dtype)
    count = F.conv1d(
        F.pad(valid.to(dtype)[:, None], (int(window) - 1, 0)), one
    )[:, 0]
    n = count.clamp(min=1)
    sx, sy, sxy, sx2, sy2 = (sums[:, i] for i in range(5))
    mx, my = sx / n, sy / n
    cov = sxy / n - mx * my
    vx = (sx2 / n - mx.square()).clamp(min=0)
    vy = (sy2 / n - my.square()).clamp(min=0)
    nan = torch.full_like(cov, float("nan"))
    good = count >= 1
    return tuple(torch.where(good, z, nan).reshape(shape) for z in (cov, vx, vy))


def _day_reduce(x, mode):
    x = _float(x)
    # Minute cube is (I,D,M); same-minute history is (I,M,D).
    axis = 1 if x.ndim == 3 and getattr(x, "_same_minute", False) else -1
    # Tensors cannot carry arbitrary metadata.  The registered B2 wrappers call
    # `_day_reduce_same_minute` instead.
    valid = torch.isfinite(x)
    clean = torch.nan_to_num(x)
    count = valid.sum(axis).clamp(min=1)
    if mode == "sum":
        return (clean * valid).sum(axis)
    if mode == "mean":
        return (clean * valid).sum(axis) / count
    if mode == "std":
        mean = (clean * valid).sum(axis) / count
        var = ((clean - mean.unsqueeze(axis)).square() * valid).sum(axis) / count
        return torch.sqrt(var.clamp(min=0))
    filled = torch.where(
        valid, clean, torch.full_like(clean, float("inf") if mode == "min" else -float("inf"))
    )
    value = filled.amin(axis) if mode == "min" else filled.amax(axis)
    return torch.where(valid.any(axis), value, torch.full_like(value, float("nan")))


def _day_reduce_same_minute(x, mode):
    # (I,M,D) -> (I,D), reducing the minute-of-day axis.
    return _day_reduce(x.permute(0, 2, 1), mode)


def _day_quantile(x, quantile):
    x = _float(x)
    valid = torch.isfinite(x)
    count = valid.sum(-1).clamp(min=1)
    pos = (float(quantile) * (count - 1)).long().clamp(min=0, max=x.shape[-1] - 1)
    # Preserve the historical kernel's exact zero-fill/sort behavior.
    sorted_values = torch.nan_to_num(x).sort(-1).values
    return sorted_values.gather(-1, pos.unsqueeze(-1)).squeeze(-1)


def _day_last(x):
    x = _float(x)
    valid = torch.isfinite(x)
    last = valid.cumsum(-1) == valid.sum(-1, keepdim=True)
    return torch.where(last, x, torch.zeros_like(x)).sum(-1)


def _day_first(x):
    x = _float(x)
    valid = torch.isfinite(x)
    first = valid.cumsum(-1) == 1
    return torch.where(first, x, torch.zeros_like(x)).sum(-1)


def _day_corr(x, y):
    x, y = _float(x), _float(y)
    valid = torch.isfinite(x) & torch.isfinite(y)
    clean_x, clean_y = torch.nan_to_num(x), torch.nan_to_num(y)
    count = valid.sum(-1).clamp(min=2).to(x.dtype)
    mx = (clean_x * valid).sum(-1) / count
    my = (clean_y * valid).sum(-1) / count
    dx, dy = clean_x - mx[..., None], clean_y - my[..., None]
    cov = (dx * dy * valid).sum(-1) / count
    vx = (dx.square() * valid).sum(-1) / count
    vy = (dy.square() * valid).sum(-1) / count
    return cov / torch.sqrt((vx * vy).clamp(min=1e-6))


def _time_barycenter(x):
    x = _float(x)
    valid = torch.isfinite(x)
    clean = torch.nan_to_num(x)
    minute = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype)
    return (clean * valid * minute).sum(-1) / (clean * valid).sum(-1).abs().clamp(min=1e-6)


def _broadcast_daily(x, *, _context):
    minute_count = next(
        value.shape[-1] for value in _context.values()
        if isinstance(value, torch.Tensor) and value.ndim >= 2
    )
    return x.unsqueeze(-1).expand(*x.shape, minute_count)


def _mask_mul(x, session_mask):
    return _float(x) * _float(session_mask).reshape(1, 1, -1)


def _mask_agg(x, mask, statistic):
    x = _float(x)
    if isinstance(mask, torch.Tensor):
        selected = _float(mask) > 0.5
    else:
        selected = torch.ones_like(x, dtype=torch.bool)
    values = torch.where(selected, x, torch.full_like(x, float("nan")))
    weights = selected.to(x.dtype)
    statistic = int(statistic)
    count = weights.sum(-1)
    if statistic == 0:
        daily = count
    elif statistic == 1:
        daily = torch.nan_to_num(values).sum(-1)
    elif statistic == 2:
        daily = torch.nan_to_num(values).sum(-1) / count.clamp(min=1)
    elif statistic in (3, 4, 5):
        mean = torch.nan_to_num(values).sum(-1) / count.clamp(min=1)
        centered = torch.nan_to_num(values - mean[..., None]) * weights
        variance = centered.square().sum(-1) / count.clamp(min=1)
        std = torch.sqrt(variance.clamp(min=1e-12))
        if statistic == 3:
            daily = torch.sqrt(variance.clamp(min=0))
        elif statistic == 4:
            daily = (centered ** 3).sum(-1) / count.clamp(min=1) / std ** 3
        else:
            daily = (centered ** 4).sum(-1) / count.clamp(min=1) / std ** 4
    elif statistic == 6:
        daily = torch.where(selected, x, torch.full_like(x, float("inf"))).amin(-1)
    elif statistic == 7:
        daily = torch.where(selected, x, torch.full_like(x, -float("inf"))).amax(-1)
    else:
        daily = torch.where(selected, x, torch.full_like(x, float("inf"))).sort(-1).values
        index = ((count.long().clamp(min=1) - 1) // 2).unsqueeze(-1)
        daily = daily.gather(-1, index).squeeze(-1)
    return torch.where(count > 0, daily, torch.full_like(daily, float("nan")))


def _mask_ratio(x, a, b, window):
    x = _float(x)
    left = (torch.nan_to_num(x) * (_float(a) > 0.5)).sum(-1)
    right = (torch.nan_to_num(x) * (_float(b) > 0.5)).sum(-1)
    left = _rolling(left, window, "sum")
    right = _rolling(right, window, "sum")
    out = left / (right + 1)
    out[..., :int(window) - 1] = float("nan")
    return out


def _dist_to_event(mask):
    mask = _float(mask) > 0.5
    minute_count = mask.shape[-1]
    dtype = _float(mask).dtype
    position = torch.arange(minute_count, device=mask.device, dtype=dtype)
    positions = torch.where(mask, position, torch.full_like(mask, float("inf"), dtype=dtype))
    following = torch.cummin(positions.flip(-1), dim=-1).values.flip(-1)
    shifted = torch.full_like(following, float("nan"))
    shifted[..., :-1] = torch.where(
        torch.isfinite(following[..., 1:]), following[..., 1:],
        torch.full_like(following[..., 1:], float("nan")),
    )
    return shifted - position


def _interval_stats(mask, statistic):
    output_dtype = _float(mask).dtype
    mask = mask.bool()
    # Preserve the legacy helper's alignment exactly: ``_shift_m(x, -1)``
    # places the next minute at t, so adjacent runs are represented by their
    # final minute.
    following = torch.zeros_like(mask)
    following[..., :-1] = mask[..., 1:]
    mask = mask & ~following
    shape = mask.shape
    rows = mask.reshape(-1, shape[-1])
    outputs = torch.full((rows.shape[0],), float("nan"), device=mask.device, dtype=output_dtype)
    # Seed anchors use this only for equivalence/reference evaluation.  The
    # event-skeleton implementation remains the high-throughput search path.
    for index, row in enumerate(rows):
        positions = torch.nonzero(row, as_tuple=False).flatten()
        event_count = positions.numel()
        positions = positions[positions > 0].float()
        if event_count < 2:
            continue
        if positions.numel() < 2:
            outputs[index] = 0
            continue
        gaps = positions.diff()
        if statistic == "std": outputs[index] = gaps.std(unbiased=False)
        elif statistic == "skew":
            d = gaps - gaps.mean(); s = torch.sqrt((d.square()).mean())
            outputs[index] = (d ** 3).mean() / s ** 3 if s > 1e-12 else 0
        elif statistic == "kurt":
            d = gaps - gaps.mean(); v = d.square().mean()
            outputs[index] = (d ** 4).mean() / v ** 2 - 3 if v > 1e-12 else 0
        else: outputs[index] = gaps.mean()
    return outputs.reshape(shape[:-1])


def _ts_quantile(x, quantile, window):
    window = int(window)
    if x.shape[-1] < window:
        return torch.full_like(x, float("nan"))
    values = x.unfold(-1, window, 1).sort(-1).values
    valid = torch.isfinite(values).sum(-1).clamp(min=1)
    position = (float(quantile) * (valid - 1)).long().clamp(max=window - 1)
    out = torch.full_like(x, float("nan"))
    out[..., window - 1:] = values.gather(-1, position.unsqueeze(-1)).squeeze(-1)
    return out


def _cs_resid(y, x):
    y, x = _float(y), _float(x)
    valid = torch.isfinite(y) & torch.isfinite(x)
    fy, fx = torch.nan_to_num(y), torch.nan_to_num(x)
    count = valid.sum(0, keepdim=True).clamp(min=2).to(y.dtype)
    mx, my = (fx * valid).sum(0, keepdim=True) / count, (fy * valid).sum(0, keepdim=True) / count
    cov = ((fx - mx) * (fy - my) * valid).sum(0, keepdim=True) / count
    var = ((fx - mx).square() * valid).sum(0, keepdim=True) / count
    beta = cov / var.clamp(min=1e-12)
    residual = y - (my - beta * mx + beta * x)
    return torch.where(valid, residual, torch.full_like(residual, float("nan")))


def _roll_cut(x, y, window, quantile):
    x, y = _float(x), _float(y)
    window = int(window)
    out = torch.full(x.shape[:2], float("nan"), device=x.device, dtype=x.dtype)
    for day in range(window - 1, x.shape[1]):
        xv = x[:, day-window+1:day+1].reshape(x.shape[0], -1)
        yv = y[:, day-window+1:day+1].reshape(y.shape[0], -1)
        valid = torch.isfinite(xv) & torch.isfinite(yv)
        high = torch.nanquantile(yv.float(), 1-float(quantile), dim=-1).to(x.dtype)
        low = torch.nanquantile(yv.float(), float(quantile), dim=-1).to(x.dtype)
        top, bottom = valid & (yv >= high[:, None]), valid & (yv <= low[:, None])
        hi = (torch.nan_to_num(xv) * top).sum(-1) / top.sum(-1).clamp(min=1)
        lo = (torch.nan_to_num(xv) * bottom).sum(-1) / bottom.sum(-1).clamp(min=1)
        out[:, day] = torch.where(valid.sum(-1) >= 30, hi-lo, torch.full_like(hi, float("nan")))
    return out


def _register_variants(registry, base, implementation, arity=1, include_scalar=False):
    kinds = {
        "minute": SemanticType.LEGACY_MINUTE,
        "daily": SemanticType.LEGACY_DAILY,
        "same_minute": SemanticType.SAME_MINUTE_HISTORY,
    }
    for suffix, kind in kinds.items():
        registry.register(OperatorSpec(
            f"seed_{base}_{suffix}", (kind,) * arity, kind, implementation,
        ))
        if include_scalar and arity == 2:
            registry.register(OperatorSpec(
                f"seed_{base}_{suffix}_scalar_right", (kind, SemanticType.SCALAR), kind, implementation,
            ))
            registry.register(OperatorSpec(
                f"seed_{base}_{suffix}_scalar_left", (SemanticType.SCALAR, kind), kind, implementation,
            ))


def register_seed_tree_operators(registry: OperatorRegistry) -> None:
    minute, daily, same = (
        SemanticType.LEGACY_MINUTE, SemanticType.LEGACY_DAILY,
        SemanticType.SAME_MINUTE_HISTORY,
    )
    for name, fn in (("add", _binary(lambda a,b:a+b)), ("sub", _binary(lambda a,b:a-b)),
                     ("mul", _binary(lambda a,b:a*b)), ("div", _safe_div)):
        _register_variants(registry, name, fn, arity=2, include_scalar=True)
    for name, fn in (("abs", lambda x: torch.abs(_float(x))), ("sqrt", _sqrt),
                     ("neg", lambda x: -_float(x)), ("f", _float)):
        _register_variants(registry, name, fn)
    for name, fn in (("ge", _comparison(lambda a,b:a>=b)),
                     ("le", _comparison(lambda a,b:a<=b)),
                     ("gt", _comparison(lambda a,b:a>b)),
                     ("lt", _comparison(lambda a,b:a<b)),
                     ("or", _binary(lambda a,b:a.bool()|b.bool()))):
        _register_variants(registry, name, fn, arity=2, include_scalar=name != "or")

    registry.register(OperatorSpec("seed_intra_shift", (minute,), minute, _shift_minute,
        parameter_domains={"shift": (-1, 1)}, intraday_lookahead_minutes=lambda p:max(0,p["shift"])))
    registry.register(OperatorSpec("seed_intra_mean", (minute,), minute, lambda x,window:_intra_roll(x,window),
        parameter_domains={"window": WINDOWS}))
    registry.register(OperatorSpec("seed_intra_std", (minute,), minute, lambda x,window:_intra_roll(x,window,True),
        parameter_domains={"window": WINDOWS}))

    for suffix, kind in (("minute", minute), ("same_minute", same)):
        reducer = _day_reduce if kind == minute else _day_reduce_same_minute
        for name in ("sum", "mean", "std", "min", "max"):
            registry.register(OperatorSpec(f"seed_day_{name}_{suffix}", (kind,), daily,
                lambda x, _n=name, _r=reducer: _r(x, _n)))
    for name, fn in (("last", _day_last), ("first", _day_first)):
        registry.register(OperatorSpec(f"seed_day_{name}", (minute,), daily, fn))
    registry.register(OperatorSpec("seed_day_ratio", (minute,), daily,
        lambda x:_day_last(x)/_day_first(x).clamp(min=1e-6)))
    registry.register(OperatorSpec("seed_day_median", (minute,), daily, lambda x:_day_quantile(x,.5)))
    registry.register(OperatorSpec("seed_day_quantile", (minute,), daily, _day_quantile,
        parameter_domains={"quantile": QUANTILES}))
    registry.register(OperatorSpec("seed_day_corr", (minute,minute), daily, _day_corr))
    registry.register(OperatorSpec("seed_time_barycenter", (minute,), daily, _time_barycenter))
    for name in ("istd", "iskew", "ikurt"):
        stat = {"istd":"std", "iskew":"skew", "ikurt":"kurt"}[name]
        registry.register(OperatorSpec(f"seed_day_{name}", (minute,), daily,
            lambda x,_s=stat:_interval_stats(x,_s)))

    for suffix, kind in (("daily",daily),("same_minute",same)):
        for name, mode in (("mean","mean"),("sum","sum"),("max","max")):
            registry.register(OperatorSpec(f"seed_ts_{name}_{suffix}", (kind,), kind,
                lambda x,window,_m=mode:_rolling(x,window,_m),
                parameter_domains={"window":WINDOWS}, needs_history=True,
                history_days=lambda p:p["window"]-1))
        # Preserve the old runtime's ts_min implementation (negative rolling max).
        registry.register(OperatorSpec(f"seed_ts_min_{suffix}", (kind,), kind,
            lambda x,window:-_rolling(x,window,"max"), parameter_domains={"window":WINDOWS},
            needs_history=True, history_days=lambda p:p["window"]-1))
        registry.register(OperatorSpec(f"seed_ts_delay_{suffix}", (kind,), kind, _ts_delay,
            parameter_domains={"window":WINDOWS}, needs_history=True,
            history_days=lambda p:p["window"]))
        registry.register(OperatorSpec(f"seed_ts_std_{suffix}", (kind,), kind, _ts_std,
            parameter_domains={"window":WINDOWS}, needs_history=True,
            history_days=lambda p:p["window"]-1))
        registry.register(OperatorSpec(f"seed_ts_corr_{suffix}", (kind,kind), kind,
            lambda x,y,window:_rolling_pair(x,y,window)[0]/torch.sqrt(
                (_rolling_pair(x,y,window)[1]*_rolling_pair(x,y,window)[2]).clamp(min=1e-6)),
            parameter_domains={"window":WINDOWS}, needs_history=True,
            history_days=lambda p:p["window"]-1, cost=3))
    registry.register(OperatorSpec("seed_ts_quantile_daily", (daily,), daily, _ts_quantile,
        parameter_domains={"quantile":QUANTILES,"window":WINDOWS}, needs_history=True,
        history_days=lambda p:p["window"]-1))

    registry.register(OperatorSpec("seed_to_same_minute", (minute,), same,
        lambda x:x.permute(0,2,1)))
    registry.register(OperatorSpec("seed_broadcast_daily", (daily,), minute, _broadcast_daily,
        passes_context=True))
    registry.register(OperatorSpec("seed_mask_mul", (minute,SemanticType.SESSION_MASK), minute, _mask_mul))
    registry.register(OperatorSpec("seed_mask_agg", (minute,minute), daily, _mask_agg,
        parameter_domains={"statistic":MASK_STATS}))
    registry.register(OperatorSpec(
        "seed_mask_agg_all", (minute,), daily,
        lambda x, statistic, all_minutes: _mask_agg(x, 1, statistic),
        parameter_domains={"statistic":MASK_STATS, "all_minutes":(True,)},
    ))
    registry.register(OperatorSpec("seed_mask_ratio", (minute,minute,minute), daily, _mask_ratio,
        parameter_domains={"window":WINDOWS}, needs_history=True,
        history_days=lambda p:p["window"]-1))
    registry.register(OperatorSpec("seed_dist_to_event", (minute,), minute, _dist_to_event))
    registry.register(OperatorSpec("seed_cs_resid", (daily,daily), daily, _cs_resid,
        needs_full_cross_section=True))
    registry.register(OperatorSpec("seed_cs_rank", (daily,), daily, cross_section_rank,
        needs_full_cross_section=True))
    registry.register(OperatorSpec("seed_roll_cut", (minute,minute), daily, _roll_cut,
        parameter_domains={"window":WINDOWS,"quantile":QUANTILES}, needs_history=True,
        history_days=lambda p:p["window"]-1, cost=8))

    # Every migrated seed kernel is at least linear in the tensor elements it
    # consumes. Without these exponents a measured small-grid runtime would be
    # treated as constant when extrapolated to the production grid.
    minute_like = {minute, same}
    for name in registry.names():
        if not name.startswith("seed_"):
            continue
        spec = registry.get(name)
        semantic_types = set(spec.input_types) | {spec.output_type}
        complexity = {"I": 1.0, "D": 1.0}
        if semantic_types & minute_like:
            complexity["M"] = 1.0
        if name in {
            "seed_day_median", "seed_day_quantile", "seed_mask_agg",
            "seed_mask_agg_all", "seed_roll_cut",
        }:
            complexity["logM"] = 1.0
        if name == "seed_cs_rank":
            complexity["logI"] = 1.0
        registry.replace(replace(
            spec,
            complexity=complexity,
            memory_complexity=complexity,
        ))

# -*- coding: utf-8 -*-
"""因子 5 用到的全部算子实现，由 AST 从项目源码机械抽取。

本文件仅供阅读核对，不是可运行模块（缺少 import 与依赖）。
每个函数上方标注了它在项目里的真实位置，请以那里为准。
生成脚本：make_package.py，抽取方式为 ast.parse，非手工复制。
"""

import torch
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────
# 分钟级因子挖掘/operators/event.py:24
# ────────────────────────────────────────────────────────────────────
def minute_return(close: torch.Tensor, horizon: int = 1) -> torch.Tensor:
    out = torch.full_like(close.float(), float("nan"))
    left, right = close[..., :-horizon].float(), close[..., horizon:].float()
    value = right / left.clamp(min=1e-12) - 1.0
    good = torch.isfinite(left) & torch.isfinite(right) & (left > 0)
    out[..., horizon:] = torch.where(
        good, value, torch.full_like(value, float("nan"))
    )
    return out

# ────────────────────────────────────────────────────────────────────
# 分钟级因子挖掘/operators/intraday.py:82
# ────────────────────────────────────────────────────────────────────
def close_minute_log_return(close, horizon=1):
    simple = minute_return(close, horizon)
    return torch.log1p(simple.clamp(min=-1 + 1e-12))

# ────────────────────────────────────────────────────────────────────
# 分钟级因子挖掘/operators/event.py:75
# ────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────
# 分钟级因子挖掘/operators/event.py:105
# ────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────
# 分钟级因子挖掘/operators/event.py:98
# ────────────────────────────────────────────────────────────────────
def masked_daily_mean(signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool() & torch.isfinite(signal)
    count = valid.sum(-1)
    value = (torch.nan_to_num(signal.float()) * valid).sum(-1) / count.clamp(min=1)
    return torch.where(count > 0, value, torch.full_like(value, float("nan")))

# ────────────────────────────────────────────────────────────────────
# 分钟级因子挖掘/operators/seed_tree.py:38
# ────────────────────────────────────────────────────────────────────
def _float(x):
    if not isinstance(x, torch.Tensor):
        return x
    return x if x.dtype.is_floating_point else x.to(torch.bfloat16)

# ────────────────────────────────────────────────────────────────────
# 分钟级因子挖掘/operators/seed_tree.py:108
# ────────────────────────────────────────────────────────────────────
def _history_rows(x):
    """Flatten D2 or (I,M,D) without losing the same-minute grouping."""
    return x.reshape(-1, x.shape[-1]), x.shape

# ────────────────────────────────────────────────────────────────────
# 分钟级因子挖掘/operators/seed_tree.py:113
# ────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────
# 分钟级因子挖掘/operators/seed_tree.py:142
# ────────────────────────────────────────────────────────────────────
def _ts_std(x, window):
    mean = _rolling(x, window, "mean")
    second = _rolling(_float(x).square(), window, "mean")
    return torch.sqrt((second - mean.square()).clamp(min=0))

# ────────────────────────────────────────────────────────────────────
# 分钟级因子挖掘/numeric/preprocessing.py:8
# ────────────────────────────────────────────────────────────────────
def remove_outliers(
    factor: torch.Tensor,
    n_mad: float = 5.0,
    dim: int = 0,
) -> torch.Tensor:
    """Winsorize each cross-section to median +/- ``n_mad`` raw MAD.

    The operation is date-local and therefore introduces no look-ahead. Only
    finite factor observations enter the median and MAD; missing/non-finite
    inputs remain missing. ``MAD`` is the unscaled median absolute deviation
    requested by the research protocol (there is deliberately no 1.4826
    normal-consistency multiplier).
    """
    if n_mad < 0:
        raise ValueError("n_mad must be non-negative")
    value = factor.float()
    finite = torch.isfinite(value)
    clean = torch.where(finite, value, torch.full_like(value, float("nan")))
    median = torch.nanquantile(clean, 0.5, dim=dim, keepdim=True)
    deviation = torch.abs(clean - median)
    mad = torch.nanquantile(deviation, 0.5, dim=dim, keepdim=True)
    lower, upper = median - n_mad * mad, median + n_mad * mad
    clipped = torch.maximum(torch.minimum(clean, upper), lower)
    return torch.where(finite, clipped, torch.full_like(clipped, float("nan")))

# ────────────────────────────────────────────────────────────────────
# 分钟级因子挖掘/numeric/preprocessing.py:66
# ────────────────────────────────────────────────────────────────────
def neutralize(
    factor: torch.Tensor,
    industry: torch.Tensor | None = None,
    levels: int | None = None,
    continuous: tuple[torch.Tensor, ...] = (),
    min_cross_section: int = 30,
) -> torch.Tensor:
    """Replace a factor with its daily cross-sectional OLS residual.

    ``Y`` is the factor and ``X`` is the industry dummy block beside every
    continuous exposure (log float market cap in the standard protocol). The
    normal equations are solved with ``pinv`` rather than a ridge-regularised
    ``solve`` because the design is routinely rank-deficient: a full industry
    one-hot spans the intercept, and on any given date a CSI 500 slice can hold
    no members of some industry at all, leaving that column exactly zero. The
    pseudo-inverse returns the minimum-norm solution in both cases, which fits
    the identical subspace and therefore yields the same residual, whereas a
    plain solve would fail or return noise scaled by the ridge term.

    Because the industry dummies already span the intercept, no separate
    constant column is added unless there are no industries at all.

    A stock is dropped (returned as NaN) on any date where the factor, its
    industry, or any exposure is missing: a name that cannot be neutralised
    must not be ranked in the same cross-section as names that were, or it
    keeps the very exposure the residual is meant to remove. Dates with fewer
    than ``min_cross_section`` usable stocks are dropped entirely.
    """
    if factor.ndim != 2:
        raise ValueError("factor must have shape (instrument, date)")
    if industry is None and not continuous:
        raise ValueError("neutralization needs an industry or a continuous exposure")
    if min_cross_section < 1:
        raise ValueError("min_cross_section must be positive")
    value = factor.float()
    if industry is not None and industry.shape != factor.shape:
        raise ValueError("industry grid does not align with the factor")
    for exposure in continuous:
        if exposure.shape != factor.shape:
            raise ValueError("continuous exposure does not align with the factor")

    # Validity is settled before the design is built, because the continuous
    # columns are standardised over exactly the rows that enter the fit.
    valid = torch.isfinite(value)
    if industry is not None:
        valid &= industry >= 0
    for exposure in continuous:
        valid &= torch.isfinite(exposure.float())

    columns = []
    if industry is not None:
        columns.append(industry_one_hot(industry, levels))
    else:
        columns.append(torch.ones(*factor.shape, 1, device=value.device))
    weight = valid.to(value.dtype)
    count = weight.sum(0, keepdim=True).clamp(min=1)
    for exposure in continuous:
        # Centre and scale each date's exposure. The industry dummies span the
        # intercept, so this leaves the column space -- and therefore the
        # residual -- unchanged, but it keeps the normal matrix well
        # conditioned. Raw ln(float market cap) sits near 23 while a dummy is
        # 1, and squaring that ratio into X'X costs enough precision that a
        # factor exactly linear in the exposure kept a residual correlation of
        # -0.03 instead of 0.
        finite = torch.nan_to_num(exposure.float())
        mean = (finite * weight).sum(0, keepdim=True) / count
        # Zeroing must follow the fill: a non-finite exposure times a zero
        # weight is still non-finite, and one such cell makes the whole date's
        # SVD fail rather than simply dropping that row from the fit.
        centered = (finite - mean) * weight
        scale = (centered.square().sum(0, keepdim=True) / count).sqrt()
        columns.append((centered / scale.clamp(min=1e-8)).unsqueeze(2))
    design = torch.cat(columns, dim=2)

    # Solved in double precision: the pseudo-inverse of a rank-deficient normal
    # matrix separates kept from discarded directions by singular-value
    # magnitude, and float32 leaves that threshold too close to the noise.
    weighted = (design * weight.unsqueeze(2)).double()
    target = torch.where(valid, value, torch.zeros_like(value)).double()
    normal = torch.einsum("idp,idq->dpq", weighted, weighted)
    moment = torch.einsum("idp,id->dp", weighted, target)
    beta = torch.linalg.pinv(normal) @ moment.unsqueeze(2)
    fitted = torch.einsum("idp,dp->id", design.double(), beta.squeeze(2))
    enough = valid.sum(0) >= min_cross_section
    keep = valid & enough.unsqueeze(0)
    residual = (value.double() - fitted).to(value.dtype)
    return torch.where(keep, residual, torch.full_like(value, float("nan")))

# ────────────────────────────────────────────────────────────────────
# 分钟级因子挖掘/numeric/ranking.py:25
# ────────────────────────────────────────────────────────────────────
def cross_section_rank(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """NaN-aware fractional rank in [0, 1] with average ranks for ties.

    A double-``argsort`` assigns arbitrary distinct ranks to equal values.
    Event factors contain many ties (event counts, masked medians), so that
    behaviour can manufacture a small IC out of row order alone. Every member
    of a tie group gets its mid-rank instead. Fully vectorised.
    """
    x = _as_float(x)   # defensive: and_/or_ may produce bool
    moved = x.movedim(dim, -1)
    width = moved.shape[-1]
    rows = moved.reshape(-1, width)
    valid = torch.isfinite(rows)
    clean = torch.where(valid, rows, torch.full_like(rows, float("inf")))
    values, order = torch.sort(clean, dim=1, stable=True)
    sorted_valid = torch.gather(valid, 1, order)
    pos = torch.arange(width, device=x.device).expand_as(order)

    new_group = sorted_valid.clone()
    if width > 1:
        new_group[:, 1:] &= (
            ~sorted_valid[:, :-1] | (values[:, 1:] != values[:, :-1])
        )
    starts = torch.where(new_group, pos, torch.zeros_like(pos))
    starts = torch.cummax(starts, dim=1).values

    end_group = sorted_valid.clone()
    if width > 1:
        end_group[:, :-1] &= (
            ~sorted_valid[:, 1:] | (values[:, :-1] != values[:, 1:])
        )
    ends = torch.where(end_group, pos, torch.full_like(pos, width - 1))
    ends = torch.flip(
        torch.cummin(torch.flip(ends, dims=(1,)), dim=1).values, dims=(1,)
    )
    sorted_rank = 0.5 * (starts + ends).to(rows.dtype)
    rank = torch.empty_like(sorted_rank).scatter(1, order, sorted_rank)
    count = valid.sum(1, keepdim=True)
    rank = rank / (count - 1).clamp(min=1).to(rank.dtype)
    rank = torch.where(valid, rank, torch.full_like(rank, float("nan")))
    return rank.reshape(moved.shape).movedim(-1, dim)

# ────────────────────────────────────────────────────────────────────
# 分钟级因子挖掘/evaluation/neutralize.py:8
# ────────────────────────────────────────────────────────────────────
def trailing_volatility(close: torch.Tensor, window: int) -> torch.Tensor:
    """Std of daily returns over the `window` days ending at each date.

    Strictly backward looking: the value at column t uses returns up to and
    including t, which is the information a signal formed at t's close has.
    """
    returns = torch.full_like(close, float("nan"))
    returns[:, 1:] = close[:, 1:] / close[:, :-1] - 1.0
    valid = torch.isfinite(returns)
    clean = torch.nan_to_num(returns)
    ones = valid.to(clean.dtype)
    csum = torch.cumsum(clean, dim=1)
    csum2 = torch.cumsum(clean * clean, dim=1)
    ccount = torch.cumsum(ones, dim=1)
    zero = torch.zeros_like(csum[:, :1])
    csum = torch.cat((zero, csum), 1)
    csum2 = torch.cat((zero, csum2), 1)
    ccount = torch.cat((zero, ccount), 1)
    total = csum[:, window:] - csum[:, :-window]
    total2 = csum2[:, window:] - csum2[:, :-window]
    count = ccount[:, window:] - ccount[:, :-window]
    mean = total / count.clamp(min=1)
    var = (total2 / count.clamp(min=1) - mean * mean).clamp(min=0)
    tail = torch.sqrt(var)
    tail = torch.where(
        count >= window * 0.8, tail, torch.full_like(tail, float("nan"))
    )
    head = torch.full(
        (close.shape[0], window - 1), float("nan"),
        device=close.device, dtype=tail.dtype,
    )
    return torch.cat((head, tail), dim=1)

# ────────────────────────────────────────────────────────────────────
# 分钟级因子挖掘/evaluation/incremental.py:287
# ────────────────────────────────────────────────────────────────────
def trailing_signal_mean(factor: torch.Tensor, window: int) -> torch.Tensor:
    """Strict trailing trading-day mean, including the current day."""
    if factor.ndim != 2:
        raise ValueError("factor must have shape (I,D)")
    if window < 1:
        raise ValueError("window must be >= 1")
    value = factor.float()
    if window == 1:
        return value
    valid = torch.isfinite(value)
    clean = torch.nan_to_num(value)
    sums = torch.cumsum(clean, dim=1)
    counts = torch.cumsum(valid.to(torch.int32), dim=1)
    prefix_sum = torch.zeros_like(sums[:, :1])
    prefix_count = torch.zeros_like(counts[:, :1])
    sums = torch.cat((prefix_sum, sums), dim=1)
    counts = torch.cat((prefix_count, counts), dim=1)
    rolling_sum = sums[:, window:] - sums[:, :-window]
    rolling_count = counts[:, window:] - counts[:, :-window]
    tail = rolling_sum / rolling_count.clamp(min=1).to(rolling_sum.dtype)
    tail = torch.where(
        rolling_count == window, tail, torch.full_like(tail, float("nan"))
    )
    head = torch.full(
        (factor.shape[0], window - 1), float("nan"),
        device=factor.device, dtype=tail.dtype,
    )
    return torch.cat((head, tail), dim=1)

# ────────────────────────────────────────────────────────────────────
# 分钟级因子挖掘/label.py:66
# ────────────────────────────────────────────────────────────────────
def week_end_mask(dates, device=None) -> torch.Tensor:
    """Mark the last available trading day in each ISO calendar week."""
    parsed = [date.fromisoformat(str(value)[:10]) for value in dates]
    mask = []
    for index, value in enumerate(parsed):
        current = value.isocalendar()[:2]
        following = (
            parsed[index + 1].isocalendar()[:2]
            if index + 1 < len(parsed) else None
        )
        mask.append(following != current)
    return torch.tensor(mask, dtype=torch.bool, device=device)

# ────────────────────────────────────────────────────────────────────
# 分钟级因子挖掘/label.py:80
# ────────────────────────────────────────────────────────────────────
def tensor_weekly_fwd_ret(close_d, dates):
    """Close-to-close return from one calendar-week end to the next.

    Values outside rebalance dates, and the final rebalance date without a
    subsequent exit, remain NaN.  Holiday-shortened weeks therefore rebalance
    on their actual last trading day rather than on a hard-coded weekday.
    """
    if close_d.ndim != 2 or close_d.shape[1] != len(dates):
        raise ValueError("close_d columns must align one-for-one with dates")
    mask = week_end_mask(dates, close_d.device)
    indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
    result = torch.full_like(close_d, float("nan"))
    if indices.numel() < 2:
        return result
    entry, exit_ = indices[:-1], indices[1:]
    result[:, entry] = close_d[:, exit_] / close_d[:, entry] - 1.0
    return result

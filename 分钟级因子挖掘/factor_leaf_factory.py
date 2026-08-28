"""Produce external-factor leaves from approved local data, with API fallback.

The factory keeps data provenance separate from the genetic representation.
Every produced value is a raw or mechanically derived data field; no complete
factor or historical candidate is installed as a leaf.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


SOURCE_LEAVES = {
    "raw_panic": {"daily_close", "market_close"},
    "rushing_forward": {
        "amount_share", "volume_share", "up_volume_down_price_mask",
    },
    "water_boat": {"high_amount", "low_amount", "float_market_cap"},
    "cooperation_effect": {
        "volume_share", "price_state", "daily_return", "pair_similarity",
    },
}


def requested_leaf_names(source_names) -> set[str]:
    return set().union(*(SOURCE_LEAVES.get(name, set()) for name in source_names))


def _boundary(x, first):
    valid = torch.isfinite(x)
    if first:
        index = valid.float().argmax(-1)
    else:
        index = x.shape[-1] - 1 - valid.flip(-1).float().argmax(-1)
    value = x.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    return torch.where(
        valid.any(-1), value,
        torch.full_like(value, float("nan")),
    )


def _daily_return(close):
    result = torch.full_like(close.float(), float("nan"))
    result[:, 1:] = close[:, 1:] / close[:, :-1].clamp(min=1e-12) - 1
    return result


def _trailing_mean(x, window):
    values = torch.nan_to_num(x.float())
    valid = torch.isfinite(x).float()
    total = values.cumsum(1)
    count = valid.cumsum(1)
    if window < x.shape[1]:
        total[:, window:] -= total[:, :-window].clone()
        count[:, window:] -= count[:, :-window].clone()
    return torch.where(
        count > 0, total / count.clamp(min=1),
        torch.full_like(total, float("nan")),
    )


def _rolling_sum_count(x, window):
    value = torch.nan_to_num(x.float())
    valid = torch.isfinite(x).float()
    total, square, count = value.cumsum(-1), value.square().cumsum(-1), valid.cumsum(-1)
    if window < x.shape[-1]:
        total[..., window:] -= total[..., :-window].clone()
        square[..., window:] -= square[..., :-window].clone()
        count[..., window:] -= count[..., :-window].clone()
    return total, square, count


def _price_state(open_, high, low, close, window=5, day_chunk=16):
    """Classify the minute price state without promoting the full cube.

    A production minute cube is close to 1 GB even in its resident precision.
    Keeping the twelve float32 rolling intermediates for all days at once can
    therefore exceed a 24 GB GPU before GP evaluation starts.  Trading days
    are independent here, so compute them in bounded slices.
    """
    result = torch.empty_like(close, dtype=torch.int8)
    for start in range(0, close.shape[1], day_chunk):
        stop = min(start + day_chunk, close.shape[1])
        total = square = count = None
        for value in (open_, high, low, close):
            part_total, part_square, part_count = _rolling_sum_count(
                value[:, start:stop], window
            )
            if total is None:
                total, square, count = part_total, part_square, part_count
            else:
                total.add_(part_total)
                square.add_(part_square)
                count.add_(part_count)
        mean = total / count.clamp(min=1)
        variance = (square / count.clamp(min=1) - mean.square()).clamp(min=0)
        std = variance.sqrt()
        current_close = close[:, start:stop].float()
        state = torch.ones_like(current_close, dtype=torch.int8)
        state = torch.where(current_close > mean + std, 2, state)
        state = torch.where(current_close < mean - std, 0, state)
        result[:, start:stop] = torch.where(
            count >= 4, state, torch.full_like(state, -1)
        )
    return result


def _cross_section_shares(close, volume, pool, day_chunk=16):
    """Build PIT cross-sectional shares in day chunks.

    Only the final low-precision leaves remain resident.  Float32 price,
    volume, amount and denominator tensors exist for at most ``day_chunk``
    days, which removes the multi-GB preprocessing spike seen in production.
    """
    amount_result = torch.full_like(close, float("nan"))
    volume_result = torch.full_like(volume, float("nan"))
    for start in range(0, close.shape[1], day_chunk):
        stop = min(start + day_chunk, close.shape[1])
        current_close = close[:, start:stop].float()
        current_volume = volume[:, start:stop].float()
        valid = (
            pool[:, start:stop].unsqueeze(-1)
            & torch.isfinite(current_close)
            & torch.isfinite(current_volume)
            & current_volume.ge(0)
        )
        safe_volume = torch.where(
            valid, current_volume, torch.zeros_like(current_volume)
        )
        amount = torch.where(
            valid, current_close * current_volume, torch.zeros_like(current_volume)
        )
        volume_denominator = safe_volume.sum(0, keepdim=True)
        amount_denominator = amount.sum(0, keepdim=True)
        volume_share = safe_volume / volume_denominator.clamp(min=1e-12)
        amount_share = amount / amount_denominator.clamp(min=1e-12)
        volume_share.masked_fill_(~(valid & volume_denominator.gt(0)), float("nan"))
        amount_share.masked_fill_(~(valid & amount_denominator.gt(0)), float("nan"))
        volume_result[:, start:stop].copy_(volume_share)
        amount_result[:, start:stop].copy_(amount_share)
    return amount_result, volume_result


def _rushing_event(close, volume, pool):
    """Audited proxy already used by prepare_rushing_forward_leaves.py."""
    valid = pool.unsqueeze(-1) & torch.isfinite(close) & torch.isfinite(volume)
    event = torch.zeros_like(valid)
    consecutive = valid[..., 1:] & valid[..., :-1]
    event[..., 1:] = (
        consecutive & volume[..., 1:].gt(volume[..., :-1])
        & close[..., 1:].lt(close[..., :-1])
    )
    return event


def _high_low_amount(open_, close, volume, window=20, day_chunk=16):
    day_open, day_close = _boundary(open_, True), _boundary(close, False)
    day_return = day_close / day_open.clamp(min=1e-12) - 1
    reasonable = _trailing_mean(day_return, window)
    high_result = torch.full_like(day_return, float("nan"))
    low_result = torch.full_like(day_return, float("nan"))
    for start in range(0, close.shape[1], day_chunk):
        stop = min(start + day_chunk, close.shape[1])
        current_close = close[:, start:stop].float()
        current_volume = volume[:, start:stop].float()
        relative = (
            current_close
            / day_open[:, start:stop].float().unsqueeze(-1).clamp(min=1e-12)
            - 1
        )
        amount = current_close * current_volume
        valid = torch.isfinite(relative) & torch.isfinite(amount)
        high = valid & (relative > reasonable[:, start:stop].unsqueeze(-1))
        low = valid & ~high
        high_amount = torch.where(high, amount, torch.zeros_like(amount)).sum(-1)
        low_amount = torch.where(low, amount, torch.zeros_like(amount)).sum(-1)
        high_result[:, start:stop] = torch.where(
            valid.any(-1), high_amount, torch.full_like(high_amount, float("nan"))
        )
        low_result[:, start:stop] = torch.where(
            valid.any(-1), low_amount, torch.full_like(low_amount, float("nan"))
        )
    return high_result, low_result


def _rolling_previous_mean(x, window=5):
    total, _, count = _rolling_sum_count(x, window)
    result = torch.full_like(total, float("nan"))
    result[..., 1:] = total[..., :-1] / count[..., :-1].clamp(min=1)
    return result


def _pair_similarity(close, volume, day_chunk=16):
    """Daily pair score from the handbook's three signs and two fallbacks.

    Stored as float16 because the score is an integer in [0, 3*M] and the
    downstream consumer only ranks it. One day is built at a time, bounding
    temporary memory at O(I^2*M) instead of O(I^2*D*M).
    """
    instruments, days, minutes = close.shape
    previous_day = torch.full((instruments, days), float("nan"), device=close.device)
    day_close = _boundary(close, False).float()
    previous_day[:, 1:] = day_close[:, :-1]
    result = torch.empty(
        (instruments, instruments, days), dtype=torch.float16,
        device=close.device,
    )
    # Equality counts are sums of three one-hot Gram matrices. Chunking days
    # avoids ever allocating (I,I,D,M), while bmm uses the GPU efficiently.
    for start in range(0, days, day_chunk):
        stop = min(start + day_chunk, days)
        current_close = close[:, start:stop].float()
        current_volume = volume[:, start:stop].float()
        one = torch.full_like(current_close, float("nan"))
        one[..., 1:] = (
            current_close[..., 1:] / current_close[..., :-1].clamp(min=1e-12) - 1
        )
        five = torch.full_like(current_close, float("nan"))
        five[..., 5:] = (
            current_close[..., 5:] / current_close[..., :-5].clamp(min=1e-12) - 1
        )
        volume_change = current_volume - _rolling_previous_mean(current_volume, 5)
        recent_price = current_close - _rolling_previous_mean(current_close, 5)
        fallback = torch.sign(recent_price)
        fallback = torch.where(
            fallback == 0,
            torch.sign(
                current_close - previous_day[:, start:stop].unsqueeze(-1)
            ),
            fallback,
        )

        def direction(value):
            sign = torch.sign(value)
            sign = torch.where(sign == 0, fallback, sign)
            return torch.where(torch.isfinite(value), sign.to(torch.int8), -2)

        directions = (
            direction(one), direction(one - five), direction(volume_change)
        )
        score = torch.zeros(
            (stop - start, instruments, instruments),
            dtype=torch.float16, device=close.device,
        )
        for values in directions:
            for state in (-1, 0, 1):
                member = values.eq(state).permute(1, 0, 2).to(torch.float16)
                score += torch.bmm(member, member.transpose(1, 2))
        result[:, :, start:stop] = score.permute(1, 2, 0)
    return result


def _load_market_close(path, dates, device):
    schema = pq.ParquetFile(path).schema_arrow
    column = next((name for name in ("close_badj", "close", "收盘") if name in schema.names), None)
    if column is None:
        raise ValueError(f"market parquet has no close column: {schema.names}")
    frame = pd.read_parquet(path, columns=["trade_date", column])
    frame["trade_date"] = frame["trade_date"].astype(str).str.slice(0, 10)
    values = frame.drop_duplicates("trade_date").set_index("trade_date")[column]
    return torch.as_tensor(values.reindex(dates).to_numpy(np.float32), device=device)


def _fetch_market_close(dates, device, cache_path):
    import akshare as ak

    frame = ak.stock_zh_index_daily_em(
        symbol="csi000905",
        start_date=str(min(dates)).replace("-", ""),
        end_date=str(max(dates)).replace("-", ""),
    )
    date_column = next(name for name in frame if "日期" in str(name) or str(name).lower() == "date")
    close_column = next(name for name in frame if "收盘" in str(name) or str(name).lower() == "close")
    saved = pd.DataFrame({
        "trade_date": pd.to_datetime(frame[date_column]).dt.strftime("%Y-%m-%d"),
        "close": pd.to_numeric(frame[close_column], errors="coerce"),
    })
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    saved.to_parquet(cache_path, index=False)
    return _load_market_close(cache_path, dates, device)


def _load_float_market_cap(path, dates, instruments, device):
    from min_gp.spectral_data import load_daily_exposures

    styles, _industry, _levels = load_daily_exposures(
        path, dates, instruments, device=device,
    )
    return styles["ln_float_market_cap"].exp()


def _fetch_float_market_cap(dates, instruments, device, cache_path):
    import akshare as ak

    rows = []
    for instrument in instruments:
        code = "".join(ch for ch in str(instrument) if ch.isdigit())[-6:]
        frame = ak.stock_value_em(symbol=code)
        date_column = next(name for name in frame if "日期" in str(name))
        cap_column = next(name for name in frame if "流通市值" in str(name))
        rows.append(pd.DataFrame({
            "instrument": str(instrument),
            "trade_date": pd.to_datetime(frame[date_column]).dt.strftime("%Y-%m-%d"),
            "float_market_cap": pd.to_numeric(frame[cap_column], errors="coerce"),
        }))
    saved = pd.concat(rows, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    saved.to_parquet(cache_path, index=False)
    index = pd.MultiIndex.from_product([instruments, dates])
    aligned = saved.drop_duplicates(["instrument", "trade_date"]).set_index(
        ["instrument", "trade_date"]
    )["float_market_cap"].reindex(index).to_numpy(np.float32).reshape(
        len(instruments), len(dates)
    )
    return torch.as_tensor(aligned, device=device)


@dataclass(frozen=True)
class LeafFactoryConfig:
    market_parquet: str | None = None
    exposures_parquet: str | None = None
    api_fallback: bool = True
    cache_directory: str | None = None
    build_pair_similarity: bool = True


def build_external_factor_leaves(
    context, daily_close, pool, dates, instruments, source_names,
    config: LeafFactoryConfig,
) -> dict:
    """Mutate ``context`` with every requested producible external leaf."""
    requested = requested_leaf_names(source_names)
    report = {"requested": sorted(requested), "built": {}, "warnings": []}
    device = daily_close.device
    cache = Path(config.cache_directory or "output/leaf_cache")
    close, volume = context.get("close"), context.get("volume")

    if "daily_close" in requested:
        context["daily_close"] = daily_close.float()
        report["built"]["daily_close"] = "local adjusted daily close parquet"
    if "daily_return" in requested:
        context["daily_return"] = _daily_return(daily_close)
        report["built"]["daily_return"] = "daily_close[t]/daily_close[t-1]-1"

    if "market_close" in requested:
        api_error = None
        try:
            cached_market = cache / "csi000905.parquet"
            # The research universe is CSI 500, so the configured local CSI
            # 500 series is the primary source. API/cache is only a fallback.
            if config.market_parquet and Path(config.market_parquet).exists():
                context["market_close"] = _load_market_close(
                    config.market_parquet, dates, device
                )
                report["built"]["market_close"] = f"local:CSI500:{config.market_parquet}"
            elif cached_market.exists():
                context["market_close"] = _load_market_close(
                    cached_market, dates, device
                )
                report["built"]["market_close"] = "cache:AkShare:csi000905"
            elif config.api_fallback:
                try:
                    context["market_close"] = _fetch_market_close(
                        dates, device, cached_market
                    )
                    report["built"]["market_close"] = "AkShare:csi000905"
                except Exception as exc:
                    api_error = exc
            if api_error is not None:
                report["warnings"].append(
                    f"CSI 500 AkShare fallback failed: {api_error!r}"
                )
        except Exception as exc:
            report["warnings"].append(f"market_close unavailable: {exc!r}")

    if requested & {"amount_share", "volume_share", "up_volume_down_price_mask"}:
        if close is not None and volume is not None:
            amount_share, volume_share = _cross_section_shares(close, volume, pool)
            if "amount_share" in requested:
                context["amount_share"] = amount_share
                report["built"]["amount_share"] = "proxy:close*volume / PIT cross-sectional sum"
            if "volume_share" in requested:
                context["volume_share"] = volume_share
                report["built"]["volume_share"] = "volume / PIT cross-sectional sum"
            if "up_volume_down_price_mask" in requested:
                context["up_volume_down_price_mask"] = _rushing_event(close, volume, pool)
                report["built"]["up_volume_down_price_mask"] = (
                    "audited proxy:volume[t]>volume[t-1] and close[t]<close[t-1]"
                )
                report["warnings"].append(
                    "rushing AmountShare denominator and 5-minute trend are not closed-form in the source; proxy is labelled"
                )

    if requested & {"high_amount", "low_amount"} and all(
        context.get(name) is not None for name in ("open", "close", "volume")
    ):
        high_amount, low_amount = _high_low_amount(
            context["open"], context["close"], context["volume"]
        )
        context["high_amount"], context["low_amount"] = high_amount, low_amount
        report["built"]["high_amount"] = "sum(close*volume where intraday RO > MA20 daily open-close return)"
        report["built"]["low_amount"] = "sum(close*volume where intraday RO <= MA20 daily open-close return)"
        report["warnings"].append("minute amount reconstructed as close*volume because source parquet has no amount")

    if "float_market_cap" in requested:
        try:
            if config.exposures_parquet and Path(config.exposures_parquet).exists():
                context["float_market_cap"] = _load_float_market_cap(
                    config.exposures_parquet, dates, instruments, device
                )
                report["built"]["float_market_cap"] = f"local:{config.exposures_parquet}"
            elif config.api_fallback:
                context["float_market_cap"] = _fetch_float_market_cap(
                    dates, instruments, device, cache / "float_market_cap.parquet"
                )
                report["built"]["float_market_cap"] = "AkShare:stock_value_em"
        except Exception as exc:
            report["warnings"].append(f"float_market_cap unavailable: {exc!r}")

    if "price_state" in requested and all(
        context.get(name) is not None for name in ("open", "high", "low", "close")
    ):
        context["price_state"] = _price_state(
            context["open"], context["high"], context["low"], context["close"]
        )
        report["built"]["price_state"] = "current close vs mean±std of trailing 5-minute OHLC 20-point window"

    if "pair_similarity" in requested and config.build_pair_similarity:
        if close is not None and volume is not None:
            estimate = len(instruments) ** 2 * len(dates) * 2
            report["pair_similarity_estimated_bytes"] = estimate
            context["pair_similarity"] = _pair_similarity(close, volume)
            report["built"]["pair_similarity"] = (
                "three intraday direction-agreement scores with two-level zero fallback; float16"
            )

    report["missing"] = sorted(requested - set(context))
    report["built_bytes"] = {
        name: int(context[name].numel() * context[name].element_size())
        for name in report["built"] if name in context
    }
    return report

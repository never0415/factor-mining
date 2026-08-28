# -*- coding: utf-8 -*-
"""Compute and export factor 5 (ts_std 60 / 60-day volatility neutralised).

The factor is a pure function of two minute panels, so it is written here as
one - no genome, no registry, no leaf factory. That is not only clearer, it
skips the eleven external leaves `_stage` builds and never uses.

Two files come out, both in the project's interchange format that
`load_factor_parquet` and `daily_gp --leaf` already read:

  *_raw     the daily factor before any cross-sectional treatment. This is
            the expensive part - it is what costs ~8 minutes of minute-level
            work - and it is independent of every downstream choice, so
            caching it is what makes neutralisation and smoothing
            experiments cost seconds instead of minutes.
  *_signal  the series that is actually ranked: raw -> pool -> MAD5 ->
            residual on 60-day volatility -> 10-day mean -> MAD5 -> x(-1).
            Reproduce or replace this cheaply from the raw file.

The three walk-forward windows carry different PIT universes, so they are
computed separately and concatenated in the long format, which handles a
changing universe without padding.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

from min_gp.config import (
    ADJUSTED_CLOSE_PARQUET, MINUTE_PARQUET, ZZ500_PIT_PARQUET,
)
from min_gp.data import build_slice, load_pit_codes, load_pit_daily_mask
from min_gp.evaluation.incremental import trailing_signal_mean
from min_gp.evaluation.neutralize import trailing_volatility
from min_gp.factor_export import factor_frame
from min_gp.numeric.preprocessing import neutralize, remove_outliers
from min_gp.operators.event import (
    forward_window_std, masked_daily_mean, topk_separated_events,
)
from min_gp.operators.intraday import close_minute_log_return
from min_gp.operators.seed_tree import _ts_std
from min_gp.spectral_data import load_daily_close_tensor
import min_gp.unified_top5_html_report as R


def factor_5_daily(
    close_minute: torch.Tensor,
    volume_minute: torch.Tensor,
    *,
    horizon: int = 5,
    response_window: int = 5,
    top_k: int = 10,
    exclude_before: int = 15,
    min_gap: int = 5,
    daily_window: int = 60,
    chunk_instruments: int = 96,
) -> torch.Tensor:
    """(I, D, M) minute close and volume -> (I, D) daily factor.

    Dispersion, over `daily_window` trading days, of the average realised
    volatility in the five minutes following each day's highest-volume
    minutes.

    Every minute operator is applied per (instrument, day) row, so nothing
    reads across a day boundary; the only look-ahead is the four minutes
    `forward_window_std` reaches forward inside the day, which the signal is
    entitled to because it is formed at that day's close.

    Instruments are streamed in blocks: a single (I, D, M) intermediate for
    the 2018-2022 window is about a gigabyte, and this holds three at once.
    """
    if close_minute.shape != volume_minute.shape:
        raise ValueError("close and volume panels must have the same shape")
    if close_minute.ndim != 3:
        raise ValueError("panels must be (instrument, date, minute)")
    rows, days, _minutes = close_minute.shape
    out = torch.full((rows, days), float("nan"),
                     device=close_minute.device, dtype=torch.float32)
    for start in range(0, rows, chunk_instruments):
        stop = min(start + chunk_instruments, rows)
        ret = close_minute_log_return(close_minute[start:stop].float(),
                                      horizon=horizon)
        response = forward_window_std(ret, window=response_window, ddof=1)
        spikes = topk_separated_events(
            volume_minute[start:stop].float(), k=top_k,
            exclude_before=exclude_before, min_gap=min_gap,
        )
        out[start:stop] = masked_daily_mean(response, spikes)
        del ret, response, spikes
        if close_minute.is_cuda:
            torch.cuda.empty_cache()
    return _ts_std(out, daily_window)


def signal_from_raw(
    raw: torch.Tensor, pool: torch.Tensor, exposure: torch.Tensor,
    *, smooth_days: int = 10, outlier_mad: float = 5.0,
    direction: int = -1, min_cross_section: int = 30,
) -> torch.Tensor:
    """Raw daily factor -> the series that is ranked. Cheap: (I, D) only.

    The volatility exposure is passed in rather than derived here, because
    where it is estimated matters: it has to see the warmup even when the
    factor and the smoothing are restricted to a window's own dates.
    """
    value = torch.where(pool, raw, torch.full_like(raw, float("nan")))
    value = remove_outliers(value, n_mad=outlier_mad, dim=0)
    residual = neutralize(value, continuous=(exposure,),
                          min_cross_section=min_cross_section)
    smoothed = trailing_signal_mean(residual, smooth_days)
    return remove_outliers(smoothed, n_mad=outlier_mad, dim=0) * float(direction)


def apply_adopted(args, path):
    """Overwrite the parameter defaults from the adopted factor definition.

    The neutralisation is part of what the factor *is*, so it has to come
    from the same place as the tree rather than from whatever --vol-window a
    caller remembers to pass. Anything the caller set explicitly on the
    command line still wins; only defaults are replaced.
    """
    record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    root = record["genome"]["root"]
    if root["operator"] != "seed_ts_std_daily":
        raise SystemExit(f"adopted root is {root['operator']}, unsupported here")
    inner = root["children"][0]["children"]
    stat, mask = inner[0], inner[1]
    ret = stat["children"][0]
    cs = record["definition"]["stage_2_cross_section"]
    return {
        "horizon": ret["params"]["horizon"],
        "response_window": stat["params"]["window"],
        "top_k": mask["params"]["k"],
        "exclude_before": mask["params"]["exclude_before"],
        "min_gap": mask["params"]["min_gap"],
        "daily_window": root["params"]["window"],
        "vol_window": cs["neutralize"]["vol_window"],
        "smooth_days": cs["smooth"]["days"],
        "outlier_mad": float(cs["winsorize"]["n_mad"]),
        "direction": cs["direction"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from-adopted",
        help="从固化的因子定义读取全部参数（含中性化窗口）；"
             "命令行显式给出的值仍然优先。")
    ap.add_argument("--out-dir", default="分钟级因子挖掘/output/factors")
    ap.add_argument("--stem", default="f5_ts60_vol60_20260828")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--response-window", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--exclude-before", type=int, default=15)
    ap.add_argument("--min-gap", type=int, default=5)
    ap.add_argument("--daily-window", type=int, default=60)
    ap.add_argument("--vol-window", type=int, default=60)
    ap.add_argument("--smooth-days", type=int, default=10)
    ap.add_argument("--outlier-mad", type=float, default=5.0)
    ap.add_argument("--direction", type=int, default=-1)
    ap.add_argument(
        "--smooth-on-warmup", action="store_true",
        help="让 10 日平滑用上预热历史（更合理，每段多约 2 周有效值），"
             "但结果不再逐位复现已发布的回测；默认关闭以保证可复现。")
    ap.add_argument("--extend-days", type=int, default=80)
    ap.add_argument("--chunk-instruments", type=int, default=96)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.from_adopted:
        given = {a.lstrip("-").replace("-", "_") for a in sys.argv[1:]
                 if a.startswith("--")}
        for name, value in apply_adopted(args, args.from_adopted).items():
            if name not in given:
                setattr(args, name, value)
        print(f"参数取自 {args.from_adopted}", flush=True)
    print(f"因子 ts_std={args.daily_window}  中性化 vol={args.vol_window}  "
          f"平滑 {args.smooth_days} 日  方向 {args.direction:+d}", flush=True)

    raw_parts, signal_parts, coverage = [], [], {}
    for key, label, begin, finish in R.WINDOWS:
        print(f"\n[{key}] {begin}..{finish}", flush=True)
        codes = load_pit_codes(str(ZZ500_PIT_PARQUET), begin, finish)
        tensors, _masks, _label, meta = build_slice(
            str(MINUTE_PARQUET), begin, finish, instruments=codes,
            device=args.device, extend_days=args.extend_days,
        )
        dates_all = [str(v) for v in meta["dates"]]
        instruments = [str(v) for v in meta["instruments"]]
        print(f"  {len(instruments)} 只 x {len(dates_all)} 日"
              f"（含预热 {meta['warmup']} 日）", flush=True)

        raw = factor_5_daily(
            tensors["close"], tensors["volume"],
            horizon=args.horizon, response_window=args.response_window,
            top_k=args.top_k, exclude_before=args.exclude_before,
            min_gap=args.min_gap, daily_window=args.daily_window,
            chunk_instruments=args.chunk_instruments,
        )
        del tensors
        if args.device == "cuda":
            torch.cuda.empty_cache()

        close_daily = load_daily_close_tensor(
            str(ADJUSTED_CLOSE_PARQUET), dates_all, instruments,
            device=args.device)
        pool = load_pit_daily_mask(
            str(ZZ500_PIT_PARQUET), dates_all, instruments, device=args.device)
        pool &= torch.isfinite(close_daily)
        # Only the window's own dates are exported; the warmup prefix exists
        # to make the rolling statistics correct, not to be published twice.
        keep = np.asarray([begin <= d <= finish for d in dates_all])
        take = torch.as_tensor(keep, device=raw.device)
        scoped = [d for d, t in zip(dates_all, keep) if t]

        # The published backtest neutralises and smooths the *scoped* slice, so
        # its 10-day mean has no history at a window's first nine dates and
        # drops them. Smoothing the warmed-up panel instead would be sounder -
        # still strictly backward, and it recovers about two weeks per window -
        # but the exported signal has to replay the numbers that were reported,
        # so the slice comes first here too. Pass --smooth-on-warmup for the
        # sounder variant when rebuilding for production rather than for parity.
        # The exposure always sees the warmup, whichever variant is chosen.
        exposure = trailing_volatility(close_daily, args.vol_window)
        if args.smooth_on_warmup:
            signal = signal_from_raw(
                raw, pool, exposure, smooth_days=args.smooth_days,
                outlier_mad=args.outlier_mad, direction=args.direction)[:, take]
        else:
            signal = signal_from_raw(
                raw[:, take], pool[:, take], exposure[:, take],
                smooth_days=args.smooth_days, outlier_mad=args.outlier_mad,
                direction=args.direction)

        raw_scoped = torch.where(pool, raw, torch.full_like(raw, float("nan")))
        raw_parts.append(factor_frame(
            raw_scoped[:, take].cpu(), instruments, scoped, column="factor"))
        signal_parts.append(factor_frame(
            signal.cpu(), instruments, scoped, column="factor"))
        live = int(torch.isfinite(signal).sum())
        coverage[key] = {
            "instruments": len(instruments), "dates": len(scoped),
            "first": scoped[0], "last": scoped[-1],
            "signal_live_cells": live,
        }
        print(f"  -> {len(scoped)} 个交易日，信号有效格 {live}", flush=True)
        del raw, signal, pool, close_daily, raw_scoped
        if args.device == "cuda":
            torch.cuda.empty_cache()

    root = pathlib.Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    meta_common = {
        "factor": "v7 因子 5",
        "expression": (
            "seed_ts_std_daily(masked_daily_mean_signal(forward_window_std("
            "close_minute_log_return(close, horizon=5), ddof=1, window=5), "
            "topk_separated_events(volume, exclude_before=15, k=10, "
            "min_gap=5)), window=60)"),
        "params": {
            "horizon": args.horizon, "response_window": args.response_window,
            "top_k": args.top_k, "exclude_before": args.exclude_before,
            "min_gap": args.min_gap, "ts_std": args.daily_window,
        },
        "universe": "中证 500 成分股（PIT）",
        "windows": coverage,
        "generated_by": "compute_f5.py",
        "warmup_trading_days_requested": args.extend_days,
        "warning": (
            "样本期起点 2018-01-02 即数据边界，实际预热 0 日；60 日 ts_std 与 "
            "60 日波动率在该段开头若干周使用不满窗口（严格后向，非前视）。"),
    }
    for name, parts, extra in (
        ("raw", raw_parts, {
            "content": "原始日频因子，仅经成分股过滤，未做任何截面处理",
            "downstream": "MAD5 -> 对 60 日波动率取截面残差 -> 10 日均值 -> MAD5 -> x(-1)",
        }),
        ("signal", signal_parts, {
            "content": "最终信号，即实际参与截面排序的序列",
            "pipeline": (f"raw -> 成分股过滤 -> MAD{args.outlier_mad:.0f} -> "
                         f"{args.vol_window} 日波动率残差 -> {args.smooth_days} "
                         f"日均值 -> MAD{args.outlier_mad:.0f} -> x({args.direction:+d})"),
            "vol_window": args.vol_window, "smooth_days": args.smooth_days,
            "direction": args.direction,
        }),
    ):
        frame = pd.concat(parts, ignore_index=True)
        frame = frame.sort_values(["instrument", "trade_date"], ignore_index=True)
        path = root / f"{args.stem}_{name}.parquet"
        frame.to_parquet(path, index=False)
        (path.with_suffix(".parquet.json")).write_text(
            json.dumps({**meta_common, **extra, "column": "factor",
                        "rows": len(frame),
                        "ok_cells": int((frame["status"] == "ok").sum())},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        ok = int((frame["status"] == "ok").sum())
        print(f"\nwrote {path}\n  {len(frame):,} 行，其中有效 {ok:,} "
              f"({ok/len(frame)*100:.1f}%)，"
              f"{path.stat().st_size/2**20:.1f} MiB")


if __name__ == "__main__":
    main()

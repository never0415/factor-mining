"""Compare candidates before and after neutralising their volatility exposure.

The v1 hold-out showed every front candidate sorting inversely on realised
volatility: group 1 carried roughly twice the weekly standard deviation of
group 5, and the two candidates that made money were the two with the flattest
slope. This residualises each factor on trailing volatility and re-measures, so
"the control worked" can be judged by the slope flattening rather than by the
return improving.

Both a level and a rank residual are produced. A monotone but non-linear
relationship survives an OLS fit on raw values, which is the same trap that
inflated the incremental IC before it was moved onto ranks.

  python -m min_gp.report_neutralized \
      --candidates .../event_skeleton_gp.jsonl --out neutralized.json
"""

import argparse
import json

import numpy as np
import torch

from min_gp.config import (
    ADJUSTED_CLOSE_PARQUET, MINUTE_PARQUET, RISK_EXPOSURES_PARQUET,
    ZZ500_PIT_PARQUET, require_path,
)
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation.incremental import trailing_signal_mean
from min_gp.evaluation.neutralize import BatchedNeutralizer, trailing_volatility
from min_gp.factors.event_skeleton import EventSkeletonGenome
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.numeric.ranking import cross_section_rank
from min_gp.report_candidates import _pearson, quantile_curves, series_stats
from min_gp.spectral_data import (
    build_minute_slice, load_daily_close_tensor, load_daily_exposures,
)


# Which risk columns each named treatment projects out. ``vol`` keeps the
# locally computed trailing volatility so earlier runs stay comparable; it
# matches the vendor's ``total_vol_60d`` at rank correlation 0.9999.
EXPOSURE_SETS = {
    "none": (),
    "vol": ("__trailing_vol__",),
    "beta": ("beta",),
    "beta_size": ("beta", "ln_amount"),
    "vol_beta": ("__trailing_vol__", "beta"),
    "vol_beta_size": ("__trailing_vol__", "beta", "ln_amount"),
    "full": ("beta", "ln_amount", "ivol_60d"),
    # Momentum sets. ``momentum_12_1`` skips the most recent month, the
    # convention that keeps the classic momentum premium separate from
    # short-horizon reversal; ``reversal_20d`` is the short leg on its own.
    "mom": ("momentum_12_1",),
    "rev": ("reversal_20d",),
    "mom_rev": ("momentum_12_1", "reversal_20d"),
    "beta_mom": ("beta", "momentum_12_1"),
    "beta_mom_rev": ("beta", "momentum_12_1", "reversal_20d"),
    "vol_beta_mom": ("__trailing_vol__", "beta", "momentum_12_1"),
}
RISK_COLUMNS = ("beta", "ln_amount", "ivol_60d", "total_vol_60d",
                "momentum_12_1", "momentum_12_0", "reversal_20d")


def build_neutralizers(names, shape, trailing_vol, styles, min_cross_section=30):
    """One rank-space projector per named exposure set, plus a pass-through."""
    projectors = {"none": None}
    for name in names:
        if name == "none":
            continue
        columns = []
        for key in EXPOSURE_SETS[name]:
            columns.append(trailing_vol if key == "__trailing_vol__" else styles[key])
        projectors[name] = BatchedNeutralizer(
            shape, continuous=tuple(columns), rank_space=True,
            min_cross_section=min_cross_section,
        )
    return projectors


def measure(factor, fwd, rank_y, direction, groups, cost_bps, periods):
    valid = torch.isfinite(factor) & torch.isfinite(fwd)
    ic = _pearson(factor, fwd, valid) * direction
    rank_ic = _pearson(cross_section_rank(factor), rank_y, valid) * direction
    series, gross, net, turnover, days = quantile_curves(
        factor, fwd, direction, groups, cost_bps
    )
    sigma = [float(np.std(_periodic(g))) * 100 for g in _cumulatives(series)]
    return dict(
        ic=series_stats(ic.cpu().numpy(), periods),
        rank_ic=series_stats(rank_ic.cpu().numpy(), periods),
        group_cumulative=[np.cumprod(1.0 + g).tolist() for g in series],
        net_long_short_cumulative=np.cumprod(1.0 + net).tolist(),
        long_short_total=float(np.prod(1.0 + gross) - 1.0) if gross.size else float("nan"),
        net_total=float(np.prod(1.0 + net) - 1.0) if net.size else float("nan"),
        sigma=sigma,
        sigma_slope=float(sigma[0] / sigma[-1]) if sigma[-1] else float("nan"),
        turnover=float(np.nanmean(turnover)) if turnover.size else float("nan"),
        days=days,
    )


def raw_exposures(factor, styles, trailing_vol, fwd):
    """Median cross-sectional rank correlation with each style.

    Neutralising a style the factor is not loaded on can only remove noise, so
    read this before reading any residualised return: it says whether the
    treatment had anything to act on. Measured on rebalance dates only, since
    ``fwd`` is NaN everywhere else.
    """
    ranked = cross_section_rank(factor.float())
    base = torch.isfinite(ranked) & torch.isfinite(fwd)
    out = {}
    for name, column in [("__trailing_vol__", trailing_vol), *sorted(styles.items())]:
        other = cross_section_rank(column.float())
        series = _pearson(ranked, other, base & torch.isfinite(other)).cpu().numpy()
        out[name] = (float(np.nanmedian(series))
                     if np.isfinite(series).any() else float("nan"))
    return out


def _cumulatives(series):
    return [np.asarray(values) for values in series]


def _periodic(values):
    return np.asarray(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--max-rank", type=int, default=0)
    parser.add_argument("--start", default="2025-01-02")
    parser.add_argument("--end", default="2026-07-31")
    parser.add_argument("--warmup-start", default="2024-08-01",
                        help="extra history so trailing volatility is defined at --start")
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--exposures", default="none,vol,beta,mom,beta_mom,full",
                        help="comma-separated exposure sets to project out; "
                             "choose from none/vol/beta/beta_size/full")
    parser.add_argument("--rebalance", default="week_end")
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument("--signal-average-days", type=int, default=5)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"
    minute_path = require_path(str(MINUTE_PARQUET), "minute parquet")
    daily_path = require_path(str(ADJUSTED_CLOSE_PARQUET), "adjusted close")
    pit_path = require_path(str(ZZ500_PIT_PARQUET), "PIT parquet")

    records, seen = [], set()
    with open(args.candidates, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not record["fitness"]["valid"]:
                continue
            if record.get("pareto_rank", 0) > args.max_rank:
                continue
            if record["expression"] in seen:
                continue
            seen.add(record["expression"])
            records.append(record)
    genomes = [EventSkeletonGenome.from_dict(r["genome"]) for r in records]
    directions = [int(r["fitness"].get("direction", 1)) for r in records]
    fields = tuple(sorted({f for g in genomes for f in g.required_fields}))

    instruments = load_pit_codes(pit_path, args.start, args.end)
    full_dates = load_pit_dates(pit_path, args.warmup_start, args.end)
    minute, meta = build_minute_slice(
        minute_path, args.start, args.end, fields=fields,
        instruments=instruments, dates=load_pit_dates(pit_path, args.start, args.end),
        device=device,
    )
    dates = meta["dates"]
    offset = full_dates.index(dates[0])
    print(f"[neutral] {len(genomes)} candidates, {meta['I']}x{meta['D']}, "
          f"warmup {offset} days", flush=True)

    close_full = load_daily_close_tensor(
        daily_path, full_dates, meta["instruments"], device=device
    )
    vol = trailing_volatility(close_full, args.vol_window)[:, offset:]
    close = close_full[:, offset:]
    pool = load_pit_daily_mask(
        pit_path, dates, meta["instruments"], device=device
    ) & torch.isfinite(close)
    fwd = tensor_rebalance_fwd_ret(close, dates, args.rebalance, args.period)
    fwd = torch.where(pool, fwd, torch.full_like(fwd, float("nan")))
    rank_y = cross_section_rank(fwd)
    periods = 52 if args.rebalance == "week_end" else 244
    covered = float(torch.isfinite(vol[pool]).float().mean())
    print(f"[neutral] trailing {args.vol_window}d volatility defined on "
          f"{covered:.1%} of pool cells", flush=True)

    modes = [name.strip() for name in args.exposures.split(",") if name.strip()]
    unknown = sorted(set(modes) - set(EXPOSURE_SETS))
    if unknown:
        raise SystemExit(f"unknown exposure set(s) {unknown}; "
                         f"choose from {sorted(EXPOSURE_SETS)}")
    styles = {}
    if any(EXPOSURE_SETS[m] and EXPOSURE_SETS[m] != ("__trailing_vol__",)
           for m in modes):
        risk_path = require_path(str(RISK_EXPOSURES_PARQUET), "risk exposures")
        loaded, _industry, _levels = load_daily_exposures(
            risk_path, full_dates, meta["instruments"],
            continuous_columns=RISK_COLUMNS, industry_column=None, device=device,
        )
        styles = {k: v[:, offset:] for k, v in loaded.items()}
        for key in sorted({c for m in modes for c in EXPOSURE_SETS[m]
                           if c != "__trailing_vol__"}):
            share = float(torch.isfinite(styles[key][pool]).float().mean())
            print(f"[neutral] {key} defined on {share:.1%} of pool cells", flush=True)
    projectors = build_neutralizers(modes, fwd.shape, vol, styles)

    report = {"dates": dates, "vol_window": args.vol_window,
              "exposure_sets": {m: list(EXPOSURE_SETS[m]) for m in modes},
              "candidates": []}
    for position, genome in enumerate(genomes):
        direction = directions[position]
        raw = genome.evaluate(minute, args.chunk_rows)
        raw = trailing_signal_mean(raw, args.signal_average_days)
        raw = torch.where(pool, raw, torch.full_like(raw, float("nan")))
        entry = {"index": position, "expression": records[position]["expression"],
                 "exposures": raw_exposures(raw, styles, vol, fwd)}
        for mode in modes:
            projector = projectors[mode]
            factor = raw if projector is None else projector(raw)
            factor = torch.where(pool, factor, torch.full_like(factor, float("nan")))
            entry[mode] = measure(
                factor, fwd, rank_y, direction, args.groups, args.cost_bps, periods
            )
            del factor
        report["candidates"].append(entry)
        del raw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[neutral] #{position} done", flush=True)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False)
    print(f"[neutral] written -> {args.out}")


if __name__ == "__main__":
    main()
